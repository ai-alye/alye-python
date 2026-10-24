"""Patches the official `openai` package: chat completions, responses and embeddings (sync, async and streaming),
and the structured-output parse() methods.

Also used for OpenAI-compatible providers reached through the same client (Azure, DeepSeek, Groq, ...):
the provider name comes from the client's base URL.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional

from ._common import AsyncStreamProxy, LLMCall, Patcher, SyncStreamProxy, call_tracer, get, note_retry

log = logging.getLogger("alye")

_HOSTS = {
    "openai.azure.com": "azure.ai.openai",
    "api.deepseek.com": "deepseek",
    "api.groq.com": "groq",
    "api.x.ai": "x_ai",
    "api.mistral.ai": "mistral_ai",
    "api.together.xyz": "together_ai",
    "openrouter.ai": "openrouter",
    "generativelanguage.googleapis.com": "gcp.gemini",
    "api.perplexity.ai": "perplexity",
    "api.fireworks.ai": "fireworks_ai",
}


def _provider_and_server(resource: Any) -> Dict[str, Any]:
    client = getattr(resource, "_client", None)
    url = getattr(client, "base_url", None)
    host = str(getattr(url, "host", "") or "")
    port = getattr(url, "port", None)
    provider = "openai"
    for suffix, name in _HOSTS.items():
        if host == suffix or host.endswith("." + suffix):
            provider = name
            break
    return {"provider": provider, "server.address": host or None, "server.port": port}


def _request_attrs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    def given(v: Any) -> Any:
        # The SDK uses sentinel objects (NOT_GIVEN / Omit) for unset parameters.
        return None if v is None or type(v).__name__ in ("NotGiven", "Omit") else v

    return {
        "gen_ai.request.temperature": given(kwargs.get("temperature")),
        "gen_ai.request.top_p": given(kwargs.get("top_p")),
        "gen_ai.request.max_tokens": given(kwargs.get("max_completion_tokens")) or given(kwargs.get("max_tokens")) or given(kwargs.get("max_output_tokens")),
        "gen_ai.request.frequency_penalty": given(kwargs.get("frequency_penalty")),
        "gen_ai.request.presence_penalty": given(kwargs.get("presence_penalty")),
        "gen_ai.request.seed": given(kwargs.get("seed")),
        "gen_ai.request.choice.count": given(kwargs.get("n")),
        "openai.request.service_tier": given(kwargs.get("service_tier")),
        "alye.streaming": True if kwargs.get("stream") is True else None,
    }


def _start(resource: Any, kwargs: Dict[str, Any], operation: str) -> Optional[LLMCall]:
    tracer = call_tracer()
    if tracer is None:
        return None
    try:
        ps = _provider_and_server(resource)
        attrs = _request_attrs(kwargs)
        attrs["server.address"] = ps["server.address"]
        attrs["server.port"] = ps["server.port"]
        call = LLMCall(tracer, provider=ps["provider"], operation=operation, model=str(kwargs.get("model")) if kwargs.get("model") else None, attributes=attrs)
        if operation == "chat":
            call.set_input(kwargs.get("messages"))
        elif operation == "embeddings":
            inp = kwargs.get("input")
            call.set("gen_ai.embeddings.input.count", len(inp) if isinstance(inp, list) else 1)
        else:
            call.set_input({"instructions": kwargs.get("instructions"), "input": kwargs.get("input")} if kwargs.get("instructions") else kwargs.get("input"))
        if operation == "chat" and kwargs.get("stream") is True and tracer.config.inject_stream_usage and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}
            call.set("alye.stream_usage_injected", True)
        return call
    except Exception:
        log.debug("alye: could not start OpenAI span", exc_info=True)
        return None


# ---- response readers ----------------------------------------------------
def _chat_usage(call: LLMCall, usage: Any) -> None:
    if usage is None:
        return
    call.set_usage(
        get(usage, "prompt_tokens"),
        get(usage, "completion_tokens"),
        cache_read_tokens=get(usage, "prompt_tokens_details", "cached_tokens"),
        cache_write_tokens=get(usage, "prompt_tokens_details", "cache_write_tokens"),
        reasoning_tokens=get(usage, "completion_tokens_details", "reasoning_tokens"),
    )


def _read_chat(call: LLMCall, resp: Any) -> None:
    choices = get(resp, "choices", default=[]) or []
    call.set_response(model=get(resp, "model"), response_id=get(resp, "id"), finish_reasons=[get(c, "finish_reason") for c in choices])
    call.set("openai.response.service_tier", get(resp, "service_tier"))
    _chat_usage(call, get(resp, "usage"))
    call.set_output([get(c, "message") for c in choices])


def _responses_usage(call: LLMCall, usage: Any) -> None:
    if usage is None:
        return
    call.set_usage(
        get(usage, "input_tokens"),
        get(usage, "output_tokens"),
        cache_read_tokens=get(usage, "input_tokens_details", "cached_tokens"),
        reasoning_tokens=get(usage, "output_tokens_details", "reasoning_tokens"),
    )


def _read_responses(call: LLMCall, resp: Any) -> None:
    status = get(resp, "status")
    call.set_response(model=get(resp, "model"), response_id=get(resp, "id"), finish_reasons=[status] if status else None)
    _responses_usage(call, get(resp, "usage"))
    text = getattr(resp, "output_text", None)
    call.set_output(text if isinstance(text, str) else get(resp, "output"))


def _read_embeddings(call: LLMCall, resp: Any) -> None:
    call.set_response(model=get(resp, "model"))
    usage = get(resp, "usage")
    if usage is not None:
        call.set_usage(get(usage, "prompt_tokens"), 0)


READERS = {"chat": _read_chat, "responses": _read_responses, "embeddings": _read_embeddings}


def _unwrap_raw(result: Any, streaming: bool) -> Any:
    """`.with_raw_response` returns a response wrapper; parse it when that is safe (not streaming)."""
    if getattr(result, "http_response", None) is None:
        return result
    if not streaming and callable(getattr(result, "parse", None)) and not inspect.iscoroutinefunction(result.parse):
        try:
            return result.parse()
        except Exception:
            return None
    return None


# ---- stream handlers -----------------------------------------------------
class _ChatStream:
    def __init__(self, call: LLMCall) -> None:
        self.call = call
        self.text: Dict[int, List[str]] = {}
        self.finish: Dict[int, str] = {}
        self.model: Optional[str] = None
        self.id: Optional[str] = None
        self.usage: Any = None

    def chunk(self, ch: Any) -> None:
        self.model = self.model or get(ch, "model")
        self.id = self.id or get(ch, "id")
        if get(ch, "usage") is not None:
            self.usage = ch.usage
        for c in get(ch, "choices", default=[]) or []:
            i = get(c, "index", default=0)
            piece = get(c, "delta", "content")
            if piece:
                self.text.setdefault(i, []).append(piece)
            fr = get(c, "finish_reason")
            if fr:
                self.finish[i] = fr

    def done(self) -> None:
        self.call.set_response(model=self.model, response_id=self.id, finish_reasons=[self.finish[k] for k in sorted(self.finish)])
        _chat_usage(self.call, self.usage)
        if self.text:
            self.call.set_output([{"role": "assistant", "content": "".join(self.text[k])} for k in sorted(self.text)])


class _ResponsesStream:
    def __init__(self, call: LLMCall) -> None:
        self.call = call
        self.final: Any = None
        self.text: List[str] = []

    def chunk(self, ev: Any) -> None:
        etype = str(get(ev, "type", default=""))
        if etype.endswith(".delta") and isinstance(get(ev, "delta"), str) and "output_text" in etype:
            self.text.append(ev.delta)
        if etype in ("response.completed", "response.incomplete", "response.failed") and get(ev, "response") is not None:
            self.final = ev.response

    def done(self) -> None:
        if self.final is not None:
            status = get(self.final, "status")
            self.call.set_response(model=get(self.final, "model"), response_id=get(self.final, "id"), finish_reasons=[status] if status else None)
            _responses_usage(self.call, get(self.final, "usage"))
        if self.text:
            self.call.set_output("".join(self.text))


STREAMS = {"chat": _ChatStream, "responses": _ResponsesStream}


def _finish(call: LLMCall, result: Any, kwargs: Dict[str, Any], operation: str, is_async: bool) -> Any:
    streaming = kwargs.get("stream") is True
    try:
        if streaming and operation in STREAMS and getattr(result, "http_response", None) is None:
            handler = STREAMS[operation](call)
            proxy = AsyncStreamProxy if is_async else SyncStreamProxy
            return proxy(result, call, handler.chunk, handler.done)
        parsed = _unwrap_raw(result, streaming)
        if parsed is not None:
            READERS[operation](call, parsed)
        else:
            call.set("alye.raw_response", True)
    except Exception:
        log.debug("alye: could not read OpenAI response", exc_info=True)
    call.end()
    return result


def _make(operation: str, is_async: bool):
    # Async methods are marked by class (Async*), not by inspection: some are plain functions returning a coroutine.
    def factory(original: Any) -> Any:
        if is_async:

            async def awrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                call = _start(self, kwargs, operation)
                if call is None:
                    return await original(self, *args, **kwargs)
                call.enter()
                try:
                    result = await original(self, *args, **kwargs)
                except BaseException as e:
                    call.exit_scope()
                    call.fail(e)
                    call.end()
                    raise
                call.exit_scope()
                return _finish(call, result, kwargs, operation, is_async=True)

            return awrapper

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            call = _start(self, kwargs, operation)
            if call is None:
                return original(self, *args, **kwargs)
            call.enter()
            try:
                result = original(self, *args, **kwargs)
            except BaseException as e:
                call.exit_scope()
                call.fail(e)
                call.end()
                raise
            call.exit_scope()
            return _finish(call, result, kwargs, operation, is_async=False)

        return wrapper

    return factory


def _retry_factory(original: Any) -> Any:
    if inspect.iscoroutinefunction(original):

        async def asleep(self: Any, *args: Any, **kwargs: Any) -> Any:
            note_retry(getattr(kwargs.get("response"), "status_code", None))
            return await original(self, *args, **kwargs)

        return asleep

    def sleep(self: Any, *args: Any, **kwargs: Any) -> Any:
        note_retry(getattr(kwargs.get("response"), "status_code", None))
        return original(self, *args, **kwargs)

    return sleep


def patch(p: Patcher) -> bool:
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    done = False
    targets = [
        ("openai.resources.chat.completions", ("Completions", "AsyncCompletions"), "create", "chat"),
        # Structured outputs (LangChain's with_structured_output): parse() posts itself, not through create().
        ("openai.resources.chat.completions", ("Completions", "AsyncCompletions"), "parse", "chat"),
        ("openai.resources.responses", ("Responses", "AsyncResponses"), "create", "responses"),
        ("openai.resources.responses", ("Responses", "AsyncResponses"), "parse", "responses"),
        ("openai.resources.embeddings", ("Embeddings", "AsyncEmbeddings"), "create", "embeddings"),
    ]
    import importlib

    for module, classes, method, op in targets:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        for cls_name in classes:
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                done |= p.wrap(cls, method, _make(op, cls_name.startswith("Async")))
    try:
        from openai import _base_client

        for cls_name in ("SyncAPIClient", "AsyncAPIClient"):
            cls = getattr(_base_client, cls_name, None)
            if cls is not None:
                p.wrap(cls, "_sleep_for_retry", _retry_factory)
    except Exception:
        log.debug("alye: OpenAI retry counting unavailable", exc_info=True)
    return done
