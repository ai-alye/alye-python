"""Patches the `google-genai` package (Gemini API and Vertex AI): generate_content, generate_content_stream, embed_content."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ._common import AsyncStreamProxy, LLMCall, Patcher, SyncStreamProxy, call_tracer, get

log = logging.getLogger("alye")


def _start(resource: Any, kwargs: Dict[str, Any], operation: str, streaming: bool) -> Optional[LLMCall]:
    tracer = call_tracer()
    if tracer is None:
        return None
    try:
        api = getattr(resource, "_api_client", None)
        vertex = bool(getattr(api, "vertexai", False))
        cfg = kwargs.get("config")
        call = LLMCall(
            tracer,
            provider="gcp.vertex_ai" if vertex else "gcp.gemini",
            operation=operation,
            model=str(kwargs.get("model")) if kwargs.get("model") else None,
            attributes={
                "gen_ai.request.max_tokens": get(cfg, "max_output_tokens"),
                "gen_ai.request.temperature": get(cfg, "temperature"),
                "gen_ai.request.top_p": get(cfg, "top_p"),
                "gen_ai.request.top_k": get(cfg, "top_k"),
                "cloud.region": getattr(api, "location", None) if vertex else None,
                "alye.streaming": True if streaming else None,
            },
        )
        if operation == "embeddings":
            contents = kwargs.get("contents")
            call.set("gen_ai.embeddings.input.count", len(contents) if isinstance(contents, list) else 1)
        else:
            call.set_input(kwargs.get("contents"))
        return call
    except Exception:
        log.debug("alye: could not start Google GenAI span", exc_info=True)
        return None


def _enum(v: Any) -> Any:
    return getattr(v, "name", v)


def _usage(call: LLMCall, md: Any) -> None:
    if md is None:
        return
    prompt = get(md, "prompt_token_count")
    out = get(md, "candidates_token_count", default=0) or 0
    thoughts = get(md, "thoughts_token_count", default=0) or 0
    if prompt is None and not out and not thoughts:
        return
    # Gemini bills thinking tokens as output.
    call.set_usage(prompt, out + thoughts, cache_read_tokens=get(md, "cached_content_token_count"), reasoning_tokens=thoughts or None)


def _text(resp: Any) -> str:
    parts: List[str] = []
    for c in get(resp, "candidates", default=[]) or []:
        for p in get(c, "content", "parts", default=[]) or []:
            t = get(p, "text")
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts)


def _read(call: LLMCall, resp: Any) -> None:
    cands = get(resp, "candidates", default=[]) or []
    call.set_response(model=get(resp, "model_version"), response_id=get(resp, "response_id"), finish_reasons=[_enum(get(c, "finish_reason")) for c in cands if get(c, "finish_reason") is not None])
    _usage(call, get(resp, "usage_metadata"))
    text = _text(resp)
    if text:
        call.set_output(text)


def _read_embed(call: LLMCall, resp: Any) -> None:
    chars = get(resp, "metadata", "billable_character_count")
    if chars is not None:
        call.set("alye.billable_characters", chars)


class _Stream:
    def __init__(self, call: LLMCall) -> None:
        self.call = call
        self.last: Any = None
        self.usage: Any = None
        self.text: List[str] = []
        self.finish: List[Any] = []

    def chunk(self, ch: Any) -> None:
        self.last = ch
        if get(ch, "usage_metadata") is not None:
            self.usage = ch.usage_metadata
        t = _text(ch)
        if t:
            self.text.append(t)
        for c in get(ch, "candidates", default=[]) or []:
            fr = get(c, "finish_reason")
            if fr is not None:
                self.finish.append(_enum(fr))

    def done(self) -> None:
        self.call.set_response(model=get(self.last, "model_version"), response_id=get(self.last, "response_id"), finish_reasons=self.finish or None)
        _usage(self.call, self.usage)
        if self.text:
            self.call.set_output("".join(self.text))


def _make(operation: str, streaming: bool, is_async: bool):
    reader = _read_embed if operation == "embeddings" else _read

    def factory(original: Any) -> Any:
        if is_async:

            async def awrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                call = _start(self, kwargs, operation, streaming)
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
                if streaming:
                    h = _Stream(call)
                    return AsyncStreamProxy(result, call, h.chunk, h.done)
                try:
                    reader(call, result)
                except Exception:
                    log.debug("alye: could not read Google GenAI response", exc_info=True)
                call.end()
                return result

            return awrapper

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            call = _start(self, kwargs, operation, streaming)
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
            if streaming:
                h = _Stream(call)
                return SyncStreamProxy(result, call, h.chunk, h.done)
            try:
                reader(call, result)
            except Exception:
                log.debug("alye: could not read Google GenAI response", exc_info=True)
            call.end()
            return result

        return wrapper

    return factory


def patch(p: Patcher) -> bool:
    try:
        from google.genai import models
    except ImportError:
        return False
    done = False
    for cls_name in ("Models", "AsyncModels"):
        cls = getattr(models, cls_name, None)
        if cls is None:
            continue
        is_async = cls_name.startswith("Async")
        done |= p.wrap(cls, "generate_content", _make("generate_content", False, is_async))
        p.wrap(cls, "generate_content_stream", _make("generate_content", True, is_async))
        p.wrap(cls, "embed_content", _make("embeddings", False, is_async))
    return done
