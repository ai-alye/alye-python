"""Patches the official `anthropic` package: messages.create (sync, async, stream=True, and through
.with_raw_response) and the messages.stream() helper, on messages and beta.messages."""

from __future__ import annotations

import inspect
import logging
import weakref
from typing import Any, Dict, List, Optional

from ._common import AsyncStreamProxy, LLMCall, Patcher, SyncStreamProxy, call_tracer, get, note_retry

log = logging.getLogger("alye")


def _given(v: Any) -> Any:
    return None if v is None or type(v).__name__ in ("NotGiven", "Omit") else v


def _start(resource: Any, kwargs: Dict[str, Any], streaming: bool) -> Optional[LLMCall]:
    tracer = call_tracer()
    if tracer is None:
        return None
    try:
        url = getattr(getattr(resource, "_client", None), "base_url", None)
        host = str(getattr(url, "host", "") or "") or None
        provider = "aws.bedrock" if host and "bedrock" in host else "gcp.vertex_ai" if host and "aiplatform" in host else "anthropic"
        call = LLMCall(
            tracer,
            provider=provider,
            operation="chat",
            model=str(kwargs.get("model")) if kwargs.get("model") else None,
            attributes={
                "server.address": host,
                "gen_ai.request.max_tokens": _given(kwargs.get("max_tokens")),
                "gen_ai.request.temperature": _given(kwargs.get("temperature")),
                "gen_ai.request.top_p": _given(kwargs.get("top_p")),
                "gen_ai.request.top_k": _given(kwargs.get("top_k")),
                "anthropic.request.service_tier": _given(kwargs.get("service_tier")),
                "alye.streaming": True if streaming else None,
            },
        )
        system = _given(kwargs.get("system"))
        call.set_input({"system": system, "messages": kwargs.get("messages")} if system else kwargs.get("messages"))
        return call
    except Exception:
        log.debug("alye: could not start Anthropic span", exc_info=True)
        return None


def _usage(call: LLMCall, usage: Any, output_override: Optional[int] = None) -> None:
    if usage is None:
        return
    base = get(usage, "input_tokens", default=0) or 0
    read = get(usage, "cache_read_input_tokens", default=0) or 0
    write = get(usage, "cache_creation_input_tokens", default=0) or 0
    # Anthropic reports cached tokens separately; total input = uncached + cache read + cache write.
    call.set_usage(base + read + write, output_override if output_override is not None else get(usage, "output_tokens"), cache_read_tokens=read, cache_write_tokens=write)


def _read_message(call: LLMCall, msg: Any) -> None:
    call.set_response(model=get(msg, "model"), response_id=get(msg, "id"), finish_reasons=[get(msg, "stop_reason")])
    call.set("anthropic.response.service_tier", get(msg, "usage", "service_tier"))
    _usage(call, get(msg, "usage"))
    call.set_output(get(msg, "content"))


class _EventStream:
    def __init__(self, call: LLMCall) -> None:
        self.call = call
        self.start_usage: Any = None
        self.model: Optional[str] = None
        self.id: Optional[str] = None
        self.stop: Optional[str] = None
        self.output_tokens: Optional[int] = None
        self.text: List[str] = []

    def chunk(self, ev: Any) -> None:
        t = get(ev, "type")
        if t == "message_start":
            m = get(ev, "message")
            self.model, self.id = get(m, "model"), get(m, "id")
            self.start_usage = get(m, "usage")
        elif t == "message_delta":
            self.stop = get(ev, "delta", "stop_reason") or self.stop
            out = get(ev, "usage", "output_tokens")
            if out is not None:
                self.output_tokens = out
        elif t == "content_block_delta":
            piece = get(ev, "delta", "text")
            if piece:
                self.text.append(piece)

    def done(self) -> None:
        self.call.set_response(model=self.model, response_id=self.id, finish_reasons=[self.stop] if self.stop else None)
        _usage(self.call, self.start_usage, self.output_tokens)
        if self.text:
            self.call.set_output("".join(self.text))


def _wrap_raw_parse(call: LLMCall, raw: Any, streaming: bool, is_async: bool) -> bool:
    """`.with_raw_response` (LangChain uses it) where the result can't be read now: a stream, or an
    async response whose parse() is a coroutine. Wrap raw.parse() so the caller's own parse reads
    the message or wraps the stream; the call ends then (or when the stream does)."""
    original = getattr(raw, "parse", None)
    if not callable(original):
        return False

    def on_parsed(value: Any) -> Any:
        if getattr(raw, "_alye_wrapped", False):
            return value
        raw._alye_wrapped = True
        if streaming:
            h = _EventStream(call)
            return (AsyncStreamProxy if is_async else SyncStreamProxy)(value, call, h.chunk, h.done)
        try:
            _read_message(call, value)
        except Exception:
            log.debug("alye: could not read Anthropic response", exc_info=True)
        call.end()
        return value

    if inspect.iscoroutinefunction(original):

        async def parse(*args: Any, **kwargs: Any) -> Any:
            return on_parsed(await original(*args, **kwargs))

    else:

        def parse(*args: Any, **kwargs: Any) -> Any:
            return on_parsed(original(*args, **kwargs))

    try:
        raw.parse = parse
    except Exception:
        return False
    try:
        weakref.finalize(raw, call.end)  # never parsed: still end the call (without usage)
    except TypeError:
        pass
    return True


def _finish(call: LLMCall, result: Any, streaming: bool, is_async: bool) -> Any:
    try:
        raw = getattr(result, "http_response", None) is not None
        if streaming and not raw:
            h = _EventStream(call)
            return (AsyncStreamProxy if is_async else SyncStreamProxy)(result, call, h.chunk, h.done)
        if raw:
            parse = getattr(result, "parse", None)
            if not streaming and callable(parse) and not inspect.iscoroutinefunction(parse):
                # The SDK caches the parsed message, so the caller's own parse() returns the same object.
                _read_message(call, parse())
            elif _wrap_raw_parse(call, result, streaming, is_async):
                return result
            else:
                call.set("alye.raw_response", True)
        else:
            _read_message(call, result)
    except Exception:
        log.debug("alye: could not read Anthropic response", exc_info=True)
    call.end()
    return result


def _make_create(is_async: bool):
    def factory(original: Any) -> Any:
        if is_async:

            async def awrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                streaming = kwargs.get("stream") is True
                call = _start(self, kwargs, streaming)
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
                return _finish(call, result, streaming, True)

            return awrapper

        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            streaming = kwargs.get("stream") is True
            call = _start(self, kwargs, streaming)
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
            return _finish(call, result, streaming, False)

        return wrapper

    return factory


# ---- messages.stream() helper ----------------------------------------------
class _ManagerProxy:
    """Wraps MessageStreamManager / AsyncMessageStreamManager. The request is sent on enter."""

    def __init__(self, inner: Any, call: LLMCall) -> None:
        self._inner = inner
        self._call = call
        self._stream: Any = None

    def _read(self) -> None:
        snap = None
        fn = getattr(self._stream, "_current_snapshot_or_none", None)
        try:
            snap = fn() if callable(fn) else getattr(self._stream, "current_message_snapshot", None)
        except Exception:
            snap = None
        if snap is not None:
            _read_message(self._call, snap)

    def __enter__(self) -> Any:
        self._call.enter()
        try:
            self._stream = self._inner.__enter__()
        except BaseException as e:
            self._call.exit_scope()
            self._call.fail(e)
            self._call.end()
            raise
        self._call.exit_scope()
        return self._stream

    def __exit__(self, et: Any, ev: Any, tb: Any) -> Any:
        try:
            return self._inner.__exit__(et, ev, tb)
        finally:
            if ev is not None:
                self._call.fail(ev)
            self._read()
            self._call.end()

    async def __aenter__(self) -> Any:
        self._call.enter()
        try:
            self._stream = await self._inner.__aenter__()
        except BaseException as e:
            self._call.exit_scope()
            self._call.fail(e)
            self._call.end()
            raise
        self._call.exit_scope()
        return self._stream

    async def __aexit__(self, et: Any, ev: Any, tb: Any) -> Any:
        try:
            return await self._inner.__aexit__(et, ev, tb)
        finally:
            if ev is not None:
                self._call.fail(ev)
            self._read()
            self._call.end()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _stream_factory(original: Any) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        manager = original(self, *args, **kwargs)
        call = _start(self, kwargs, True)
        return manager if call is None else _ManagerProxy(manager, call)

    return wrapper


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
        from anthropic.resources import messages as m
    except ImportError:
        return False
    modules = [m]
    try:
        from anthropic.resources.beta import messages as beta  # beta.messages: LangChain uses it for beta features

        modules.append(beta)
    except ImportError:
        pass
    done = False
    for mod in modules:
        for cls_name in ("Messages", "AsyncMessages"):
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            done |= p.wrap(cls, "create", _make_create(cls_name.startswith("Async")))
            p.wrap(cls, "stream", _stream_factory)
    try:
        from anthropic import _base_client

        for cls_name in ("SyncAPIClient", "AsyncAPIClient"):
            cls = getattr(_base_client, cls_name, None)
            if cls is not None:
                p.wrap(cls, "_sleep_for_retry", _retry_factory)
    except Exception:
        log.debug("alye: Anthropic retry counting unavailable", exc_info=True)
    return done
