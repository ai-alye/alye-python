"""Shared pieces for provider patches: the LLM call span, error classification, retry counting and stream wrappers."""

from __future__ import annotations

import contextvars
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ..context import reset_current_span, set_current_span
from ..span import NOOP_SPAN, Span
from ..tracer import Tracer, get_tracer

log = logging.getLogger("alye")

# Set while a patched call is running so the patched retry sleep can count attempts.
_retry_box: contextvars.ContextVar[Optional[List[Any]]] = contextvars.ContextVar("alye_retry_box", default=None)


def note_retry(status: Optional[int]) -> None:
    box = _retry_box.get()
    if box is not None:
        box.append(status)


def classify_error(exc: BaseException) -> str:
    """Map provider exceptions to a short, stable error.type."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    by_name = {
        "RateLimitError": "rate_limited",
        "APITimeoutError": "timeout",
        "DeadlineExceededError": "timeout",
        "TimeoutError": "timeout",
        "ReadTimeout": "timeout",
        "ConnectTimeout": "timeout",
        "APIConnectionError": "connection_error",
        "ConnectError": "connection_error",
        "AuthenticationError": "auth",
        "PermissionDeniedError": "auth",
        "BadRequestError": "invalid_request",
        "UnprocessableEntityError": "invalid_request",
        "NotFoundError": "not_found",
        "OverloadedError": "overloaded",
        "ServiceUnavailableError": "overloaded",
        "InternalServerError": "server_error",
        "RequestTooLargeError": "invalid_request",
        "ContentFilterFinishReasonError": "content_filter",
        "CancelledError": "cancelled",
    }
    if name in by_name:
        return by_name[name]
    if isinstance(status, int):
        if status == 429:
            return "rate_limited"
        if status in (401, 403):
            return "auth"
        if status == 408 or status == 504:
            return "timeout"
        if status == 529 or status == 503:
            return "overloaded"
        if status >= 500:
            return "server_error"
        if status >= 400:
            return "invalid_request"
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code == 429:
        return "rate_limited"
    return name


# The LangChain run (instrumentation/langchain.py) in progress on this context, if any. A call recorded
# by a client patch inside it claims it, so the LangChain callback doesn't record the same call again.
langchain_run: contextvars.ContextVar[Any] = contextvars.ContextVar("alye_langchain_run", default=None)


class LLMCall:
    """One model call. Created by provider patches, or by `alye.llm_call()` for anything else."""

    def __init__(self, tracer: Optional[Tracer], *, provider: str, operation: str, model: Optional[str], attributes: Optional[Dict[str, Any]] = None) -> None:
        self.tracer = tracer
        self.start = time.monotonic()
        self.first_token_at: Optional[float] = None
        self._retries: List[Any] = []
        self._retry_token: Optional[contextvars.Token] = None
        self._span_token: Optional[contextvars.Token] = None
        self._usage_seen = False
        self._ended = False
        name = f"{operation} {model}" if model else operation
        run = langchain_run.get()
        if run is not None and (attributes or {}).get("alye.integration") != "langchain":
            run.claimed = True
        if tracer is None:
            self.span: Any = NOOP_SPAN
        else:
            self.span = tracer.start_span(
                name,
                kind="client",
                attributes={
                    "gen_ai.provider.name": provider,
                    "gen_ai.operation.name": operation,
                    "gen_ai.request.model": model,
                    **(attributes or {}),
                },
            )

    # ---- lifecycle --------------------------------------------------------
    def enter(self) -> "LLMCall":
        self._retry_token = _retry_box.set(self._retries)
        if isinstance(self.span, Span):
            self._span_token = set_current_span(self.span)
        return self

    def exit_scope(self) -> None:
        """Leave the call's context (the span may still be open while a stream is read)."""
        if self._retry_token is not None:
            try:
                _retry_box.reset(self._retry_token)
            except ValueError:  # reset from a different context (e.g. stream consumed in another task)
                _retry_box.set(None)
            self._retry_token = None
        if self._span_token is not None:
            try:
                reset_current_span(self._span_token)
            except ValueError:
                pass
            self._span_token = None

    def __enter__(self) -> "LLMCall":
        return self.enter()

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> None:
        self.exit_scope()
        if exc is not None:
            self.fail(exc)
        self.end()

    # ---- recording helpers --------------------------------------------------
    def set(self, key: str, value: Any) -> "LLMCall":
        self.span.set_attribute(key, value)
        return self

    def set_usage(
        self,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        *,
        cache_read_tokens: Optional[int] = None,
        cache_write_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
    ) -> "LLMCall":
        if input_tokens is None and output_tokens is None:
            return self
        self._usage_seen = True
        s = self.span
        s.set_attribute("gen_ai.usage.input_tokens", _int(input_tokens))
        s.set_attribute("gen_ai.usage.output_tokens", _int(output_tokens))
        s.set_attribute("gen_ai.usage.cache_read.input_tokens", _int(cache_read_tokens) or None)
        s.set_attribute("gen_ai.usage.cache_creation.input_tokens", _int(cache_write_tokens) or None)
        s.set_attribute("gen_ai.usage.reasoning.output_tokens", _int(reasoning_tokens) or None)
        return self

    def set_response(self, *, model: Optional[str] = None, response_id: Optional[str] = None, finish_reasons: Optional[List[str]] = None) -> "LLMCall":
        self.span.set_attribute("gen_ai.response.model", model)
        self.span.set_attribute("gen_ai.response.id", response_id)
        if finish_reasons:
            self.span.set_attribute("gen_ai.response.finish_reasons", [str(r) for r in finish_reasons if r is not None])
        return self

    def set_input(self, value: Any) -> "LLMCall":
        if self.tracer is not None:
            self.span.set_attribute("gen_ai.input.messages", self.tracer.content(value))
        return self

    def set_output(self, value: Any) -> "LLMCall":
        if self.tracer is not None:
            self.span.set_attribute("gen_ai.output.messages", self.tracer.content(value))
        return self

    def first_token(self) -> None:
        if self.first_token_at is None:
            self.first_token_at = time.monotonic()
            self.span.set_attribute("gen_ai.response.time_to_first_chunk", round(self.first_token_at - self.start, 4))

    def fail(self, exc: BaseException) -> None:
        err = classify_error(exc)
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            self.span.set_attribute("http.response.status_code", status)
        self.span.record_exception(exc, err)

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        s = self.span
        if self._retries:
            s.set_attribute("alye.retries", len(self._retries))
            statuses = [x for x in self._retries if isinstance(x, int)]
            if statuses:
                s.set_attribute("alye.retry_statuses", statuses)
        if not self._usage_seen and getattr(s, "status", "") != "error":
            # The call returned but the provider sent no usage: it may still have been billed.
            s.set_attribute("alye.usage_unknown", True)
        s.end()


def _int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def get(obj: Any, *path: str, default: Any = None) -> Any:
    """getattr/dict lookup along a path, returning default on any miss."""
    cur = obj
    for p in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            cur = getattr(cur, p, None)
    return default if cur is None else cur


def call_tracer() -> Optional[Tracer]:
    return get_tracer()


# ---- stream wrappers ------------------------------------------------------
class SyncStreamProxy:
    """Wraps a provider stream: passes every chunk through unchanged and ends the span when the stream ends."""

    def __init__(self, inner: Any, call: LLMCall, on_chunk: Callable[[Any], None], on_done: Callable[[], None]) -> None:
        self._inner = inner
        self._call = call
        self._on_chunk = on_chunk
        self._on_done = on_done
        self._iter: Any = None
        self._finished = False

    def _finish(self, exc: Optional[BaseException] = None) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            if exc is not None:
                self._call.fail(exc)
            else:
                self._on_done()
        except Exception:
            log.debug("alye: stream finish failed", exc_info=True)
        self._call.end()

    def __iter__(self) -> "SyncStreamProxy":
        return self

    def __next__(self) -> Any:
        if self._iter is None:
            self._iter = iter(self._inner)
        try:
            chunk = next(self._iter)
        except StopIteration:
            self._finish()
            raise
        except BaseException as e:
            self._finish(e)
            raise
        self._call.first_token()
        try:
            self._on_chunk(chunk)
        except Exception:
            log.debug("alye: chunk handler failed", exc_info=True)
        return chunk

    def __enter__(self) -> "SyncStreamProxy":
        enter = getattr(self._inner, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, *exc: Any) -> Any:
        try:
            ex = getattr(self._inner, "__exit__", None)
            return ex(*exc) if ex is not None else None
        finally:
            self._finish(exc[1] if exc and exc[1] is not None else None)

    def close(self) -> None:
        try:
            fn = getattr(self._inner, "close", None)
            if fn is not None:
                fn()
        finally:
            self._finish()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __del__(self) -> None:
        if not self._finished:
            # Stream dropped without being fully read: record what we have.
            self._call.span.set_attribute("alye.stream_abandoned", True)
            self._finish()


class AsyncStreamProxy:
    def __init__(self, inner: Any, call: LLMCall, on_chunk: Callable[[Any], None], on_done: Callable[[], None]) -> None:
        self._inner = inner
        self._call = call
        self._on_chunk = on_chunk
        self._on_done = on_done
        self._iter: Any = None
        self._finished = False

    _finish = SyncStreamProxy._finish

    def __aiter__(self) -> "AsyncStreamProxy":
        return self

    async def __anext__(self) -> Any:
        if self._iter is None:
            self._iter = self._inner.__aiter__()
        try:
            chunk = await self._iter.__anext__()
        except StopAsyncIteration:
            self._finish()
            raise
        except BaseException as e:
            self._finish(e)
            raise
        self._call.first_token()
        try:
            self._on_chunk(chunk)
        except Exception:
            log.debug("alye: chunk handler failed", exc_info=True)
        return chunk

    async def __aenter__(self) -> "AsyncStreamProxy":
        enter = getattr(self._inner, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        try:
            ex = getattr(self._inner, "__aexit__", None)
            return await ex(*exc) if ex is not None else None
        finally:
            self._finish(exc[1] if exc and exc[1] is not None else None)

    async def close(self) -> None:
        try:
            fn = getattr(self._inner, "close", None) or getattr(self._inner, "aclose", None)
            if fn is not None:
                res = fn()
                if hasattr(res, "__await__"):
                    await res
        finally:
            self._finish()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __del__(self) -> None:
        if not self._finished:
            self._call.span.set_attribute("alye.stream_abandoned", True)
            self._finish()


# ---- patch bookkeeping ----------------------------------------------------
class Patcher:
    """Replaces methods on classes and can put the originals back."""

    def __init__(self) -> None:
        self._saved: List[Any] = []

    def wrap(self, owner: Any, name: str, make: Callable[[Callable[..., Any]], Callable[..., Any]]) -> bool:
        original = owner.__dict__.get(name)
        if original is None or getattr(original, "__alye_wrapped__", False):
            return False
        wrapped = make(original)
        try:
            import functools

            functools.update_wrapper(wrapped, original)
        except Exception:
            pass
        wrapped.__alye_wrapped__ = True  # type: ignore[attr-defined]
        setattr(owner, name, wrapped)
        self._saved.append((owner, name, original))
        return True

    def undo(self) -> None:
        for owner, name, original in reversed(self._saved):
            setattr(owner, name, original)
        self._saved.clear()

    @property
    def count(self) -> int:
        return len(self._saved)
