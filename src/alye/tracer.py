"""Creates spans, links them into traces, and applies sampling and context tags."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar

from .config import Config
from .context import get_current_span, reset_current_span, set_current_span, tag_attributes
from .redact import Redactor
from .span import NOOP_SPAN, Span, new_trace_id

F = TypeVar("F", bound=Callable[..., Any])


class Tracer:
    def __init__(self, config: Config, processor: Any) -> None:
        self.config = config
        self.processor = processor
        self.redactor = Redactor(config.redact_pii, config.redact_patterns, config.redact)

    def _sample(self, trace_id: str) -> bool:
        rate = self.config.sample_rate
        if rate >= 1.0:
            return True
        if rate <= 0.0:
            return False
        # Deterministic per trace so every span in a trace gets the same decision.
        return int(trace_id[-8:], 16) / 0xFFFFFFFF < rate

    def start_span(self, name: str, *, kind: str = "internal", attributes: Optional[Dict[str, Any]] = None) -> Span:
        parent = get_current_span()
        if isinstance(parent, Span):
            trace_id, parent_id, sampled = parent.trace_id, parent.span_id, parent.sampled
        else:
            trace_id, parent_id = new_trace_id(), None
            sampled = self._sample(trace_id)
        span = Span(name, trace_id=trace_id, parent_span_id=parent_id, kind=kind, sampled=sampled, on_end=self.processor.on_end)
        span.set_attributes(tag_attributes())
        if attributes:
            span.set_attributes(attributes)
        return span

    @contextmanager
    def span(self, name: str, *, kind: str = "internal", **attributes: Any) -> Iterator[Span]:
        s = self.start_span(name, kind=kind, attributes=attributes)
        token = set_current_span(s)
        try:
            yield s
        except BaseException as e:
            if not isinstance(e, (GeneratorExit, asyncio.CancelledError)):
                s.record_exception(e)
            raise
        finally:
            reset_current_span(token)
            s.end()

    # ---- content capture --------------------------------------------------
    def content(self, value: Any) -> Optional[str]:
        """Serialize, redact and truncate a prompt or response, or return None when capture is off."""
        if not self.config.capture_content or value is None:
            return None
        try:
            text = value if isinstance(value, str) else json.dumps(value, default=_jsonable, ensure_ascii=False)
        except Exception:
            text = str(value)
        text = self.redactor(text)
        limit = self.config.max_content_chars
        return text if len(text) <= limit else text[:limit] + f"…[truncated {len(text) - limit} chars]"


def _jsonable(o: Any) -> Any:
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(o, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return str(o)


# ---- module-level helpers that work even before init() -------------------
_tracer: Optional[Tracer] = None


def get_tracer() -> Optional[Tracer]:
    return _tracer


def set_tracer(t: Optional[Tracer]) -> None:
    global _tracer
    _tracer = t


@contextmanager
def span(name: str, *, kind: str = "internal", **attributes: Any) -> Iterator[Any]:
    """Group work under a named span, for example one agent run:

        with alye.span("answer_ticket", ticket_id=t.id):
            ...
    """
    t = _tracer
    if t is None:
        yield NOOP_SPAN
        return
    with t.span(name, kind=kind, **attributes) as s:
        yield s


def trace(name: Optional[str] = None, *, kind: str = "internal") -> Callable[[F], F]:
    """Decorator that wraps a function (sync or async) in a span named after it."""

    def deco(fn: F) -> F:
        span_name = name or fn.__qualname__
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                with span(span_name, kind=kind, **{"code.function": fn.__qualname__}):
                    return await fn(*args, **kwargs)

            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name, kind=kind, **{"code.function": fn.__qualname__}):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return deco


def current_span() -> Any:
    s = get_current_span()
    return s if s is not None else NOOP_SPAN
