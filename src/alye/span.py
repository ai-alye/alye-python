"""A minimal span model that serializes to OTLP/HTTP JSON."""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

SPAN_KIND = {"internal": 1, "server": 2, "client": 3, "producer": 4, "consumer": 5}
STATUS = {"unset": 0, "ok": 1, "error": 2}


def new_trace_id() -> str:
    return os.urandom(16).hex()


def new_span_id() -> str:
    return os.urandom(8).hex()


def _otlp_value(v: Any) -> Dict[str, Any]:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [_otlp_value(x) for x in v]}}
    return {"stringValue": str(v)}


def otlp_attributes(attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"key": k, "value": _otlp_value(v)} for k, v in attrs.items() if v is not None]


class Span:
    __slots__ = (
        "trace_id",
        "span_id",
        "parent_span_id",
        "name",
        "kind",
        "start_ns",
        "end_ns",
        "attributes",
        "events",
        "status",
        "status_message",
        "sampled",
        "_on_end",
    )

    def __init__(
        self,
        name: str,
        *,
        trace_id: str,
        parent_span_id: Optional[str],
        kind: str = "internal",
        sampled: bool = True,
        attributes: Optional[Dict[str, Any]] = None,
        on_end: Optional[Callable[["Span"], None]] = None,
    ) -> None:
        self.trace_id = trace_id
        self.span_id = new_span_id()
        self.parent_span_id = parent_span_id
        self.name = name
        self.kind = kind
        self.start_ns = time.time_ns()
        self.end_ns: Optional[int] = None
        self.attributes: Dict[str, Any] = {}
        self.events: List[Tuple[str, int, Dict[str, Any]]] = []
        self.status = "unset"
        self.status_message: Optional[str] = None
        self.sampled = sampled
        self._on_end = on_end
        if attributes:
            self.set_attributes(attributes)

    # ---- recording -------------------------------------------------------
    @property
    def is_recording(self) -> bool:
        return self.end_ns is None

    def set_attribute(self, key: str, value: Any) -> "Span":
        if value is not None and self.end_ns is None:
            self.attributes[key] = value
        return self

    def set_attributes(self, attrs: Dict[str, Any]) -> "Span":
        for k, v in attrs.items():
            self.set_attribute(k, v)
        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "Span":
        if self.end_ns is None:
            self.events.append((name, time.time_ns(), dict(attributes or {})))
        return self

    def set_status(self, status: str, message: Optional[str] = None) -> "Span":
        if self.end_ns is None:
            self.status = status
            self.status_message = message
        return self

    def record_exception(self, exc: BaseException, error_type: Optional[str] = None) -> "Span":
        self.set_attribute("error.type", error_type or type(exc).__qualname__)
        self.add_event(
            "exception",
            {
                "exception.type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "exception.message": str(exc)[:2000],
                "exception.stacktrace": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:],
            },
        )
        self.set_status("error", str(exc)[:500])
        return self

    def end(self) -> None:
        if self.end_ns is not None:
            return
        self.end_ns = time.time_ns()
        if self._on_end is not None:
            self._on_end(self)

    @property
    def duration_ms(self) -> float:
        end = self.end_ns if self.end_ns is not None else time.time_ns()
        return (end - self.start_ns) / 1e6

    # ---- export ----------------------------------------------------------
    def to_otlp(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "name": self.name,
            "kind": SPAN_KIND.get(self.kind, 1),
            "startTimeUnixNano": str(self.start_ns),
            "endTimeUnixNano": str(self.end_ns or time.time_ns()),
            "attributes": otlp_attributes(self.attributes),
            "status": {"code": STATUS[self.status]},
        }
        if self.parent_span_id:
            out["parentSpanId"] = self.parent_span_id
        if self.status_message:
            out["status"]["message"] = self.status_message
        if self.events:
            out["events"] = [{"name": n, "timeUnixNano": str(t), "attributes": otlp_attributes(a)} for n, t, a in self.events]
        return out

    def __repr__(self) -> str:
        return f"<Span {self.name!r} trace={self.trace_id[:8]} {self.status} {self.duration_ms:.1f}ms>"


class NoopSpan:
    """Returned when the SDK is not initialized or is disabled, so instrumented code never fails."""

    trace_id = "0" * 32
    span_id = "0" * 16
    parent_span_id = None
    name = ""
    attributes: Dict[str, Any] = {}
    is_recording = False
    sampled = False

    def set_attribute(self, *_: Any, **__: Any) -> "NoopSpan":
        return self

    set_attributes = add_event = set_status = record_exception = set_attribute

    def end(self) -> None:
        return None


NOOP_SPAN = NoopSpan()
