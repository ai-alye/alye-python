"""Exporters and the background batch processor.

Spans are queued without blocking the caller and sent in batches from a daemon thread.
If the queue is full, new spans are dropped and counted (reported in the heartbeat) rather
than slowing your service down.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import queue
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ._version import __version__
from .span import Span, otlp_attributes

log = logging.getLogger("alye")


class SpanExporter(Protocol):
    def export(self, spans: Sequence[Span]) -> bool: ...

    def shutdown(self) -> None: ...


class OTLPHttpExporter:
    """Sends spans as OTLP/HTTP JSON to `{endpoint}/v1/traces`, gzip-compressed."""

    def __init__(self, endpoint: str, api_key: Optional[str], resource: Dict[str, Any], timeout: float = 5.0, max_attempts: int = 3) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.resource = resource
        self.timeout = timeout
        self.max_attempts = max_attempts

    def _headers(self) -> Dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Content-Encoding": "gzip",
            "User-Agent": f"alye-python/{__version__} python/{sys.version_info[0]}.{sys.version_info[1]}",
        }
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _post(self, path: str, body: Dict[str, Any]) -> bool:
        data = gzip.compress(json.dumps(body, separators=(",", ":"), default=str).encode("utf-8"))
        url = self.endpoint + path
        for attempt in range(1, self.max_attempts + 1):
            req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp.read()
                    return True
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    log.error("alye: the ingest endpoint rejected the API key (HTTP %s). Check ALYE_API_KEY.", e.code)
                    return False
                if e.code not in (408, 429) and e.code < 500:
                    log.warning("alye: ingest returned HTTP %s; dropping %s", e.code, path)
                    return False
                wait = _retry_after(e.headers.get("Retry-After")) if e.headers else None
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                log.debug("alye: send to %s failed: %s", url, e)
                wait = None
            if attempt < self.max_attempts:
                time.sleep(wait if wait is not None else min(8.0, 0.5 * 2 ** (attempt - 1)) * (0.5 + random.random()))
        log.warning("alye: could not reach %s after %s attempts", url, self.max_attempts)
        return False

    def export(self, spans: Sequence[Span]) -> bool:
        body = {
            "resourceSpans": [
                {
                    "resource": {"attributes": otlp_attributes(self.resource)},
                    "scopeSpans": [{"scope": {"name": "alye", "version": __version__}, "spans": [s.to_otlp() for s in spans]}],
                }
            ]
        }
        return self._post("/v1/traces", body)

    def send_heartbeat(self, payload: Dict[str, Any]) -> bool:
        return self._post("/v1/heartbeat", payload)

    def shutdown(self) -> None:
        return None


class ConsoleExporter:
    """Prints one line per model call. Useful in local development."""

    def __init__(self, stream: Any = None) -> None:
        self.stream = stream or sys.stderr

    def export(self, spans: Sequence[Span]) -> bool:
        for s in spans:
            a = s.attributes
            if "gen_ai.operation.name" not in a and s.parent_span_id:
                continue
            model = a.get("gen_ai.response.model") or a.get("gen_ai.request.model") or ""
            tokens = ""
            if "gen_ai.usage.input_tokens" in a or "gen_ai.usage.output_tokens" in a:
                tokens = f" in={a.get('gen_ai.usage.input_tokens', '?')} out={a.get('gen_ai.usage.output_tokens', '?')}"
            status = "ERROR " + str(a.get("error.type", "")) if s.status == "error" else "ok"
            retries = f" retries={a['alye.retries']}" if a.get("alye.retries") else ""
            cust = f" customer={a['alye.customer_id']}" if a.get("alye.customer_id") else ""
            print(f"[alye] {s.name} {model}{tokens} {s.duration_ms:.0f}ms {status}{retries}{cust}", file=self.stream)
        return True

    def send_heartbeat(self, payload: Dict[str, Any]) -> bool:
        return True

    def shutdown(self) -> None:
        return None


class InMemoryExporter:
    """Keeps finished spans in a list. For tests."""

    def __init__(self) -> None:
        self.spans: List[Span] = []
        self.heartbeats: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def export(self, spans: Sequence[Span]) -> bool:
        with self._lock:
            self.spans.extend(spans)
        return True

    def send_heartbeat(self, payload: Dict[str, Any]) -> bool:
        self.heartbeats.append(payload)
        return True

    def clear(self) -> None:
        with self._lock:
            self.spans.clear()

    def shutdown(self) -> None:
        return None


class NoopExporter:
    def export(self, spans: Sequence[Span]) -> bool:
        return True

    def send_heartbeat(self, payload: Dict[str, Any]) -> bool:
        return True

    def shutdown(self) -> None:
        return None


def _retry_after(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    try:
        return max(0.0, min(30.0, float(raw)))
    except ValueError:
        return None


_FLUSH = object()


class BatchProcessor:
    """Queues finished spans and exports them in batches from a daemon thread."""

    def __init__(self, exporter: Any, *, max_queue: int, max_batch: int, flush_interval: float) -> None:
        self.exporter = exporter
        self.max_batch = max_batch
        self.flush_interval = flush_interval
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        self.exported = 0
        self.failed = 0
        self._stop = threading.Event()
        self._flushed = threading.Condition()
        self._flush_seq = 0
        self._thread: Optional[threading.Thread] = None
        self._start()
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork)

    def _start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="alye-export", daemon=True)
        self._thread.start()

    def _after_fork(self) -> None:
        # Threads do not survive fork; start a fresh queue and worker in the child (e.g. gunicorn workers).
        if self._stop.is_set():
            return
        self._queue = queue.Queue(maxsize=self._queue.maxsize)
        self._flushed = threading.Condition()
        self._start()

    def on_end(self, span: Span) -> None:
        if not span.sampled and span.status != "error":
            return
        try:
            self._queue.put_nowait(span)
        except queue.Full:
            self.dropped += 1

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def _export(self, batch: List[Span]) -> None:
        if not batch:
            return
        try:
            ok = self.exporter.export(batch)
        except Exception:  # never let an exporter bug kill the worker
            log.exception("alye: exporter raised")
            ok = False
        if ok:
            self.exported += len(batch)
        else:
            self.failed += len(batch)

    def _run(self) -> None:
        batch: List[Span] = []
        deadline = time.monotonic() + self.flush_interval
        while True:
            timeout = max(0.0, deadline - time.monotonic())
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                item = None
            if item is _FLUSH:
                self._export(batch)
                batch = []
                with self._flushed:
                    self._flush_seq += 1
                    self._flushed.notify_all()
                deadline = time.monotonic() + self.flush_interval
                continue
            if item is not None:
                batch.append(item)
            if len(batch) >= self.max_batch or time.monotonic() >= deadline:
                self._export(batch)
                batch = []
                deadline = time.monotonic() + self.flush_interval
            if self._stop.is_set() and self._queue.empty():
                self._export(batch)
                return

    def flush(self, timeout: float = 5.0) -> bool:
        """Block until everything queued so far has been handed to the exporter."""
        if self._thread is None or not self._thread.is_alive():
            return False
        with self._flushed:
            target = self._flush_seq + 1
            try:
                self._queue.put(_FLUSH, timeout=timeout)
            except queue.Full:
                return False
            end = time.monotonic() + timeout
            while self._flush_seq < target:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._flushed.wait(left)
        return True

    def shutdown(self, timeout: float = 5.0) -> None:
        self.flush(timeout)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.exporter.shutdown()
        except Exception:
            log.debug("alye: exporter shutdown failed", exc_info=True)
