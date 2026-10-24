"""Alye: cost, latency and error tracking for LLM calls.

    import alye

    alye.init(api_key=os.environ["ALYE_API_KEY"], service="support-reply", env="production")

    with alye.context(customer_id="northwind", session_id=ticket.id):   # optional
        answer = client.chat.completions.create(...)                     # recorded automatically
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence

from . import instrumentation
from ._version import __version__
from .config import DEFAULT_ENDPOINT, Config, build_config
from .context import clear_tags, context, current_tags, set_tags
from .exporter import BatchProcessor, ConsoleExporter, InMemoryExporter, NoopExporter, OTLPHttpExporter
from .heartbeat import Heartbeat, base_payload
from .instrumentation._common import LLMCall
from .span import NOOP_SPAN, Span
from .tracer import Tracer, current_span, get_tracer, set_tracer, span, trace

__all__ = [
    "__version__",
    "init",
    "shutdown",
    "flush",
    "is_enabled",
    "context",
    "set_tags",
    "clear_tags",
    "current_tags",
    "span",
    "trace",
    "llm_call",
    "current_span",
    "Span",
    "LLMCall",
    "InMemoryExporter",
    "ConsoleExporter",
    "OTLPHttpExporter",
]

log = logging.getLogger("alye")

_lock = threading.Lock()
_processor: Optional[BatchProcessor] = None
_heartbeat: Optional[Heartbeat] = None
_config: Optional[Config] = None


def _make_exporter(cfg: Config, resource: Dict[str, Any]) -> Any:
    exp = cfg.exporter
    if not isinstance(exp, str):
        return exp  # a custom exporter object
    if exp == "none":
        return NoopExporter()
    if exp == "console":
        return ConsoleExporter()
    if exp != "otlp":
        log.warning("alye: unknown exporter %r; using console", exp)
        return ConsoleExporter()
    endpoint = cfg.endpoint
    if not endpoint:
        if not cfg.api_key:
            log.warning("alye: no API key set (ALYE_API_KEY); printing calls to the console instead of sending them.")
            return ConsoleExporter()
        endpoint = DEFAULT_ENDPOINT
    local = any(h in endpoint for h in ("localhost", "127.0.0.1", "0.0.0.0"))
    if not cfg.api_key and not local:
        log.warning("alye: no API key set (ALYE_API_KEY); calls will not be sent.")
        return NoopExporter()
    return OTLPHttpExporter(endpoint, cfg.api_key, resource, timeout=cfg.timeout)


def init(
    api_key: Optional[str] = None,
    *,
    service: Optional[str] = None,
    env: Optional[str] = None,
    endpoint: Optional[str] = None,
    capture_content: Optional[bool] = None,
    redact_pii: Optional[bool] = None,
    sample_rate: Optional[float] = None,
    exporter: Any = None,
    instrument: Optional[Sequence[str]] = None,
    debug: Optional[bool] = None,
    disabled: Optional[bool] = None,
    **options: Any,
) -> None:
    """Start Alye. Call once at startup, in each service.

    Every argument can also come from an environment variable: ALYE_API_KEY, ALYE_SERVICE
    (or OTEL_SERVICE_NAME), ALYE_ENV, ALYE_ENDPOINT, ALYE_CAPTURE_CONTENT, ALYE_REDACT_PII,
    ALYE_SAMPLE_RATE, ALYE_EXPORTER, ALYE_DEBUG, ALYE_DISABLED, ALYE_REGION.

    ``region="eu"`` (or a cloud region like ``"europe-west4"``) tells the backend where calls are
    processed so regional prices are used; Bedrock and Vertex regions are detected automatically.
    """
    global _processor, _heartbeat, _config
    with _lock:
        if _processor is not None:
            log.warning("alye: init() called again; shutting down the previous setup first")
            _shutdown_locked(timeout=2.0)

        cfg = build_config(
            api_key=api_key,
            service=service,
            env=env,
            endpoint=endpoint,
            capture_content=capture_content,
            redact_pii=redact_pii,
            sample_rate=sample_rate,
            exporter=exporter,
            instrument=instrument,
            debug=debug,
            disabled=disabled,
            **options,
        )
        if cfg.debug:
            logging.getLogger("alye").setLevel(logging.DEBUG)
        if cfg.disabled:
            log.info("alye: disabled; no calls will be recorded")
            return

        resource = {
            "service.name": cfg.service,
            "deployment.environment.name": cfg.env,
            "service.version": cfg.release,
            "cloud.region": cfg.region,
            "telemetry.sdk.name": "alye",
            "telemetry.sdk.language": "python",
            "telemetry.sdk.version": __version__,
            **cfg.resource_attributes,
        }
        exp = _make_exporter(cfg, resource)
        _processor = BatchProcessor(exp, max_queue=cfg.max_queue, max_batch=cfg.max_batch, flush_interval=cfg.flush_interval)
        set_tracer(Tracer(cfg, _processor))
        _config = cfg
        patched = instrumentation.instrument(cfg.instrument)

        base = base_payload(cfg.service, cfg.env, cfg.release, patched)
        proc = _processor

        def payload() -> Dict[str, Any]:
            return {**base, "sent_at": time.time(), "queue_size": proc.queue_size, "spans_exported": proc.exported, "spans_failed": proc.failed, "spans_dropped": proc.dropped}

        send = getattr(exp, "send_heartbeat", None)
        if callable(send):
            _heartbeat = Heartbeat(send, payload, cfg.heartbeat_interval)
        log.info("alye %s: tracking service %r (%s); patched %s", __version__, cfg.service, cfg.env, ", ".join(patched) or "nothing")


def _shutdown_locked(timeout: float) -> None:
    global _processor, _heartbeat, _config
    if _heartbeat is not None:
        _heartbeat.stop()
        _heartbeat = None
    if _processor is not None:
        _processor.shutdown(timeout)
        _processor = None
    set_tracer(None)
    instrumentation.uninstrument()
    _config = None


def shutdown(timeout: float = 5.0) -> None:
    """Send anything still queued, stop background threads and remove the provider patches."""
    with _lock:
        _shutdown_locked(timeout)


def flush(timeout: float = 5.0) -> bool:
    """Send queued calls now. Useful at the end of scripts, jobs and serverless handlers."""
    p = _processor
    return p.flush(timeout) if p is not None else True


def is_enabled() -> bool:
    return get_tracer() is not None


def config() -> Optional[Config]:
    return _config


@contextmanager
def llm_call(provider: str, model: Optional[str] = None, *, operation: str = "chat", **attributes: Any) -> Iterator[LLMCall]:
    """Record a model call to a provider the SDK does not patch automatically (Voyage, Bedrock via boto3, local models...).

        with alye.llm_call("voyage", "voyage-3", operation="embeddings") as call:
            res = vo.embed(texts, model="voyage-3")
            call.set_usage(input_tokens=res.total_tokens, output_tokens=0)
    """
    call = LLMCall(get_tracer(), provider=provider, operation=operation, model=model, attributes=attributes)
    with call:
        yield call


@atexit.register
def _atexit() -> None:
    try:
        shutdown(timeout=float(os.environ.get("ALYE_SHUTDOWN_TIMEOUT", "3")))
    except Exception:
        pass
