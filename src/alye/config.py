"""SDK settings, read from `init()` arguments first and environment variables second."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple, Union

log = logging.getLogger("alye")

TRUE = {"1", "true", "yes", "on"}
FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    v = raw.strip().lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    log.warning("alye: ignoring %s=%r, expected true or false", name, raw)
    return None


def _env_float(name: str) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        log.warning("alye: ignoring %s=%r, expected a number", name, raw)
        return None


RedactPattern = Tuple[Union[str, "re.Pattern[str]"], str]

# Alye's hosted ingest. Used when an API key is set and no endpoint is given;
# ALYE_ENDPOINT overrides it (self-hosted, or the local dev server).
DEFAULT_ENDPOINT = "https://api.alye.ai/ingest"


@dataclass
class Config:
    service: str
    api_key: Optional[str] = None
    env: str = "production"
    endpoint: Optional[str] = None
    # Prompts and responses are not stored unless you turn this on.
    capture_content: bool = False
    redact_pii: bool = True
    redact_patterns: Sequence[RedactPattern] = ()
    redact: Optional[Callable[[str], str]] = None
    max_content_chars: int = 8000
    sample_rate: float = 1.0
    # "otlp" sends to `endpoint`; "console" prints; "none" drops; or pass an exporter object.
    exporter: Any = "otlp"
    # Which provider SDKs to patch. None means every supported one that is installed.
    instrument: Optional[Sequence[str]] = None
    # Ask OpenAI streams to include token usage (adds stream_options.include_usage when you did not set it).
    inject_stream_usage: bool = True
    flush_interval: float = 2.0
    max_batch: int = 256
    max_queue: int = 10_000
    timeout: float = 5.0
    heartbeat_interval: float = 60.0
    release: Optional[str] = None
    # Where this service's model calls are processed, for regional pricing: "global", a data zone
    # ("us", "eu", "apac") or a cloud region ("europe-west4"). Usually not needed: Bedrock and
    # Vertex regions are read from the endpoint. Set it for Azure or when you know better.
    region: Optional[str] = None
    debug: bool = False
    disabled: bool = False
    resource_attributes: dict = field(default_factory=dict)


def build_config(**kwargs: Any) -> Config:
    """Merge explicit arguments with environment variables. Explicit arguments win."""

    def pick(key: str, env_value: Any) -> Any:
        v = kwargs.get(key)
        return v if v is not None else env_value

    service = pick("service", os.environ.get("ALYE_SERVICE") or os.environ.get("OTEL_SERVICE_NAME"))
    if not service:
        log.warning("alye: no service name given; using 'unknown_service'. Pass service='your-service' to init().")
        service = "unknown_service"

    cfg = Config(service=str(service))
    cfg.api_key = pick("api_key", os.environ.get("ALYE_API_KEY"))
    cfg.env = pick("env", os.environ.get("ALYE_ENV")) or cfg.env
    cfg.endpoint = pick("endpoint", os.environ.get("ALYE_ENDPOINT"))
    cfg.capture_content = bool(pick("capture_content", _env_bool("ALYE_CAPTURE_CONTENT")) or False)
    rp = pick("redact_pii", _env_bool("ALYE_REDACT_PII"))
    cfg.redact_pii = True if rp is None else bool(rp)
    sr = pick("sample_rate", _env_float("ALYE_SAMPLE_RATE"))
    cfg.sample_rate = 1.0 if sr is None else float(sr)
    cfg.exporter = pick("exporter", os.environ.get("ALYE_EXPORTER")) or "otlp"
    cfg.debug = bool(pick("debug", _env_bool("ALYE_DEBUG")) or False)
    cfg.disabled = bool(pick("disabled", _env_bool("ALYE_DISABLED")) or False)
    cfg.release = pick("release", os.environ.get("ALYE_RELEASE"))
    region = pick("region", os.environ.get("ALYE_REGION"))
    cfg.region = str(region).strip().lower() if region else None

    for key in (
        "redact_patterns",
        "redact",
        "max_content_chars",
        "instrument",
        "inject_stream_usage",
        "flush_interval",
        "max_batch",
        "max_queue",
        "timeout",
        "heartbeat_interval",
        "resource_attributes",
    ):
        if kwargs.get(key) is not None:
            setattr(cfg, key, kwargs[key])

    if not 0.0 <= cfg.sample_rate <= 1.0:
        log.warning("alye: sample_rate must be between 0 and 1; using 1.0")
        cfg.sample_rate = 1.0

    unknown = set(kwargs) - {f for f in Config.__dataclass_fields__}
    if unknown:
        log.warning("alye: unknown init() options ignored: %s", ", ".join(sorted(unknown)))
    return cfg
