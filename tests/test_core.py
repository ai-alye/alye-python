import asyncio
import io

import pytest

import alye
from alye import InMemoryExporter
from alye.devserver import DevServer
from alye.redact import Redactor


def test_noop_before_init():
    # Instrumented code must work even when init() was never called.
    with alye.span("x") as s:
        s.set_attribute("a", 1)
    with alye.llm_call("voyage", "voyage-3") as call:
        call.set_usage(10, 0)
    assert not alye.is_enabled()


def test_spans_nest_into_one_trace(mon):
    with alye.span("agent_run") as root:
        with alye.span("retrieve") as child:
            pass
    spans = {s.name: s for s in mon.spans()}
    assert spans["retrieve"].trace_id == spans["agent_run"].trace_id
    assert spans["retrieve"].parent_span_id == spans["agent_run"].span_id
    assert spans["agent_run"].parent_span_id is None


def test_context_tags_are_attached(mon):
    with alye.context(customer_id="northwind", session_id="s1", agent="reply-agent", region="eu"):
        with alye.context(session_id="s2"):
            with alye.llm_call("voyage", "voyage-3", operation="embeddings") as call:
                call.set_usage(42, 0)
    s = mon.one()
    assert s.attributes["alye.customer_id"] == "northwind"
    assert s.attributes["session.id"] == "s2"
    assert s.attributes["gen_ai.agent.name"] == "reply-agent"
    assert s.attributes["alye.tag.region"] == "eu"
    assert s.attributes["gen_ai.usage.input_tokens"] == 42
    assert s.name == "embeddings voyage-3"
    assert alye.current_tags() == {}


def test_trace_decorator_sync_and_async(mon):
    @alye.trace()
    def work(x):
        return x * 2

    @alye.trace("async_work")
    async def awork(x):
        await asyncio.sleep(0)
        return x + 1

    assert work(2) == 4
    assert asyncio.run(awork(1)) == 2
    names = {s.name for s in mon.spans()}
    assert "test_trace_decorator_sync_and_async.<locals>.work" in names
    assert "async_work" in names


def test_exception_is_recorded_and_reraised(mon):
    with pytest.raises(ValueError):
        with alye.span("boom"):
            raise ValueError("bad input")
    s = mon.spans()[0]
    assert s.status == "error"
    assert s.attributes["error.type"] == "ValueError"
    assert s.events[0][0] == "exception"


def test_sampling_drops_ok_traces_but_keeps_errors(make_mon):
    m = make_mon(sample_rate=0.0)
    with alye.span("fine"):
        pass
    with pytest.raises(RuntimeError):
        with alye.span("broken"):
            raise RuntimeError("x")
    assert [s.name for s in m.spans()] == ["broken"]


def test_content_not_captured_by_default(mon):
    with alye.llm_call("x", "m") as call:
        call.set_input([{"role": "user", "content": "hello"}])
    assert "gen_ai.input.messages" not in mon.one().attributes


def test_content_captured_and_redacted(make_mon):
    m = make_mon(capture_content=True, max_content_chars=200)
    with alye.llm_call("x", "m") as call:
        call.set_input("Mail me at jane.doe@example.com or +1 (415) 555-0134, card 4111 1111 1111 1111, key sk-abcdefghijklmnopqrstuv. Order #A-55120.")
    text = m.one().attributes["gen_ai.input.messages"]
    assert "[EMAIL]" in text and "[PHONE]" in text and "[CARD]" in text and "[SECRET]" in text
    assert "jane.doe" not in text and "4111" not in text
    assert "#A-55120" in text  # short order numbers are left alone


def test_redactor_custom_patterns():
    r = Redactor(True, extra=[(r"TICKET-\d+", "[TICKET]")], custom=lambda t: t.replace("Acme", "[ORG]"))
    assert r("Acme TICKET-991 from a@b.co") == "[ORG] [TICKET] from [EMAIL]"
    assert Redactor(False)("a@b.co") == "a@b.co"


def test_long_content_is_truncated(make_mon):
    m = make_mon(capture_content=True, max_content_chars=50)
    with alye.llm_call("x", "m") as call:
        call.set_output("y" * 500)
    assert m.one().attributes["gen_ai.output.messages"].endswith("[truncated 450 chars]")


def test_usage_unknown_flag(mon):
    with alye.llm_call("x", "m"):
        pass
    assert mon.one().attributes["alye.usage_unknown"] is True


def test_end_to_end_over_http_to_devserver():
    out = io.StringIO()
    srv = DevServer(port=0, out=out)
    srv.start_background()
    port = srv.server_address[1]
    try:
        alye.init(service="http-svc", env="staging", endpoint=f"http://127.0.0.1:{port}", flush_interval=0.05, heartbeat_interval=0.2)
        with alye.context(customer_id="contoso"):
            with alye.llm_call("anthropic", "claude-sonnet") as call:
                call.set_usage(100, 20, cache_read_tokens=60)
        assert alye.flush(5)
        alye.shutdown(2)
    finally:
        srv.shutdown()
    assert len(srv.spans) == 1
    s = srv.spans[0]
    assert s["service"] == "http-svc" and s["env"] == "staging"
    assert s["attributes"]["gen_ai.usage.input_tokens"] == 100
    assert s["attributes"]["gen_ai.usage.cache_read.input_tokens"] == 60
    assert s["attributes"]["alye.customer_id"] == "contoso"
    assert srv.heartbeats and srv.heartbeats[0]["service"] == "http-svc"
    assert "claude-sonnet" in out.getvalue()


def test_queue_full_drops_instead_of_blocking():
    import threading
    import time

    release = threading.Event()

    class SlowExporter(InMemoryExporter):
        def export(self, spans):
            release.wait(5)
            return super().export(spans)

    exp = SlowExporter()
    alye.init(service="q", exporter=exp, max_queue=1, max_batch=1, flush_interval=0.01, heartbeat_interval=0)
    start = time.monotonic()
    for i in range(20):
        with alye.span(f"s{i}"):
            pass
    assert time.monotonic() - start < 1.0  # never blocks the caller
    assert alye.__dict__["_processor"].dropped > 0
    release.set()
    alye.shutdown(2)


def test_init_twice_reconfigures():
    a, b = InMemoryExporter(), InMemoryExporter()
    alye.init(service="one", exporter=a, heartbeat_interval=0, flush_interval=0.05)
    alye.init(service="two", exporter=b, heartbeat_interval=0, flush_interval=0.05)
    with alye.span("x"):
        pass
    alye.flush()
    alye.shutdown()
    assert len(b.spans) == 1 and not a.spans


def test_disabled_records_nothing():
    alye.init(service="off", disabled=True)
    assert not alye.is_enabled()
    alye.shutdown()


def test_env_vars(monkeypatch):
    monkeypatch.setenv("ALYE_SERVICE", "from-env")
    monkeypatch.setenv("ALYE_ENV", "uat")
    monkeypatch.setenv("ALYE_SAMPLE_RATE", "0.5")
    exp = InMemoryExporter()
    alye.init(exporter=exp, heartbeat_interval=0)
    cfg = alye.config()
    assert (cfg.service, cfg.env, cfg.sample_rate) == ("from-env", "uat", 0.5)
    alye.shutdown()


def test_region_setting(monkeypatch):
    from alye.config import build_config

    assert build_config(service="s").region is None
    assert build_config(service="s", region=" EU ").region == "eu"
    monkeypatch.setenv("ALYE_REGION", "europe-west4")
    assert build_config(service="s").region == "europe-west4"
    assert build_config(service="s", region="us").region == "us"  # explicit argument wins


def test_endpoint_selection(monkeypatch):
    from alye import _make_exporter
    from alye.config import DEFAULT_ENDPOINT, build_config
    from alye.exporter import ConsoleExporter, NoopExporter, OTLPHttpExporter

    for var in ("ALYE_ENDPOINT", "ALYE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # No key, no endpoint: print locally.
    assert isinstance(_make_exporter(build_config(service="s"), {}), ConsoleExporter)
    # A key and no endpoint: the hosted ingest.
    exp = _make_exporter(build_config(service="s", api_key="om_live_x"), {})
    assert isinstance(exp, OTLPHttpExporter) and exp.endpoint == DEFAULT_ENDPOINT == "https://api.alye.ai/ingest"
    # An explicit endpoint wins (self-hosted, local dev server).
    monkeypatch.setenv("ALYE_ENDPOINT", "https://alye.example.com/ingest/")
    exp = _make_exporter(build_config(service="s", api_key="om_live_x"), {})
    assert exp.endpoint == "https://alye.example.com/ingest"
    # Remote endpoint without a key: nothing is sent.
    assert isinstance(_make_exporter(build_config(service="s"), {}), NoopExporter)
