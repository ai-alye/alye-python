import anthropic
import httpx2
import pytest

import alye
from conftest import sse

MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "Refund issued."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 20, "output_tokens": 15, "cache_read_input_tokens": 300, "cache_creation_input_tokens": 50},
}

EVENTS = [
    {"type": "message_start", "message": {**MESSAGE, "content": [], "stop_reason": None, "usage": {"input_tokens": 12, "output_tokens": 1, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 0}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 7}},
    {"type": "message_stop"},
]


def client(handler, *, max_retries=2, is_async=False):
    if is_async:
        return anthropic.AsyncAnthropic(api_key="sk-ant-test", base_url="http://test", max_retries=max_retries, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    return anthropic.Anthropic(api_key="sk-ant-test", base_url="http://test", max_retries=max_retries, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


def stream_response(req):
    return httpx2.Response(200, content=sse(EVENTS, named=True), headers={"content-type": "text/event-stream"})


def test_messages_create(mon):
    msg = client(lambda r: httpx2.Response(200, json=MESSAGE)).messages.create(model="claude-sonnet-5-5", max_tokens=500, messages=[{"role": "user", "content": "hi"}])
    assert msg.content[0].text == "Refund issued."
    s = mon.one()
    a = s.attributes
    assert s.name == "chat claude-sonnet-5-5"
    assert a["gen_ai.provider.name"] == "anthropic"
    # total input = uncached + cache read + cache write
    assert a["gen_ai.usage.input_tokens"] == 370
    assert a["gen_ai.usage.cache_read.input_tokens"] == 300 and a["gen_ai.usage.cache_creation.input_tokens"] == 50
    assert a["gen_ai.usage.output_tokens"] == 15
    assert a["gen_ai.response.finish_reasons"] == ["end_turn"]
    assert a["gen_ai.request.max_tokens"] == 500


def test_create_stream_true(mon):
    stream = client(stream_response).messages.create(model="claude-sonnet-5-5", max_tokens=100, messages=[], stream=True)
    types = [e.type for e in stream]
    assert types[0] == "message_start" and types[-1] == "message_stop"
    a = mon.one().attributes
    assert a["gen_ai.usage.input_tokens"] == 112 and a["gen_ai.usage.output_tokens"] == 7
    assert a["gen_ai.response.finish_reasons"] == ["end_turn"]
    assert a["gen_ai.response.time_to_first_chunk"] >= 0


def test_stream_helper(make_mon):
    m = make_mon(capture_content=True)
    with client(stream_response).messages.stream(model="claude-sonnet-5-5", max_tokens=100, messages=[{"role": "user", "content": "hi"}]) as stream:
        text = "".join(stream.text_stream)
    assert text == "Hello"
    a = m.one().attributes
    assert a["gen_ai.usage.output_tokens"] == 7 and a["gen_ai.usage.input_tokens"] == 112
    assert a["alye.streaming"] is True
    assert "Hello" in a["gen_ai.output.messages"]


def test_overloaded_then_success_counts_retry(mon):
    n = {"i": 0}

    def handler(r):
        n["i"] += 1
        if n["i"] == 1:
            return httpx2.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}, headers={"retry-after-ms": "1"})
        return httpx2.Response(200, json=MESSAGE)

    client(handler).messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[])
    a = mon.one().attributes
    assert a["alye.retries"] == 1 and a["alye.retry_statuses"] == [529]


def test_overloaded_error(mon):
    c = client(lambda r: httpx2.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}), max_retries=0)
    with pytest.raises(anthropic.APIStatusError):
        c.messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[])
    s = mon.one()
    assert s.status == "error" and s.attributes["error.type"] == "overloaded"


async def test_async_create_and_stream(mon):
    def handler(r):
        import json

        return stream_response(r) if json.loads(r.content).get("stream") else httpx2.Response(200, json=MESSAGE)

    c = client(handler, is_async=True)
    await c.messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[])
    stream = await c.messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[], stream=True)
    async for _ in stream:
        pass
    async with c.messages.stream(model="claude-sonnet-5-5", max_tokens=10, messages=[]) as s:
        await s.get_final_message()
    outs = sorted(x.attributes["gen_ai.usage.output_tokens"] for x in mon.llm())
    assert outs == [7, 7, 15]


def test_with_raw_response(mon):
    # LangChain calls messages.with_raw_response.create and parses the result itself
    c = client(lambda r: stream_response(r) if b'"stream":true' in r.content.replace(b" ", b"") else httpx2.Response(200, json=MESSAGE))
    raw = c.messages.with_raw_response.create(model="claude-sonnet-5-5", max_tokens=10, messages=[])
    assert raw.parse().content[0].text == "Refund issued."
    raw = c.messages.with_raw_response.create(model="claude-sonnet-5-5", max_tokens=10, messages=[], stream=True)
    assert [e.type for e in raw.parse()][-1] == "message_stop"
    outs = sorted(x.attributes["gen_ai.usage.output_tokens"] for x in mon.llm())
    assert outs == [7, 15]


async def test_async_with_raw_response(mon):
    def handler(r):
        import json

        return stream_response(r) if json.loads(r.content).get("stream") else httpx2.Response(200, json=MESSAGE)

    c = client(handler, is_async=True)
    raw = await c.messages.with_raw_response.create(model="claude-sonnet-5-5", max_tokens=10, messages=[])
    assert (await raw.parse()).content[0].text == "Refund issued."
    raw = await c.messages.with_raw_response.create(model="claude-sonnet-5-5", max_tokens=10, messages=[], stream=True)
    async for _ in await raw.parse():
        pass
    outs = sorted(x.attributes["gen_ai.usage.output_tokens"] for x in mon.llm())
    assert outs == [7, 15]


def test_beta_messages(mon):
    msg = client(lambda r: httpx2.Response(200, json=MESSAGE)).beta.messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[])
    assert msg.content[0].text == "Refund issued."
    assert mon.one().attributes["gen_ai.usage.output_tokens"] == 15


def test_nested_in_agent_span(mon):
    with alye.span("answer_ticket"):
        client(lambda r: httpx2.Response(200, json=MESSAGE)).messages.create(model="claude-sonnet-5-5", max_tokens=10, messages=[])
    spans = {s.name: s for s in mon.spans()}
    assert spans["chat claude-sonnet-5-5"].parent_span_id == spans["answer_ticket"].span_id
