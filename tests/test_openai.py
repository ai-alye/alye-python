import json

import httpx2
import openai
import pytest

import alye
from conftest import sse

CHAT = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-5-mini-2026-01-01",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi there"}, "finish_reason": "stop"}],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 100},
        "completion_tokens_details": {"reasoning_tokens": 12},
    },
}


def client(handler, *, base_url="http://test/v1", max_retries=2, is_async=False):
    if is_async:
        return openai.AsyncOpenAI(api_key="sk-test", base_url=base_url, max_retries=max_retries, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))
    return openai.OpenAI(api_key="sk-test", base_url=base_url, max_retries=max_retries, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


def test_chat_completion(mon):
    c = client(lambda req: httpx2.Response(200, json=CHAT))
    with alye.context(customer_id="northwind"):
        r = c.chat.completions.create(model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}], temperature=0.2, max_completion_tokens=100)
    assert r.choices[0].message.content == "Hi there"
    s = mon.one()
    a = s.attributes
    assert s.name == "chat gpt-5-mini" and s.kind == "client"
    assert a["gen_ai.provider.name"] == "openai"
    assert a["gen_ai.response.model"] == "gpt-5-mini-2026-01-01"
    assert a["gen_ai.usage.input_tokens"] == 120 and a["gen_ai.usage.output_tokens"] == 30
    assert a["gen_ai.usage.cache_read.input_tokens"] == 100
    assert a["gen_ai.usage.reasoning.output_tokens"] == 12
    assert a["gen_ai.response.finish_reasons"] == ["stop"]
    assert a["gen_ai.request.temperature"] == 0.2 and a["gen_ai.request.max_tokens"] == 100
    assert a["alye.customer_id"] == "northwind"
    assert "alye.usage_unknown" not in a
    assert s.status != "error"


def test_streaming_chat_injects_usage_and_records_ttft(mon):
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        chunks = [
            {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-5-mini", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}]},
            {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-5-mini", "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
            {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-5-mini", "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11}},
        ]
        return httpx2.Response(200, content=sse(chunks, done=True), headers={"content-type": "text/event-stream"})

    c = client(handler)
    stream = c.chat.completions.create(model="gpt-5-mini", messages=[{"role": "user", "content": "hi"}], stream=True)
    text = "".join((ch.choices[0].delta.content or "") for ch in stream if ch.choices)
    assert text == "Hello"
    assert seen["stream_options"] == {"include_usage": True}
    a = mon.one().attributes
    assert a["gen_ai.usage.input_tokens"] == 9 and a["gen_ai.usage.output_tokens"] == 2
    assert a["alye.streaming"] is True
    assert a["gen_ai.response.time_to_first_chunk"] >= 0
    assert a["gen_ai.response.finish_reasons"] == ["stop"]


def test_stream_used_as_context_manager(mon):
    def handler(req):
        chunks = [{"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}]}]
        return httpx2.Response(200, content=sse(chunks, done=True), headers={"content-type": "text/event-stream"})

    with client(handler).chat.completions.create(model="m", messages=[], stream=True, stream_options={"include_usage": False}) as stream:
        for _ in stream:
            pass
    a = mon.one().attributes
    assert a["alye.usage_unknown"] is True  # caller turned usage off
    assert "alye.stream_usage_injected" not in a


def test_retry_after_429_is_counted(mon):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx2.Response(429, json={"error": {"message": "slow down", "type": "rate_limit"}}, headers={"retry-after-ms": "1"})
        return httpx2.Response(200, json=CHAT)

    client(handler).chat.completions.create(model="gpt-5-mini", messages=[])
    a = mon.one().attributes
    assert a["alye.retries"] == 1 and a["alye.retry_statuses"] == [429]
    assert a["gen_ai.usage.output_tokens"] == 30


def test_error_is_recorded_even_when_swallowed(mon):
    c = client(lambda req: httpx2.Response(400, json={"error": {"message": "bad model", "type": "invalid_request_error"}}), max_retries=0)
    try:
        c.chat.completions.create(model="nope", messages=[])
    except openai.BadRequestError:
        pass  # the app swallows it
    s = mon.one()
    assert s.status == "error"
    assert s.attributes["error.type"] == "invalid_request"
    assert s.attributes["http.response.status_code"] == 400


def test_rate_limit_after_retries_exhausted(mon):
    c = client(lambda req: httpx2.Response(429, json={"error": {"message": "x"}}, headers={"retry-after-ms": "1"}), max_retries=2)
    with pytest.raises(openai.RateLimitError):
        c.chat.completions.create(model="gpt-5-mini", messages=[])
    a = mon.one().attributes
    assert a["error.type"] == "rate_limited" and a["alye.retries"] == 2


async def test_async_chat_and_stream(mon):
    def handler(req):
        body = json.loads(req.content)
        if body.get("stream"):
            chunks = [
                {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": "stop"}]},
                {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}},
            ]
            return httpx2.Response(200, content=sse(chunks, done=True), headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=CHAT)

    c = client(handler, is_async=True)
    r = await c.chat.completions.create(model="gpt-5-mini", messages=[])
    assert r.usage.total_tokens == 150
    stream = await c.chat.completions.create(model="m", messages=[], stream=True)
    got = [ch async for ch in stream]
    assert len(got) == 2
    calls = mon.llm()
    assert len(calls) == 2
    assert {c.attributes["gen_ai.usage.input_tokens"] for c in calls} == {120, 3}


def test_responses_api(mon):
    resp = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "gpt-5",
        "status": "completed",
        "output": [{"type": "message", "id": "m1", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "Done", "annotations": []}]}],
        "usage": {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60, "input_tokens_details": {"cached_tokens": 20}, "output_tokens_details": {"reasoning_tokens": 4}},
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    r = client(lambda req: httpx2.Response(200, json=resp)).responses.create(model="gpt-5", input="hi")
    assert r.id == "resp_1"
    a = mon.one().attributes
    assert a["gen_ai.operation.name"] == "responses"
    assert a["gen_ai.usage.input_tokens"] == 50 and a["gen_ai.usage.cache_read.input_tokens"] == 20
    assert a["gen_ai.usage.reasoning.output_tokens"] == 4
    assert a["gen_ai.response.finish_reasons"] == ["completed"]


def test_embeddings(mon):
    body = {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}], "model": "text-embedding-3-small", "usage": {"prompt_tokens": 8, "total_tokens": 8}}
    client(lambda req: httpx2.Response(200, json=body)).embeddings.create(model="text-embedding-3-small", input=["a", "b"])
    a = mon.one().attributes
    assert a["gen_ai.operation.name"] == "embeddings"
    assert a["gen_ai.usage.input_tokens"] == 8
    assert a["gen_ai.embeddings.input.count"] == 2


def test_openai_compatible_provider_detected(mon):
    client(lambda req: httpx2.Response(200, json=CHAT), base_url="https://api.deepseek.com/v1").chat.completions.create(model="deepseek-chat", messages=[])
    a = mon.one().attributes
    assert a["gen_ai.provider.name"] == "deepseek" and a["server.address"] == "api.deepseek.com"


def test_with_raw_response(mon):
    raw = client(lambda req: httpx2.Response(200, json=CHAT)).chat.completions.with_raw_response.create(model="gpt-5-mini", messages=[])
    assert raw.parse().usage.prompt_tokens == 120
    assert mon.one().attributes["gen_ai.usage.input_tokens"] == 120


def test_structured_output_parse(mon):
    # LangChain's with_structured_output calls chat.completions.parse (through with_raw_response)
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    body = {**CHAT, "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"answer": "Paris"}'}, "finish_reason": "stop"}]}
    c = client(lambda req: httpx2.Response(200, json=body))
    assert c.chat.completions.parse(model="gpt-5-mini", messages=[], response_format=Answer).choices[0].message.parsed.answer == "Paris"
    assert c.chat.completions.with_raw_response.parse(model="gpt-5-mini", messages=[], response_format=Answer).parse().choices[0].message.parsed.answer == "Paris"
    calls = mon.llm()
    assert len(calls) == 2
    assert all(s.attributes["gen_ai.usage.input_tokens"] == 120 and s.attributes["gen_ai.operation.name"] == "chat" for s in calls)


def test_capture_content(make_mon):
    m = make_mon(capture_content=True)
    client(lambda req: httpx2.Response(200, json=CHAT)).chat.completions.create(model="gpt-5-mini", messages=[{"role": "user", "content": "email me at a@b.co"}])
    a = m.one().attributes
    assert "[EMAIL]" in a["gen_ai.input.messages"] and "a@b.co" not in a["gen_ai.input.messages"]
    assert "Hi there" in a["gen_ai.output.messages"]


def test_shutdown_removes_patches():
    from openai.resources.chat.completions import Completions

    alye.init(service="x", exporter="none", heartbeat_interval=0)
    assert getattr(Completions.create, "__alye_wrapped__", False)
    alye.shutdown()
    assert not getattr(Completions.create, "__alye_wrapped__", False)
