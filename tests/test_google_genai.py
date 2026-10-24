import json

import httpx
import pytest
from google import genai
from google.genai import types

import alye

RESP = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "Bonjour"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 5, "thoughtsTokenCount": 3, "cachedContentTokenCount": 30, "totalTokenCount": 48},
    "modelVersion": "gemini-2.5-flash",
    "responseId": "r-1",
}


def handler(req: httpx.Request) -> httpx.Response:
    path = req.url.path
    if path.endswith(":streamGenerateContent"):
        chunks = [
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "Bon"}]}}], "modelVersion": "gemini-2.5-flash"},
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "jour"}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2, "totalTokenCount": 12}, "modelVersion": "gemini-2.5-flash"},
        ]
        body = "".join(f"data: {json.dumps(c)}\r\n\r\n" for c in chunks)
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})
    if path.endswith(":generateContent"):
        return httpx.Response(200, json=RESP)
    if path.endswith(":batchEmbedContents") or path.endswith(":embedContent"):
        return httpx.Response(200, json={"embeddings": [{"values": [0.1, 0.2]}]})
    return httpx.Response(404, json={"error": {"code": 404, "message": path}})


def client(h=handler):
    opts = types.HttpOptions(base_url="http://test", httpx_client=httpx.Client(transport=httpx.MockTransport(h)), httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(h)))
    return genai.Client(api_key="test-key", http_options=opts)


def test_generate_content(mon):
    r = client().models.generate_content(model="gemini-2.5-flash", contents="hello", config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=200))
    assert r.text == "Bonjour"
    s = mon.one()
    a = s.attributes
    assert s.name == "generate_content gemini-2.5-flash"
    assert a["gen_ai.provider.name"] == "gcp.gemini"
    assert a["gen_ai.usage.input_tokens"] == 40
    assert a["gen_ai.usage.output_tokens"] == 8  # 5 + 3 thinking
    assert a["gen_ai.usage.reasoning.output_tokens"] == 3
    assert a["gen_ai.usage.cache_read.input_tokens"] == 30
    assert a["gen_ai.response.finish_reasons"] == ["STOP"]
    assert a["gen_ai.request.temperature"] == 0.3 and a["gen_ai.request.max_tokens"] == 200


def test_generate_content_stream(mon):
    text = "".join(ch.text or "" for ch in client().models.generate_content_stream(model="gemini-2.5-flash", contents="hello"))
    assert text == "Bonjour"
    a = mon.one().attributes
    assert a["gen_ai.usage.input_tokens"] == 10 and a["gen_ai.usage.output_tokens"] == 2
    assert a["alye.streaming"] is True


async def test_async_generate_and_stream(mon):
    c = client()
    await c.aio.models.generate_content(model="gemini-2.5-flash", contents="hello")
    stream = await c.aio.models.generate_content_stream(model="gemini-2.5-flash", contents="hello")
    async for _ in stream:
        pass
    assert sorted(s.attributes["gen_ai.usage.input_tokens"] for s in mon.llm()) == [10, 40]


def test_error(mon):
    c = client(lambda req: httpx.Response(429, json={"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}))
    with pytest.raises(Exception):
        c.models.generate_content(model="gemini-2.5-flash", contents="hello", config=types.GenerateContentConfig(http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))))
    s = mon.one()
    assert s.status == "error" and s.attributes["error.type"] == "rate_limited"
