"""Runs a small fake "support agent" without real API keys and sends its calls to the local dev server.

    python -m alye.devserver            # terminal 1
    python examples/demo_offline.py               # terminal 2

The OpenAI and Anthropic clients are real; only their HTTP transport is faked.
"""

import os

import anthropic
import httpx2
import openai

import alye

alye.init(
    service="support-reply",
    env="local",
    endpoint=os.environ.get("ALYE_ENDPOINT", "http://localhost:4318"),
    flush_interval=0.5,
)

attempts = {"n": 0}


def fake_openai(req: httpx2.Request) -> httpx2.Response:
    attempts["n"] += 1
    if attempts["n"] == 1:  # first call is rate limited, the SDK retries
        return httpx2.Response(429, json={"error": {"message": "slow down"}}, headers={"retry-after-ms": "50"})
    return httpx2.Response(
        200,
        json={
            "id": "chatcmpl-demo",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-5-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "billing"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 310, "completion_tokens": 3, "total_tokens": 313, "prompt_tokens_details": {"cached_tokens": 256}},
        },
    )


def fake_anthropic(req: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "msg_demo",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": "I've refunded the duplicate charge."}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 180, "cache_read_input_tokens": 9800, "cache_creation_input_tokens": 0},
        },
    )


oai = openai.OpenAI(api_key="sk-demo", base_url="http://fake/v1", http_client=httpx2.Client(transport=httpx2.MockTransport(fake_openai)))
claude = anthropic.Anthropic(api_key="sk-ant-demo", base_url="http://fake", http_client=httpx2.Client(transport=httpx2.MockTransport(fake_anthropic)))


@alye.trace("answer_ticket")
def answer_ticket(ticket: str) -> str:
    oai.chat.completions.create(model="gpt-5-mini", messages=[{"role": "user", "content": f"Classify: {ticket}"}])
    with alye.span("lookup_order", tool="orders.get"):
        pass
    reply = claude.messages.create(model="claude-sonnet-5-5", max_tokens=800, messages=[{"role": "user", "content": ticket}])
    return reply.content[0].text


with alye.context(customer_id="northwind", session_id="ses_demo"):
    print("agent replied:", answer_ticket("I was charged twice for order A-55120"))

alye.flush()
