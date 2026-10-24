"""LangChain models whose client Alye doesn't patch are recorded through a LangChain callback, and
calls a patched client already recorded are not recorded twice."""

import asyncio
from typing import Any, List, Optional

import pytest

import alye

lc = pytest.importorskip("langchain_core")
from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


class FakeChat(BaseChatModel):
    """Answers with usage_metadata, like ChatBedrockConverse or ChatMistralAI."""

    provider: str = "mistral"
    model_name: str = "mistral-large-latest"
    through_patched_client: bool = False

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _get_ls_params(self, stop: Optional[List[str]] = None, **kwargs: Any) -> dict:
        return {"ls_provider": self.provider, "ls_model_name": self.model_name, "ls_model_type": "chat"}

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        if self.through_patched_client:
            # What a patched client (openai, anthropic, google-genai) does inside a LangChain model.
            with alye.llm_call("openai", self.model_name) as call:
                call.set_usage(input_tokens=50, output_tokens=9)
        msg = AIMessage(
            content="Sunny",
            usage_metadata={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150, "input_token_details": {"cache_read": 100}},
            response_metadata={"model_name": self.model_name + "-2026", "finish_reason": "stop"},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_unpatched_provider_is_recorded(mon):
    FakeChat().invoke("hi")
    s = mon.one()
    a = s.attributes
    assert a["gen_ai.provider.name"] == "mistral_ai" and a["alye.integration"] == "langchain"
    assert a["gen_ai.request.model"] == "mistral-large-latest" and a["gen_ai.response.model"] == "mistral-large-latest-2026"
    assert (a["gen_ai.usage.input_tokens"], a["gen_ai.usage.output_tokens"], a["gen_ai.usage.cache_read.input_tokens"]) == (120, 30, 100)
    assert a["gen_ai.response.finish_reasons"] == ["stop"]


def test_bedrock_name_and_async_batch(mon):
    m = FakeChat(provider="amazon_bedrock", model_name="anthropic.claude-3-5-sonnet-20240620-v1:0")
    asyncio.run(m.ainvoke("hi"))
    m.batch(["a", "b"])
    calls = mon.llm()
    assert len(calls) == 3 and all(c.attributes["gen_ai.provider.name"] == "aws.bedrock" for c in calls)


def test_patched_client_inside_langchain_is_not_counted_twice(mon):
    FakeChat(through_patched_client=True, provider="deepseek").invoke("hi")
    s = mon.one()
    assert s.attributes["gen_ai.provider.name"] == "openai"  # the client patch's call wins
    assert "alye.integration" not in s.attributes


def test_failures_are_recorded(mon):
    class Broken(FakeChat):
        def _generate(self, *a: Any, **k: Any) -> ChatResult:
            raise RuntimeError("upstream 500")

    with pytest.raises(RuntimeError):
        Broken().invoke("hi")
    s = mon.one()
    assert s.attributes["error.type"] and s.attributes["alye.integration"] == "langchain"


def test_nests_under_alye_span(mon):
    with alye.span("answer_ticket"):
        FakeChat().invoke("hi")
    spans = mon.spans()
    root = next(s for s in spans if s.name == "answer_ticket")
    call = mon.one()
    assert call.trace_id == root.trace_id and call.parent_span_id == root.span_id
