"""Records LangChain chat models whose provider client Alye doesn't patch (Bedrock, Mistral, Groq,
Ollama, Vertex AI, ...), through a LangChain callback registered for every run.

Models built on a patched client (ChatOpenAI, AzureChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI,
and the OpenAI-compatible ones such as ChatDeepSeek or ChatXAI) are already recorded by that patch,
with more detail. To never count a call twice, the callback marks the LangChain run in a context
variable; a patched client call made inside it claims the run, and the callback then drops its own.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from typing import Any, Dict, List, Optional

from ._common import LLMCall, call_tracer, langchain_run as _current

log = logging.getLogger("alye")

# LangChain's ls_provider -> the provider names Alye prices with.
PROVIDERS = {
    "openai": "openai",
    "azure": "azure.ai.openai",
    "anthropic": "anthropic",
    "google_genai": "gcp.gemini",
    "google_vertexai": "gcp.vertex_ai",
    "amazon_bedrock": "aws.bedrock",
    "mistral": "mistral_ai",
    "mistralai": "mistral_ai",
    "groq": "groq",
    "deepseek": "deepseek",
    "xai": "x_ai",
    "together": "together_ai",
    "fireworks": "fireworks_ai",
    "perplexity": "perplexity",
    "openrouter": "openrouter",
    "cohere": "cohere",
    "ollama": "ollama",
}

class _Run:
    __slots__ = ("call", "claimed", "token")

    def __init__(self, call: LLMCall) -> None:
        self.call = call
        self.claimed = False
        self.token: Optional[contextvars.Token] = None


def _int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _usage(call: LLMCall, response: Any) -> None:
    gens = getattr(response, "generations", None) or []
    gen = gens[0][0] if gens and gens[0] else None
    msg = getattr(gen, "message", None)
    um = getattr(msg, "usage_metadata", None)
    if um:
        inp, out = _int(_get(um, "input_tokens")), _int(_get(um, "output_tokens"))
        details_in = _get(um, "input_token_details") or {}
        details_out = _get(um, "output_token_details") or {}
        call.set_usage(inp, out, cache_read_tokens=_int(_get(details_in, "cache_read")),
                       cache_write_tokens=_int(_get(details_in, "cache_creation")), reasoning_tokens=_int(_get(details_out, "reasoning")))
    else:
        out = getattr(response, "llm_output", None) or {}
        tu = out.get("token_usage") or out.get("usage") or {}
        call.set_usage(_int(_get(tu, "prompt_tokens") or _get(tu, "input_tokens")), _int(_get(tu, "completion_tokens") or _get(tu, "output_tokens")))
    meta = getattr(msg, "response_metadata", None) or getattr(gen, "generation_info", None) or {}
    llm_out = getattr(response, "llm_output", None) or {}
    model = meta.get("model_name") or meta.get("model") or meta.get("model_id") or llm_out.get("model_name") or llm_out.get("model_id")
    finish = meta.get("finish_reason") or meta.get("stop_reason") or meta.get("stopReason") or meta.get("done_reason")
    call.set_response(model=model, response_id=meta.get("id"), finish_reasons=[str(finish)] if finish else None)


def make_handler() -> Any:
    from langchain_core.callbacks import BaseCallbackHandler

    class AlyeLangChainHandler(BaseCallbackHandler):
        """Adds a span per chat-model call that no patched client recorded."""

        run_inline = True  # run in the caller's context, so the run marker reaches the client call
        raise_error = False

        def __init__(self) -> None:
            self._runs: Dict[uuid.UUID, _Run] = {}

        def _start(self, run_id: uuid.UUID, metadata: Optional[Dict[str, Any]], invocation_params: Optional[Dict[str, Any]]) -> None:
            tracer = call_tracer()
            if tracer is None:
                return
            md = metadata or {}
            ls_provider = str(md.get("ls_provider") or "").lower()
            provider = PROVIDERS.get(ls_provider, ls_provider or "unknown")
            model = md.get("ls_model_name") or (invocation_params or {}).get("model") or (invocation_params or {}).get("model_id")
            attrs: Dict[str, Any] = {
                "gen_ai.request.temperature": md.get("ls_temperature"),
                "gen_ai.request.max_tokens": md.get("ls_max_tokens"),
                "alye.integration": "langchain",
            }
            region = (invocation_params or {}).get("region_name") or (invocation_params or {}).get("region")
            if region:
                attrs["cloud.region"] = str(region)
            run = _Run(LLMCall(tracer, provider=provider, operation="chat", model=str(model) if model else None, attributes=attrs))
            run.token = _current.set(run)
            self._runs[run_id] = run

        def _finish(self, run_id: uuid.UUID) -> Optional[_Run]:
            run = self._runs.pop(run_id, None)
            if run is not None and run.token is not None:
                try:
                    _current.reset(run.token)
                except ValueError:  # ended in another context (e.g. a stream consumed elsewhere)
                    if _current.get() is run:
                        _current.set(None)
            return run

        def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: uuid.UUID, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
            self._start(run_id, metadata, kwargs.get("invocation_params"))

        def on_llm_start(self, serialized: Any, prompts: List[str], *, run_id: uuid.UUID, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
            self._start(run_id, metadata, kwargs.get("invocation_params"))

        def on_llm_new_token(self, token: str, *, run_id: uuid.UUID, **kwargs: Any) -> None:
            run = self._runs.get(run_id)
            if run is not None:
                run.call.first_token()

        def on_llm_end(self, response: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
            run = self._finish(run_id)
            if run is None or run.claimed:
                return  # the patched client already recorded this call
            try:
                _usage(run.call, response)
            except Exception:
                log.debug("alye: could not read a LangChain response", exc_info=True)
            run.call.end()

        def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> None:
            run = self._finish(run_id)
            if run is None or run.claimed:
                return
            run.call.fail(error)
            run.call.end()

    return AlyeLangChainHandler()


_registered = False


def patch(_patcher: Any) -> bool:
    """Register the handler for every LangChain run (once per process; it does nothing after shutdown)."""
    global _registered
    try:
        from langchain_core.tracers.context import register_configure_hook
    except ImportError:
        return False
    if not _registered:
        register_configure_hook(contextvars.ContextVar("alye_langchain_handler", default=make_handler()), inheritable=True)
        _registered = True
    return True
