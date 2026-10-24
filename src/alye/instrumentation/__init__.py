"""Turns provider patches on and off."""

from __future__ import annotations

import importlib
import logging
from typing import List, Optional, Sequence

from ._common import Patcher

log = logging.getLogger("alye")

SUPPORTED = {
    "openai": "alye.instrumentation.openai",
    "anthropic": "alye.instrumentation.anthropic",
    "google_genai": "alye.instrumentation.google_genai",
    # Last: LangChain models whose client none of the above patches (Bedrock, Mistral, Groq, Ollama, ...).
    "langchain": "alye.instrumentation.langchain",
}

_patcher = Patcher()
_patched: List[str] = []


def instrument(names: Optional[Sequence[str]] = None) -> List[str]:
    """Patch the named providers (default: every supported one that is installed). Returns what was patched."""
    wanted = list(names) if names is not None else list(SUPPORTED)
    for name in wanted:
        if name in _patched:
            continue
        module = SUPPORTED.get(name)
        if module is None:
            log.warning("alye: unknown provider %r; supported: %s", name, ", ".join(SUPPORTED))
            continue
        try:
            if importlib.import_module(module).patch(_patcher):
                _patched.append(name)
        except Exception:
            # A provider SDK changed shape: keep the app running and say so once.
            log.warning("alye: could not instrument %s; its calls will not be recorded", name, exc_info=True)
    return list(_patched)


def uninstrument() -> None:
    _patcher.undo()
    _patched.clear()


def patched() -> List[str]:
    return list(_patched)
