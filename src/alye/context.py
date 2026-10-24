"""Request-scoped context: the current span and tags such as customer_id, carried through sync and async code."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

_current_span: contextvars.ContextVar[Any] = contextvars.ContextVar("alye_current_span", default=None)
_tags: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("alye_tags", default={})

# Known tags map to these span attribute names. Anything else goes under alye.tag.<name>.
TAG_ATTRIBUTES = {
    "customer_id": "alye.customer_id",
    "session_id": "session.id",
    "user_id": "enduser.id",
    "agent": "gen_ai.agent.name",
    "data_source": "alye.data_source",
    "operation": "alye.operation",
}


def get_current_span() -> Any:
    return _current_span.get()


def set_current_span(span: Any) -> contextvars.Token:
    return _current_span.set(span)


def reset_current_span(token: contextvars.Token) -> None:
    _current_span.reset(token)


def current_tags() -> Dict[str, Any]:
    return dict(_tags.get())


def tag_attributes() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in _tags.get().items():
        if v is None:
            continue
        out[TAG_ATTRIBUTES.get(k, f"alye.tag.{k}")] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return out


@contextmanager
def context(**tags: Any) -> Iterator[Dict[str, Any]]:
    """Tag every LLM call made inside this block.

        with alye.context(customer_id="northwind", session_id=ticket.id):
            answer = run_agent(ticket)

    Nested blocks add to (and can override) the outer tags. Works across `await`.
    For new threads, run the target with `contextvars.copy_context().run(...)` to keep the tags.
    """
    merged = {**_tags.get(), **tags}
    token = _tags.set(merged)
    try:
        yield dict(merged)
    finally:
        _tags.reset(token)


def set_tags(**tags: Any) -> None:
    """Set tags for the rest of the current context without a `with` block (for example in middleware)."""
    _tags.set({**_tags.get(), **tags})


def clear_tags(*names: Optional[str]) -> None:
    if not names:
        _tags.set({})
        return
    cur = dict(_tags.get())
    for n in names:
        cur.pop(n, None)
    _tags.set(cur)
