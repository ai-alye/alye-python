import json
from typing import Any, Callable, Dict, List

import pytest

import alye
from alye import InMemoryExporter


class Mon:
    def __init__(self, exporter: InMemoryExporter) -> None:
        self.exporter = exporter

    def spans(self) -> List[Any]:
        alye.flush(5)
        return list(self.exporter.spans)

    def llm(self) -> List[Any]:
        return [s for s in self.spans() if "gen_ai.operation.name" in s.attributes]

    def one(self) -> Any:
        calls = self.llm()
        assert len(calls) == 1, f"expected one LLM span, got {calls}"
        return calls[0]


@pytest.fixture
def make_mon() -> Callable[..., Mon]:
    def _make(**opts: Any) -> Mon:
        exp = InMemoryExporter()
        alye.init(service="test-svc", env="test", exporter=exp, flush_interval=0.05, heartbeat_interval=0, **opts)
        return Mon(exp)

    yield _make
    alye.shutdown(2)


@pytest.fixture
def mon(make_mon: Callable[..., Mon]) -> Mon:
    return make_mon()


def sse(events: List[Dict[str, Any]], named: bool = False, done: bool = False) -> bytes:
    out = []
    for e in events:
        if named:
            out.append(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n")
        else:
            out.append(f"data: {json.dumps(e)}\n\n")
    if done:
        out.append("data: [DONE]\n\n")
    return "".join(out).encode()
