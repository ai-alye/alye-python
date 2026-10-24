"""A tiny local receiver for development: accepts OTLP/HTTP JSON spans and heartbeats and prints them.

    python -m alye.devserver            # listens on http://localhost:4318
    ALYE_ENDPOINT=http://localhost:4318 python your_service.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, TextIO


def _attr_value(v: Dict[str, Any]) -> Any:
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return v["doubleValue"]
    if "boolValue" in v:
        return v["boolValue"]
    if "arrayValue" in v:
        return [_attr_value(x) for x in v["arrayValue"].get("values", [])]
    return None


def _attrs(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {i["key"]: _attr_value(i.get("value", {})) for i in items or []}


def flatten(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn an OTLP traces payload into simple dicts: one per span, with resource attributes merged in."""
    out = []
    for rs in body.get("resourceSpans", []):
        res = _attrs(rs.get("resource", {}).get("attributes", []))
        for ss in rs.get("scopeSpans", []):
            for s in ss.get("spans", []):
                a = _attrs(s.get("attributes", []))
                out.append(
                    {
                        "service": res.get("service.name"),
                        "env": res.get("deployment.environment.name"),
                        "trace_id": s.get("traceId"),
                        "span_id": s.get("spanId"),
                        "parent_span_id": s.get("parentSpanId"),
                        "name": s.get("name"),
                        "start_ns": int(s.get("startTimeUnixNano", 0)),
                        "duration_ms": (int(s.get("endTimeUnixNano", 0)) - int(s.get("startTimeUnixNano", 0))) / 1e6,
                        "status": {0: "unset", 1: "ok", 2: "error"}.get(s.get("status", {}).get("code", 0)),
                        "attributes": a,
                    }
                )
    return out


def _line(s: Dict[str, Any]) -> str:
    a = s["attributes"]
    ts = datetime.fromtimestamp(s["start_ns"] / 1e9).strftime("%H:%M:%S")
    model = a.get("gen_ai.response.model") or a.get("gen_ai.request.model") or ""
    tok = ""
    if "gen_ai.usage.input_tokens" in a:
        cached = a.get("gen_ai.usage.cache_read.input_tokens")
        tok = f"  in {a['gen_ai.usage.input_tokens']}" + (f" ({cached} cached)" if cached else "") + f"  out {a.get('gen_ai.usage.output_tokens', 0)}"
    elif a.get("alye.usage_unknown"):
        tok = "  usage unknown"
    status = f"ERROR {a.get('error.type', '')}" if s["status"] == "error" else "ok"
    extra = []
    if a.get("alye.retries"):
        extra.append(f"retries {a['alye.retries']}")
    if a.get("gen_ai.response.time_to_first_chunk") is not None:
        extra.append(f"ttft {a['gen_ai.response.time_to_first_chunk'] * 1000:.0f}ms")
    if a.get("alye.customer_id"):
        extra.append(f"customer {a['alye.customer_id']}")
    indent = "  └ " if s.get("parent_span_id") else ""
    return f"{ts}  {s['service']:<16} {indent}{s['name']:<34} {model:<22}{tok}  {s['duration_ms']:.0f}ms  {status}" + (f"  [{', '.join(extra)}]" if extra else "")


class _Handler(BaseHTTPRequestHandler):
    server: "DevServer"  # type: ignore[assignment]

    def log_message(self, *args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return
        if self.path.rstrip("/") == "/v1/traces":
            spans = flatten(body)
            self.server.record(spans)
        elif self.path.rstrip("/") == "/v1/heartbeat":
            self.server.heartbeat(body)
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")


class DevServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int = 4318, out: Optional[TextIO] = None, jsonl: Optional[str] = None, quiet: bool = False) -> None:
        super().__init__(("127.0.0.1", port), _Handler)
        self.spans: List[Dict[str, Any]] = []
        self.heartbeats: List[Dict[str, Any]] = []
        self.out = out or sys.stdout
        self.jsonl = jsonl
        self.quiet = quiet
        self._lock = threading.Lock()

    def record(self, spans: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.spans.extend(spans)
            if self.jsonl:
                with open(self.jsonl, "a", encoding="utf-8") as f:
                    for s in spans:
                        f.write(json.dumps(s) + "\n")
        if not self.quiet:
            for s in sorted(spans, key=lambda x: x["start_ns"]):
                print(_line(s), file=self.out, flush=True)

    def heartbeat(self, body: Dict[str, Any]) -> None:
        with self._lock:
            self.heartbeats.append(body)
        if not self.quiet:
            libs = ", ".join(f"{k} {v}" for k, v in (body.get("libraries") or {}).items()) or "no providers"
            print(f"♥ heartbeat  {body.get('service')} ({body.get('env')})  sdk {body.get('sdk', {}).get('version')}  {libs}  dropped {body.get('spans_dropped', 0)}", file=self.out, flush=True)

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        return t


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Local receiver for alye spans")
    ap.add_argument("--port", type=int, default=4318)
    ap.add_argument("--jsonl", help="also append every span to this file")
    args = ap.parse_args(argv)
    srv = DevServer(args.port, jsonl=args.jsonl)
    print(f"alye dev server listening on http://localhost:{args.port}  (set ALYE_ENDPOINT to this)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
