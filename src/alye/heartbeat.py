"""Periodic heartbeat so the backend can tell a quiet service from one that has stopped reporting."""

from __future__ import annotations

import logging
import os
import platform
import socket
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

from ._version import __version__

log = logging.getLogger("alye")


def _lib_versions(names: list) -> Dict[str, str]:
    out: Dict[str, str] = {}
    dist = {"openai": "openai", "anthropic": "anthropic", "google_genai": "google-genai", "langchain": "langchain-core"}
    try:
        from importlib.metadata import version
    except ImportError:  # pragma: no cover
        return out
    for n in names:
        try:
            out[n] = version(dist.get(n, n))
        except Exception:
            out[n] = "unknown"
    return out


class Heartbeat:
    def __init__(self, send: Callable[[Dict[str, Any]], bool], payload: Callable[[], Dict[str, Any]], interval: float) -> None:
        self._send = send
        self._payload = payload
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if interval > 0:
            self._start()
            if hasattr(os, "register_at_fork"):
                os.register_at_fork(after_in_child=self._restart_in_child)

    def _restart_in_child(self) -> None:
        if not self._stop.is_set():
            self._start()

    def _start(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="alye-heartbeat", daemon=True)
        self._thread.start()

    def beat(self) -> None:
        try:
            self._send(self._payload())
        except Exception:
            log.debug("alye: heartbeat failed", exc_info=True)

    def _run(self) -> None:
        self.beat()
        while not self._stop.wait(self.interval):
            self.beat()

    def stop(self) -> None:
        self._stop.set()


def base_payload(service: str, env: str, release: Optional[str], patched: list) -> Dict[str, Any]:
    return {
        "service": service,
        "env": env,
        "release": release,
        "sdk": {"name": "alye-python", "version": __version__},
        "runtime": {"python": platform.python_version(), "implementation": platform.python_implementation(), "platform": sys.platform},
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "libraries": _lib_versions(patched),
        "started_at": time.time(),
    }
