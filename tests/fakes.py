"""Test doubles: a scripted LLM client and a fake Ollama HTTP server."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


class ScriptedLLM:
    """Returns queued responses (str, or callable(system, prompt) -> str). Records every call."""

    def __init__(self, responses: list[Any] | Callable[[str, str], str], model: str = "fake:1b"):
        self.model = model
        self._resp = responses
        self.calls: list[tuple[str, str]] = []
        self.unloaded = False

    def generate_json(self, system, prompt, schema, *, num_predict=None, cancel=None) -> str:
        self.calls.append((system, prompt))
        if callable(self._resp):
            return self._resp(system, prompt)
        r = self._resp.pop(0) if len(self._resp) > 1 else self._resp[0]
        return r(system, prompt) if callable(r) else r

    def unload(self):
        self.unloaded = True


class FakeOllama:
    """Tiny /api/tags + /api/chat server. `chat_handler(payload) -> (status, [ndjson dicts] | error dict)`."""

    def __init__(self, models=("qwen3:4b",), chat_handler=None):
        self.models = list(models)
        self.chat_handler = chat_handler or (lambda p: (200, [{"message": {"content": '{"moments": []}'}, "done": True}]))
        self.requests: list[dict] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):
                body = json.dumps({"models": [{"name": m} for m in outer.models]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append(payload)
                status, body = outer.chat_handler(payload)
                self.send_response(status)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                if isinstance(body, dict):
                    self.wfile.write(json.dumps(body).encode())
                else:
                    for obj in body:
                        self.wfile.write((json.dumps(obj) + "\n").encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()
