"""Minimal Ollama HTTP client (local only). Streams so jobs can be cancelled mid-generation."""
from __future__ import annotations

import ipaddress
import json
import logging
import threading
from typing import Any, Protocol
from urllib.parse import urlparse

import requests

from config import LLMConfig
from src.core.exceptions import ConfigError, JobCancelled, LLMModelMissingError, LLMUnavailableError, ResourceExhaustedError

log = logging.getLogger(__name__)


class LLMClient(Protocol):
    """What the pipeline needs from a language model. Tests provide a scripted fake."""

    model: str

    def generate_json(
        self, system: str, prompt: str, schema: dict[str, Any], *,
        num_predict: int | None = None, cancel: threading.Event | None = None,
    ) -> str: ...

    def unload(self) -> None: ...


def assert_local_host(host: str, allow_remote: bool) -> None:
    """Privacy guard: only loopback Ollama servers are accepted unless explicitly overridden."""
    name = urlparse(host if "://" in host else "http://" + host).hostname or ""
    if allow_remote:
        return
    if name == "localhost":
        return
    try:
        if ipaddress.ip_address(name).is_loopback:
            return
    except ValueError:
        pass
    raise ConfigError(
        f"llm.host '{host}' is not a local address; refusing to send transcripts off this machine.",
        recovery="Use http://127.0.0.1:11434, or set llm.allow_remote_host: true if you really run Ollama on another machine you control.",
    )


class OllamaClient:
    def __init__(self, cfg: LLMConfig):
        assert_local_host(cfg.host, cfg.allow_remote_host)
        self.cfg = cfg
        self.model = cfg.model
        self.base = cfg.host.rstrip("/")
        self._send_think = True

    # ---- availability --------------------------------------------------------------------------
    def list_models(self) -> list[str]:
        try:
            r = requests.get(self.base + "/api/tags", timeout=5)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise LLMUnavailableError(f"Cannot reach Ollama at {self.base} ({type(exc).__name__}).") from exc
        return [m.get("name", "") for m in r.json().get("models", [])]

    def ensure_ready(self) -> None:
        """Raise a helpful error unless the server is up and the model is installed."""
        names = self.list_models()
        want = self.model if ":" in self.model else self.model + ":latest"
        if want not in names and self.model not in names:
            raise LLMModelMissingError(
                f"Ollama is running but model '{self.model}' is not installed (installed: {', '.join(names) or 'none'}).",
                recovery=f"Run: ollama pull {self.model}",
            )

    # ---- generation ----------------------------------------------------------------------------
    def generate_json(self, system, prompt, schema, *, num_predict=None, cancel=None) -> str:
        sys_text = system + ("\n/no_think" if self.model.lower().startswith("qwen3") else "")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": prompt}],
            "stream": True,
            "format": schema,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "temperature": self.cfg.temperature, "seed": self.cfg.seed,
                "num_ctx": self.cfg.num_ctx, "num_predict": num_predict or self.cfg.num_predict,
            },
        }
        if self._send_think:
            payload["think"] = False
        try:
            return self._stream(payload, cancel)
        except _ThinkUnsupported:
            self._send_think = False
            payload.pop("think", None)
            return self._stream(payload, cancel)

    def _stream(self, payload: dict[str, Any], cancel: threading.Event | None) -> str:
        try:
            resp = requests.post(self.base + "/api/chat", json=payload, stream=True, timeout=(5, self.cfg.timeout_s))
        except requests.ConnectionError as exc:
            raise LLMUnavailableError(f"Cannot connect to Ollama at {self.base}. Is it running?") from exc
        except requests.Timeout as exc:
            raise LLMUnavailableError("Ollama did not respond in time.") from exc
        try:
            if resp.status_code != 200:
                self._raise_for_status(resp)
            parts: list[str] = []
            self.last_done_reason: str | None = None
            for raw in resp.iter_lines():
                if cancel is not None and cancel.is_set():
                    raise JobCancelled("Cancelled during LLM generation")
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except ValueError:
                    continue
                if obj.get("error"):
                    raise _classify(str(obj["error"]), self.model)
                parts.append((obj.get("message") or {}).get("content", ""))
                if obj.get("done"):
                    self.last_done_reason = obj.get("done_reason")
            return "".join(parts)
        except requests.Timeout as exc:
            raise LLMUnavailableError(
                f"Ollama stalled for more than {self.cfg.timeout_s:.0f}s.",
                recovery="The model may be too large for this machine (swapping). Use a smaller llm.model or performance.mode: low.",
            ) from exc
        except requests.RequestException as exc:
            raise LLMUnavailableError(f"Connection to Ollama failed mid-response: {exc}") from exc
        finally:
            resp.close()

    def _raise_for_status(self, resp: requests.Response) -> None:
        text = ""
        try:
            text = resp.json().get("error", "") or resp.text
        except ValueError:
            text = resp.text
        if resp.status_code == 400 and "think" in text.lower():
            raise _ThinkUnsupported()
        raise _classify(text or f"HTTP {resp.status_code}", self.model, resp.status_code)

    def unload(self) -> None:
        """Ask Ollama to drop the model from memory now (frees VRAM for FFmpeg/NVENC and other apps)."""
        try:
            requests.post(self.base + "/api/chat", json={"model": self.model, "messages": [], "keep_alive": 0}, timeout=5)
        except requests.RequestException:
            pass


class _ThinkUnsupported(Exception):
    pass


def _classify(text: str, model: str, status: int = 0) -> Exception:
    low = text.lower()
    if status == 404 or ("model" in low and "not found" in low):
        return LLMModelMissingError(f"Ollama does not have model '{model}'.", recovery=f"Run: ollama pull {model}")
    if "memory" in low or "oom" in low or "cuda" in low:
        return ResourceExhaustedError(f"Ollama ran out of memory loading '{model}': {text[:200]}",
                                      recovery="Use a smaller model (llm.model: qwen3:1.7b), reduce llm.num_ctx, or close GPU-heavy programs.")
    return LLMUnavailableError(f"Ollama error: {text[:300]}")
