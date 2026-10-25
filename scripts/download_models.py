"""One-time model download (the only step that needs internet besides `pip install`).

    python scripts/download_models.py                       # whisper small + qwen3:4b
    python scripts/download_models.py --whisper medium --llm qwen3:8b
    python scripts/download_models.py --skip-llm

After this, everything - including model loading - runs offline.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.hardware_detection import detect_ollama  # noqa: E402

WHISPER_CHOICES = ("tiny", "base", "small", "medium", "large-v2")


def download_whisper(size: str) -> bool:
    try:
        from faster_whisper.utils import download_model
    except ImportError:
        print("faster-whisper is not installed: pip install -r requirements.txt")
        return False
    print(f"Downloading Whisper '{size}' (Systran/faster-whisper-{size}, MIT licence)...")
    try:
        path = download_model(size)
    except Exception as exc:
        print(f"FAILED: {exc}\nCheck your internet connection / proxy. huggingface.co must be reachable for this one-off step.")
        return False
    print(f"OK -> {path}")
    return True


def pull_ollama(model: str, host: str) -> bool:
    exe = shutil.which("ollama")
    if not exe:
        print("Ollama not found. Install it from https://ollama.com/download, start it, then re-run this script.")
        return False
    info = detect_ollama(host)
    if not info["server"]:
        print("Ollama is installed but its server is not running. Start the Ollama app (or run `ollama serve`) and retry.")
        return False
    if any(m == model or m.startswith(model + ":") or m.split(":")[0] == model for m in info["models"]):
        print(f"Ollama model '{model}' is already installed.")
        return True
    print(f"Pulling '{model}' with Ollama (Qwen3 weights are Apache-2.0)...")
    return subprocess.run([exe, "pull", model]).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--whisper", default="small", choices=WHISPER_CHOICES)
    ap.add_argument("--llm", default="qwen3:4b")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--skip-whisper", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    a = ap.parse_args()
    ok = True
    if not a.skip_whisper:
        ok &= download_whisper(a.whisper)
    if not a.skip_llm:
        ok &= pull_ollama(a.llm, a.ollama_host)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
