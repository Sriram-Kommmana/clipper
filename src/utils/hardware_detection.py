"""Hardware / environment detection and model recommendations.

Everything here describes *the machine the code runs on*. When the cloud dev sandbox runs it, the report
describes the sandbox, not the user's laptop - run `python scripts/check_environment.py` on the laptop.
"""
from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any


def register_nvidia_dll_dirs() -> list[str]:
    """Windows: make pip-installed CUDA libs (nvidia-cublas-cu12, nvidia-cudnn-cu12) discoverable.

    faster-whisper/CTranslate2 needs cuBLAS 12 DLLs; `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` places
    them under site-packages/nvidia/*/bin, which is not on the DLL search path. Ollama's copy is also used if present.
    """
    added: list[str] = []
    if os.name != "nt":
        return added
    import site

    roots = set(site.getsitepackages() + [site.getusersitepackages()])
    candidates: list[str] = []
    for root in roots:
        nv = os.path.join(root, "nvidia")
        if os.path.isdir(nv):
            candidates += [os.path.join(nv, sub, "bin") for sub in os.listdir(nv)]
    candidates += _ollama_cuda_dirs()
    for b in candidates:
        if os.path.isdir(b):
            try:
                os.add_dll_directory(b)
                os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")
                added.append(b)
            except OSError:
                pass
    return added


def _ollama_cuda_dirs() -> list[str]:
    """Ollama for Windows ships the CUDA 12 runtime; using it avoids a separate download of more than 1 GB."""
    exe = shutil.which("ollama")
    roots = [os.path.dirname(exe)] if exe else []
    if os.environ.get("LOCALAPPDATA"):
        roots.append(os.path.join(os.environ["LOCALAPPDATA"], "Programs", "Ollama"))
    return [os.path.join(r, "lib", "ollama", "cuda_v12") for r in dict.fromkeys(roots)]


def _run(cmd: list[str], timeout: float = 10) -> str | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return (p.stdout or "") + (p.stderr or "") if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def detect_gpu() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False, "reason": "nvidia-smi not found (no NVIDIA driver, or not on PATH)"}
    out = _run([exe, "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"])
    if not out:
        return {"available": False, "reason": "nvidia-smi failed to run"}
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                gpus.append({"name": parts[0], "vram_total_mb": int(parts[1]), "vram_free_mb": int(parts[2]), "driver": parts[3]})
            except ValueError:
                continue
    return {"available": bool(gpus), "gpus": gpus} if gpus else {"available": False, "reason": "no GPU rows parsed"}


def detect_ctranslate2() -> dict[str, Any]:
    info: dict[str, Any] = {"installed": False}
    try:
        register_nvidia_dll_dirs()
        import ctranslate2

        info.update(installed=True, version=ctranslate2.__version__)
        try:
            n = ctranslate2.get_cuda_device_count()
            info["cuda_devices"] = n
            if n:
                info["cuda_compute_types"] = sorted(ctranslate2.get_supported_compute_types("cuda"))
        except Exception as exc:  # missing cublas/cudnn surfaces here on Windows
            info["cuda_devices"] = 0
            info["cuda_error"] = str(exc)[:200]
        info["cpu_compute_types"] = sorted(ctranslate2.get_supported_compute_types("cpu"))
    except ImportError:
        info["reason"] = "ctranslate2 not installed (pip install -r requirements.txt)"
    return info


def detect_ffmpeg() -> dict[str, Any]:
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    info: dict[str, Any] = {"ffmpeg": ff, "ffprobe": fp}
    if ff:
        out = _run([ff, "-hide_banner", "-version"]) or ""
        info["version"] = out.splitlines()[0] if out else "unknown"
        enc = _run([ff, "-hide_banner", "-encoders"]) or ""
        info["libx264"] = " libx264 " in enc
        info["h264_nvenc"] = " h264_nvenc " in enc
        info["aac"] = " aac " in enc
        flt = _run([ff, "-hide_banner", "-filters"]) or ""
        info["ass_filter"] = " ass " in flt
    return info


def detect_ollama(host: str = "http://127.0.0.1:11434") -> dict[str, Any]:
    info: dict[str, Any] = {"binary": shutil.which("ollama"), "server": False, "models": []}
    try:
        import requests

        r = requests.get(host.rstrip("/") + "/api/tags", timeout=2)
        if r.ok:
            info["server"] = True
            info["models"] = sorted(m.get("name", "") for m in r.json().get("models", []))
    except Exception as exc:
        info["server_error"] = type(exc).__name__
    return info


def detect_system() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    return {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()[:60]})",
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores_logical": os.cpu_count(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / 2**30, 1),
        "ram_available_gb": round(vm.available / 2**30, 1),
    }


def build_report(ollama_host: str = "http://127.0.0.1:11434") -> dict[str, Any]:
    report = {
        "system": detect_system(),
        "gpu": detect_gpu(),
        "ctranslate2": detect_ctranslate2(),
        "ffmpeg": detect_ffmpeg(),
        "ollama": detect_ollama(ollama_host),
    }
    report["recommendation"] = recommend(report)
    return report


def recommend(report: dict[str, Any]) -> dict[str, Any]:
    """Pick conservative defaults from a report dict (pure function: unit-tested)."""
    notes: list[str] = []
    ram = report["system"].get("ram_total_gb", 0)
    gpu = report.get("gpu", {})
    ct2 = report.get("ctranslate2", {})
    cuda_ok = bool(gpu.get("available")) and ct2.get("cuda_devices", 0) > 0
    vram_gb = (gpu["gpus"][0]["vram_total_mb"] / 1024) if gpu.get("available") else 0.0

    rec: dict[str, Any] = {"notes": notes}
    if cuda_ok and vram_gb >= 5.5:
        rec.update(whisper_model="large-v3-turbo", whisper_device="cuda", whisper_compute="float16", llm_model="qwen3:4b")
        if vram_gb >= 7.5:
            notes.append("8 GB VRAM: qwen3:8b also fits fully on the GPU; models run one at a time.")
        else:
            notes.append("6 GB VRAM: Whisper 'large-v3-turbo' and qwen3:4b fit. Models are never loaded together. qwen3:8b works but spills onto the CPU.")
    elif cuda_ok and vram_gb >= 3.5:
        rec.update(whisper_model="base", whisper_device="cuda", whisper_compute="int8_float16", llm_model="qwen3:1.7b")
        notes.append("Limited VRAM: using smaller models.")
    else:
        rec.update(whisper_model="base", whisper_device="cpu", whisper_compute="int8", llm_model="qwen3:1.7b")
        if gpu.get("available") and not ct2.get("cuda_devices"):
            notes.append(
                "An NVIDIA GPU exists but CTranslate2 cannot use CUDA "
                f"({ct2.get('cuda_error', 'no CUDA device found')}). Install CUDA 12 + cuDNN 9 (or "
                "'pip install nvidia-cublas-cu12 nvidia-cudnn-cu12') and update the NVIDIA driver."
            )
        else:
            notes.append("No usable GPU: CPU transcription and LLM will be slow; expect several x real-time.")
    if ram and ram < 12:
        rec["performance_mode"] = "low"
        notes.append("Under 12 GB RAM: enable performance.mode: low.")
    else:
        rec["performance_mode"] = "normal"
    ff = report.get("ffmpeg", {})
    if not ff.get("ffmpeg") or not ff.get("ffprobe"):
        notes.append("FFmpeg/FFprobe not found on PATH - install FFmpeg (e.g. 'winget install Gyan.FFmpeg').")
    elif not ff.get("libx264") or not ff.get("ass_filter"):
        notes.append("This FFmpeg build lacks libx264 or the 'ass' filter - install a full build (gyan.dev 'full' / BtbN GPL).")
    ol = report.get("ollama", {})
    if not ol.get("server"):
        notes.append("Ollama server not reachable: install from ollama.com, start it, then 'ollama pull " + rec["llm_model"] + "'.")
    elif rec["llm_model"] not in ol.get("models", []) and not any(m.startswith(rec["llm_model"]) for m in ol.get("models", [])):
        notes.append(f"Model not installed: run 'ollama pull {rec['llm_model']}'.")
    return rec


def is_gpu_memory_error(exc: BaseException) -> bool:
    s = f"{type(exc).__name__}: {exc}".lower()
    return any(t in s for t in ("out of memory", "cuda failed", "cudamalloc", "cublas", "cudnn", "cuda error", "alloc"))


def format_report(report: dict[str, Any]) -> str:
    s, g, c, f, o, r = (report[k] for k in ("system", "gpu", "ctranslate2", "ffmpeg", "ollama", "recommendation"))
    L = [f"Machine: {s['hostname']}  (this report describes THIS machine only)"]
    L.append(f"OS: {s['os']}\nPython: {s['python']}\nCPU: {s['cpu']} ({s['cpu_cores_physical']} physical / {s['cpu_cores_logical']} logical cores)")
    L.append(f"RAM: {s['ram_total_gb']} GB total, {s['ram_available_gb']} GB available")
    if g.get("available"):
        for i, d in enumerate(g["gpus"]):
            L.append(f"GPU {i}: {d['name']} - {d['vram_total_mb']} MB VRAM ({d['vram_free_mb']} MB free), driver {d['driver']}")
    else:
        L.append(f"GPU: none usable ({g.get('reason')})")
    if c.get("installed"):
        L.append(f"CTranslate2 {c['version']}: CUDA devices={c.get('cuda_devices', 0)}" + (f" [{c['cuda_error']}]" if c.get("cuda_error") else "")
                 + (f", GPU compute types={c.get('cuda_compute_types')}" if c.get("cuda_compute_types") else ""))
    else:
        L.append(f"CTranslate2: {c.get('reason')}")
    L.append(f"FFmpeg: {f.get('version', 'NOT FOUND')} | libx264={f.get('libx264')} aac={f.get('aac')} ass={f.get('ass_filter')} nvenc={f.get('h264_nvenc')}")
    L.append(f"Ollama: binary={o['binary'] or 'not found'}, server={'running' if o['server'] else 'not reachable'}, models={o['models'] or '-'}")
    L.append("\nRecommended: Whisper '{whisper_model}' on {whisper_device} ({whisper_compute}); LLM {llm_model}; performance.mode={performance_mode}".format(**r))
    L.extend("  * " + n for n in r["notes"])
    return "\n".join(L)
