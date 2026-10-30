import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

import app as app_module
from config import load_config
from src.core.exceptions import ConfigError, ResourceExhaustedError
from src.core.job_manager import JobManager, JobReport
from src.review.interface import RES_PRESETS, collect_overrides
from src.review.workbench import Workbench
from src.transcription.transcriber import FasterWhisperBackend
from src.transcription.transcript_models import Transcript
from src.utils.hardware_detection import is_gpu_memory_error, recommend
from src.utils.logging_setup import detach_handler
from tests.conftest import needs_ffmpeg

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- resource handling
def test_gpu_oom_falls_back_to_cpu_and_says_so(monkeypatch):
    cfg = load_config(overrides={"transcription": {"device": "cuda"}}, local_path=None)
    be = FasterWhisperBackend(cfg, 2)
    be.device, be.compute_type, be._model = "cuda", "float16", object()
    calls = []

    def run(wav, language, duration, on_progress, cancel):
        calls.append(be.device)
        if be.device == "cuda":
            raise RuntimeError("CUDA failed with error out of memory")
        return Transcript(language="en")

    def load():
        if be._model is None:      # like the real _load: only loads when nothing is loaded
            be.device, be.compute_type, be._model = "cpu", "int8", object()

    monkeypatch.setattr(be, "_run", run)
    monkeypatch.setattr(be, "_load", load)
    out = be.transcribe(Path("x.wav"), language=None, duration=1, on_progress=None, cancel=None)
    assert calls == ["cuda", "cpu"] and out.language == "en"
    assert "CPU" in be.fallback_note


def test_gpu_oom_without_fallback_is_actionable(monkeypatch):
    cfg = load_config(overrides={"transcription": {"device": "cuda", "allow_cpu_fallback": False}}, local_path=None)
    be = FasterWhisperBackend(cfg, 2)
    be.device, be._model = "cuda", object()
    monkeypatch.setattr(be, "_run", lambda *a: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")))
    with pytest.raises(ResourceExhaustedError) as ei:
        be.transcribe(Path("x.wav"), language=None, duration=1, on_progress=None, cancel=None)
    assert "performance.mode" in ei.value.recovery and "smaller" in ei.value.recovery


def test_error_classification():
    assert is_gpu_memory_error(RuntimeError("CUDA failed with error out of memory"))
    assert is_gpu_memory_error(RuntimeError("Library cublas64_12.dll is not found"))
    assert not is_gpu_memory_error(ValueError("bad value"))


def rep(vram_mb=None, cuda=0, ram=16.0, ollama=True, models=("qwen3:4b",)):
    return {"system": {"ram_total_gb": ram}, "gpu": {"available": bool(vram_mb), "gpus": [{"vram_total_mb": vram_mb or 0}]},
            "ctranslate2": {"cuda_devices": cuda}, "ffmpeg": {"ffmpeg": "x", "ffprobe": "x", "libx264": True, "ass_filter": True},
            "ollama": {"server": ollama, "models": list(models)}}


def test_recommendations_for_target_hardware():
    r = recommend(rep(6144, 1))
    assert (r["whisper_model"], r["whisper_device"], r["llm_model"], r["performance_mode"]) == ("small", "cuda", "qwen3:4b", "normal")
    assert not r["notes"] or "6 GB" in r["notes"][0]
    assert any("medium" in n for n in recommend(rep(8192, 1))["notes"])
    small = recommend(rep(3072, 1))
    assert small["llm_model"] == "qwen3:1.7b"
    cpu = recommend(rep(None, 0, ram=8, ollama=False))
    assert cpu["whisper_device"] == "cpu" and cpu["performance_mode"] == "low" and any("Ollama" in n for n in cpu["notes"])
    broken = recommend(rep(6144, 0))
    assert broken["whisper_device"] == "cpu" and any("CUDA" in n for n in broken["notes"])
    assert any("ollama pull" in n for n in recommend(rep(6144, 1, models=()))["notes"])


# ---------------------------------------------------------------- UI settings mapping / CLI guards
def ui_values(**kw):
    v = dict(whisper="medium", device="cpu", language=" en ", llm_model="qwen3:1.7b", use_llm=False, perf="low", min_d=10, target_d=30,
             max_d=50, max_clips=4, min_score=55, separation=20, layout="smart_crop", resolution="720 × 1280 (faster)", hw="never",
             subs_on=True, subs_mode="plain", sub_pos="middle", font_size=64, hi_color="#00FF00")
    v.update(kw)
    return v


def test_ui_settings_become_valid_config():
    cfg = load_config(overrides=collect_overrides(ui_values()), local_path=None)
    assert cfg.transcription.model == "medium" and cfg.transcription.language == "en" and not cfg.llm.use_llm
    assert (cfg.render.width, cfg.render.height, cfg.render.preview_width) == (720, 1280, 360)
    assert cfg.selection.max_clips == 4 and cfg.subtitles.highlight_color == "#00FF00" and cfg.performance.mode == "low"
    assert load_config(overrides=collect_overrides(ui_values(language="")), local_path=None).transcription.language is None
    for preset in RES_PRESETS:
        load_config(overrides=collect_overrides(ui_values(resolution=preset)), local_path=None)


def test_cli_refuses_non_local_bind(capsys):
    rc = app_module.main(["ui", "--host", "0.0.0.0", "--no-browser"])
    assert rc == 2 and "Refusing" in capsys.readouterr().err


def test_cli_check_runs(capsys):
    assert app_module.main(["check"]) == 0
    assert "Recommended" in capsys.readouterr().out


def test_cli_run_reports_missing_file(capsys):
    assert app_module.main(["run", "/nonexistent/video.mp4"]) == 1
    assert "File not found" in capsys.readouterr().err


@needs_ffmpeg
def test_cli_run_writes_the_project_log(tmp_path, sample_info, monkeypatch):
    def fake_run(self, **kw):
        logging.getLogger("src.core.job_manager").warning("stage ran")
        return JobReport(success=True)

    monkeypatch.setattr(JobManager, "run", fake_run)
    before = set(logging.getLogger().handlers)
    try:
        assert app_module.main(["run", sample_info.path, "--no-llm", "--out", str(tmp_path)]) == 0
    finally:
        for h in set(logging.getLogger().handlers) - before:
            detach_handler(h)
    log = next(tmp_path.glob("*/logs/clipper.log"))
    assert "stage ran" in log.read_text(encoding="utf-8")


def test_settings_keep_the_startup_config(tmp_path):
    custom = tmp_path / "custom.yaml"
    custom.write_text("selection: {max_clips: 3}\npaths: {projects_dir: elsewhere}\n", encoding="utf-8")
    wb = Workbench(load_config(custom), preview_dir=tmp_path)
    wb.set_config(collect_overrides(ui_values(max_clips=4)))
    assert wb.cfg.selection.max_clips == 4 and wb.cfg.render.mode == "smart_crop", "UI values apply"
    assert wb.cfg.paths.projects_dir == "elsewhere", "--config and --projects-dir survive"
    wb.set_config({"subtitles": {"enabled": False}})
    assert wb.cfg.selection.max_clips == 3 and wb.cfg.render.mode == "blur", "each call starts again from the startup config"
    with pytest.raises(ConfigError):
        wb.set_config({"selection": {"min_duration_s": 90}})


@needs_ffmpeg
def test_opening_another_project_clears_the_old_progress(tmp_path, sample_info):
    wb = Workbench(load_config(local_path=None), preview_dir=tmp_path / "ui")
    wb.progress.update(overall=0.4, message="Stopped: old failure", report=JobReport(success=False, error="old failure"))
    try:
        wb.open_video(sample_info.path, str(tmp_path / "projects"))
        assert "old failure" not in wb.progress_html() and "Idle" in wb.progress_html()
    finally:
        detach_handler(wb._log_handler)


# ---------------------------------------------------------------- zero paid API / privacy audit
FORBIDDEN_IMPORTS = re.compile(r"^\s*(?:import|from)\s+(anthropic|openai|google\.generativeai|google\.genai|cohere|mistralai|replicate|together|groq|boto3|azure)\b", re.M)
FORBIDDEN_ENV = re.compile(r"ANTHROPIC_API_KEY|OPENAI_API_KEY|GEMINI_API_KEY|GOOGLE_API_KEY|api\.openai\.com|api\.anthropic\.com", re.I)


def project_sources():
    files = [ROOT / "app.py", ROOT / "config.py", *sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "scripts").rglob("*.py"))]
    return files


def test_no_paid_api_clients_or_keys_anywhere():
    for f in project_sources():
        text = f.read_text(encoding="utf-8")
        assert not FORBIDDEN_IMPORTS.search(text), f"{f} imports a cloud AI SDK"
        assert not FORBIDDEN_ENV.search(text), f"{f} references a paid-API key or endpoint"
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in ("anthropic", "openai", "google-generativeai", "ultralytics", "boto3"):
        assert pkg not in reqs, f"{pkg} must not be a requirement"


def test_only_local_network_use_in_runtime_code():
    """Runtime code may only talk HTTP to the (loopback-guarded) Ollama client; no other URLs are hard-coded."""
    offenders = []
    for f in project_sources():
        if f.name == "download_models.py":
            continue
        for m in re.finditer(r"https?://[^\s\"'`)>]+", f.read_text(encoding="utf-8")):
            url = m.group(0)
            if not re.match(r"https?://(127\.0\.0\.1|localhost|\[::1\])", url) and "ollama.com" not in url and "ffmpeg.org" not in url:
                offenders.append((f.name, url))
    assert not offenders, offenders


def test_ui_binds_localhost_by_default_and_telemetry_is_off():
    assert app_module.LOOPBACK >= {"127.0.0.1", "localhost"}
    import os

    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False" and os.environ["HF_HUB_OFFLINE"] == "1"
    assert "analytics_enabled=False" in (ROOT / "src/review/interface.py").read_text(encoding="utf-8")


def test_gitignore_blocks_media_and_secrets():
    gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pat in ("*.mp4", ".env", "venv/", "projects/", "*.pt", "*.safetensors", "cache/"):
        assert pat in gi
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert not [t for t in tracked if t.endswith((".mp4", ".mov", ".mkv", ".wav", ".pt", ".bin", ".env"))]
    assert sys.version_info >= (3, 10)
