"""Shared fixtures. Media fixtures are generated on the fly with FFmpeg (nothing binary is committed)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from config import AppConfig, load_config
from src.core.models import VideoInfo
from src.rendering.ffmpeg_runner import FFmpegRunner

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")


@pytest.fixture(scope="session")
def cfg() -> AppConfig:
    return load_config(local_path=None)


@pytest.fixture(scope="session")
def runner() -> FFmpegRunner:
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not available")
    return FFmpegRunner()


def _gen(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """24 s, 640x360@25, with a scene change every 8 s and a loud burst at 10-14 s over quiet tone."""
    d = tmp_path_factory.mktemp("media")
    out = d / "sample.mp4"
    video = (
        "color=c=white:s=640x360:r=25:d=8[a];color=c=0x202060:s=640x360:r=25:d=8[b];"
        "testsrc2=s=640x360:r=25:d=8[c];[a][b][c]concat=n=3:v=1:a=0[v]"
    )
    audio = "sine=f=220:d=24,volume='if(between(t,10,14),1.0,0.05)':eval=frame[a]"
    _gen(["-filter_complex", f"{video};{audio}", "-map", "[v]", "-map", "[a]",
          "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)])
    return out


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("media_silent")
    out = d / "silent.mp4"
    _gen(["-f", "lavfi", "-i", "testsrc2=s=320x240:r=25:d=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)])
    return out


@pytest.fixture(scope="session")
def sample_info(sample_video, runner) -> VideoInfo:
    from src.ingestion.video_probe import inspect_video

    return inspect_video(sample_video, runner)
