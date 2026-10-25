"""Encoder selection and quality arguments. Hardware (NVENC) is used only after a real test encode succeeds."""
from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass

from config import RenderConfig
from src.rendering.ffmpeg_runner import FFmpegRunner, _popen_kwargs

log = logging.getLogger(__name__)
_nvenc_cache: dict[str, bool] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class EncoderChoice:
    name: str            # libx264 | h264_nvenc
    video_args: tuple[str, ...]
    hardware: bool

    @property
    def label(self) -> str:
        return "NVENC (GPU)" if self.hardware else "x264 (CPU)"


def nvenc_usable(runner: FFmpegRunner) -> bool:
    """True iff h264_nvenc is compiled in AND can actually encode here (driver/GPU present). Cached per ffmpeg binary."""
    with _lock:
        if runner.ffmpeg in _nvenc_cache:
            return _nvenc_cache[runner.ffmpeg]
        ok = False
        try:
            cmd = [runner.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
                   "-pix_fmt", "yuv420p", "-c:v", "h264_nvenc", "-f", "null", "-"]
            ok = subprocess.run(cmd, capture_output=True, timeout=20, stdin=subprocess.DEVNULL, **_popen_kwargs(False)).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        _nvenc_cache[runner.ffmpeg] = ok
        log.info("NVENC test encode: %s", "OK" if ok else "unavailable - using x264")
        return ok


def choose_encoder(runner: FFmpegRunner, cfg: RenderConfig, *, preview: bool, allow_hardware: bool = True) -> EncoderChoice:
    crf = cfg.preview_crf if preview else cfg.crf
    if cfg.hardware_encoding == "auto" and allow_hardware and nvenc_usable(runner):
        return EncoderChoice("h264_nvenc", ("-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(min(51, crf + 3)),
                                          "-b:v", "0", "-pix_fmt", "yuv420p", "-profile:v", "high"), True)
    preset = "ultrafast" if preview else cfg.preset
    return EncoderChoice("libx264", ("-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                                     "-profile:v", "high", *(["-threads", str(runner.threads)] if runner.threads else [])), False)


def audio_args(bitrate: str, has_audio: bool) -> list[str]:
    if not has_audio:
        return ["-an"]
    return ["-map", "0:a:0", "-c:a", "aac", "-b:a", bitrate, "-ar", "48000", "-ac", "2", "-af", "aresample=async=1:first_pts=0"]
