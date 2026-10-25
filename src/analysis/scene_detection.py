"""Scene-change detection using FFmpeg's scene score (fast, multithreaded decode, tiny memory)."""
from __future__ import annotations

import re
import threading
from typing import Callable

from pydantic import BaseModel, Field

from config import ScenesConfig
from src.core.models import VideoInfo
from src.rendering.ffmpeg_runner import FFmpegRunner

_PTS = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
MIN_SCENE_S = 1.0


class Scene(BaseModel):
    start: float
    end: float


class SceneAnalysis(BaseModel):
    cuts: list[float] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    thumbnails: list[tuple[float, str]] = Field(default_factory=list)   # (time, filename in cache/frames)
    keyframes_only: bool = False

    def cuts_in(self, start: float, end: float) -> int:
        return sum(1 for c in self.cuts if start < c < end)

    def cut_rate(self, start: float, end: float) -> float:
        """Cuts per second inside [start, end]."""
        return self.cuts_in(start, end) / max(end - start, 1e-6)


def parse_showinfo(stderr: str) -> list[float]:
    return [float(m.group(1)) for line in stderr.splitlines() if "showinfo" in line for m in [_PTS.search(line)] if m]


def build_scene_command(info: VideoInfo, cfg: ScenesConfig) -> list[str]:
    args = ["-loglevel", "info"]
    if cfg.keyframes_only:
        args += ["-skip_frame", "nokey"]
    args += ["-i", info.path, "-an", "-sn", "-vf", f"scale=320:-2,select='gt(scene,{cfg.threshold})',showinfo", "-f", "null", "-"]
    return args


def build_scenes(cuts: list[float], duration: float) -> list[Scene]:
    kept: list[float] = []
    last = 0.0
    for c in sorted(cuts):
        if c - last >= MIN_SCENE_S and duration - c >= MIN_SCENE_S:
            kept.append(round(c, 3))
            last = c
    bounds = [0.0, *kept, duration]
    return [Scene(start=a, end=b) for a, b in zip(bounds, bounds[1:])]


def detect_scenes(
    info: VideoInfo, runner: FFmpegRunner, cfg: ScenesConfig,
    *, cancel: threading.Event | None = None, on_progress: Callable[[float], None] | None = None,
) -> SceneAnalysis:
    res = runner.run(build_scene_command(info, cfg), duration=info.duration, on_progress=on_progress, cancel=cancel, keep_stderr=True)
    scenes = build_scenes(parse_showinfo(res.stderr), info.duration)
    cuts = [s.start for s in scenes[1:]]
    return SceneAnalysis(cuts=cuts, scenes=scenes, keyframes_only=cfg.keyframes_only)
