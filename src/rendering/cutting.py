"""One-pass clip rendering: accurate cut + layout + captions + H.264/AAC, with a content-addressed render cache."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from config import AppConfig
from src.core.exceptions import FFmpegError, JobCancelled, RenderError
from src.core.models import Candidate, SubtitleLine, VideoInfo
from src.rendering import subtitles as subs
from src.rendering.encoding import EncoderChoice, audio_args, choose_encoder
from src.rendering.ffmpeg_runner import FFmpegRunner
from src.rendering.subject_tracking import CropPlan, make_detector, plan_crop_path, sample_subject_track
from src.rendering.vertical_video import blur_layout, crop_layout, original_layout
from src.storage.project_store import ProjectStore

log = logging.getLogger(__name__)
RENDER_VERSION = 3
ProgressFn = Callable[[float], None]


@dataclass
class RenderResult:
    path: Path
    cached: bool
    seconds: float
    encoder: str
    notes: list[str]


def output_size(cfg: AppConfig, info: VideoInfo, preview: bool) -> tuple[int, int]:
    r = cfg.render
    if r.mode == "original":
        # keep the source orientation/aspect, bounded by the configured size
        w, h = (r.preview_width, r.preview_height) if preview else (r.width, r.height)
        if info.width >= info.height:
            w, h = max(w, h), min(w, h)
        scale = min(w / info.width, h / info.height, 1.0)
        return max(2, int(info.width * scale) // 2 * 2), max(2, int(info.height * scale) // 2 * 2)
    return (r.preview_width, r.preview_height) if preview else (r.width, r.height)


def build_render_args(
    info: VideoInfo, start: float, end: float, cfg: AppConfig, encoder: EncoderChoice, out_path: Path,
    *, preview: bool, ass_name: str | None, crop_points: list[tuple[float, float]] | None,
) -> list[str]:
    """Complete ffmpeg argument list (without the executable). Input-side -ss + re-encode = frame-accurate cut."""
    w, h = output_size(cfg, info, preview)
    mode = cfg.render.mode
    if mode == "smart_crop":
        layout = crop_layout(info.width, info.height, w, h, crop_points)
    elif mode == "original":
        layout = original_layout(w, h)
    else:
        layout = blur_layout(info.width, info.height, w, h, cfg.render)
    fps = min(60.0, info.fps) if info.fps >= 5 else 30.0
    graph = f"[0:v:0]fps={fps:.3f},{layout}"      # every layout ends with a filter, so a label can follow
    graph += f",ass={ass_name}[vout]" if ass_name else "[vout]"
    return [
        "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", info.path, "-t", f"{end - start:.3f}",
        "-filter_complex", graph, "-map", "[vout]", *encoder.video_args, "-r", f"{fps:.3f}",
        *audio_args(cfg.render.audio_bitrate, info.has_audio), "-movflags", "+faststart", "-sn", "-dn", str(out_path),
    ]


class ClipRenderer:
    def __init__(self, info: VideoInfo, store: ProjectStore, cfg: AppConfig, runner: FFmpegRunner):
        self.info, self.store, self.cfg, self.runner = info, store, cfg, runner

    # ---- subtitles -----------------------------------------------------------------------------
    def subtitle_lines(self, c: Candidate, transcript) -> list[SubtitleLine]:
        if not self.cfg.subtitles.enabled:
            return []
        if c.subtitle_lines is not None:
            return subs.clip_lines(c.subtitle_lines, c.start, c.end)
        return subs.lines_from_transcript(transcript, c.start, c.end, self.cfg.subtitles)

    def render_key(self, c: Candidate, lines: list[SubtitleLine], preview: bool) -> str:
        blob = json.dumps({
            "v": RENDER_VERSION, "fp": self.info.fingerprint, "s": round(c.start, 3), "e": round(c.end, 3), "p": preview,
            "render": self.cfg.render.model_dump(), "subs": self.cfg.subtitles.model_dump() if self.cfg.subtitles.enabled else None,
            "sc": self.cfg.smart_crop.model_dump() if self.cfg.render.mode == "smart_crop" else None,
            "lines": [l.model_dump() for l in lines],
        }, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def render_path(self, c: Candidate, lines: list[SubtitleLine], preview: bool) -> Path:
        return self.store.renders_dir / f"{c.id}_{'preview' if preview else 'final'}_{self.render_key(c, lines, preview)}.mp4"

    # ---- render --------------------------------------------------------------------------------
    def render(self, c: Candidate, transcript, *, preview: bool = False, cancel: threading.Event | None = None,
               on_progress: ProgressFn | None = None, force: bool = False) -> RenderResult:
        t0 = time.monotonic()
        lines = self.subtitle_lines(c, transcript)
        out = self.render_path(c, lines, preview)
        if out.exists() and out.stat().st_size > 0 and not force:
            return RenderResult(out, True, 0.0, "cache", [])
        self.store.renders_dir.mkdir(parents=True, exist_ok=True)
        notes: list[str] = []

        ass_name = None
        if lines:
            w, h = output_size(self.cfg, self.info, preview)
            ass_name = out.stem + ".ass"
            (self.store.renders_dir / ass_name).write_text(
                subs.build_ass(lines, c.start, c.end, w, h, self.cfg.subtitles), encoding="utf-8")

        crop_points = None
        if self.cfg.render.mode == "smart_crop":
            plan = self.plan_crop(c, preview, cancel)
            crop_points = plan.keypoints
            notes += plan.notes

        tmp = out.with_suffix(".part.mp4")
        encoder = choose_encoder(self.runner, self.cfg.render, preview=preview)
        for attempt in (0, 1):
            args = build_render_args(self.info, c.start, c.end, self.cfg, encoder, tmp, preview=preview, ass_name=ass_name, crop_points=crop_points)
            try:
                # cwd = renders dir lets the filter refer to the .ass by bare name (no Windows drive-letter escaping)
                self.runner.run(args, duration=c.end - c.start, on_progress=on_progress, cancel=cancel, cwd=self.store.renders_dir)
                break
            except JobCancelled:
                tmp.unlink(missing_ok=True)
                raise
            except FFmpegError as exc:
                tmp.unlink(missing_ok=True)
                if attempt == 0 and encoder.hardware:
                    log.warning("NVENC encode failed (%s); retrying with x264", (exc.details or "")[-200:])
                    notes.append("GPU encoder failed; used CPU x264 instead.")
                    encoder = choose_encoder(self.runner, self.cfg.render, preview=preview, allow_hardware=False)
                    continue
                raise RenderError(f"Rendering clip {c.id} failed.", details=exc.details) from exc
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RenderError(f"FFmpeg produced no output for clip {c.id}.")
        tmp.replace(out)
        for stale in self.store.renders_dir.glob(f"{c.id}_{'preview' if preview else 'final'}_*"):
            if stale.stem != out.stem:
                stale.unlink(missing_ok=True)
        secs = time.monotonic() - t0
        self.store.mutate(lambda s: setattr(s, "render_seconds", s.render_seconds + secs))
        return RenderResult(out, False, secs, encoder.name, notes)

    def plan_crop(self, c: Candidate, preview: bool, cancel) -> CropPlan:
        info, sc = self.info, self.cfg.smart_crop
        w, h = output_size(self.cfg, info, preview)
        crop_w = min(info.width, int(round(info.height * w / h / 2)) * 2)
        if info.width / info.height < w / h or sc.detector == "none":
            return CropPlan([(0.0, (info.width - crop_w) / 2)], 0.0, True, ["Smart crop: centre crop (no detector or source not wider than target)."])
        detector = make_detector(sc.detector)
        if detector is None:
            return CropPlan([(0.0, (info.width - crop_w) / 2)], 0.0, True, ["Smart crop: detector unavailable - centre crop used."])
        samples = sample_subject_track(Path(info.path), c.start, c.end, detector, sc, cancel)
        plan = plan_crop_path(samples, info.width, crop_w, sc)
        if not plan.fallback:
            plan.notes.append(f"Smart crop followed a subject ({plan.detection_ratio:.0%} of samples, {len(plan.keypoints)} keypoints).")
        return plan
