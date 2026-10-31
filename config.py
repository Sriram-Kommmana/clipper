"""Centralised, validated configuration.

Layering (later wins): configs/default.yaml -> configs/local.yaml (optional, git-ignored)
-> explicit overrides passed in code / from the UI.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.core.exceptions import ConfigError

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "configs" / "default.yaml"
LOCAL_CONFIG_PATH = APP_DIR / "configs" / "local.yaml"

CATEGORIES = ("story", "funny", "emotional", "surprising", "insight", "quote", "reaction")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PathsConfig(_Section):
    projects_dir: str = "projects"
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


class PerformanceConfig(_Section):
    mode: Literal["normal", "low"] = "normal"
    cpu_threads: int = Field(0, ge=0, le=64)
    low_priority: bool = False


class TranscriptionConfig(_Section):
    transcript_file: str | None = None
    model: str = "large-v3-turbo"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: Literal["auto", "float16", "int8_float16", "int8", "float32"] = "auto"
    language: str | None = None
    beam_size: int = Field(5, ge=1, le=10)
    vad_filter: bool = True
    allow_cpu_fallback: bool = True


class AudioConfig(_Section):
    window_s: float = Field(0.5, gt=0.05, le=5)
    min_silence_s: float = Field(0.8, gt=0)
    silence_floor_db: float = -50.0
    sustained_threshold: float = Field(0.75, ge=0, le=1)
    sustained_min_s: float = Field(3.0, gt=0)
    loud_margin_db: float = Field(6.0, ge=0, le=30)
    change_sigma: float = Field(2.5, gt=0)
    peak_prominence: float = Field(0.25, gt=0, le=1)


class ScenesConfig(_Section):
    enabled: bool = True
    threshold: float = Field(0.30, gt=0, lt=1)
    keyframes_only: bool = False
    max_thumbnails: int = Field(120, ge=0, le=1000)


class LLMConfig(_Section):
    host: str = "http://127.0.0.1:11434"
    allow_remote_host: bool = False
    model: str = "qwen3:4b"
    num_ctx: int = Field(8192, ge=1024, le=131072)
    num_predict: int = Field(1800, ge=64, le=16384)
    temperature: float = Field(0.3, ge=0, le=2)
    seed: int = 7
    keep_alive: str = "5m"
    timeout_s: float = Field(300, gt=1)
    max_retries: int = Field(2, ge=0, le=5)
    section_seconds: float = Field(600, ge=60)
    section_overlap_s: float = Field(90, ge=0)
    section_max_words: int = Field(1400, ge=100)
    scoring_batch_size: int = Field(4, ge=1, le=16)
    use_llm: bool = True

    @model_validator(mode="after")
    def _check(self) -> "LLMConfig":
        if self.section_overlap_s >= self.section_seconds:
            raise ValueError("llm.section_overlap_s must be smaller than llm.section_seconds")
        return self


class BoundariesConfig(_Section):
    pad_before_s: float = Field(0.25, ge=0, le=3)
    pad_after_s: float = Field(0.35, ge=0, le=3)
    sentence_snap_s: float = Field(3.0, ge=0, le=15)
    reaction_lead_in_s: float = Field(8.0, ge=0, le=30)


class SelectionConfig(_Section):
    min_duration_s: float = Field(15, gt=1)
    max_duration_s: float = Field(90, gt=1)
    target_duration_s: float = Field(45, gt=1)
    overflow_tolerance: float = Field(1.2, ge=1.0, le=2.0)
    max_clips: int = Field(10, ge=1, le=100)
    min_score: float = Field(45, ge=0, le=100)
    min_separation_s: float = Field(5, ge=0)
    diversity_penalty: float = Field(6, ge=0, le=50)
    category_preferences: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "SelectionConfig":
        if not self.min_duration_s < self.max_duration_s:
            raise ValueError("selection.min_duration_s must be smaller than selection.max_duration_s")
        if not self.min_duration_s <= self.target_duration_s <= self.max_duration_s:
            raise ValueError("selection.target_duration_s must lie between min and max duration")
        bad = [k for k in self.category_preferences if k not in CATEGORIES]
        if bad:
            raise ValueError(f"unknown categories in selection.category_preferences: {bad}; valid: {CATEGORIES}")
        return self


class ScoringConfig(_Section):
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "hook": 0.22, "clarity": 0.18, "completeness": 0.18, "payoff": 0.17,
            "escalation": 0.08, "format_fit": 0.09, "av_suitability": 0.08,
        }
    )

    @model_validator(mode="after")
    def _check(self) -> "ScoringConfig":
        expected = {"hook", "clarity", "completeness", "payoff", "escalation", "format_fit", "av_suitability"}
        if set(self.weights) != expected:
            raise ValueError(f"scoring.weights must have exactly the keys {sorted(expected)}")
        if any(w < 0 for w in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("scoring.weights must be non-negative with a positive sum")
        return self


class SubtitlesConfig(_Section):
    enabled: bool = True
    mode: Literal["karaoke", "plain"] = "karaoke"
    font: str = "Arial"
    font_size: int = Field(78, ge=10, le=300)
    bold: bool = True
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFE135"
    outline_color: str = "#000000"
    outline_px: float = Field(5, ge=0, le=30)
    shadow_px: float = Field(2, ge=0, le=30)
    position: Literal["bottom", "middle", "top"] = "bottom"
    margin_v_pct: float = Field(0.20, ge=0, le=0.45)
    margin_h_pct: float = Field(0.07, ge=0, le=0.3)
    max_words: int = Field(4, ge=1, le=12)
    max_chars: int = Field(26, ge=8, le=80)
    pause_s: float = Field(0.45, gt=0)
    uppercase: bool = False
    pop_animation: bool = True

    @model_validator(mode="after")
    def _colors(self) -> "SubtitlesConfig":
        for name in ("primary_color", "highlight_color", "outline_color"):
            v = getattr(self, name)
            if not (isinstance(v, str) and len(v) == 7 and v[0] == "#" and all(c in "0123456789abcdefABCDEF" for c in v[1:])):
                raise ValueError(f"subtitles.{name} must look like '#RRGGBB', got {v!r}")
        return self


class RenderConfig(_Section):
    mode: Literal["blur", "smart_crop", "original"] = "blur"
    width: int = Field(1080, ge=144, le=4320)
    height: int = Field(1920, ge=144, le=7680)
    crf: int = Field(20, ge=0, le=51)
    preset: str = "veryfast"
    audio_bitrate: str = "160k"
    hardware_encoding: Literal["auto", "never"] = "auto"
    blur_sigma: float = Field(24, ge=0, le=100)
    background_dim: float = Field(0.15, ge=0, le=0.9)
    preview_width: int = Field(540, ge=144, le=2160)
    preview_height: int = Field(960, ge=144, le=3840)
    preview_crf: int = Field(30, ge=0, le=51)

    @model_validator(mode="after")
    def _even(self) -> "RenderConfig":
        for n in ("width", "height", "preview_width", "preview_height"):
            if getattr(self, n) % 2:
                raise ValueError(f"render.{n} must be even (H.264 requirement)")
        return self


class SmartCropConfig(_Section):
    detector: Literal["face", "person", "none"] = "face"
    sample_interval_s: float = Field(0.75, ge=0.1, le=5)
    min_detection_ratio: float = Field(0.25, ge=0, le=1)
    deadband_pct: float = Field(0.10, ge=0, le=0.5)
    smoothing: float = Field(0.35, ge=0, le=1)
    max_keypoints: int = Field(24, ge=2, le=100)


class ExportConfig(_Section):
    write_srt: bool = True
    write_metadata: bool = True


class LoggingConfig(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class AppConfig(_Section):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    scenes: ScenesConfig = Field(default_factory=ScenesConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    boundaries: BoundariesConfig = Field(default_factory=BoundariesConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    subtitles: SubtitlesConfig = Field(default_factory=SubtitlesConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    smart_crop: SmartCropConfig = Field(default_factory=SmartCropConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def effective(self) -> "AppConfig":
        """Return a copy with performance-mode adjustments applied (does not mutate self)."""
        cfg = self.model_copy(deep=True)
        if cfg.performance.mode == "low":
            cfg.performance.low_priority = True
            cfg.scenes.keyframes_only = True
            cfg.transcription.beam_size = 1
            cfg.llm.num_ctx = min(cfg.llm.num_ctx, 4096)
            cfg.llm.section_max_words = min(cfg.llm.section_max_words, 700)
            cfg.llm.num_predict = min(cfg.llm.num_predict, 1200)
            cfg.llm.scoring_batch_size = min(cfg.llm.scoring_batch_size, 3)
            cfg.llm.keep_alive = "0"  # unload the LLM from VRAM/RAM as soon as a request finishes
        return cfg

    def resolved_threads(self) -> int:
        """CPU threads to use for FFmpeg / CTranslate2 / OpenCV."""
        cores = os.cpu_count() or 2
        if self.performance.cpu_threads > 0:
            return self.performance.cpu_threads
        if self.performance.mode == "low":
            return max(1, cores // 2 - 1) if cores > 3 else 1
        return max(1, min(cores, 8))

    def fingerprint(self, *sections: str) -> str:
        """Stable hash of selected config sections, used for cache keys."""
        data = {s: getattr(self, s).model_dump(mode="json") for s in sorted(sections)}
        return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]

    def with_overrides(self, overrides: dict[str, Any]) -> "AppConfig":
        """A validated copy with `overrides` applied on top. Raises ConfigError with a readable message."""
        return _validate(_deep_merge(self.model_dump(mode="json"), overrides))


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read configuration file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {path} must contain a YAML mapping at top level")
    return data


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    local_path: str | Path | None = LOCAL_CONFIG_PATH,
) -> AppConfig:
    """Load and validate configuration. Raises ConfigError with a readable message."""
    data = _read_yaml(Path(path) if path else DEFAULT_CONFIG_PATH)
    if local_path and Path(local_path).is_file() and path is None:
        data = _deep_merge(data, _read_yaml(Path(local_path)))
    if overrides:
        data = _deep_merge(data, overrides)
    return _validate(data)


def _validate(data: dict[str, Any]) -> AppConfig:
    try:
        return AppConfig.model_validate(data)
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            lines.append(f"  - {loc}: {err['msg']}")
        raise ConfigError("Invalid configuration:\n" + "\n".join(lines)) from exc
