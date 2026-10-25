"""Persisted domain models (pydantic: validated on load, JSON-serialisable)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["story", "funny", "emotional", "surprising", "insight", "quote", "reaction"]
ReviewStatus = Literal["pending", "approved", "rejected"]
StageStatus = Literal["pending", "running", "done", "failed", "cancelled", "interrupted", "skipped"]

SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class VideoInfo(BaseModel):
    path: str
    filename: str
    size_bytes: int
    container: str = ""
    duration: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = 0.0
    video_codec: str = ""
    audio_codec: str | None = None
    has_audio: bool = False
    audio_channels: int = 0
    audio_sample_rate: int = 0
    rotation: int = 0
    bit_rate: int = 0
    fingerprint: str = ""
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width

    def summary(self) -> str:
        audio = f"{self.audio_codec} {self.audio_channels}ch {self.audio_sample_rate} Hz" if self.has_audio else "NO AUDIO"
        return (
            f"{self.filename}: {self.width}x{self.height} @ {self.fps:.2f} fps, {self.duration:.1f}s "
            f"({self.video_codec}), audio: {audio}, {self.size_bytes / 1e6:.1f} MB"
        )


class SubtitleWord(BaseModel):
    text: str
    start: float
    end: float


class SubtitleLine(BaseModel):
    """One caption line in *source-video* time (so it survives boundary edits)."""

    start: float
    end: float
    text: str
    words: list[SubtitleWord] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    hook: float = Field(0, ge=0, le=100)
    clarity: float = Field(0, ge=0, le=100)
    completeness: float = Field(0, ge=0, le=100)
    payoff: float = Field(0, ge=0, le=100)
    escalation: float = Field(0, ge=0, le=100)
    format_fit: float = Field(0, ge=0, le=100)
    av_suitability: float = Field(0, ge=0, le=100)
    total: float = Field(0, ge=0, le=100)
    llm_scored: bool = False
    notes: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    id: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    category: Category
    title: str
    explanation: str = ""
    hook: str = ""
    caption: str = ""
    excerpt: str = ""
    context_completeness: float = Field(50, ge=0, le=100)
    payoff_estimate: float = Field(50, ge=0, le=100)
    confidence: float = Field(50, ge=0, le=100)
    warnings: list[str] = Field(default_factory=list)
    source: Literal["llm", "audio", "manual"] = "llm"
    scores: ScoreBreakdown | None = None
    selected: bool = False
    filter_reason: str | None = None
    status: ReviewStatus = "pending"
    subtitle_lines: list[SubtitleLine] | None = None   # user-edited override (None = auto from transcript)
    boundaries_edited: bool = False
    title_edited: bool = False
    merged_from: list[str] = Field(default_factory=list)
    section: int | None = None
    updated_at: str = Field(default_factory=now_iso)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def total_score(self) -> float:
        return self.scores.total if self.scores else 0.0

    def filter_reason_blocks(self) -> bool:
        """True if the candidate failed hard validation (cannot be rendered/approved in bulk)."""
        return bool(self.filter_reason and self.filter_reason.startswith("invalid:"))


class StageState(BaseModel):
    status: StageStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    cache_key: str | None = None
    message: str = ""
    progress: float = 0.0
    error: str | None = None
    recovery: str | None = None
    cached: bool = False        # True when the stage was satisfied from cache on the last run
    device: str | None = None   # e.g. which device transcription actually used


class ProjectState(BaseModel):
    schema_version: int = SCHEMA_VERSION
    project_id: str
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    source: VideoInfo
    stages: dict[str, StageState] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    render_seconds: float = 0.0
    export_seconds: float = 0.0
    last_run_total_s: float | None = None
