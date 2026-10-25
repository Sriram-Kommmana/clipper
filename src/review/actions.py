"""Review operations (approve / reject / edit bounds / edit subtitles). UI-independent and unit-tested."""
from __future__ import annotations

from config import AppConfig
from src.analysis.audio_analysis import AudioAnalysis
from src.candidates.scorer import compute_breakdown, format_fit
from src.candidates.validator import Limits, validate_manual_bounds
from src.core.exceptions import CandidateValidationError
from src.core.models import Candidate, ScoreBreakdown, SubtitleLine
from src.rendering import subtitles as subs
from src.storage.project_store import ProjectStore
from src.transcription.transcript_models import Transcript


def set_status(store: ProjectStore, ids: list[str], status: str) -> int:
    """Approve / reject / reset. Pure metadata change: nothing is re-rendered."""
    assert status in ("approved", "rejected", "pending")
    n = 0
    for cid in ids:
        store.update_candidate(cid, status=status)
        n += 1
    return n


def bulk_approve(store: ProjectStore, min_score: float, *, only_selected: bool = True) -> list[str]:
    ids = [c.id for c in store.load_candidates()
           if c.status == "pending" and (c.selected or not only_selected) and not c.filter_reason_blocks() and c.total_score >= min_score]
    set_status(store, ids, "approved")
    return ids


def rescore_after_edit(c: Candidate, transcript: Transcript, audio: AudioAnalysis | None, cfg: AppConfig) -> ScoreBreakdown:
    """Refresh duration/audio terms after a boundary edit; language-model terms stay from the original range."""
    old = c.scores
    if old is None:
        return compute_breakdown(c, None, transcript, audio, None, cfg)
    vals = {"hook": old.hook, "clarity": old.clarity, "completeness": old.completeness, "payoff": old.payoff, "escalation": old.escalation}
    fresh = compute_breakdown(c, vals, transcript, audio, None, cfg)
    notes = [n for n in old.notes if not n.startswith("Boundaries edited")]
    notes.append("Boundaries edited: duration/audio terms recomputed; language-model terms are from the original range.")
    return fresh.model_copy(update={"llm_scored": old.llm_scored, "notes": notes})


def update_bounds(store: ProjectStore, cid: str, start: float, end: float, transcript: Transcript, audio: AudioAnalysis | None, cfg: AppConfig) -> tuple[Candidate, list[str]]:
    """Validate and apply manual boundaries. Raises CandidateValidationError for impossible ranges."""
    c = store.get_candidate(cid)
    warns = validate_manual_bounds(float(start), float(end), store.state.source.duration, Limits.from_config(cfg.selection))
    start, end = round(float(start), 3), round(float(end), 3)
    updated = c.model_copy(update={"start": start, "end": end, "excerpt": transcript.text_in_range(start, end), "boundaries_edited": True})
    if c.subtitle_lines is not None:
        updated = updated.model_copy(update={"subtitle_lines": _extend_override(c.subtitle_lines, transcript, start, end, cfg)})
    updated = updated.model_copy(update={"scores": rescore_after_edit(updated, transcript, audio, cfg),
                                         "warnings": [w for w in c.warnings if not w.startswith("Manual:")] + [f"Manual: {w}" for w in warns]})
    out = store.update_candidate(cid, **{k: getattr(updated, k) for k in ("start", "end", "excerpt", "boundaries_edited", "subtitle_lines", "scores", "warnings")})
    return out, warns


def _extend_override(old: list[SubtitleLine], transcript: Transcript, start: float, end: float, cfg: AppConfig) -> list[SubtitleLine]:
    """Keep the user's edited captions and fill any newly covered time from the transcript."""
    if not old:
        return old
    lo, hi = min(l.start for l in old), max(l.end for l in old)
    auto = subs.lines_from_transcript(transcript, start, end, cfg.subtitles)
    extra = [l for l in auto if l.end <= lo + 0.01 or l.start >= hi - 0.01]
    return sorted([*old, *extra], key=lambda l: l.start)


def current_subtitle_lines(c: Candidate, transcript: Transcript, cfg: AppConfig) -> list[SubtitleLine]:
    if c.subtitle_lines is not None:
        return subs.clip_lines(c.subtitle_lines, c.start, c.end)
    return subs.lines_from_transcript(transcript, c.start, c.end, cfg.subtitles)


def subtitle_editor_text(c: Candidate, transcript: Transcript, cfg: AppConfig) -> str:
    return subs.format_for_editor(current_subtitle_lines(c, transcript, cfg), c.start)


def update_subtitles(store: ProjectStore, cid: str, text: str, transcript: Transcript, cfg: AppConfig) -> Candidate:
    """Apply edited caption text/timing. Raises ValueError (with the offending line number) on bad syntax."""
    c = store.get_candidate(cid)
    specs = subs.parse_editor_text(text, c.start, c.end)
    lines = subs.reconcile(current_subtitle_lines(c, transcript, cfg), specs)
    return store.update_candidate(cid, subtitle_lines=[l.model_dump() for l in lines])


def reset_subtitles(store: ProjectStore, cid: str) -> Candidate:
    return store.update_candidate(cid, subtitle_lines=None)


def update_title(store: ProjectStore, cid: str, title: str, caption: str | None = None) -> Candidate:
    title = " ".join(title.split())[:120]
    if not title:
        raise CandidateValidationError("The title cannot be empty.")
    changes: dict = {"title": title, "title_edited": True}
    if caption is not None:
        changes["caption"] = " ".join(caption.split())[:300]
    return store.update_candidate(cid, **changes)


def add_manual_candidate(store: ProjectStore, start: float, end: float, title: str, transcript: Transcript, audio: AudioAnalysis | None, cfg: AppConfig, fingerprint: str) -> Candidate:
    """User-defined clip: validated like an edit, never removed by re-runs, selected for review."""
    from src.candidates.generator import candidate_id

    warns = validate_manual_bounds(float(start), float(end), store.state.source.duration, Limits.from_config(cfg.selection))
    c = Candidate(id=candidate_id(fingerprint, start, end, "quote") + "m", start=round(start, 3), end=round(end, 3), category="quote",
                  title=" ".join(title.split())[:120] or "Manual clip", excerpt=transcript.text_in_range(start, end), source="manual",
                  selected=True, boundaries_edited=True, confidence=100, warnings=[f"Manual: {w}" for w in warns])
    c = c.model_copy(update={"scores": compute_breakdown(c, None, transcript, audio, None, cfg)})
    store.save_candidates([*store.load_candidates(), c])
    return c


def duration_fit(c: Candidate, cfg: AppConfig) -> float:
    s = cfg.selection
    return format_fit(c.duration, s.target_duration_s, s.min_duration_s, s.max_duration_s)
