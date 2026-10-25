"""Candidate scoring (Stage E).

Seven components, each 0-100, combined with configurable weights (scoring.weights):

  hook, clarity, completeness, payoff, escalation
      LLM rubric (1-10 -> x10) judged on the clip's own text plus a little context.
      `completeness` is blended 70 % LLM / 30 % deterministic boundary quality
      (does it start on a sentence start and end on a sentence end?).
  format_fit      deterministic: Gaussian around the target duration (50 at the min/max edges).
  av_suitability  deterministic: 45 % speech density (words/s, ideal 2-3.5), 25 % non-silence,
                  15 % audio energy, 15 % visual check (black / flat frames).

If the LLM is unavailable or a batch fails, neutral LLM-derived values come from the generation-time
ratings and the breakdown says so (`llm_scored=False`).  These numbers are ranking heuristics for
ordering your review queue - they are NOT a prediction of views or engagement.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Callable

from config import AppConfig
from src.analysis.audio_analysis import AudioAnalysis
from src.analysis.frame_extraction import VisualCheck
from src.core.exceptions import LLMOutputError
from src.core.models import Candidate, ScoreBreakdown
from src.core.schemas import ScoreList
from src.llm.ollama_client import LLMClient
from src.llm.output_parser import structured_call
from src.llm.prompts import score_prompt
from src.transcription.transcript_models import Transcript

log = logging.getLogger(__name__)
CONTEXT_S = 10.0


def format_fit(duration: float, target: float, min_s: float, max_s: float) -> float:
    sigma = (max_s - min_s) / 2.4
    score = 100.0 * math.exp(-0.5 * ((duration - target) / sigma) ** 2)
    if duration < min_s or duration > max_s:
        score = min(score, 40.0)
    return max(0.0, min(100.0, score))


def boundary_quality(c: Candidate, transcript: Transcript, tol: float = 0.9) -> float:
    """0 / 50 / 100: starts at a sentence start, ends at a sentence end (padding tolerance `tol`)."""
    pts = 0.0
    if any(abs(c.start - s) <= tol for s in transcript.sentence_starts()):
        pts += 50
    if any(abs(c.end - e) <= tol for e in transcript.sentence_ends()):
        pts += 50
    return pts


def speech_density_score(words: int, duration: float) -> float:
    wps = words / max(duration, 1e-6)
    if wps < 2.0:
        return 100.0 * wps / 2.0
    if wps <= 3.5:
        return 100.0
    return max(40.0, 100.0 - (wps - 3.5) * 25.0)  # extremely dense speech is hard to caption/read


def av_suitability(c: Candidate, transcript: Transcript, audio: AudioAnalysis | None, visual: VisualCheck | None) -> tuple[float, list[str]]:
    notes: list[str] = []
    density = speech_density_score(transcript.word_count_in_range(c.start, c.end), c.duration)
    if audio and audio.has_audio:
        non_silence = 100.0 * (1.0 - audio.silence_ratio(c.start, c.end))
        energy = 40.0 + 60.0 * audio.mean_energy(c.start, c.end)
    else:
        non_silence, energy = 70.0, 60.0
        notes.append("no audio analysis available")
    vis = visual.score if visual else 80.0
    if visual:
        notes.extend(visual.warnings)
    return 0.45 * density + 0.25 * non_silence + 0.15 * energy + 0.15 * vis, notes


def _context(transcript: Transcript, c: Candidate) -> tuple[str, str]:
    return (
        transcript.text_in_range(max(0.0, c.start - CONTEXT_S), c.start)[-400:],
        transcript.text_in_range(c.end, c.end + CONTEXT_S)[:400],
    )


def _llm_scores(
    cands: list[Candidate], transcript: Transcript, cfg: AppConfig, llm: LLMClient,
    cancel: threading.Event | None, on_progress: Callable[[float, str], None] | None,
) -> dict[str, dict[str, float]]:
    """Batch LLM scoring; a failed batch just leaves its candidates unscored (never aborts the stage)."""
    out: dict[str, dict[str, float]] = {}
    size = cfg.effective().llm.scoring_batch_size
    batches = [cands[i:i + size] for i in range(0, len(cands), size)]
    for n, batch in enumerate(batches):
        if on_progress:
            on_progress(n / max(1, len(batches)), f"Scoring candidates: batch {n + 1}/{len(batches)}")
        blocks = []
        for c in batch:
            before, after = _context(transcript, c)
            blocks.append({"id": c.id, "duration": c.duration, "category": c.category,
                           "before": before or "(start of video)", "text": c.excerpt[:1800], "after": after or "(end of video)"})
        p = score_prompt(blocks)
        try:
            res, _ = structured_call(llm, p.system, p.user, ScoreList, retries=cfg.llm.max_retries, cancel=cancel)
        except LLMOutputError as exc:
            log.warning("Scoring batch %d failed: %s", n, exc)
            continue
        valid = {c.id for c in batch}
        for s in res.scores:
            if s.id in valid:
                out[s.id] = {k: getattr(s, k) * 10.0 for k in ("hook", "clarity", "completeness", "payoff", "escalation")}
                out[s.id]["_reason"] = s.reason  # type: ignore[assignment]
    return out


def compute_breakdown(
    c: Candidate, llm_vals: dict | None, transcript: Transcript, audio: AudioAnalysis | None,
    visual: VisualCheck | None, cfg: AppConfig,
) -> ScoreBreakdown:
    sel = cfg.selection
    notes: list[str] = []
    bq = boundary_quality(c, transcript)
    if llm_vals:
        hook, clarity, payoff, esc = (llm_vals[k] for k in ("hook", "clarity", "payoff", "escalation"))
        completeness = 0.7 * llm_vals["completeness"] + 0.3 * bq
        if llm_vals.get("_reason"):
            notes.append(str(llm_vals["_reason"]))
    else:
        hook, esc = 50.0, 50.0
        clarity, payoff = c.context_completeness, c.payoff_estimate
        completeness = 0.5 * c.context_completeness + 0.5 * bq
        notes.append("LLM scoring unavailable - hook/escalation neutral, others from generation-time ratings")
    ff = format_fit(c.duration, sel.target_duration_s, sel.min_duration_s, sel.max_duration_s)
    av, av_notes = av_suitability(c, transcript, audio, visual)
    notes.extend(av_notes)
    comps = {"hook": hook, "clarity": clarity, "completeness": completeness, "payoff": payoff,
             "escalation": esc, "format_fit": ff, "av_suitability": av}
    comps = {k: max(0.0, min(100.0, v)) for k, v in comps.items()}
    w = cfg.scoring.weights
    total = sum(w[k] * comps[k] for k in comps) / sum(w.values())
    return ScoreBreakdown(**{k: round(v, 1) for k, v in comps.items()}, total=round(total, 1),
                          llm_scored=llm_vals is not None, notes=notes)


def score_candidates(
    cands: list[Candidate], transcript: Transcript, audio: AudioAnalysis | None, cfg: AppConfig, llm: LLMClient | None,
    *, visual_fn: Callable[[Candidate], VisualCheck] | None = None, cancel: threading.Event | None = None,
    on_progress: Callable[[float, str], None] | None = None,
) -> list[Candidate]:
    """Returns new Candidates with `.scores` set. Visual checks run only for the best-looking ones (I/O cost)."""
    llm_vals = _llm_scores(cands, transcript, cfg, llm, cancel, on_progress) if (llm and cfg.llm.use_llm and cands) else {}
    first = {c.id: compute_breakdown(c, llm_vals.get(c.id), transcript, audio, None, cfg) for c in cands}
    visuals: dict[str, VisualCheck] = {}
    if visual_fn:
        top = sorted(cands, key=lambda c: first[c.id].total, reverse=True)[: cfg.selection.max_clips * 3]
        for c in top:
            if cancel is not None and cancel.is_set():
                break
            try:
                visuals[c.id] = visual_fn(c)
            except Exception as exc:  # a visual check must never kill scoring
                log.warning("Visual check failed for %s: %s", c.id, exc)
    out: list[Candidate] = []
    for c in cands:
        bd = compute_breakdown(c, llm_vals.get(c.id), transcript, audio, visuals.get(c.id), cfg) if c.id in visuals else first[c.id]
        warnings = list(c.warnings)
        v = visuals.get(c.id)
        if v:
            warnings += [w for w in v.warnings if w not in warnings]
        out.append(c.model_copy(update={"scores": bd, "warnings": warnings}))
    return out
