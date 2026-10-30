"""Narrative candidate generation (Stage D).

1. Cut the transcript into overlapping sections that fit the local model's context.
2. Ask the LLM for self-contained moments per section (schema-validated).
3. Verify everything against the transcript (clamp times, rebuild excerpt, check the hook is a real quote).
4. Add audio-event candidates as a *supplement* for strong moments the LLM did not cover.
5. Merge duplicates (including the same moment seen from two overlapping sections).
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass
from typing import Callable


from config import AppConfig, LLMConfig
from src.analysis.audio_analysis import AudioAnalysis
from src.candidates.merger import merge_candidates
from src.core.exceptions import LLMOutputError
from src.core.models import Candidate
from src.core.schemas import MomentList, RawMoment
from src.llm.ollama_client import LLMClient
from src.llm.output_parser import structured_call
from src.llm.prompts import moment_prompt
from src.transcription.transcript_models import Transcript, TranscriptSegment

log = logging.getLogger(__name__)

MAX_MOMENTS_PER_SECTION = 6
MAX_AUDIO_CANDIDATES = 8


@dataclass
class Section:
    index: int
    start: float
    end: float
    segments: list[TranscriptSegment]

    def render(self) -> str:
        return "\n".join(f"{i}: ({s.end - s.start:.0f}s) {s.text.strip()}" for i, s in enumerate(self.segments))


def candidate_id(fingerprint: str, start: float, end: float, category: str) -> str:
    h = hashlib.sha1(f"{fingerprint}:{round(start)}:{round(end)}:{category}".encode()).hexdigest()
    return "c" + h[:8]


def make_sections(transcript: Transcript, cfg: LLMConfig) -> list[Section]:
    """Greedy sections bounded by time and word count, each overlapping the previous by ~section_overlap_s."""
    segs = [s for s in transcript.segments if s.text.strip()]
    sections: list[Section] = []
    i = 0
    while i < len(segs):
        j, words = i, 0
        while j < len(segs):
            n = len(segs[j].text.split())
            if j > i and (segs[j].end - segs[i].start > cfg.section_seconds or words + n > cfg.section_max_words):
                break
            words += n
            j += 1
        chunk = segs[i:j]
        sections.append(Section(len(sections), chunk[0].start, chunk[-1].end, chunk))
        if j >= len(segs):
            break
        target = chunk[-1].end - cfg.section_overlap_s
        k = next((k for k in range(i + 1, j) if segs[k].start >= target), j - 1 if j - 1 > i else j)
        i = max(k, i + 1)
    return sections


def _norm_words(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower(), flags=re.UNICODE).split()


def hook_is_quote(hook: str, excerpt: str) -> bool:
    """True if the hook's words appear, in order and contiguously, in the excerpt (punctuation/case ignored)."""
    h, e = _norm_words(hook), _norm_words(excerpt)
    if not h:
        return False
    n = len(h)
    return any(e[i:i + n] == h for i in range(len(e) - n + 1))


def first_sentence(text: str, max_words: int = 25) -> str:
    m = re.match(r"(.+?[.!?…])(\s|$)", text.strip())
    sent = m.group(1) if m else text.strip()
    return " ".join(sent.split()[:max_words])


def moment_to_candidate(m: RawMoment, sec: Section, transcript: Transcript, cfg: AppConfig, fingerprint: str) -> tuple[Candidate | None, str | None]:
    """Validate/repair one LLM moment. Returns (candidate, None) or (None, reason_dropped)."""
    sel = cfg.selection
    first, last = max(0, m.first_line), min(len(sec.segments) - 1, m.last_line)
    if last < first:
        return None, f"lines {m.first_line}-{m.last_line} are not a valid range in this section"
    start, end = sec.segments[first].start, sec.segments[last].end
    dur = end - start
    if dur < sel.min_duration_s * 0.5:
        return None, f"too short ({dur:.0f}s)"
    if dur > sel.max_duration_s * sel.overflow_tolerance * 2:
        return None, f"too long ({dur:.0f}s)"
    excerpt = transcript.text_in_range(start, end)
    if not excerpt:
        return None, "no transcript text in range"
    warnings: list[str] = []
    hook = m.hook
    if not hook_is_quote(hook, excerpt):
        hook = first_sentence(excerpt)
        warnings.append("Model's hook was not a verbatim quote; replaced with the clip's first sentence.")
    return Candidate(
        id=candidate_id(fingerprint, start, end, m.category), start=start, end=end, category=m.category,
        title=m.title or first_sentence(excerpt, 8), explanation=m.explanation, hook=hook, caption=m.caption,
        excerpt=excerpt, context_completeness=m.context_completeness * 10, payoff_estimate=m.payoff * 10,
        confidence=m.confidence * 10, warnings=warnings, source="llm", section=sec.index,
    ), None


def llm_candidates(
    transcript: Transcript, cfg: AppConfig, llm: LLMClient, fingerprint: str,
    *, cancel: threading.Event | None = None, on_progress: Callable[[float, str], None] | None = None,
) -> tuple[list[Candidate], dict]:
    sections = make_sections(transcript, cfg.effective().llm)
    stats = {"sections": len(sections), "raw_moments": 0, "dropped": [], "failed_sections": [], "retries": 0}
    out: list[Candidate] = []
    for n, sec in enumerate(sections):
        if on_progress:
            on_progress(n / max(1, len(sections)), f"Finding moments: section {n + 1}/{len(sections)}")
        p = moment_prompt(
            transcript=sec.render(), a=sec.start, b=sec.end, language=transcript.language,
            min_s=cfg.selection.min_duration_s, max_s=cfg.selection.max_duration_s, max_moments=MAX_MOMENTS_PER_SECTION,
        )
        try:
            result, attempts = structured_call(llm, p.system, p.user, MomentList, retries=cfg.llm.max_retries, cancel=cancel)
        except LLMOutputError as exc:
            log.warning("Section %d skipped: %s", sec.index, exc)
            stats["failed_sections"].append(sec.index)
            continue
        stats["retries"] += attempts - 1
        stats["raw_moments"] += len(result.moments)
        for m in result.moments:
            cand, why = moment_to_candidate(m, sec, transcript, cfg, fingerprint)
            if cand:
                out.append(cand)
            else:
                stats["dropped"].append(f"section {sec.index}: {why}")
    if sections and len(stats["failed_sections"]) == len(sections):
        raise LLMOutputError(
            "The local model failed to produce valid output for every section of the transcript.",
            recovery="Try a larger model (llm.model: qwen3:8b), lower llm.section_max_words, or set llm.use_llm: false for the heuristic fallback.",
        )
    return out, stats


def audio_candidates(
    transcript: Transcript, audio: AudioAnalysis | None, cfg: AppConfig, existing: list[Candidate],
    fingerprint: str, source_duration: float,
) -> list[Candidate]:
    """Strong audio events (sustained excitement / big peaks) that the LLM candidates do not already cover.

    Timing rule specific to this category: the cause of a reaction usually precedes the audio, so the
    window starts up to `boundaries.reaction_lead_in_s` before the event, on a sentence edge.
    """
    if not audio or not audio.has_audio:
        return []
    lead = cfg.boundaries.reaction_lead_in_s
    sel = cfg.selection
    events = [e for e in audio.events if e.kind == "sustained" or (e.kind == "peak" and e.strength >= 0.85)]
    events.sort(key=lambda e: e.strength * (1 + (e.end - e.start)), reverse=True)
    out: list[Candidate] = []
    for ev in events:
        if len(out) >= MAX_AUDIO_CANDIDATES:
            break
        mid = (ev.start + ev.end) / 2
        if any(c.start <= mid <= c.end for c in existing + out):
            continue
        start = max(0.0, ev.start - lead)
        end = min(source_duration, max(ev.end + 3.0, start + sel.min_duration_s))
        start = max(0.0, min(start, end - sel.min_duration_s))
        if end - start < sel.min_duration_s * 0.9:
            continue
        excerpt = transcript.text_in_range(start, end)
        warnings = ["Triggered by audio energy only - confirm in the preview that it is a real highlight."]
        if transcript.word_count_in_range(start, end) < 8:
            warnings.append("Little or no speech around this event; it relies on visuals/sound.")
        out.append(Candidate(
            id=candidate_id(fingerprint, start, end, "reaction"), start=start, end=end, category="reaction",
            title=f"High-energy moment at {int(ev.start // 60)}:{int(ev.start % 60):02d}",
            explanation="Sustained loud or energetic audio (e.g. laughter, applause or excitement) with its lead-in.",
            hook=first_sentence(excerpt) if excerpt else "", excerpt=excerpt, context_completeness=40, payoff_estimate=55,
            confidence=25, warnings=warnings, source="audio",
        ))
    return out


def heuristic_candidates(transcript: Transcript, audio: AudioAnalysis | None, cfg: AppConfig, fingerprint: str, source_duration: float) -> list[Candidate]:
    """Fallback when no LLM is available: sentence-aligned windows around the target duration, picked by
    speech density and audio energy. No semantic understanding - clearly flagged as such."""
    sel = cfg.selection
    starts, ends = transcript.sentence_starts(), transcript.sentence_ends()
    if not starts or not ends:
        return []
    scored: list[tuple[float, float, float]] = []
    for s in starts:
        e = next((x for x in ends if x >= s + sel.target_duration_s * 0.8), None)
        if e is None or e - s > sel.max_duration_s:
            continue
        wps = transcript.word_count_in_range(s, e) / (e - s)
        energy = audio.mean_energy(s, e) if audio and audio.has_audio else 0.3
        scored.append((min(wps / 3.0, 1.0) * 0.6 + energy * 0.4, s, e))
    scored.sort(reverse=True)
    out: list[Candidate] = []
    for _, s, e in scored[:40]:
        if any(min(e, c.end) - max(s, c.start) > 0 for c in out):
            continue
        ex = transcript.text_in_range(s, e)
        out.append(Candidate(
            id=candidate_id(fingerprint, s, e, "insight"), start=s, end=e, category="insight",
            title=first_sentence(ex, 8) or "Untitled moment", explanation="Chosen by speech density and audio energy (no language model used).",
            hook=first_sentence(ex), excerpt=ex, context_completeness=45, payoff_estimate=45, confidence=30,
            warnings=["Heuristic candidate: no language model was used, so completeness/payoff are guesses."], source="llm",
        ))
        if len(out) >= 12:
            break
    return out


def generate_candidates(
    transcript: Transcript, audio: AudioAnalysis | None, cfg: AppConfig, llm: LLMClient | None, fingerprint: str,
    source_duration: float, *, cancel: threading.Event | None = None, on_progress: Callable[[float, str], None] | None = None,
) -> tuple[list[Candidate], dict]:
    stats: dict = {"mode": "llm"}
    if cfg.llm.use_llm and llm is not None:
        base, st = llm_candidates(transcript, cfg, llm, fingerprint, cancel=cancel, on_progress=on_progress)
        stats.update(st)
    else:
        stats["mode"] = "heuristic"
        base = heuristic_candidates(transcript, audio, cfg, fingerprint, source_duration)
    extra = audio_candidates(transcript, audio, cfg, base, fingerprint, source_duration)
    stats["audio_candidates"] = len(extra)
    merged, n = merge_candidates(base + extra, cfg.selection.max_duration_s * cfg.selection.overflow_tolerance)
    stats["merged"] = n
    stats["candidates"] = len(merged)
    return merged, stats
