"""Stage K: export approved clips (MP4 + SRT + metadata + caption) and the project summary."""
from __future__ import annotations

import json
import re
import shutil
import threading
from typing import Callable

from config import AppConfig
from src.core.exceptions import ClipperError
from src.core.models import Candidate, ProjectState
from src.core.project_manager import slugify
from src.rendering import subtitles as subs
from src.rendering.cutting import ClipRenderer
from src.storage.project_store import ProjectStore, atomic_write_json, atomic_write_text
from src.transcription.transcript_models import Transcript

ProgressFn = Callable[[float, str], None]


def metadata_for(c: Candidate, state: ProjectState) -> dict:
    return {
        "id": c.id, "title": c.title, "caption": c.caption, "hook": c.hook, "category": c.category,
        "start": round(c.start, 3), "end": round(c.end, 3), "duration": round(c.duration, 2),
        "explanation": c.explanation, "excerpt": c.excerpt, "source": c.source,
        "scores": c.scores.model_dump() if c.scores else None, "warnings": c.warnings,
        "source_video": state.source.filename, "note": "Scores are ranking heuristics, not predictions of performance.",
    }


def export_clips(
    store: ProjectStore, cfg: AppConfig, renderer: ClipRenderer, transcript: Transcript, candidates: list[Candidate],
    *, on_progress: ProgressFn | None = None, cancel: threading.Event | None = None,
) -> list[dict]:
    """Render (or reuse) final-quality clips and write the export folder for each. Returns per-clip results."""
    results: list[dict] = []
    n = len(candidates)
    for i, c in enumerate(candidates):
        if cancel is not None and cancel.is_set():
            break
        cb = (lambda f, _i=i: on_progress((_i + f) / max(1, n), f"Exporting {_i + 1}/{n}: {c.title[:40]}")) if on_progress else None
        folder = store.exports_dir / f"{c.id}_{slugify(c.title, 30)}"
        try:
            res = renderer.render(c, transcript, preview=False, cancel=cancel, on_progress=cb)
            folder.mkdir(parents=True, exist_ok=True)
            base = slugify(c.title, 50)
            mp4 = folder / f"{base}.mp4"
            tmp = mp4.with_suffix(".part")
            shutil.copyfile(res.path, tmp)
            tmp.replace(mp4)
            files = [mp4.name]
            lines = renderer.subtitle_lines(c, transcript)
            if cfg.export.write_srt and lines:
                atomic_write_text(folder / f"{base}.srt", subs.build_srt(lines, c.start, c.end))
                files.append(f"{base}.srt")
            if cfg.export.write_metadata:
                atomic_write_json(folder / f"{base}.json", metadata_for(c, store.state))
                files.append(f"{base}.json")
            desc = (c.caption or c.explanation or "").strip()
            atomic_write_text(folder / "description.txt", f"{c.title}\n\n{desc}\n")
            files.append("description.txt")
            results.append({"id": c.id, "ok": True, "folder": str(folder), "files": files, "notes": res.notes, "cached": res.cached})
        except ClipperError as exc:
            results.append({"id": c.id, "ok": False, "error": str(exc), "details": exc.details})
        if on_progress:
            on_progress((i + 1) / max(1, n), f"Exported {i + 1}/{n}")
    return results


def build_summary(store: ProjectStore) -> dict:
    s = store.state
    cands = store.load_candidates()
    st = lambda n: (s.stages[n].duration_s or 0.0) if n in s.stages else 0.0
    gen = st("candidates") + st("scoring") + st("selection")
    analysis_total = sum((x.duration_s or 0.0) for x in s.stages.values())
    return {
        "source": {"file": s.source.filename, "duration_s": round(s.source.duration, 1), "resolution": f"{s.source.width}x{s.source.height}",
                   "fps": round(s.source.fps, 2), "has_audio": s.source.has_audio},
        "timing_s": {"transcription": round(st("transcribe"), 1), "audio_and_scene_analysis": round(st("audio") + st("scenes"), 1),
                     "candidate_generation_and_scoring": round(gen, 1), "rendering": round(s.render_seconds, 1),
                     "total_processing": round(analysis_total + s.render_seconds, 1)},
        "candidates": {"total": len(cands), "selected": sum(c.selected for c in cands),
                       "approved": sum(c.status == "approved" for c in cands), "rejected": sum(c.status == "rejected" for c in cands),
                       "filtered": sum(1 for c in cands if c.filter_reason and not c.selected)},
        "stages": {n: {"status": x.status, "seconds": x.duration_s, "cached": x.cached, "device": x.device} for n, x in s.stages.items()},
        "warnings": s.warnings, "errors": s.errors[-10:],
        "disclaimer": "Scores are ranking heuristics for ordering your review queue, not predictions of views or engagement.",
    }


def summary_markdown(sm: dict) -> str:
    t, c = sm["timing_s"], sm["candidates"]
    lines = [
        f"# Project summary - {sm['source']['file']}", "",
        f"* Source: {sm['source']['duration_s']} s, {sm['source']['resolution']} @ {sm['source']['fps']} fps",
        f"* Transcription: {t['transcription']} s; audio+scene analysis: {t['audio_and_scene_analysis']} s; candidate generation+scoring: {t['candidate_generation_and_scoring']} s",
        f"* Rendering: {t['rendering']} s; **total processing: {t['total_processing']} s**",
        f"* Candidates: {c['total']} found, {c['selected']} selected, {c['approved']} approved, {c['rejected']} rejected, {c['filtered']} filtered out",
        "", "## Warnings", *(f"* {w}" for w in sm["warnings"] or ["none"]),
        "", "## Errors", *("* " + _oneline(e) for e in sm["errors"] or ["none"]),
        "", f"_{sm['disclaimer']}_", "",
    ]
    return "\n".join(lines)


def _oneline(text: str) -> str:
    return re.sub(r"\s+", " ", text)[:300]


def write_summary(store: ProjectStore) -> dict:
    sm = build_summary(store)
    atomic_write_text(store.root / "summary.json", json.dumps(sm, indent=2, ensure_ascii=False))
    atomic_write_text(store.root / "summary.md", summary_markdown(sm))
    return sm
