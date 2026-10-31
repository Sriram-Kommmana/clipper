"""Resumable analysis job: ingest -> transcribe -> audio -> scenes -> candidates -> scoring -> selection.

* Every stage has a cache key built from the source fingerprint, the config it depends on and its upstream
  keys. A stage is skipped only if its stored key still matches, so changing a setting re-runs exactly what
  it affects and nothing else.
* Stage status is persisted before and after each stage; a crash leaves "running", which the next run
  reports as "interrupted" and simply re-runs.
* Failure of a stage keeps everything before it. Cancellation is cooperative (threading.Event).
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from config import AppConfig
from src.analysis.audio_analysis import AudioAnalysis, analyze_audio_file
from src.analysis.frame_extraction import extract_thumbnails, visual_suitability
from src.analysis.scene_detection import SceneAnalysis, detect_scenes
from src.candidates.generator import generate_candidates
from src.candidates.merger import merge_candidates
from src.candidates.scorer import score_candidates
from src.candidates.selector import select_candidates
from src.candidates.validator import Limits, hard_failures, refine_boundaries
from src.core.exceptions import ClipperError, JobCancelled, ResourceExhaustedError
from src.core.models import Candidate, VideoInfo, now_iso
from src.llm.ollama_client import LLMClient, OllamaClient
from src.rendering.ffmpeg_runner import FFmpegRunner
from src.storage.cache import StageCache, make_key
from src.storage.project_store import ProjectStore
from src.transcription.importer import store_import, stored_import
from src.transcription.transcriber import WhisperBackend, extract_audio_wav, transcribe_video, transcript_cache_key
from src.transcription.transcript_models import Transcript

log = logging.getLogger(__name__)

STAGES = ["transcribe", "audio", "scenes", "candidates", "scoring", "selection"]
STAGE_LABELS = {
    "transcribe": "Transcription", "audio": "Audio analysis", "scenes": "Scene analysis",
    "candidates": "Candidate generation", "scoring": "Scoring", "selection": "Selection",
}
STAGE_WEIGHT = {"transcribe": 0.40, "audio": 0.05, "scenes": 0.10, "candidates": 0.22, "scoring": 0.18, "selection": 0.05}
VERSIONS = {"audio": 2, "scenes": 1, "cand": 5, "score": 3}

ProgressCb = Callable[[str, float, float, str], None]   # stage, stage_fraction, overall_fraction, message


@dataclass
class JobReport:
    success: bool
    completed: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    failed_stage: str | None = None
    error: str | None = None
    recovery: str | None = None
    cancelled: bool = False
    seconds: float = 0.0


def load_transcript(store: ProjectStore) -> Transcript | None:
    data = StageCache(store.cache_dir).load_any("transcript")
    return Transcript.model_validate(data) if data else None


def load_audio(store: ProjectStore) -> AudioAnalysis | None:
    data = StageCache(store.cache_dir).load_any("audio")
    return AudioAnalysis.model_validate(data) if data else None


def load_scenes(store: ProjectStore) -> SceneAnalysis | None:
    data = StageCache(store.cache_dir).load_any("scenes")
    return SceneAnalysis.model_validate(data) if data else None


def merge_user_decisions(fresh: list[Candidate], live: list[Candidate]) -> list[Candidate]:
    """Carry review decisions (status, edited bounds/subtitles/title) from the live list onto fresh results.

    Manual candidates and anything the user touched are never dropped by a re-run.
    """
    old = {c.id: c for c in live}
    out: list[Candidate] = []
    seen: set[str] = set()
    for c in fresh:
        prev = old.get(c.id)
        if prev:
            keep = {"status": prev.status, "subtitle_lines": prev.subtitle_lines}
            if prev.title_edited:
                keep.update(title=prev.title, caption=prev.caption, title_edited=True)
            if prev.boundaries_edited:
                keep.update(start=prev.start, end=prev.end, boundaries_edited=True, excerpt=prev.excerpt)
            c = c.model_copy(update=keep)
        out.append(c)
        seen.add(c.id)
    for prev in live:
        if prev.id not in seen and (prev.source == "manual" or prev.status != "pending" or prev.boundaries_edited or prev.title_edited):
            out.append(prev.model_copy(update={"selected": False}))
    return out


class JobManager:
    def __init__(
        self, store: ProjectStore, cfg: AppConfig, runner: FFmpegRunner, *,
        llm: LLMClient | None = None, whisper: WhisperBackend | None = None,
    ):
        self.store, self.cfg, self.runner = store, cfg, runner
        self.eff = cfg.effective()
        self.info: VideoInfo = store.state.source
        self.cache = StageCache(store.cache_dir)
        self._llm = llm
        self._whisper = whisper

    # ---- cache keys ----------------------------------------------------------------------------
    def keys(self) -> dict[str, str]:
        c, fp = self.eff, self.info.fingerprint
        k: dict[str, str] = {"transcribe": transcript_cache_key(self.info, self.cfg, self._imported_transcript())}
        k["audio"] = make_key("audio", VERSIONS["audio"], fp, c.audio.model_dump())
        k["scenes"] = make_key("scenes", VERSIONS["scenes"], fp, c.scenes.model_dump())
        sel = {n: getattr(c.selection, n) for n in ("min_duration_s", "max_duration_s", "target_duration_s", "overflow_tolerance")}
        llm = {n: getattr(c.llm, n) for n in ("model", "use_llm", "num_ctx", "temperature", "seed", "section_seconds", "section_overlap_s", "section_max_words")}
        k["candidates"] = make_key("cand", VERSIONS["cand"], fp, k["transcribe"], k["audio"], llm, sel, c.boundaries.model_dump())
        k["scoring"] = make_key("score", VERSIONS["score"], k["candidates"], c.llm.model, c.llm.use_llm, c.llm.scoring_batch_size, c.scoring.model_dump(), sel)
        k["selection"] = make_key("select", k["scoring"], c.selection.model_dump())
        return k

    # ---- run -----------------------------------------------------------------------------------
    def run(self, stages: list[str] | None = None, *, on_progress: ProgressCb | None = None,
            cancel: threading.Event | None = None, force: set[str] | None = None) -> JobReport:
        cancel = cancel or threading.Event()
        stages = stages or STAGES
        force = force or set()
        report = JobReport(success=False)
        started = time.monotonic()
        store = self.store
        for name, st in store.state.stages.items():          # crashed earlier run?
            if st.status == "running":
                store.update_stage(name, status="interrupted", message="Previous run was interrupted; will re-run.")
        done_weight = 0.0
        try:
            keys = self.keys()
            for name in stages:
                if cancel.is_set():
                    raise JobCancelled("Cancelled before stage '%s'" % name)

                def prog(frac: float, msg: str = "", _n=name, _base=done_weight) -> None:
                    frac = max(0.0, min(1.0, frac))
                    if on_progress:
                        on_progress(_n, frac, _base + STAGE_WEIGHT[_n] * frac, msg or STAGE_LABELS[_n])

                if name not in force and self._is_fresh(name, keys[name]):
                    store.update_stage(name, cached=True, message="Reused from cache", progress=1.0)
                    report.cached.append(name)
                    done_weight += STAGE_WEIGHT[name]
                    prog(1.0, "Reused from cache")
                    continue
                if name == "scenes" and not self.eff.scenes.enabled:
                    store.update_stage(name, status="skipped", message="Disabled in settings")
                    done_weight += STAGE_WEIGHT[name]
                    continue
                t0 = time.monotonic()
                store.update_stage(name, status="running", started_at=now_iso(), finished_at=None, error=None, recovery=None,
                                   cache_key=None, cached=False, progress=0.0, message="Starting")
                self._run_stage(name, keys, prog, cancel)
                details = store.stage(name).message            # keep what the stage reported, e.g. language and confidence
                store.update_stage(name, status="done", finished_at=now_iso(), duration_s=round(time.monotonic() - t0, 2),
                                   cache_key=keys[name], progress=1.0, message=details if details != "Starting" else "Done")
                report.completed.append(name)
                done_weight += STAGE_WEIGHT[name]
                prog(1.0, "Done")
            report.success = True
            self._cleanup_work()
        except JobCancelled as exc:
            report.cancelled, report.error, report.recovery = True, str(exc), exc.recovery
            self._mark_failed(stages, "cancelled", str(exc), exc.recovery, report)
        except ClipperError as exc:
            report.error, report.recovery = str(exc), exc.recovery
            self._mark_failed(stages, "failed", str(exc), exc.recovery, report)
            store.add_error(f"[{report.failed_stage}] {exc} | {exc.details or ''}"[:1500])
            log.error("Stage %s failed: %s", report.failed_stage, exc)
        except MemoryError as exc:
            err = ResourceExhaustedError("The system ran out of RAM.")
            report.error, report.recovery = str(err), err.recovery
            self._mark_failed(stages, "failed", str(err), err.recovery, report)
            store.add_error(f"[{report.failed_stage}] MemoryError {exc}")
        except Exception as exc:  # unexpected: keep the traceback in the log, show a clear message
            log.error("Unexpected error:\n%s", traceback.format_exc())
            report.error = f"Unexpected error: {type(exc).__name__}: {exc}"
            report.recovery = "See logs/clipper.log in the project folder; progress so far is saved and Resume will continue."
            self._mark_failed(stages, "failed", report.error, report.recovery, report)
            store.add_error(f"[{report.failed_stage}] {report.error}")
        finally:
            self._unload_models()
            report.seconds = time.monotonic() - started
            store.mutate(lambda s: setattr(s, "last_run_total_s", round(report.seconds, 1)))
        return report

    def _mark_failed(self, stages: list[str], status: str, error: str, recovery: str | None, report: JobReport) -> None:
        for name in stages:
            st = self.store.stage(name)
            if st.status == "running":
                self.store.update_stage(name, status=status, error=error, recovery=recovery, message=("Cancelled" if status == "cancelled" else "Failed"))
                report.failed_stage = name

    def _is_fresh(self, name: str, key: str) -> bool:
        st = self.store.stage(name)
        if st.status not in ("done",) or st.cache_key != key:
            return False
        if name == "selection":
            return self.store.candidates_path.exists()
        cache_name = {"transcribe": "transcript", "candidates": "candidates_raw", "scoring": "candidates_scored"}.get(name, name)
        return self.cache.load(cache_name, key) is not None

    def _cleanup_work(self) -> None:
        for f in list(self.store.work_dir.glob("*")):
            try:
                f.unlink()
            except OSError:
                pass

    def _unload_models(self) -> None:
        if self._llm is not None:
            try:
                self._llm.unload()
            except Exception:
                pass

    # ---- stages --------------------------------------------------------------------------------
    def _run_stage(self, name: str, keys: dict[str, str], prog, cancel: threading.Event) -> None:
        getattr(self, f"_stage_{name}")(keys[name], prog, cancel)

    def _wav(self, cancel) -> Path:
        wav = self.store.work_dir / "audio_16k.wav"
        if not wav.exists():
            extract_audio_wav(self.info, wav, self.runner, cancel=cancel)
        return wav

    def _stage_transcribe(self, key, prog, cancel) -> None:
        t, from_cache = transcribe_video(self.info, self.store, self.cfg, self.runner, backend=self._whisper, on_progress=prog, cancel=cancel,
                                         imported=self._imported_transcript())
        confidence = t.mean_word_confidence()
        quality = "imported transcript" if t.model == "imported" else f"word confidence {confidence:.0%}" if confidence is not None else "no confidence data"
        self.store.update_stage("transcribe", device=f"{t.device}/{t.compute_type}" if t.device else None,
                                message=f"{len(t.segments)} segments, language {t.language}, {quality}")

    def _imported_transcript(self) -> Path | None:
        """A transcript given for this run is kept in the project, so later runs reuse it instead of running Whisper."""
        given = self.cfg.transcription.transcript_file
        return store_import(given, self.store.root) if given else stored_import(self.store.root)

    def _stage_audio(self, key, prog, cancel) -> None:
        if not self.info.has_audio:
            self.cache.save("audio", key, AudioAnalysis(window_s=self.eff.audio.window_s, has_audio=False, warnings=["No audio stream"]).model_dump(mode="json"))
            return
        wav = self._wav(cancel)
        a = analyze_audio_file(wav, self.eff.audio, cancel=cancel, on_progress=lambda f: prog(f, "Analysing audio"))
        for w in a.warnings:
            self.store.add_warning("Audio: " + w)
        self.cache.save("audio", key, a.model_dump(mode="json"))

    def _stage_scenes(self, key, prog, cancel) -> None:
        sa = detect_scenes(self.info, self.runner, self.eff.scenes, cancel=cancel, on_progress=lambda f: prog(0.9 * f, "Detecting scene changes"))
        frames_dir = self.store.cache_dir / "frames"
        try:
            sa.thumbnails = extract_thumbnails(Path(self.info.path), sa.scenes, frames_dir, self.eff.scenes.max_thumbnails)
        except Exception as exc:  # thumbnails are optional
            log.warning("Thumbnail extraction failed: %s", exc)
        prog(1.0, f"{len(sa.cuts)} scene changes")
        self.cache.save("scenes", key, sa.model_dump(mode="json"))

    def get_llm(self) -> LLMClient | None:
        if not self.eff.llm.use_llm:
            return None
        if self._llm is None:
            self._llm = OllamaClient(self.eff.llm)
        return self._llm

    def _ensure_llm_ready(self) -> LLMClient | None:
        llm = self.get_llm()
        if llm is not None and hasattr(llm, "ensure_ready"):
            llm.ensure_ready()  # raises LLMUnavailable/LLMModelMissing with install instructions
        return llm

    def _stage_candidates(self, key, prog, cancel) -> None:
        transcript, audio = load_transcript(self.store), load_audio(self.store)
        scenes = load_scenes(self.store)
        assert transcript is not None
        llm = self._ensure_llm_ready()
        c = self.eff
        cands, stats = generate_candidates(transcript, audio, c, llm, self.info.fingerprint, self.info.duration, cancel=cancel,
                                           on_progress=lambda f, m: prog(0.9 * f, m), title=Path(self.info.filename).stem.replace("_", " "))
        limits = Limits.from_config(c.selection)
        refined = [refine_boundaries(x, transcript, c.boundaries, limits, self.info.duration) for x in cands]
        refined, n = merge_candidates(refined, limits.hard_max)
        stats["merged"] = stats.get("merged", 0) + n
        final: list[Candidate] = []
        used: set[str] = set()
        for x in refined:
            cid = x.id
            while cid in used:
                cid += "x"
            used.add(cid)
            warnings = list(x.warnings)
            if scenes and scenes.cut_rate(x.start, x.end) > 0.3:
                warnings.append("Frequent scene cuts - the blurred-background layout is safer than smart crop here.")
            why = hard_failures(x, transcript, audio, self.info.duration, limits)
            final.append(x.model_copy(update={"id": cid, "warnings": warnings, "filter_reason": ("invalid: " + "; ".join(why)) if why else None}))
        stats["invalid"] = sum(1 for x in final if x.filter_reason)
        self.cache.save("candidates_raw", key, {"candidates": [x.model_dump(mode="json") for x in final], "stats": stats})
        prog(1.0, f"{len(final)} candidates")

    def _stage_scoring(self, key, prog, cancel) -> None:
        raw = self.cache.load_any("candidates_raw")
        transcript, audio = load_transcript(self.store), load_audio(self.store)
        assert raw and transcript
        cands = [Candidate.model_validate(x) for x in raw["candidates"]]
        valid = [x for x in cands if not x.filter_reason]
        llm = self._ensure_llm_ready()
        video = Path(self.info.path)
        scored = score_candidates(valid, transcript, audio, self.eff, llm, cancel=cancel,
                                  visual_fn=lambda c: visual_suitability(video, c.start, c.end), on_progress=prog,
                                  context=raw.get("stats", {}).get("video_brief", ""))
        out = scored + [x for x in cands if x.filter_reason]
        llm_scored = sum(1 for x in scored if x.scores and x.scores.llm_scored)
        if scored and llm is not None and llm_scored < len(scored):
            self.store.add_warning(f"{len(scored) - llm_scored} of {len(scored)} candidates could not be scored by the language model; neutral values were used.")
        self.cache.save("candidates_scored", key, {"candidates": [x.model_dump(mode="json") for x in out]})

    def _stage_selection(self, key, prog, cancel) -> None:
        data = self.cache.load_any("candidates_scored")
        assert data
        fresh = [Candidate.model_validate(x) for x in data["candidates"]]
        merged = merge_user_decisions(fresh, self.store.load_candidates())
        self.store.save_candidates(select_candidates(merged, self.eff.selection))
