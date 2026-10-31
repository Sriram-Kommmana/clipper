"""UI-independent application controller used by the Gradio interface (and unit-tested without it)."""
from __future__ import annotations

import html
import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from config import AppConfig, load_config
from src.core.exceptions import ClipperError, FFmpegNotFoundError, ProjectError
from src.core.exporter import export_clips, write_summary
from src.core.job_manager import STAGE_LABELS, STAGES, JobManager, JobReport, load_audio, load_transcript
from src.core.models import Candidate, VideoInfo
from src.core.project_manager import create_or_open_project, list_projects, open_project
from src.ingestion.video_probe import inspect_video
from src.rendering.cutting import ClipRenderer
from src.rendering.ffmpeg_runner import FFmpegRunner
from src.review import actions
from src.storage.project_store import ProjectStore
from src.utils.logging_setup import attach_project_log, detach_handler, recent_log_lines

log = logging.getLogger(__name__)

STATUS_ICON = {"pending": "⏳ pending", "approved": "✅ approved", "rejected": "❌ rejected"}
STAGE_ICON = {"pending": "·", "running": "⏳", "done": "✅", "failed": "❌", "cancelled": "⏹", "interrupted": "⚠", "skipped": "—"}
TABLE_HEADERS = ["#", "ID", "Title", "Type", "Score", "Len", "Review", "Note"]
IDLE_PROGRESS = {"running": False, "stage": "", "overall": 0.0, "message": "Idle", "report": None}


class Busy(ClipperError):
    default_recovery = "Wait for the current job to finish, or press Cancel."


class Workbench:
    def __init__(self, base_cfg: AppConfig | None = None, preview_dir: Path | None = None):
        self.base_cfg = base_cfg or load_config()
        self.cfg = self.base_cfg
        self.store: ProjectStore | None = None
        self.info: VideoInfo | None = None
        self._runner: FFmpegRunner | None = None
        self._runner_key: tuple | None = None
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.RLock()
        self._log_handler: logging.Handler | None = None
        self.preview_dir = preview_dir or Path(tempfile.gettempdir()) / "clipper_ui"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.progress: dict[str, Any] = dict(IDLE_PROGRESS)
        self._tcache: tuple[str, Any, Any] | None = None   # (project id, transcript, audio)

    # ---- config / tools ------------------------------------------------------------------------
    def set_config(self, overrides: dict[str, Any]) -> AppConfig:
        self.cfg = self.base_cfg.with_overrides(overrides)
        return self.cfg

    def runner(self) -> FFmpegRunner:
        eff = self.cfg.effective()
        key = (self.cfg.paths.ffmpeg_path, self.cfg.paths.ffprobe_path, eff.performance.low_priority, self.cfg.resolved_threads())
        if self._runner is None or key != self._runner_key:
            self._runner = FFmpegRunner(self.cfg.paths.ffmpeg_path, self.cfg.paths.ffprobe_path, low_priority=eff.performance.low_priority,
                                        threads=self.cfg.resolved_threads())
            self._runner_key = key
        return self._runner

    def tools_ok(self) -> tuple[bool, str]:
        try:
            self.runner()
            return True, "FFmpeg found."
        except FFmpegNotFoundError as exc:
            return False, exc.user_message()

    # ---- projects ------------------------------------------------------------------------------
    def inspect(self, path: str) -> VideoInfo:
        return inspect_video(path, self.runner())

    def open_video(self, path: str, projects_dir: str | None = None) -> tuple[ProjectStore, VideoInfo]:
        self._require_idle()
        info = self.inspect(path)
        root = Path(projects_dir or self.cfg.paths.projects_dir).expanduser()
        if not root.is_absolute():
            root = (Path(__file__).resolve().parents[2] / root)
        root.mkdir(parents=True, exist_ok=True)
        self._switch(create_or_open_project(info, root))
        self.info = info
        return self.store, info  # type: ignore[return-value]

    def open_existing(self, folder: str) -> ProjectStore:
        self._require_idle()
        store = open_project(Path(folder))
        self._switch(store)
        self.info = store.state.source
        return store

    def _switch(self, store: ProjectStore) -> None:
        if self._log_handler:
            detach_handler(self._log_handler)
        self.store = store
        self._log_handler = attach_project_log(store.logs_dir)
        self._tcache = None
        self.progress = dict(IDLE_PROGRESS)

    def project_choices(self, projects_dir: str | None = None) -> list[tuple[str, str]]:
        root = Path(projects_dir or self.cfg.paths.projects_dir).expanduser()
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[2] / root
        return [(n, str(p)) for n, p in list_projects(root)]

    def need_store(self) -> ProjectStore:
        if self.store is None:
            raise ProjectError("No project is open.", recovery="Import a video (or open a project) on the first tab.")
        return self.store

    # ---- job control ---------------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _require_idle(self) -> None:
        if self.running:
            raise Busy("A job is already running.")

    def start_job(self, *, force_all: bool = False) -> None:
        with self._lock:
            self._require_idle()
            store = self.need_store()
            self._cancel = threading.Event()
            self.progress = {"running": True, "stage": "", "overall": 0.0, "message": "Starting…", "report": None}
            job = JobManager(store, self.cfg, self.runner())
            force = set(STAGES) if force_all else set()

            def cb(stage: str, frac: float, overall: float, msg: str) -> None:
                self.progress.update(stage=stage, overall=overall, message=f"{STAGE_LABELS.get(stage, stage)}: {msg}")

            def work() -> None:
                rep: JobReport = job.run(on_progress=cb, cancel=self._cancel, force=force)
                self._tcache = None
                self.progress.update(running=False, report=rep, overall=1.0 if rep.success else self.progress["overall"],
                                     message="Finished" if rep.success else ("Cancelled" if rep.cancelled else f"Stopped: {rep.error}"))

            self._thread = threading.Thread(target=work, name="clipper-job", daemon=True)
            self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float = 600) -> None:
        if self._thread:
            self._thread.join(timeout)

    # ---- status views --------------------------------------------------------------------------
    def stage_rows(self) -> list[list[str]]:
        if not self.store:
            return [[STAGE_LABELS[s], "·", "", ""] for s in STAGES]
        rows = []
        for s in STAGES:
            st = self.store.state.stages.get(s)
            if not st:
                rows.append([STAGE_LABELS[s], "·  waiting", "", ""])
                continue
            detail = st.message
            if st.status in ("failed", "cancelled") and st.error:
                detail = st.error.splitlines()[0][:140]
            if st.device:
                detail += f"  [{st.device}]"
            secs = "cached" if st.cached and st.status == "done" else (f"{st.duration_s:.1f}s" if st.duration_s is not None else "")
            rows.append([STAGE_LABELS[s], f"{STAGE_ICON.get(st.status, '')}  {st.status}", secs, detail])
        return rows

    def progress_html(self) -> str:
        p = self.progress
        pct = int(100 * p["overall"])
        rep: JobReport | None = p.get("report")
        extra = ""
        if rep and not rep.success and not p["running"]:
            extra = f"<div class='clp-err'><b>{html.escape(rep.error or '')}</b><br>{html.escape(rep.recovery or '')}</div>"
        elif rep and rep.success and not p["running"]:
            extra = f"<div class='clp-ok'>Done in {rep.seconds:.0f}s. Open the <b>Review</b> tab.</div>"
        return (f"<div class='clp-prog'><div class='clp-bar'><div style='width:{pct}%'></div></div>"
                f"<div class='clp-prog-label'>{pct}% · {html.escape(p['message'])}</div>{extra}</div>")

    def log_text(self, n: int = 30) -> str:
        return "\n".join(recent_log_lines(n))

    # ---- candidates ----------------------------------------------------------------------------
    def candidates(self, view: str = "Selected") -> list[Candidate]:
        cs = self.need_store().load_candidates()
        if view == "Selected":
            cs = [c for c in cs if c.selected or c.status == "approved"]
        elif view == "Approved":
            cs = [c for c in cs if c.status == "approved"]
        elif view == "Rejected":
            cs = [c for c in cs if c.status == "rejected"]
        elif view == "Pending":
            cs = [c for c in cs if c.status == "pending" and c.selected]
        return sorted(cs, key=lambda c: (not c.selected, -c.total_score))

    def table_rows(self, view: str = "Selected") -> list[list[Any]]:
        out = []
        for i, c in enumerate(self.candidates(view), 1):
            out.append([i, c.id, c.title, c.category, round(c.total_score), f"{c.duration:.0f}s", STATUS_ICON[c.status],
                        "" if c.selected else (c.filter_reason or "")[:60]])
        return out

    def _data(self):
        store = self.need_store()
        pid = store.state.project_id
        if self._tcache is None or self._tcache[0] != pid:
            self._tcache = (pid, load_transcript(store), load_audio(store))
        if self._tcache[1] is None:
            raise ProjectError("This project has no transcript yet.", recovery="Run the analysis on the first tab.")
        return self._tcache[1], self._tcache[2]

    def detail(self, cid: str) -> dict[str, Any]:
        store = self.need_store()
        c = store.get_candidate(cid)
        transcript, _ = self._data()
        return {
            "candidate": c, "subtitles": actions.subtitle_editor_text(c, transcript, self.cfg),
            "score_html": score_html(c), "info_md": info_markdown(c),
        }

    def renderer(self) -> ClipRenderer:
        store = self.need_store()
        return ClipRenderer(store.state.source, store, self.cfg, self.runner())

    def preview(self, cid: str) -> tuple[str, str]:
        """Render (or reuse) a low-res preview; returns (path served to the browser, note)."""
        self._require_idle()
        store = self.need_store()
        c = store.get_candidate(cid)
        transcript, _ = self._data()
        res = self.renderer().render(c, transcript, preview=True)
        served = self.preview_dir / res.path.name
        if not served.exists():
            shutil.copyfile(res.path, served)
        for old in self.preview_dir.glob(f"{c.id}_preview_*.mp4"):
            if old.name != served.name:
                old.unlink(missing_ok=True)
        note = "reused cached preview" if res.cached else f"rendered in {res.seconds:.1f}s ({res.encoder})"
        return str(served), note + ("; " + "; ".join(res.notes) if res.notes else "")

    def approve(self, ids: list[str], status: str) -> int:
        return actions.set_status(self.need_store(), ids, status)

    def bulk_approve(self, min_score: float) -> int:
        return len(actions.bulk_approve(self.need_store(), min_score))

    def apply_bounds(self, cid: str, start: float, end: float):
        transcript, audio = self._data()
        return actions.update_bounds(self.need_store(), cid, start, end, transcript, audio, self.cfg)

    def apply_subtitles(self, cid: str, text: str):
        transcript, _ = self._data()
        return actions.update_subtitles(self.need_store(), cid, text, transcript, self.cfg)

    def export(self, ids: list[str] | None = None, on_progress=None) -> list[dict]:
        self._require_idle()
        store = self.need_store()
        transcript, _ = self._data()
        chosen = [c for c in store.load_candidates() if c.status == "approved" and (ids is None or c.id in ids)]
        if not chosen:
            raise ProjectError("No approved clips to export.", recovery="Approve at least one clip in the Review tab.")
        results = export_clips(store, self.cfg, self.renderer(), transcript, chosen, on_progress=on_progress)
        write_summary(store)
        return results


# ---- presentation helpers (pure; unit-tested) ---------------------------------------------------
SCORE_LABELS = [("hook", "Hook"), ("clarity", "Clarity"), ("completeness", "Completeness"), ("payoff", "Payoff"),
                ("escalation", "Escalation"), ("format_fit", "Length fit"), ("av_suitability", "Audio/visual")]


def score_html(c: Candidate) -> str:
    if not c.scores:
        return "<i>Not scored.</i>"
    rows = []
    for key, label in SCORE_LABELS:
        v = getattr(c.scores, key)
        rows.append(f"<div class='clp-row'><span>{label}</span><div class='clp-bar sm'><div style='width:{v:.0f}%'></div></div><b>{v:.0f}</b></div>")
    tag = "language model + checks" if c.scores.llm_scored else "checks only (language model unavailable)"
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in c.scores.notes)
    return (f"<div class='clp-scores'><div class='clp-total'>{c.scores.total:.0f}<small>/100</small></div>{''.join(rows)}"
            f"<div class='clp-note'>Scored by {tag}. A ranking heuristic, not a prediction of views.</div>"
            f"<ul class='clp-notes'>{notes}</ul></div>")


def info_markdown(c: Candidate) -> str:
    lines = [f"**{c.category.title()}** · {c.duration:.0f}s · {c.start:.1f}s → {c.end:.1f}s · source: {c.source}",
             f"**Why:** {c.explanation or '—'}", f"**Hook:** “{c.hook}”" if c.hook else "", f"**Caption idea:** {c.caption}" if c.caption else "",
             f"**Context completeness:** {c.context_completeness:.0f}% · **Payoff:** {c.payoff_estimate:.0f}% · **Model confidence:** {c.confidence:.0f}%",
             f"> {c.excerpt[:600]}{'…' if len(c.excerpt) > 600 else ''}"]
    if c.warnings:
        lines.append("**Warnings**\n" + "\n".join(f"- ⚠ {w}" for w in c.warnings))
    if c.merged_from:
        lines.append(f"_Merged from {len(c.merged_from)} overlapping candidate(s)._")
    return "\n\n".join(x for x in lines if x)
