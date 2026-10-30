"""Clipper - local AI video clipping.

    python app.py                        # start the review UI on http://127.0.0.1:7860
    python app.py run VIDEO [options]    # headless: analyse a video, optionally export the top clips
    python app.py check                  # hardware / dependency report
"""
from __future__ import annotations

import os

# No telemetry, no update pings, no implicit model downloads: set BEFORE importing gradio / huggingface.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from config import load_config  # noqa: E402
from src.core.exceptions import ClipperError  # noqa: E402
from src.utils.logging_setup import attach_project_log, setup_logging  # noqa: E402

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def cmd_ui(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.host not in LOOPBACK and not args.allow_lan:
        print(f"Refusing to bind to '{args.host}': the UI has no login. Use --allow-lan only on a network you trust.", file=sys.stderr)
        return 2
    setup_logging(cfg.logging.level)
    if args.projects_dir:
        cfg.paths.projects_dir = args.projects_dir
    from src.review.interface import launch
    from src.review.workbench import Workbench

    launch(Workbench(cfg), host=args.host, port=args.port, inbrowser=not args.no_browser)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from src.core.exporter import export_clips, write_summary
    from src.core.job_manager import STAGE_LABELS, JobManager, load_transcript
    from src.core.project_manager import create_or_open_project
    from src.ingestion.video_probe import inspect_video
    from src.rendering.cutting import ClipRenderer
    from src.rendering.ffmpeg_runner import FFmpegRunner
    from src.review import actions

    overrides: dict = {}
    if args.no_llm:
        overrides.setdefault("llm", {})["use_llm"] = False
    if args.model:
        overrides.setdefault("llm", {})["model"] = args.model
    if args.whisper:
        overrides.setdefault("transcription", {})["model"] = args.whisper
    if args.perf:
        overrides.setdefault("performance", {})["mode"] = args.perf
    if args.layout:
        overrides.setdefault("render", {})["mode"] = args.layout
    if args.max_clips:
        overrides.setdefault("selection", {})["max_clips"] = args.max_clips
    cfg = load_config(args.config, overrides)
    setup_logging(cfg.logging.level)
    eff = cfg.effective()
    runner = FFmpegRunner(cfg.paths.ffmpeg_path, cfg.paths.ffprobe_path, low_priority=eff.performance.low_priority, threads=cfg.resolved_threads())
    info = inspect_video(args.video, runner)
    print(info.summary())
    for w in info.warnings:
        print("  warning:", w)
    projects = Path(args.out or cfg.paths.projects_dir).expanduser().resolve()
    store = create_or_open_project(info, projects)
    attach_project_log(store.logs_dir)
    print(f"Project: {store.root}")

    def show(stage, frac, overall, msg):
        print(f"\r[{overall * 100:5.1f}%] {msg[:90]:<90}", end="", flush=True)

    rep = JobManager(store, cfg, runner).run(on_progress=show, force=set(STAGE_LABELS) if args.force else None)
    print()
    if not rep.success:
        print(f"\nStopped at '{rep.failed_stage}': {rep.error}\n  How to fix: {rep.recovery}", file=sys.stderr)
        return 1
    print(f"Done in {rep.seconds:.0f}s (reused from cache: {', '.join(rep.cached) or 'nothing'})")
    cands = [c for c in store.load_candidates() if c.selected]
    for i, c in enumerate(cands, 1):
        print(f"{i:2d}. [{c.total_score:3.0f}] {c.start:7.1f}-{c.end:7.1f}s {c.category:<10} {c.title}")
    if args.export_top and cands:
        top = cands[: args.export_top]
        actions.set_status(store, [c.id for c in top], "approved")
        transcript = load_transcript(store)
        res = export_clips(store, cfg, ClipRenderer(info, store, cfg, runner), transcript, top,
                           on_progress=lambda f, m: print(f"\r[{f * 100:5.1f}%] {m[:90]:<90}", end="", flush=True))
        print()
        for r in res:
            print(("OK   " if r["ok"] else "FAIL ") + (r.get("folder") or r.get("error", "")))
        write_summary(store)
        print(f"Exports: {store.exports_dir}\nSummary: {store.root / 'summary.md'}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from src.utils.hardware_detection import build_report, format_report

    print(format_report(build_report(load_config(args.config).llm.host)))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clipper", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="alternative YAML config (default: configs/default.yaml + configs/local.yaml)")
    sub = ap.add_subparsers(dest="cmd")
    ui = sub.add_parser("ui", help="start the review interface (default)")
    for p in (ap, ui):
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=7860)
        p.add_argument("--no-browser", action="store_true")
        p.add_argument("--allow-lan", action="store_true", help="allow binding to a non-localhost address (no authentication!)")
        p.add_argument("--projects-dir", default=None)
    run = sub.add_parser("run", help="analyse a video headlessly")
    run.add_argument("video")
    run.add_argument("--out", help="projects folder")
    run.add_argument("--no-llm", action="store_true", help="heuristic mode without Ollama")
    run.add_argument("--model", help="Ollama model, e.g. qwen3:4b")
    run.add_argument("--whisper", help="Whisper model size")
    run.add_argument("--perf", choices=["normal", "low"])
    run.add_argument("--layout", choices=["blur", "smart_crop", "original"])
    run.add_argument("--max-clips", type=int)
    run.add_argument("--force", action="store_true", help="ignore caches")
    run.add_argument("--export-top", type=int, metavar="N", help="approve and export the N best clips")
    sub.add_parser("check", help="hardware / dependency report")
    args = ap.parse_args(argv)
    try:
        return {"run": cmd_run, "check": cmd_check}.get(args.cmd, cmd_ui)(args)
    except ClipperError as exc:
        print(exc.user_message(), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Finished stages are saved; run the same command to resume.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
