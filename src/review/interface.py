"""Gradio review interface (bound to localhost by default). All logic lives in Workbench / actions."""
from __future__ import annotations

import functools
import logging
import traceback
from pathlib import Path
from typing import Any, Callable

import gradio as gr

from src.core.exceptions import CandidateValidationError, ClipperError
from src.core.exporter import build_summary, summary_markdown
from src.review.workbench import STATUS_ICON, TABLE_HEADERS, Workbench
from src.utils.hardware_detection import build_report, format_report

log = logging.getLogger(__name__)

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2"]
LLM_MODELS = ["qwen3:4b", "qwen3:1.7b", "qwen3:8b"]
RES_PRESETS = {"1080 × 1920 (full quality)": (1080, 1920), "720 × 1280 (faster)": (720, 1280)}

CSS = """
.clp-hero{padding:14px 18px;border-radius:14px;background:linear-gradient(135deg,#4338ca,#7c3aed);color:#fff;margin-bottom:6px}
.clp-hero h1{margin:0;font-size:1.5rem;color:#fff}.clp-hero p{margin:2px 0 0;opacity:.9}
.clp-prog{margin:6px 0}.clp-bar{height:12px;border-radius:8px;background:rgba(127,127,127,.25);overflow:hidden}
.clp-bar>div{height:100%;background:linear-gradient(90deg,#6366f1,#a855f7);transition:width .4s}
.clp-bar.sm{height:8px;flex:1}.clp-prog-label{margin-top:4px;font-size:.9rem;opacity:.85}
.clp-err{margin-top:8px;padding:10px;border-radius:10px;background:rgba(220,38,38,.12);border:1px solid rgba(220,38,38,.5)}
.clp-ok{margin-top:8px;padding:10px;border-radius:10px;background:rgba(22,163,74,.12);border:1px solid rgba(22,163,74,.5)}
.clp-scores .clp-row{display:flex;align-items:center;gap:10px;margin:4px 0}.clp-scores .clp-row span{width:110px;font-size:.88rem}
.clp-scores .clp-row b{width:30px;text-align:right}.clp-total{font-size:2.2rem;font-weight:700;line-height:1}.clp-total small{font-size:1rem;opacity:.6}
.clp-note{font-size:.8rem;opacity:.7;margin-top:6px}.clp-notes{font-size:.8rem;opacity:.8;margin:4px 0 0 16px}
"""


def guard(fn: Callable) -> Callable:
    """Show friendly toasts for expected errors; log + show a clear message for unexpected ones."""
    @functools.wraps(fn)
    def inner(*a, **k):
        try:
            return fn(*a, **k)
        except gr.Error:
            raise
        except ClipperError as exc:
            raise gr.Error(exc.user_message(), duration=12) from exc
        except Exception as exc:
            log.error("UI handler %s failed:\n%s", fn.__name__, traceback.format_exc())
            raise gr.Error(f"Unexpected error: {type(exc).__name__}: {exc} (details are in the log tab)", duration=12) from exc
    return inner


def collect_overrides(v: dict[str, Any]) -> dict[str, Any]:
    """Map UI control values to config overrides (pure; unit-tested)."""
    w, h = RES_PRESETS.get(v["resolution"], (1080, 1920))
    pw, ph = (540, 960) if w == 1080 else (360, 640)
    return {
        "transcription": {"model": v["whisper"], "device": v["device"], "language": (v["language"] or "").strip() or None},
        "llm": {"model": (v["llm_model"] or "qwen3:4b").strip(), "use_llm": bool(v["use_llm"])},
        "performance": {"mode": v["perf"]},
        "selection": {"min_duration_s": v["min_d"], "target_duration_s": v["target_d"], "max_duration_s": v["max_d"],
                      "max_clips": int(v["max_clips"]), "min_score": v["min_score"], "min_separation_s": v["separation"]},
        "render": {"mode": v["layout"], "width": w, "height": h, "preview_width": pw, "preview_height": ph, "hardware_encoding": v["hw"]},
        "subtitles": {"enabled": bool(v["subs_on"]), "mode": v["subs_mode"], "font_size": int(v["font_size"]),
                      "highlight_color": v["hi_color"][:7] if v["hi_color"] and v["hi_color"].startswith("#") else "#FFE135", "position": v["sub_pos"]},
    }


def info_card(info) -> str:
    warn = "".join(f"\n- ⚠ {w}" for w in info.warnings)
    return (f"**{info.filename}** — {info.width}×{info.height} @ {info.fps:.2f} fps · {info.duration / 60:.1f} min · "
            f"{info.video_codec} · audio: {('%s %dch' % (info.audio_codec, info.audio_channels)) if info.has_audio else '**none**'} · "
            f"{info.size_bytes / 1e6:.0f} MB{warn}")


def build_ui(wb: Workbench) -> gr.Blocks:
    with gr.Blocks(title="Clipper", analytics_enabled=False, fill_width=True) as demo:
        gr.HTML("<div class='clp-hero'><h1>🎬 Clipper</h1><p>Local AI clip maker — transcription, moment finding and rendering all run on this computer. "
                "No cloud AI, no uploads. Only use footage you are authorised to edit and republish.</p></div>")
        selected_id = gr.State("")

        with gr.Tabs():
            # ------------------------------------------------------------ 1. import & process
            with gr.Tab("1 · Import & process", id="import"):
                with gr.Row():
                    with gr.Column(scale=3):
                        path_tb = gr.Textbox(label="Video file (path on this computer)", placeholder=r"C:\Videos\podcast.mp4",
                                             info="MP4, MOV or MKV. Tip: shift+right-click the file in Explorer → Copy as path.")
                        upload = gr.File(label="…or pick a file", file_count="single", type="filepath", file_types=[".mp4", ".mov", ".mkv"], height=90)
                        out_tb = gr.Textbox(label="Output folder (projects are stored here)", value=str(wb.cfg.paths.projects_dir))
                        with gr.Row():
                            inspect_btn = gr.Button("🔍 Inspect & open project", variant="primary")
                            proj_dd = gr.Dropdown(label="…or reopen a project", choices=wb.project_choices(), value=None, scale=2)
                            reopen_btn = gr.Button("Open")
                        info_md = gr.Markdown("No project open.")
                    with gr.Column(scale=2):
                        with gr.Accordion("Settings", open=True):
                            with gr.Row():
                                whisper_dd = gr.Dropdown(WHISPER_MODELS, value=wb.cfg.transcription.model, label="Whisper model", info="small fits 6 GB VRAM")
                                device_dd = gr.Dropdown(["auto", "cuda", "cpu"], value=wb.cfg.transcription.device, label="Device")
                            with gr.Row():
                                llm_dd = gr.Dropdown(LLM_MODELS, value=wb.cfg.llm.model, allow_custom_value=True, label="Ollama model")
                                use_llm_cb = gr.Checkbox(value=wb.cfg.llm.use_llm, label="Use local LLM", info="Off = rough heuristic mode")
                            lang_tb = gr.Textbox(label="Spoken language (blank = auto-detect)", value=wb.cfg.transcription.language or "", max_lines=1)
                            perf_rd = gr.Radio(["normal", "low"], value=wb.cfg.performance.mode, label="Performance mode", info="low = fewer threads, smaller contexts, lower priority (slower)")
                            with gr.Row():
                                min_d = gr.Slider(5, 60, value=wb.cfg.selection.min_duration_s, step=1, label="Min length (s)")
                                target_d = gr.Slider(10, 90, value=wb.cfg.selection.target_duration_s, step=1, label="Target (s)")
                                max_d = gr.Slider(20, 120, value=wb.cfg.selection.max_duration_s, step=1, label="Max (s)")
                            with gr.Row():
                                max_clips = gr.Slider(1, 30, value=wb.cfg.selection.max_clips, step=1, label="Max clips")
                                min_score = gr.Slider(0, 90, value=wb.cfg.selection.min_score, step=1, label="Min score")
                                separation = gr.Slider(0, 120, value=wb.cfg.selection.min_separation_s, step=1, label="Min gap (s)")
                            with gr.Row():
                                layout_dd = gr.Dropdown([("Blurred background (9:16)", "blur"), ("Smart crop (follow face)", "smart_crop"), ("Original framing", "original")],
                                                        value=wb.cfg.render.mode, label="Layout")
                                res_dd = gr.Dropdown(list(RES_PRESETS), value="1080 × 1920 (full quality)", label="Output size")
                                hw_dd = gr.Dropdown(["auto", "never"], value=wb.cfg.render.hardware_encoding, label="GPU encoding", info="auto = only if it works")
                            with gr.Row():
                                subs_on = gr.Checkbox(value=wb.cfg.subtitles.enabled, label="Captions")
                                subs_mode = gr.Dropdown([("Animated word highlight", "karaoke"), ("Plain captions", "plain")], value=wb.cfg.subtitles.mode, label="Caption style")
                                sub_pos = gr.Dropdown(["bottom", "middle", "top"], value=wb.cfg.subtitles.position, label="Position")
                            with gr.Row():
                                font_size = gr.Slider(30, 140, value=wb.cfg.subtitles.font_size, step=2, label="Font size (at 1080 px)")
                                hi_color = gr.ColorPicker(value=wb.cfg.subtitles.highlight_color, label="Highlight colour")
                with gr.Row():
                    run_btn = gr.Button("▶ Run / resume analysis", variant="primary", scale=2)
                    cancel_btn = gr.Button("■ Cancel", variant="stop")
                    rerun_btn = gr.Button("↻ Re-run everything (ignore cache)")
                prog_html = gr.HTML(wb.progress_html())
                stage_df = gr.Dataframe(headers=["Stage", "Status", "Time", "Details"], value=wb.stage_rows(), interactive=False, wrap=True, label="Stages")
                log_tb = gr.Textbox(label="Live log", lines=7, max_lines=7, interactive=False)

            # ------------------------------------------------------------ 2. review
            with gr.Tab("2 · Review clips", id="review") as review_tab:
                with gr.Row():
                    view_rd = gr.Radio(["Selected", "Pending", "Approved", "Rejected", "All"], value="Selected", label="Show", scale=3)
                    refresh_btn = gr.Button("↻ Refresh list")
                    bulk_score = gr.Slider(0, 100, value=60, step=1, label="Bulk-approve selected clips scoring at least", scale=2)
                    bulk_btn = gr.Button("✅ Bulk approve")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        cand_df = gr.Dataframe(headers=TABLE_HEADERS, value=[], interactive=False, wrap=True, label="Candidates (click a row)",
                                               column_widths=["5%", "13%", "30%", "12%", "9%", "8%", "14%", "15%"], max_height=420)
                        auto_prev = gr.Checkbox(value=True, label="Render a low-resolution preview when I select a clip (cached; approving never re-renders)")
                        score_html = gr.HTML()
                    with gr.Column(scale=4):
                        video = gr.Video(label="Preview (9:16)", height=480, interactive=False, autoplay=False)
                        prev_note = gr.Markdown("")
                        with gr.Row():
                            approve_btn = gr.Button("✅ Approve", variant="primary")
                            reject_btn = gr.Button("❌ Reject")
                            pending_btn = gr.Button("↩ Back to pending")
                            prev_btn = gr.Button("🎞 Render preview")
                status_md = gr.Markdown("")
                with gr.Row():
                    with gr.Column():
                        title_tb = gr.Textbox(label="Title", max_lines=1)
                        caption_tb = gr.Textbox(label="Caption / description", lines=2)
                        save_title_btn = gr.Button("Save title & caption")
                        info_md2 = gr.Markdown()
                    with gr.Column():
                        with gr.Row():
                            start_n = gr.Number(label="Start (s in source)", precision=2)
                            end_n = gr.Number(label="End (s in source)", precision=2)
                        bounds_btn = gr.Button("Apply boundaries")
                        subs_tb = gr.Textbox(label="Captions — one per line: start-end | text  (seconds from clip start)", lines=9)
                        with gr.Row():
                            subs_btn = gr.Button("Apply caption edits")
                            subs_reset = gr.Button("Reset captions")

            # ------------------------------------------------------------ 3. export
            with gr.Tab("3 · Export", id="export"):
                gr.Markdown("Exports every **approved** clip as a final-quality MP4 (+ SRT, metadata JSON, description) and writes a project summary.")
                export_btn = gr.Button("📤 Export approved clips", variant="primary")
                export_md = gr.Markdown()
                summary_md = gr.Markdown()

            # ------------------------------------------------------------ 4. diagnostics
            with gr.Tab("Diagnostics", id="diag"):
                hw_btn = gr.Button("Check this computer (GPU, RAM, FFmpeg, Ollama)")
                hw_out = gr.Textbox(label="Hardware & dependency report", lines=14, interactive=False)
                err_md = gr.Markdown()
                log_btn = gr.Button("Show recent log")
                log_full = gr.Textbox(label="Log", lines=14, interactive=False)

        gr.Markdown("<small>All processing is local. Scores rank clips for your review; they do not predict views or earnings.</small>")

        # ================================================================ handlers
        setting_inputs = [whisper_dd, device_dd, lang_tb, llm_dd, use_llm_cb, perf_rd, min_d, target_d, max_d, max_clips, min_score,
                          separation, layout_dd, res_dd, hw_dd, subs_on, subs_mode, sub_pos, font_size, hi_color]
        keys = ["whisper", "device", "language", "llm_model", "use_llm", "perf", "min_d", "target_d", "max_d", "max_clips", "min_score",
                "separation", "layout", "resolution", "hw", "subs_on", "subs_mode", "sub_pos", "font_size", "hi_color"]

        def apply_settings(*vals):
            wb.set_config(collect_overrides(dict(zip(keys, vals))))

        @guard
        def on_inspect(path, picked, out_dir):
            ok, msg = wb.tools_ok()
            if not ok:
                raise gr.Error(msg, duration=15)
            _, info = wb.open_video((path or "").strip() or picked, out_dir)
            return info_card(info) + f"\n\nProject folder: `{wb.store.root}`", gr.update(choices=wb.project_choices(out_dir)), wb.stage_rows(), wb.progress_html()

        @guard
        def on_reopen(folder):
            if not folder:
                raise gr.Error("Pick a project first.")
            store = wb.open_existing(folder)
            return info_card(store.state.source) + f"\n\nProject folder: `{store.root}`", store.state.source.path, wb.stage_rows(), wb.progress_html()

        @guard
        def on_run(*vals, force=False):
            apply_settings(*vals)
            wb.start_job(force_all=force)
            return wb.progress_html()

        def on_tick():
            return wb.progress_html(), wb.stage_rows(), wb.log_text()

        @guard
        def on_cancel():
            wb.cancel()
            gr.Info("Cancelling… the current step will stop shortly; finished work is kept.")

        @guard
        def load_list(view):
            return wb.table_rows(view)

        def empty_detail():
            return ("", "", "", "", "<i>Select a clip.</i>", None, None, "", None, "", "")

        @guard
        def on_select(evt: gr.SelectData, view):
            row = evt.row_value if getattr(evt, "row_value", None) else wb.table_rows(view)[evt.index[0]]
            cid = str(row[1])
            d = wb.detail(cid)
            c = d["candidate"]
            return (cid, c.title, c.caption, d["info_md"], d["score_html"], round(c.start, 2), round(c.end, 2), d["subtitles"],
                    None, f"**Status:** {STATUS_ICON[c.status]}" + (f" · not auto-selected: {c.filter_reason}" if c.filter_reason and not c.selected else ""), "")

        select_outputs = [selected_id, title_tb, caption_tb, info_md2, score_html, start_n, end_n, subs_tb, video, status_md, prev_note]

        @guard
        def on_preview(cid):
            if not cid:
                raise gr.Error("Select a clip first.")
            path, note = wb.preview(cid)
            return path, f"_{note}_"

        def maybe_preview(cid, auto):
            return on_preview(cid) if (auto and cid) else (gr.update(), gr.update())

        @guard
        def set_status(cid, view, status):
            if not cid:
                raise gr.Error("Select a clip first.")
            wb.approve([cid], status)
            return wb.table_rows(view), f"**Status:** {STATUS_ICON[status]}"

        @guard
        def on_bulk(score, view):
            n = wb.bulk_approve(score)
            gr.Info(f"Approved {n} clip(s).")
            return wb.table_rows(view)

        @guard
        def on_save_title(cid, title, caption, view):
            from src.review import actions

            actions.update_title(wb.need_store(), cid, title, caption)
            gr.Info("Saved.")
            return wb.table_rows(view)

        @guard
        def on_bounds(cid, s, e, view):
            if not cid:
                raise gr.Error("Select a clip first.")
            try:
                c, warns = wb.apply_bounds(cid, s, e)
            except CandidateValidationError as exc:
                raise gr.Error(str(exc)) from exc
            for w in warns:
                gr.Warning(w)
            gr.Info("Boundaries updated. Press ‘Render preview’ to see the change.")
            d = wb.detail(cid)
            return wb.table_rows(view), d["info_md"], d["score_html"], d["subtitles"], round(c.start, 2), round(c.end, 2)

        @guard
        def on_subs(cid, text):
            if not cid:
                raise gr.Error("Select a clip first.")
            try:
                wb.apply_subtitles(cid, text)
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            gr.Info("Captions updated. Press ‘Render preview’ to see them.")
            return wb.detail(cid)["subtitles"]

        @guard
        def on_subs_reset(cid):
            from src.review import actions

            actions.reset_subtitles(wb.need_store(), cid)
            return wb.detail(cid)["subtitles"]

        @guard
        def on_export(progress=gr.Progress()):
            res = wb.export(on_progress=lambda f, m: progress(f, desc=m))
            ok = [r for r in res if r["ok"]]
            lines = [f"**Exported {len(ok)} of {len(res)} clip(s)** to `{wb.store.exports_dir}`"]
            lines += [f"- ✅ `{Path(r['folder']).name}` — {', '.join(r['files'])}" + (f" · {'; '.join(r['notes'])}" if r['notes'] else "") for r in ok]
            lines += [f"- ❌ {r['id']}: {r['error']}" for r in res if not r["ok"]]
            return "\n".join(lines), summary_markdown(build_summary(wb.store))

        @guard
        def on_hw():
            return format_report(build_report(wb.cfg.llm.host))

        def on_errors():
            if not wb.store:
                return ""
            errs = wb.store.state.errors[-5:]
            return "**Recent errors**\n" + "\n".join(f"- `{e[:300]}`" for e in errs) if errs else ""

        inspect_btn.click(on_inspect, [path_tb, upload, out_tb], [info_md, proj_dd, stage_df, prog_html])
        reopen_btn.click(on_reopen, [proj_dd], [info_md, path_tb, stage_df, prog_html])
        run_btn.click(on_run, setting_inputs, [prog_html])
        rerun_btn.click(functools.partial(on_run, force=True), setting_inputs, [prog_html])
        cancel_btn.click(on_cancel)
        gr.Timer(1.2).tick(on_tick, None, [prog_html, stage_df, log_tb], show_progress="hidden")

        review_tab.select(load_list, [view_rd], [cand_df])
        view_rd.change(load_list, [view_rd], [cand_df])
        refresh_btn.click(load_list, [view_rd], [cand_df])
        cand_df.select(on_select, [view_rd], select_outputs).then(maybe_preview, [selected_id, auto_prev], [video, prev_note])
        prev_btn.click(on_preview, [selected_id], [video, prev_note])
        approve_btn.click(functools.partial(set_status, status="approved"), [selected_id, view_rd], [cand_df, status_md])
        reject_btn.click(functools.partial(set_status, status="rejected"), [selected_id, view_rd], [cand_df, status_md])
        pending_btn.click(functools.partial(set_status, status="pending"), [selected_id, view_rd], [cand_df, status_md])
        bulk_btn.click(on_bulk, [bulk_score, view_rd], [cand_df])
        save_title_btn.click(on_save_title, [selected_id, title_tb, caption_tb, view_rd], [cand_df])
        bounds_btn.click(on_bounds, [selected_id, start_n, end_n, view_rd], [cand_df, info_md2, score_html, subs_tb, start_n, end_n])
        subs_btn.click(on_subs, [selected_id, subs_tb], [subs_tb])
        subs_reset.click(on_subs_reset, [selected_id], [subs_tb])
        export_btn.click(on_export, None, [export_md, summary_md])
        hw_btn.click(on_hw, None, [hw_out])
        log_btn.click(lambda: wb.log_text(300), None, [log_full]).then(on_errors, None, [err_md])
    return demo


def launch(wb: Workbench, host: str = "127.0.0.1", port: int = 7860, inbrowser: bool = True) -> None:
    demo = build_ui(wb)
    demo.queue(default_concurrency_limit=4)
    demo.launch(server_name=host, server_port=port, inbrowser=inbrowser, show_error=True, quiet=False,
                allowed_paths=[str(wb.preview_dir)], theme=gr.themes.Soft(primary_hue="indigo"), css=CSS, footer_links=[])
