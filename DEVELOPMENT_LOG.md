# Development log

Newest last. Test results are what was actually executed in the cloud dev sandbox (Linux, no GPU,
no Ollama, Hugging Face/Ollama hosts blocked) — not on the target Windows laptop.

## Milestone 1 — scaffold, config, reference analysis
* Changed: created package skeleton, `config.py` + `configs/default.yaml` (validated, layered), exception
  hierarchy, `.gitignore`, requirements, `docs/REFERENCE_ANALYSIS.md`, `docs/ARCHITECTURE.md`, roadmap.
* Reference repo: cloned depth-1, read source. **No LICENSE found → nothing copied.**
* Tests run: none yet.
* Commit: (recorded in next entry)
* Next: core models, storage, ingestion.
* Commit: `002a443 chore: initialize project structure, config system and reference analysis`

## Milestone 2 — ingestion, storage, FFmpeg runner
* Added: `ffmpeg_runner` (arg lists only, progress, cancel, timeout, low priority), `input_validator`
  (absolute paths, extension/size checks, content fingerprint), `video_probe` (rotation, no-audio and
  corrupt-file handling), atomic JSON store, keyed `StageCache`, `ProjectStore`/project manager,
  logging, pydantic models + LLM schemas, transcript model.
* Tests actually run: `pytest` → 27 passed (config, ingestion on generated MP4s, storage/cache/atomic writes).
* Known gaps: nothing renders yet.
* Commit: `db44a80 feat: add video ingestion, metadata probing and project storage`

## Milestone 3 — transcription + diagnostics
* Added: `Transcriber` (faster-whisper backend with CUDA→int8_float16→CPU int8 plan, OOM fallback that is
  recorded as a project warning, `local_files_only` so there are no hidden downloads), stage wrapper with
  cache key (fingerprint+model+language+vad+beam), JSON + SRT output, `hardware_detection` +
  `scripts/check_environment.py`, `scripts/download_models.py`.
* Tests: 35 passed (fake backend; cache hit/invalidate; no-speech and no-audio errors; cancel; device plan).
* **Not verified:** the real faster-whisper inference path. Hugging Face is blocked in the sandbox
  (`download_models.py --whisper tiny` → 403), so only model *loading failure messaging* was exercised.
* Commit: `2847a6e feat: implement local transcription with device fallback and hardware diagnostics`

## Milestone 4 — audio, scene and frame analysis
* Added: streaming librosa RMS analysis (silences, peaks, sustained spans, sudden changes, quality
  warnings), FFmpeg scene detection (+ keyframe-only fast mode), OpenCV frame sampling, visual checks and
  thumbnails.
* Bugs found by tests and fixed: single-window bursts lost by smoothing (peaks now use raw energy);
  noise-floor stat polluted by digital silence.
* Tests: 44 passed total (new: synthetic energy arrays, streamed WAV, real FFmpeg scene cuts on a generated MP4).
* Commit: `06def31 feat: add audio, scene and frame analysis`

## Milestone 5 — Ollama integration (`feat: implement Ollama integration…`)
* `OllamaClient` (streaming, cancel, JSON-schema `format`, `think:false` with old-server fallback, loopback-only guard,
  unload), tolerant JSON extraction (fences, `<think>`, prose), pydantic validation, bounded retries with error feedback.
* Tests: 11 new, against a fake Ollama HTTP server and a scripted LLM. **Real Qwen3 never ran** (sandbox blocks ollama.com).

## Milestone 6 — candidates (`feat: narrative candidate generation, merging, scoring, selection…`)
* Sectioned generation, verification of times/excerpts/hooks against transcript, audio-event supplement with
  reaction lead-in rule, heuristic no-LLM fallback, merger, boundary refinement, hard validation, 7-component
  scorer, selector (separation / diversity / preferences / user decisions kept).
* Bugs found by tests: merger let a high-confidence audio-only trigger outrank an LLM candidate (now LLM always
  wins); misc. test-data errors.
* Tests: 88 passed total.
* Commit: `15c1dea`

## Milestone 7 — rendering (`feat: synchronized subtitles, vertical conversion and one-pass renderer`)
* ASS karaoke/plain captions + SRT + edit round-trip, blur / smart-crop / original layouts, OpenCV Haar-face and
  HOG-person detectors behind `SubjectDetector`, dead-banded crop path planner, NVENC only after a real test encode with
  automatic x264 retry, one-pass accurate cut + layout + captions, content-hash render cache.
* Verified with real FFmpeg: output size/codecs/duration; audio onset after a cut lands within 0.15 s of expected;
  yellow highlight pixels appear at the right clip time and not before the first word; centre-crop fallback;
  moving crop; no-audio source; cancel leaves no files; NVENC-failure fallback. Visually inspected a full-size frame.
* Tests: 102 passed total (this includes ~8 real-render tests).
* Commit: `64bceca feat: add synchronized subtitle rendering and vertical video conversion`
* Correction: the previous entry said "102 passed"; the actual count at that commit was **114 passed**.

## Milestone 8 — resumable job manager, review actions, export
* `JobManager` (stage keys from fingerprint+config+upstream keys, persisted status, interrupted-run recovery,
  cancellation, failure keeps earlier work, model unload, temp cleanup), `merge_user_decisions` (approve/reject,
  edited bounds/subtitles/manual clips survive re-runs), `review/actions.py`, `exporter` (+ summary.json/.md).
* Bug found by the new integration tests: the transcribe stage was never recognised as cached (stage vs cache
  name mismatch) so resume would have re-transcribed. Fixed.
* Integration tests (real FFmpeg/librosa/OpenCV; fake Whisper + scripted LLM): full run, resume, selective
  invalidation (selection-only, scoring-only, model change), failure+resume, crash-mid-stage, cancel, decisions
  surviving re-runs, heuristic mode, dead-Ollama failure message, approval does not render, bounds/subtitle/title
  edits, export artifacts + summary.
* Tests: **125 passed** (26 s).
* Commit: `c8810c5 feat: add caching, resumable jobs, review actions and clip export`

## Milestone 9 — review interface, CLI, hardening
* Gradio 6 UI (4 tabs), `Workbench` controller, CLI (`app.py ui|run|check`), Windows `setup_windows.bat` / `run_clipper.bat`.
* **Driven in headless Chromium** (Playwright): open project, select clip → auto-render preview (7.5 s at 540×960 on 4 CPU
  cores), score bars, approve, export 1 clip, Whisper-missing error panel with fix command, hardware report. Screenshots inspected.
* Real bugs found only by this browser run, fixed: Approve/Reject/Pending buttons crashed (`functools.partial` keyword collided with
  Gradio's positional args); table headers cramped; caption grouping left orphan words (now balanced 3+3+3).
* Review of merge logic: added `title_edited` so edited titles survive re-runs but regenerated ones are not frozen.
* Privacy: `GRADIO_ANALYTICS_ENABLED=False`, `HF_HUB_OFFLINE=1`, UI binds 127.0.0.1 (CLI refuses others). Proxy log showed **no
  outbound attempts** during UI/pipeline runs (only my deliberate probes to ollama.com / huggingface.co, which are blocked here).
* Scale check: 40-min 640×360 video → audio extraction 10 s, audio analysis 1.1 s, scene detection 33 s (5 s keyframe-only), peak RSS 251 MB;
  all 8 planted loud bursts detected.
* Docs: README, TROUBLESHOOTING, THIRD_PARTY_NOTICES (licences marked *metadata-verified* vs *from publisher docs, re-check*).
* Tests: **137 passed, 1 deselected** (the opt-in Ollama smoke test) in ≈30 s. Browser smoke test saved in `tests/ui/` (manual).
* Python compatibility: `vermin` reports minimum 3.9 syntax; project documents 3.10–3.13 (only 3.13 was actually run).

## Milestone 10 — first real-model run (Windows 11, Python 3.10, RTX 3050 6 GB)
* Run on a 205 s text-to-speech test video with real Whisper `small` (CPU) and real `qwen3:latest` (8.2B). NVENC and the browser UI also worked.
* Bugs the fake LLM could not reveal, now fixed:
  * `json_schema_for` stripped every key named `title`, including the moment's `title` field, so every real LLM call failed validation.
  * The model returned the timestamps of one transcript line (2-6 s), so every moment was dropped as too short. It now picks numbered
    lines (`first_line`, `last_line`, after the title) and the code converts them to times.
  * `opencv-python-headless>=4.8` resolved to 5.0, which removed the Haar and HOG detectors, so smart crop always fell back to a centre crop. Now `<5`.
* Result after the fixes: 5 moments from 1 section, none dropped, no retries, all scored by the model; 2 selected, the others within `min_separation_s`.
* Tests: 138 passed, 2 failed. Both failures are faulty tests: `test_robustness.py:151` reads a UTF-8 file with the system encoding, and
  `test_transcription.py:122` assumes the machine has no CUDA.

## Known bugs / limitations
* Whisper on a GPU, `qwen3:4b` and smart crop on real faces are unverified (see ROADMAP `[~]`).
* Transcription/LLM stages cannot resume mid-stage; a cancelled/failed stage restarts from its beginning.
* Haar face detector is frontal-only; untested on real faces.
* One open project / one job at a time (single-user local app).

## Next recommended actions (for you, on the laptop)
1. `setup_windows.bat`, install Ollama, `python scripts\download_models.py`, `python scripts\check_environment.py`.
2. `pytest` (140 tests; see Milestone 10 for the 2 known failures) and `pytest -m ollama tests/integration/test_ollama_smoke.py`.
3. Run a 3–5 min clip of authorised content through the UI; compare chosen moments against your judgement; tune `configs/local.yaml`.
4. Report anything odd (device shown in the Transcription stage row, caption sync, crop quality).
