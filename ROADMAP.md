# Roadmap

Legend: `[x]` implemented **and** exercised by tests or a real run in the dev sandbox (Linux, no GPU, no Ollama, no model downloads);
`[~]` implemented, but a key part could not be verified in the sandbox (stated); `[ ]` not done.

## Phase 1 — Inspection and planning
- [x] Inspect reference repository + licence check (no LICENSE → nothing copied) → `docs/REFERENCE_ANALYSIS.md`
- [x] Architecture doc, Git workflow, scaffold, validated layered config

## Phase 2 — Minimum end-to-end pipeline
- [x] Ingestion: probe, validation, rotation, no-audio/corrupt detection, content fingerprint
- [x] Transcription (faster-whisper, word timestamps, JSON+SRT, device/compute plan, OOM→CPU fallback, language retry) — real inference run on the GPU and CPU; or import a timed `.srt`/`.vtt`
- [~] Ollama client + structured output + retries — run with real `qwen3:latest` (8.2B); **`qwen3:4b` not run**
- [x] Narrative candidate generation (sections, verification against transcript, merge)
- [x] Accurate one-pass FFmpeg cut + captions + 9:16 render (audio onset accuracy and caption sync measured)
- [x] Review UI + export (driven in Chromium: open, preview, approve, export, error panel)

## Phase 3 — Candidate quality
- [x] Audio analysis (RMS, silences, peaks, sustained, changes, quality) — streamed; 40-min file in 1 s / 251 MB
- [x] Scene detection (FFmpeg), frame sampling, visual checks
- [x] Deduplication / cross-section merge
- [x] Scoring (7 components), selection, diversity, category preferences
- [x] Timestamp validation & boundary refinement (sentence/word snapping, safe padding)
- [ ] Measuring candidate *quality* with real models on real videos (needs the user's hardware)

## Phase 4 — Editing and rendering
- [x] Blurred-background vertical layout
- [~] Smart crop: path planning/dead-band/centre fallback/moving crop verified; **face detector not tried on real faces**
- [x] Karaoke + plain captions, SRT, caption editing with timing preservation
- [x] Editable boundaries (validated, rescored), batch export
- [~] NVENC: only used after a real test encode; fallback to x264 tested by simulation; **never run on a GPU**

## Phase 5 — Reliability
- [x] Keyed caches with selective invalidation; atomic writes
- [x] Resumable jobs (failure, crash mid-stage, cancel), user decisions survive re-runs
- [x] Performance mode, thread limits, sequential models, unload, temp cleanup
- [x] Actionable errors with recovery text; live log; stage table
- [ ] Mid-stage resume inside Whisper / LLM sections (a stage restarts from its beginning)

## Phase 6 — Verification
- [x] Unit tests (137 passing incl. real FFmpeg renders)
- [x] Integration tests on a generated MP4 with fake Whisper/scripted LLM
- [x] Zero-paid-API audit (static test + observed: no outbound connection attempts during UI/pipeline runs)
- [~] Real pipeline with real models on Windows — run on a real 20-minute Hindi and English interview; **clips still to be judged by a person**
- [~] Windows install check — existing venv and `scripts/check_environment.py` run on Windows 11 / Python 3.10; `setup_windows.bat` itself not run

## Phase 7 — Finalisation
- [x] README, troubleshooting, third-party notices, architecture, reference analysis
- [x] Development log kept per milestone

## Phase 8 — Robustness pass after the competitor review (2026-11)
- [x] Per-section / per-batch LLM answers kept; failed sections reported and retried alone
- [x] Scoring judges the clip text alone (**whether this raises quality is unmeasured**; use `python app.py eval`)
- [x] Optional `llm.audio_hints` (off by default until compared on real videos)
- [x] Opening / sponsor-read warnings (English, Hindi, Romanised Hindi)
- [x] Smart-crop path planned per scene cut; per-shot centre fallback
- [x] Two-pass linear loudness normalisation of exports
- [x] Evaluation harness: `python app.py eval` + `docs/EVALUATION.md`
- [x] GitHub Actions workflow running the test suite (**not yet seen to run on GitHub**)
- [x] WebM / M4V / AVI inputs
- [ ] Run `eval` on real labelled videos and decide the `audio_hints` default
- [ ] A licence for the repository (the owner has to choose it)

## Ideas not started
Better face detector than Haar (YuNet / MediaPipe; needs a model download) · standalone-clip gate and ending/opening review passes with the LLM · head-to-head re-rank of finalists · video-kind routing of prompts · Roman-script (Hinglish) captions · Hinglish-tuned Whisper models (need conversion to CTranslate2) · feedback loop from rejected clips (learning from examples moved results most in published research) · speaker diarisation · local VLM visual validation · batch of several videos · mid-stage checkpointing
