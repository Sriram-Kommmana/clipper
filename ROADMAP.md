# Roadmap

Legend: `[x]` implemented **and** exercised by tests or a real run in the dev sandbox (Linux, no GPU, no Ollama, no model downloads);
`[~]` implemented, but a key part could not be verified in the sandbox (stated); `[ ]` not done.

## Phase 1 — Inspection and planning
- [x] Inspect reference repository + licence check (no LICENSE → nothing copied) → `docs/REFERENCE_ANALYSIS.md`
- [x] Architecture doc, Git workflow, scaffold, validated layered config

## Phase 2 — Minimum end-to-end pipeline
- [x] Ingestion: probe, validation, rotation, no-audio/corrupt detection, content fingerprint
- [~] Transcription (faster-whisper, word timestamps, JSON+SRT, device/compute plan, OOM→CPU fallback) — real inference run on CPU; **GPU inference not run** (cuBLAS missing, fell back to CPU as designed)
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
- [~] Real pipeline with real models on Windows — run on a text-to-speech test video; **real footage still to try**
- [~] Windows install check — existing venv and `scripts/check_environment.py` run on Windows 11 / Python 3.10; `setup_windows.bat` itself not run

## Phase 7 — Finalisation
- [x] README, troubleshooting, third-party notices, architecture, reference analysis
- [x] Development log kept per milestone

## Ideas not started
Speaker diarisation · local VLM visual validation · feedback loop from rejected clips · batch of several videos · mid-stage checkpointing
