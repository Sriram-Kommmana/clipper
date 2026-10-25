# Architecture

Everything runs locally: FFmpeg for media, faster-whisper for speech, Ollama (Qwen3) for language
understanding, OpenCV/librosa/SciPy for signal analysis, Gradio (bound to 127.0.0.1) for review.
No network calls are made except to the **local** Ollama server (the client refuses non-loopback hosts
unless `llm.allow_remote_host: true`).

## Data flow

```
 local file ──► [A] ingest ──► [B] transcribe ──► [C] audio + scene analysis
                 probe,validate   faster-whisper     RMS/silence/peaks, ffmpeg scene cuts
                 fingerprint      words + SRT        quality checks, thumbnails
                                        │                      │
                                        ▼                      ▼
                          [D] candidate generation  ◄──  audio-event candidates (supplement)
                              sectioned LLM passes → verify against transcript → merge duplicates
                                        │
                                        ▼
                          [E] scoring  (LLM rubric  +  deterministic terms)  → 0-100 + components
                                        │
                                        ▼
                          [F] selection (duration, score, separation, diversity, category prefs)
                                        │   [G] boundary refinement + timestamp validation
                                        ▼
                      review UI  (approve / reject / edit bounds / edit subtitles / preview)
                                        │
                                        ▼
                  [H][I] one-pass FFmpeg render: accurate cut + 9:16 + ASS captions + H.264/AAC
                                        │
                                        ▼
                          [K] export: mp4 + srt + metadata json + summary
```

Stages A–G are the **analysis job** (`JobManager`), resumable and cached. Rendering (H–K) is on demand
from the review UI or CLI, with its own content-hash cache so approving never re-renders.

## Package layout (`src/`)

| Package | Responsibility |
|---|---|
| `core` | pydantic models & schemas, exception hierarchy (each with a recovery hint), `project_manager`, `job_manager` (stages, persistence, cancellation, progress), `exporter` (clips + summary) |
| `ingestion` | `ffprobe` metadata, input validation, content fingerprint |
| `transcription` | `Transcriber` with pluggable backend (faster-whisper; device/OOM fallback), transcript models, JSON/SRT |
| `analysis` | audio features/events/quality, scene detection, frame extraction & visual checks |
| `llm` | Ollama client (JSON-schema constrained output), prompts, tolerant parser + retrying structured call |
| `candidates` | generator (sections), merger, scorer, selector, validator/boundary refiner |
| `rendering` | FFmpeg runner (argument lists only), command builders, ASS subtitles, vertical layouts, subject detectors, encoders |
| `review` | `workbench` (UI-independent controller), `actions` (approve/reject/edit bounds/captions), `interface` (Gradio) |
| `storage` | atomic JSON IO, stage cache with keys, per-project store |
| `utils` | logging, hardware detection |

Top level: `app.py` (CLI + UI launcher), `config.py` (validated config), `scripts/` (diagnostics, model download).

## Project folder

```
projects/<slug>-<fingerprint8>/
  project.json            # source info, settings snapshot, job/stage state, timings, log of errors
  cache/                  # stage outputs; each file = {key, created, version, data}
    transcript.json  transcript.srt  audio.json  scenes.json  candidates.json  scores.json
    frames/               # scene thumbnails
  work/                   # temporary (16 kHz wav) — removed after a successful run
  renders/                # content-hash named previews + finals (cache for the review UI)
  exports/NN_title/       # clip.mp4, clip.srt, clip.json
  logs/clipper.log
  summary.json  summary.md
```

All JSON is written atomically (`tmp` file + `os.replace`).

## Caching & resume

Each stage has a **cache key** = hash(source fingerprint, stage-relevant config sections, stage code
version, upstream keys). A stage is skipped iff its cache file exists and its key matches. Changing the
Whisper model invalidates the transcript *and everything downstream*; changing `selection.*` only re-runs
selection. Job state (`pending/running/done/failed/cancelled/skipped`) is persisted before and after
every stage, so a crash leaves `running` which is reported as `interrupted` on next open and simply
re-run. Completed stages are never repeated. Candidate edits (status, bounds, subtitle overrides) live in
`candidates.json` and survive re-selection because decisions are merged by stable candidate ID.

## Candidate IDs

`c` + first 8 hex of `sha1(fingerprint : round(start) : round(end) : category)` at creation; persisted and
never recomputed after the user edits bounds.

## Scoring (documented in `candidates/scorer.py`)

Seven components, each 0-100: **hook, clarity, completeness, payoff, escalation** (LLM 1-10 rubric ×10,
blended with deterministic boundary/speech checks), **format_fit** (duration vs target; deterministic),
**av_suitability** (speech density, silence ratio, energy, black/blurry frames; deterministic).
`total = Σ weight·component` (weights in config). It is a *ranking heuristic*, not a virality prediction.

## Anti-hallucination measures

* The model is asked only about the supplied timestamps; times are clamped into the section.
* Excerpts are always taken from the transcript by timestamp, never from model text.
* The `hook` must be (fuzzily) present in the excerpt; otherwise it's replaced by the first sentence and a
  warning is added.
* All responses are parsed through pydantic schemas; invalid → bounded retries → structured error.
* Model output is data only: never executed, never used in a shell command or file path.

## Resource strategy

Models are loaded one at a time: Whisper is released before the LLM stages begin; Ollama's `keep_alive`
is set to unload the model when the pipeline ends (`0` in performance "low" mode). GPU failure/OOM during
transcription falls back (visibly) to CPU int8 when `allow_cpu_fallback` is on. Audio is analysed in
streaming blocks; frames are read one at a time. FFmpeg threads/priority follow `performance.*`.

## Extension points

* `WhisperBackend` protocol (transcription), `LLMClient` protocol (llm), `SubjectDetector` protocol
  (smart crop) — swap implementations without touching callers.
