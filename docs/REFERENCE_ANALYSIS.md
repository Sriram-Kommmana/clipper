# Reference analysis — `LanderIglesias/Clipper`

Inspected on 2026-10-25 from a depth-1 clone (HEAD `754a2a4`, "docs: add demo video…"). Everything below
is from reading the actual source (≈3.9k lines of Python), not just the README.

## 1. Licence — read this first

* The repository tree contains **no `LICENSE` / `COPYING` file** (checked the full file list). With no
  licence, default copyright applies: nobody may copy, modify or redistribute the code without the
  author's permission.
* **Decision: no code, prompts or text were copied.** This project is an independent implementation. Only
  *ideas and observable behaviour* (which are not copyrightable) informed the design, and each is
  re-derived and documented below. If the author later adds a licence, this can be revisited.
* The repo also commits `yolo11n.pt`. Ultralytics YOLO code and weights are **AGPL-3.0** (or a paid
  enterprise licence). That is unsuitable as a mandatory dependency for something that may become
  commercial, which matches the instruction not to make YOLO mandatory.

## 2. What the reference does (verified in code)

| Stage | Reference implementation | Notes from the code |
|---|---|---|
| Transcribe | `faster-whisper`, `large-v2`, `device="cuda"` hard-coded (`config.py`) | No `compute_type`, no CPU fallback, no OOM handling. `word_timestamps=True`. Cache keyed by **file stem only**. |
| Audio | `librosa` RMS in 0.5 s windows; peaks = top 15 % of normalised energy; silence < 5 % | Windows are normalised per video, so a quiet video still has "peaks". |
| Narrative candidates | **One** Claude call with the whole transcript (`[12.3s] text` lines) | Transcripts over 8000 words are **thinned by dropping every other segment** (`segments[::2]`) — the model then reasons over half the text. Candidates below virality 7 / clarity 6 are dropped. |
| Audio-peak candidates | `_find_action_peaks` adds "action" windows from sustained high energy | Supplements the LLM list. |
| Scoring | Claude, batches of 10, rubric: hook / escalation / payoff / clarity / format / "content virality", composite weights 0.30/0.15/0.20/0.15/0.05/0.15 | Scores 1–10, trusted as-is. |
| Vision | Claude Vision over 5 frames per clip; approve / adjust / reject; final = 0.6 LLM + 0.4 vision | Needs a paid API. |
| Cutting | `ffmpeg -ss … -c copy -map 0` | README claims "re-encode from source for accurate seeking" — the cutter itself uses **stream copy** (keyframe-snapped, inexact); accuracy is only restored when the later subtitle step re-encodes. Action clips start 5 s earlier, funny 2 s (fixed offsets). |
| Subtitles | Word timestamps → ≤4-word lines → ASS with `Default`/`Highlight` styles, one dialogue per word state | Good idea; hard-coded Arial, fixed margins. |
| Vertical | `blur_background` (`filter_complex`, bg scaled+blurred, fg overlaid) or YOLO11 person detection on 10 sampled frames → **single static** crop at the mean x | No temporal smoothing/tracking despite the "tracking" label; whole clip uses one crop. |
| UI | Gradio: load analysis JSON, per-clip video, Approve/Reject, subtitle textbox → regenerate, export approved | State lives in module-level globals / files named by rank (`_find_clip_video(rank)`), so re-ranking or re-runs can mismatch files. |
| Other | Google-Drive sync script, `.env` with `ANTHROPIC_API_KEY` | Out of scope (network/cloud). |

## 3. Engineering decisions worth keeping (re-implemented, not copied)

1. **Narrative candidates beat fixed windows** — transcripts describe events with a lag and windows cut
   the setup/payoff. We keep this, but replace the single whole-transcript call with **overlapping
   transcript sections** (a 4B local model cannot take a 1-hour transcript, and thinning is lossy), then
   merge duplicates across section seams.
2. **Audio supplements, never decides.** We keep energy peaks as a *secondary* candidate source and as a
   scoring input only.
3. **Re-encode from the source for subtitle sync.** We do cut + reframe + burn in **one FFmpeg pass** with
   input-side `-ss` (frame-accurate on re-encode), so audio/video/subtitles share one clock.
4. **Blurred-background 9:16 via `filter_complex`** — same concept; our background is blurred at low
   resolution for speed.
5. **Word-level ASS highlighting** — one dialogue event per word state; works in any libass build.
6. **Batch scoring** to amortise per-call overhead — kept, with small batches (4) sized for a 4B model.
7. **JSON caches** — kept, but keyed by *content fingerprint + config hash + version* rather than a filename.

## 4. Things deliberately not adopted

| Reference behaviour | Why not | Replacement here |
|---|---|---|
| Claude API for narrative, scoring, vision | Paid, cloud, forbidden by requirement | Ollama + `qwen3:4b` with schema-validated JSON; deterministic scoring terms; OpenCV frame checks |
| Fixed 5 s backward shift for "action" | Arbitrary; hides a timing problem | Boundaries refined from word timestamps; for audio-triggered *reaction* moments only, include the preceding sentence(s) as setup (bounded, documented, configurable) |
| `-c copy` cuts | Keyframe-inexact | Accurate re-encode in one pass |
| `segments[::2]` thinning | Loses half the story | Sectioned LLM passes with overlap |
| Static single crop | Subject moves | Sampled detections → smoothed, dead-banded, piece-wise crop path; centre-crop fallback |
| Hard-coded `cuda`/`large-v2` | Breaks on weaker machines | Device/compute auto-selection, OOM → CPU int8 fallback (reported, never silent) |
| Global-state UI keyed by rank | Fragile | Stable candidate IDs persisted per project |
| Trusting LLM numbers/text | Hallucination risk | Pydantic schemas; timestamps clamped to transcript; excerpts and hooks are **verified against the transcript**, not taken from the model |
| Ultralytics YOLO | AGPL | OpenCV Haar-face / HOG-person detectors behind a `SubjectDetector` interface |
| Google Drive sync | Cloud dependency, out of scope | Local files only |

## 5. Component → replacement map

| Capability | Local replacement | Licence (see THIRD_PARTY_NOTICES.md) |
|---|---|---|
| Speech-to-text | faster-whisper (CTranslate2) `small` default | MIT / MIT; weights MIT |
| Narrative reasoning, titles, hooks, categories | Ollama `qwen3:4b` (opt. `qwen3:1.7b` / `qwen3:8b`) | Ollama MIT; Qwen3 weights Apache-2.0 |
| Scoring | LLM rubric + deterministic terms (duration fit, boundaries, speech density, silence, brightness/sharpness) | — |
| "Vision validation" | OpenCV: black-frame, brightness, sharpness over sampled frames | Apache-2.0 |
| Subject tracking | OpenCV Haar face / HOG person (swappable) | Apache-2.0 / BSD-style cascade licence |
| Rendering | FFmpeg (libx264 / NVENC if verified) | LGPL/GPL depending on build — not bundled |

## 6. What can be adapted vs deferred

* **Adapt now:** all of the above rows.
* **Deferred:** speaker diarisation (no reliable permissive local option on 6 GB alongside the other
  models); learned vision validation with a local VLM (e.g. a small Qwen-VL) — the interface leaves room
  but it would compete for VRAM; "feedback loop from rejected clips" (reference's own TODO); direct upload
  to social platforms (network + account concerns).

## 7. Caveats about this analysis

* It reflects the repository at the commit above; behaviour I describe was read from code, not executed
  (the reference needs a paid API key to run).
* I did not verify the reference's claimed results (80 % approval, 4-minute runtime).
