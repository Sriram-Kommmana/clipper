# Clipper — local AI video clipper

Turn long videos you are **authorised to edit and republish** into short vertical clips with animated captions, entirely on your own
computer. Transcription (faster-whisper), moment finding and scoring (Ollama + Qwen3), analysis (librosa/OpenCV) and rendering (FFmpeg)
all run locally. **No paid API, no API key, no cloud service, no telemetry.** After the one-time model download it works offline.

```
video → transcribe → audio & scene analysis → find moments (local LLM) → score → select
      → review in browser (preview, approve/reject, edit bounds & captions) → export MP4 + SRT + metadata
```

> **Honest status** (see [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) for details): the full pipeline has now run on a Windows laptop with real
> Whisper (`small`, CPU) and real Qwen3 (`qwen3:latest`, the 8.2B model), plus NVENC rendering, on one 3.4-minute text-to-speech test video.
> **Not yet tried:** Whisper on a GPU, `qwen3:4b`, smart crop on real faces, long videos, and clip quality on real footage.

## Requirements
* Windows 10/11 (also works on Linux/macOS), Python **3.10 – 3.13** (3.11/3.12 recommended)
* 16 GB RAM; NVIDIA GPU with 6–8 GB VRAM recommended (CPU works, slower)
* [FFmpeg](https://ffmpeg.org) *full* build with libx264 (`winget install Gyan.FFmpeg`)
* [Ollama](https://ollama.com/download)

## Install (Windows)
```bat
setup_windows.bat                 :: creates venv, installs requirements, prints a hardware report
venv\Scripts\activate
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12   :: only for NVIDIA GPU transcription
python scripts\download_models.py                   :: one-time: Whisper "small" + `ollama pull qwen3:4b`
python scripts\check_environment.py                 :: confirm GPU/CUDA/FFmpeg/Ollama and see recommended settings
```
Manual equivalent (any OS): `python -m venv venv`, activate, `pip install -r requirements.txt`.

## First run
```bat
run_clipper.bat                   :: or: python app.py        → http://127.0.0.1:7860
```
1. **Import & process** – paste the video path (or pick a file), *Inspect & open project*, adjust settings, *Run / resume analysis*.
   Stages show status, timing, device (`cuda/float16` or `cpu/int8`) and whether they were reused from cache.
2. **Review clips** – candidates sorted by score; click one to see a rendered 9:16 preview, the score breakdown, why it was picked, the transcript
   excerpt and warnings. Edit title/caption, start/end (validated), captions (`start-end | text`), then Approve / Reject. Rejected clips stay in
   the project history (*Show → Rejected*). Approving never re-renders anything.
3. **Export** – every approved clip → `projects/<video>/exports/<id>_<title>/` with `.mp4` (H.264/AAC), `.srt`, `.json` (metadata, scores),
   `description.txt`; plus `summary.json` / `summary.md` (durations, counts, warnings, errors).

Headless: `python app.py run my_video.mp4 --export-top 5` · diagnostics: `python app.py check` · help: `python app.py -h`.

## How candidates are found and scored
* The transcript is cut into overlapping sections (≈10 min, 90 s overlap, ≤1400 words) so a 4B model sees all of it. Each section is sent to the local model
  with a JSON schema; invalid answers are retried (max 2 times) with the validation error fed back.
* Everything the model says is **checked against the transcript**: times are clamped, the excerpt is rebuilt from the transcript (never taken from the model),
  and a "hook" that is not a verbatim quote is replaced by the clip's first sentence with a warning.
* Strong audio events (sustained excitement/laughter, big peaks, clearly louder than the recording's typical level) that the LLM did not cover become
  lower-confidence *reaction* candidates with a lead-in so the setup is kept, widened to whole sentences. They supplement the semantic candidates; loudness alone never makes a clip good.
* Duplicates (including the same moment seen by two sections) are merged; boundaries snap to sentence edges using word timestamps; invalid ranges are filtered with a reason.
* **Score (0–100)** = weighted sum of hook, clarity, completeness, payoff, escalation (LLM rubric), length fit and audio/visual suitability (deterministic).
  Weights live in `configs/default.yaml`. **It is a ranking heuristic for ordering your review queue — not a prediction of views, engagement or revenue.**
* Selection honours min/max/target duration, max clips, min score, minimum gap between clips, category preferences and a diversity penalty. Your approvals/rejections/edits survive re-runs.

## Configuration
Defaults: `configs/default.yaml` (documented). Put overrides in `configs/local.yaml` (git-ignored), e.g.
```yaml
transcription: {model: medium}          # bigger Whisper
llm: {model: qwen3:8b}                  # needs ~6+ GB free VRAM; models never run at the same time
performance: {mode: low}                # slower but keeps the laptop responsive
render: {mode: smart_crop}              # follow a face instead of the blurred-background layout
```
Unknown keys / bad values are rejected at start-up with a readable message.

**Performance mode `low`**: half the CPU threads, below-normal priority for FFmpeg, keyframe-only scene detection, greedy Whisper decoding, smaller LLM context and
batch sizes, LLM unloaded after every request. Models are always loaded one at a time (Whisper is released before the LLM stages); Ollama is told to unload at the end of a run.
GPU out-of-memory during transcription falls back to CPU int8 **and says so** (never silently, never to a cloud service).

## Project layout
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Reference-repo findings and licence check: [docs/REFERENCE_ANALYSIS.md](docs/REFERENCE_ANALYSIS.md).
Licences of dependencies/models: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Problems: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Tests
```bash
pip install -r requirements-dev.txt
pytest                                   # 147 tests, ~30 s, needs FFmpeg, no models/internet
pytest -m ollama tests/integration/test_ollama_smoke.py   # optional: real local Ollama model
node tests/ui/ui_smoke.js …              # optional browser smoke test (see tests/ui/README.md)
```
Unit tests use a scripted LLM/fake Whisper; integration tests render real video with FFmpeg.

## Privacy & safety
Binds to `127.0.0.1` (refuses other addresses without `--allow-lan`); `GRADIO_ANALYTICS_ENABLED=False`, `HF_HUB_OFFLINE=1`; the Ollama client refuses non-loopback hosts;
subprocesses use argument lists (no shell) with absolute, validated paths; model output is data only and never executed. Media, models, caches and outputs are git-ignored.

## Known limitations
* Real-model quality is unmeasured. Qwen3-4B may miss moments or mis-categorise; the review step exists for that. Non-English quality depends on Whisper and Qwen.
* No speaker diarisation. Smart crop uses OpenCV's frontal-face Haar detector (cheap, misses profiles/groups); it falls back to a centre crop and reports it.
* One job at a time; the UI is single-user. Captions burn in with libass using your installed fonts.
* Only local MP4/MOV/MKV files; no URL download, no auto-upload.
