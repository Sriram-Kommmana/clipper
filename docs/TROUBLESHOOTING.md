# Troubleshooting

Run `python scripts/check_environment.py` first; it prints what is missing and what it recommends for *your* machine.
Errors in the UI/CLI always end with a "How to fix" line. Project logs: `projects/<name>/logs/clipper.log`.

| Symptom | Cause | Fix |
|---|---|---|
| `'ffmpeg' was not found on PATH` | FFmpeg not installed | `winget install Gyan.FFmpeg`, open a **new** terminal; or set `paths.ffmpeg_path` / `paths.ffprobe_path` in `configs/local.yaml`. Use a *full* build (needs libx264 + the `ass` filter). |
| `Whisper model 'small' is not downloaded` | Weights are fetched once, never at run time | With internet: `python scripts/download_models.py --whisper small` |
| `Cannot connect to Ollama` | Server not running | Start the Ollama app (or `ollama serve`). Check `ollama list`. |
| `Ollama … model 'qwen3:4b' is not installed` | Model not pulled | `ollama pull qwen3:4b` (≈2.6 GB) |
| Clips are poor and scores are low; the transcript looks garbled | Weak speech recognition, common with Hindi or mixed Hindi-English on `small` | Check `cache/transcript.srt`. Use `large-v3-turbo` (`python scripts\download_models.py --whisper large-v3-turbo`), set the spoken language, or import a timed `.srt`/`.vtt` on the first tab. |
| `The transcript has no timestamps` | Plain text was given | Clips are cut by time, so use an `.srt` or `.vtt`. |
| Want Whisper again after importing a transcript | The imported file is reused by later runs | Delete `projects/<name>/imported_transcript.*`, then Run. |
| Transcription is slow / says "CPU" | CUDA not usable by CTranslate2 | Update the NVIDIA driver; `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` (faster-whisper 1.x needs cuBLAS 12 + cuDNN 9); re-run the check script — it prints the exact CUDA error. Until fixed it falls back to CPU int8 and tells you (stage "Details" column + project warnings). |
| `Library cublas64_12.dll is not found` | CUDA 12 DLLs not on the DLL path | The app first looks for Ollama's copy (`Ollama\lib\ollama\cuda_v12`), so check Ollama is installed. Otherwise install the two pip packages above (the app registers `site-packages\nvidia\*\bin` automatically) or the CUDA 12 toolkit. |
| `ResourceExhaustedError` / CUDA out of memory | VRAM full (browser, games, another model) | Close GPU apps; `performance.mode: low`; `transcription.model: base`; `llm.model: qwen3:1.7b`; or `transcription.device: cpu`. Ollama unloads the LLM when a run ends (`keep_alive`). |
| The laptop becomes sluggish while processing | All cores busy | `performance.mode: low` (half the threads, below-normal priority, smaller LLM context) or set `performance.cpu_threads`. |
| LLM stage: "did not produce valid output" | 4B model returned malformed/invalid JSON repeatedly | Retry (seeded, so use a different `llm.seed`), lower `llm.section_max_words`, try `qwen3:8b` if you have ≥ 8 GB VRAM free, or untick "Use local LLM" for the heuristic mode (lower quality, clearly labelled). |
| No clips found | Short/quiet/no-speech video; `selection.min_score` too high; durations too strict | Lower *Min score*; widen min/max length; check the transcript at `cache/transcript.srt`. Filtered candidates are kept: choose **All** in the Review tab to see why each was filtered. |
| Captions use the wrong font / boxes | Font missing for FFmpeg's libass | Set `subtitles.font` to an installed font (e.g. `Segoe UI`, `Arial`). CJK text needs a CJK font such as `Microsoft YaHei`. |
| Preview video does not play in the browser | Browser lacks H.264 (rare) | Open the file in `projects/…/renders/` with VLC. |
| Smart crop always centres | No face detected (Haar detector only finds frontal faces) | Try `smart_crop.detector: person`, or use the blurred-background layout. |
| NVENC errors | Old driver / no NVENC | `render.hardware_encoding: never`. The app already retries once with x264 automatically. |
| UI port in use | Another instance | `python app.py --port 7870` |
| A stage says `interrupted` | App was closed mid-stage | Press **Run / resume**; finished stages are reused. |
| Want a clean start | | Delete `projects/<name>/cache` (keeps your exports) or press *Re-run everything*. |

## Where things are stored
`project.json` (state), `cache/` (transcript, audio/scene analysis, candidates; user decisions in `candidates.json`),
`renders/` (preview + final render cache), `exports/` (deliverables), `logs/`.
Nothing is written outside the project folder except temporary UI previews in your system temp folder (`clipper_ui`).
