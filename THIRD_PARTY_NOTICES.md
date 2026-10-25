# Third-party notices

This project's own code is unlicensed so far (the owner has not chosen a licence yet — decide before distributing).
**No code, prompts or text were copied from the reference repository** `LanderIglesias/Clipper`; it has no LICENSE file
(see `docs/REFERENCE_ANALYSIS.md`). Nothing below is bundled in this repository: users install everything themselves.

How to read the "Verified" column: **metadata** = read from the installed package's own metadata in the dev sandbox on
2026-10-25; **docs** = from the publisher's published licence as I know it — I could *not* re-fetch it here (the sandbox blocks
huggingface.co / ollama.com), so confirm it on your machine before relying on it commercially.

## Python packages (installed via `pip install -r requirements.txt`)
| Package | Licence | Verified | Commercial notes |
|---|---|---|---|
| faster-whisper 1.2.1 | MIT | metadata | permissive |
| ctranslate2 4.8.3 | MIT | metadata | permissive |
| librosa 1.0.0 | ISC | metadata | permissive |
| numpy 2.5 | BSD-3-Clause (+ bundled 0BSD/MIT/Zlib/CC0 parts) | metadata | |
| scipy 1.18 | BSD-3-Clause | metadata | |
| soundfile 0.14 | BSD-3-Clause | metadata | wheels bundle libsndfile (LGPL-2.1): dynamic linking; keep the notice if you redistribute wheels |
| opencv-python(-headless) 5.0 | Apache-2.0 (packaging MIT) | metadata | wheels also bundle an LGPL FFmpeg; Haar cascade XML uses the Intel/OpenCV BSD-style licence (docs) |
| gradio 6.30 | Apache-2.0 | metadata | pulls many transitive packages — run `pip-licenses` before shipping a product |
| pydantic 2.14 | MIT | metadata | |
| PyYAML 6.0 | MIT | metadata | |
| requests 2.34 | Apache-2.0 | metadata | |
| psutil 7.2 | BSD-3-Clause | metadata | |
| pytest 9.1 (dev) | MIT | metadata | |
| NVIDIA cuBLAS / cuDNN pip wheels (optional) | NVIDIA proprietary licence | docs | free to use, **redistribution restricted** — never bundle them, let users `pip install` |

Deliberately **not** used: `ultralytics` / YOLO (AGPL-3.0 or paid licence), any cloud AI SDK (`anthropic`, `openai`, …).
`tests/unit/test_robustness.py` fails if one of those is imported or listed in requirements.

## Model weights (downloaded once by the user)
| Model | Licence | Verified | Notes |
|---|---|---|---|
| Whisper (OpenAI) weights, as converted by Systran for faster-whisper (`Systran/faster-whisper-*`) | MIT | docs | commercial use allowed; Whisper can hallucinate text on silence — hence the VAD + no-speech filter |
| Qwen3 (`qwen3:1.7b`, `qwen3:4b`, `qwen3:8b`) via Ollama | Apache-2.0 | docs | check the licence of the exact tag you pull: `ollama show qwen3:4b --license`. Model output is not guaranteed accurate; this app verifies excerpts/hooks against the transcript |
| Ollama (runtime) | MIT | docs | separate install, not bundled |

## Tools
| Tool | Licence | Notes |
|---|---|---|
| FFmpeg / FFprobe | LGPL-2.1+ core; **GPL-2/3 when built with libx264** (typical Windows "essentials/full" builds, e.g. gyan.dev, BtbN) | Not bundled. If you ever *ship* FFmpeg inside a product you must follow the GPL/LGPL terms (offer source, licence text). Using it as an external program, as here, is the simple case |
| x264 (H.264 encoder) | GPL-2.0+ (commercial licence available) | H.264 and AAC are also covered by patent pools; creating/streaming content is generally fine, but **shipping an encoder in a product** needs legal review |
| NVENC | part of NVIDIA drivers (proprietary) | used only through FFmpeg when a test encode succeeds |
| Arial (default caption font) | proprietary (Microsoft) | read from the user's system, never redistributed. Burned-in caption pixels are normal output, but if you publish at scale consider an open font (OFL, e.g. Noto Sans / Open Sans) via `subtitles.font` |

## Content rights
The app does not download media and cannot grant rights to it. Only clip footage you are authorised to edit and republish.
