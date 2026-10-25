# Optional browser smoke test (not part of `pytest`)

Drives the real Gradio UI in Chromium: open a project, select a clip, render a preview, approve, export, and check
the error panel for a missing Whisper model. Needs Node + Playwright (`npm i -g playwright`).

```bash
S=/tmp/clipper_ui; mkdir -p $S
python tests/ui/seed_project.py $S                         # builds a project with a fake Whisper/LLM + demo.mp4
ffmpeg -y -f lavfi -i testsrc2=s=640x360:r=25:d=20 -f lavfi -i sine=f=300:d=20 -c:v libx264 -pix_fmt yuv420p -c:a aac $S/demo2.mp4
python app.py --projects-dir $S/uiproj --no-browser --port 7861 &
NODE_PATH=$(npm root -g) node tests/ui/ui_smoke.js $S
```
Last run: 2026-10-25 in the dev sandbox, all steps passed (see DEVELOPMENT_LOG.md).
