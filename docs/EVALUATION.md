# Measuring clip quality

Everything else in this project can be tested without a model; **whether the clips are good cannot.** `python app.py eval`
compares a project's clips with moments you marked by hand, so that a change (a prompt, a weight, `llm.audio_hints`, a different
model or Whisper size) can be judged by numbers instead of by feel.

## 1. Mark the good moments (10 minutes per video)
Watch a video you know and write down the start and end (in seconds of the *source* video) of every moment you would be happy to
post. Be generous with the edges; a clip that covers most of a moment still counts.

```json
{"moments": [
  {"start": 312.0, "end": 361.5, "note": "the broken guitar story"},
  {"start": 905.0, "end": 951.0}
]}
```
Five to ten videos with five to fifteen moments each is enough to see a real difference. Keep the labels out of git if the videos are private.

## 2. Run the analysis, then evaluate
```bash
python app.py run my_video.mp4                       # creates projects/<video>-<id>/
python app.py eval projects/<video>-<id> --labels labels/my_video.json
```
```
  moments found by any candidate    80.0%   <- the ceiling: a moment never proposed cannot be recovered later
  moments found among selected      60.0%
  selected clips that are real      50.0%
  mean reciprocal rank              0.55   (1.00 = a matching clip is always ranked first)
  median start / end error          1.4s / 2.2s
  score gap (matching - others)     +8.5 points   (should be clearly positive)
  MISSED 905-951s
```

| Number | Meaning | What to do if it is low |
|---|---|---|
| found by any candidate | Did generation propose the moment at all? | Prompts, model size, section settings. Nothing downstream can fix it. |
| found among selected | Did selection keep it? | Scoring weights, `selection.min_score`, `max_clips`, `min_separation_s`. |
| selected clips that are real | How much of what you would review is worth it? | Scoring, stricter `min_score`. |
| mean reciprocal rank / score gap | Are good clips ranked above bad ones? | Scoring prompt and weights. |
| start / end error | How close are the cuts? | Boundary settings (`boundaries.*`). |

## 3. A/B a setting
Run the same video twice with different settings into two project folders (`--out`), then compare:
```bash
python app.py run my_video.mp4 --out projects_a
python app.py run my_video.mp4 --out projects_b            # with configs/local.yaml changed, e.g. llm: {audio_hints: true}
python app.py eval projects_a/<id> --labels labels/my_video.json --compare projects_b/<id>
```
`--iou` sets how much overlap counts as "found" (default 0.4); `--json` prints the numbers for scripts.

## Limits
A handful of labelled videos is a smoke detector, not a benchmark: differences of a few points are noise. Labels reflect *your*
taste. The numbers say nothing about views or engagement.
