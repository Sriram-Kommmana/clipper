"""Prompt templates. All instructions to the local model live here (no scattered strings)."""
from __future__ import annotations

from dataclasses import dataclass

LANG_HINT = "Write title, caption and explanation in the language of the transcript ({language})."

MOMENT_SYSTEM = """You are an assistant to a video editor. You read a timestamped transcript and find \
self-contained moments that work as standalone short vertical clips.

STRICT RULES
1. Use ONLY information present in the transcript. Never invent events, quotes, names, numbers or claims.
2. Timestamps are seconds from the video start and must be copied from the transcript markers [start-end]. \
Start at the beginning of the sentence where the setup begins; end right after the payoff. Never start or end mid-sentence.
3. A moment must make sense without the rest of the video and must contain its own setup AND its payoff.
4. Moment length must be between {min_s} and {max_s} seconds. Moments must not overlap each other.
5. Categories: story (a narrated event with beginning and end), funny, emotional, surprising (surprising statement or \
revelation), insight (useful or illuminating explanation), quote (a memorable standalone line), reaction (a strong reaction \
to something just said or done).
6. "hook" is the opening line of the moment copied word for word from the transcript (at most 25 words).
7. "title": at most 8 words, descriptive and honest. "caption": one sentence for a social post. "explanation": one short \
sentence on why this moment works. {lang_hint}
8. context_completeness, payoff and confidence are honest integer ratings 1-10. Use low numbers when unsure.
9. If nothing qualifies, return {{"moments": []}}. Do not pad the list.
Reply with JSON only."""

MOMENT_USER = """Transcript section ({a:.0f}s - {b:.0f}s), language: {language}.
Moments must lie fully inside this section.

{transcript}

Find at most {max_moments} moments. Reply as {{"moments": [{{"start": 0.0, "end": 0.0, "category": "story", "title": "", \
"explanation": "", "hook": "", "caption": "", "context_completeness": 5, "payoff": 5, "confidence": 5}}]}}."""

SCORE_SYSTEM = """You are a strict editor judging short-video clip candidates. For each candidate you see the clip's \
transcript and a little surrounding context. Judge ONLY the text shown; do not assume anything about the video.

Rate each from 1 to 10 (most clips deserve 4-7; 9-10 is rare):
- hook: do the first seconds grab attention (a question, surprise, bold claim, vivid start)?
- clarity: can a stranger understand it without any other context?
- completeness: does it contain its setup and a finished ending (not cut mid-thought)?
- payoff: is there a satisfying punchline, insight, resolution or emotional beat?
- escalation: does tension, curiosity, humour or emotion build as it goes?
Give one short factual "reason" (max 20 words) that refers only to the shown text. Reply with JSON only."""

SCORE_USER = """Rate these {n} candidates.

{blocks}

Reply as {{"scores": [{{"id": "<candidate id>", "hook": 5, "clarity": 5, "completeness": 5, "payoff": 5, "escalation": 5, \
"reason": ""}}]}} with exactly one entry per candidate id."""

SCORE_BLOCK = """### Candidate {id} ({duration:.0f}s, category: {category})
[context before] {before}
[CLIP] {text}
[context after] {after}"""


@dataclass(frozen=True)
class PromptPair:
    system: str
    user: str


def moment_prompt(*, transcript: str, a: float, b: float, language: str, min_s: float, max_s: float, max_moments: int) -> PromptPair:
    return PromptPair(
        MOMENT_SYSTEM.format(min_s=int(min_s), max_s=int(max_s), lang_hint=LANG_HINT.format(language=language)),
        MOMENT_USER.format(a=a, b=b, language=language, transcript=transcript, max_moments=max_moments),
    )


def score_prompt(blocks: list[dict]) -> PromptPair:
    text = "\n\n".join(SCORE_BLOCK.format(**b) for b in blocks)
    return PromptPair(SCORE_SYSTEM, SCORE_USER.format(n=len(blocks), blocks=text))
