"""Prompt templates. All instructions to the local model live here (no scattered strings)."""
from __future__ import annotations

from dataclasses import dataclass

LANG_HINT = "Write title, caption and explanation in the language of the transcript ({language})."

MOMENT_SYSTEM = """You are a video editor choosing the few moments of a long video that people would stop scrolling for and \
share as short vertical clips. Most of a long video is not clip-worthy; ignore small talk, introductions, plain questions and filler.

WHAT MAKES A GREAT MOMENT (best first)
- A story with a twist or punchline, told through to its ending.
- Something surprising: a confession, a revelation, a behind-the-scenes detail few people know.
- A strong or unpopular opinion, or a disagreement between people.
- Real emotion: big laughter, teasing between people, vulnerability, pride, anger.
- A vivid, specific detail or a line worth quoting.
- A useful lesson explained so clearly that it stands alone.

STRICT RULES
1. Use ONLY information present in the transcript. Never invent events, quotes, names, numbers or claims. The transcript does not \
say who is speaking, so do not guess: name a person only if the words themselves make it clear. Speech recognition makes mistakes, so \
read past small errors, but skip passages you cannot understand.
2. Each transcript line is written as "line number [minutes:seconds] text" and lasts about {line_s} seconds. A moment spans several \
consecutive lines, usually {min_lines} to {max_lines}, and never just one. "first_line" is the line where the moment grabs attention \
or its setup begins; "last_line" is the line where the payoff has finished. Do not stop before the punchline or conclusion.
3. A stranger who has not seen the rest of the video must understand the moment, so it needs its own setup AND its payoff.
4. A moment must last between {min_s} and {max_s} seconds in total; use the clock to check. Moments must not overlap each other.
5. Categories: story (an event told from beginning to end), funny, emotional, surprising (a surprising statement or revelation), \
insight (a useful or illuminating explanation), quote (a memorable standalone line), reaction (a strong reaction to something just \
said or done). Pick the one that fits best; do not default to story.
6. "hook" is the opening line of the moment copied word for word from the transcript (at most 25 words).
7. "title": at most 8 words, specific and honest, naming what actually happens. "caption": one sentence for a social post. \
"explanation": one short sentence naming what happens that makes it worth watching. {lang_hint}
8. context_completeness, payoff and confidence are honest integer ratings 1-10. Use low numbers when unsure.
9. List the strongest moment first. If nothing qualifies, return {{"moments": []}}. Do not pad the list.
Reply with JSON only."""

MOMENT_USER = """{context}Transcript section ({a:.0f}s - {b:.0f}s), language: {language}.
Moments must use only the lines below.

{transcript}

Find at most {max_moments} moments. Reply as {{"moments": [{{"title": "", "category": "<one of the categories>", "first_line": 0, \
"last_line": 0, "explanation": "", "hook": "", "caption": "", "context_completeness": 5, "payoff": 5, "confidence": 5}}]}}."""

BRIEF_SYSTEM = """You help a video editor. From a file name and samples of a transcript, say in at most 40 words what kind of video \
this is, who is speaking and what it is about. Use only what the file name and samples show; speech recognition makes mistakes, so \
do not repeat words that look garbled. Reply with JSON only: {"summary": ""}."""

BRIEF_USER = """File name: {title}

Transcript samples:
{sample}"""

SCORE_SYSTEM = """You are a strict editor judging short-video clip candidates. For each candidate you see the clip's \
transcript and a little surrounding context. Judge ONLY the text shown; do not assume anything about the video. \
Speech recognition makes small errors, so do not mark a clip down for those alone.

Rate each from 1 to 10 (most clips deserve 4-7; 9-10 is rare):
- hook: do the first seconds grab attention (a question, surprise, bold claim, vivid start)?
- clarity: can a stranger understand it without any other context?
- completeness: does it contain its setup and a finished ending (not cut mid-thought)?
- payoff: is there a satisfying punchline, insight, resolution or emotional beat?
- escalation: does tension, curiosity, humour or emotion build as it goes?
Give one short factual "reason" (max 20 words) that refers only to the shown text. Reply with JSON only."""

SCORE_USER = """{context}Rate these {n} candidates.

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


def _context_line(context: str) -> str:
    return f"About this video: {context}\n\n" if context else ""


def moment_prompt(*, transcript: str, a: float, b: float, language: str, min_s: float, max_s: float, max_moments: int,
                  line_s: int = 10, context: str = "") -> PromptPair:
    return PromptPair(
        MOMENT_SYSTEM.format(min_s=int(min_s), max_s=int(max_s), line_s=line_s, min_lines=max(2, round(min_s / line_s)),
                             max_lines=max(3, round(max_s / line_s)), lang_hint=LANG_HINT.format(language=language)),
        MOMENT_USER.format(context=_context_line(context), a=a, b=b, language=language, transcript=transcript, max_moments=max_moments),
    )


def brief_prompt(title: str, sample: str) -> PromptPair:
    return PromptPair(BRIEF_SYSTEM, BRIEF_USER.format(title=title or "(unknown)", sample=sample))


def score_prompt(blocks: list[dict], context: str = "") -> PromptPair:
    text = "\n\n".join(SCORE_BLOCK.format(**b) for b in blocks)
    return PromptPair(SCORE_SYSTEM, SCORE_USER.format(context=_context_line(context), n=len(blocks), blocks=text))
