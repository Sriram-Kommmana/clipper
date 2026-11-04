"""Cheap, language-aware checks that put a *warning* on a candidate. They never drop or re-score a clip.

They look for the things a viewer who starts at the clip's first word would trip over, and for sponsor reads, which a
language model reading a transcript tends to rate as "clear and well structured". Word lists are short on purpose and cover
English, Hindi (Devanagari) and common Romanised Hindi; a miss just means no warning, a false hit costs the reviewer a glance.
"""
from __future__ import annotations

import re

# \w does not include Devanagari vowel signs (they are combining marks), which would cut Hindi words into pieces.
_WORDS = re.compile(r"[\w\u0900-\u097F']+", re.UNICODE)

# Openers that continue a sentence that began before the clip.
CONTINUATION = {
    "and", "but", "because", "which", "or", "plus", "whereas",
    "और", "लेकिन", "मगर", "क्योंकि", "इसलिए", "जो", "या", "परंतु",
    "aur", "lekin", "magar", "kyunki", "isliye", "jo",
}
# Openers that point at something said earlier.
POINTER = {
    "he", "she", "they", "it", "this", "that", "those", "these", "him", "her", "them",
    "वो", "वह", "यह", "ये", "उसने", "उन्होंने", "उसे", "इसने",
    "woh", "wo", "yeh", "ye", "usne", "unhone",
}
# Phrases (matched on the first words) that refer back to the rest of the video.
BACKREFERENCE = (
    "as i said", "like i said", "as i mentioned", "like i mentioned", "as we discussed", "as we saw", "going back to",
    "earlier i", "as i was saying", "remember when",
    "जैसा कि मैंने", "जैसा मैंने", "जैसा हमने", "मैंने पहले", "पहले जो",
    "jaisa maine", "jaisa humne", "maine pehle",
)
OPENING_WORDS = 8

STRONG_PROMO = (
    "sponsored by", "brought to you by", "promo code", "discount code", "coupon code", "use code", "affiliate link",
    "link in the description", "link in the bio", "link in my bio", "paid partnership",
    "स्पॉन्सर", "प्रोमो कोड", "डिस्काउंट कोड", "कूपन कोड", "लिंक डिस्क्रिप्शन", "लिंक बायो",
)
WEAK_PROMO = ("subscribe", "sign up", "download the app", "check out", "free trial", "limited time", "सब्सक्राइब", "फ्री ट्रायल", "डाउनलोड")


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORDS.findall(text or "")]


def opening_warnings(text: str) -> list[str]:
    """Reasons a cold viewer may not follow the first words of `text`."""
    words = _words(text)
    if not words:
        return []
    head = " ".join(words[:OPENING_WORDS])
    out: list[str] = []
    if any(head.startswith(p) for p in BACKREFERENCE):
        out.append("Opens by referring to something said earlier in the video; a new viewer will not know what.")
    elif words[0] in CONTINUATION:
        out.append(f"Opens on '{words[0]}', which continues a sentence that began before the clip; check the start.")
    elif words[0] in POINTER:
        out.append(f"Opens on '{words[0]}' with nothing before it that says who or what that is; check the start.")
    return out


def promo_warnings(text: str) -> list[str]:
    """Flag likely sponsor reads / self-promotion (one strong cue, or several weak ones)."""
    low = " ".join(_words(text))
    strong = [p for p in STRONG_PROMO if p in low]
    weak = [p for p in WEAK_PROMO if p in low]
    if strong or len(weak) >= 2:
        return ["Looks like a sponsor read or self-promotion; check that it is content people would want to share."]
    return []


def review_warnings(text: str) -> list[str]:
    return opening_warnings(text) + promo_warnings(text)
