import re

import pytest

from config import SubtitlesConfig
from src.core.models import SubtitleLine, SubtitleWord
from src.rendering.subtitles import (ass_color, ass_time, build_ass, build_srt, clip_lines, escape_ass, format_for_editor, group_words,
                                     lines_from_transcript, parse_editor_text, reconcile, words_for_range)
from tests.helpers import make_transcript

CFG = SubtitlesConfig()


def W(text, a, b):
    return SubtitleWord(text=text, start=a, end=b)


def test_grouping_rules():
    words = [W(t, i * 0.4, i * 0.4 + 0.3) for i, t in enumerate("one two three four five six seven".split())]
    lines = group_words(words, CFG)
    assert [l.text for l in lines] == ["one two three four", "five six seven"]
    assert all(len(l.text) <= CFG.max_chars for l in lines)
    gap = [W("a", 0, 0.3), W("b", 0.4, 0.7), W("c", 2.0, 2.3)]
    assert [l.text for l in group_words(gap, CFG)] == ["a b", "c"], "a pause starts a new caption"
    nine = [W(t, i * 0.4, i * 0.4 + 0.3) for i, t in enumerate("a b c d e f g h i.".split())]
    assert [len(l.words) for l in group_words(nine, CFG)] == [3, 3, 3], "balanced split, no orphan word"
    sent = [W("Hi.", 0, 0.3), W("there", 0.4, 0.7), W("friend", 0.8, 1.1)]
    assert [l.text for l in group_words(sent, CFG)] == ["Hi.", "there friend"]
    long = [W("extraordinarily", 0, .4), W("complicated", .5, .9), W("pronunciation", 1.0, 1.4)]
    assert all(len(l.text) <= CFG.max_chars or len(l.words) == 1 for l in group_words(long, CFG))
    assert group_words([], CFG) == []


def test_words_for_range_with_and_without_word_timing():
    t = make_transcript([(0, "alpha beta gamma delta."), (10, "epsilon zeta eta theta.")])
    ws = words_for_range(t, 9.0, 12.0)
    assert [w.text for w in ws][:2] == ["epsilon", "zeta"]
    seg_only = t.model_copy(deep=True)
    for s in seg_only.segments:
        s.words = []
    ws2 = words_for_range(seg_only, 0, 100)
    assert len(ws2) == 8 and ws2[0].start == 0 and ws2[-1].end == pytest.approx(seg_only.segments[1].end)


def test_ass_structure_and_karaoke_events():
    t = make_transcript([(10.0, "Here we go again friend.")])
    lines = lines_from_transcript(t, 9.5, 14.0, CFG)
    ass = build_ass(lines, 9.5, 14.0, 1080, 1920, CFG)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    style = re.search(r"^Style: Default,(.*)$", ass, re.M).group(1).split(",")
    assert style[0] == "Arial" and style[1] == "78" and style[2] == "&H00FFFFFF"
    assert style[17] == "2" and style[20] == str(round(0.20 * 1920)), (style[17], style[20])
    events = [l for l in ass.splitlines() if l.startswith("Dialogue")]
    first_line_words = lines[0].text.split()
    assert len(events) >= len(first_line_words)
    assert "&H0035E1FF" in events[0], "active word uses the highlight colour (BGR order)"
    assert events[0].split(",")[1] == ass_time(10.0 - 9.5), "event times are shifted to clip time"
    # events within one caption line never overlap
    def secs(s):
        h, m, rest = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)

    spans = [(secs(e.split(",")[1]), secs(e.split(",")[2])) for e in events]
    assert all(a < b for a, b in spans)
    for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
        assert b1 <= a2 + 0.011


def test_plain_mode_and_options():
    t = make_transcript([(0.0, "Short plain caption here.")])
    cfg = SubtitlesConfig(mode="plain", uppercase=True, position="top", font="Verdana", font_size=60)
    ass = build_ass(lines_from_transcript(t, 0, 5, cfg), 0, 5, 540, 960, cfg)
    ev = [l for l in ass.splitlines() if l.startswith("Dialogue")]
    assert len(ev) == 1 and "SHORT PLAIN CAPTION HERE." in ev[0] and "\\1c" not in ev[0]
    style = re.search(r"^Style: Default,(.*)$", ass, re.M).group(1).split(",")
    assert style[0] == "Verdana" and style[1] == "30" and style[17] == "8"   # font scales with output size


def test_escaping_unicode_and_punctuation():
    assert escape_ass("a{b}c\\d\ne") == "a(b)c/d e"
    t = make_transcript([(0.0, "Привет, мир! 你好 — «quotes» {braces} ñandú.")])
    ass = build_ass(lines_from_transcript(t, 0, 10, CFG), 0, 10, 1080, 1920, CFG)
    assert "Привет," in ass and "你好" in ass and "«quotes»" in ass and "{braces}" not in ass
    assert ass_color("#FFE135") == "&H0035E1FF" and ass_time(3661.239) == "1:01:01.24"
    assert build_ass([], 0, 5, 100, 100, CFG).rstrip().endswith("Text")


def test_srt_relative_times_and_clipping():
    lines = [SubtitleLine(start=10, end=12, text="inside"), SubtitleLine(start=40, end=42, text="outside")]
    srt = build_srt(lines, 9.0, 20.0)
    assert "00:00:01,000 --> 00:00:03,000\ninside" in srt and "outside" not in srt


def test_clip_lines_trims_words_when_boundaries_move():
    ln = SubtitleLine(start=5, end=8, text="a b c", words=[W("a", 5, 6), W("b", 6, 7), W("c", 7, 8)])
    out = clip_lines([ln], 6.5, 20)
    assert len(out) == 1 and [w.text for w in out[0].words] == ["b", "c"] and out[0].start == 6.5
    assert clip_lines([ln], 9, 20) == []


def test_editor_roundtrip_preserves_word_timing_when_text_edited_in_place():
    t = make_transcript([(10.0, "Here we go again friend.")])
    lines = lines_from_transcript(t, 9.5, 14.0, CFG)
    text = format_for_editor(lines, 9.5)
    assert text.splitlines()[0].startswith("0.50-")
    specs = parse_editor_text(text, 9.5, 14.0)
    again = reconcile(lines, specs)
    assert [l.text for l in again] == [l.text for l in lines]
    assert [(w.start, w.end) for w in again[0].words][1] == pytest.approx((lines[0].words[1].start, lines[0].words[1].end), abs=0.011)
    edited = text.replace("Here we", "There we").replace("go", "run")
    out = reconcile(lines, parse_editor_text(edited, 9.5, 14.0))
    assert out[0].text.startswith("There we run")
    assert out[0].words[1].start == pytest.approx(lines[0].words[1].start, abs=0.011), "same word count keeps timings"
    # different word count -> re-distributed evenly inside the line span
    longer = text.replace("Here we go", "Here we all go")
    out = reconcile(lines, parse_editor_text(longer, 9.5, 14.0))
    ws = out[0].words
    assert len(ws) == 4 and ws[0].start == pytest.approx(out[0].start) and ws[-1].end == pytest.approx(out[0].end)
    # deleting a line (empty text) removes it
    assert reconcile(lines, parse_editor_text("0.5-1.0 | ", 9.5, 14.0)) == []


@pytest.mark.parametrize("bad", ["no separator here", "2.0-1.0 | backwards", "99-100 | way after the clip", "abc-1 | x"])
def test_editor_parse_errors(bad):
    with pytest.raises(ValueError, match="Line 1"):
        parse_editor_text(bad, 0.0, 10.0)
