import json

import pytest

import app
from src.core.project_manager import create_or_open_project
from src.evaluation.metrics import LabelError, Span, compare, evaluate, format_report, iou, load_labels
from tests.helpers import cand

LABELS = [Span(100, 150), Span(300, 340), Span(900, 960)]


def S(a, b, score=50.0):
    return Span(a, b, score)


def test_iou():
    assert iou(S(0, 10), S(5, 15)) == pytest.approx(5 / 15)
    assert iou(S(0, 10), S(20, 30)) == 0.0 and iou(S(0, 10), S(0, 10)) == 1.0


def test_perfect_and_empty():
    r = evaluate(LABELS, [S(100, 150, 90), S(300, 340, 80), S(900, 960, 70)], [S(100, 150), S(300, 340), S(900, 960)])
    assert r.recall_candidates == r.recall_selected == r.precision_selected == 1.0
    assert r.mean_reciprocal_rank == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
    assert r.start_error_s == 0 and r.missed == []
    e = evaluate(LABELS, [], [])
    assert e.recall_candidates == 0 and e.precision_selected == 0 and e.start_error_s is None and len(e.missed) == 3


def test_candidates_vs_selected_and_boundary_error():
    cands = [S(95, 148, 80), S(310, 345, 40), S(500, 540, 90), S(20, 40, 10)]        # the third is a false positive ranked first
    r = evaluate(LABELS, cands, [S(95, 148), S(500, 540)])
    assert r.recall_candidates == pytest.approx(2 / 3), "the 900-960 moment was never proposed"
    assert r.recall_selected == pytest.approx(1 / 3) and r.precision_selected == 0.5
    assert r.start_error_s == pytest.approx(7.5) and r.end_error_s == pytest.approx(3.5)
    assert [m["start"] for m in r.missed] == [900]
    assert r.score_gap == pytest.approx(10.0), "matching clips (80, 40) average 60; the others (90, 10) average 50"
    # ranks: label 100 -> clip at 95 is 2nd (score 80 < 90) ; label 300 -> 3rd ; label 900 -> none
    assert r.mean_reciprocal_rank == pytest.approx((1 / 2 + 1 / 3 + 0) / 3)


def test_threshold_matters():
    near = [S(100, 112, 50)]                                       # captures only a quarter of a 50 s moment
    assert evaluate([Span(100, 150)], near, near, iou_min=0.4).recall_candidates == 0
    assert evaluate([Span(100, 150)], near, near, iou_min=0.2).recall_candidates == 1


def test_load_labels_validation(tmp_path):
    def write(obj):
        p = tmp_path / "l.json"
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")
        return p

    assert [(s.start, s.end) for s in load_labels(write({"moments": [{"start": 1, "end": 5, "note": "x"}]}))] == [(1.0, 5.0)]
    assert len(load_labels(write([{"start": 1, "end": 5}]))) == 1, "a bare list is accepted"
    for bad in ("{oops", {"moments": []}, {"moments": [{"start": 5, "end": 1}]}, {"moments": [{"start": "a", "end": 3}]},
                {"moments": [{"end": 3}]}, {"moments": [{"start": -1, "end": 3}]}, 5):
        with pytest.raises(LabelError):
            load_labels(write(bad))
    with pytest.raises(LabelError):
        load_labels(tmp_path / "missing.json")


def test_report_text_and_compare():
    a = evaluate(LABELS, [S(100, 150, 90)], [S(100, 150)])
    b = evaluate(LABELS, [S(100, 150, 90), S(300, 340, 80)], [S(100, 150), S(300, 340)])
    text = format_report(b, "demo")
    assert "demo" in text and "66.7%" in text and "MISSED 900-960s" in text
    cmp = compare(a, b)
    assert "+33.3pp" in cmp


def test_eval_cli_on_a_real_project(sample_info, tmp_path, capsys):
    store = create_or_open_project(sample_info, tmp_path / "projects")
    cs = [cand("a", 100, 150, score=80), cand("b", 300, 340, score=60), cand("c", 500, 540, score=70)]
    cs[0] = cs[0].model_copy(update={"selected": True})
    cs[2] = cs[2].model_copy(update={"filter_reason": "invalid: too short"})        # invalid clips are not candidates
    store.save_candidates(cs)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"moments": [{"start": 100, "end": 150}, {"start": 300, "end": 340}, {"start": 500, "end": 540}]}))
    assert app.main(["eval", str(store.root), "--labels", str(labels)]) == 0
    out = capsys.readouterr().out
    assert "3 labelled moments, 2 candidates, 1 selected" in out and "66.7%" in out and "MISSED 500-540s" in out
    assert app.main(["eval", str(store.root), "--labels", str(labels), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert next(iter(data.values()))["recall_candidates"] == pytest.approx(2 / 3)
    assert app.main(["eval", str(store.root), "--labels", str(tmp_path / "nope.json")]) == 2
    assert app.main(["eval", str(store.root), "--labels", str(labels), "--compare", str(store.root)]) == 0
    assert "B - A" in capsys.readouterr().out
