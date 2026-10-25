import json
import threading

import pytest

from src.core.exceptions import ProjectError
from src.core.models import Candidate
from src.core.project_manager import create_or_open_project, list_projects, slugify
from src.storage.cache import StageCache, make_key
from src.storage.project_store import atomic_write_json, read_json


def test_atomic_write_leaves_no_temp_and_survives_failure(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_json(p, {"a": 1})
    assert read_json(p) == {"a": 1}
    with pytest.raises(TypeError):
        atomic_write_json(p, {"a": object()})
    assert read_json(p) == {"a": 1}, "failed write must not corrupt the previous file"
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]


def test_cache_key_invalidation(tmp_path):
    c = StageCache(tmp_path)
    k1 = make_key("fp", {"model": "small"})
    k2 = make_key("fp", {"model": "medium"})
    assert k1 != k2 and make_key("fp", {"model": "small"}) == k1
    c.save("transcript", k1, {"x": 1})
    assert c.load("transcript", k1) == {"x": 1}
    assert c.load("transcript", k2) is None
    assert c.load("missing", k1) is None
    c.invalidate("transcript")
    assert c.load("transcript", k1) is None


def test_corrupt_cache_is_a_miss(tmp_path):
    c = StageCache(tmp_path)
    c.path("a").write_text("{not json")
    assert c.load("a", "k") is None
    c.path("b").write_text(json.dumps({"key": "k", "schema": 999, "data": 1}))
    assert c.load("b", "k") is None, "schema mismatch is a miss"


def test_project_roundtrip_and_resume(tmp_path, sample_info):
    store = create_or_open_project(sample_info, tmp_path)
    assert store.state_path.exists() and store.cache_dir.is_dir()
    store.update_stage("transcribe", status="done", cache_key="abc")
    again = create_or_open_project(sample_info, tmp_path)
    assert again.root == store.root
    assert again.state.stages["transcribe"].status == "done", "re-opening must keep completed work"
    assert [n for n, _ in list_projects(tmp_path)] == [store.root.name]


def test_corrupt_project_file_raises_project_error(tmp_path, sample_info):
    store = create_or_open_project(sample_info, tmp_path)
    store.state_path.write_text("garbage")
    with pytest.raises(ProjectError):
        create_or_open_project(sample_info, tmp_path)


def _cand(i, **kw):
    return Candidate(id=f"c{i}", start=10 * i, end=10 * i + 20, category="story", title=f"t{i}", **kw)


def test_candidate_updates_are_thread_safe(tmp_path, sample_info):
    store = create_or_open_project(sample_info, tmp_path)
    store.save_candidates([_cand(i) for i in range(5)])

    def work(i):
        store.update_candidate(f"c{i}", status="approved")

    ts = [threading.Thread(target=work, args=(i,)) for i in range(5)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert all(c.status == "approved" for c in store.load_candidates())
    with pytest.raises(ProjectError):
        store.update_candidate("nope", status="approved")


def test_slugify():
    assert slugify("Mein Video (Final) é!.mp4") == "mein-video-final-e-mp4"
    assert slugify("???") == "video"
