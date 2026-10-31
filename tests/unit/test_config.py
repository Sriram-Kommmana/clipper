
import pytest

from config import load_config
from src.core.exceptions import ConfigError


def test_defaults_load_and_validate(cfg):
    assert cfg.llm.model == "qwen3:4b"
    assert cfg.transcription.model == "large-v3-turbo"
    assert cfg.selection.min_duration_s == 15 and cfg.selection.max_duration_s == 90
    assert cfg.selection.max_clips == 10
    assert abs(sum(cfg.scoring.weights.values()) - 1.0) < 1e-9
    assert cfg.render.width == 1080 and cfg.render.height == 1920


def test_overrides_deep_merge(tmp_path):
    cfg = load_config(overrides={"selection": {"max_clips": 3}, "llm": {"model": "qwen3:1.7b"}}, local_path=None)
    assert cfg.selection.max_clips == 3 and cfg.selection.min_duration_s == 15
    assert cfg.llm.model == "qwen3:1.7b"


def test_local_yaml_layer(tmp_path):
    local = tmp_path / "local.yaml"
    local.write_text("transcription:\n  model: medium\n")
    assert load_config(local_path=local).transcription.model == "medium"


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="selectoin|Extra|extra"):
        load_config(overrides={"selectoin": {"max_clips": 3}}, local_path=None)


@pytest.mark.parametrize(
    "override",
    [
        {"selection": {"min_duration_s": 70}},
        {"selection": {"target_duration_s": 5}},
        {"selection": {"category_preferences": {"nonsense": 1.0}}},
        {"render": {"width": 1081}},
        {"subtitles": {"primary_color": "red"}},
        {"llm": {"section_overlap_s": 700}},
        {"scoring": {"weights": {"bogus": 1.0}}},
    ],
)
def test_invalid_values_raise_readable_error(override):
    with pytest.raises(ConfigError) as ei:
        load_config(overrides=override, local_path=None)
    assert "Invalid configuration" in str(ei.value)


def test_missing_file_is_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(path=tmp_path / "nope.yaml")


def test_performance_mode_reduces_resources(cfg):
    low = load_config(overrides={"performance": {"mode": "low"}}, local_path=None)
    eff = low.effective()
    assert eff.llm.num_ctx <= 4096 and eff.scenes.keyframes_only and eff.performance.low_priority
    assert eff.llm.keep_alive == "0"
    assert low.llm.num_ctx == 8192, "effective() must not mutate the original"
    assert low.resolved_threads() <= cfg.resolved_threads()


def test_fingerprint_changes_only_with_section(cfg):
    a = cfg.fingerprint("transcription")
    other = load_config(overrides={"selection": {"max_clips": 2}}, local_path=None)
    assert other.fingerprint("transcription") == a
    changed = load_config(overrides={"transcription": {"model": "base"}}, local_path=None)
    assert changed.fingerprint("transcription") != a
