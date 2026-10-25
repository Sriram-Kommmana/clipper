import json
import threading

import pytest

from config import LLMConfig
from src.core.exceptions import ConfigError, JobCancelled, LLMModelMissingError, LLMOutputError, LLMUnavailableError
from src.core.schemas import MomentList, RawMoment, ScoreList, json_schema_for
from src.llm.ollama_client import OllamaClient, assert_local_host
from src.llm.output_parser import extract_json, parse_model, strip_thinking, structured_call
from src.llm.prompts import moment_prompt, score_prompt
from tests.fakes import FakeOllama, ScriptedLLM

GOOD = {"moments": [{"start": 1, "end": 30, "category": "story", "title": "A title", "explanation": "x", "hook": "h",
                     "caption": "c", "context_completeness": 8, "payoff": 7, "confidence": 6}]}


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure! Here you go:\n```json\n{"a": [1, 2]}\n```\nHope it helps') == {"a": [1, 2]}
    assert extract_json('<think>hmm {"x": 0}</think>\n{"a": 2}') == {"a": 2}
    assert extract_json('prefix text [1, 2, 3] suffix') == [1, 2, 3]
    assert strip_thinking("<think>unclosed reasoning") == ""
    for bad in ("", "no json here", "{broken"):
        with pytest.raises(ValueError):
            extract_json(bad)


def test_parse_model_accepts_bare_list_and_validates():
    m = parse_model(json.dumps(GOOD["moments"]), MomentList)
    assert len(m.moments) == 1 and m.moments[0].category == "story"
    with pytest.raises(Exception):
        parse_model(json.dumps({"moments": [{**GOOD["moments"][0], "category": "cooking"}]}), MomentList)
    with pytest.raises(Exception):
        parse_model(json.dumps({"moments": [{**GOOD["moments"][0], "payoff": 11}]}), MomentList)
    t = RawMoment(**{**GOOD["moments"][0], "title": "  spaced \n out  title "})
    assert t.title == "spaced out title"


def test_json_schema_is_flat():
    s = json.dumps(json_schema_for(MomentList))
    assert "$ref" not in s and "$defs" not in s and '"enum"' in s
    assert "scores" in json_schema_for(ScoreList)["properties"]


def test_structured_call_retries_then_succeeds():
    llm = ScriptedLLM(["not json", '{"moments": [{"start": 1}]}', json.dumps(GOOD)])
    obj, attempts = structured_call(llm, "sys", "prompt", MomentList, retries=2)
    assert attempts == 3 and len(obj.moments) == 1
    assert "rejected" in llm.calls[1][1] and "rejected" in llm.calls[2][1]


def test_structured_call_gives_up():
    llm = ScriptedLLM(["garbage"])
    with pytest.raises(LLMOutputError) as ei:
        structured_call(llm, "s", "p", MomentList, retries=1)
    assert len(llm.calls) == 2 and "garbage" in ei.value.details


def test_prompts_contain_guardrails():
    p = moment_prompt(transcript="[0.0-1.0] hi", a=0, b=600, language="en", min_s=15, max_s=60, max_moments=5)
    assert "Never invent" in p.system and "15 and 60" in p.system and "[0.0-1.0] hi" in p.user
    sp = score_prompt([{"id": "c1", "duration": 30, "category": "story", "before": "", "text": "t", "after": ""}])
    assert "c1" in sp.user and "ONLY the text shown" in sp.system


def test_local_host_guard():
    for ok in ("http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"):
        assert_local_host(ok, False)
    with pytest.raises(ConfigError, match="not a local address"):
        assert_local_host("http://example.com:11434", False)
    with pytest.raises(ConfigError):
        assert_local_host("http://192.168.1.5:11434", False)
    assert_local_host("http://192.168.1.5:11434", True)


def _client(url, **kw):
    return OllamaClient(LLMConfig(host=url, **kw))


def test_client_streams_and_sends_expected_payload():
    def handler(p):
        return 200, [{"message": {"content": '{"mom'}}, {"message": {"content": 'ents": []}'}, "done": True, "done_reason": "stop"}]

    with FakeOllama(chat_handler=handler) as srv:
        c = _client(srv.url)
        c.ensure_ready()
        out = c.generate_json("system", "user", {"type": "object"})
        assert out == '{"moments": []}'
        p = srv.requests[0]
        assert p["model"] == "qwen3:4b" and p["format"] == {"type": "object"} and p["think"] is False
        assert p["options"]["num_ctx"] == 8192 and p["options"]["seed"] == 7 and p["keep_alive"] == "5m"
        assert p["messages"][0]["content"].endswith("/no_think")
        c.unload()
        assert srv.requests[-1]["keep_alive"] == 0


def test_client_think_fallback_for_old_ollama():
    def handler(p):
        if "think" in p:
            return 400, {"error": "json: unknown field \"think\""}
        return 200, [{"message": {"content": "{}"}, "done": True}]

    with FakeOllama(chat_handler=handler) as srv:
        c = _client(srv.url)
        assert c.generate_json("s", "u", {}) == "{}"
        assert c._send_think is False and "think" not in srv.requests[-1]


def test_client_error_mapping():
    with FakeOllama(models=["other:1b"]) as srv:
        with pytest.raises(LLMModelMissingError) as ei:
            _client(srv.url).ensure_ready()
        assert "ollama pull qwen3:4b" in ei.value.recovery
    with FakeOllama(chat_handler=lambda p: (404, {"error": "model 'qwen3:4b' not found"})) as srv:
        with pytest.raises(LLMModelMissingError):
            _client(srv.url).generate_json("s", "u", {})
    with FakeOllama(chat_handler=lambda p: (500, {"error": "CUDA out of memory"})) as srv:
        from src.core.exceptions import ResourceExhaustedError

        with pytest.raises(ResourceExhaustedError):
            _client(srv.url).generate_json("s", "u", {})
    with pytest.raises(LLMUnavailableError, match="Cannot"):
        _client("http://127.0.0.1:9").list_models()


def test_client_cancel():
    ev = threading.Event()

    def handler(p):
        ev.set()
        return 200, [{"message": {"content": "a"}}] * 5 + [{"message": {"content": ""}, "done": True}]

    with FakeOllama(chat_handler=handler) as srv:
        with pytest.raises(JobCancelled):
            _client(srv.url).generate_json("s", "u", {}, cancel=ev)
