"""Opt-in smoke test against a real local Ollama.   Run:  pytest -m ollama tests/integration/test_ollama_smoke.py"""
import json

import pytest

from config import load_config
from src.core.schemas import MomentList
from src.llm.ollama_client import OllamaClient
from src.llm.output_parser import structured_call
from src.llm.prompts import moment_prompt

pytestmark = pytest.mark.ollama


def test_real_model_returns_valid_schema():
    cfg = load_config(local_path=None)
    client = OllamaClient(cfg.llm)
    client.ensure_ready()
    transcript = "\n".join(f"{i}: (5s) {t}" for i, t in enumerate([
        "So last summer I decided to climb the mountain behind my house.", "I had no map and only one bottle of water.",
        "Halfway up a goat blocked the path and refused to move.", "I tried everything, shouting, clapping, even singing.",
        "Finally I offered it my sandwich and it walked right off.", "At the top I realised the goat had eaten my map too.",
        "That is the story of how a goat taught me to pack better."]))
    p = moment_prompt(transcript=transcript, a=0, b=35, language="en", min_s=10, max_s=40, max_moments=3)
    result, attempts = structured_call(client, p.system, p.user, MomentList, retries=2)
    print(json.dumps(result.model_dump(), indent=2), "attempts:", attempts)
    for m in result.moments:
        assert 0 <= m.first_line < m.last_line <= 6
    client.unload()
