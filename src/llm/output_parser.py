"""Tolerant extraction of JSON from model text + strict schema validation + bounded retries."""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.core.exceptions import LLMOutputError
from src.core.schemas import json_schema_for
from src.llm.ollama_client import LLMClient

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def strip_thinking(text: str) -> str:
    text = _THINK.sub("", text)
    if "<think>" in text.lower():  # unclosed block (truncated generation)
        text = re.split(r"<think>", text, flags=re.I)[0]
    return text.strip()


def extract_json(text: str) -> Any:
    """First JSON value in `text` (handles code fences, prose around it, think-blocks). Raises ValueError."""
    text = strip_thinking(text)
    if not text:
        raise ValueError("empty response")
    candidates = [text]
    candidates += [m.group(1).strip() for m in _FENCE.finditer(text)]
    dec = json.JSONDecoder()
    for cand in candidates:
        try:
            return json.loads(cand)
        except ValueError:
            pass
        for m in re.finditer(r"[\{\[]", cand):
            try:
                value, _ = dec.raw_decode(cand[m.start():])
                if isinstance(value, (dict, list)):
                    return value
            except ValueError:
                continue
    raise ValueError("no JSON object found in response")


def parse_model(text: str, model_cls: type[T]) -> T:
    """Validate model output against `model_cls`. A bare list is accepted for single-list-field schemas."""
    data = extract_json(text)
    if isinstance(data, list):
        list_fields = [n for n, f in model_cls.model_fields.items() if "list" in str(f.annotation).lower()]
        if len(list_fields) == 1:
            data = {list_fields[0]: data}
    return model_cls.model_validate(data)


def structured_call(
    client: LLMClient, system: str, prompt: str, model_cls: type[T], *,
    retries: int = 2, num_predict: int | None = None, cancel: threading.Event | None = None,
) -> tuple[T, int]:
    """Ask for JSON, validate, and retry (with the error fed back) a bounded number of times.

    Returns (validated object, attempts used). Raises LLMOutputError when all attempts fail.
    Model output is data only - it is never executed or used as a path/command.
    """
    schema = json_schema_for(model_cls)
    current = prompt
    last_err = ""
    last_text = ""
    for attempt in range(1, retries + 2):
        last_text = client.generate_json(system, current, schema, num_predict=num_predict, cancel=cancel)
        try:
            return parse_model(last_text, model_cls), attempt
        except (ValueError, ValidationError) as exc:
            last_err = _short_error(exc)
            log.warning("LLM response invalid (attempt %d/%d): %s", attempt, retries + 1, last_err)
            current = (
                prompt
                + f"\n\nYour previous reply was rejected: {last_err}\n"
                "Reply again with ONLY one valid JSON object that matches the required schema. "
                "Keep titles and text short."
            )
    raise LLMOutputError(
        f"The local model did not produce valid output after {retries + 1} attempts ({last_err}).",
        details=last_text[:1500],
    )


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        errs = exc.errors()[:3]
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in errs)
    return str(exc)[:200]
