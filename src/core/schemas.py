"""Schemas for *untrusted* LLM output. Everything the model says is validated here before use."""
from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

CATEGORY_VALUES = ("story", "funny", "emotional", "surprising", "insight", "quote", "reaction")


def _clean(s: Any, limit: int) -> str:
    return " ".join(str(s).split())[:limit]


class RawMoment(BaseModel):
    start: float
    end: float
    category: Literal["story", "funny", "emotional", "surprising", "insight", "quote", "reaction"]
    title: str = Field(max_length=120)
    explanation: str = Field("", max_length=400)
    hook: str = Field("", max_length=300)
    caption: str = Field("", max_length=300)
    context_completeness: int = Field(5, ge=1, le=10)
    payoff: int = Field(5, ge=1, le=10)
    confidence: int = Field(5, ge=1, le=10)

    @field_validator("title", "explanation", "hook", "caption", mode="before")
    @classmethod
    def _tidy(cls, v: Any) -> str:
        return _clean(v if v is not None else "", 280)


class MomentList(BaseModel):
    moments: list[RawMoment] = Field(default_factory=list, max_length=20)


class RawScore(BaseModel):
    id: str
    hook: int = Field(ge=1, le=10)
    clarity: int = Field(ge=1, le=10)
    completeness: int = Field(ge=1, le=10)
    payoff: int = Field(ge=1, le=10)
    escalation: int = Field(ge=1, le=10)
    reason: str = Field("", max_length=300)

    @field_validator("reason", mode="before")
    @classmethod
    def _tidy(cls, v: Any) -> str:
        return _clean(v if v is not None else "", 280)


class ScoreList(BaseModel):
    scores: list[RawScore] = Field(default_factory=list, max_length=32)


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """JSON schema with `$ref`s inlined (constrained decoders cope better with flat schemas)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return inline(copy.deepcopy(defs[name]))
            return {k: inline(v) for k, v in node.items() if k not in ("title", "default")}
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    return inline(schema)
