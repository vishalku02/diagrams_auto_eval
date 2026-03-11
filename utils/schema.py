"""Pydantic models for CTML pedagogical quality structured outputs."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

ValueYesNo = Literal["Yes", "No"]
ValueYesNoNA = Literal["Yes", "No", "N/A"]


class YesNoCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str = Field(min_length=1)
    value: ValueYesNo


class YesNoNACriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str = Field(min_length=1)
    value: ValueYesNoNA


class PedagogicalJudgeOutput(BaseModel):
    """4 dimensions aligned with prompts/pedagogical_quality.md."""
    model_config = ConfigDict(extra="forbid")

    coherence: YesNoCriterion
    signaling: YesNoNACriterion
    label_accuracy: YesNoNACriterion
    labeling: YesNoNACriterion


STRICT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "PedagogicalJudgeRubric",
        "schema": PedagogicalJudgeOutput.model_json_schema(),
        "strict": True,
    },
}
