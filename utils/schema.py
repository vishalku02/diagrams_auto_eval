"""Pydantic models for CTML pedagogical quality structured outputs."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict

ValueYesNo = Literal["Yes", "No"]


class YesNoCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str = Field(min_length=1)
    value: ValueYesNo


class PedagogicalJudgeOutput(BaseModel):
    """5 binary dimensions derived from Mayer's CTML principles."""
    model_config = ConfigDict(extra="forbid")

    coherence: YesNoCriterion
    signaling: YesNoCriterion
    spatial_contiguity: YesNoCriterion
    segmenting: YesNoCriterion
    appropriate_labeling: YesNoCriterion


STRICT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "PedagogicalJudgeRubric",
        "schema": PedagogicalJudgeOutput.model_json_schema(),
        "strict": True,
    },
}
