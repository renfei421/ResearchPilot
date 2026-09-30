"""Semantic assessments, separate from retrieval fusion scores."""

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from researchpilot.ranked_paper_group import RankedPaperGroup


class RelevanceAssessment(BaseModel):
    """Title/abstract relevance to a question, with a concise explanation."""

    model_config = ConfigDict(extra="forbid")

    category: Literal["direct", "supporting", "off_target"]
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(strict=True, min_length=1)

    @field_validator("reason")
    @classmethod
    def require_non_blank_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must be a non-empty string.")
        return value


class SemanticallyRankedGroup(BaseModel):
    """An original retrieval result and its independent semantic assessment."""

    ranked_group: RankedPaperGroup
    assessment: RelevanceAssessment


class RelevanceClient(Protocol):
    """The synchronous assessment interface required by SemanticReranker."""

    def assess(
        self,
        research_question: str,
        title: str,
        abstract: str | None,
    ) -> RelevanceAssessment: ...
