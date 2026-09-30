"""Validated planning inputs and the provider-independent one-shot interface."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from researchpilot.query_plan import SearchPlan


class PlannerRequest(BaseModel):
    """Research question and contextual limits for a single planning request."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    research_question: str = Field(strict=True, min_length=1)
    year_from: int | None = Field(default=None, strict=True)
    year_to: int | None = Field(default=None, strict=True)
    max_queries: int = Field(default=5, strict=True, ge=3, le=5)
    project_context: dict | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_year_order(self) -> "PlannerRequest":
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must be less than or equal to year_to.")
        return self


class QueryPlanner(Protocol):
    """Plan once, without executing queries or consuming retrieval results."""

    def plan(self, request: PlannerRequest) -> SearchPlan: ...
