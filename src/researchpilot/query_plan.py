"""Planning-layer output contracts, separate from executable retrieval queries."""

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _duplicate_key(value: str) -> str:
    """Compare text conservatively without changing its stored display form."""
    return " ".join(value.split()).casefold()


class PlannedQuery(BaseModel):
    """One search expression with its planning role and rationale."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query_id: str = Field(strict=True, min_length=1)
    text: str = Field(strict=True, min_length=1)
    role: Literal["core", "facet", "bridge", "synonym"]
    rationale: str = Field(strict=True, min_length=1)

    @field_validator("text", mode="before")
    @classmethod
    def require_single_line(cls, value: object) -> object:
        # Check before trimming so an edge newline is not silently discarded.
        if isinstance(value, str) and re.search(r"[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]", value):
            raise ValueError("text must be a single-line search expression.")
        return value

    @field_validator("text")
    @classmethod
    def reject_request_syntax(cls, value: str) -> str:
        if re.search(r"https?://", value, flags=re.IGNORECASE):
            raise ValueError("text must not contain an HTTP/HTTPS URL.")
        if re.search(r"\b(?:api_key|per_page|filter)\s*=", value, flags=re.IGNORECASE):
            raise ValueError("text must not contain API parameters (api_key=, per_page=, filter=).")
        return value


class SearchPlan(BaseModel):
    """A research question, ordered concepts and three to five planned queries."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    research_question: str = Field(strict=True, min_length=1)
    concepts: list[Annotated[str, Field(strict=True, min_length=1)]] = Field(min_length=1)
    queries: list[PlannedQuery] = Field(min_length=3, max_length=5)

    @field_validator("concepts")
    @classmethod
    def require_unique_concepts(cls, values: list[str]) -> list[str]:
        keys = [_duplicate_key(value) for value in values]
        if len(set(keys)) != len(keys):
            raise ValueError("concepts must be unique after whitespace normalization and casefold.")
        return values

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, values: list[PlannedQuery]) -> list[PlannedQuery]:
        query_ids = [query.query_id for query in values]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("query_id values must be unique within a search plan.")
        text_keys = [_duplicate_key(query.text) for query in values]
        if len(set(text_keys)) != len(text_keys):
            raise ValueError("query texts must be unique after whitespace normalization and casefold.")
        if not any(query.role == "core" for query in values):
            raise ValueError("queries must include at least one query with role='core'.")
        return values
