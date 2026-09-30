"""Normalized paper metadata shared across research sources."""

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """A paper's identity, metadata, and source provenance."""

    paper_id: str
    title: str
    abstract: str | None
    authors: list[str]
    publication_year: int | None
    doi: str | None
    citation_count: int = Field(ge=0)
    source: str
    source_id: str
    open_access_url: str | None
    relevance_score: float | None = Field(default=None, ge=0.0, le=1.0)
