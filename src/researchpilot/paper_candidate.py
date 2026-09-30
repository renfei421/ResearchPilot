"""Candidate-pool models with query-level retrieval provenance."""

from pydantic import BaseModel, ConfigDict, Field

from researchpilot.paper import Paper


class SearchQuery(BaseModel):
    """A named query, trimmed at the edges to match the OpenAlex search text."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query_id: str = Field(strict=True, min_length=1)
    text: str = Field(strict=True, min_length=1)


class QueryHit(BaseModel):
    """A paper's original one-based position in one query's results."""

    query_id: str
    query_text: str
    rank: int = Field(ge=1)


class PaperCandidate(BaseModel):
    """The first-seen paper and its distinct query hits, without scoring."""

    paper: Paper
    hits: list[QueryHit]
