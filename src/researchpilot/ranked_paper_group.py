"""A version group with fused retrieval hits and an RRF score."""

from pydantic import BaseModel

from researchpilot.paper_candidate import QueryHit
from researchpilot.paper_version import PaperVersionGroup


class RankedPaperGroup(BaseModel):
    """Retrieval fusion results, without a stored final position."""

    group: PaperVersionGroup
    fused_hits: list[QueryHit]
    rrf_score: float
