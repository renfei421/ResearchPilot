"""Non-destructive groups of candidate publication versions."""

from pydantic import BaseModel

from researchpilot.paper_candidate import PaperCandidate


class PaperVersionGroup(BaseModel):
    """Related bibliographic records, each retaining its own metadata and hits."""

    group_id: str
    members: list[PaperCandidate]
