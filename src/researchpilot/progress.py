"""Small public progress snapshots; never prompts or reasoning traces."""

from time import monotonic
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Stage = Literal["queued", "planning", "searching", "reranking", "acquiring",
                "retrieving", "selecting", "synthesizing", "verifying", "assessing",
                "rescuing", "follow_up", "finalizing", "completed", "failed"]


class ResearchProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_stage: Stage = "queued"
    current_round: int = Field(default=0, ge=0)
    selected_papers: int = Field(default=0, ge=0)
    evidence_collected: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


def state_progress(stage: Stage, state: dict) -> ResearchProgress:
    return ResearchProgress(
        current_stage=stage, current_round=state.get("search_round", 1),
        selected_papers=len(state.get("selected_papers", [])),
        evidence_collected=len(state.get("selected_evidence", {})),
        elapsed_seconds=max(0, round(monotonic() - state.get("started_at", monotonic()), 3)),
        warnings=list(state.get("warnings", [])),
    )
