"""Bounded Idea Check source-recovery contracts, not a second coverage system."""

from typing import Literal
import unicodedata

from pydantic import Field

from researchpilot.evidence import EvidenceModel, Text


MAX_RECOVERY_ROUNDS = 2
MAX_QUERIES_PER_ROUND = 3
MAX_QUERIES_PER_CLAIM = 6
MAX_PAPERS_PER_CLAIM = 8
# Shared additional budgets, so five weak assertions cannot each acquire eight PDFs.
MAX_ADDITIONAL_DOCUMENTS = 8
MAX_ADDITIONAL_CANDIDATES = 100


def recovery_query_key(text: str) -> str:
    """Collapse typographic variations, retaining words, order and math symbols."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())


class RecoveryQuery(EvidenceModel):
    text: Text
    role: Literal["direct", "terminology", "bridge", "method_neighbor"]


class RecoveryPlan(EvidenceModel):
    claim_id: Text
    queries: list[RecoveryQuery] = Field(max_length=MAX_QUERIES_PER_ROUND)


class RecoveryDiagnostic(EvidenceModel):
    claim_id: Text
    trigger_reason: Text
    coverage_before: Literal["weak"] = "weak"
    coverage_after: Literal["weak", "moderate", "strong"] = "weak"
    rounds_used: int = Field(default=0, ge=0, le=MAX_RECOVERY_ROUNDS)
    generated_queries: list[RecoveryQuery] = Field(default_factory=list)
    recovery_queries: list[str] = Field(default_factory=list, max_length=MAX_QUERIES_PER_CLAIM)
    successful_queries: list[str] = Field(default_factory=list)
    skipped_duplicate_queries: list[str] = Field(default_factory=list)
    new_candidate_papers: list[str] = Field(default_factory=list)
    new_selected_papers: list[str] = Field(default_factory=list, max_length=MAX_PAPERS_PER_CLAIM)
    new_evidence_count: int = Field(default=0, ge=0)
    new_relations_count: int = Field(default=0, ge=0)
    incomplete: bool = False
    stop_reason: Literal["coverage_sufficient", "budget_exhausted", "no_new_papers",
                         "no_new_substantive_evidence", "duplicate_queries",
                         "recovery_unavailable"] = "budget_exhausted"

    def summary(self):
        action = "performed" if self.recovery_queries else "attempted"
        text = (f"Additional targeted literature recovery was {action} because initial coverage for "
                f"{self.claim_id} was weak; coverage {self.coverage_before} -> {self.coverage_after}.")
        if self.coverage_after == "weak":
            text += " Coverage remains weak."
        if self.incomplete:
            text += " Supplementary recovery was incomplete; successful earlier results are retained."
        return text
