"""Public request/result contracts for the synchronous research agent."""

from typing import Literal

from pydantic import Field, model_validator

from researchpilot.evidence import (
    EvidenceModel, EvidencePassage, EvidenceSelection, RetrievedPassage, Text,
    VerifiedClaim, RetrievalView, passage_index, require_known_ids, require_unique,
)
from researchpilot.evidence_bundle import AssemblyLimits, EvidenceBundle, validate_bundle
from researchpilot.paper import Paper
from researchpilot.hybrid_retrieval import RetrievalDiagnostics
from researchpilot.document_rescue import RescueLimits, RescueTrace
from researchpilot.math_evidence import VisualEvidenceRecord
from researchpilot.query_plan import SearchPlan
from researchpilot.paper_candidate import SearchQuery
from researchpilot.research_iteration import EvidenceAssessment, EvidenceGapRecord, FollowUpSearchPlan
from researchpilot.project_models import ProjectRunRequest, ProjectUsage


class ResearchRequest(ProjectRunRequest):
    question: Text
    year_from: int | None = Field(default=None, strict=True)
    year_to: int | None = Field(default=None, strict=True)
    max_papers: int = Field(default=6, strict=True, ge=1, le=20)
    max_queries: int = Field(default=5, strict=True, ge=3, le=5)
    top_passages: int = Field(default=20, strict=True, ge=1, le=50)
    max_search_rounds: int = Field(default=2, strict=True, ge=1, le=3)
    max_follow_up_queries: int = Field(default=3, strict=True, ge=1, le=3)
    rescue_limits: RescueLimits = Field(default_factory=RescueLimits)
    assembly_limits: AssemblyLimits = Field(default_factory=AssemblyLimits)

    @model_validator(mode="after")
    def ordered_years(self) -> "ResearchRequest":
        if self.year_from is not None and self.year_to is not None:
            if self.year_from > self.year_to:
                raise ValueError("year_from must be less than or equal to year_to.")
        return self


class SelectedPaper(EvidenceModel):
    citation_label: Text
    group_id: Text
    paper: Paper
    source_type: Literal["pdf", "abstract", "unavailable"] = "unavailable"
    cache_hit: bool = False


class RunStats(EvidenceModel):
    planned_queries: int = 0
    # Candidates delivered by PaperSearchService, before version grouping.
    raw_candidates: int = 0
    retrieved_query_hits: int = 0
    version_groups: int = 0
    selected_papers: int = 0
    full_text_papers: int = 0
    abstract_only_papers: int = 0
    passages_created: int = 0
    passages_retrieved: int = 0
    evidence_selected: int = 0
    claims_drafted: int = 0
    supported: int = 0
    partially_supported: int = 0
    unsupported: int = 0
    conflicting: int = 0
    paper_ranking: Literal["not_run", "semantic", "rrf_fallback"] = "not_run"
    elapsed_seconds: float = 0.0
    search_rounds: int = 0
    synthesis_rounds: int = 0
    verification_calls: int = 0
    local_rescue_passes: int = 0
    visual_pages_rendered: int = 0
    visual_evidence_calls: int = 0
    embedding_cache_hits: int = 0
    embedding_cache_misses: int = 0
    passage_embeddings_generated: int = 0
    query_embeddings_generated: int = 0
    embedding_batches: int = 0
    embedding_input_tokens: int = 0
    embedding_elapsed_seconds: float = 0.0
    hybrid_retrieval_elapsed_seconds: float = 0.0
    dense_fallback_used: bool = False
    bundles_created: int = 0
    bundle_members: int = 0
    assembly_local_seconds: float = 0.0
    assembly_model_calls: int = 0

    def record_retrieval(self, diagnostics: RetrievalDiagnostics) -> None:
        self.embedding_cache_hits += diagnostics.cache_hits
        self.embedding_cache_misses += diagnostics.cache_misses
        for name in ("passage_embeddings_generated", "query_embeddings_generated", "embedding_batches",
                     "embedding_input_tokens", "embedding_elapsed_seconds"):
            setattr(self, name, getattr(self, name) + getattr(diagnostics, name))
        self.hybrid_retrieval_elapsed_seconds += diagnostics.hybrid_elapsed_seconds
        self.dense_fallback_used |= diagnostics.dense_fallback_used


TerminationReason = Literal[
    "sufficient_evidence", "max_search_rounds", "no_meaningful_gaps",
    "no_follow_up_queries", "duplicate_queries", "no_new_papers", "no_new_evidence",
    "follow_up_planner_failed", "evidence_assessor_failed", "follow_up_search_failed",
    "no_literature_found",
]


class RoundTrace(EvidenceModel):
    round: int
    queries: int
    query_ids: list[str]
    candidate_papers: int = 0
    new_candidate_papers: int = 0
    already_seen_papers: int = 0
    new_papers: int = 0  # Newly selected representatives, not raw query rows.
    new_passages: int = 0
    new_evidence: int = 0
    evidence_passages: int = 0  # Accumulated selected evidence.
    claims_drafted: int = 0
    supported_claims: int = 0
    partial_claims: int = 0
    unsupported_claims: int = 0
    conflicting_claims: int = 0
    remaining_gaps: int = 0
    synthesis_performed: bool = False
    ranking_path: Literal["not_run", "semantic", "rrf_fallback"] = "not_run"
    retrieval_views: list[RetrievalView] = Field(default_factory=list)
    passage_retrieval: RetrievalDiagnostics | None = None
    gap_ids_carried: list[str] = Field(default_factory=list)
    gap_ids_new: list[str] = Field(default_factory=list)
    gap_ids_resolved: list[str] = Field(default_factory=list)
    gap_ids_partially_resolved: list[str] = Field(default_factory=list)
    gap_ids_superseded: list[str] = Field(default_factory=list)
    local_rescue_triggered: bool = False


from researchpilot.claim_stability import DowngradeAudit


class ResearchResult(EvidenceModel):
    project_usage: ProjectUsage = Field(default_factory=ProjectUsage)
    question: Text
    answer: Text
    claims: list[VerifiedClaim]
    evidence: list[EvidencePassage]
    papers: list[SelectedPaper]
    search_plan: SearchPlan
    warnings: list[str]
    run_stats: RunStats
    evidence_selection: EvidenceSelection | None = None
    retrieved_passages: list[RetrievedPassage] = Field(default_factory=list)
    draft_limitations: list[str] = Field(default_factory=list)
    evidence_assessment: EvidenceAssessment | None = None
    round_trace: list[RoundTrace] = Field(default_factory=list)
    executed_queries: list[SearchQuery] = Field(default_factory=list)
    follow_up_plans: list[FollowUpSearchPlan] = Field(default_factory=list)
    termination_reason: TerminationReason | None = None
    gap_ledger: list[EvidenceGapRecord] = Field(default_factory=list)
    rescue_trace: list[RescueTrace] = Field(default_factory=list)
    visual_evidence: list[VisualEvidenceRecord] = Field(default_factory=list)
    evidence_bundles: list[EvidenceBundle] = Field(default_factory=list)
    assembly_visual_trace: list[RescueTrace] = Field(default_factory=list)
    downgrade_audit: list[DowngradeAudit] = Field(default_factory=list)
    atom_bundle_support: dict[str, dict[str, list[str]]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolve_references(self) -> "ResearchResult":
        available = passage_index(self.evidence)
        papers = {p.paper.paper_id: p for p in self.papers}
        require_unique([p.paper.paper_id for p in self.papers], "selected paper IDs")
        require_unique([p.citation_label for p in self.papers], "citation labels")
        require_unique([c.claim.claim_id for c in self.claims], "claim IDs")
        for passage in self.evidence:
            if passage.paper_id not in papers:
                raise ValueError("Evidence must originate from a selected paper.")
            if passage.title != papers[passage.paper_id].paper.title:
                raise ValueError("Evidence title must match its source paper.")
        for record in self.claims:
            require_known_ids(record.claim.evidence_ids, available, "claim evidence IDs")
        visuals = {r.evidence.evidence_id: r for r in self.visual_evidence}
        require_unique([r.evidence.evidence_id for r in self.visual_evidence], "visual evidence IDs")
        for passage in self.evidence:
            if passage.passage_id.startswith("visual:"):
                record = visuals.get(passage.passage_id)
                if record is None or (record.evidence.paper_id, record.evidence.page_number) != (passage.paper_id, passage.page_number):
                    raise ValueError("Visual evidence must resolve to its original PDF/page audit record.")
        require_unique([b.bundle_id for b in self.evidence_bundles], "bundle IDs")
        for bundle in self.evidence_bundles:
            validate_bundle(bundle, available)
        return self
