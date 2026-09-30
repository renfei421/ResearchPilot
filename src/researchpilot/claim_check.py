"""Public contracts for bounded, evidence-grounded prior-work alignment."""

from typing import Annotated, Literal
import re

from pydantic import Field, field_validator, model_validator

from researchpilot.evidence import EvidenceModel, EvidencePassage, Text, passage_index, require_known_ids, require_unique
from researchpilot.evidence_bundle import EvidenceBundle, validate_bundle
from researchpilot.document_rescue import RescueTrace
from researchpilot.math_evidence import VisualEvidenceRecord
from researchpilot.research_models import SelectedPaper, RunStats
from researchpilot.project_models import ProjectRunRequest, ProjectUsage
from researchpilot.claim_recovery import RecoveryDiagnostic

CAVEAT = ("This analysis identifies overlap and apparent gaps within the literature retrieved by "
          "ResearchPilot. Failure to identify a close match is not proof that no prior work exists.")
_FORBIDDEN = re.compile(r"\b(?:your|this|the) idea is novel\b|\bno one has done this\b|"
                        r"\bthis is the first work to\b|\b(?:proves?|establishes?) (?:originality|novelty)\b", re.I)


def safe_generated_text(text: str) -> str:
    if _FORBIDDEN.search(text):
        raise ValueError("Unsupported novelty certification is forbidden.")
    return text


class ClaimCheckRequest(ProjectRunRequest):
    claim: Annotated[str, Field(strict=True, min_length=1, max_length=6000)]
    field: Annotated[str, Field(strict=True, max_length=300)] | None = None
    context: Annotated[str, Field(strict=True, max_length=4000)] | None = None
    year_from: int | None = Field(default=None, strict=True, ge=1, le=9999)
    year_to: int | None = Field(default=None, strict=True, ge=1, le=9999)
    max_papers_per_claim: int = Field(default=8, strict=True, ge=1, le=8)
    max_search_rounds: int = Field(default=2, strict=True, ge=1, le=2)
    claim_title: Annotated[str, Field(strict=True, max_length=200)] | None = None

    @model_validator(mode="after")
    def ordered_years(self):
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from must not exceed year_to.")
        return self


class ClaimAssertion(EvidenceModel):
    claim_id: Text
    text: Text
    claim_type: Literal["theoretical", "methodological", "empirical", "mechanistic", "comparative",
                        "application", "assumption", "other"]
    concepts: list[Text] = Field(min_length=1, max_length=8)
    depends_on: list[Text]
    importance: Literal["core", "supporting"]


class ClaimDecomposition(EvidenceModel):
    main_claim: Text
    assertions: list[ClaimAssertion] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def references(self):
        index = {a.claim_id: a for a in self.assertions}
        require_unique([a.claim_id for a in self.assertions], "claim IDs")
        core = [a for a in self.assertions if a.importance == "core"]
        if not 1 <= len(core) <= 5:
            raise ValueError("Decomposition requires 1 to 5 core assertions.")
        def visit(key, ancestors):
            if key in ancestors:
                raise ValueError("Claim dependencies must be acyclic.")
            for dep in index[key].depends_on:
                visit(dep, ancestors | {key})
        for assertion in self.assertions:
            require_known_ids(assertion.depends_on, index, "dependency IDs")
            visit(assertion.claim_id, set())
            safe_generated_text(assertion.text)
        return self


class ClaimQuery(EvidenceModel):
    text: Text
    role: Literal["direct", "terminology", "mechanism", "assumption", "bridge"]


def query_key(text: str) -> str:
    return " ".join(text.split()).casefold()


class ClaimSearchPlan(EvidenceModel):
    claim_id: Text
    queries: list[ClaimQuery] = Field(min_length=2, max_length=5)

    @model_validator(mode="after")
    def complementary(self):
        if "direct" not in [q.role for q in self.queries]:
            raise ValueError("Each claim plan requires a direct query.")
        require_unique([query_key(q.text) for q in self.queries], "queries within a claim")
        if len({q.role for q in self.queries}) < 2:
            raise ValueError("Claim queries must have complementary roles.")
        return self


class ClaimSearchPlans(EvidenceModel):
    plans: list[ClaimSearchPlan] = Field(min_length=1, max_length=5)

    def validate_targets(self, claim_ids):
        ids = require_unique([p.claim_id for p in self.plans], "search plan claim IDs")
        if set(ids) != set(claim_ids):
            raise ValueError("Search plans must cover exactly the requested core assertions.")
        if sum(len(p.queries) for p in self.plans) > 20:
            raise ValueError("At most 20 planned queries per round.")
        return self


Relation = Literal["DIRECT_OVERLAP", "PARTIAL_OVERLAP", "SUPPORTING", "BRIDGING",
                   "CONTRADICTING", "METHOD_SIMILAR", "ADJACENT"]
DIMENSIONS = ("setting", "assumptions", "mechanism", "method", "quantity", "conclusion", "scope")


class DimensionComparison(EvidenceModel):
    dimension: Literal["setting", "assumptions", "mechanism", "method", "quantity", "conclusion", "scope"]
    claim_scope: Text
    paper_scope: Text
    alignment: Literal["same", "partial", "different", "unknown", "not_applicable"]
    evidence_ids: list[Text]


class ClaimPaperRelation(EvidenceModel):
    claim_id: Text
    paper_id: Text
    relation: Relation
    relation_summary: Text
    matched_dimensions: list[Text]
    differences: list[Text]
    evidence_ids: list[Text]
    scope_notes: list[Text]
    confidence: Literal["high", "medium", "low"]
    comparisons: list[DimensionComparison] = Field(min_length=7, max_length=7)
    evidence_quality: Literal["substantive", "bibliographic_only", "none"]
    explicit_incompatibility: bool = Field(strict=True)

    @model_validator(mode="after")
    def grounded_scope(self):
        require_unique([d.dimension for d in self.comparisons], "comparison dimensions")
        if set(d.dimension for d in self.comparisons) != set(DIMENSIONS):
            raise ValueError("All seven comparison dimensions are required.")
        require_unique(self.evidence_ids, "relation evidence IDs")
        for d in self.comparisons:
            require_unique(d.evidence_ids, "dimension evidence IDs")
            if set(d.evidence_ids) - set(self.evidence_ids):
                raise ValueError("Dimension evidence must be cited by the relation.")
            if d.alignment in ("same", "partial", "different") and not d.evidence_ids:
                raise ValueError("Observed dimension comparisons require evidence.")
        if self.relation != "ADJACENT" and (not self.evidence_ids or self.evidence_quality != "substantive"):
            raise ValueError("A substantive relation requires substantive cited evidence.")
        if self.relation in ("DIRECT_OVERLAP", "PARTIAL_OVERLAP") and (not self.matched_dimensions or not self.differences):
            raise ValueError("Overlap requires both similarities and differences (or explicit no observed difference).")
        by_dim = {d.dimension: d.alignment for d in self.comparisons}
        if self.relation == "DIRECT_OVERLAP":
            if any(d.alignment not in ("same", "not_applicable") for d in self.comparisons):
                raise ValueError("Direct overlap requires all applicable dimensions to match.")
            if by_dim["conclusion"] != "same" or by_dim["scope"] != "same":
                raise ValueError("Direct overlap requires the same conclusion and scope.")
        if self.relation == "CONTRADICTING":
            if not self.explicit_incompatibility or self.confidence != "high":
                raise ValueError("Contradiction requires explicit, high-confidence incompatible evidence.")
            if any(by_dim[d] not in ("same", "not_applicable") for d in ("setting", "assumptions", "quantity", "scope")) or by_dim["scope"] != "same":
                raise ValueError("Contradiction requires comparable scope and assumptions.")
        for text in [self.relation_summary, *self.matched_dimensions, *self.differences, *self.scope_notes]:
            safe_generated_text(text)
        return self


def validate_relation(relation: ClaimPaperRelation, assertions, evidence):
    if relation.claim_id not in {a.claim_id for a in assertions}:
        raise ValueError("Unknown relation claim ID.")
    available = passage_index(evidence)
    require_known_ids(relation.evidence_ids, available, "relation evidence IDs")
    if any(available[key].paper_id != relation.paper_id for key in relation.evidence_ids):
        raise ValueError("Relation evidence must belong to its paper.")


class RelationBatch(EvidenceModel):
    relations: list[ClaimPaperRelation]


class ClaimLiteratureMap(EvidenceModel):
    claim_id: Text
    claim_text: Text
    direct_overlap_papers: list[str]
    partial_overlap_papers: list[str]
    supporting_papers: list[str]
    bridging_papers: list[str]
    contradicting_papers: list[str]
    method_similar_papers: list[str]
    no_close_match_found: bool
    coverage_status: Literal["strong", "moderate", "weak"]
    search_limitations: list[str]


class BoundaryFinding(EvidenceModel):
    text: Text
    claim_ids: list[Text]
    paper_ids: list[Text]
    evidence_ids: list[Text]

    _safe = field_validator("text")(safe_generated_text)


class NoveltyBoundary(EvidenceModel):
    established_components: list[BoundaryFinding]
    partially_established_components: list[BoundaryFinding]
    known_combinations: list[BoundaryFinding]
    potentially_underexplored_connections: list[BoundaryFinding]
    contradictions: list[BoundaryFinding]
    missing_evidence: list[BoundaryFinding]
    caveat: Literal[CAVEAT] = CAVEAT


class ClaimSearchTrace(EvidenceModel):
    round: int
    claim_id: Text
    query_id: Text
    query: Text
    role: str
    reused: bool
    status: Literal["completed", "failed"]
    candidate_ids: list[str]
    limitation: str | None = None


class ClaimCheckStats(RunStats):
    assertions: int = 0
    papers_evaluated: int = 0
    relation_counts: dict[str, int] = Field(default_factory=dict)
    queries_per_claim: dict[str, int] = Field(default_factory=dict)
    relation_calls: int = 0


class ClaimCheckResult(EvidenceModel):
    source_recovery: list[RecoveryDiagnostic] = Field(default_factory=list)
    project_usage: ProjectUsage = Field(default_factory=ProjectUsage)
    request: ClaimCheckRequest
    decomposition: ClaimDecomposition
    claim_maps: list[ClaimLiteratureMap]
    closest_prior_work: list[str]
    novelty_boundary: NoveltyBoundary
    relations: list[ClaimPaperRelation]
    search_trace: list[ClaimSearchTrace]
    sources: list[SelectedPaper]
    evidence: list[EvidencePassage]
    evidence_bundles: list[EvidenceBundle]
    visual_evidence: list[VisualEvidenceRecord]
    rescue_trace: list[RescueTrace]
    warnings: list[str]
    limitations: list[str]
    run_stats: ClaimCheckStats
    termination_reason: str

    @model_validator(mode="after")
    def references(self):
        if self.request.claim != self.decomposition.main_claim:
            raise ValueError("Decomposition must preserve the user's original claim.")
        claims = {a.claim_id: a for a in self.decomposition.assertions}
        require_unique([d.claim_id for d in self.source_recovery], "recovered claim IDs")
        for diagnostic in self.source_recovery:
            if diagnostic.claim_id not in claims or claims[diagnostic.claim_id].importance != "core":
                raise ValueError("Source recovery is restricted to core claims.")
        available = passage_index(self.evidence)
        sources = {s.paper.paper_id: s for s in self.sources}
        require_unique([s.paper.paper_id for s in self.sources], "source paper IDs")
        require_unique([s.citation_label for s in self.sources], "citation labels")
        require_known_ids(self.closest_prior_work, sources, "closest paper IDs")
        require_unique([r.claim_id+"\0"+r.paper_id for r in self.relations], "claim-paper pairs")
        for passage in self.evidence:
            if passage.paper_id not in sources or passage.title != sources[passage.paper_id].paper.title:
                raise ValueError("Evidence source identity mismatch.")
        for relation in self.relations:
            if relation.paper_id not in sources:
                raise ValueError("Unknown relation paper.")
            validate_relation(relation, self.decomposition.assertions, self.evidence)
        for trace in self.search_trace:
            if trace.claim_id not in claims:
                raise ValueError("Unknown search-trace claim ID.")
        # Validate stored/externally constructed results too: a fabricated strong
        # coverage field must not certify search completion without provenance.
        from researchpilot.claim_literature_map import build_maps
        derived = {m.claim_id: m for m in build_maps(self.decomposition, self.relations, self.search_trace)}
        require_unique([m.claim_id for m in self.claim_maps], "claim map IDs")
        if set(m.claim_id for m in self.claim_maps) != set(claims):
            raise ValueError("Every assertion needs exactly one literature map.")
        for m in self.claim_maps:
            if m.claim_text != claims[m.claim_id].text:
                raise ValueError("Claim map changed an assertion.")
            for label, name in RELATION_FIELDS.items():
                expected = [r.paper_id for r in self.relations if r.claim_id == m.claim_id and r.relation == label]
                if getattr(m, name) != expected:
                    raise ValueError("Claim map must exactly reflect evaluated relations.")
            if m.no_close_match_found and (m.direct_overlap_papers or m.partial_overlap_papers or m.coverage_status == "weak"):
                raise ValueError("No-close-match requires completed coverage without direct or partial overlap.")
            if (m.coverage_status, m.no_close_match_found) != (derived[m.claim_id].coverage_status, derived[m.claim_id].no_close_match_found):
                raise ValueError("Coverage and no-close-match must be derived from actual search and relation evidence.")
        for diagnostic in self.source_recovery:
            if diagnostic.coverage_after != derived[diagnostic.claim_id].coverage_status:
                raise ValueError("Recovery coverage must match the final literature map.")
            require_known_ids(diagnostic.new_selected_papers, sources, "recovered selected papers")
        for name in NoveltyBoundary.model_fields:
            if name == "caveat":
                continue
            for finding in getattr(self.novelty_boundary, name):
                require_known_ids(finding.claim_ids, claims, "boundary claim IDs")
                require_known_ids(finding.paper_ids, sources, "boundary paper IDs")
                require_known_ids(finding.evidence_ids, available, "boundary evidence IDs")
                if any(available[key].paper_id not in finding.paper_ids for key in finding.evidence_ids):
                    raise ValueError("Boundary evidence must match its listed papers.")
                if name not in ("missing_evidence", "potentially_underexplored_connections") and not finding.evidence_ids:
                    raise ValueError("Established findings require evidence.")
        visuals = {v.evidence.evidence_id: v for v in self.visual_evidence}
        for p in self.evidence:
            if p.passage_id.startswith("visual:") and p.passage_id not in visuals:
                raise ValueError("Visual passages require their original visual evidence record.")
        for bundle in self.evidence_bundles:
            validate_bundle(bundle, available)
        return self


RELATION_FIELDS = {"DIRECT_OVERLAP": "direct_overlap_papers", "PARTIAL_OVERLAP": "partial_overlap_papers",
                   "SUPPORTING": "supporting_papers", "BRIDGING": "bridging_papers",
                   "CONTRADICTING": "contradicting_papers", "METHOD_SIMILAR": "method_similar_papers"}
