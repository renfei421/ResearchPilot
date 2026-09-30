"""Evidence coverage and gap-directed follow-up contracts, without orchestration."""

from typing import Literal, Protocol
import unicodedata

from pydantic import Field, field_validator, model_validator

from researchpilot.evidence import (
    EvidenceModel, EvidencePassage, Text, VerifiedClaim, passage_index,
    require_known_ids, require_unique, supported_findings,
)
from researchpilot.paper_candidate import SearchQuery
from researchpilot.query_plan import PlannedQuery


def query_key(text: str) -> str:
    """Conservative equality only; never rewrite the actual search expression."""
    return " ".join(text.split()).casefold()


class EvidenceGap(EvidenceModel):
    gap_id: Text
    description: Text
    related_claim_ids: list[Text]
    severity: Literal["critical", "useful"]
    search_focus: Text
    relevance_to_question: Literal["core", "peripheral"] = "core"

    @field_validator("related_claim_ids")
    @classmethod
    def unique_claims(cls, values: list[str]) -> list[str]:
        return require_unique(values, "related claim IDs")


GapStatus = Literal["unresolved", "partially_resolved", "resolved", "superseded"]


class EvidenceGapRecord(EvidenceGap):
    first_seen_round: int = Field(strict=True, ge=1)
    last_updated_round: int = Field(strict=True, ge=1)
    status: GapStatus = "unresolved"
    resolution_reason: str = ""
    evidence_ids: list[Text] = Field(default_factory=list)
    superseded_by: str | None = None
    rescue_attempted: bool = False
    rescued_evidence_ids: list[Text] = Field(default_factory=list)
    rescue_round: int | None = None
    rescue_result: str | None = None


class GapUpdate(EvidenceModel):
    gap_id: Text
    status: GapStatus
    reason: Text
    evidence_ids: list[Text]
    related_claim_ids: list[Text]
    superseded_by: str | None


def active_gaps(ledger: list[EvidenceGapRecord]) -> list[EvidenceGapRecord]:
    return [g for g in ledger if g.status in ("unresolved", "partially_resolved")]


def gap_uncertainty(gap: EvidenceGapRecord) -> str:
    """Show current unresolved scope, not a stale pre-resolution description."""
    return f"{gap.severity} / {gap.status}: {gap.resolution_reason or gap.description}"


def searchable_gaps(ledger: list[EvidenceGapRecord]) -> list[EvidenceGap]:
    return [EvidenceGap.model_validate(g.model_dump(include=set(EvidenceGap.model_fields)) |
            ({"description": g.resolution_reason} if g.resolution_reason else {}))
            for g in active_gaps(ledger) if g.relevance_to_question == "core"]


def _gap_key(gap: EvidenceGap) -> tuple[str, str]:
    return tuple(query_key(unicodedata.normalize("NFKC", s))
                 for s in (gap.description, gap.search_focus))


def update_gap_ledger(previous: list[EvidenceGapRecord], assessment: "EvidenceAssessment",
                      round_number: int) -> list[EvidenceGapRecord]:
    """Exact IDs/text only; missing decisions carry forward, never imply resolution."""
    ledger = {g.gap_id: g.model_copy(deep=True) for g in previous}
    keys = {_gap_key(g): g.gap_id for g in previous}
    for gap in assessment.gaps:
        key = keys.get(_gap_key(gap), gap.gap_id)
        if key in ledger and _gap_key(ledger[key]) != _gap_key(gap):
            raise ValueError("A new gap cannot reuse an existing ID for different information.")
        if key not in ledger:
            ledger[key] = EvidenceGapRecord(**gap.model_dump(), first_seen_round=round_number,
                                           last_updated_round=round_number)
            keys[_gap_key(gap)] = key
        # Existing gap wording, severity, identity and original-question anchor
        # remain stable. Status changes require an explicit update below.
    for update in assessment.gap_updates:
        if update.gap_id not in {g.gap_id for g in active_gaps(previous)}:
            raise ValueError("Gap update must reference an existing active gap ID.")
        if update.status == "superseded":
            replacement = ledger.get(update.superseded_by)
            if replacement is None or replacement.gap_id == update.gap_id:
                raise ValueError("Superseded gap requires a distinct known replacement.")
            if replacement.status in ("resolved", "superseded"):
                raise ValueError("Superseded gap replacement must remain active.")
            original = ledger[update.gap_id]
            if (original.relevance_to_question == "core" and replacement.relevance_to_question != "core"
                    or original.severity == "critical" and replacement.severity != "critical"):
                raise ValueError("Superseding must preserve the original gap's importance and question scope.")
        ledger[update.gap_id] = ledger[update.gap_id].model_copy(update=dict(
            status=update.status, last_updated_round=round_number, resolution_reason=update.reason,
            evidence_ids=list(update.evidence_ids), related_claim_ids=list(update.related_claim_ids),
            superseded_by=update.superseded_by))
    return list(ledger.values())


class EvidenceAssessment(EvidenceModel):
    sufficient: bool = Field(strict=True)
    gaps: list[EvidenceGap] = Field(max_length=8)
    rationale: Text
    gap_updates: list[GapUpdate] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_gaps(self) -> "EvidenceAssessment":
        require_unique([g.gap_id for g in self.gaps], "gap IDs")
        require_unique([g.gap_id for g in self.gap_updates], "gap update IDs")
        if self.sufficient and self.gaps:
            raise ValueError("A sufficient assessment must have no remaining evidence gaps.")
        return self


class FollowUpQuery(EvidenceModel):
    query_id: Text
    text: Text
    gap_id: Text
    rationale: Text

    # Reuse the original planner's search-expression guards without adding roles.
    @field_validator("text", mode="before")
    @classmethod
    def single_line(cls, value: object) -> object:
        return PlannedQuery.require_single_line(value)

    @field_validator("text")
    @classmethod
    def search_expression(cls, value: str) -> str:
        return PlannedQuery.reject_request_syntax(value)


class FollowUpSearchPlan(EvidenceModel):
    # An empty plan is a valid explicit decision to stop searching.
    queries: list[FollowUpQuery] = Field(max_length=3)

    @field_validator("queries")
    @classmethod
    def unique_query_ids(cls, values: list[FollowUpQuery]) -> list[FollowUpQuery]:
        require_unique([q.query_id for q in values], "follow-up query IDs")
        return values


class EvidenceAssessor(Protocol):
    def assess(self, question: str, claims: list[VerifiedClaim],
               evidence: list[EvidencePassage], *,
               prior_gaps: list[EvidenceGapRecord] = ()) -> EvidenceAssessment: ...


class FollowUpPlanner(Protocol):
    def plan(self, question: str, gaps: list[EvidenceGap], executed_queries: list[SearchQuery],
             concepts: list[str], max_queries: int = 3) -> FollowUpSearchPlan: ...


def validate_assessment(assessment: EvidenceAssessment, claims: list[VerifiedClaim],
                        evidence: list[EvidencePassage],
                        prior_gaps: list[EvidenceGapRecord] = ()) -> None:
    available = passage_index(evidence)
    claim_ids = {c.claim.claim_id: c for c in claims}
    require_unique([c.claim.claim_id for c in claims], "claim IDs")
    for record in claims:
        require_known_ids(record.claim.evidence_ids, available, "claim evidence IDs")
    for gap in assessment.gaps:
        require_known_ids(gap.related_claim_ids, claim_ids, "related claim IDs")
    if assessment.sufficient and (not evidence or not claims):
        raise ValueError("An assessment cannot declare sufficiency without evidence and verified claims.")
    if assessment.sufficient and any(c.verification.status != "supported" for c in claims):
        raise ValueError("Sufficiency cannot override unsupported, partial, or conflicting claim statuses.")
    expected = {g.gap_id for g in active_gaps(prior_gaps)}
    if {u.gap_id for u in assessment.gap_updates} != expected:
        raise ValueError("Every prior active gap requires exactly one explicit status update.")
    for update in assessment.gap_updates:
        require_known_ids(update.evidence_ids, available, "gap evidence IDs")
        require_known_ids(update.related_claim_ids, claim_ids, "related claim IDs")
        if update.status in ("resolved", "partially_resolved") and not update.evidence_ids:
            raise ValueError("Gap resolution requires actual evidence IDs.")
        if update.status == "resolved":
            supported_ids = {key for c in claims for _, ids, _ in supported_findings(c) for key in ids}
            if not set(update.evidence_ids) & supported_ids:
                raise ValueError("A resolved gap must cite evidence used by a supported finding.")
    if assessment.sufficient and any(u.status in ("unresolved", "partially_resolved")
                                    for u in assessment.gap_updates):
        raise ValueError("Sufficiency cannot silently discard a prior unresolved gap.")


def new_follow_up_queries(plan: FollowUpSearchPlan, gaps: list[EvidenceGap],
                          executed: list[SearchQuery], max_queries: int) -> list[SearchQuery]:
    if type(max_queries) is not int or not 1 <= max_queries <= 3:
        raise ValueError("max_queries must be an integer between 1 and 3.")
    if len(plan.queries) > max_queries:
        raise ValueError("Follow-up plan exceeds the requested query budget.")
    known_gaps = {g.gap_id for g in gaps}
    seen_texts = {query_key(q.text) for q in executed}
    seen_ids = {q.query_id for q in executed}
    accepted = []
    for query in plan.queries:
        if query.gap_id not in known_gaps:
            raise ValueError("Follow-up query references an unknown gap ID.")
        key = query_key(query.text)
        if key in seen_texts:
            continue
        if query.query_id in seen_ids:
            raise ValueError("A new query must have an unused query_id across the entire run.")
        seen_texts.add(key)
        seen_ids.add(query.query_id)
        accepted.append(SearchQuery(query_id=query.query_id, text=query.text))
    return accepted
