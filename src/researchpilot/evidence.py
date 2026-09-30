"""Evidence contracts and cross-reference checks shared by all agent stages."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Text = Annotated[str, Field(strict=True, min_length=1)]


class EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DocumentPage(EvidenceModel):
    # Preserve source metadata verbatim; the existing Paper model permits edge
    # whitespace in titles. Trimming it here would break source identity checks.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    paper_id: Text
    title: Text
    page_number: int | None = Field(strict=True, ge=1)
    source_type: Literal["pdf", "abstract"]
    text: Text
    source_url: str | None = None

    @field_validator("paper_id", "title", "text")
    @classmethod
    def nonblank_source_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Source fields must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_page(self) -> "DocumentPage":
        if (self.source_type == "pdf") != (self.page_number is not None):
            raise ValueError("PDF evidence requires a page number; abstracts must have none.")
        return self


class AcquiredDocument(EvidenceModel):
    pages: list[DocumentPage]
    warnings: list[str] = Field(default_factory=list)
    cache_hit: bool = False


class EvidencePassage(DocumentPage):
    passage_id: Text

    @field_validator("passage_id")
    @classmethod
    def nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("passage_id must not be blank.")
        return value


class RetrievalView(EvidenceModel):
    view_id: Text
    text: Text
    origin_text: str | None = None


class PassageViewHit(EvidenceModel):
    view_id: Text
    rank: int = Field(strict=True, ge=1)


class RetrievedPassage(EvidenceModel):
    """Lexical score is diagnostic only and never sent to the LLM."""

    passage: EvidencePassage
    lexical_score: float = Field(ge=0, allow_inf_nan=False)
    fusion_score: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    view_hits: list[PassageViewHit] = Field(default_factory=list)


def require_unique(values: list[str], name: str) -> list[str]:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique.")
    return values


def passage_index(passages: list[EvidencePassage]) -> dict[str, EvidencePassage]:
    require_unique([p.passage_id for p in passages], "passage IDs")
    return {p.passage_id: p for p in passages}


def require_known_ids(ids: list[str], available: dict, name: str) -> None:
    require_unique(ids, name)
    unknown = set(ids) - available.keys()
    if unknown:
        raise ValueError(f"Unknown {name}: {', '.join(sorted(unknown))}.")


class EvidenceSelection(EvidenceModel):
    selected_passage_ids: list[Text]
    coverage_notes: Text
    insufficient_evidence: bool = Field(strict=True)

    @field_validator("selected_passage_ids")
    @classmethod
    def unique_ids(cls, values: list[str]) -> list[str]:
        return require_unique(values, "selected passage IDs")


class AnswerClaim(EvidenceModel):
    claim_id: Text
    text: Text
    evidence_ids: list[Text] = Field(min_length=1)

    @field_validator("evidence_ids")
    @classmethod
    def unique_ids(cls, values: list[str]) -> list[str]:
        return require_unique(values, "claim evidence IDs")


class AnswerDraft(EvidenceModel):
    claims: list[AnswerClaim] = Field(max_length=12)
    limitations: list[Text]

    @field_validator("claims")
    @classmethod
    def unique_claims(cls, values: list[AnswerClaim]) -> list[AnswerClaim]:
        require_unique([c.claim_id for c in values], "claim IDs")
        return values


SupportStatus = Literal["supported", "partially_supported", "unsupported", "conflicting"]


class AtomicAssertion(EvidenceModel):
    """A material assertion and its evidence result, never private reasoning."""

    assertion_id: Text
    text: Text
    kind: Literal["factual", "quantitative", "comparative", "causal", "theoretical", "scope", "historical"]
    status: SupportStatus
    evidence_ids: list[Text]
    reason: Text
    scope_supported: bool = Field(strict=True)
    inference_supported: bool = Field(strict=True)
    quantities_supported: bool = Field(strict=True)

    @model_validator(mode="after")
    def check_support(self) -> "AtomicAssertion":
        require_unique(self.evidence_ids, "assertion evidence IDs")
        if self.status == "supported" and (not self.evidence_ids or not all((
                self.scope_supported, self.inference_supported, self.quantities_supported))):
            self.status = "partially_supported" if self.evidence_ids else "unsupported"
        return self


def aggregate_assertions(assertions: list[AtomicAssertion]) -> SupportStatus:
    statuses = {a.status for a in assertions}
    if "conflicting" in statuses:
        return "conflicting"
    if statuses == {"supported"}:
        return "supported"
    if statuses & {"supported", "partially_supported"}:
        return "partially_supported"
    return "unsupported"


class ClaimVerification(EvidenceModel):
    claim_id: Text
    status: SupportStatus
    # Used evidence, not a copy of the synthesis citations. A rejected claim
    # may have no supporting evidence at all.
    evidence_ids: list[Text]
    reason: Text
    # Default preserves readability of previously persisted runs.
    assertions: list[AtomicAssertion] = Field(default_factory=list)

    @field_validator("evidence_ids")
    @classmethod
    def unique_ids(cls, values: list[str]) -> list[str]:
        return require_unique(values, "verification evidence IDs")


def validate_draft(draft: AnswerDraft, evidence: list[EvidencePassage]) -> None:
    available = passage_index(evidence)
    require_unique([c.claim_id for c in draft.claims], "claim IDs")
    for claim in draft.claims:
        require_known_ids(claim.evidence_ids, available, "claim evidence IDs")


def validate_verification(claim: AnswerClaim, verification: ClaimVerification) -> None:
    if verification.claim_id != claim.claim_id:
        raise ValueError("Unknown claim ID returned by verifier.")
    allowed = dict.fromkeys(require_unique(claim.evidence_ids, "claim evidence IDs"))
    require_known_ids(verification.evidence_ids, allowed, "verification evidence IDs")
    require_unique([a.assertion_id for a in verification.assertions], "assertion IDs")
    for assertion in verification.assertions:
        require_known_ids(assertion.evidence_ids, allowed, "assertion evidence IDs")
        if assertion.status != "unsupported":
            require_known_ids(assertion.evidence_ids, dict.fromkeys(verification.evidence_ids),
                              "verified assertion evidence IDs")
        if assertion.status == "supported" and (not assertion.evidence_ids or not all((
                assertion.scope_supported, assertion.inference_supported, assertion.quantities_supported))):
            raise ValueError("Supported assertion requires evidence and all support checks.")
    if verification.status == "supported" and not verification.evidence_ids:
        raise ValueError("Supported claim requires verified evidence.")
    if verification.assertions and verification.status == "supported":
        if aggregate_assertions(verification.assertions) != "supported":
            raise ValueError("Supported claim requires all material assertions to be supported.")


class VerifiedClaim(EvidenceModel):
    """Audit record; unsupported claims stay here but never enter the answer."""

    claim: AnswerClaim
    verification: ClaimVerification

    @model_validator(mode="after")
    def validate_references(self) -> "VerifiedClaim":
        validate_verification(self.claim, self.verification)
        return self


def atomic_evidence_ids(claim: AnswerClaim, assertions: list[AtomicAssertion]) -> list[str]:
    """Used support/conflict evidence in original citation order; no new IDs."""
    used = {key for atom in assertions if atom.status != "unsupported" for key in atom.evidence_ids}
    return [key for key in claim.evidence_ids if key in used]


def supported_findings(record: VerifiedClaim) -> list[tuple[str, list[str], str]]:
    """Authoritative surviving factual content for rendering and trusted ingestion.

    Old saved audits may include unused IDs in their claim-level union. Where
    atoms exist they remain authoritative; atomless legacy results retain their
    explicitly verified citations. Never modify the original synthesis record.
    """
    validate_verification(record.claim, record.verification)
    verification = record.verification
    if verification.status == "unsupported":
        return []
    if verification.status == "supported":
        ids = (atomic_evidence_ids(record.claim, verification.assertions) if verification.assertions
               else [key for key in record.claim.evidence_ids if key in verification.evidence_ids])
        return [(record.claim.text, ids, verification.reason)]
    return [(atom.text, [key for key in record.claim.evidence_ids if key in atom.evidence_ids], atom.reason)
            for atom in verification.assertions if atom.status == "supported"]
