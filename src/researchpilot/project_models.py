"""Explicit local research records; notes and historical assessments are not proof."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from researchpilot.evidence import EvidenceModel, EvidencePassage, Text
from researchpilot.math_evidence import VisualEvidenceRecord
from researchpilot.paper import Paper
from researchpilot.paper_candidate import PaperCandidate


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identity(prefix: str, *parts) -> str:
    return prefix + ":" + sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:32]


def text_key(text: str) -> str:
    return " ".join(text.split()).casefold()


ShortText = Annotated[str, Field(strict=True, min_length=1, max_length=6000)]


class ProjectRunRequest(EvidenceModel):
    project_id: Text | None = None
    target_question_id: Text | None = None
    target_claim_id: Text | None = None
    target_gap_id: Text | None = None
    refresh_search: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def project_target(self):
        targets = [self.target_question_id, self.target_claim_id, self.target_gap_id]
        if sum(x is not None for x in targets) > 1:
            raise ValueError("Choose one project question, claim or gap target.")
        if any(targets) and self.project_id is None:
            raise ValueError("A project target requires project_id.")
        return self


class ProjectCreate(EvidenceModel):
    title: Annotated[str, Field(strict=True, min_length=1, max_length=200)]
    description: Annotated[str, Field(strict=True, max_length=6000)] = ""
    field: Annotated[str, Field(strict=True, max_length=300)] | None = None


class ResearchProject(ProjectCreate):
    project_id: Text = Field(default_factory=lambda: uuid4().hex)
    created_at: str = Field(default_factory=now)
    updated_at: str = Field(default_factory=now)
    status: Literal["active", "archived"] = "active"


class ProjectUpdate(EvidenceModel):
    title: Annotated[str, Field(strict=True, min_length=1, max_length=200)] | None = None
    description: Annotated[str, Field(strict=True, max_length=6000)] | None = None
    field: Annotated[str, Field(strict=True, max_length=300)] | None = None
    status: Literal["active", "archived"] | None = None


class ProjectRecord(EvidenceModel):
    project_id: Text
    created_at: str = Field(default_factory=now)
    updated_at: str = Field(default_factory=now)


class ProjectQuestion(ProjectRecord):
    question_id: Text = Field(default_factory=lambda: "question:" + uuid4().hex)
    text: ShortText
    status: Literal["open", "partially_answered", "answered", "archived"] = "open"


class ProjectClaim(ProjectRecord):
    claim_id: Text = Field(default_factory=lambda: "claim:" + uuid4().hex)
    text: ShortText
    claim_type: Text = "hypothesis"
    status: Literal["proposed", "partially_supported", "supported", "contradicted", "unresolved", "archived"] = "proposed"
    parent_claim_id: Text | None = None
    originating_run_id: Text | None = None
    evidence_ids: list[Text] = Field(default_factory=list)


class ProjectNote(ProjectRecord):
    note_id: Text = Field(default_factory=lambda: "note:" + uuid4().hex)
    text: ShortText


class ProjectPaper(ProjectRecord):
    paper_id: Text
    paper: Paper
    version_group_id: Text
    local_document_reference: str | None = None
    first_seen_run_id: Text
    last_used_at: str = Field(default_factory=now)

    @model_validator(mode="after")
    def consistent_paper(self):
        if self.paper_id != self.paper.paper_id:
            raise ValueError("Project paper identity must match canonical Paper.")
        return self


class EvidenceAttestation(EvidenceModel):
    run_id: Text
    claim_id: Text
    kind: Literal["atomic_verification", "prior_work_relation"]
    status: Text
    reason: Text


class ProjectEvidence(ProjectRecord):
    evidence_id: Text
    passage: EvidencePassage
    related_claim_ids: list[Text] = Field(default_factory=list)
    related_question_ids: list[Text] = Field(default_factory=list)
    verification_status: Literal["supported", "relation_assessed"]
    originating_run_id: Text
    attestations: list[EvidenceAttestation] = Field(min_length=1)
    visual_record: VisualEvidenceRecord | None = None

    @model_validator(mode="after")
    def visual_source(self):
        if self.passage.passage_id.startswith("visual:"):
            if self.visual_record is None or self.visual_record.evidence.evidence_id != self.passage.passage_id:
                raise ValueError("Project visual evidence needs its original page audit.")
        return self


class ProjectGap(ProjectRecord):
    gap_id: Text
    description: ShortText
    search_focus: ShortText
    severity: Literal["critical", "useful"] = "useful"
    related_claim_ids: list[Text] = Field(default_factory=list)
    related_question_ids: list[Text] = Field(default_factory=list)
    status: Literal["unresolved", "partially_resolved", "resolved", "archived"] = "unresolved"
    first_seen_run_id: Text
    last_updated_run_id: Text
    evidence_ids: list[Text] = Field(default_factory=list)
    resolution_reason: str = ""


class ProjectFinding(ProjectRecord):
    finding_id: Text
    text: Text
    evidence_ids: list[Text] = Field(min_length=1)
    claim_ids: list[Text]
    status: Literal["verified", "relation_assessed"]
    originating_run_id: Text
    last_updated_run_id: Text
    relation: str | None = None


class ProjectQuery(ProjectRecord):
    query_id: Text
    query: Text
    request_scope: Text
    per_query: int = Field(ge=1, le=100)
    candidates: list[PaperCandidate]
    originating_run_id: Text


class ProjectUsage(EvidenceModel):
    project_id: Text | None = None
    context_record_ids: list[str] = Field(default_factory=list)
    reused_paper_ids: list[str] = Field(default_factory=list)
    # Actual source identity survives; historical judgments are never verification input.
    evidence_origins: dict[str, Literal["reused_project_evidence", "newly_retrieved_evidence"]] = Field(default_factory=dict)
    queries_reused: int = 0
    new_papers: int = 0
    new_evidence: int = 0
    gaps_carried_in: int = 0
    gaps_resolved: int = 0


class ProjectContinue(EvidenceModel):
    target_type: Literal["question", "claim", "gap"]
    target_id: Text
    instruction: Annotated[str, Field(strict=True, max_length=2000)] = ""
    mode: Literal["research", "claim_check"] = "research"
    refresh_search: bool = Field(default=False, strict=True)
    year_from: int | None = Field(default=None, strict=True)
    year_to: int | None = Field(default=None, strict=True)
