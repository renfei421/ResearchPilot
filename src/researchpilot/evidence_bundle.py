"""Auditable same-paper evidence sets. Completeness is availability, not truth."""

from hashlib import sha256
import json
from typing import Literal

from pydantic import Field, model_validator

from researchpilot.evidence import EvidenceModel, EvidencePassage, Text, require_unique

ASSEMBLY_VERSION = "claim-dependencies-v2"
Role = Literal["assumption", "condition", "theorem", "proposition", "definition",
               "formula", "proof", "consequence", "scope", "remark", "supporting_context"]


class AssemblyLimits(EvidenceModel):
    enabled: bool = True
    max_page_radius: int = Field(default=2, strict=True, ge=0, le=2)
    max_bundle_members: int = Field(default=14, strict=True, ge=1, le=20)
    max_bundles_per_gap: int = Field(default=2, strict=True, ge=1, le=3)
    max_bundles_per_run: int = Field(default=6, strict=True, ge=1, le=8)
    max_bundles_per_paper: int = Field(default=2, strict=True, ge=1, le=4)
    max_reference_lookups: int = Field(default=12, strict=True, ge=1, le=20)
    max_reference_depth: int = Field(default=3, strict=True, ge=1, le=4)
    max_dependency_pages: int = Field(default=8, strict=True, ge=1, le=12)
    max_visual_pages_per_bundle: int = Field(default=4, strict=True, ge=0, le=4)
    max_document_passages: int = Field(default=2000, strict=True, ge=1, le=3000)


class EvidenceBundleMember(EvidenceModel):
    evidence_id: Text
    passage_id: Text | None = None
    math_evidence_id: Text | None = None
    page_number: int = Field(strict=True, ge=1)
    source_type: Literal["pdf"] = "pdf"
    role: Role

    @model_validator(mode="after")
    def original_identity(self):
        if (self.passage_id is None) == (self.math_evidence_id is None):
            raise ValueError("A bundle member requires exactly one original evidence identity.")
        if (self.passage_id or self.math_evidence_id) != self.evidence_id:
            raise ValueError("Bundle evidence identity must equal its original ID.")
        if self.math_evidence_id and not self.math_evidence_id.startswith("visual:"):
            raise ValueError("Math evidence must have a visual: identity.")
        return self


class DependencyEdge(EvidenceModel):
    referenced_by: Text
    target_id: Text
    statement_id: Text
    dependency_type: Literal["reference", "component", "proof", "continuation"]


class EvidenceBundle(EvidenceModel):
    bundle_id: Text
    paper_id: Text
    title: Text
    anchor_evidence_id: Text
    members: list[EvidenceBundleMember] = Field(min_length=1, max_length=20)
    purpose: Text
    complete: bool
    missing_components: list[Text]
    resolved_references: dict[str, list[str]] = Field(default_factory=dict)
    target_claim_ids: list[str] = Field(default_factory=list)
    target_descriptions: list[str] = Field(default_factory=list)
    required_components: dict[str, list[str]] = Field(default_factory=dict)
    dependencies: list[DependencyEdge] = Field(default_factory=list)
    assembly_version: str = ASSEMBLY_VERSION

    @model_validator(mode="after")
    def consistent(self):
        ids = require_unique([m.evidence_id for m in self.members], "bundle members")
        if self.anchor_evidence_id not in ids:
            raise ValueError("Bundle must retain its anchor.")
        if self.complete != (not self.missing_components):
            raise ValueError("Completeness must agree with missing components.")
        if any(key not in ids for values in self.resolved_references.values() for key in values):
            raise ValueError("References must resolve to actual bundle members.")
        if any(key not in ids for values in self.required_components.values() for key in values):
            raise ValueError("Components must resolve to actual bundle members.")
        if any(e.referenced_by not in ids or e.target_id not in ids for e in self.dependencies):
            raise ValueError("Dependency edges must connect actual bundle members.")
        return self


def bundle_identity(paper_id: str, anchor: str, ids: list[str], purpose: str = "",
                    version: str = ASSEMBLY_VERSION) -> str:
    parts = [version, paper_id, anchor, ids]
    if version != "structured-evidence-v1":
        parts.append(purpose)
    body = json.dumps(parts, ensure_ascii=False)
    return "bundle:" + sha256(body.encode()).hexdigest()[:24]


def validate_bundle(bundle: EvidenceBundle, available: dict[str, EvidencePassage]) -> None:
    for member in bundle.members:
        p = available.get(member.evidence_id)
        if p is None or (p.paper_id, p.title, p.page_number, p.source_type) != (
                bundle.paper_id, bundle.title, member.page_number, member.source_type):
            raise ValueError("Bundle member must retain its original same-paper/page provenance.")
    if bundle.bundle_id != bundle_identity(bundle.paper_id, bundle.anchor_evidence_id,
            [m.evidence_id for m in bundle.members], bundle.purpose, bundle.assembly_version):
        raise ValueError("Bundle ID must match the ordered source identities.")
