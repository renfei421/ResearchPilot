"""Page-derived evidence and its audit trail, separate from parser passages."""

from typing import Literal
import re

from pydantic import Field

from researchpilot.evidence import (
    ClaimVerification, EvidenceModel, EvidencePassage, Text, aggregate_assertions,
)


class MathEvidence(EvidenceModel):
    evidence_id: Text
    paper_id: Text
    page_number: int = Field(strict=True, ge=1)
    statement_type: Literal["theorem", "proposition", "lemma", "assumption", "condition", "equation", "table", "other"]
    statement_text: Text
    formula_text: Text | None
    assumptions: list[Text]
    conclusion: Text | None
    relevant_to_gap: bool = Field(strict=True)
    ambiguity: Text | None


class VisualEvidenceRecord(EvidenceModel):
    evidence: MathEvidence
    pdf_sha256: Text
    image_sha256: Text
    render_version: Text
    extraction_version: Text
    model: Text
    gap_id: Text
    parser_passage_ids: list[Text]
    parser_passages: list[EvidencePassage] = Field(default_factory=list)


def visual_passage(record: VisualEvidenceRecord, source: EvidencePassage) -> EvidencePassage:
    item = record.evidence
    if (item.paper_id, item.page_number) != (source.paper_id, source.page_number):
        raise ValueError("Visual evidence must match the source paper and physical page.")
    if not item.evidence_id.startswith("visual:") or not item.relevant_to_gap:
        raise ValueError("Only relevant visual evidence with a visual: ID may enter synthesis.")
    parts = ["Target-page visual transcription (not parser text).", item.statement_text]
    if item.formula_text:
        parts.append("Visible formula: " + item.formula_text)
    if item.assumptions:
        parts.append("Visible assumptions: " + "; ".join(item.assumptions))
    if item.conclusion:
        parts.append("Visible conclusion: " + item.conclusion)
    if item.ambiguity:
        parts.append("AMBIGUITY — not exact formula support: " + item.ambiguity)
    return EvidencePassage(**(source.model_dump() | {
        "passage_id": item.evidence_id, "text": "\n".join(parts),
    }))


def guard_ambiguous_visual_support(verification: ClaimVerification,
                                   records: dict[str, VisualEvidenceRecord]) -> ClaimVerification:
    """Never promote a verdict; reject exact math backed only by ambiguous images.

    Clear scope/qualifier atoms may remain supported. The current verifier still
    decides substantive support, including assumptions and mixed evidence sets.
    """
    ambiguous = {key for key, r in records.items() if r.evidence.ambiguity}
    result = verification.model_copy(deep=True)
    changed = False
    for atom in result.assertions:
        mathematical = atom.kind in ("quantitative", "theoretical") or bool(re.search(r"[=≤≥√^]|\\(?:frac|sqrt)", atom.text))
        if (mathematical and atom.status == "supported" and atom.evidence_ids
                and set(atom.evidence_ids) <= ambiguous):
            atom.status = "partially_supported"
            atom.quantities_supported = False
            atom.reason += " Visual transcription is ambiguous; it cannot alone establish exact mathematics."
            changed = True
    if changed and result.status not in ("unsupported", "conflicting"):
        result.status = aggregate_assertions(result.assertions)
    if (not result.assertions and result.status == "supported"
            and set(result.evidence_ids) <= ambiguous):
        result.status = "partially_supported"
        changed = True
    if changed:
        result.reason += " Exact-formula support is limited by recorded visual ambiguity."
    return result
