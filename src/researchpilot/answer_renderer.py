"""Deterministic final-answer policy and source citations, with no LLM calls."""

import re
from urllib.parse import quote, urlsplit

from researchpilot.evidence import (
    EvidencePassage, VerifiedClaim, atomic_evidence_ids, passage_index,
    require_known_ids, supported_findings,
)
from researchpilot.research_models import SelectedPaper
from researchpilot.research_iteration import EvidenceGapRecord, gap_uncertainty


def _plain(text: str) -> str:
    """Render model/source prose as text, not user-controlled Markdown structure."""
    return re.sub(r"([\\`*_{}\[\]<>#])", r"\\\1", " ".join(text.split()))


def source_url(paper: SelectedPaper) -> str | None:
    record = paper.paper
    if record.open_access_url:
        try:
            parsed = urlsplit(record.open_access_url)
            if parsed.scheme in ("http", "https") and parsed.hostname and not parsed.username:
                return record.open_access_url
        except ValueError:
            pass
    if record.doi:
        return "https://doi.org/" + quote(record.doi, safe="/")
    if record.source == "openalex" and re.fullmatch(r"W\d+", record.source_id):
        return "https://openalex.org/" + record.source_id
    return None


def rendered_findings(record: VerifiedClaim) -> list[tuple[str, list[str]]]:
    """No unsupported atomic text is copied into the factual answer."""
    verification = record.verification
    supported = supported_findings(record)  # Also validates mutated/corrupt audit records.
    if verification.status == "unsupported":
        return []
    if verification.status == "supported":
        return [(text, ids) for text, ids, _ in supported]
    findings = [("Supported portion only: " + text, ids) for text, ids, _ in supported]
    if verification.status == "conflicting":
        conflicts = [a for a in verification.assertions if a.status == "conflicting"]
        ids = (atomic_evidence_ids(record.claim, conflicts or verification.assertions)
               if verification.assertions else verification.evidence_ids)
        findings.append(("Unresolved disagreement in cited evidence; no combined conclusion is established.",
                         ids))
    elif not findings:
        findings.append(("Partially supported; not established in full. Unverified wording omitted.",
                         []))
    return findings


def rendered_evidence_ids(record: VerifiedClaim) -> list[str]:
    """Deterministically reconstruct the final citation audit from stored atoms."""
    used = {key for _, ids in rendered_findings(record) for key in ids}
    return [key for key in record.claim.evidence_ids if key in used]


def render_answer(claims: list[VerifiedClaim], evidence: list[EvidencePassage],
                  papers: list[SelectedPaper], *, incomplete: bool = False,
                  gaps: list[EvidenceGapRecord] = ()) -> str:
    available = passage_index(evidence)
    sources = {p.paper.paper_id: p for p in papers}
    lines = []
    for record in claims:
        claim, verification = record.claim, record.verification
        require_known_ids(claim.evidence_ids, available, "claim evidence IDs")
        for text, ids in rendered_findings(record):
            citations = []
            for key in ids:
                passage = available[key]
                if passage.paper_id not in sources:
                    raise ValueError("Citation does not resolve to a selected paper.")
                label = sources[passage.paper_id].citation_label
                location = f"p. {passage.page_number}" if passage.source_type == "pdf" else "abstract"
                citation = f"[{label}, {location}]"
                if citation not in citations:
                    citations.append(citation)
            lines.append(f"- {_plain(text)} {' '.join(citations)}")
    if not lines:
        lines.append("Insufficient evidence to provide a verified substantive answer.")
    elif incomplete:
        lines.insert(0, "Partial answer: the available evidence does not adequately cover the whole question.\n")
    remaining = [g for g in gaps if g.status in ("unresolved", "partially_resolved")]
    if remaining:
        lines.extend(["", "## Remaining uncertainty", ""])
        for gap in sorted(remaining, key=lambda g: g.severity != "critical"):
            lines.append(f"- {_plain(gap_uncertainty(gap))}")
    if papers:
        lines.extend(["", "## Sources", ""])
        for selected in papers:
            paper = selected.paper
            identifier = f"DOI: {paper.doi}" if paper.doi else f"{paper.source}: {paper.source_id}"
            url = source_url(selected)
            url_text = f"; {_plain(url)}" if url else ""
            lines.append(
                f"- [{selected.citation_label}] {_plain(paper.title)} "
                f"({paper.publication_year if paper.publication_year is not None else 'year unavailable'}); "
                f"{_plain(identifier)}; {selected.source_type}{url_text}"
            )
    return "\n".join(lines)
