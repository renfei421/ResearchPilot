"""Conservative availability guards and persistent gaps; never promote a verdict."""

from hashlib import sha256
import re

from researchpilot.evidence import ClaimVerification, aggregate_assertions
from researchpilot.evidence_bundle import EvidenceBundle
from researchpilot.evidence_assembly import references, declarations
from researchpilot.research_iteration import EvidenceGapRecord


def bundle_gap_id(bundle: EvidenceBundle) -> str:
    return "assembly:" + sha256((bundle.paper_id + "\n" + bundle.purpose).encode()).hexdigest()[:20]


def guard_bundle_support(verification: ClaimVerification, bundles: list[EvidenceBundle], evidence=()) -> ClaimVerification:
    result = verification.model_copy(deep=True)
    changed = False
    for atom in result.assertions:
        if atom.status != "supported" or atom.kind not in ("theoretical", "quantitative", "causal"):
            continue
        cited = [p for p in evidence if p.passage_id in atom.evidence_ids]
        cited_refs = {key for p in cited for key in references(p.text)}
        cited_pages = {(p.paper_id, p.page_number) for p in cited}
        for bundle in bundles:
            if bundle.complete or not set(atom.evidence_ids) & {m.evidence_id for m in bundle.members}:
                continue
            # An incomplete curvature chain must not suppress an independently
            # supported graph-assumption atom. Only guard the missing dimension.
            relevant = []
            for missing in bundle.missing_components:
                if "formula" in missing and atom.kind == "quantitative":
                    page = re.search(r"page (\d+)", missing)
                    if not evidence or (page and (bundle.paper_id, int(page[1])) in cited_pages):
                        relevant.append(missing)
                for target, pattern in (("curvature", r"curvature"), ("recovery", r"recover")):
                    if target in missing and re.search(pattern, atom.text, re.I):
                        has_result = any(re.search(pattern, p.text, re.I) and any(
                            role in ("theorem", "proposition", "consequence")
                            for role in declarations(p.text).values()) for p in cited)
                        if not has_result:
                            relevant.append(missing)
                if missing.startswith("statement "):
                    key = missing.split()[1]
                    # A statement can be reported without every lemma of its
                    # proof. Required conditions/definitions are different:
                    # they constrain what that statement actually establishes.
                    if (key.startswith(("assumption:", "condition:", "definition:"))
                            and (not evidence or key in cited_refs) and atom.kind == "theoretical"):
                        relevant.append(missing)
            if relevant:
                atom.status = "partially_supported"
                atom.scope_supported = False
                atom.reason += " Incomplete cited evidence chain: " + "; ".join(relevant) + "."
                changed = True
                break
    if changed and result.status not in ("unsupported", "conflicting"):
        result.status = aggregate_assertions(result.assertions)
        result.reason += " Missing bundle components prevent unconditional formal support."
    return result


def reconcile_bundle_gaps(ledger: list[EvidenceGapRecord], bundles: list[EvidenceBundle],
                          round_number: int, claims=()) -> list[EvidenceGapRecord]:
    """Missing evidence cannot be resolved by an unrelated supported finding.

    Completion alone never closes a gap. An explicit assessor resolution and a
    supported atomic finding citing the assembled chain are both required.
    """
    records = {g.gap_id: g.model_copy(deep=True) for g in ledger}
    by_gap = {}
    for b in bundles:
        by_gap.setdefault(bundle_gap_id(b), []).append(b)
    for key, items in by_gap.items():
        missing = list(dict.fromkeys(m for b in items for m in b.missing_components))
        supported_chain = any(c.verification.status == "supported" and any(
            b.complete and {m.evidence_id for m in b.members if m.passage_id} <= set(a.evidence_ids)
            for b in items for a in c.verification.assertions if a.status == "supported"
            and a.kind == "theoretical") for c in claims)
        if missing:
            description = "Incomplete mathematical evidence chain in " + items[0].title + ": " + "; ".join(missing)
            if key not in records:
                records[key] = EvidenceGapRecord(gap_id=key, description=description,
                    search_focus=items[0].purpose, related_claim_ids=[], severity="critical",
                    first_seen_round=round_number, last_updated_round=round_number)
            record = records[key]
            record.status, record.superseded_by = "unresolved", None
            record.resolution_reason = description
            record.last_updated_round = round_number
        elif key in records:
            record = records[key]
            if record.status in ("resolved", "superseded") and not supported_chain:
                record.status, record.superseded_by = "unresolved", None
                record.resolution_reason = "Components located; scoped joint atomic support is still required."
    return list(records.values())
