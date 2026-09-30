"""Additive evidence and immutable supported findings across targeted rescue."""

import re
from typing import Literal

from researchpilot.evidence import AnswerDraft, EvidenceModel, Text, VerifiedClaim
from researchpilot.passage_retrieval import _tokens


class DowngradeAudit(EvidenceModel):
    claim_id: Text
    reason: Literal["conflicting_new_evidence", "prior_support_invalidated",
                    "claim_text_changed", "evidence_lost", "verifier_changed"]
    old_status: str = "supported"
    new_status: Text
    prior_evidence_ids: list[str]
    current_evidence_ids: list[str]


def assert_evidence_union(previous, current):
    for key, passage in previous.items():
        if current.get(key) != passage:
            raise ValueError(f"evidence_lost: original passage {key} was removed or altered")


def recheck_supported(record, additions, evidence, gaps):
    """Recheck an unchanged finding only for a targeted gap or potential conflict.

    This gate never judges support. Only the unchanged atomic verifier can do so.
    Same-paper topical negations are deliberately conservative conflict triggers.
    """
    if any(record.claim.claim_id in g.related_claim_ids for g in gaps):
        return True
    papers = {evidence[key].paper_id for key in record.claim.evidence_ids}
    topic = set(_tokens(record.claim.text))
    return any(p.paper_id in papers and len(topic & set(_tokens(sentence))) >= 3
               and re.search(r"\b(?:not|fails?|counterexample|contradict\w*|incorrect)\b", sentence, re.I)
               for p in additions for sentence in re.split(r"(?<=[.!?])\s+", p.text))


def merge_targeted_draft(previous: list[VerifiedClaim], proposed: AnswerDraft,
                         additions, evidence, bundles=()) -> AnswerDraft:
    """Preserve supported IDs/text; retain old evidence for each updated target.

    A supplied target ID is required for replacement. New IDs remain new claims;
    collisions with supported findings cannot overwrite them. Omitted unresolved
    findings remain available for re-verification, never silently disappear.
    """
    updates = {c.claim_id: c for c in proposed.claims}
    result = []
    old_ids = {c.claim.claim_id for c in previous}
    for record in previous:
        old = record.claim
        if record.verification.status == "supported":
            result.append(old.model_copy(deep=True))
            continue
        candidate = updates.get(old.claim_id, old).model_copy(deep=True)
        papers = {evidence[key].paper_id for key in old.evidence_ids}
        extra = [p.passage_id for p in additions if p.paper_id in papers]
        extra += [m.evidence_id for b in bundles if old.claim_id in b.target_claim_ids for m in b.members]
        candidate.evidence_ids = list(dict.fromkeys([*old.evidence_ids, *candidate.evidence_ids, *extra]))
        result.append(candidate)
    for claim in proposed.claims:
        if claim.claim_id not in old_ids and len(result) < 12:
            result.append(claim)
    return AnswerDraft(claims=result, limitations=proposed.limitations)


def audit_downgrade(previous: VerifiedClaim, current: VerifiedClaim):
    if not set(previous.claim.evidence_ids) <= set(current.claim.evidence_ids):
        raise ValueError("evidence_lost: previously cited evidence omitted from verification")
    if previous.verification.status != "supported" or current.verification.status == "supported":
        return None
    if previous.claim.text != current.claim.text:
        reason = "claim_text_changed"
    elif current.verification.status == "conflicting":
        reason = "conflicting_new_evidence"
    else:
        reason = "prior_support_invalidated"
    return DowngradeAudit(claim_id=previous.claim.claim_id, reason=reason,
        new_status=current.verification.status, prior_evidence_ids=previous.claim.evidence_ids,
        current_evidence_ids=current.claim.evidence_ids)
