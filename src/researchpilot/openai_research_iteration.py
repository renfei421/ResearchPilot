"""Structured coverage assessment and gap-directed follow-up planning."""

from researchpilot.evidence import EvidencePassage, VerifiedClaim, passage_index, require_known_ids
from researchpilot.openai_evidence_client import _BOUNDARY, _StructuredClient, _passages_payload
from researchpilot.paper_candidate import SearchQuery
from researchpilot.research_iteration import (
    EvidenceAssessment, EvidenceGap, EvidenceGapRecord, FollowUpSearchPlan, new_follow_up_queries,
    validate_assessment,
)


_COVERAGE = _BOUNDARY + """Assess whether the current verified claims and evidence
adequately answer the ORIGINAL research question. Consider completely uncovered
aspects as well as meaningful unsupported, partially supported, or conflicting
claims. A support status is an assessment, not new source evidence. Do not treat
the number of papers or claims as coverage. Return sufficient=true only when
the meaningful aspects are covered without unresolved material gaps; then gaps
must be empty. A completely uncovered aspect may have no related_claim_ids.
Use only actual claim IDs for other gaps.
If any drafted answer claim is unsupported, partially supported or conflicting,
return sufficient=false and identify the missing support or unresolved conflict;
do not override its verification status by declaring the whole draft sufficient.

Identify missing INFORMATION in at most 8 concise gaps with unique gap IDs.
critical means needed for answering the question; useful means an important
qualification/comparison is still missing. search_focus must describe the
specific missing information to search for, not a generic topic repetition.
Bad gap: 'Need more evidence'. Good gap: 'Evidence shows trainable-parameter
reduction but does not establish GPU optimizer-memory savings.'
Do not invent findings, paper titles, or a reason to keep searching when the
question is already covered. Return only concise coverage rationale and gaps,
not private reasoning or a chain-of-thought.

The prior_gaps ledger is persistent. Return exactly one gap_updates decision for
EVERY supplied prior gap, using its original gap_id: unresolved,
partially_resolved, resolved, or superseded. Missing from the new gaps list does
not mean resolved. Cite actual selected evidence IDs for any resolution/partial
resolution and briefly say what is now supported or still missing. Preserve
unresolved portions explicitly. superseded is only for replacement by another
explicit active gap; provide superseded_by, otherwise null. Do not change core
gaps into peripheral gaps to declare success. Related claim IDs refer to CURRENT
claims; leave empty when no current claim addresses the gap.
gaps contains NEW gaps only (no paraphrased duplicates of prior gaps). Anchor
each to the ORIGINAL question. Set relevance_to_question=peripheral for optional
extensions/new subtopics outside it; such gaps must not force another round.
A useful qualification of an original-question finding may still be core.
Judge only verified supported portions of partial claims as findings.
No extra search is needed just to strengthen an already qualified answer.
"""

_FOLLOW_UP = _BOUNDARY + """Plan NEW academic searches to fill the supplied
evidence gaps. Do not repeat the initial broad search strategy. Every query must
target one actual gap_id, with a concise rationale for that missing information.
Return 1 to max_queries queries (at most 3), or an empty list when no useful new
search is possible. Prefer critical gaps. Use single-line academic Boolean or
keyword expressions, not URLs or API parameters. Do not include publication-year
filters; the caller retains the original bounds. Query texts must be distinct
after stripping, collapsing whitespace, and casefold, including against ALL
previously executed queries. Query IDs must also be new across the entire run.
Example: for missing optimizer-memory evidence, search '"LoRA" AND "optimizer
memory" AND "fine-tuning"', not a generic repetition of parameter-efficient
fine-tuning. Use the actual supplied gap, not this example, to plan. Never use
outside findings as evidence. Return only query metadata, not private reasoning.
"""


class OpenAIEvidenceAssessor(_StructuredClient):
    def assess(self, question: str, claims: list[VerifiedClaim],
               evidence: list[EvidencePassage], *,
               prior_gaps: list[EvidenceGapRecord] = ()) -> EvidenceAssessment:
        available = passage_index(evidence)
        for record in claims:
            require_known_ids(record.claim.evidence_ids, available, "claim evidence IDs")
        result = self._parse(_COVERAGE, {
            "research_question": question,
            "verified_claims": [c.model_dump() for c in claims],
            "evidence": _passages_payload(evidence),
            "prior_gaps": [g.model_dump() for g in prior_gaps],
        }, EvidenceAssessment)
        validate_assessment(result, claims, evidence, prior_gaps)
        return result


class OpenAIFollowUpPlanner(_StructuredClient):
    def plan(self, question: str, gaps: list[EvidenceGap], executed_queries: list[SearchQuery],
             concepts: list[str], max_queries: int = 3) -> FollowUpSearchPlan:
        if type(max_queries) is not int or not 1 <= max_queries <= 3:
            raise ValueError("max_queries must be an integer between 1 and 3.")
        result = self._parse(_FOLLOW_UP, {
            "research_question": question, "evidence_gaps": [g.model_dump() for g in gaps],
            "executed_queries": [q.model_dump() for q in executed_queries],
            "concepts": concepts, "max_queries": max_queries,
        }, FollowUpSearchPlan)
        # Validate links/budget now; retain original proposals for the audit trace.
        new_follow_up_queries(result, gaps, executed_queries, max_queries)
        return result
