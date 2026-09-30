"""Deterministic summaries of validated relations; no originality inference."""

from researchpilot.claim_check import (
    BoundaryFinding, ClaimLiteratureMap, NoveltyBoundary, RELATION_FIELDS, query_key,
)


def build_maps(decomposition, relations, traces):
    maps = []
    for assertion in decomposition.assertions:
        rows = [r for r in relations if r.claim_id == assertion.claim_id]
        substantial = [r for r in rows if r.relation != "ADJACENT" and r.confidence != "low"]
        queries = [t for t in traces if t.claim_id == assertion.claim_id]
        completed = {query_key(t.query) for t in queries if t.status == "completed"}
        failed = any(t.status == "failed" for t in queries)
        searched = len(completed) >= 2 and not failed
        fields = {field: [r.paper_id for r in rows if r.relation == label]
                  for label, field in RELATION_FIELDS.items()}
        # Coverage is a bounded-search diagnostic, never a truth/novelty score.
        strong = searched and (any(r.relation == "DIRECT_OVERLAP" for r in substantial)
            or (len(substantial) >= 3 and any(r.relation == "PARTIAL_OVERLAP" for r in substantial)
                and any(r.relation in ("SUPPORTING", "BRIDGING") for r in substantial)))
        moderate = searched and len(substantial) >= 2
        coverage = "strong" if strong else "moderate" if moderate else "weak"
        limits = ["Coverage is limited to the bounded OpenAlex search and accessible evidence."]
        if not searched:
            limits.append("Fewer than two successful distinct searches or an incomplete search; absence is inconclusive.")
        if not substantial:
            limits.append("No sufficiently confident substantive relation established from the available passages.")
        if not fields["direct_overlap_papers"]:
            limits.append("No direct overlap established; unexamined literature and missing evidence remain possible.")
        if assertion.importance == "supporting":
            limits.append("Supporting assertion assessed opportunistically; no dedicated exhaustive search.")
        maps.append(ClaimLiteratureMap(claim_id=assertion.claim_id, claim_text=assertion.text,
            **fields, no_close_match_found=(coverage != "weak" and not fields["direct_overlap_papers"]
                                           and not fields["partial_overlap_papers"]),
            coverage_status=coverage, search_limitations=limits))
    return maps


def build_boundary(decomposition, relations, maps):
    fields = {name: [] for name in NoveltyBoundary.model_fields if name != "caveat"}
    def finding(text, claim_ids, rows):
        return BoundaryFinding(text=text, claim_ids=list(dict.fromkeys(claim_ids)),
            paper_ids=list(dict.fromkeys(r.paper_id for r in rows)),
            evidence_ids=list(dict.fromkeys(k for r in rows for k in r.evidence_ids)))
    for assertion in decomposition.assertions:
        rows = [r for r in relations if r.claim_id == assertion.claim_id and r.relation != "ADJACENT"]
        for row in rows:
            if row.relation in ("DIRECT_OVERLAP", "SUPPORTING", "BRIDGING"):
                # The finding is the evidenced *narrow relation*, not a certification
                # that the user's entire (possibly stronger) assertion is established.
                fields["established_components"].append(finding(
                    f"Evidence for {assertion.claim_id} ({row.relation}): {row.relation_summary}", [assertion.claim_id], [row]))
            elif row.relation == "PARTIAL_OVERLAP":
                fields["partially_established_components"].append(finding(
                    row.relation_summary + " Limits: " + "; ".join(row.differences), [assertion.claim_id], [row]))
            elif row.relation == "CONTRADICTING":
                fields["contradictions"].append(finding(row.relation_summary, [assertion.claim_id], [row]))
        direct = [r for r in rows if r.relation == "DIRECT_OVERLAP"]
        if assertion.depends_on and direct:
            fields["known_combinations"].append(finding(
                "Retrieved evidence overlaps the combined proposition " + assertion.claim_id + ": " + direct[0].relation_summary,
                [*assertion.depends_on, assertion.claim_id], direct))
        if not direct:
            m = next(m for m in maps if m.claim_id == assertion.claim_id)
            fields["missing_evidence"].append(finding(
                f"{assertion.claim_id}: direct evidence for the full proposition was not established within this run; "
                + ("partial overlap was found, but it does not establish the full proposition." if m.partial_overlap_papers
                   else "no close match was identified within retrieved literature." if m.no_close_match_found
                   else "coverage remains insufficient for a no-close-match conclusion."),
                [assertion.claim_id], rows))
            ingredient_rows = [r for r in relations if r.claim_id in assertion.depends_on
                               and r.relation in ("DIRECT_OVERLAP", "SUPPORTING", "BRIDGING", "PARTIAL_OVERLAP")]
            if len(assertion.depends_on) >= 2 and all(any(r.claim_id == dep for r in ingredient_rows) for dep in assertion.depends_on):
                fields["potentially_underexplored_connections"].append(finding(
                    "Retrieved passages cover ingredients " + ", ".join(assertion.depends_on)
                    + " separately, while the full connection in " + assertion.claim_id
                    + " was not directly established. This is a potential boundary to investigate, not proof of novelty.",
                    [*assertion.depends_on, assertion.claim_id], ingredient_rows))
    return NoveltyBoundary(**fields)


def closest_papers(relations, evidence, limit=8):
    index = {p.passage_id: p for p in evidence}
    tiers = {"DIRECT_OVERLAP": 5, "PARTIAL_OVERLAP": 4, "CONTRADICTING": 4,
             "SUPPORTING": 3, "BRIDGING": 3, "METHOD_SIMILAR": 2, "ADJACENT": 1}
    best = {}
    for r in relations:
        if r.relation == "ADJACENT":
            continue
        strength = (tiers[r.relation], {"high": 3, "medium": 2, "low": 1}[r.confidence],
                    sum(d.alignment == "same" and bool(d.evidence_ids) for d in r.comparisons),
                    any(index[key].source_type == "pdf" for key in r.evidence_ids))
        if r.paper_id not in best or best[r.paper_id] < strength:
            best[r.paper_id] = strength
    # Stable first appearance for ties; not a novelty number or nearest-neighbor metric.
    return sorted(best, key=best.get, reverse=True)[:limit]
