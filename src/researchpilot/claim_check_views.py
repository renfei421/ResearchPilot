"""Deterministic, escaped presentations of the same persisted Idea Check result."""

from html import escape

from researchpilot.answer_renderer import _plain, source_url
from researchpilot.claim_check import ClaimCheckResult, CAVEAT


def _sections(result):
    """Text plus actual evidence IDs; formatting cannot invent citations."""
    sources = {s.paper.paper_id: s for s in result.sources}
    sections = [("Your Research Claim", [(result.request.claim, [])])]
    if result.request.context:
        sections[0][1].append(("Context: " + result.request.context, []))
    sections.append(("Claim Decomposition", [(f"{a.claim_id} [{a.importance}, {a.claim_type}]: {a.text}"
        + (" Depends on: " + ", ".join(a.depends_on) if a.depends_on else ""), []) for a in result.decomposition.assertions]))
    sections.append(("Closest Prior Work", [(f"{sources[pid].citation_label}: {sources[pid].paper.title}",
        list(dict.fromkeys(key for r in result.relations if r.paper_id == pid and r.relation != "ADJACENT" for key in r.evidence_ids)))
        for pid in result.closest_prior_work] or [("No substantive prior-work relation established in the retrieved evidence.", [])]))
    rows = []
    for m in result.claim_maps:
        rows.append((f"{m.claim_id}: {m.claim_text} — coverage: {m.coverage_status}", []))
        if m.no_close_match_found:
            rows.append(("No close match was identified within the retrieved literature; this is not proof of novelty.", []))
        for r in result.relations:
            if r.claim_id != m.claim_id or r.relation == "ADJACENT":
                continue
            rows.append((f"{sources[r.paper_id].citation_label} — {r.relation} ({r.confidence}): {r.relation_summary}", r.evidence_ids))
            if r.matched_dimensions:
                rows.append(("Same / useful: " + "; ".join(r.matched_dimensions), r.evidence_ids))
            if r.differences:
                rows.append(("Different / not established: " + "; ".join(r.differences), r.evidence_ids))
            if r.scope_notes:
                rows.append(("Scope: " + "; ".join(r.scope_notes), r.evidence_ids))
        rows.extend((text, []) for text in m.search_limitations)
    sections.append(("Claim-by-Claim Literature Map", rows))
    rows = []
    for field, title in (("established_components", "Evidenced components"),
                         ("partially_established_components", "Partially established components"),
                         ("known_combinations", "Known combinations"),
                         ("potentially_underexplored_connections", "Potentially underexplored connections")):
        rows.extend((title + ": " + f.text, f.evidence_ids) for f in getattr(result.novelty_boundary, field))
    rows.append((CAVEAT, []))
    sections.append(("Potential Novelty Boundary", rows))
    sections.append(("Contradictory / Limiting Evidence",
        [(f.text, f.evidence_ids) for f in result.novelty_boundary.contradictions + result.novelty_boundary.missing_evidence]
        or [("No comparable-scope contradiction was established in this run. This is not proof of consistency.", [])]))
    sections.append(("Search Limitations", [(text, []) for text in result.limitations + result.warnings]))
    stats = result.run_stats
    sections.append(("Search Trace", [(f"{stats.assertions} assertions; {stats.raw_candidates} candidate records; "
        f"{stats.papers_evaluated} papers evaluated; {stats.search_rounds} rounds; {len(result.evidence)} evidence passages; "
        f"{stats.visual_evidence_calls} vision calls; {stats.elapsed_seconds:.1f}s. Termination: {result.termination_reason}", []),
        ("Relations: " + "; ".join(f"{key}: {value}" for key, value in stats.relation_counts.items()), [])]
        + [(f"Round {t.round} / {t.claim_id} / {t.role}: {t.query} ({'reused' if t.reused else 'executed'}, {t.status}, "
            f"{len(t.candidate_ids)} records)", []) for t in result.search_trace]))
    return sections


def _references(result):
    sources = {s.paper.paper_id: s for s in result.sources}
    evidence = {p.passage_id: p for p in result.evidence}
    def token(key):
        p = evidence[key]
        place = f"p. {p.page_number}" if p.source_type == "pdf" else "abstract"
        return f"[{sources[p.paper_id].citation_label}, {place}]"
    return evidence, token


def claim_markdown(result: ClaimCheckResult) -> str:
    evidence, token = _references(result)
    lines = ["# ResearchPilot — Idea Check", ""]
    for title, rows in _sections(result):
        lines += ["## " + title, ""]
        for text, ids in rows:
            lines.append("- " + _plain(text) + (" " + " ".join(dict.fromkeys(token(key) for key in ids)) if ids else ""))
        lines.append("")
    lines += ["## Sources", ""]
    for s in result.sources:
        url = source_url(s)
        lines.append(f"- [{s.citation_label}] {_plain(s.paper.title)} ({s.paper.publication_year or 'year unavailable'})"
                     + (f" — {url}" if url else "") + f" — {s.source_type}")
    lines += ["", "## Cited evidence", ""]
    cited = dict.fromkeys(key for _, rows in _sections(result) for _, ids in rows for key in ids)
    for key in cited:
        lines += [f"### {token(key)}", "", _plain(evidence[key].text), ""]
    return "\n".join(lines)


def claim_html(result: ClaimCheckResult) -> str:
    evidence, token = _references(result)
    cited = dict.fromkeys(key for _, rows in _sections(result) for _, ids in rows for key in ids)
    anchors = {key: f"idea-evidence-{i}" for i, key in enumerate(cited, 1)}
    html = ['<article class="idea-result">']
    for title, rows in _sections(result):
        html.append(f"<section><h2>{escape(title)}</h2><ul>")
        for text, ids in rows:
            refs = " ".join(f'<a class="citation" href="#{anchors[key]}">{escape(token(key))}</a>' for key in ids)
            html.append(f"<li>{escape(text)} {refs}</li>")
        html.append("</ul></section>")
    html.append("<section><h2>Sources</h2><ul>")
    for s in result.sources:
        url = source_url(s)
        title = escape(s.paper.title)
        if url:
            title = f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{title}</a>'
        html.append(f"<li>[{escape(s.citation_label)}] {title} — {s.paper.publication_year or 'year unavailable'} — {s.source_type}</li>")
    html.append("</ul></section><section><h2>Cited evidence</h2>")
    for key in cited:
        p = evidence[key]
        note = " · visual transcription" if key.startswith("visual:") else ""
        html.append(f'<details id="{anchors[key]}" class="evidence"><summary tabindex="0">{escape(token(key)+note+" — "+p.title)}</summary>'
                    f'<blockquote>{escape(p.text)}</blockquote></details>')
    return "".join(html) + "</section></article>"
