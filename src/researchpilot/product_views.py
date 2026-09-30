"""Escaped HTML and Markdown presentations of already verified results."""

from html import escape
import re

from researchpilot.answer_renderer import _plain, rendered_findings, source_url
from researchpilot.research_iteration import active_gaps, gap_uncertainty
from researchpilot.research_models import ResearchResult


def uncertainties(result: ResearchResult) -> list[str]:
    if result.gap_ledger:
        gaps = [gap_uncertainty(g) for g in
                sorted(active_gaps(result.gap_ledger), key=lambda g: g.severity != "critical")]
    else:
        gaps = [gap.description for gap in result.evidence_assessment.gaps] if result.evidence_assessment else []
    return list(dict.fromkeys([*result.warnings, *result.draft_limitations, *gaps]))


def markdown_export(result: ResearchResult) -> str:
    # The plain-text/CLI answer also contains the ledger. Exports present that
    # ledger once, alongside other warnings in their existing uncertainty section.
    answer = re.sub(r"\n## Remaining uncertainty\n.*?(?=\n## Sources|\Z)", "", result.answer, flags=re.S)
    lines = ["# ResearchPilot", "", "## Research question", "", _plain(result.question),
             "", "## Answer and key findings", "", answer, "", "## Remaining uncertainty", ""]
    lines.extend(f"- {_plain(item)}" for item in uncertainties(result))
    if not uncertainties(result):
        lines.append("No remaining uncertainty was reported; this is not a guarantee of completeness.")
    lines.extend(["", f"Termination: {result.termination_reason or 'unavailable'}", ""])
    return "\n".join(lines)


def result_html(result: ResearchResult) -> str:
    sources = {p.paper.paper_id: p for p in result.papers}
    passages = {p.passage_id: p for p in result.evidence}
    citations: dict[str, list] = {}
    for record in result.claims:
        if record.verification.status == "unsupported":
            continue
        for key in dict.fromkeys(key for _, ids in rendered_findings(record) for key in ids):
            passage = passages[key]
            label = sources[passage.paper_id].citation_label
            location = f"p. {passage.page_number}" if passage.source_type == "pdf" else "abstract"
            token = f"[{label}, {location}]"
            found = citations.setdefault(token, [])
            if passage not in found:
                found.append(passage)
    anchors = {token: f"citation-{i}" for i, token in enumerate(citations, 1)}

    def plain(text: str) -> str:
        # Undo only the escaping introduced by the deterministic answer renderer,
        # then HTML-escape; source/model text can never introduce HTML or links.
        return escape(re.sub(r"\\([\\`*_{}\[\]<>#])", r"\1", text))

    def inline(text: str) -> str:
        parts, start = [], 0
        for match in re.finditer(r"(?<!\\)\[P\d+, (?:p\. \d+|abstract)\]", text):
            parts.append(plain(text[start:match.start()]))
            token = match.group()
            if token in anchors:
                parts.append(f'<a class="citation" href="#{anchors[token]}">{escape(token)}</a>')
            else:
                parts.append(escape(token))
            start = match.end()
        return "".join(parts) + plain(text[start:])

    body = ['<section class="answer"><h2>回答与主要发现</h2>']
    # Source cards below use structured metadata, not parsed Markdown links.
    for line in result.answer.split("\n## Sources", 1)[0].split("\n## Remaining uncertainty", 1)[0].splitlines():
        if line.startswith("- "):
            body.append(f'<p class="finding">{inline(line[2:])}</p>')
        elif line.strip():
            body.append(f"<p>{inline(line)}</p>")
    body.append('</section><section><h2>剩余不确定性</h2>')
    notes = uncertainties(result)
    if notes:
        body.append("<ul>" + "".join(f"<li>{escape(note)}</li>" for note in notes) + "</ul>")
    else:
        body.append('<p class="muted">本轮未报告剩余证据缺口；这不保证结论完整无误。</p>')
    body.append('</section><section><h2>引用证据</h2><p class="muted">点击回答中的引用或展开下列条目查看原文。PDF 页码为文件中的物理页码。</p>')
    for token, evidence in citations.items():
        selected = sources[evidence[0].paper_id]
        url = source_url(selected)
        body.append(f'<details class="evidence" id="{anchors[token]}"><summary>{escape(token)} · {escape(selected.paper.title)}</summary>')
        for passage in evidence:
            if passage.passage_id.startswith("visual:"):
                body.append('<p class="muted">目标页视觉转录；其中标注的歧义不作为精确公式依据。</p>')
            body.append(f'<blockquote>{escape(passage.text)}</blockquote>')
        if selected.paper.doi:
            body.append(f"<p>DOI: {escape(selected.paper.doi)}</p>")
        if url:
            body.append(f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">查看来源 ↗</a>')
        body.append("</details>")
    body.append('</section><section><h2>来源文献</h2><div class="source-grid">')
    for selected in result.papers:
        paper, url = selected.paper, source_url(selected)
        kind = {"pdf": "全文 / PDF", "abstract": "仅摘要", "unavailable": "无可用正文"}[selected.source_type]
        body.append(f'<article class="source-card"><span class="tag">{escape(selected.citation_label)} · {kind}</span>')
        body.append(f"<h3>{escape(paper.title)}</h3><p>{paper.publication_year if paper.publication_year is not None else '年份未知'}</p>")
        if paper.doi:
            body.append(f"<p>DOI: {escape(paper.doi)}</p>")
        if url:
            body.append(f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">查看来源 ↗</a>')
        body.append("</article>")
    body.append('</div></section><details class="trace"><summary>运行轨迹</summary>')
    body.append(f"<p>停止原因：{escape(result.termination_reason or 'unavailable')}</p>")
    for trace in result.round_trace:
        body.append(f"<p>Round {trace.round} · {trace.queries} queries · {trace.new_papers} new papers · "
                    f"{trace.new_evidence} new evidence · {trace.supported_claims} supported claims · "
                    f"{trace.remaining_gaps} remaining gaps</p>")
    body.append("</details>")
    return "".join(body)
