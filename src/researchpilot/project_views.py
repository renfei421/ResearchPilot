"""Deterministic, escaped project views; no model or generated manuscript."""

from html import escape
from copy import deepcopy


def project_export_data(data):
    """Export durable project state and result audits without retrieval buffers.

    Query caches and pre-selection passages remain in the local store for reuse
    and run diagnostics. They are not project findings or verified evidence.
    Never alter those original records while preparing an export.
    """
    exported = deepcopy(data)
    for query in exported.get("queries", []):
        query.pop("candidates", None)
    for run in exported.get("runs", []):
        if run.get("result") is not None:
            run["result"].pop("retrieved_passages", None)
    return exported


def project_html(data):
    def e(value):
        return escape(str(value), quote=True)
    project = data["project"]
    active = project["status"] == "active"
    parts = [f'<h2>{e(project["title"])}</h2><p>{e(project["description"])}</p>',
             f'<p class="muted">{e(project.get("field") or "")} · {e(project["status"])} · {e(project["project_id"])}</p>']
    def section(key, title, render):
        parts.append(f'<section class="project-section" id="project-{key}"><h3>{title}</h3>')
        parts.extend(render(row) for row in data[key])
        if not data[key]:
            parts.append('<p class="muted">暂无记录</p>')
        parts.append('</section>')
    def action(kind, key, mode="research"):
        if not active:
            return ""
        return f'<button class="text-button" type="button" data-continue="{kind}" data-id="{e(key)}" data-mode="{mode}">{"继续检查主张" if mode == "claim_check" else "继续研究"}</button>'
    def archive(kind, key):
        return f'<button class="text-button" type="button" data-archive="{kind}" data-id="{e(key)}">归档</button>' if active else ""
    def refs(ids):
        return " ".join(f'<a href="#project-evidence-{e(key)}" data-evidence-link>{e(key[:22])}</a>' for key in ids)
    def question(row):
        return f'<article><p>{e(row["text"])}</p><small>{e(row["status"])}</small> ' + (action("question", row["question_id"]) + archive("questions", row["question_id"]) if row["status"] != "archived" else "") + '</article>'
    def claim(row):
        cid = row["claim_id"]
        findings = [f["text"] for f in data["findings"] if cid in f["claim_ids"]]
        gaps = [g["description"] for g in data["gaps"] if cid in g["related_claim_ids"] and g["status"] in ("unresolved", "partially_resolved")]
        return f'<article><p>{e(row["text"])}</p><small>{e(row["status"])} · {e(row["claim_type"])}</small><p>{refs(row["evidence_ids"])}</p>' + ''.join(f'<p class="muted">文献关系／发现：{e(f)}</p>' for f in findings) + ''.join(f'<p class="muted">缺口：{e(g)}</p>' for g in gaps) + (action("claim", cid, "claim_check") + action("claim", cid) + archive("claims", cid) if row["status"] != "archived" else "") + '</article>'
    def paper(row):
        p = row["paper"]
        ids = [v["evidence_id"] for v in data["evidence"] if v["passage"]["paper_id"] == row["paper_id"]]
        return f'<article><strong>{e(p["title"])}</strong><p>{e(p["publication_year"] or "年份未知")} · {e(p["doi"] or p["source_id"])} · {e(p["source"])}</p><small>{len(ids)} 条证据 · 首次运行 {e(row["first_seen_run_id"])}</small><p>{refs(ids)}</p></article>'
    def evidence(row):
        p = row["passage"]
        return f'<details id="project-evidence-{e(row["evidence_id"])}"><summary>{e(p["title"])} · {e(p["source_type"])} · {e(p["page_number"] or "摘要")} · {e(row["verification_status"])}</summary><p class="passage-text">{e(p["text"])}</p><small>来源：{e(p.get("source_url") or p["paper_id"])} · 原始运行 {e(row["originating_run_id"])}</small></details>'
    def gap(row):
        return f'<article><p>{e(row["description"])}</p><p class="muted">{e(row["search_focus"])}</p><small>{e(row["status"])} · {e(row["severity"])}</small><p>{e(row["resolution_reason"])} {refs(row["evidence_ids"])}</p>' + (action("gap", row["gap_id"]) + archive("gaps", row["gap_id"]) if row["status"] in ("unresolved", "partially_resolved") else "") + '</article>'
    section("questions", "研究问题", question)
    section("claims", "主张与假设", claim)
    section("papers", "文献", paper)
    section("evidence", "已核验的证据与文献关系", evidence)
    section("findings", "证据支持的发现", lambda f: f'<article><p>{e(f["text"])}</p><small>{e(f["status"])} · 原始运行 {e(f["originating_run_id"])}</small><p>{refs(f["evidence_ids"])}</p></article>')
    section("gaps", "研究缺口", gap)
    section("notes", "用户笔记 · 未核验", lambda n: f'<article><p>{e(n["text"])}</p><small>{e(n["created_at"])}</small></article>')
    section("runs", "项目历史", lambda r: f'<article><button class="text-button" type="button" data-run="{e(r["run_id"])}">{e(r["question"])}</button><small>{e(r["mode"])} · {e(r["status"])}</small></article>')
    return '\n'.join(parts)


def project_markdown(data):
    p = data["project"]
    lines = [f'# {p["title"]}', '', p["description"], '', f'Project: {p["project_id"]} | Status: {p["status"]}',
             f'Created: {p["created_at"]} | Updated: {p["updated_at"]}', '',
             'Historical evidence is scoped to its original claims. User notes are unverified.', '']
    for key, title in [("questions", "Research Questions"), ("claims", "Claims"), ("findings", "Evidence-grounded Findings"),
                       ("gaps", "Research Gaps"), ("papers", "Papers"), ("evidence", "Evidence"),
                       ("notes", "User Notes (unverified)"), ("runs", "Run History")]:
        lines += ['## ' + title, '']
        for row in data[key]:
            if key == "papers":
                body = f'{row["paper"]["title"]} ({row["paper"].get("publication_year")}) — {row["paper_id"]}; first run {row["first_seen_run_id"]}'
            elif key == "evidence":
                v = row["passage"]
                body = f'{row["evidence_id"]} [{row["verification_status"]}] {v["title"]}, {v["source_type"]} page {v["page_number"]}; original run {row["originating_run_id"]}\n\n  {v["text"]}'
            elif key == "runs":
                body = f'{row["run_id"]} [{row["mode"]}, {row["status"]}]: {row["question"]}'
            else:
                body = f'[{row.get("status", "unverified")}] {row.get("text", row.get("description", ""))}'
                if row.get("evidence_ids"):
                    body += '\n\n  Evidence: ' + ', '.join(row["evidence_ids"])
                if row.get("resolution_reason"):
                    body += '\n\n  Resolution: ' + row["resolution_reason"]
            lines += ['- ' + body, '']
        if not data[key]:
            lines += ['No records.', '']
    return '\n'.join(lines)
