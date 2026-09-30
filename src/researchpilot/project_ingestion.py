"""Transactional ingestion of validated final results, never transient candidate output."""

import json

from researchpilot.evidence import supported_findings
from researchpilot.project_models import (
    EvidenceAttestation, ProjectClaim, ProjectEvidence, ProjectFinding, ProjectGap,
    ProjectPaper, identity, now, text_key,
)


def union(*values):
    return list(dict.fromkeys(x for group in values for x in group))


def ingest_result(store, db, run_id, request, result, *, queries=(), document_refs=None):
    from researchpilot.claim_check import ClaimCheckRequest, ClaimCheckResult
    from researchpilot.research_models import ResearchResult

    if request.project_id is None:
        return {"new_papers": 0, "new_evidence": 0, "gaps_resolved": 0}
    project_id = request.project_id
    project = store._project(db, project_id)
    existing = db.execute("SELECT counts_json FROM project_ingestions WHERE project_id=? AND run_id=?",
                          (project_id, run_id)).fetchone()
    if existing:
        return json.loads(existing[0])
    idea = isinstance(request, ClaimCheckRequest)
    result = (ClaimCheckResult if idea else ResearchResult).model_validate(result.model_dump())
    if (result.request != request if idea else result.question != request.question):
        raise ValueError("Cannot ingest a result for a different project request.")
    row = db.execute("SELECT project_id,status FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None or row["project_id"] != project_id or row["status"] not in ("running", "completed"):
        raise ValueError("Project ingestion requires an associated completing run.")
    timestamp = now()
    counts = {"new_papers": 0, "new_evidence": 0, "gaps_resolved": 0}
    papers = result.sources if idea else result.papers
    evidence = {p.passage_id: p for p in result.evidence}
    visuals = {v.evidence.evidence_id: v for v in result.visual_evidence}
    qids = [request.target_question_id] if request.target_question_id else []
    if request.target_gap_id:
        target = store._get(db, project_id, "gaps", request.target_gap_id)
        if target is None:
            raise ValueError("Unknown project gap target.")
        qids = target.related_question_ids

    for source in papers:
        pid = source.paper.paper_id
        prior = store._get(db, project_id, "papers", pid)
        counts["new_papers"] += prior is None
        # Preserve first-seen identity; missing metadata can be enriched without merging versions.
        paper = source.paper
        if prior:
            fields = prior.paper.model_dump()
            for key, val in paper.model_dump().items():
                if fields.get(key) in (None, "", []) and val not in (None, "", []):
                    fields[key] = val
            paper = type(paper).model_validate(fields)
        record = ProjectPaper(project_id=project_id, paper_id=pid, paper=paper,
            version_group_id=source.group_id, first_seen_run_id=prior.first_seen_run_id if prior else run_id,
            local_document_reference=(document_refs or {}).get(pid) or (prior.local_document_reference if prior else None),
            created_at=prior.created_at if prior else timestamp, updated_at=timestamp, last_used_at=timestamp)
        store._put(db, "papers", record)

    claim_map, claim_rows = {}, {}
    inputs = [(a.claim_id, a.text, a.claim_type) for a in result.decomposition.assertions] if idea else [
        (c.claim.claim_id, c.claim.text, "research_finding") for c in result.claims]
    saved_claims = store._list(db, project_id, "claims")
    for cid, text, kind in inputs:
        prior = next((c for c in saved_claims if text_key(c.text) == text_key(text)), None)
        key = prior.claim_id if prior else identity("claim", project_id, text_key(text))
        claim_map[cid] = key
        row = prior or ProjectClaim(project_id=project_id, claim_id=key, text=text, claim_type=kind,
            parent_claim_id=request.target_claim_id if key != request.target_claim_id else None,
            originating_run_id=run_id, status="proposed" if idea else "unresolved")
        store._put(db, "claims", row)
        claim_rows[cid] = row

    evidence_map = {}
    def trusted(ids, cid, *, kind, reason):
        saved = []
        for key in ids:
            passage = evidence[key]
            eid = identity("evidence", passage.paper_id, passage.source_type, passage.page_number, passage.text)
            prior = store._get(db, project_id, "evidence", eid)
            attestation = EvidenceAttestation(run_id=run_id, claim_id=claim_map[cid], kind=kind,
                status="supported" if kind == "atomic_verification" else "relation_assessed", reason=reason)
            attestations = list(prior.attestations) if prior else []
            if attestation not in attestations:
                attestations.append(attestation)
            status = "supported" if kind == "atomic_verification" or prior and prior.verification_status == "supported" else "relation_assessed"
            row = ProjectEvidence(project_id=project_id, evidence_id=eid,
                passage=prior.passage if prior else passage,
                related_claim_ids=union(prior.related_claim_ids if prior else [], [claim_map[cid]]),
                related_question_ids=union(prior.related_question_ids if prior else [], qids),
                verification_status=status, originating_run_id=prior.originating_run_id if prior else run_id,
                created_at=prior.created_at if prior else timestamp, updated_at=timestamp, attestations=attestations,
                visual_record=prior.visual_record if prior else visuals.get(key))
            store._put(db, "evidence", row)
            counts["new_evidence"] += prior is None
            evidence_map[key] = eid
            saved.append(eid)
        return saved

    def finding(text, cid, eids, status, relation=None):
        key = identity("finding", project_id, text_key(text), claim_map[cid], relation)
        prior = store._get(db, project_id, "findings", key)
        record = ProjectFinding(project_id=project_id, finding_id=key, text=text,
            evidence_ids=union(prior.evidence_ids if prior else [], eids), claim_ids=[claim_map[cid]],
            status=status, originating_run_id=prior.originating_run_id if prior else run_id,
            last_updated_run_id=run_id, created_at=prior.created_at if prior else timestamp,
            updated_at=timestamp, relation=relation)
        store._put(db, "findings", record)

    if idea:
        for relation in result.relations:
            if relation.relation == "ADJACENT" or relation.evidence_quality != "substantive":
                continue
            ids = trusted(relation.evidence_ids, relation.claim_id, kind="prior_work_relation",
                          reason=relation.relation_summary)
            # Persist the narrow, scoped relationship, never promote the proposed claim.
            scope = "; ".join(union(relation.differences, relation.scope_notes))
            finding(f"{relation.relation}: {relation.relation_summary} Scope: {scope}",
                    relation.claim_id, ids, "relation_assessed", relation.relation)
    else:
        for record in result.claims:
            cid, verification = record.claim.claim_id, record.verification
            items = supported_findings(record)
            ids = []
            for text, cited, reason in items:
                saved = trusted(cited, cid, kind="atomic_verification", reason=reason)
                ids = union(ids, saved)
                finding(text, cid, saved, "verified")
            prior = claim_rows[cid]
            # Never copy a relation label or a run completion flag into support status.
            status = "supported" if verification.status == "supported" else "partially_supported" if ids else "unresolved"
            if prior.status == "archived":
                status = "archived"
            store._put(db, "claims", prior.model_copy(update={"status": status,
                "evidence_ids": union(prior.evidence_ids, ids), "updated_at": timestamp}))

    if not idea and any(c.verification.status == "supported" for c in result.claims):
        for qid in qids:
            question = store._get(db, project_id, "questions", qid)
            if question and question.status == "open":
                store._put(db, "questions", question.model_copy(update={"status": "partially_answered", "updated_at": timestamp}))

    if idea:
        gap_inputs = []
        for item in result.novelty_boundary.missing_evidence:
            focus = " ".join(claim_rows[cid].text for cid in item.claim_ids)
            gap_inputs.append(dict(gap_id=identity("gap", project_id, text_key(item.text), text_key(focus)),
                description=item.text, search_focus=focus or item.text, severity="useful",
                related_claim_ids=item.claim_ids, status="unresolved", evidence_ids=[], resolution_reason=""))
    else:
        gap_inputs = [g.model_dump() for g in result.gap_ledger]
        if not gap_inputs and result.evidence_assessment:
            gap_inputs = [g.model_dump() | {"status": "unresolved", "evidence_ids": [], "resolution_reason": ""}
                          for g in result.evidence_assessment.gaps]
    for g in gap_inputs:
        prior = store._get(db, project_id, "gaps", g["gap_id"])
        gid = prior.gap_id if prior else identity("gap", project_id, text_key(g["description"]), text_key(g["search_focus"]))
        prior = prior or store._get(db, project_id, "gaps", gid)
        eids = [evidence_map[key] for key in g["evidence_ids"] if key in evidence_map]
        status = "archived" if g["status"] == "superseded" else g["status"]
        if status in ("resolved", "partially_resolved") and (not eids or len(eids) != len(g["evidence_ids"])):
            status = prior.status if prior else "unresolved"
        if prior and prior.status == "archived":
            status = "archived"
        cids = [claim_map[key] for key in g["related_claim_ids"] if key in claim_map]
        row = ProjectGap(project_id=project_id, gap_id=gid,
            description=prior.description if prior else g["description"], search_focus=prior.search_focus if prior else g["search_focus"],
            severity=prior.severity if prior else g["severity"], status=status,
            first_seen_run_id=prior.first_seen_run_id if prior else run_id, last_updated_run_id=run_id,
            related_claim_ids=union(prior.related_claim_ids if prior else [], cids),
            related_question_ids=union(prior.related_question_ids if prior else [], qids),
            evidence_ids=union(prior.evidence_ids if prior else [], eids),
            resolution_reason=g["resolution_reason"] if eids else prior.resolution_reason if prior else "",
            created_at=prior.created_at if prior else timestamp, updated_at=timestamp)
        store._put(db, "gaps", row)
        counts["gaps_resolved"] += status == "resolved" and (prior is None or prior.status != "resolved")
    for query in queries:
        if query.project_id != project_id or query.originating_run_id != run_id:
            raise ValueError("Query audit belongs to another project or run.")
        store._put(db, "queries", query)
    store._save_project(db, project.model_copy(update={"updated_at": timestamp}))
    db.execute("INSERT INTO project_ingestions VALUES (?,?,?,?)", (project_id, run_id, timestamp, json.dumps(counts)))
    return counts
