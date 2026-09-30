"""Small project fixtures using real bounded agents with synthetic providers."""

from types import SimpleNamespace

from test_research_graph import loop_fixture
from researchpilot.query_plan import SearchPlan
from researchpilot.project_models import ProjectCreate
from researchpilot.project_context import ProjectContextBuilder, project_usage
from researchpilot.project_reuse import ProjectSearchSession
from researchpilot.research_models import ResearchRequest
from researchpilot.research_iteration import EvidenceAssessment, EvidenceGap, GapUpdate

QUESTION = "How does spectral structure affect recovery?"
A_TEXT = "Spectral structure controls recovery under fixed support and regularity assumptions."
B_TEXT = "Spectral recovery under nonuniform sampling additionally requires bounded weighted degrees."


def research_fixture(*, second=False, supported=True):
    f = loop_fixture()
    f.a = f.a.model_copy(update={"title": "Spectral recovery A", "abstract": A_TEXT})
    f.b = f.b.model_copy(update={"title": "Nonuniform spectral recovery B", "abstract": B_TEXT})
    f.raw.search_works.side_effect = lambda **kw: [f.b] if second else [f.a]
    f.planner.plan.side_effect = lambda r: SearchPlan(research_question=r.research_question,
        concepts=["spectral", "recovery"], queries=[dict(query_id=f"q{i}", text=text, role=role, rationale="Synthetic facet")
            for i, (text, role) in enumerate([( "spectral recovery", "core"), ("fixed support", "facet"), ("nonuniform recovery", "bridge")], 1)])
    if not supported:
        f.statuses[f.a.paper_id] = "unsupported"
    def assess(question, claims, evidence, prior_gaps=()):
        resolved = second and any(p.paper_id == f.b.paper_id for p in evidence)
        return EvidenceAssessment(sufficient=resolved, rationale="Based on cited synthetic evidence.",
            gaps=[] if prior_gaps or resolved else [EvidenceGap(gap_id="G1", description="Nonuniform spectral recovery assumptions are missing.",
                search_focus="nonuniform spectral recovery", related_claim_ids=[], severity="critical")],
            gap_updates=[GapUpdate(gap_id=g.gap_id, status="resolved" if resolved else "unresolved",
                reason="Weighted-degree evidence supplied." if resolved else "Still missing.",
                evidence_ids=[p.passage_id for p in evidence if p.paper_id == f.b.paper_id] if resolved else [],
                related_claim_ids=[], superseded_by=None) for g in prior_gaps])
    f.assessor.assess.side_effect = assess
    return f


def project(store):
    return store.projects.create(ProjectCreate(title="Low-Rank Recovery", description="Spectral recovery on fixed support."))


def finish(store, project_id, fixture=None, **updates):
    f = fixture or research_fixture()
    request = ResearchRequest(question=QUESTION, project_id=project_id, max_search_rounds=1, **updates)
    run = store.create(request)
    request = store.get(run.run_id).request
    store.start(run.run_id)
    context = ProjectContextBuilder(store.projects).build(project_id, request) if project_id else None
    if context:
        f.agent.project_context = context
        store.set_project_context(run.run_id, context)
        session = ProjectSearchSession(f.agent.paper_search, store.projects, request, run.run_id)
        f.agent.paper_search = session
    result = f.agent.run(request)
    result.project_usage = project_usage(context, result.papers, result.evidence,
        queries_reused=session.reused if context else 0)
    store.complete(run.run_id, result, queries=list(session.records.values()) if context else ())
    return SimpleNamespace(run=store.get(run.run_id), fixture=f, context=context)
