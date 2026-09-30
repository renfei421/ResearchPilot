"""Small fictional sources; no real paper metadata or network calls."""

from types import SimpleNamespace

from researchpilot.claim_check import *
from researchpilot.claim_recovery import RecoveryPlan
from researchpilot.claim_novelty_agent import ClaimNoveltyAgent
from researchpilot.document_rescue import DocumentRescuer
from researchpilot.evidence import AcquiredDocument, DocumentPage
from researchpilot.hybrid_retrieval import HybridPassageRetriever
from researchpilot.paper import Paper
from researchpilot.paper_search_service import PaperSearchService
from researchpilot.relevance import RelevanceAssessment, SemanticallyRankedGroup
from researchpilot.research_agent import ResearchAgent

CLAIM = "Optimizing fixed graph weights improves downstream curvature."
TEXTS = {
    "A": "A weight optimization algorithm reduces a graph spectral ratio on fixed support. It does not study downstream curvature.",
    "B": "A spectral condition implies a downstream curvature bound under regularity assumptions. No optimization of weights is studied.",
    "C": "A downstream recovery procedure uses uniform graph weights and a spectral condition. Optimized nonuniform weights are not considered.",
}


def assertion(key="C1", text="Optimize graph weights.", **kwargs):
    return ClaimAssertion(claim_id=key, text=text, claim_type="theoretical", concepts=["graph", "weights"],
                          depends_on=[], importance="core", **kwargs)


def decomposition(main=CLAIM):
    return ClaimDecomposition(main_claim=main, assertions=[
        assertion("C1", "Fixed-support weight optimization reduces a spectral ratio."),
        assertion("C2", "Spectral control implies downstream curvature under regularity assumptions."),
        assertion("C3", "Optimizing fixed-support weights improves downstream curvature.").model_copy(update={"depends_on": ["C1", "C2"]})])


def relation(cid="C1", pid="A", label="SUPPORTING", ids=None, **updates):
    ids = ids if ids is not None else ["eA"]
    data = dict(claim_id=cid, paper_id=pid, relation=label,
        relation_summary="The cited passage establishes a useful component under its stated assumptions.",
        matched_dimensions=["fixed graph setting"], differences=["The full combined result is not established here."],
        evidence_ids=ids, scope_notes=["Limited to the actual supplied passage."], confidence="high",
        comparisons=[DimensionComparison(dimension=d, claim_scope="the assertion's stated scope",
            paper_scope="the source's stated scope", alignment="same", evidence_ids=ids) for d in DIMENSIONS],
        evidence_quality="substantive", explicit_incompatibility=label == "CONTRADICTING")
    if label == "ADJACENT":
        data.update(evidence_ids=[], evidence_quality="none", matched_dimensions=[], differences=[],
            comparisons=[DimensionComparison(dimension=d, claim_scope="claim", paper_scope="unknown",
                alignment="unknown", evidence_ids=[]) for d in DIMENSIONS])
    data.update(updates)
    return ClaimPaperRelation(**data)


def fixture(*, follow_up=False, blank=False):
    client = SimpleNamespace(calls=[])
    papers = [Paper(paper_id=k, title="Fictional study "+k, abstract=text, authors=["Example Author"],
        publication_year=2023, doi=None, citation_count=0, source="fixture", source_id=k, open_access_url=None)
        for k, text in TEXTS.items()]
    def search_works(**kw):
        client.calls.append(kw)
        return [] if blank else papers
    client.search_works = search_works
    fetcher = SimpleNamespace(calls=[])
    def acquire(paper):
        fetcher.calls.append(paper.paper_id)
        return AcquiredDocument(pages=[DocumentPage(paper_id=paper.paper_id, title=paper.title,
            page_number=1, source_type="pdf", text=TEXTS[paper.paper_id])])
    fetcher.acquire = acquire
    reranker = SimpleNamespace(calls=[])
    def rerank(question, groups):
        reranker.calls.append(question)
        return [SemanticallyRankedGroup(ranked_group=g, assessment=RelevanceAssessment(
            category="direct", score=.8, reason="Fictional relevant evidence.")) for g in groups]
    reranker.rerank = rerank
    planner = SimpleNamespace(calls=[])
    def plan(request, decomposed, targets, previous, gaps):
        planner.calls.append((targets, previous, gaps))
        return ClaimSearchPlans(plans=[ClaimSearchPlan(claim_id=cid, queries=[
            ClaimQuery(text="graph weights" if not previous or not follow_up else "weighted support", role="direct"),
            ClaimQuery(text="spectral geometry" if not previous or not follow_up else "spectral mechanism", role="terminology"),
            ClaimQuery(text="weights curvature" if not previous or not follow_up else "curvature connection", role="bridge")]) for cid in targets])
    planner.plan = plan
    analyzer = SimpleNamespace(calls=[])
    def analyze(assertions, pid, evidence, bundles):
        analyzer.calls.append((assertions, pid, evidence, bundles))
        labels = {("C1", "A"): "DIRECT_OVERLAP", ("C2", "B"): "DIRECT_OVERLAP",
                  ("C3", "A"): "BRIDGING", ("C3", "B"): "SUPPORTING", ("C3", "C"): "PARTIAL_OVERLAP"}
        return RelationBatch(relations=[relation(a.claim_id, pid, labels.get((a.claim_id,pid), "ADJACENT"),
            [evidence[0].passage_id]) for a in assertions])
    analyzer.analyze = analyze
    services = ResearchAgent(paper_search=PaperSearchService(client), semantic_reranker=reranker,
        document_fetcher=fetcher, passage_retriever=HybridPassageRetriever(), document_rescuer=DocumentRescuer())
    decomposer = SimpleNamespace(decompose=lambda request: decomposition(request.claim))
    recovery_planner = SimpleNamespace(plan=lambda request, assertion, *args:
        RecoveryPlan(claim_id=assertion.claim_id, queries=[]))
    agent = ClaimNoveltyAgent(services=services, decomposer=decomposer, search_planner=planner,
        relation_analyzer=analyzer, recovery_planner=recovery_planner)
    return SimpleNamespace(agent=agent, client=client, fetcher=fetcher, planner=planner, analyzer=analyzer,
                           services=services, decomposer=decomposer, reranker=reranker)
