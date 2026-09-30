"""Fake downloaded documents -> hybrid retrieval -> atomic verification -> citation."""

from copy import deepcopy
from unittest.mock import Mock

from phase5_helpers import OfflineTest, page, paper, passage
from test_hybrid_retrieval import QUERY, HIDDEN, EXACT, fake_hybrid
from test_research_graph import loop_fixture
from researchpilot.document_rescue import DocumentRescuer, RescueLimits
from researchpilot.evidence import (AcquiredDocument, AnswerClaim, AnswerDraft, AtomicAssertion,
                                   ClaimVerification, EvidenceSelection, RetrievalView)
from researchpilot.passage_retrieval import LexicalRetriever, PageChunker
from researchpilot.research_graph import create_memory_checkpointer
from researchpilot.research_iteration import EvidenceAssessment, EvidenceGap, GapUpdate
from researchpilot.research_models import ResearchRequest as _ResearchRequest, ResearchResult, SelectedPaper
from researchpilot.evidence_bundle import AssemblyLimits


def ResearchRequest(**kwargs):
    """Isolate existing rescue/hybrid contracts from the new assembly stage."""
    kwargs.setdefault("assembly_limits", AssemblyLimits(enabled=False))
    return _ResearchRequest(**kwargs)



class HybridGraphTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.root = self.temporary_directory()
        self.hybrid, self.client = fake_hybrid(self.root, {QUERY: [1, 0, 0], HIDDEN: [1, 0, 0], EXACT: [0, 0, 1]})
        self.paper = paper(title="Local guarantees for factor objectives", abstract=QUERY)
        self.pages = [page(self.paper, 1, EXACT), page(self.paper, 7, HIDDEN)]
        self.corpus = PageChunker().chunk(self.pages)
        self.papers = [SelectedPaper(paper=self.paper, citation_label="P1", group_id="g", source_type="pdf")]
        self.gap = EvidenceGap(gap_id="curvature", description="Need the local positivity guarantee.",
            search_focus=QUERY, related_claim_ids=[], severity="critical")

    def fixture(self):
        f = loop_fixture()
        f.raw.search_works.side_effect = lambda **kw: [self.paper]
        f.planner.plan.return_value = f.planner.plan.return_value.model_copy(update={
            "research_question": QUERY, "concepts": ["Assumption A1"]})
        f.fetcher.acquire.side_effect = lambda p: AcquiredDocument(pages=self.pages, cache_hit=True)
        f.agent.passage_retriever = self.hybrid
        f.agent.document_rescuer = DocumentRescuer(passage_retriever=self.hybrid)
        f.synthesizer.synthesize.side_effect = lambda q, ps: AnswerDraft(claims=[AnswerClaim(
            claim_id="scope", text=HIDDEN, evidence_ids=[next(p.passage_id for p in ps if p.text == HIDDEN)])], limitations=[])
        f.verifier.verify.side_effect = lambda c, ps: ClaimVerification(claim_id=c.claim_id,
            status="supported", evidence_ids=c.evidence_ids, reason="Only the stated tangent-space scope.",
            assertions=[AtomicAssertion(assertion_id="scope", text=c.text, kind="theoretical", status="supported",
                evidence_ids=c.evidence_ids, reason="The passage states positivity on admissible perturbations.",
                scope_supported=True, quantities_supported=True, inference_supported=True)])
        f.assessor.assess.side_effect = lambda q, cs, ps, prior_gaps=(): EvidenceAssessment(
            sufficient=True, gaps=[], rationale="Tangent-space claim is supported.",
            gap_updates=[GapUpdate(gap_id=g.gap_id, status="resolved", reason="Scoped positivity established.",
                evidence_ids=cs[0].claim.evidence_ids, related_claim_ids=["scope"], superseded_by=None) for g in prior_gaps])
        return f

    def test_normal_hybrid_selector_atomic_verification_and_real_page_citation(self):
        f = self.fixture()
        self.assertEqual(LexicalRetriever().retrieve(QUERY, self.corpus, 2), [])
        original = deepcopy(self.corpus)
        result = f.agent.run(ResearchRequest(question=QUERY, max_queries=3, top_passages=2))
        self.assertEqual(result.termination_reason, "sufficient_evidence")
        self.assertEqual(result.run_stats.search_rounds, 1)
        self.assertEqual(result.run_stats.supported, 1)
        self.assertEqual(result.run_stats.local_rescue_passes, 0)
        self.assertEqual(result.run_stats.visual_evidence_calls, 0)
        self.assertIn("[P1, p. 7]", result.answer)
        self.assertIn(HIDDEN, result.answer)
        self.assertEqual(f.raw.search_works.call_count, 3)
        f.follow_up.plan.assert_not_called()
        f.fetcher.acquire.assert_called_once()
        offered = f.selector.select.call_args.args[1]
        self.assertIn(HIDDEN, [p.text for p in offered])
        self.assertTrue(all(type(p) is type(self.corpus[0]) for p in offered))
        self.assertEqual(self.corpus, original)
        self.assertEqual(ResearchResult.model_validate_json(result.model_dump_json()), result)

    def test_gap_resolved_by_dense_local_rescue_without_external_search(self):
        f = self.fixture()
        complete_assessment = f.assessor.assess.side_effect
        f.assessor.assess.side_effect = lambda q, cs, ps, prior_gaps=(): (
            complete_assessment(q, cs, ps, prior_gaps) if cs else
            EvidenceAssessment(sufficient=False, gaps=[self.gap], rationale="Positivity evidence needed."))
        # A bounded first selector sees the fused set but asks for focused evidence;
        # routing then uses hybrid local rescue over the SAME acquired PDF corpus.
        f.selector.select.side_effect = lambda q, ps: EvidenceSelection(
            selected_passage_ids=[] if f.selector.select.call_count == 1 else
                [p.passage_id for p in ps if p.text == HIDDEN],
            insufficient_evidence=f.selector.select.call_count == 1, coverage_notes="Need scoped positivity.")
        result = f.agent.run(ResearchRequest(question=QUERY, max_queries=3, top_passages=2))
        self.assertEqual(result.run_stats.local_rescue_passes, 1)
        self.assertEqual(result.run_stats.search_rounds, 1)
        self.assertTrue(result.gap_ledger)
        self.assertTrue(all(g.status == "resolved" for g in result.gap_ledger))
        self.assertTrue(result.rescue_trace[0].added_evidence_ids)
        self.assertEqual(result.rescue_trace[0].passage_retrieval[0].mode, "hybrid")
        self.assertIn("[P1, p. 7]", result.answer)
        self.assertEqual(result.claims[0].verification.assertions[0].status, "supported")
        f.follow_up.plan.assert_not_called()
        f.fetcher.acquire.assert_called_once()

    def test_local_rescue_planning_is_pure_and_actual_rescue_uses_hybrid(self):
        rescuer = DocumentRescuer(passage_retriever=self.hybrid)
        limits = RescueLimits()
        selected = {self.corpus[0].passage_id: self.corpus[0]}
        plans = rescuer.plan(QUERY, [self.gap], self.papers, self.corpus, selected, [], [], limits)
        self.assertTrue(plans)  # No lexical hit, but scoped unseen document evidence.
        self.assertEqual(self.client.calls, [])
        before = deepcopy(plans)
        outcome = rescuer.rescue(QUERY, plans, self.papers, limits, [], 1, "synthetic")
        self.assertIn(HIDDEN, [p.text for p in outcome.passages])
        self.assertNotIn(EXACT, [p.text for p in outcome.passages])
        self.assertEqual(outcome.trace.passage_retrieval[0].mode, "hybrid")
        self.assertEqual(plans, before)
        self.assertEqual(outcome.trace.visual_pages, [])

    def test_dense_rescue_preserves_structural_priority_and_limits(self):
        corpus = [*self.corpus, passage("structure", self.paper, 8, "Theorem 2. A restricted curvature guarantee holds.")]
        rescuer = DocumentRescuer(passage_retriever=self.hybrid)
        limits = RescueLimits(max_rescue_passages_per_gap=2, max_rescue_passages_per_run=2)
        plans = rescuer.plan(QUERY, [self.gap], self.papers, corpus, {}, [], [], limits)
        outcome = rescuer.rescue(QUERY, plans, self.papers, limits, [], 1, "synthetic")
        self.assertEqual(outcome.passages[0].passage_id, "structure")
        self.assertLessEqual(len(outcome.passages), 2)

    def test_rescue_dense_failure_warns_and_keeps_lexical_evidence(self):
        self.client.embed_texts = Mock(side_effect=RuntimeError("offline"))
        corpus = [passage("lexical", self.paper, 7, "A restricted curvature guarantee under bounded perturbations.")]
        rescuer = DocumentRescuer(passage_retriever=self.hybrid)
        plans = rescuer.plan(QUERY, [self.gap], self.papers, corpus, {}, [], [], RescueLimits())
        outcome = rescuer.rescue(QUERY, plans, self.papers, RescueLimits(), [], 1, "synthetic")
        self.assertTrue(outcome.trace.passage_retrieval[0].dense_fallback_used)
        self.assertEqual(outcome.passages, corpus)
        self.assertTrue(outcome.warnings)

    def test_hybrid_trace_survives_checkpoint(self):
        f = self.fixture()
        f.agent._checkpointer = create_memory_checkpointer()
        result = f.agent.run(ResearchRequest(question=QUERY, max_queries=3, top_passages=2))
        self.assertEqual(result.round_trace[0].passage_retrieval.mode, "hybrid")

    def test_dense_similarity_never_enters_selector_or_verifier_payload(self):
        f = self.fixture()
        f.agent.run(ResearchRequest(question=QUERY, max_queries=3, top_passages=2))
        for record in f.selector.select.call_args.args[1] + f.verifier.verify.call_args.args[1]:
            self.assertEqual(set(record.model_dump()), set(self.corpus[0].model_dump()))
            self.assertNotIn("similarity", record.model_dump_json())
