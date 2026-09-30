"""Actual LangGraph routing with fake academic/LLM dependencies, never network."""

from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langgraph.graph.state import CompiledStateGraph

from phase5_helpers import OfflineTest, QUESTION, candidate, page, paper, plan
from researchpilot.evidence import (
    AcquiredDocument, AnswerClaim, AnswerDraft, ClaimVerification, EvidenceSelection,
)
from researchpilot.paper_search_service import PaperSearchService
from researchpilot.relevance import RelevanceAssessment
from researchpilot.research_agent import ResearchAgent
from researchpilot.hybrid_retrieval import HybridPassageRetriever
from researchpilot.research_graph import build_research_graph, create_memory_checkpointer, merge_candidates
from researchpilot.research_iteration import EvidenceAssessment, EvidenceGap, GapUpdate, FollowUpQuery, FollowUpSearchPlan
from researchpilot.research_models import ResearchRequest, ResearchResult
from researchpilot.semantic_reranker import SemanticReranker


A_TEXT = "Shared caches reduce storage latency for repeated reads in a fixed-capacity experiment."
B_TEXT = "Cache allocation reduces storage latency under contention by isolating competing requests."
FOLLOW_TEXT = '"cache contention" AND latency'


def missing_b(prior_gaps=()):
    return EvidenceAssessment(sufficient=False, gaps=[] if prior_gaps else [EvidenceGap(
        gap_id="contention", description="Repeated-read evidence does not establish latency under contention.",
        related_claim_ids=[], severity="critical", search_focus="cache contention allocation latency",
    )], rationale="The contention setting is not yet covered.", gap_updates=[GapUpdate(
        gap_id=g.gap_id, status="unresolved", reason="Still missing.", evidence_ids=[],
        related_claim_ids=[], superseded_by=None) for g in prior_gaps])


def follow_plan(*texts):
    return FollowUpSearchPlan(queries=[FollowUpQuery(query_id=f"follow{i}", text=text,
        gap_id="contention", rationale="Find the missing contention-specific latency evidence.")
        for i, text in enumerate(texts or (FOLLOW_TEXT,), 1)])


def loop_fixture():
    f = SimpleNamespace()
    f.a = paper("W1", title="Repeated-read shared caches", abstract=A_TEXT)
    f.b = paper("W2", title="Contention-aware cache allocation", abstract=B_TEXT)
    f.initial_records = [f.a]
    f.follow_records = [f.a, f.b]
    f.statuses = {}
    f.messages = []
    f.planner = Mock()
    f.planner.plan.return_value = plan()
    initial_texts = {q.text for q in plan().queries}
    f.raw = Mock()
    def search(query, **kwargs):
        if query in initial_texts:
            return f.initial_records
        return f.follow_records(query) if callable(f.follow_records) else f.follow_records
    f.raw.search_works.side_effect = search
    f.relevance = Mock()
    f.relevance.assess.return_value = RelevanceAssessment(category="direct", score=0.9, reason="Relevant setting.")
    f.fetcher = Mock()
    f.fetcher.acquire.side_effect = lambda p: AcquiredDocument(pages=[page(p, 3 if p.source_id == "W1" else None, p.abstract)])
    f.selector = Mock()
    f.selector.select.side_effect = lambda q, ps: EvidenceSelection(
        selected_passage_ids=[p.passage_id for p in ps], coverage_notes="Observed results only.", insufficient_evidence=False)
    f.synthesizer = Mock()
    def synthesize(question, evidence):
        by_paper = {p.paper_id: p for p in evidence}
        return AnswerDraft(claims=[AnswerClaim(claim_id=f"claim:{p.paper_id}", text=p.text,
            evidence_ids=[p.passage_id]) for p in by_paper.values()], limitations=[])
    f.synthesizer.synthesize.side_effect = synthesize
    f.verifier = Mock()
    f.verifier.verify.side_effect = lambda c, ps: ClaimVerification(claim_id=c.claim_id,
        status=f.statuses.get(ps[0].paper_id, "supported"), evidence_ids=c.evidence_ids,
        reason="Decision based only on the cited passage.")
    f.assessor = Mock()
    f.assessor.assess.side_effect = lambda question, claims, evidence, prior_gaps=(): (
        EvidenceAssessment(sufficient=True, gaps=[], rationale="Repeated reads and contention are covered.",
            gap_updates=[GapUpdate(gap_id=g.gap_id, status="resolved", reason="Contention evidence now present.",
                evidence_ids=[p.passage_id for p in evidence if p.paper_id == f.b.paper_id],
                related_claim_ids=[], superseded_by=None) for g in prior_gaps])
        if any(p.paper_id == f.b.paper_id for p in evidence) else missing_b(prior_gaps))
    f.follow_up = Mock()
    f.follow_up.plan.return_value = follow_plan()
    f.agent = ResearchAgent(planner=f.planner, paper_search=PaperSearchService(f.raw),
        semantic_reranker=SemanticReranker(f.relevance), document_fetcher=f.fetcher,
        evidence_selector=f.selector, synthesizer=f.synthesizer, verifier=f.verifier,
        evidence_assessor=f.assessor, follow_up_planner=f.follow_up, progress=f.messages.append,
        passage_retriever=HybridPassageRetriever())
    return f


class ResearchGraphTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.f = loop_fixture()
        self.request = ResearchRequest(question=QUESTION, max_queries=3)

    def run_agent(self, **updates):
        return self.f.agent.run(ResearchRequest(**(self.request.model_dump() | updates)))

    def test_real_graph_two_round_end_to_end_supports_a_and_b(self):
        graph = build_research_graph(self.f.agent)
        self.assertIsInstance(graph, CompiledStateGraph)
        events = list(graph.stream({"request": self.request}, config={"recursion_limit": 32}, stream_mode="updates"))
        names = [next(iter(e)) for e in events]
        self.assertEqual(names, ["initial_plan", "search_and_rank", "acquire_documents", "retrieve_evidence",
            "synthesize_claims", "verify_claims", "assess_evidence", "plan_follow_up", "search_and_rank",
            "acquire_documents", "retrieve_evidence", "synthesize_claims", "verify_claims", "assess_evidence", "finalize"])
        result = events[-1]["finalize"]["final_result"]
        self.assertIn(A_TEXT, result.answer)
        self.assertIn(B_TEXT, result.answer)
        self.assertIn("[P1, p. 3]", result.answer)
        self.assertIn("[P2, abstract]", result.answer)
        self.assertEqual(result.termination_reason, "sufficient_evidence")
        self.assertEqual(result.run_stats.search_rounds, 2)
        self.assertEqual(result.run_stats.synthesis_rounds, 2)
        self.assertEqual(result.run_stats.verification_calls, 2)
        self.assertEqual(result.run_stats.supported, 2)
        self.assertEqual(result.run_stats.raw_candidates, 2)
        self.assertEqual(result.run_stats.selected_papers, 2)
        self.assertEqual(len(result.evidence), 2)
        self.assertEqual(self.f.planner.plan.call_count, 1)
        self.assertEqual(self.f.raw.search_works.call_count, 4)
        self.assertEqual(self.f.fetcher.acquire.call_count, 2)
        self.assertEqual(self.f.relevance.assess.call_count, 2)
        self.assertEqual(ResearchResult.model_validate_json(result.model_dump_json()), result)

    def test_round_trace_counts_and_query_history(self):
        result = self.run_agent()
        self.assertEqual([t.round for t in result.round_trace], [1, 2])
        self.assertEqual([t.queries for t in result.round_trace], [3, 1])
        self.assertEqual([t.candidate_papers for t in result.round_trace], [1, 2])
        self.assertEqual([t.new_papers for t in result.round_trace], [1, 1])
        self.assertEqual([t.new_passages for t in result.round_trace], [1, 1])
        self.assertEqual([t.new_evidence for t in result.round_trace], [1, 1])
        self.assertEqual([t.evidence_passages for t in result.round_trace], [1, 2])
        self.assertEqual([t.supported_claims for t in result.round_trace], [1, 2])
        self.assertEqual([t.remaining_gaps for t in result.round_trace], [1, 0])
        self.assertEqual([t.already_seen_papers for t in result.round_trace], [0, 1])
        self.assertEqual([q.text for q in result.executed_queries], [q.text for q in plan().queries] + [FOLLOW_TEXT])
        self.assertEqual(result.follow_up_plans[0].queries[0].gap_id, "contention")

    def test_first_round_sufficient_skips_follow_up(self):
        self.f.assessor.assess.side_effect = lambda *args, **kwargs: EvidenceAssessment(sufficient=True, gaps=[], rationale="Question already covered.")
        result = self.run_agent()
        self.assertEqual(len(result.round_trace), 1)
        self.assertEqual(result.termination_reason, "sufficient_evidence")
        self.f.follow_up.plan.assert_not_called()

    def test_follow_up_receives_actual_gaps_question_history_and_budget(self):
        result = self.run_agent(max_follow_up_queries=1)
        call = self.f.follow_up.plan.call_args.kwargs
        self.assertEqual(call["question"], QUESTION)
        self.assertEqual(call["gaps"], missing_b().gaps)
        self.assertEqual([q.text for q in call["executed_queries"]], [q.text for q in plan().queries])
        self.assertEqual(call["max_queries"], 1)
        self.assertEqual(call["concepts"], plan().concepts)
        self.assertEqual(result.follow_up_plans[0].queries[0].gap_id, call["gaps"][0].gap_id)

    def test_follow_up_preserves_original_year_bounds(self):
        self.run_agent(year_from=2020, year_to=2026)
        for call in self.f.raw.search_works.call_args_list:
            self.assertEqual((call.kwargs["year_from"], call.kwargs["year_to"]), (2020, 2026))

    def test_duplicate_query_filtered_before_network(self):
        self.f.follow_up.plan.return_value = follow_plan('  "SHARED   CACHES" and latency ', FOLLOW_TEXT)
        result = self.run_agent()
        self.assertEqual(result.round_trace[1].queries, 1)
        self.assertEqual(result.executed_queries[-1].query_id, "follow2")
        self.assertTrue(any("Duplicate" in w for w in result.warnings))

    def test_all_duplicates_stop_without_second_search(self):
        self.f.follow_up.plan.return_value = follow_plan('"SHARED  CACHES" and latency')
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "duplicate_queries")
        self.assertEqual(len(result.round_trace), 1)
        self.assertEqual(self.f.raw.search_works.call_count, 3)

    def test_round_one_evidence_retained_and_reverified_after_new_evidence(self):
        result = self.run_agent()
        first = self.f.synthesizer.synthesize.call_args_list[0].args[1]
        second = self.f.synthesizer.synthesize.call_args_list[1].args[1]
        self.assertEqual(second[:len(first)], first)
        self.assertEqual(len(second), 2)
        self.assertEqual([p.passage_id for p in result.evidence], [p.passage_id for p in second])
        calls = self.f.verifier.verify.call_args_list
        self.assertNotEqual(calls[0].args[0].claim_id, calls[1].args[0].claim_id)  # Stable finding reused.
        for call in calls:
            claim, cited = call.args
            self.assertEqual(claim.evidence_ids, [p.passage_id for p in cited])

    def test_follow_up_fills_an_unsupported_topic_and_removes_old_draft(self):
        original = self.f.synthesizer.synthesize.side_effect
        def synthesize(question, evidence):
            draft = original(question, evidence)
            if len(evidence) == 1:
                draft.claims.append(AnswerClaim(claim_id="unsupported-B", text="UNSUPPORTED initial conjecture on contention.", evidence_ids=[evidence[0].passage_id]))
            else:
                draft.claims[-1].claim_id="unsupported-B"  # Target identity retained during revision.
            return draft
        self.f.synthesizer.synthesize.side_effect = synthesize
        original_verify = self.f.verifier.verify.side_effect
        self.f.verifier.verify.side_effect = lambda c, ps: (ClaimVerification(claim_id=c.claim_id, status="unsupported", evidence_ids=c.evidence_ids, reason="Topic B absent.")
            if c.text.startswith("UNSUPPORTED") else original_verify(c, ps))
        result = self.run_agent()
        self.assertEqual(result.round_trace[0].unsupported_claims, 1)
        self.assertEqual(result.round_trace[1].unsupported_claims, 0)
        self.assertNotIn("UNSUPPORTED", result.answer)
        self.assertIn(B_TEXT, result.answer)

    def test_max_rounds_one_is_one_shot_with_explicit_assessment(self):
        result = self.run_agent(max_search_rounds=1)
        self.assertEqual(result.termination_reason, "max_search_rounds")
        self.assertEqual(len(result.round_trace), 1)
        self.f.follow_up.plan.assert_not_called()
        self.f.assessor.assess.assert_called_once()
        self.assertIn(A_TEXT, result.answer)

    def test_max_three_rounds_bounds_an_always_insufficient_model(self):
        self.f.assessor.assess.side_effect = lambda *args, prior_gaps=(): missing_b(prior_gaps)
        self.f.follow_up.plan.side_effect = [follow_plan(), FollowUpSearchPlan(queries=[FollowUpQuery(
            query_id="third", text='"cache allocation" AND latency', gap_id="contention", rationale="A new gap-directed expression.")])]
        self.f.follow_records = lambda text: [self.f.b] if text == FOLLOW_TEXT else [paper("W3", title="New cache result", abstract=B_TEXT)]
        result = self.run_agent(max_search_rounds=3)
        self.assertEqual(result.run_stats.search_rounds, 3)
        self.assertEqual(result.termination_reason, "max_search_rounds")
        self.assertEqual(self.f.follow_up.plan.call_count, 2)
        self.assertEqual(self.f.synthesizer.synthesize.call_count, 3)
        self.assertEqual(len(result.papers), 3)
        self.assertTrue(any("remaining evidence gaps: 1" in w for w in result.warnings))

    def test_no_new_papers_stops_and_preserves_verified_claims(self):
        self.f.follow_records = [self.f.a]
        result = self.run_agent(max_search_rounds=3)
        self.assertEqual(result.termination_reason, "no_new_papers")
        self.assertEqual(self.f.fetcher.acquire.call_count, 1)
        self.assertEqual(self.f.synthesizer.synthesize.call_count, 1)
        self.assertIn(A_TEXT, result.answer)
        self.assertEqual(result.round_trace[-1].new_evidence, 0)

    def test_new_version_does_not_reprocess_existing_work_when_group_id_changes(self):
        self.f.follow_records = [paper("W9", title=self.f.a.title, publication_year=2025, abstract=A_TEXT)]
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "no_new_papers")
        self.assertEqual(result.round_trace[-1].new_candidate_papers, 1)
        self.assertEqual(result.round_trace[-1].already_seen_papers, 1)
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(result.papers[0].paper.paper_id, self.f.a.paper_id)
        self.assertEqual(self.f.fetcher.acquire.call_count, 1)

    def test_seen_but_unselected_record_is_not_misreported_as_new_work(self):
        self.f.initial_records = [self.f.a, self.f.b]
        self.f.follow_records = [self.f.b]
        result = self.run_agent(max_papers=1)
        self.assertEqual(result.termination_reason, "no_new_papers")
        self.assertEqual(self.f.fetcher.acquire.call_count, 1)

    def test_no_new_passages_stops_before_resynthesis(self):
        original = self.f.fetcher.acquire.side_effect
        self.f.fetcher.acquire.side_effect = lambda p: AcquiredDocument(pages=[]) if p.source_id == "W2" else original(p)
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "no_new_evidence")
        self.assertEqual(result.round_trace[-1].new_papers, 1)
        self.assertEqual(result.round_trace[-1].new_passages, 0)
        self.assertEqual(self.f.synthesizer.synthesize.call_count, 1)

    def test_new_passages_but_no_new_selected_evidence_stops(self):
        self.f.selector.select.side_effect = lambda q, ps: EvidenceSelection(
            selected_passage_ids=[p.passage_id for p in ps if p.paper_id == self.f.a.paper_id],
            coverage_notes="Other passages did not support the question.", insufficient_evidence=True)
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "no_new_evidence")
        self.assertEqual(result.round_trace[-1].new_passages, 1)
        self.assertEqual(result.round_trace[-1].new_evidence, 0)
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(self.f.synthesizer.synthesize.call_count, 1)

    def test_no_meaningful_gaps_stops(self):
        self.f.assessor.assess.side_effect = lambda *args, **kwargs: EvidenceAssessment(sufficient=False, gaps=[], rationale="No actionable academic search remains.")
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "no_meaningful_gaps")
        self.f.follow_up.plan.assert_not_called()

    def test_empty_follow_up_plan_stops(self):
        self.f.follow_up.plan.return_value = FollowUpSearchPlan(queries=[])
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "no_follow_up_queries")
        self.assertEqual(len(result.round_trace), 1)

    def test_failed_follow_up_planner_keeps_current_verified_answer(self):
        self.f.follow_up.plan.side_effect = RuntimeError("unavailable")
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "follow_up_planner_failed")
        self.assertIn(A_TEXT, result.answer)
        self.assertEqual(self.f.raw.search_works.call_count, 3)

    def test_bad_gap_reference_never_executes_follow_up(self):
        self.f.follow_up.plan.return_value = FollowUpSearchPlan(queries=[FollowUpQuery(
            query_id="new", text="invented focus", gap_id="unknown", rationale="x")])
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "follow_up_planner_failed")
        self.assertEqual(self.f.raw.search_works.call_count, 3)

    def test_assessor_failure_does_not_invent_a_search_reason(self):
        self.f.assessor.assess.side_effect = RuntimeError("unavailable")
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "evidence_assessor_failed")
        self.assertIsNone(result.evidence_assessment)
        self.f.follow_up.plan.assert_not_called()

    def test_assessor_unknown_claim_id_rejected(self):
        invalid = missing_b()
        invalid.gaps[0].related_claim_ids = ["invented"]
        self.f.assessor.assess.side_effect = None
        self.f.assessor.assess.return_value = invalid
        with self.assertRaisesRegex(ValueError, "Unknown related"):
            self.run_agent()

    def test_assessor_cannot_mark_unsupported_final_claims_sufficient(self):
        self.f.statuses[self.f.b.paper_id] = "unsupported"
        with self.assertRaisesRegex(ValueError, "cannot override"):
            self.run_agent()

    def test_follow_up_search_failure_keeps_verified_answer(self):
        original = self.f.raw.search_works.side_effect
        def search(query, **kwargs):
            if query == FOLLOW_TEXT:
                raise RuntimeError("search unavailable")
            return original(query, **kwargs)
        self.f.raw.search_works.side_effect = search
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "follow_up_search_failed")
        self.assertIn(A_TEXT, result.answer)

    def test_follow_up_pdf_failure_uses_abstract_and_preserves_pdf_citation(self):
        original = self.f.fetcher.acquire.side_effect
        def fetch(p):
            if p.source_id == "W2":
                raise RuntimeError("malformed PDF")
            return original(p)
        self.f.fetcher.acquire.side_effect = fetch
        result = self.run_agent()
        self.assertIn("[P1, p. 3]", result.answer)
        self.assertIn("[P2, abstract]", result.answer)
        self.assertTrue(any("document fetcher failed" in w for w in result.warnings))

    def test_final_unsupported_removed_partial_qualified_and_conflict_explicit(self):
        for status in ("unsupported", "partially_supported", "conflicting"):
            with self.subTest(status=status):
                f = loop_fixture()
                f.statuses[f.b.paper_id] = status
                f.assessor.assess.side_effect = lambda *args, prior_gaps=(): missing_b(prior_gaps)
                result = f.agent.run(self.request)
                self.assertEqual(result.termination_reason, "max_search_rounds")
                if status == "unsupported":
                    self.assertNotIn(B_TEXT, result.answer)
                elif status == "partially_supported":
                    self.assertIn("Partially supported; not established in full", result.answer)
                else:
                    self.assertIn("Unresolved disagreement", result.answer)
                available = {p.passage_id: p for p in result.evidence}
                for record in result.claims:
                    self.assertTrue(record.claim.evidence_ids)
                    for key in record.claim.evidence_ids:
                        self.assertIn(key, available)
                        p = available[key]
                        self.assertEqual(p.source_type == "pdf", p.page_number is not None)

    def test_unknown_citation_in_second_synthesis_rejected(self):
        original = self.f.synthesizer.synthesize.side_effect
        def synthesize(q, evidence):
            draft = original(q, evidence)
            if len(evidence) > 1:
                draft.claims[-1].evidence_ids = ["invented"]
            return draft
        self.f.synthesizer.synthesize.side_effect = synthesize
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.run_agent()

    def test_unknown_claim_from_second_round_verifier_rejected(self):
        original = self.f.verifier.verify.side_effect
        def verify(c, ps):
            v = original(c, ps)
            if ps[0].paper_id == self.f.b.paper_id:
                return v.model_copy(update={"claim_id": "invented"})
            return v
        self.f.verifier.verify.side_effect = verify
        with self.assertRaisesRegex(ValueError, "Unknown claim"):
            self.run_agent()

    def test_no_evidence_never_synthesizes_factual_answer(self):
        self.f.initial_records = []
        self.f.follow_records = []
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "no_literature_found")
        self.assertEqual(result.run_stats.search_rounds, 1)
        self.f.assessor.assess.assert_not_called()
        self.f.follow_up.plan.assert_not_called()
        self.assertIn("Insufficient evidence", result.answer)
        self.assertEqual(result.evidence, [])
        self.f.synthesizer.synthesize.assert_not_called()
        self.f.verifier.verify.assert_not_called()

    def test_agent_reuse_does_not_leak_state_between_runs(self):
        first = self.run_agent()
        second = self.run_agent()
        def stable_trace(result):
            traces = [t.model_dump() for t in result.round_trace]
            for trace in traces:
                trace["passage_retrieval"].pop("hybrid_elapsed_seconds")
            return traces
        self.assertEqual(stable_trace(first), stable_trace(second))
        self.assertEqual(first.evidence, second.evidence)
        self.assertEqual([p.citation_label for p in second.papers], ["P1", "P2"])

    def test_optional_memory_checkpointer_and_old_states_are_not_mutated(self):
        with create_memory_checkpointer() as checkpointer:
            graph = build_research_graph(self.f.agent, checkpointer=checkpointer)
            config = {"recursion_limit": 32, "configurable": {"thread_id": "offline-two-round"}}
            result = graph.invoke({"request": self.request}, config)["final_result"]
            history = list(graph.get_state_history(config))
            round_one = [s.values for s in history if s.values.get("search_round") == 1 and len(s.values.get("claims", [])) == 1]
            self.assertTrue(round_one)
            self.assertTrue(all(len(s["selected_evidence"]) == 1 for s in round_one))
            self.assertEqual(result.run_stats.supported, 2)

    def test_graph_has_no_eval_dependency_or_file_reads(self):
        import researchpilot.research_graph as module
        self.assertNotIn("from eval", inspect.getsource(module))
        self.assertNotIn("import eval", inspect.getsource(module))
        with patch("builtins.open", side_effect=AssertionError("No benchmark files may be opened")), \
             patch.object(Path, "read_text", side_effect=AssertionError("No benchmark files may be read")):
            self.run_agent()

    def test_candidate_merge_keeps_first_metadata_and_best_hits_without_mutation(self):
        first = candidate(self.f.a, rank=4)
        incoming = [candidate(self.f.a.model_copy(update={"title": "changed"}), rank=1),
                    candidate(self.f.a, rank=3, query_id="follow1")]
        prior = {self.f.a.paper_id: first}
        original = deepcopy((prior, incoming))
        merged = merge_candidates(prior, incoming)
        result = merged[self.f.a.paper_id]
        self.assertEqual(result.paper.title, self.f.a.title)
        self.assertEqual([(h.query_id, h.rank) for h in result.hits], [("q1", 1), ("follow1", 3)])
        self.assertEqual((prior, incoming), original)
