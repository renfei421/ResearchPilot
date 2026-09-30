from copy import deepcopy
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace
from unittest.mock import Mock, patch

from phase5_helpers import OfflineTest, QUESTION, TEXT, candidate, page, paper, passage, plan
from researchpilot.answer_renderer import render_answer, source_url
from researchpilot.evidence import AcquiredDocument, AnswerClaim, AnswerDraft, ClaimVerification, EvidenceSelection, VerifiedClaim
from researchpilot.paper_search_service import PaperSearchService
from researchpilot.relevance import RelevanceAssessment
from researchpilot.research_agent import ResearchAgent
from researchpilot.hybrid_retrieval import HybridPassageRetriever
from researchpilot.research_models import ResearchRequest, ResearchResult, SelectedPaper
from researchpilot.research_iteration import EvidenceAssessment, FollowUpSearchPlan
from researchpilot.semantic_reranker import SemanticReranker
from scripts.run_research_agent import main


def fake_dependencies(records=None):
    records = records if records is not None else [paper(), paper("W2", title="Cache contention")]
    planner = Mock()
    planner.plan.return_value = plan()
    raw_client = Mock()
    raw_client.search_works.return_value = records
    relevance = Mock()
    relevance.assess.return_value = RelevanceAssessment(category="direct", score=0.8, reason="Useful evidence.")
    fetcher = Mock()
    fetcher.acquire.side_effect = lambda p: AcquiredDocument(pages=[page(p, None if p.source_id == "W2" else 3)])
    selector = Mock()
    selector.select.side_effect = lambda q, ps: EvidenceSelection(
        selected_passage_ids=[p.passage_id for p in ps], coverage_notes="Evidence covers repeated reads.", insufficient_evidence=False)
    synth = Mock()
    def synthesize(q, ps):
        return AnswerDraft(claims=[
            AnswerClaim(claim_id="c1", text="Shared caches reduce latency for repeated reads.", evidence_ids=[ps[0].passage_id]),
            AnswerClaim(claim_id="c2", text="UNSUPPORTED: all workloads always improve.", evidence_ids=[ps[-1].passage_id]),
        ], limitations=[])
    synth.synthesize.side_effect = synthesize
    verifier = Mock()
    verifier.verify.side_effect = lambda c, ps: ClaimVerification(claim_id=c.claim_id,
        status="supported" if c.claim_id == "c1" else "unsupported", evidence_ids=c.evidence_ids,
        reason="Only repeated reads are covered by the cited evidence.")
    # These tests retain their Phase 5 assertions; the new graph dependencies
    # are explicitly offline and propose no additional searches.
    assessor = Mock()
    assessor.assess.return_value = EvidenceAssessment(sufficient=False, gaps=[], rationale="Fixture has no actionable follow-up gaps.")
    follow_up = Mock()
    follow_up.plan.return_value = FollowUpSearchPlan(queries=[])
    agent = ResearchAgent(planner=planner, paper_search=PaperSearchService(raw_client),
                          semantic_reranker=SemanticReranker(relevance), document_fetcher=fetcher,
                          evidence_selector=selector, synthesizer=synth, verifier=verifier,
                          evidence_assessor=assessor, follow_up_planner=follow_up,
                          passage_retriever=HybridPassageRetriever())
    return SimpleNamespace(agent=agent, planner=planner, raw_client=raw_client, relevance=relevance,
                           fetcher=fetcher, selector=selector, synth=synth, verifier=verifier, records=records)


class AgentTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.f = fake_dependencies()
        self.request = ResearchRequest(question=QUESTION, year_from=2020, year_to=2026, max_queries=3)

    def test_complete_fake_end_to_end_reuses_retrieval_and_ranking(self):
        original = deepcopy(self.f.records)
        result = self.f.agent.run(self.request)
        self.assertEqual(self.f.raw_client.search_works.call_count, 3)
        self.assertEqual(self.f.relevance.assess.call_count, 2)
        for call in self.f.raw_client.search_works.call_args_list:
            self.assertEqual(call.kwargs["year_from"], 2020)
            self.assertEqual(call.kwargs["year_to"], 2026)
            self.assertEqual(call.kwargs["per_page"], 10)
        self.assertEqual(self.f.records, original)
        self.assertEqual(self.request.question, QUESTION)
        self.assertEqual(result.run_stats.paper_ranking, "semantic")
        self.assertEqual(result.run_stats.planned_queries, 3)
        self.assertEqual(result.run_stats.raw_candidates, 2)
        self.assertEqual(result.run_stats.retrieved_query_hits, 6)
        self.assertEqual(result.run_stats.version_groups, 2)
        self.assertEqual(result.run_stats.full_text_papers, 1)
        self.assertEqual(result.run_stats.abstract_only_papers, 1)
        self.assertEqual(result.run_stats.passages_created, 2)
        self.assertEqual(result.run_stats.evidence_selected, 2)
        self.assertEqual(result.run_stats.claims_drafted, 2)
        self.assertEqual(result.run_stats.supported, 1)
        self.assertEqual(result.run_stats.unsupported, 1)
        self.assertIn("[P1, p. 3]", result.answer)
        self.assertNotIn("UNSUPPORTED:", result.answer)
        self.assertIn("UNSUPPORTED:", result.claims[1].claim.text)
        self.assertEqual({p.paper.paper_id for p in result.papers}, {p.paper_id for p in original})
        available = {p.passage_id for p in result.evidence}
        for record in result.claims:
            self.assertTrue(set(record.claim.evidence_ids) <= available)
        self.assertEqual(ResearchResult.model_validate_json(result.model_dump_json()), result)

    def test_verifier_gets_exact_cited_evidence_only(self):
        self.f.agent.run(self.request)
        for call in self.f.verifier.verify.call_args_list:
            claim, evidence = call.args
            self.assertEqual(claim.evidence_ids, [p.passage_id for p in evidence])
            self.assertEqual(len(evidence), 1)

    def test_semantic_failure_uses_marked_rrf_fallback(self):
        self.f.relevance.assess.side_effect = RuntimeError("Provider unavailable")
        result = self.f.agent.run(self.request)
        self.assertEqual(result.run_stats.paper_ranking, "rrf_fallback")
        self.assertTrue(any("using original RRF" in w for w in result.warnings))
        self.assertEqual(result.papers[0].paper.paper_id, self.f.records[0].paper_id)

    def test_semantic_validation_error_does_not_hide_defect(self):
        self.f.relevance.assess.side_effect = ValueError("Invalid schema")
        with self.assertRaises(ValueError):
            self.f.agent.run(self.request)

    def test_max_papers_limits_document_acquisition(self):
        result = self.f.agent.run(self.request.model_copy(update={"max_papers": 1}))
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(self.f.fetcher.acquire.call_count, 1)

    def test_representative_prefers_usable_abstract_then_url(self):
        a = paper("W1", title="Same work", abstract=None)
        b = paper("W2", title="Same work", open_access_url=None)
        c = paper("W3", title="Same work")
        f = fake_dependencies([a, b, c])
        result = f.agent.run(self.request)
        self.assertEqual(len(result.papers), 1)
        self.assertEqual(result.papers[0].paper.paper_id, c.paper_id)
        self.assertIsNone(f.relevance.assess.call_args.kwargs["abstract"])

    def test_one_document_failure_falls_back_and_continues(self):
        def acquire(p):
            if p.source_id == "W1":
                raise RuntimeError("download failed")
            return AcquiredDocument(pages=[page(p, 2)])
        self.f.fetcher.acquire.side_effect = acquire
        result = self.f.agent.run(self.request)
        self.assertEqual(result.run_stats.abstract_only_papers, 1)
        self.assertEqual(result.run_stats.full_text_papers, 1)
        self.assertTrue(any("fetcher failed" in w for w in result.warnings))

    def test_wrong_document_identity_is_rejected(self):
        self.f.fetcher.acquire.side_effect = lambda p: AcquiredDocument(pages=[page(paper("W999"))])
        with self.assertRaisesRegex(ValueError, "source paper"):
            self.f.agent.run(self.request)

    def test_source_title_whitespace_is_preserved_without_identity_failure(self):
        f = fake_dependencies([paper(title="  Shared caches  ")])
        result = f.agent.run(self.request)
        self.assertEqual(result.papers[0].paper.title, "  Shared caches  ")
        self.assertTrue(all(p.title == "  Shared caches  " for p in result.evidence))

    def test_no_papers_skips_later_stages(self):
        self.f.raw_client.search_works.return_value = []
        result = self.f.agent.run(self.request)
        self.assertEqual(result.papers, [])
        self.assertIn("Insufficient evidence", result.answer)
        self.f.relevance.assess.assert_not_called()
        self.f.selector.select.assert_not_called()

    def test_no_document_text_skips_generation(self):
        self.f.fetcher.acquire.side_effect = lambda p: AcquiredDocument(pages=[])
        result = self.f.agent.run(self.request)
        self.assertEqual(result.evidence, [])
        self.assertIn("Insufficient evidence", result.answer)
        self.f.synth.synthesize.assert_not_called()

    def test_no_lexical_match_skips_selection(self):
        self.f.fetcher.acquire.side_effect = lambda p: AcquiredDocument(pages=[page(p, text="Ocean temperatures and marine ecosystems influence distant weather systems.")])
        result = self.f.agent.run(self.request)
        self.assertEqual(result.run_stats.passages_retrieved, 0)
        self.f.selector.select.assert_not_called()

    def test_no_selected_evidence_skips_synthesis(self):
        self.f.selector.select.side_effect = lambda q, ps: EvidenceSelection(selected_passage_ids=[], coverage_notes="No support", insufficient_evidence=True)
        result = self.f.agent.run(self.request)
        self.assertEqual(result.claims, [])
        self.f.synth.synthesize.assert_not_called()
        self.f.verifier.verify.assert_not_called()

    def test_insufficient_selection_with_real_evidence_returns_partial_answer(self):
        self.f.selector.select.side_effect = lambda q, ps: EvidenceSelection(selected_passage_ids=[ps[0].passage_id], coverage_notes="Missing scope", insufficient_evidence=True)
        result = self.f.agent.run(self.request)
        self.assertIn("Partial answer", result.answer)
        self.assertTrue(any("insufficient coverage" in w for w in result.warnings))

    def test_unknown_evidence_from_injected_selector_rejected(self):
        self.f.selector.select.side_effect = lambda q, ps: EvidenceSelection(selected_passage_ids=["invented"], coverage_notes="x", insufficient_evidence=False)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.f.agent.run(self.request)
        self.f.synth.synthesize.assert_not_called()

    def test_unknown_evidence_from_injected_synthesizer_rejected(self):
        self.f.synth.synthesize.side_effect = lambda q, ps: AnswerDraft(claims=[AnswerClaim(claim_id="c", text="x", evidence_ids=["invented"])], limitations=[])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.f.agent.run(self.request)
        self.f.verifier.verify.assert_not_called()

    def test_unknown_claim_from_injected_verifier_rejected(self):
        self.f.verifier.verify.side_effect = lambda c, ps: ClaimVerification(claim_id="invented", status="supported", evidence_ids=c.evidence_ids, reason="x")
        with self.assertRaisesRegex(ValueError, "Unknown claim"):
            self.f.agent.run(self.request)

    def test_all_unsupported_gives_no_substantive_answer(self):
        self.f.verifier.verify.side_effect = lambda c, ps: ClaimVerification(claim_id=c.claim_id, status="unsupported", evidence_ids=c.evidence_ids, reason="No support")
        result = self.f.agent.run(self.request)
        self.assertIn("Insufficient evidence", result.answer)
        self.assertNotIn("Shared caches reduce latency for repeated reads.", result.answer)
        self.assertNotIn("UNSUPPORTED:", result.answer)

    def test_verifier_failure_propagates_without_partial_answer(self):
        original = self.f.verifier.verify.side_effect
        def verify(c, ps):
            if c.claim_id == "c2":
                raise RuntimeError("API failure")
            return original(c, ps)
        self.f.verifier.verify.side_effect = verify
        with self.assertRaisesRegex(RuntimeError, "API failure"):
            self.f.agent.run(self.request)

    def test_retrieval_failure_propagates(self):
        self.f.raw_client.search_works.side_effect = RuntimeError("OpenAlex unavailable")
        with self.assertRaisesRegex(RuntimeError, "OpenAlex unavailable"):
            self.f.agent.run(self.request)
        self.f.relevance.assess.assert_not_called()

    def test_invalid_planner_question_is_rejected(self):
        self.f.planner.plan.return_value = plan().model_copy(update={"research_question": "different"})
        with self.assertRaisesRegex(ValueError, "Planner returned"):
            self.f.agent.run(self.request)
        self.f.raw_client.search_works.assert_not_called()

    def test_production_run_does_not_read_benchmark_files(self):
        with patch("builtins.open", side_effect=AssertionError("No files should be read in the fake run")):
            self.f.agent.run(self.request)


class RenderingTests(OfflineTest):
    def test_all_statuses_have_correct_final_policy(self):
        statuses = ["supported", "partially_supported", "unsupported", "conflicting"]
        records = [VerifiedClaim(claim=AnswerClaim(claim_id=f"c{i}", text=f"finding_{status}", evidence_ids=["e1"]),
                    verification=ClaimVerification(claim_id=f"c{i}", status=status, evidence_ids=["e1"], reason="reason"))
                   for i, status in enumerate(statuses)]
        result = render_answer(records, [passage()], [SelectedPaper(citation_label="P1", group_id="g", paper=paper(), source_type="pdf")])
        self.assertIn("finding\\_supported", result)
        self.assertNotIn("finding\\_unsupported", result)
        self.assertIn("Partially supported; not established in full", result)
        self.assertIn("Unresolved disagreement", result)
        # The partial warning has no surviving supported assertion to cite.
        self.assertEqual(result.count("[P1, p. 1]"), 2)

    def test_abstract_citation_has_no_page_number(self):
        record = VerifiedClaim(claim=AnswerClaim(claim_id="c", text="Finding", evidence_ids=["e1"]),
                   verification=ClaimVerification(claim_id="c", status="supported", evidence_ids=["e1"], reason="r"))
        result = render_answer([record], [passage(number=None)], [SelectedPaper(citation_label="P1", group_id="g", paper=paper(), source_type="abstract")])
        self.assertIn("[P1, abstract]", result)
        self.assertNotIn("p. None", result)
        self.assertIn("2024", result)
        self.assertIn("openalex: W1", result)

    def test_unresolved_citation_is_an_error(self):
        record = VerifiedClaim(claim=AnswerClaim(claim_id="c", text="Finding", evidence_ids=["e1"]),
                   verification=ClaimVerification(claim_id="c", status="supported", evidence_ids=["e1"], reason="r"))
        with self.assertRaises(ValueError):
            render_answer([record], [], [])

    def test_malformed_oa_url_cannot_break_final_rendering(self):
        selected = SelectedPaper(citation_label="P1", group_id="g", paper=paper(open_access_url="https://[broken"))
        self.assertEqual(source_url(selected), "https://openalex.org/W1")
        self.assertIn("Sources", render_answer([], [], [selected]))


class CLITests(OfflineTest):
    def test_cli_passes_iteration_budgets(self):
        result = fake_dependencies().agent.run(ResearchRequest(question=QUESTION))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "offline"}), patch(
            "scripts.run_research_agent.ResearchAgent") as factory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            factory.return_value.run.return_value = result
            status = main(["--question", QUESTION, "--max-search-rounds", "3", "--max-follow-up-queries", "1"])
        self.assertEqual(status, 0)
        request = factory.return_value.run.call_args.args[0]
        self.assertEqual((request.max_search_rounds, request.max_follow_up_queries), (3, 1))

    def test_invalid_request_has_actionable_validation_message(self):
        with redirect_stderr(io.StringIO()) as stderr:
            status = main(["--question", QUESTION, "--max-queries", "1"])
        self.assertEqual(status, 1)
        self.assertIn("max_queries", stderr.getvalue())

    def test_cli_writes_complete_json_and_prints_answer(self):
        result = fake_dependencies().agent.run(ResearchRequest(question=QUESTION))
        output = self.temporary_directory() / "result.json"
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-do-not-save"}), patch(
            "scripts.run_research_agent.ResearchAgent") as factory, redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
            factory.return_value.run.return_value = result
            status = main(["--question", QUESTION, "--output", str(output)])
        self.assertEqual(status, 0)
        self.assertIn("[P1, p. 3]", stdout.getvalue())
        saved = output.read_text(encoding="utf-8")
        self.assertEqual(ResearchResult.model_validate_json(saved), result)
        self.assertNotIn("fake-do-not-save", saved)

    def test_missing_key_fails_before_agent_construction(self):
        with patch.dict("os.environ", {}, clear=True), patch("scripts.run_research_agent.ResearchAgent") as factory, redirect_stderr(io.StringIO()) as stderr:
            status = main(["--question", QUESTION])
        self.assertEqual(status, 2)
        self.assertIn("OPENAI_API_KEY", stderr.getvalue())
        factory.assert_not_called()

    def test_failed_run_does_not_overwrite_existing_output_or_expose_secrets(self):
        output = self.temporary_directory() / "result.json"
        output.write_text("previous run", encoding="utf-8")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "secret"}), patch("scripts.run_research_agent.ResearchAgent") as factory, redirect_stderr(io.StringIO()) as stderr:
            factory.return_value.run.side_effect = RuntimeError("https://api?api_key=secret")
            status = main(["--question", QUESTION, "--output", str(output)])
        self.assertEqual(status, 1)
        self.assertEqual(output.read_text(encoding="utf-8"), "previous run")
        self.assertNotIn("secret", stderr.getvalue())
