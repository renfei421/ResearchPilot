"""Bounded cross-run context and cache reuse, without external services."""

from datetime import datetime, timedelta, timezone
import json
from unittest.mock import Mock, patch

import httpx

from phase5_helpers import OfflineTest, paper, candidate, page, pdf_bytes
from phase12_helpers import project, finish, research_fixture, QUESTION
from claim_check_helpers import fixture as idea_fixture, CLAIM
from test_hybrid_retrieval import FakeEmbeddingClient
from researchpilot.project_models import ProjectClaim, ProjectPaper, ProjectGap, ProjectUpdate
from researchpilot.project_context import ProjectContextBuilder, project_usage
from researchpilot.project_reuse import ProjectSearchSession, seed_project
from researchpilot.project_api import continuation_request
from researchpilot.project_models import ProjectContinue
from researchpilot.research_models import ResearchRequest
from researchpilot.claim_check import ClaimCheckRequest
from researchpilot.run_store import RunStore
from researchpilot.paper_candidate import SearchQuery
from researchpilot.document_acquisition import DocumentAcquirer, PdfTextExtractor
from researchpilot.embedding_cache import EmbeddingCache, EmbeddingStats
from researchpilot.openai_query_planner import OpenAIQueryPlanner
from researchpilot.query_planner import PlannerRequest


class ProjectContextTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.root = self.temporary_directory()
        self.store = RunStore(self.root / "runs.db")
        self.p = project(self.store)
        self.ps = self.store.projects
        self.first = finish(self.store, self.p.project_id)
        self.request = ResearchRequest(question=QUESTION, project_id=self.p.project_id, max_search_rounds=1)

    def context(self, request=None, **limits):
        return ProjectContextBuilder(self.ps, **limits).build(self.p.project_id, request or self.request)

    def test_relevant_claim_selected(self):
        self.assertTrue(self.context().claims)

    def test_unrelated_claim_excluded(self):
        claim = self.ps.add_claim(self.p.project_id, "Ocean plankton migration")
        self.assertNotIn(claim.claim_id, self.context().record_ids())

    def test_relevant_evidence_selected(self):
        self.assertEqual(len(self.context().evidence), 1)

    def test_unrelated_evidence_excluded(self):
        self.assertEqual(self.context("Ocean plankton migration").evidence, [])

    def test_related_unresolved_gap_included(self):
        self.assertEqual(self.context().gaps[0].status, "unresolved")

    def test_resolved_gap_excluded(self):
        finish(self.store, self.p.project_id, research_fixture(second=True), refresh_search=True)
        self.assertEqual(self.context().gaps, [])

    def test_archived_claim_excluded(self):
        row = self.context().claims[0]
        self.ps.archive_record(self.p.project_id, "claims", row.claim_id)
        self.assertNotIn(row.claim_id, self.context().record_ids())

    def test_record_limit(self):
        self.assertLessEqual(len(self.context(max_records=2).record_ids()), 2)

    def test_character_limit_with_long_records(self):
        for i in range(8):
            self.ps.add_note(self.p.project_id, "Spectral recovery " + str(i) + "x" * 5000)
        c = self.context(max_characters=1500)
        self.assertLessEqual(len(c.model_dump_json()), 1500)
        self.assertTrue(all(e.passage.paper_id in {p.paper_id for p in c.papers} for e in c.evidence))

    def test_no_full_history_dump(self):
        for i in range(30):
            self.ps.add_claim(self.p.project_id, f"Ocean {i} plankton sentinel")
        payload = self.context().planner_payload()
        self.assertNotIn("sentinel", json.dumps(payload))
        self.assertNotIn("runs", payload)
        self.assertNotIn("local_document_reference", json.dumps(payload))

    def test_evidence_provenance_unchanged(self):
        c = self.context()
        e = c.evidence[0]
        self.assertEqual(e.originating_run_id, self.first.run.run_id)
        self.assertEqual(e.passage, self.first.run.result.evidence[0])

    def test_notes_are_explicitly_unverified(self):
        self.ps.add_note(self.p.project_id, "Spectral recovery note")
        payload = self.context().planner_payload()
        self.assertEqual(payload["user_notes_unverified"], ["Spectral recovery note"])

    def test_context_is_project_isolated(self):
        other = project(self.store)
        self.ps.add_claim(other.project_id, "Spectral private sentinel")
        self.assertNotIn("private sentinel", self.context().model_dump_json())

    def test_date_bounds_apply_to_reused_papers(self):
        r = self.request.model_copy(update={"year_from": 2025})
        self.assertEqual(self.context(r).papers, [])
        self.assertEqual(self.context(r).evidence, [])

    def test_seed_does_not_mutate_context(self):
        c = self.context()
        before = c.model_dump()
        sources, evidence, _, _ = seed_project(c, max_papers=1)
        sources[0].paper.title = "Modified only in run"
        next(iter(evidence.values())).text = "Modified only in run"
        self.assertEqual(c.model_dump(), before)

    def test_reused_evidence_is_offered_to_current_verifier(self):
        f = research_fixture(second=True)
        finish(self.store, self.p.project_id, f, refresh_search=True)
        cited = [p.passage_id for call in f.verifier.verify.call_args_list for p in call.args[1]]
        self.assertIn(self.context().evidence[0].passage.passage_id, cited)

    def test_historical_support_does_not_bypass_new_verification(self):
        f = research_fixture(supported=False)
        second = finish(self.store, self.p.project_id, f)
        self.assertGreater(f.verifier.verify.call_count, 0)
        self.assertTrue(all(c.verification.status == "unsupported" for c in second.run.result.claims))
        # Historical finding remains historical, the current run is not promoted.
        self.assertEqual(len(self.ps.list_records(self.p.project_id, "findings")), 1)

    def test_reuse_diagnostics_identify_old_and_new_evidence(self):
        second = finish(self.store, self.p.project_id, research_fixture(second=True), refresh_search=True)
        usage = second.run.result.project_usage
        self.assertEqual(set(usage.evidence_origins.values()), {"reused_project_evidence", "newly_retrieved_evidence"})
        self.assertEqual(len(usage.reused_paper_ids), 1)
        self.assertEqual((usage.new_papers, usage.new_evidence), (1, 1))

    def test_research_planner_receives_bounded_context(self):
        f = research_fixture()
        finish(self.store, self.p.project_id, f)
        payload = f.planner.plan.call_args.args[0].project_context
        self.assertTrue(payload["open_gaps"])
        self.assertTrue(payload["evidence"])
        self.assertTrue(payload["previous_queries"])

    def test_idea_reanalyzes_prior_evidence_for_current_assertions(self):
        f = idea_fixture()
        r = ClaimCheckRequest(claim=CLAIM + " Investigate spectral recovery.", project_id=self.p.project_id, max_search_rounds=1)
        f.agent.project_context = self.context(r)
        result = f.agent.run(r)
        old = self.first.run.result.evidence[0].passage_id
        self.assertTrue(any(old in [p.passage_id for p in call[2]] for call in f.analyzer.calls))
        self.assertTrue(any(s.paper.paper_id == self.first.run.result.papers[0].paper.paper_id for s in result.sources))
        self.assertTrue(f.planner.calls[0][2])

    def test_continue_gap_becomes_bounded_research_request(self):
        gap = self.context().gaps[0]
        r = continuation_request(self.ps, self.p.project_id, ProjectContinue(target_type="gap", target_id=gap.gap_id))
        self.assertEqual(r.target_gap_id, gap.gap_id)
        self.assertIn(gap.description, r.question)
        self.assertEqual(r.max_search_rounds, 2)

    def test_continue_idea_preserves_hypothesis(self):
        claim = self.ps.add_claim(self.p.project_id, CLAIM)
        r = continuation_request(self.ps, self.p.project_id, ProjectContinue(target_type="claim", target_id=claim.claim_id,
            mode="claim_check", instruction="Check recent work", refresh_search=True))
        self.assertEqual(r.claim, CLAIM)
        self.assertEqual(r.context, "Check recent work")
        self.assertTrue(r.refresh_search)

    def test_exact_query_reused_with_new_query_id(self):
        backend = Mock()
        session = ProjectSearchSession(backend, self.ps, self.request, "new")
        rows = session.search_candidates([SearchQuery(query_id="renamed", text="spectral recovery")])
        backend.search_candidates.assert_not_called()
        self.assertEqual(rows[0].hits[0].query_id, "renamed")
        self.assertEqual(session.reused, 1)

    def test_new_question_requires_new_search(self):
        backend = Mock(); backend.search_candidates.return_value = []
        session = ProjectSearchSession(backend, self.ps, self.request.model_copy(update={"question": "New spectral theorem?"}), "new")
        session.search_candidates([SearchQuery(query_id="q1", text="spectral recovery")])
        backend.search_candidates.assert_called_once()

    def test_explicit_refresh_requires_new_search(self):
        backend = Mock(); backend.search_candidates.return_value = []
        session = ProjectSearchSession(backend, self.ps, self.request.model_copy(update={"refresh_search": True}), "new")
        session.search_candidates([SearchQuery(query_id="q1", text="spectral recovery")])
        backend.search_candidates.assert_called_once()

    def test_changed_year_requires_new_search(self):
        backend = Mock(); backend.search_candidates.return_value = []
        session = ProjectSearchSession(backend, self.ps, self.request, "new")
        session.search_candidates([SearchQuery(query_id="q1", text="spectral recovery")], year_from=2020)
        backend.search_candidates.assert_called_once()

    def test_changed_page_size_requires_new_search(self):
        backend = Mock(); backend.search_candidates.return_value = []
        session = ProjectSearchSession(backend, self.ps, self.request, "new")
        session.search_candidates([SearchQuery(query_id="q1", text="spectral recovery")], per_query=5)
        backend.search_candidates.assert_called_once()

    def test_stale_query_requires_new_search(self):
        for row in self.ps.list_records(self.p.project_id, "queries"):
            self.ps.put("queries", row.model_copy(update={"updated_at": (datetime.now(timezone.utc)-timedelta(days=2)).isoformat()}))
        backend = Mock(); backend.search_candidates.return_value = []
        ProjectSearchSession(backend, self.ps, self.request, "new").search_candidates([SearchQuery(query_id="q1", text="spectral recovery")])
        backend.search_candidates.assert_called_once()

    def test_search_failure_not_cached(self):
        backend = Mock(); backend.search_candidates.side_effect = RuntimeError("failed")
        session = ProjectSearchSession(backend, self.ps, self.request, "new")
        with self.assertRaises(RuntimeError):
            session.search_candidates([SearchQuery(query_id="q1", text="new search")])
        self.assertEqual(session.records, {})

    def test_cached_pdf_not_redownloaded_for_project_paper(self):
        self.check_document_cache(check_extractor=False)

    def test_extracted_pages_reused_for_project_paper(self):
        self.check_document_cache(check_extractor=True)

    def check_document_cache(self, check_extractor):
        p = self.context().papers[0].paper
        transport = Mock(return_value=httpx.Response(200, content=pdf_bytes()))
        extractor = Mock(wraps=PdfTextExtractor())
        with httpx.Client(transport=httpx.MockTransport(transport)) as client:
            first = DocumentAcquirer(self.root / "docs", http_client=client, extractor=extractor).acquire(p)
            second = DocumentAcquirer(self.root / "docs", http_client=client, extractor=extractor).acquire(p)
        self.assertTrue(second.cache_hit)
        self.assertEqual(first.pages, second.pages)
        self.assertEqual(extractor.extract.call_count if check_extractor else transport.call_count, 1)

    def test_project_evidence_embeddings_reused_after_restart(self):
        text = self.context().evidence[0].passage.text
        client = FakeEmbeddingClient()
        first_stats, second_stats = EmbeddingStats(), EmbeddingStats()
        a = EmbeddingCache(self.root / "embeddings", client).embed([text], first_stats, passages=True)
        b = EmbeddingCache(self.root / "embeddings", client).embed([text], second_stats, passages=True)
        self.assertEqual(a, b)
        self.assertEqual((len(client.calls), second_stats.cache_hits), (1, 1))

    def test_invalid_context_limits(self):
        with self.assertRaises(ValueError):
            ProjectContextBuilder(self.ps, max_records=True)

    def test_research_rejects_context_from_another_project(self):
        f = research_fixture()
        f.agent.project_context = self.context()
        with self.assertRaisesRegex(ValueError, "current request"):
            f.agent.run(self.request.model_copy(update={"project_id": "other"}))

    def test_idea_rejects_context_from_another_project(self):
        f = idea_fixture()
        f.agent.project_context = self.context()
        with self.assertRaisesRegex(ValueError, "current request"):
            f.agent.run(ClaimCheckRequest(claim=CLAIM, project_id="other"))
