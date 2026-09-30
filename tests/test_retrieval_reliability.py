"""Offline regressions for production query isolation, retries and diagnostics."""

import json
from unittest.mock import Mock, patch

import httpx

from phase5_helpers import OfflineTest, QUESTION, paper
from test_research_graph import loop_fixture
from researchpilot.openalex_client import OpenAlexClient
from researchpilot.research_iteration import EvidenceAssessment
from researchpilot.research_models import ResearchRequest
from researchpilot.retrieval_diagnostics import error_details
from researchpilot.run_manager import safe_error


def status_error(status):
    response = httpx.Response(status, request=httpx.Request("GET", "https://api.openalex.org/works"))
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        return error
    raise AssertionError("Expected HTTP error")


class ProductionQueryTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.f = loop_fixture()
        base = self.f.planner.plan.return_value
        self.queries = [base.queries[0].model_copy(update={"query_id": f"q{i}", "text": f"query {i}"})
                        for i in range(1, 6)]
        self.f.planner.plan.return_value = base.model_copy(update={"queries": self.queries})
        self.f.assessor.assess.side_effect = None
        self.f.assessor.assess.return_value = EvidenceAssessment(sufficient=True, gaps=[], rationale="Covered.")
        self.records = [paper(f"W{i}", title=f"Distinct study {i}") for i in range(1, 6)]
        self.responses = {q.text: [p] for q, p in zip(self.queries, self.records)}
        def search(query, **kwargs):
            value = self.responses[query]
            if isinstance(value, Exception):
                raise value
            return value
        self.f.raw.search_works.side_effect = search

    def run_agent(self):
        return self.f.agent.run(ResearchRequest(question=QUESTION, year_from=2010, year_to=2020,
                                               max_papers=5, max_search_rounds=1))

    def test_five_successes_preserve_order_provenance_and_bounds(self):
        result = self.run_agent()
        self.assertEqual([p.paper.paper_id for p in result.papers], [p.paper_id for p in self.records])
        self.assertEqual(result.run_stats.raw_candidates, 5)
        self.assertEqual(result.run_stats.retrieved_query_hits, 5)
        self.assertEqual([q.query_id for q in result.executed_queries], [q.query_id for q in self.queries])
        for call in self.f.raw.search_works.call_args_list:
            self.assertEqual((call.kwargs["year_from"], call.kwargs["year_to"], call.kwargs["per_page"]),
                             (2010, 2020, 10))

    def test_four_successes_one_timeout_continue_with_explicit_warning(self):
        self.responses["query 3"] = httpx.ReadTimeout("temporary timeout")
        result = self.run_agent()
        self.assertEqual(len(result.papers), 4)
        self.assertEqual(self.f.raw.search_works.call_count, 5)
        self.assertTrue(any("q3" in w and "successful queries only" in w for w in result.warnings))
        self.assertEqual(result.run_stats.retrieved_query_hits, 4)

    def test_one_success_four_network_failures_continue(self):
        for q in self.queries[1:]:
            self.responses[q.text] = httpx.ConnectError("unreachable")
        result = self.run_agent()
        self.assertEqual([p.paper.paper_id for p in result.papers], [self.records[0].paper_id])
        self.assertEqual(sum("failed temporarily" in w for w in result.warnings), 4)
        self.assertEqual(self.f.raw.search_works.call_count, 5)

    def test_five_successful_empty_queries_stop_without_llm_assessment_or_follow_up(self):
        self.responses = {q.text: [] for q in self.queries}
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "no_literature_found")
        self.assertEqual(result.papers, [])
        self.assertEqual(result.evidence, [])
        self.assertIn("Insufficient evidence", result.answer)
        self.assertTrue(any("No relevant literature found" in w for w in result.warnings))
        self.assertEqual(self.f.raw.search_works.call_count, 5)
        for dependency in (self.f.relevance.assess, self.f.assessor.assess, self.f.follow_up.plan,
                           self.f.synthesizer.synthesize):
            dependency.assert_not_called()

    def test_empty_success_and_failures_are_not_called_total_service_outage(self):
        self.responses = {q.text: httpx.ConnectError("unreachable") for q in self.queries}
        self.responses["query 4"] = []
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "no_literature_found")
        self.assertEqual(sum("failed temporarily" in w for w in result.warnings), 4)

    def test_five_network_failures_propagate_and_never_publish_partial_result(self):
        self.responses = {q.text: httpx.ConnectError("unreachable") for q in self.queries}
        with self.assertRaises(httpx.ConnectError):
            self.run_agent()
        self.assertEqual(self.f.raw.search_works.call_count, 5)
        self.f.relevance.assess.assert_not_called()
        self.f.synthesizer.synthesize.assert_not_called()

    def test_malformed_response_is_not_skipped_after_a_success(self):
        failure = ValueError("OpenAlex response must contain a 'results' list.")
        self.responses["query 2"] = failure
        with self.assertRaises(ValueError) as caught:
            self.run_agent()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.f.raw.search_works.call_count, 2)
        self.f.relevance.assess.assert_not_called()

    def test_rate_limit_or_server_error_allows_successful_queries(self):
        for status in (429, 503):
            with self.subTest(status=status):
                self.responses["query 1"] = status_error(status)
                result = self.run_agent()
                self.assertEqual(len(result.papers), 4)
                self.assertTrue(any("q1" in w and "failed temporarily" in w for w in result.warnings))

    def test_bad_request_and_programming_errors_abort_without_partial_answer(self):
        for failure in (status_error(400), status_error(401), RuntimeError("closed client"), TypeError("bug")):
            with self.subTest(failure=type(failure).__name__):
                self.responses["query 2"] = failure
                self.f.raw.search_works.reset_mock()
                with self.assertRaises(type(failure)):
                    self.run_agent()
                self.assertEqual(self.f.raw.search_works.call_count, 2)

    def test_repeated_papers_keep_best_rank_and_all_successful_query_hits(self):
        self.responses["query 1"] = [self.records[1], self.records[0], self.records[0]]
        self.responses["query 2"] = [self.records[0]]
        self.responses["query 3"] = httpx.ReadTimeout("timeout")
        self.responses["query 4"] = self.responses["query 5"] = []
        result = self.run_agent()
        self.assertEqual(result.run_stats.raw_candidates, 2)
        self.assertEqual(result.run_stats.retrieved_query_hits, 3)
        groups = self.f.relevance.assess.call_count
        self.assertEqual(groups, 2)
        self.assertEqual(result.papers[0].paper.paper_id, self.records[0].paper_id)


class OpenAlexRetryTests(OfflineTest):
    def make_client(self, handler, retries=2):
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        return OpenAlexClient(http_client=client, max_retries=retries)

    def test_transient_failures_retry_then_return_real_response(self):
        for failure in (httpx.ConnectError("connect"), httpx.ReadTimeout("timeout"), 429, 503):
            calls = []
            def handler(request):
                calls.append(request)
                if len(calls) < 3:
                    if isinstance(failure, Exception):
                        raise failure
                    return httpx.Response(failure)
                return httpx.Response(200, json={"results": [{"id": "https://openalex.org/W1", "title": "Study"}]})
            with self.subTest(failure=failure), patch("researchpilot.openalex_client.sleep") as sleep:
                result = self.make_client(handler).search_works("original query", year_from=2010, year_to=2020)
                self.assertEqual(result[0].source_id, "W1")
                self.assertEqual(len(calls), 3)
                self.assertEqual([x.args[0] for x in sleep.call_args_list], [0.5, 1.0])
                self.assertTrue(all(r.url == calls[0].url for r in calls))

    def test_retry_exhaustion_is_bounded_and_propagates_original_exception(self):
        calls = []
        failure = httpx.ConnectTimeout("timeout")
        def handler(request):
            calls.append(request)
            raise failure
        with patch("researchpilot.openalex_client.sleep") as sleep:
            with self.assertRaises(httpx.ConnectTimeout) as caught:
                self.make_client(handler).search_works("query")
        self.assertIs(caught.exception, failure)
        self.assertEqual((len(calls), sleep.call_count), (3, 2))

    def test_invalid_request_json_schema_and_closed_client_are_never_retried(self):
        responses = [httpx.Response(400), httpx.Response(401), httpx.Response(403),
                     httpx.Response(200, content=b"not json"), httpx.Response(200, json={}),
                     httpx.Response(200, json={"results": [{"id": "W1", "title": None}]})]
        for response in responses:
            handler = Mock(return_value=response)
            with self.subTest(response=response), patch("researchpilot.openalex_client.sleep") as sleep:
                with self.assertRaises((httpx.HTTPStatusError, ValueError)):
                    self.make_client(handler).search_works("query")
                self.assertEqual(handler.call_count, 1)
                sleep.assert_not_called()
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        client.close()
        with patch("researchpilot.openalex_client.sleep") as sleep, self.assertRaises(RuntimeError):
            OpenAlexClient(http_client=client, max_retries=2).search_works("query")
        sleep.assert_not_called()

    def test_retry_after_is_respected_without_unbounded_waiting(self):
        for value, attempts in (("2", 3), ("60", 1), ("120", 1), ("Wed, 21 Oct 2030 07:28:00 GMT", 1)):
            handler = Mock(side_effect=lambda r: httpx.Response(429, headers={"Retry-After": value}))
            with self.subTest(value=value), patch("researchpilot.openalex_client.sleep") as sleep:
                with self.assertRaises(httpx.HTTPStatusError):
                    self.make_client(handler).search_works("query")
                self.assertEqual(handler.call_count, attempts)
                self.assertEqual([c.args[0] for c in sleep.call_args_list], [2.0, 2.0] if attempts == 3 else [])

    def test_invalid_retry_budget_rejected(self):
        for value in (-1, 3, True, 1.5, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                OpenAlexClient(max_retries=value)


class RetrievalErrorTests(OfflineTest):
    def test_failure_categories_remain_distinct(self):
        self.assertIn("temporarily unavailable", safe_error(httpx.ReadTimeout("timeout"), "searching"))
        self.assertIn("temporarily unavailable", safe_error(status_error(429), "searching"))
        self.assertIn("Invalid/generated search query", safe_error(status_error(400), "searching"))
        self.assertIn("rejected access", safe_error(status_error(403), "searching"))
        self.assertIn("Internal research pipeline error", safe_error(ValueError("bad JSON"), "searching"))
        self.assertIn("Internal research pipeline error", safe_error(RuntimeError("closed"), "searching"))

    def test_http_diagnostics_redact_keys_urls_and_keep_status_and_stack(self):
        request = httpx.Request("GET", "https://api.openalex.org/works?api_key=private-test-key&search=secret-query")
        try:
            httpx.Response(400, request=request).raise_for_status()
        except httpx.HTTPStatusError as error:
            data = error_details(error)
        text = json.dumps(data)
        self.assertEqual(data["http_status"], 400)
        self.assertEqual(data["error_type"], "HTTPStatusError")
        self.assertTrue(data["traceback"])
        self.assertNotIn("private-test-key", text)
        self.assertNotIn("secret-query", text)
        self.assertNotIn("api_key", text)

    def test_non_http_exception_message_is_not_logged(self):
        self.assertNotIn("private prompt", json.dumps(error_details(ValueError("private prompt"))))

    def test_anonymous_search_pause_retains_503_and_retry_after_diagnostics(self):
        request = httpx.Request("GET", "https://api.openalex.org/works")
        try:
            httpx.Response(503, headers={"Retry-After": "60"}, request=request,
                           json={"error": "Search temporarily unavailable",
                                 "message": "Anonymous search is paused."}).raise_for_status()
        except httpx.HTTPStatusError as error:
            details = error_details(error)
        self.assertEqual(details["http_status"], 503)
        self.assertEqual(details["retry_after_seconds"], 60)
        self.assertIn("503 Service Unavailable", details["error_message"])
