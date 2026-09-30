"""Offline planner adapter tests using SDK injection and local MockTransport."""

from copy import deepcopy
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import ValidationError

from researchpilot import openai_query_planner as planner_module
from researchpilot.openai_query_planner import OpenAIQueryPlanner
from researchpilot.query_plan import SearchPlan
from researchpilot.query_planner import PlannerRequest, QueryPlanner


QUESTION = "How do shared caches affect storage latency?"


def plan_data(question: str = QUESTION, count: int = 3) -> dict:
    # Independent handcrafted fixtures; no benchmark artifacts are loaded.
    texts = [
        '"shared cache" AND latency',
        '"cache contention" AND throughput',
        '"request scheduling" AND "cache allocation"',
        '"buffer sharing" AND performance',
        '"cache isolation" AND "tail latency"',
    ]
    roles = ["core", "facet", "bridge", "synonym", "facet"]
    return {
        "research_question": question,
        "concepts": ["shared caches", "storage latency"],
        "queries": [
            {"query_id": f"q{i}", "text": text, "role": role,
             "rationale": f"Covers fixture search dimension {i}."}
            for i, (text, role) in enumerate(zip(texts[:count], roles[:count]), 1)
        ],
    }


def sdk_response_data(plan: dict) -> dict:
    return {
        "id": "resp_offline", "object": "response", "created_at": 0,
        "status": "completed", "model": "gpt-5.6-terra",
        "output": [{
            "id": "msg_offline", "type": "message", "status": "completed",
            "role": "assistant", "content": [
                {"type": "output_text", "text": json.dumps(plan), "annotations": []}
            ],
        }],
    }


class OpenAIQueryPlannerTests(unittest.TestCase):
    def setUp(self):
        self.request = PlannerRequest(research_question=QUESTION)
        self.search_plan = SearchPlan.model_validate(plan_data())
        self.response = SimpleNamespace(status="completed", output=[], output_parsed=self.search_plan)
        self.sdk = MagicMock()
        self.sdk.with_options.return_value = self.sdk
        self.sdk.responses.parse.return_value = self.response
        self.planner = OpenAIQueryPlanner(client=self.sdk)
        for target in ("socket.create_connection", "socket.socket.connect"):
            guard = patch(target, side_effect=AssertionError("Tests must remain offline"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_successful_structured_parse_uses_default_model_and_no_tools(self):
        planner: QueryPlanner = self.planner
        result = planner.plan(self.request)
        self.assertIs(result, self.search_plan)
        self.sdk.responses.parse.assert_called_once()
        params = self.sdk.responses.parse.call_args.kwargs
        self.assertEqual(params["model"], "gpt-5.6-terra")
        self.assertIs(params["text_format"], SearchPlan)
        self.assertIs(params["store"], False)
        self.assertEqual(set(params), {"model", "store", "instructions", "input", "text_format"})

    def test_only_normalized_request_fields_are_sent_as_json(self):
        request = PlannerRequest(
            research_question='  How do "shared caches" affect latency at α scale?\n',
            year_from=2020, year_to=2025, max_queries=4,
        )
        self.response.output_parsed = SearchPlan.model_validate(plan_data(request.research_question))
        before = request.model_dump()
        self.planner.plan(request)
        payload = json.loads(self.sdk.responses.parse.call_args.kwargs["input"])
        self.assertEqual(payload, before)
        self.assertEqual(set(payload), {"research_question", "year_from", "year_to", "max_queries"})
        self.assertEqual(request.model_dump(), before)

    def test_absent_years_are_serialized_as_null(self):
        self.planner.plan(self.request)
        payload = json.loads(self.sdk.responses.parse.call_args.kwargs["input"])
        self.assertIsNone(payload["year_from"])
        self.assertIsNone(payload["year_to"])

    def test_configured_model_is_passed(self):
        OpenAIQueryPlanner(model="configured-model", client=self.sdk).plan(self.request)
        self.assertEqual(self.sdk.responses.parse.call_args.kwargs["model"], "configured-model")

    def test_instructions_are_static_and_have_no_handcrafted_search_examples(self):
        instructions = []
        for question in (QUESTION, "How does coastal shading affect local temperature?"):
            request = PlannerRequest(research_question=question)
            self.response.output_parsed = SearchPlan.model_validate(plan_data(question))
            # The planner has no reason to load any benchmark or other files.
            with patch("pathlib.Path.open", side_effect=AssertionError("No artifact reads")):
                self.planner.plan(request)
            value = self.sdk.responses.parse.call_args.kwargs["instructions"]
            self.assertNotIn(question, value)
            # Forbid quoted Boolean search examples, without reading gold or
            # handcrafted benchmark queries to construct a comparison fixture.
            self.assertNotRegex(value, r'["“][^"”\n]+["”]\s+(?:AND|OR|NOT)\b')
            self.assertNotRegex(value.casefold(), r"for example|e\.g\.|benchmark_id|eval/datasets")
            instructions.append(value)
        self.assertEqual(instructions[0], instructions[1])
        self.assertEqual(instructions[0], planner_module._PLANNER_INSTRUCTIONS)

    def test_instructions_define_strategy_roles_and_year_handling(self):
        self.planner.plan(self.request)
        instructions = " ".join(self.sdk.responses.parse.call_args.kwargs["instructions"].split())
        for phrase in (
            "small complementary academic search strategy",
            "not an answer to the research question",
            "Use only the supplied request",
            "between 3 and request.max_queries",
            "rather than repeatedly paraphrasing the same idea",
            "Balance precision and recall", "severe semantic drift",
            "requiring every concept from a multi-part research question to occur in one paper",
            "single-line academic search expression", "quotes and Boolean operators",
            "Do not produce API URLs or API parameters",
            "Do not put publication-year filtering into query text",
            "year_from and year_to are contextual constraints only",
            "downstream retrieval handles them structurally",
            "core: targets the central research problem",
            "facet: targets an important dimension or constraint",
            "bridge: connects concepts that may live in different literatures",
            "synonym: targets alternative terminology or lexical formulations",
            "At least one query must have role=core",
            "rationale explaining the distinct part of the search space",
            "Concepts should summarize the important concepts",
            "Copy research_question exactly as supplied",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, instructions)

    def test_injected_client_remains_caller_owned_with_retries_disabled(self):
        self.planner.plan(self.request)
        self.sdk.with_options.assert_called_once_with(max_retries=0, timeout=60.0)
        self.sdk.close.assert_not_called()
        self.sdk.__exit__.assert_not_called()

    def test_incomplete_responses_rejected_even_with_parsed_plan(self):
        for status in ("incomplete", "failed", "queued", "in_progress", "cancelled", None):
            with self.subTest(status=status):
                self.response.status = status
                self.sdk.responses.parse.reset_mock()
                with self.assertRaisesRegex(RuntimeError, "not completed"):
                    self.planner.plan(self.request)
                self.sdk.responses.parse.assert_called_once()

    def test_missing_or_wrong_parsed_output_rejected(self):
        for value in (None, plan_data(), "unparsed text"):
            with self.subTest(value_type=type(value).__name__):
                self.response.output_parsed = value
                with self.assertRaisesRegex(RuntimeError, "no parsed SearchPlan"):
                    self.planner.plan(self.request)

    def test_explicit_refusal_rejected_even_alongside_a_parsed_plan(self):
        self.response.output = [SimpleNamespace(
            type="message", content=[SimpleNamespace(type="refusal", refusal="Cannot comply")]
        )]
        for value in (None, self.search_plan):
            with self.subTest(has_parsed_plan=value is not None):
                self.response.output_parsed = value
                with self.assertRaisesRegex(RuntimeError, "refused"):
                    self.planner.plan(self.request)

    def test_question_mismatch_raises_without_rewriting_output(self):
        self.response.output_parsed = SearchPlan.model_validate(plan_data(QUESTION.upper()))
        before = self.response.output_parsed.model_dump()
        with self.assertRaisesRegex(ValueError, "research_question must match"):
            self.planner.plan(self.request)
        self.assertEqual(self.response.output_parsed.model_dump(), before)
        self.sdk.responses.parse.assert_called_once()

    def test_output_above_request_limit_rejected_without_truncation(self):
        request = PlannerRequest(research_question=QUESTION, max_queries=3)
        self.response.output_parsed = SearchPlan.model_validate(plan_data(count=4))
        before = self.response.output_parsed.model_dump()
        with self.assertRaisesRegex(ValueError, "exceeding request.max_queries=3"):
            self.planner.plan(request)
        self.assertEqual(self.response.output_parsed.model_dump(), before)
        self.sdk.responses.parse.assert_called_once()

    def test_exactly_max_queries_accepted(self):
        for count in (3, 4, 5):
            with self.subTest(count=count):
                request = PlannerRequest(research_question=QUESTION, max_queries=count)
                self.response.output_parsed = SearchPlan.model_validate(plan_data(count=count))
                self.assertEqual(len(self.planner.plan(request).queries), count)

    def test_three_queries_accepted_when_max_queries_is_larger(self):
        self.assertEqual(len(self.planner.plan(self.request).queries), 3)
        self.assertEqual(self.request.max_queries, 5)

    def test_sdk_and_transport_exceptions_propagate_without_fallback_or_retry(self):
        for failure in (
            RuntimeError("SDK failed"),
            APIConnectionError(request=httpx.Request("POST", "https://example.invalid/responses")),
        ):
            with self.subTest(error=type(failure).__name__):
                self.sdk.responses.parse.reset_mock()
                self.sdk.responses.parse.side_effect = failure
                result = None
                with self.assertRaises(type(failure)) as caught:
                    result = self.planner.plan(self.request)
                self.assertIs(caught.exception, failure)
                self.assertIsNone(result)
                self.sdk.responses.parse.assert_called_once()
                self.sdk.close.assert_not_called()

    def test_owned_client_uses_environment_and_closes_after_success_or_failure(self):
        for failure in (None, RuntimeError("SDK failed")):
            with self.subTest(failure=failure is not None):
                created = []

                def create_sdk(**kwargs):
                    sdk = OpenAI(**kwargs)
                    created.append(sdk)
                    return sdk

                with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-environment-key"}):
                    with patch.object(planner_module, "OpenAI", side_effect=create_sdk) as factory:
                        with patch.object(OpenAI, "post", return_value=self.response, side_effect=failure) as post:
                            planner = OpenAIQueryPlanner()
                            factory.assert_not_called()
                            if failure is None:
                                self.assertIs(planner.plan(self.request), self.search_plan)
                            else:
                                with self.assertRaises(RuntimeError) as caught:
                                    planner.plan(self.request)
                                self.assertIs(caught.exception, failure)
                            post.assert_called_once()
                            factory.assert_called_once_with(max_retries=0, timeout=60.0)
                self.assertEqual(created[0].api_key, "offline-environment-key")
                self.assertEqual(created[0].max_retries, 0)
                self.assertEqual(created[0].timeout, 60.0)
                self.assertTrue(created[0].is_closed())

    def test_real_sdk_parses_search_plan_and_sends_strict_schema_offline(self):
        captured = []

        def respond(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=sdk_response_data(plan_data()))

        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            with OpenAI(api_key="offline-test-key", http_client=http) as sdk:
                result = OpenAIQueryPlanner(client=sdk).plan(self.request)
                self.assertIsInstance(result, SearchPlan)
                self.assertEqual(result.model_dump(), plan_data())
                self.assertFalse(sdk.is_closed())
                self.assertEqual(sdk.max_retries, 2)
        self.assertEqual(len(captured), 1)
        body = captured[0]
        self.assertIs(body["store"], False)
        self.assertNotIn("tools", body)
        self.assertEqual(json.loads(body["input"]), self.request.model_dump())
        output_format = body["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertIs(output_format["strict"], True)
        schema = output_format["schema"]
        self.assertEqual(set(schema["required"]), {"research_question", "concepts", "queries"})
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["queries"]["minItems"], 3)
        self.assertEqual(schema["properties"]["queries"]["maxItems"], 5)
        planned_query = schema["$defs"]["PlannedQuery"]
        self.assertFalse(planned_query["additionalProperties"])
        self.assertEqual(planned_query["properties"]["role"]["enum"], ["core", "facet", "bridge", "synonym"])

    def test_real_sdk_validation_error_propagates_without_partial_plan(self):
        invalid = deepcopy(plan_data())
        invalid["queries"][0]["role"] = "invalid-role"
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(200, json=sdk_response_data(invalid))

        result = None
        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            with OpenAI(api_key="offline-test-key", http_client=http) as sdk:
                with self.assertRaises(ValidationError):
                    result = OpenAIQueryPlanner(client=sdk).plan(self.request)
                self.assertFalse(sdk.is_closed())
        self.assertIsNone(result)
        self.assertEqual(len(calls), 1)

    def test_retryable_http_error_makes_only_one_offline_request(self):
        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(500, json={"error": {"message": "Offline failure"}})

        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            with OpenAI(api_key="offline-test-key", http_client=http) as sdk:
                with self.assertRaises(APIStatusError):
                    OpenAIQueryPlanner(client=sdk).plan(self.request)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
