"""Offline validation of the planning request and provider-independent contract."""

from copy import deepcopy
import unittest

from pydantic import ValidationError

from researchpilot.query_plan import SearchPlan
from researchpilot.query_planner import PlannerRequest, QueryPlanner


class PlannerRequestTests(unittest.TestCase):
    def test_valid_request_and_defaults(self):
        request = PlannerRequest(research_question="How does caching affect storage latency?")
        self.assertEqual(request.model_dump(), {
            "research_question": "How does caching affect storage latency?",
            "year_from": None, "year_to": None, "max_queries": 5,
        })

    def test_question_trimming_preserves_internal_text(self):
        question = '  How do "shared caches" affect\n  storage latency?\t '
        request = PlannerRequest(research_question=question)
        self.assertEqual(request.research_question, question.strip())

    def test_blank_question_rejected(self):
        for value in ("", " \t\n "):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                PlannerRequest(research_question=value)

    def test_non_string_question_rejected(self):
        for value in (None, 123, True, b"question", []):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                PlannerRequest(research_question=value)

    def test_question_is_required(self):
        with self.assertRaises(ValidationError):
            PlannerRequest()

    def test_none_years_are_valid(self):
        request = PlannerRequest(research_question="Question", year_from=None, year_to=None)
        self.assertIsNone(request.year_from)
        self.assertIsNone(request.year_to)

    def test_single_year_bound_is_valid(self):
        for field in ("year_from", "year_to"):
            with self.subTest(field=field):
                request = PlannerRequest(research_question="Question", **{field: 2024})
                self.assertEqual(getattr(request, field), 2024)

    def test_ordered_and_equal_years_have_no_arbitrary_bounds(self):
        for lower, upper in ((2020, 2025), (2024, 2024), (-10, 0), (10000, 11000)):
            with self.subTest(lower=lower, upper=upper):
                request = PlannerRequest(research_question="Question", year_from=lower, year_to=upper)
                self.assertEqual((request.year_from, request.year_to), (lower, upper))

    def test_reversed_year_bounds_rejected(self):
        with self.assertRaisesRegex(ValidationError, "year_from must be less than or equal to year_to"):
            PlannerRequest(research_question="Question", year_from=2025, year_to=2024)

    def test_non_integer_years_rejected(self):
        for field in ("year_from", "year_to"):
            for value in ("2024", 2024.0, 2024.5, b"2024", []):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    PlannerRequest(research_question="Question", **{field: value})

    def test_bool_years_rejected(self):
        for field in ("year_from", "year_to"):
            for value in (True, False):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    PlannerRequest(research_question="Question", **{field: value})

    def test_max_queries_inclusive_boundaries_and_middle(self):
        for count in (3, 4, 5):
            with self.subTest(count=count):
                self.assertEqual(PlannerRequest(research_question="Question", max_queries=count).max_queries, count)

    def test_max_queries_below_three_rejected(self):
        with self.assertRaises(ValidationError):
            PlannerRequest(research_question="Question", max_queries=2)

    def test_max_queries_above_five_rejected(self):
        with self.assertRaises(ValidationError):
            PlannerRequest(research_question="Question", max_queries=6)

    def test_bool_and_non_integer_max_queries_rejected(self):
        for value in (True, False, "3", 3.0, None):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                PlannerRequest(research_question="Question", max_queries=value)

    def test_extra_fields_rejected(self):
        with self.assertRaisesRegex(ValidationError, "Extra inputs"):
            PlannerRequest(research_question="Question", extra_context="not allowed")

    def test_caller_owned_input_is_not_mutated_on_success_or_failure(self):
        for extra in ({}, {"year_from": 2025, "year_to": 2024}, {"extra_context": ["unchanged"]}):
            data = {"research_question": "  Question  ", **extra}
            before = deepcopy(data)
            if extra:
                with self.assertRaises(ValidationError):
                    PlannerRequest.model_validate(data)
            else:
                PlannerRequest.model_validate(data)
            self.assertEqual(data, before)


class QueryPlannerProtocolTests(unittest.TestCase):
    def test_fake_planner_can_implement_contract_without_openai(self):
        result = SearchPlan(
            research_question="Question", concepts=["caching"],
            queries=[
                {"query_id": f"q{i}", "text": f"cache fixture {i}",
                 "role": "core" if i == 1 else "facet", "rationale": "Fixture scope"}
                for i in range(1, 4)
            ],
        )

        class FakeQueryPlanner:
            def plan(self, request: PlannerRequest) -> SearchPlan:
                return result

        planner: QueryPlanner = FakeQueryPlanner()
        self.assertIs(planner.plan(PlannerRequest(research_question="Question")), result)


if __name__ == "__main__":
    unittest.main()
