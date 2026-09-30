"""Offline validation tests for planning-layer output contracts."""

from copy import deepcopy
import unittest

from pydantic import ValidationError

from researchpilot.query_plan import PlannedQuery, SearchPlan


def planned_query_data() -> dict:
    return {
        "query_id": "q1",
        "text": '"matrix completion" AND deterministic',
        "role": "core",
        "rationale": "Targets the central completion problem.",
    }


def search_plan_data(count: int = 3) -> dict:
    queries = [
        planned_query_data(),
        {"query_id": "q2", "text": '"matrix completion" AND "sampling pattern"',
         "role": "facet", "rationale": "Targets the observation constraint."},
        {"query_id": "q3", "text": '"matrix completion" AND "spectral gap"',
         "role": "bridge", "rationale": "Connects completion to graph theory."},
        {"query_id": "q4", "text": '"low-rank recovery" AND deterministic',
         "role": "synonym", "rationale": "Uses alternative terminology."},
        {"query_id": "q5", "text": '"fixed observation patterns"',
         "role": "facet", "rationale": "Targets fixed observations."},
    ]
    return {
        "research_question": "How does spectral expansion affect deterministic matrix completion?",
        "concepts": ["matrix completion", "spectral expansion", "fixed observation patterns"],
        "queries": queries[:count],
    }


class PlannedQueryTests(unittest.TestCase):
    def test_valid_planned_query(self):
        data = planned_query_data()
        self.assertEqual(PlannedQuery.model_validate(data).model_dump(), data)

    def test_trims_strings_but_preserves_query_case_punctuation_and_internal_whitespace(self):
        data = planned_query_data() | {
            "query_id": " \tq1 ",
            "text": '  ("MATRIX   Completion" AND deterministic) OR low-rank\trecovery  ',
            "rationale": " \nTargets the core problem.\t ",
        }
        query = PlannedQuery.model_validate(data)
        self.assertEqual(query.query_id, "q1")
        self.assertEqual(query.text, data["text"].strip())
        self.assertEqual(query.rationale, "Targets the core problem.")

    def test_blank_query_id_rejected(self):
        for value in ("", " \t\n "):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                PlannedQuery.model_validate(planned_query_data() | {"query_id": value})

    def test_non_string_query_id_rejected(self):
        for value in (None, 123, True, b"q1", []):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                PlannedQuery.model_validate(planned_query_data() | {"query_id": value})

    def test_blank_text_rejected(self):
        for value in ("", " \t "):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                PlannedQuery.model_validate(planned_query_data() | {"text": value})

    def test_blank_rationale_rejected(self):
        for value in ("", " \t\n "):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                PlannedQuery.model_validate(planned_query_data() | {"rationale": value})

    def test_text_and_rationale_require_strict_strings(self):
        for field in ("text", "rationale"):
            for value in (None, 123, True, b"text", []):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    PlannedQuery.model_validate(planned_query_data() | {field: value})

    def test_only_four_exact_roles_accepted(self):
        for role in ("core", "facet", "bridge", "synonym"):
            with self.subTest(role=role):
                self.assertEqual(PlannedQuery(**(planned_query_data() | {"role": role})).role, role)
        for role in ("other", "CORE", " core ", "", None, True, b"core"):
            with self.subTest(role=role), self.assertRaises(ValidationError):
                PlannedQuery.model_validate(planned_query_data() | {"role": role})

    def test_extra_fields_rejected(self):
        with self.assertRaisesRegex(ValidationError, "Extra inputs"):
            PlannedQuery.model_validate(planned_query_data() | {"weight": 1})

    def test_all_fields_required(self):
        for field in planned_query_data():
            data = planned_query_data()
            del data[field]
            with self.subTest(field=field), self.assertRaises(ValidationError):
                PlannedQuery.model_validate(data)

    def test_multiline_text_rejected_including_edge_and_unicode_line_breaks(self):
        for separator in ("\n", "\r", "\r\n", "\v", "\f", "\x85", "\u2028", "\u2029"):
            for text in (f"matrix{separator}completion", f"{separator}matrix", f"matrix{separator}"):
                with self.subTest(text=text), self.assertRaisesRegex(ValidationError, "single-line"):
                    PlannedQuery.model_validate(planned_query_data() | {"text": text})

    def test_http_and_https_urls_rejected_anywhere_in_query(self):
        for text in ("http://example.org/paper", "HTTPS://example.org/paper",
                     'reasoning AND "https://example.org/paper"', "https://api.openalex.org/works"):
            with self.subTest(text=text), self.assertRaisesRegex(ValidationError, "HTTP/HTTPS URL"):
                PlannedQuery.model_validate(planned_query_data() | {"text": text})

    def test_api_parameter_assignments_rejected(self):
        for text in ("api_key=fake", "per_page=10", "filter=publication_year:2024",
                     "matrix completion&per_page=10", "API_KEY = fake", "PER_PAGE\t= 10",
                     "FILTER = publication_year:2024"):
            with self.subTest(text=text), self.assertRaisesRegex(ValidationError, "API parameters"):
                PlannedQuery.model_validate(planned_query_data() | {"text": text})

    def test_years_and_parameter_words_without_assignments_are_allowed(self):
        for text in ('"reasoning accuracy" AND 2024', '"filter" AND "api_key" AND per_page',
                     '("A-B" OR "A/B") AND C++'):
            with self.subTest(text=text):
                self.assertEqual(PlannedQuery(**(planned_query_data() | {"text": text})).text, text)


class SearchPlanTests(unittest.TestCase):
    def test_valid_three_query_plan(self):
        data = search_plan_data()
        plan = SearchPlan.model_validate(data)
        self.assertEqual(plan.model_dump(), data)
        self.assertTrue(all(isinstance(query, PlannedQuery) for query in plan.queries))

    def test_valid_five_query_plan(self):
        data = search_plan_data(5)
        self.assertEqual(SearchPlan.model_validate(data).model_dump(), data)

    def test_fewer_than_three_queries_rejected(self):
        for count in (0, 1, 2):
            with self.subTest(count=count), self.assertRaises(ValidationError):
                SearchPlan.model_validate(search_plan_data(count))

    def test_more_than_five_queries_rejected(self):
        data = search_plan_data(5)
        data["queries"].append(planned_query_data() | {"query_id": "q6", "text": "extra query"})
        with self.assertRaises(ValidationError):
            SearchPlan.model_validate(data)

    def test_blank_research_question_rejected(self):
        for value in ("", " \t\n "):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                SearchPlan.model_validate(search_plan_data() | {"research_question": value})

    def test_research_question_is_trimmed_and_strict(self):
        data = search_plan_data() | {"research_question": "  What affects reasoning? \n"}
        self.assertEqual(SearchPlan.model_validate(data).research_question, "What affects reasoning?")
        for value in (None, 42, True, b"question", []):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                SearchPlan.model_validate(data | {"research_question": value})

    def test_empty_concepts_rejected(self):
        with self.assertRaises(ValidationError):
            SearchPlan.model_validate(search_plan_data() | {"concepts": []})

    def test_blank_concept_rejected(self):
        for value in ("", " \t\n "):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                SearchPlan.model_validate(search_plan_data() | {"concepts": ["valid", value]})

    def test_concepts_require_strict_strings(self):
        for value in (None, 123, True, b"concept", []):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                SearchPlan.model_validate(search_plan_data() | {"concepts": [value]})

    def test_duplicate_concepts_rejected_after_casefold_and_whitespace_normalization(self):
        for concepts in (["matrix completion", "  MATRIX   COMPLETION\t"], ["Straße", "STRASSE"]):
            with self.subTest(concepts=concepts), self.assertRaisesRegex(ValidationError, "concepts must be unique"):
                SearchPlan.model_validate(search_plan_data() | {"concepts": concepts})

    def test_duplicate_query_id_rejected_after_trimming(self):
        data = search_plan_data()
        data["queries"][1]["query_id"] = " q1 "
        with self.assertRaisesRegex(ValidationError, "query_id values must be unique"):
            SearchPlan.model_validate(data)

    def test_query_id_uniqueness_is_case_sensitive(self):
        data = search_plan_data()
        data["queries"][1]["query_id"] = "Q1"
        self.assertEqual([query.query_id for query in SearchPlan.model_validate(data).queries], ["q1", "Q1", "q3"])

    def test_duplicate_query_text_rejected_after_casefold_and_whitespace_normalization(self):
        data = search_plan_data()
        data["queries"][1]["text"] = '  "MATRIX   COMPLETION" AND DETERMINISTIC  '
        with self.assertRaisesRegex(ValidationError, "query texts must be unique"):
            SearchPlan.model_validate(data)
        data["queries"][0]["text"] = "Straße"
        data["queries"][1]["text"] = "STRASSE"
        with self.assertRaisesRegex(ValidationError, "query texts must be unique"):
            SearchPlan.model_validate(data)

    def test_duplicate_detection_preserves_punctuation_boolean_operators_and_word_forms(self):
        data = search_plan_data(5)
        texts = ['"A-B" AND C', '"A B" AND C', '"A-B" OR C', '"A-B" AND Cs', '“A-B” AND C']
        for query, text in zip(data["queries"], texts):
            query["text"] = text
        data["concepts"] = ["A-B", "A B", "Graph", "Graphs"]
        plan = SearchPlan.model_validate(data)
        self.assertEqual([query.text for query in plan.queries], texts)
        self.assertEqual(plan.concepts, data["concepts"])

    def test_query_order_preserved(self):
        data = search_plan_data()
        data["queries"] = [data["queries"][2], data["queries"][0], data["queries"][1]]
        self.assertEqual([query.query_id for query in SearchPlan.model_validate(data).queries], ["q3", "q1", "q2"])

    def test_concepts_trimmed_without_reordering_or_changing_display_text(self):
        data = search_plan_data() | {"concepts": ["  Spectral   Gap  ", "Matrix Completion\t", "fixed patterns"]}
        self.assertEqual(SearchPlan.model_validate(data).concepts, ["Spectral   Gap", "Matrix Completion", "fixed patterns"])

    def test_plan_without_core_query_rejected(self):
        data = search_plan_data()
        data["queries"][0]["role"] = "synonym"
        with self.assertRaisesRegex(ValidationError, "at least one.*core"):
            SearchPlan.model_validate(data)

    def test_valid_mix_of_all_four_roles(self):
        plan = SearchPlan.model_validate(search_plan_data(4))
        self.assertEqual([query.role for query in plan.queries], ["core", "facet", "bridge", "synonym"])

    def test_extra_plan_fields_rejected(self):
        with self.assertRaisesRegex(ValidationError, "Extra inputs"):
            SearchPlan.model_validate(search_plan_data() | {"max_queries": 5})

    def test_all_plan_fields_required(self):
        for field in search_plan_data():
            data = search_plan_data()
            del data[field]
            with self.subTest(field=field), self.assertRaises(ValidationError):
                SearchPlan.model_validate(data)

    def test_caller_owned_lists_and_dicts_not_mutated_on_success(self):
        data = search_plan_data() | {"concepts": [" Spectral   Gap ", " Matrix Completion "]}
        data["queries"][0]["text"] = '  "Matrix   completion" AND deterministic  '
        data["queries"][0]["query_id"] = " q1 "
        before = deepcopy(data)
        plan = SearchPlan.model_validate(data)
        self.assertEqual(data, before)
        self.assertIsNot(plan.concepts, data["concepts"])
        self.assertIsNot(plan.queries, data["queries"])

    def test_caller_owned_containers_not_mutated_when_validation_fails(self):
        for field in ("concepts", "query_id", "text"):
            data = search_plan_data()
            if field == "concepts":
                data["concepts"] = [" Matrix completion ", "MATRIX  COMPLETION"]
            else:
                data["queries"][1][field] = "  " + data["queries"][0][field] + "  "
            before = deepcopy(data)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                SearchPlan.model_validate(data)
            self.assertEqual(data, before)

    def test_prebuilt_planned_queries_are_accepted_without_mutation(self):
        data = search_plan_data()
        queries = [PlannedQuery.model_validate(query) for query in data["queries"]]
        before = [query.model_dump() for query in queries]
        plan = SearchPlan.model_validate(data | {"queries": queries})
        self.assertEqual([query.model_dump() for query in queries], before)
        self.assertEqual([query.model_dump() for query in plan.queries], before)


if __name__ == "__main__":
    unittest.main()
