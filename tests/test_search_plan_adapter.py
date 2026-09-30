"""Pure adapter tests; planning and search providers are not involved."""

import unittest

from researchpilot.paper_candidate import SearchQuery
from researchpilot.query_plan import SearchPlan
from researchpilot.search_plan_adapter import search_plan_to_queries


class SearchPlanAdapterTests(unittest.TestCase):
    def setUp(self):
        self.plan = SearchPlan(
            research_question="How do shared caches affect latency?",
            concepts=["shared caches", "latency"],
            queries=[
                {"query_id": "Q-2", "text": '"shared cache"  AND Latency',
                 "role": "core", "rationale": "Study the main problem."},
                {"query_id": "q_a", "text": '"cache contention" OR "buffer sharing"',
                 "role": "synonym", "rationale": "Search alternative terms."},
                {"query_id": "bridge:α", "text": '"request scheduling" AND performance',
                 "role": "bridge", "rationale": "Find useful scheduling methods."},
            ],
        )

    def test_converts_all_queries_preserving_exact_order_ids_and_text(self):
        queries = search_plan_to_queries(self.plan)
        self.assertEqual(len(queries), 3)
        self.assertTrue(all(isinstance(query, SearchQuery) for query in queries))
        self.assertEqual([query.query_id for query in queries], ["Q-2", "q_a", "bridge:α"])
        self.assertEqual([query.text for query in queries], [
            '"shared cache"  AND Latency',
            '"cache contention" OR "buffer sharing"',
            '"request scheduling" AND performance',
        ])

    def test_role_and_rationale_are_not_transferred(self):
        for query in search_plan_to_queries(self.plan):
            self.assertEqual(set(query.model_dump()), {"query_id", "text"})

    def test_source_plan_is_not_mutated_and_outputs_are_independent(self):
        before = self.plan.model_dump(mode="json")
        queries = search_plan_to_queries(self.plan)
        self.assertEqual(self.plan.model_dump(mode="json"), before)
        queries[0].text = "changed output"
        queries.reverse()
        self.assertEqual(self.plan.model_dump(mode="json"), before)


if __name__ == "__main__":
    unittest.main()
