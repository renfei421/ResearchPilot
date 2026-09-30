"""Pure metric tests and the fully judged 22-group matrix-completion baseline."""

from collections import Counter
import json
from math import log2
from pathlib import Path
import unittest

from eval.retrieval.metrics import ndcg_at_k, precision_at_k, strict_precision_at_k


METRICS = (precision_at_k, strict_precision_at_k, ndcg_at_k)
DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "eval"
    / "datasets"
    / "matrix_completion_v1.json"
)
CANDIDATE_PATH = DATASET_PATH.with_name("matrix_completion_candidates_v1.json")


class RetrievalMetricTests(unittest.TestCase):
    def test_precision_counts_core_and_supporting_items(self) -> None:
        self.assertEqual(
            precision_at_k(["a", "b", "c", "d"], {"a": 2, "b": 0, "c": 1, "d": 2}, 4),
            0.75,
        )

    def test_strict_precision_counts_only_core_items(self) -> None:
        self.assertEqual(
            strict_precision_at_k(
                ["a", "b", "c", "d"], {"a": 2, "b": 0, "c": 1, "d": 2}, 4
            ),
            0.5,
        )

    def test_perfect_ndcg_is_one(self) -> None:
        judgments = {"a": 2, "b": 0, "c": 1, "d": 2}

        self.assertEqual(ndcg_at_k(["a", "d", "c", "b"], judgments, 4), 1.0)

    def test_degraded_ordering_has_lower_ndcg(self) -> None:
        judgments = {"a": 2, "b": 0, "c": 1, "d": 2}

        score = ndcg_at_k(["b", "c", "d", "a"], judgments, 4)

        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_cutoff_smaller_than_ranking_uses_only_prefix(self) -> None:
        ranked = ["b", "a", "c", "d"]
        judgments = {"a": 2, "b": 0, "c": 1, "d": 2}

        self.assertEqual(precision_at_k(ranked, judgments, 2), 0.5)
        self.assertEqual(strict_precision_at_k(ranked, judgments, 2), 0.5)
        # The ideal prefix has two core items; the actual first item has gain 0.
        expected = (3 / log2(3)) / (3 + 3 / log2(3))
        self.assertAlmostEqual(ndcg_at_k(ranked, judgments, 2), expected, places=12)

    def test_large_cutoff_counts_missing_slots_for_precision(self) -> None:
        ranked = ["a", "b"]
        judgments = {"a": 2, "b": 1}

        self.assertEqual(precision_at_k(ranked, judgments, 5), 0.4)
        self.assertEqual(strict_precision_at_k(ranked, judgments, 5), 0.2)
        self.assertEqual(ndcg_at_k(ranked, judgments, 5), 1.0)

    def test_ndcg_ideal_includes_judged_items_missing_from_results(self) -> None:
        judgments = {"core": 2, "supporting": 1}

        self.assertAlmostEqual(
            ndcg_at_k(["supporting"], judgments, 1), 1 / 3, places=12
        )

    def test_empty_rankings_return_float_zero(self) -> None:
        for metric in METRICS:
            for judgments in ({}, {"a": 2}):
                with self.subTest(metric=metric.__name__, judgments=judgments):
                    score = metric([], judgments, 10)
                    self.assertEqual(score, 0.0)
                    self.assertIsInstance(score, float)

    def test_no_relevant_documents_return_zero(self) -> None:
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                self.assertEqual(metric(["a", "b"], {"a": 0, "b": 0}, 2), 0.0)

    def test_invalid_k_is_rejected(self) -> None:
        for metric in METRICS:
            for k in (0, -1, True, False, None, "10", 10.0):
                with self.subTest(metric=metric.__name__, k=k):
                    with self.assertRaisesRegex(
                        ValueError, "k must be a positive integer"
                    ):
                        metric([], {}, k)

    def test_unjudged_items_are_not_silently_treated_as_irrelevant(self) -> None:
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                with self.assertRaisesRegex(ValueError, "Missing relevance judgment"):
                    metric(["unknown"], {"known": 2}, 1)

    def test_duplicate_top_k_items_are_rejected(self) -> None:
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                with self.assertRaisesRegex(ValueError, "must be unique"):
                    metric(["a", "a"], {"a": 2}, 2)

    def test_items_beyond_cutoff_do_not_affect_evaluation(self) -> None:
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                self.assertEqual(metric(["a", "unknown", "a"], {"a": 2}, 1), 1.0)

    def test_invalid_relevance_grades_are_rejected(self) -> None:
        for metric in METRICS:
            for relevance in (-1, 3, True, None, "2", 2.0):
                with self.subTest(metric=metric.__name__, relevance=relevance):
                    with self.assertRaisesRegex(ValueError, "integers 0, 1, or 2"):
                        metric(["a"], {"a": relevance}, 1)

    def test_metrics_do_not_mutate_inputs(self) -> None:
        ranked = ["b", "a"]
        judgments = {"a": 2, "b": 1, "c": 0}

        for metric in METRICS:
            metric(ranked, judgments, 2)

        self.assertEqual(ranked, ["b", "a"])
        self.assertEqual(judgments, {"a": 2, "b": 1, "c": 0})


class FrozenBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        self.candidates = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))[
            "candidates"
        ]
        self.ranked = self.dataset["baseline"]["ranked_item_ids"]
        self.judgments = {
            item_id: item["relevance"]
            for item_id, item in self.dataset["judgments"].items()
        }

    def test_benchmark_question_and_handcrafted_queries_are_frozen(self) -> None:
        self.assertEqual(self.dataset["benchmark_id"], "matrix_completion_v1")
        self.assertEqual(
            self.dataset["research_question"],
            "How does spectral expansion affect deterministic matrix completion "
            "under fixed observation patterns?",
        )
        self.assertEqual(
            self.dataset["search_queries"],
            [
                {"query_id": "q1", "text": '"matrix completion" AND deterministic'},
                {
                    "query_id": "q2",
                    "text": '"matrix completion" AND "sampling pattern"',
                },
                {"query_id": "q3", "text": '"matrix completion" AND "spectral gap"'},
            ],
        )

    def test_judgments_use_identifiers_and_have_explanations(self) -> None:
        self.assertEqual(len(self.ranked), 22)
        self.assertEqual(len(set(self.ranked)), 22)
        self.assertEqual(set(self.ranked), set(self.judgments))
        for item_id, item in self.dataset["judgments"].items():
            with self.subTest(item_id=item_id):
                self.assertEqual(item_id, item["group_id"])
                self.assertTrue(item["title"].strip())
                self.assertTrue(item["reason"].strip())
                self.assertIn(item["relevance"], (0, 1, 2))
        self.assertEqual(
            [self.judgments[item_id] for item_id in self.ranked],
            [2, 0, 2, 2, 0, 1, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        )

    def test_complete_pool_preserves_candidate_snapshot_ids_and_rrf_order(self) -> None:
        self.assertEqual(
            [item["rrf_rank"] for item in self.candidates], list(range(1, 23))
        )
        self.assertEqual(
            self.ranked, [item["group_id"] for item in self.candidates]
        )
        for candidate in self.candidates:
            with self.subTest(group_id=candidate["group_id"]):
                judgment = self.dataset["judgments"][candidate["group_id"]]
                self.assertEqual(judgment["title"], candidate["title"])
                self.assertIn(candidate["paper_id"], judgment["known_paper_ids"])

    def test_complete_gold_judgment_counts(self) -> None:
        self.assertEqual(Counter(self.judgments.values()), {2: 7, 1: 5, 0: 10})

    def test_known_canonical_paper_ids_are_preserved(self) -> None:
        item_id = self.candidates[0]["group_id"]
        item = self.dataset["judgments"][item_id]

        self.assertEqual(
            item["known_paper_ids"],
            ["doi:10.1109/jstsp.2016.2537145", "doi:10.1109/allerton.2015.7447128"],
        )
        self.assertEqual(item["group_id"], item_id)

    def test_baseline_reproduces_expected_metrics(self) -> None:
        cutoff = self.dataset["baseline"]["k"]
        expected = self.dataset["baseline"]["expected_metrics"]
        values = {
            "precision_at_10": precision_at_k(self.ranked, self.judgments, cutoff),
            "strict_precision_at_10": strict_precision_at_k(
                self.ranked, self.judgments, cutoff
            ),
            "ndcg_at_10": ndcg_at_k(self.ranked, self.judgments, cutoff),
        }

        self.assertAlmostEqual(values["precision_at_10"], 0.60, places=12)
        self.assertAlmostEqual(values["strict_precision_at_10"], 0.50, places=12)
        self.assertAlmostEqual(values["ndcg_at_10"], 0.6781498023908038, places=12)
        for name, value in values.items():
            with self.subTest(metric=name):
                self.assertAlmostEqual(value, expected[name], places=12)

    def test_top_ten_ndcg_uses_the_complete_gold_pool_for_ideal_dcg(self) -> None:
        # The ideal Top 10 includes seven core and three supporting groups,
        # including relevant candidates that the RRF Top 10 did not retrieve.
        top_ten = self.ranked[:10]
        dcg = sum(
            (2 ** self.judgments[item_id] - 1) / log2(rank + 1)
            for rank, item_id in enumerate(top_ten, start=1)
        )
        ideal_dcg = sum(
            (2**grade - 1) / log2(rank + 1)
            for rank, grade in enumerate([2] * 7 + [1] * 3, start=1)
        )

        score = ndcg_at_k(top_ten, self.judgments, 10)

        self.assertAlmostEqual(score, dcg / ideal_dcg, places=12)
        partial_judgments = {item_id: self.judgments[item_id] for item_id in top_ten}
        self.assertLess(score, ndcg_at_k(top_ten, partial_judgments, 10))

    def test_judgment_storage_order_does_not_change_scores(self) -> None:
        reversed_judgments = dict(reversed(list(self.judgments.items())))

        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                self.assertEqual(
                    metric(self.ranked, self.judgments, 10),
                    metric(self.ranked, reversed_judgments, 10),
                )

    def test_reordering_items_keeps_labels_attached_to_identifiers(self) -> None:
        ideal_order = sorted(self.ranked, key=self.judgments.__getitem__, reverse=True)

        self.assertEqual(ndcg_at_k(ideal_order, self.judgments, 10), 1.0)
        self.assertEqual(precision_at_k(ideal_order, self.judgments, 10), 1.0)
        self.assertEqual(strict_precision_at_k(ideal_order, self.judgments, 10), 0.7)
