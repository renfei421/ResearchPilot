"""Offline regression tests for the frozen 22-group CoT human gold pool."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import unittest

from eval.benchmark import load_benchmark
from eval.retrieval.metrics import ndcg_at_k, precision_at_k, strict_precision_at_k


DATASET_DIR = Path(__file__).resolve().parents[1] / "eval" / "datasets"
CANDIDATE_PATH = DATASET_DIR / "cot_reasoning_faithfulness_v1_candidates.json"
GOLD_PATH = DATASET_DIR / "cot_reasoning_faithfulness_v1.json"
# Normalize line endings only, so Git's LF/CRLF conversion does not break the pin.
FROZEN_SNAPSHOT_SHA256 = "d0b11f4de81a4234a306b87ca51f6a0625e86a6e6eca6f151961f1731ad6f8d2"


class CotReasoningFaithfulnessBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.snapshot_text = CANDIDATE_PATH.read_text(encoding="utf-8")
        self.snapshot = json.loads(self.snapshot_text)
        self.candidates = self.snapshot["candidates"]
        self.dataset = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
        self.ranked = self.dataset["baseline"]["ranked_item_ids"]
        self.judgments = {
            group_id: judgment["relevance"]
            for group_id, judgment in self.dataset["judgments"].items()
        }

    def test_gold_contains_exactly_the_22_frozen_group_ids(self) -> None:
        candidate_ids = [item["group_id"] for item in self.candidates]
        self.assertEqual(len(candidate_ids), 22)
        self.assertEqual(len(set(candidate_ids)), 22)
        self.assertEqual(len(self.judgments), 22)
        self.assertEqual(set(self.judgments), set(candidate_ids))
        for candidate in self.candidates:
            with self.subTest(group_id=candidate["group_id"]):
                judgment = self.dataset["judgments"][candidate["group_id"]]
                self.assertEqual(judgment["group_id"], candidate["group_id"])
                self.assertEqual(judgment["title"], candidate["title"])
                self.assertEqual(judgment["known_paper_ids"], [candidate["paper_id"]])
                self.assertTrue(judgment["reason"].strip())

    def test_rrf_order_and_all_human_labels_are_preserved(self) -> None:
        self.assertEqual(
            [item["rrf_rank"] for item in self.candidates], list(range(1, 23))
        )
        self.assertEqual(self.ranked, [item["group_id"] for item in self.candidates])
        self.assertEqual(
            [self.judgments[group_id] for group_id in self.ranked],
            [2, 1, 1, 1, 0, 1, 2, 1, 2, 0, 2, 1, 0, 1, 1, 1, 2, 2, 1, 1, 0, 0],
        )

    def test_human_label_counts_are_six_eleven_five(self) -> None:
        self.assertEqual(Counter(self.judgments.values()), {2: 6, 1: 11, 0: 5})

    def test_official_baseline_uses_all_22_judgments(self) -> None:
        before = CANDIDATE_PATH.read_bytes()
        baseline = self.dataset["baseline"]
        self.assertEqual(
            (baseline["method"], baseline["rrf_k"], baseline["k"]),
            ("group_rrf", 60, 10),
        )
        values = {
            "precision_at_10": precision_at_k(self.ranked, self.judgments, 10),
            "strict_precision_at_10": strict_precision_at_k(self.ranked, self.judgments, 10),
            "ndcg_at_10": ndcg_at_k(self.ranked, self.judgments, 10),
        }
        expected = {
            "precision_at_10": 0.8,
            "strict_precision_at_10": 0.3,
            "ndcg_at_10": 0.6398670761344659,
        }
        for name, value in values.items():
            with self.subTest(metric=name):
                self.assertAlmostEqual(value, expected[name], places=12)
                self.assertAlmostEqual(value, baseline["expected_metrics"][name], places=12)
        # Core groups at ranks 11, 17 and 18 contribute to the full-pool IDCG.
        self.assertAlmostEqual(
            ndcg_at_k(self.ranked[:10], self.judgments, 10),
            expected["ndcg_at_10"], places=12,
        )
        partial_gold = {group_id: self.judgments[group_id] for group_id in self.ranked[:10]}
        self.assertLess(
            values["ndcg_at_10"], ndcg_at_k(self.ranked[:10], partial_gold, 10)
        )
        self.assertEqual(CANDIDATE_PATH.read_bytes(), before)

    def test_frozen_candidate_snapshot_has_not_changed(self) -> None:
        self.assertEqual(
            sha256(self.snapshot_text.encode("utf-8")).hexdigest(),
            FROZEN_SNAPSHOT_SHA256,
        )

    def test_conservative_judgments_explain_the_frozen_evidence_limits(self) -> None:
        for rank in (2, 4, 12, 16):
            with self.subTest(rank=rank):
                candidate = self.candidates[rank - 1]
                judgment = self.dataset["judgments"][candidate["group_id"]]
                self.assertEqual(judgment["relevance"], 1)
                self.assertIn("Association for Computational Linguistics", candidate["abstract"])
                self.assertIn("bibliographic information", judgment["reason"])
                self.assertIn("no substantive", judgment["reason"])
                self.assertIn("without inferring", judgment["reason"])

        candidate = self.candidates[5]
        judgment = self.dataset["judgments"][candidate["group_id"]]
        self.assertEqual(
            candidate["title"], "Chain-Of-Thought Prompting Elicits Reasoning in Large Language Models"
        )
        self.assertIsNone(candidate["abstract"])
        self.assertEqual(judgment["relevance"], 1)
        self.assertIn("abstract is null", judgment["reason"])
        self.assertIn("without using outside knowledge", judgment["reason"])

    def test_question_queries_and_gold_schema_match_the_benchmark(self) -> None:
        definition = load_benchmark("cot_reasoning_faithfulness_v1")
        self.assertEqual(self.dataset["schema_version"], 1)
        self.assertEqual(self.dataset["benchmark_id"], definition.benchmark_id)
        self.assertEqual(self.snapshot["benchmark_id"], definition.benchmark_id)
        self.assertEqual(self.dataset["research_question"], definition.question)
        self.assertEqual(self.dataset["research_question"], self.snapshot["question"])
        self.assertEqual(
            self.dataset["search_queries"], [query.model_dump() for query in definition.queries]
        )
        self.assertEqual(
            self.dataset["relevance_scale"],
            {"2": "Direct/Core", "1": "Bridge/Supporting", "0": "Off-target for this benchmark"},
        )
        for existing in ("matrix_completion_v1", "rag_hallucination_v1"):
            with self.subTest(existing=existing):
                other = json.loads((DATASET_DIR / f"{existing}.json").read_text(encoding="utf-8"))
                self.assertEqual(set(self.dataset), set(other))
                self.assertEqual(set(self.dataset["baseline"]), set(other["baseline"]))


if __name__ == "__main__":
    unittest.main()
