"""Offline regression tests for the frozen 17-group RAG human gold pool."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import unittest

from eval.benchmark import load_benchmark
from eval.retrieval.metrics import ndcg_at_k, precision_at_k, strict_precision_at_k


DATASET_DIR = Path(__file__).resolve().parents[1] / "eval" / "datasets"
CANDIDATE_PATH = DATASET_DIR / "rag_hallucination_v1_candidates.json"
GOLD_PATH = DATASET_DIR / "rag_hallucination_v1.json"
# Normalize line endings only, so Git's LF/CRLF conversion does not break the pin.
FROZEN_SNAPSHOT_SHA256 = "47a5230a37f52051df8ad17b0e0e6c6f6afaad758606611f6d6d7daedcd0961a"


class RagHallucinationBenchmarkTests(unittest.TestCase):
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

    def test_gold_contains_exactly_the_17_frozen_group_ids(self) -> None:
        candidate_ids = [item["group_id"] for item in self.candidates]
        self.assertEqual(len(candidate_ids), 17)
        self.assertEqual(len(set(candidate_ids)), 17)
        self.assertEqual(len(self.judgments), 17)
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
            [item["rrf_rank"] for item in self.candidates], list(range(1, 18))
        )
        self.assertEqual(self.ranked, [item["group_id"] for item in self.candidates])
        self.assertEqual(
            [self.judgments[group_id] for group_id in self.ranked],
            [2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 1],
        )

    def test_human_label_counts_are_four_twelve_one(self) -> None:
        self.assertEqual(Counter(self.judgments.values()), {2: 4, 1: 12, 0: 1})

    def test_official_baseline_uses_all_17_judgments(self) -> None:
        before = CANDIDATE_PATH.read_bytes()
        baseline = self.dataset["baseline"]
        self.assertEqual((baseline["method"], baseline["rrf_k"], baseline["k"]), ("group_rrf", 60, 10))
        values = {
            "precision_at_10": precision_at_k(self.ranked, self.judgments, 10),
            "strict_precision_at_10": strict_precision_at_k(self.ranked, self.judgments, 10),
            "ndcg_at_10": ndcg_at_k(self.ranked, self.judgments, 10),
        }
        expected = {
            "precision_at_10": 1.0,
            "strict_precision_at_10": 0.3,
            "ndcg_at_10": 0.8104156585300659,
        }
        for name, value in values.items():
            with self.subTest(metric=name):
                self.assertAlmostEqual(value, expected[name], places=12)
                self.assertAlmostEqual(value, baseline["expected_metrics"][name], places=12)
        # A core group at RRF rank 16 must still contribute to ideal DCG@10.
        self.assertAlmostEqual(
            ndcg_at_k(self.ranked[:10], self.judgments, 10), expected["ndcg_at_10"], places=12
        )
        self.assertEqual(CANDIDATE_PATH.read_bytes(), before)

    def test_frozen_candidate_snapshot_has_not_changed(self) -> None:
        self.assertEqual(
            sha256(self.snapshot_text.encode("utf-8")).hexdigest(),
            FROZEN_SNAPSHOT_SHA256,
        )

    def test_conservative_judgments_explain_the_frozen_evidence_limits(self) -> None:
        for rank in (2, 5):
            with self.subTest(rank=rank):
                candidate = self.candidates[rank - 1]
                judgment = self.dataset["judgments"][candidate["group_id"]]
                self.assertEqual(judgment["relevance"], 1)
                self.assertIn("Proceedings", candidate["abstract"])
                self.assertIn("bibliographic metadata", judgment["reason"])
                self.assertIn("no substantive", judgment["reason"])

        corrective = self.candidates[13]
        judgment = self.dataset["judgments"][corrective["group_id"]]
        self.assertIsNone(corrective["abstract"])
        self.assertEqual(judgment["relevance"], 1)
        self.assertIn("abstract is null", judgment["reason"])
        self.assertIn("without inferring", judgment["reason"])

        off_target = self.candidates[10]
        judgment = self.dataset["judgments"][off_target["group_id"]]
        self.assertEqual(off_target["title"], "Retrieval-Augmented Generation (RAG)")
        self.assertEqual(judgment["relevance"], 0)
        self.assertIn("enterprise", off_target["abstract"])
        self.assertIn("no evidence about hallucination, factuality, grounding, or LLM generation", judgment["reason"])

    def test_question_queries_and_gold_schema_match_the_benchmark(self) -> None:
        definition = load_benchmark("rag_hallucination_v1")
        self.assertEqual(self.dataset["schema_version"], 1)
        self.assertEqual(self.dataset["benchmark_id"], definition.benchmark_id)
        self.assertEqual(self.dataset["research_question"], definition.question)
        self.assertEqual(self.dataset["research_question"], self.snapshot["question"])
        self.assertEqual(
            self.dataset["search_queries"], [query.model_dump() for query in definition.queries]
        )
        self.assertEqual(
            self.dataset["relevance_scale"],
            {"2": "Direct/Core", "1": "Bridge/Supporting", "0": "Off-target for this benchmark"},
        )


if __name__ == "__main__":
    unittest.main()
