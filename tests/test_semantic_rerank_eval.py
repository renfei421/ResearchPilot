"""Offline tests for snapshot evaluation, gold isolation, and run artifacts."""

from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from eval.benchmark import PROJECT_ROOT, benchmark_paths
from researchpilot.relevance import RelevanceAssessment, RelevanceClient
from scripts import run_semantic_rerank_eval as runner


class SemanticRerankEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Normal mkdir preserves inherited Windows ACLs in restricted workspaces.
        root = Path(__file__).resolve().parent / f"semantic-eval-{uuid4().hex}"
        root.mkdir()
        self.root = root
        self.candidate_path = root / "candidates.json"
        self.gold_path = root / "gold.json"
        self.output_path = root / "runs" / "completed.json"
        self.benchmark_id = "test_retrieval_v1"
        self.definition_path = benchmark_paths(self.benchmark_id, root=root).definition
        self.addCleanup(self._cleanup)
        self.question = "  How do fixed patterns affect completion?\n"
        self.definition_path.parent.mkdir(parents=True)
        self.definition_path.write_text(
            json.dumps({
                "benchmark_id": self.benchmark_id,
                "question": self.question,
                "year_from": 2015,
                "year_to": 2026,
                "per_query": 10,
                "queries": [{"query_id": "q1", "text": "fixed patterns"}],
            }),
            encoding="utf-8",
        )
        self.snapshot = {
            "benchmark_id": self.benchmark_id,
            "question": self.question,
            "candidates": [
                {
                    "group_id": f"group-{rank}",
                    "rrf_rank": rank,
                    "rrf_score": 1 / (60 + rank),
                    # Same titles ensure that judgments are joined by group_id.
                    "title": "  Shared paper title\n",
                    "abstract": f"  Abstract {rank}\n" if rank != 3 else None,
                }
                for rank in range(1, 23)
            ],
        }
        self.rrf_ids = [item["group_id"] for item in self.snapshot["candidates"]]
        self.gold = {
            "benchmark_id": self.benchmark_id,
            "research_question": self.question,
            "baseline": {"ranked_item_ids": self.rrf_ids},
            "judgments": {
                group_id: {
                    "relevance": {"group-1": 2, "group-3": 1}.get(group_id, 0),
                    "reason": "GOLD ONLY: never disclose this judgment to the client.",
                }
                for group_id in reversed(self.rrf_ids)
            },
        }
        self._write_inputs()
        self.client = Mock(spec=RelevanceClient)
        # Fixed fake outputs are independent of gold and have deliberate ties.
        self.client.assess.side_effect = [
            RelevanceAssessment(category="supporting", score=score, reason="Fake evidence")
            for score in [0.1, 0.9, 0.9] + [0.2] * 19
        ]
        # Tests must explicitly inject a fake and must never construct an API client.
        api_constructor = patch.object(
            runner, "OpenAIRelevanceClient", side_effect=AssertionError("No real API")
        )
        api_constructor.start()
        self.addCleanup(api_constructor.stop)

    def _write_inputs(self) -> None:
        self.candidate_path.write_text(json.dumps(self.snapshot), encoding="utf-8")
        self.gold_path.write_text(json.dumps(self.gold), encoding="utf-8")

    def _cleanup(self) -> None:
        for path in (
            self.output_path, self.candidate_path, self.gold_path, self.definition_path
        ):
            path.unlink(missing_ok=True)
        if self.output_path.parent.exists():
            self.output_path.parent.rmdir()
        self.definition_path.parent.rmdir()
        self.definition_path.parent.parent.rmdir()
        self.root.rmdir()

    def _run(self, **overrides):
        arguments = {
            "benchmark_id": self.benchmark_id,
            "root": self.root,
            "client": self.client,
            "model": "offline-test-model",
            "candidate_path": self.candidate_path,
            "gold_path": self.gold_path,
            "output_path": self.output_path,
        }
        arguments.update(overrides)
        return runner.run_evaluation(**arguments)

    def test_complete_run_is_gold_blind_and_saves_ranking_with_group_id_mapping(self):
        # File storage order must not replace the explicit original RRF ranks.
        self.snapshot["candidates"].reverse()
        self._write_inputs()
        before = [path.read_bytes() for path in (self.candidate_path, self.gold_path)]
        original_read = Path.read_text
        gold_reads = []

        def observed_read(path, *args, **kwargs):
            if path == self.gold_path:
                self.assertEqual(self.client.assess.call_count, 22)
                gold_reads.append(path)
            return original_read(path, *args, **kwargs)

        output = io.StringIO()
        with patch.object(Path, "read_text", autospec=True, side_effect=observed_read):
            with redirect_stdout(output):
                result = self._run()

        self.assertEqual(gold_reads, [self.gold_path])
        self.assertEqual(self.client.assess.call_count, 22)
        for rank, call in enumerate(self.client.assess.call_args_list, start=1):
            self.assertEqual(call.args, ())
            self.assertEqual(
                call.kwargs,
                {
                    "research_question": self.question,
                    "title": "  Shared paper title\n",
                    "abstract": f"  Abstract {rank}\n" if rank != 3 else None,
                },
            )
        self.assertEqual(
            result["semantic_ranking"], self.rrf_ids[1:] + self.rrf_ids[:1]
        )
        self.assertEqual(
            [item["group_id"] for item in result["assessments"]], self.rrf_ids
        )
        self.assertEqual(
            [item["original_rrf_rank"] for item in result["assessments"]],
            list(range(1, 23)),
        )
        self.assertEqual(
            [item["original_rrf_score"] for item in result["assessments"]],
            [1 / (60 + rank) for rank in range(1, 23)],
        )
        self.assertEqual(result["semantic_metrics"]["precision_at_10"], 0.1)
        self.assertEqual(result["semantic_metrics"]["strict_precision_at_10"], 0.0)
        self.assertEqual(result["baseline_metrics"]["precision_at_10"], 0.2)
        self.assertEqual(result["baseline_metrics"]["strict_precision_at_10"], 0.1)
        self.assertEqual(result["benchmark_id"], self.benchmark_id)
        self.assertEqual(result["question"], self.question)
        self.assertEqual(result["model"], "offline-test-model")
        self.assertEqual(
            datetime.fromisoformat(result["timestamp"]).utcoffset(), timedelta(0)
        )
        self.assertEqual(json.loads(self.output_path.read_text(encoding="utf-8")), result)
        self.assertEqual(
            set(result),
            {"benchmark_id", "question", "model", "timestamp", "assessments",
             "semantic_ranking", "baseline_metrics", "semantic_metrics"},
        )
        for item in result["assessments"]:
            self.assertEqual(
                set(item),
                {"group_id", "original_rrf_rank", "original_rrf_score", "title",
                 "category", "semantic_score", "reason"},
            )
        self.assertNotIn("GOLD ONLY", self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [path.read_bytes() for path in (self.candidate_path, self.gold_path)], before
        )
        printed = output.getvalue()
        self.assertIn("[1/22] assessing", printed)
        self.assertIn("[22/22] assessing", printed)
        self.assertIn("RRF baseline", printed)
        self.assertIn("Semantic reranker", printed)
        self.assertIn("nDCG@10", printed)
        self.assertIn("Semantic Top 10:", printed)
        self.assertEqual(printed.count("original RRF rank="), 10)

    def test_invalid_snapshot_fails_before_any_assessment_or_gold_read(self):
        mutations = [
            ("missing question", lambda data: data.pop("question")),
            ("blank question", lambda data: data.update(question=" \n")),
            ("non-string question", lambda data: data.update(question=7)),
            ("missing candidates", lambda data: data.pop("candidates")),
            ("not a candidate list", lambda data: data.update(candidates={})),
            ("missing title", lambda data: data["candidates"][0].pop("title")),
            ("blank title", lambda data: data["candidates"][0].update(title=" ")),
            ("missing group_id", lambda data: data["candidates"][0].pop("group_id")),
            ("blank group_id", lambda data: data["candidates"][0].update(group_id="")),
            ("duplicate group_id", lambda data: data["candidates"][0].update(group_id="group-2")),
            ("duplicate RRF rank", lambda data: data["candidates"][0].update(rrf_rank=2)),
            ("invalid RRF score", lambda data: data["candidates"][0].update(rrf_score=None)),
            ("wrong benchmark", lambda data: data.update(benchmark_id="another_v1")),
        ]
        for name, mutate in mutations:
            with self.subTest(case=name):
                snapshot = deepcopy(self.snapshot)
                mutate(snapshot)
                self.candidate_path.write_text(json.dumps(snapshot), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self._run(gold_path=self.root / "unreadable.json")
                self.client.assess.assert_not_called()
                self.assertFalse(self.output_path.exists())

    def test_failed_assessment_propagates_without_gold_metrics_or_overwriting_run(self):
        failure = RuntimeError("Assessment failed")
        self.client.assess.side_effect = [
            RelevanceAssessment(category="direct", score=0.8, reason="Fake evidence"),
            failure,
        ]
        self.output_path.parent.mkdir()
        self.output_path.write_text("previous completed run", encoding="utf-8")
        output = io.StringIO()
        result = None

        with redirect_stdout(output), self.assertRaises(RuntimeError) as caught:
            result = self._run(gold_path=self.root / "unreadable.json")

        self.assertIs(caught.exception, failure)
        self.assertIsNone(result)
        self.assertEqual(self.client.assess.call_count, 2)
        self.assertNotIn("RRF baseline", output.getvalue())
        self.assertNotIn("Semantic Top 10", output.getvalue())
        self.assertEqual(
            self.output_path.read_text(encoding="utf-8"), "previous completed run"
        )

    def test_missing_gold_outside_semantic_top_ten_is_detected(self):
        del self.gold["judgments"]["group-22"]
        self._write_inputs()
        output = io.StringIO()

        with redirect_stdout(output), self.assertRaisesRegex(ValueError, "Missing gold"):
            self._run()

        self.assertEqual(self.client.assess.call_count, 22)
        self.assertNotIn("RRF baseline", output.getvalue())
        self.assertFalse(self.output_path.exists())

    def test_mismatched_gold_question_is_rejected_after_assessment(self):
        self.gold["research_question"] = "Different benchmark question"
        self._write_inputs()

        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "question"):
            self._run()

        self.assertEqual(self.client.assess.call_count, 22)
        self.assertFalse(self.output_path.exists())

    def test_output_cannot_overwrite_frozen_inputs(self):
        for path in (self.candidate_path, self.gold_path):
            with self.subTest(path=path):
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "must not overwrite"):
                    self._run(output_path=path)
                self.assertEqual(path.read_bytes(), before)
        self.client.assess.assert_not_called()

    def test_command_line_passes_model_to_evaluation(self):
        with patch.object(runner, "run_evaluation") as evaluate:
            runner.main(["--benchmark", self.benchmark_id, "--model", "configured-model"])
        evaluate.assert_called_once_with(self.benchmark_id, model="configured-model")

    def test_candidate_count_comes_from_snapshot_including_empty_and_small_pools(self):
        candidates = deepcopy(self.snapshot["candidates"])
        judgments = deepcopy(self.gold["judgments"])
        self.client.assess.side_effect = None
        self.client.assess.return_value = RelevanceAssessment(
            category="supporting", score=0.5, reason="Offline fixture"
        )
        for count in (0, 3, 21):
            with self.subTest(count=count):
                self.client.reset_mock()
                self.snapshot["candidates"] = candidates[:count]
                ids = [item["group_id"] for item in candidates[:count]]
                self.gold["baseline"]["ranked_item_ids"] = ids
                self.gold["judgments"] = {group_id: judgments[group_id] for group_id in ids}
                self._write_inputs()
                output = io.StringIO()

                with redirect_stdout(output):
                    result = self._run()

                self.assertEqual(self.client.assess.call_count, count)
                self.assertEqual(len(result["assessments"]), count)
                self.assertEqual(result["semantic_ranking"], ids)
                if count:
                    self.assertIn(f"[{count}/{count}] assessing", output.getvalue())

    def test_unknown_benchmark_fails_before_any_assessment(self):
        with self.assertRaisesRegex(ValueError, "Unknown benchmark id"):
            self._run(benchmark_id="unknown_benchmark_v1")
        self.client.assess.assert_not_called()
        self.assertFalse(self.output_path.exists())

    def test_existing_matrix_benchmark_works_with_canonical_paths_and_fake_assessments(self):
        self.client.assess.side_effect = None
        self.client.assess.return_value = RelevanceAssessment(
            category="supporting", score=0.5, reason="Offline regression fixture"
        )

        with redirect_stdout(io.StringIO()):
            result = self._run(
                benchmark_id="matrix_completion_v1",
                root=PROJECT_ROOT,
                candidate_path=None,
                gold_path=None,
            )

        self.assertEqual(result["benchmark_id"], "matrix_completion_v1")
        self.assertEqual(self.client.assess.call_count, 22)
        self.assertEqual(result["semantic_metrics"], result["baseline_metrics"])
        self.assertEqual(result["baseline_metrics"]["precision_at_10"], 0.6)
        self.assertEqual(result["baseline_metrics"]["strict_precision_at_10"], 0.5)
        self.assertAlmostEqual(
            result["baseline_metrics"]["ndcg_at_10"], 0.6781498023908037, places=12
        )


if __name__ == "__main__":
    unittest.main()
