"""Offline checks of artifact consistency, summaries and report publication."""

from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

from eval.benchmark import PROJECT_ROOT, benchmark_paths
from eval.retrieval.metrics import ndcg_at_k, precision_at_k, strict_precision_at_k
from eval.semantic_summary import build_report, generate_report, render_markdown
from scripts import summarize_semantic_benchmarks as cli


def metrics(ids, grades):
    return {
        "precision_at_10": precision_at_k(ids, grades, 10),
        "strict_precision_at_10": strict_precision_at_k(ids, grades, 10),
        "ndcg_at_10": ndcg_at_k(ids, grades, 10),
    }


class SemanticBenchmarkSummaryTests(unittest.TestCase):
    def setUp(self):
        # Ordinary mkdir preserves inherited Windows ACLs in the workspace.
        self.root = Path(__file__).resolve().parent / f"summary-fixture-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self._cleanup)
        for target in ("socket.create_connection", "socket.socket.connect"):
            guard = patch(target, side_effect=AssertionError("Offline tests must not connect"))
            guard.start()
            self.addCleanup(guard.stop)
        self.ids = ["first_topic_v1", "second_topic_v1"]
        self.fixtures = {
            self.ids[0]: self._fixture(self.ids[0], [2, 1, 0] * 4,
                                       ["direct", "direct", "supporting"] * 4,
                                       list(range(7, 13)) + list(range(1, 7))),
            self.ids[1]: self._fixture(self.ids[1], [0, 0, 1, 2],
                                       ["off_target", "off_target", "supporting", "direct"],
                                       [4, 3, 2, 1]),
        }
        self._write_inputs()

    def _cleanup(self):
        assert self.root.resolve().is_relative_to(Path(__file__).resolve().parent)
        for path in sorted(self.root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        self.root.rmdir()

    def _fixture(self, benchmark_id, relevances, categories, semantic_order):
        question = f"Question | for {benchmark_id}\nwith formatting?"
        ids = [f"group-{i}" for i in range(1, len(relevances) + 1)]
        candidates = [
            {"group_id": gid, "title": "Shared | title\nwith formatting", "abstract": None,
             "rrf_rank": rank, "rrf_score": 1 / (60 + rank)}
            for rank, gid in enumerate(ids, 1)
        ]
        grades = dict(zip(ids, relevances))
        semantic_ids = [f"group-{i}" for i in semantic_order]
        semantic_scores = {gid: 1 - rank / (len(ids) + 1) for rank, gid in enumerate(semantic_ids, 1)}
        baseline_metrics = metrics(ids, grades)
        return {
            "definition": {
                "benchmark_id": benchmark_id, "question": question,
                "year_from": None, "year_to": None, "per_query": 10,
                "queries": [{"query_id": "q1", "text": "fixture query"}],
            },
            "candidates": {"benchmark_id": benchmark_id, "question": question, "candidates": candidates},
            "gold": {
                "benchmark_id": benchmark_id, "research_question": question,
                "baseline": {"ranked_item_ids": ids, "expected_metrics": baseline_metrics},
                # Reverse dictionary order and use identical titles to test joins by id.
                "judgments": {gid: {"group_id": gid, "title": "Shared title", "relevance": grades[gid]}
                              for gid in reversed(ids)},
            },
            "run": {
                "benchmark_id": benchmark_id, "question": question, "model": "fixture-model",
                "timestamp": "2026-09-16T10:00:00+00:00",
                "assessments": [
                    {"group_id": item["group_id"], "title": item["title"],
                     "original_rrf_rank": item["rrf_rank"], "original_rrf_score": item["rrf_score"],
                     "category": category, "semantic_score": semantic_scores[item["group_id"]],
                     "reason": "Offline fixture"}
                    for item, category in zip(candidates, categories)
                ][::-1],
                "semantic_ranking": semantic_ids,
                "baseline_metrics": baseline_metrics,
                "semantic_metrics": metrics(semantic_ids, grades),
            },
        }

    def _write_inputs(self):
        for benchmark_id, artifacts in self.fixtures.items():
            paths = benchmark_paths(benchmark_id, root=self.root)
            for name, value in artifacts.items():
                path = getattr(paths, name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")

    def _build(self, ids=None):
        return build_report("fixture_report", self.ids if ids is None else ids, root=self.root)

    def _generate(self, **kwargs):
        return generate_report("fixture_report", self.ids, root=self.root, **kwargs)

    def _outputs(self):
        return [self.root / "eval/reports" / f"fixture_report{suffix}" for suffix in (".json", ".md")]

    def test_successful_two_benchmark_aggregation_and_group_id_mapping(self):
        report = self._build()
        self.assertEqual(report["benchmark_ids"], self.ids)
        self.assertEqual(report["benchmark_count"], 2)
        self.assertEqual(datetime.fromisoformat(report["generated_at"]).utcoffset(), timedelta(0))
        first, second = [report["per_benchmark"][bid] for bid in self.ids]
        self.assertEqual(first["candidate_count"], 12)
        self.assertEqual(second["candidate_count"], 4)
        self.assertEqual((first["direct_count"], first["supporting_count"], first["off_target_count"]), (4, 4, 4))
        self.assertEqual(second["semantic_top_10"][0]["group_id"], "group-4")
        self.assertEqual(second["semantic_top_10"][0]["gold_relevance"], 2)
        # Keep imperfect model predictions distinct from human labels.
        disagreement = first["rank_movements"][1]
        self.assertEqual(disagreement["gold_label_name"], "supporting")
        self.assertEqual(disagreement["model_category"], "direct")

    def test_duplicate_benchmark_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "Requested benchmark.*unique"):
            self._build([self.ids[0], self.ids[0]])

    def test_empty_benchmark_list_and_unsafe_ids_rejected(self):
        for ids in ([], ["../other"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self._build(ids)
        with self.assertRaisesRegex(ValueError, "report_id"):
            build_report("../source", self.ids, root=self.root)

    def test_missing_artifacts_rejected_without_partial_report(self):
        paths = benchmark_paths(self.ids[1], root=self.root)
        for name in ("definition", "candidates", "gold", "run"):
            with self.subTest(artifact=name):
                path = getattr(paths, name)
                data = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(ValueError, self.ids[1]):
                    self._generate()
                self.assertFalse(any(path.exists() for path in self._outputs()))
                path.write_bytes(data)

    def test_mismatching_benchmark_ids_and_questions_rejected(self):
        for name in ("definition", "candidates", "gold", "run"):
            artifact = self.fixtures[self.ids[0]][name]
            question_key = "research_question" if name == "gold" else "question"
            for field in ("benchmark_id", question_key):
                with self.subTest(artifact=name, field=field):
                    original = artifact[field]
                    artifact[field] = "wrong_value"
                    self._write_inputs()
                    with self.assertRaisesRegex(ValueError, "benchmark_id|question"):
                        self._build()
                    artifact[field] = original
                    self._write_inputs()

    def test_legacy_snapshot_without_id_still_checks_question_and_pool(self):
        snapshot = self.fixtures[self.ids[0]]["candidates"]
        del snapshot["benchmark_id"]
        self._write_inputs()
        self.assertIn("predates benchmark_id", self._build()["per_benchmark"][self.ids[0]]["validation_notes"][0])
        snapshot["question"] = "Wrong question"
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "question"):
            self._build()

    def test_duplicate_candidate_group_ids_rejected(self):
        snapshot = self.fixtures[self.ids[0]]["candidates"]
        snapshot["candidates"][1]["group_id"] = snapshot["candidates"][0]["group_id"]
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "Candidates.*unique"):
            self._build()

    def test_candidate_gold_mismatch_rejected(self):
        judgments = self.fixtures[self.ids[0]]["gold"]["judgments"]
        judgments["unexpected"] = judgments.pop("group-12")
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "Gold judgment.*exactly match"):
            self._build()

    def test_missing_and_duplicate_assessments_rejected(self):
        run = self.fixtures[self.ids[0]]["run"]
        original = deepcopy(run["assessments"])
        for replacement in (original[:-1], original + [original[0]]):
            with self.subTest(count=len(replacement)):
                run["assessments"] = replacement
                self._write_inputs()
                with self.assertRaisesRegex(ValueError, "Assessment.*unique|Assessment.*exactly match"):
                    self._build()

    def test_semantic_ranking_missing_item_rejected(self):
        self.fixtures[self.ids[0]]["run"]["semantic_ranking"].pop()
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "Semantic ranking.*exactly match"):
            self._build()

    def test_semantic_ranking_duplicate_item_rejected(self):
        ranking = self.fixtures[self.ids[0]]["run"]["semantic_ranking"]
        ranking[-1] = ranking[0]
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "Semantic ranking.*unique"):
            self._build()

    def test_semantic_ranking_must_match_scores_and_stable_ties(self):
        run = self.fixtures[self.ids[1]]["run"]
        for item in run["assessments"]:
            item["semantic_score"] = 0.5
        # Saved descending-score order is no longer a valid stable tie order.
        self._write_inputs()
        with self.assertRaisesRegex(ValueError, "stable RRF ties"):
            self._build()
        run["semantic_ranking"] = ["group-1", "group-2", "group-3", "group-4"]
        run["semantic_metrics"] = run["baseline_metrics"]
        self._write_inputs()
        summary = self._build()["per_benchmark"][self.ids[1]]
        self.assertEqual([item["group_id"] for item in summary["semantic_top_10"]], run["semantic_ranking"])

    def test_baseline_and_assessment_rrf_metadata_checked(self):
        artifacts = self.fixtures[self.ids[0]]
        original = deepcopy(artifacts)
        mutations = [
            lambda a: a["gold"]["baseline"]["ranked_item_ids"].reverse(),
            lambda a: a["run"]["assessments"][0].update(original_rrf_rank=1),
            lambda a: a["run"]["assessments"][0].update(original_rrf_score=0.9),
            lambda a: a["run"]["assessments"][0].update(title="Wrong title"),
        ]
        for mutate in mutations:
            artifacts = deepcopy(original)
            mutate(artifacts)
            self.fixtures[self.ids[0]] = artifacts
            self._write_inputs()
            with self.assertRaises(ValueError):
                self._build()

    def test_recomputed_metrics_match_saved_metrics_with_float_tolerance(self):
        self.fixtures[self.ids[0]]["run"]["semantic_metrics"]["ndcg_at_10"] += 1e-13
        self._write_inputs()
        report = self._build()
        for bid in self.ids:
            summary = report["per_benchmark"][bid]
            run = self.fixtures[bid]["run"]
            for key, value in run["baseline_metrics"].items():
                self.assertAlmostEqual(summary[f"rrf_{key}"], value, places=12)
                self.assertAlmostEqual(summary[f"semantic_{key}"], run["semantic_metrics"][key], places=12)
                self.assertEqual(summary[f"delta_{key}"], summary[f"semantic_{key}"] - summary[f"rrf_{key}"])

    def test_metric_mismatch_rejected_in_run_and_gold(self):
        artifacts = self.fixtures[self.ids[0]]
        for saved in (artifacts["run"]["baseline_metrics"], artifacts["run"]["semantic_metrics"],
                      artifacts["gold"]["baseline"]["expected_metrics"]):
            original = saved["ndcg_at_10"]
            saved["ndcg_at_10"] = 0.01
            self._write_inputs()
            with self.assertRaisesRegex(ValueError, "ndcg_at_10 mismatch"):
                self._build()
            saved["ndcg_at_10"] = original

    def test_nonfinite_nonnumeric_and_bool_saved_metrics_rejected(self):
        saved = self.fixtures[self.ids[0]]["run"]["semantic_metrics"]
        for bad in (float("nan"), float("inf"), float("-inf"), "0.5", True, None):
            with self.subTest(value=bad):
                saved["extra_saved_metric"] = bad
                self._write_inputs()
                with self.assertRaisesRegex(ValueError, "finite numeric"):
                    self._build()

    def test_invalid_assessment_scores_and_categories_rejected(self):
        item = self.fixtures[self.ids[0]]["run"]["assessments"][0]
        original = deepcopy(item)
        for update in ({"semantic_score": 1.1}, {"semantic_score": -0.1},
                       {"semantic_score": float("nan")}, {"category": "unknown"}):
            item.update(original | update)
            self._write_inputs()
            with self.subTest(update=update), self.assertRaises(ValueError):
                self._build()

    def test_rank_change_and_stable_top_five_movement_order(self):
        summary = self._build()["per_benchmark"][self.ids[0]]
        for item in summary["rank_movements"]:
            self.assertEqual(item["rank_change"], item["original_rrf_rank"] - item["semantic_rank"])
        self.assertEqual([item["original_rrf_rank"] for item in summary["top_5_upward_movements"]], [7, 8, 9, 10, 11])
        self.assertEqual([item["original_rrf_rank"] for item in summary["top_5_downward_movements"]], [1, 2, 3, 4, 5])
        self.assertTrue(all(item["rank_change"] == 6 for item in summary["top_5_upward_movements"]))
        self.assertTrue(all(item["rank_change"] == -6 for item in summary["top_5_downward_movements"]))

    def test_confusion_matrix_and_category_accuracy(self):
        first, second = self._build()["per_benchmark"].values()
        matrix = first["confusion_matrix"]
        self.assertEqual(matrix["row_labels"], ["direct", "supporting", "off_target"])
        self.assertEqual(matrix["column_labels"], matrix["row_labels"])
        self.assertEqual(matrix["counts"], [[4, 0, 0], [4, 0, 0], [0, 4, 0]])
        self.assertAlmostEqual(first["category_accuracy"], 1 / 3)
        self.assertEqual(second["category_accuracy"], 1.0)

    def test_macro_averages_give_different_sized_pools_equal_weight(self):
        report = self._build()
        first, second = report["per_benchmark"].values()
        aggregate = report["aggregate"]
        self.assertEqual(aggregate["number_of_benchmarks"], 2)
        for key in ("precision_at_10", "strict_precision_at_10", "ndcg_at_10"):
            for prefix in ("rrf", "semantic", "delta"):
                self.assertAlmostEqual(aggregate[key][f"mean_{prefix}"],
                                       (first[f"{prefix}_{key}"] + second[f"{prefix}_{key}"]) / 2)
        self.assertAlmostEqual(aggregate["precision_at_10"]["mean_rrf"], 0.45)

    def test_report_json_markdown_and_sources_unchanged(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*.json")}
        report = self._generate()
        json_path, md_path = self._outputs()
        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), report)
        markdown = md_path.read_text(encoding="utf-8")
        for heading in ("# Semantic Reranking Evaluation Report", "## Experimental Setup",
                        "## Overall Results", "## Aggregate Results", "## Observations", "## Limitations",
                        *[f"## Benchmark: {bid}" for bid in self.ids], "### Semantic Top 10",
                        "### Rank Movements", "### Category Calibration"):
            self.assertIn(heading, markdown)
        self.assertIn("Shared \\| title with formatting", markdown)
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_either_existing_output_protects_both_files_without_overwrite(self):
        for path in self._outputs():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("existing report", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                self._generate()
            self.assertEqual(path.read_text(encoding="utf-8"), "existing report")
            other = next(item for item in self._outputs() if item != path)
            self.assertFalse(other.exists())
            path.unlink()

    def test_explicit_overwrite_replaces_both_reports(self):
        for path in self._outputs():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("existing report", encoding="utf-8")
        report = self._generate(overwrite=True)
        self.assertEqual(json.loads(self._outputs()[0].read_text(encoding="utf-8")), report)
        self.assertTrue(self._outputs()[1].read_text(encoding="utf-8").startswith("# Semantic"))

    def test_invalid_second_benchmark_does_not_overwrite_existing_reports(self):
        self._generate()
        before = {path: path.read_bytes() for path in self._outputs()}
        benchmark_paths(self.ids[1], root=self.root).run.unlink()
        with self.assertRaises(ValueError):
            self._generate(overwrite=True)
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_cli_forwards_only_explicit_ids_and_overwrite_option(self):
        for overwrite in (False, True):
            with patch.object(cli, "generate_report", return_value={"benchmark_count": 2}) as generate:
                with redirect_stdout(io.StringIO()):
                    cli.main(["--report-id", "fixture_report", "--benchmarks", *self.ids]
                             + (["--overwrite"] if overwrite else []))
                generate.assert_called_once_with("fixture_report", self.ids, overwrite=overwrite)

    def test_unrequested_invalid_artifacts_are_not_scanned(self):
        (self.root / "eval/runs/unrelated_semantic_v1.json").write_text("not JSON", encoding="utf-8")
        self.assertEqual(self._build()["benchmark_count"], 2)

    def test_frozen_repository_results_and_legacy_run_copy(self):
        ids = ["matrix_completion_v1", "rag_hallucination_v1"]
        sources = []
        for bid in ids:
            paths = benchmark_paths(bid)
            sources.extend([paths.definition, paths.candidates, paths.gold, paths.run])
        before = {path: path.read_bytes() for path in sources}
        report = build_report("regression_report", ids)
        expected = {
            ids[0]: (22, 0.6, 0.5, 0.6781498023908037, 1.0, 0.7, 1.0),
            ids[1]: (17, 1.0, 0.3, 0.8104156585300656, 1.0, 0.4, 0.9588342406066944),
        }
        fields = ("candidate_count", "rrf_precision_at_10", "rrf_strict_precision_at_10",
                  "rrf_ndcg_at_10", "semantic_precision_at_10", "semantic_strict_precision_at_10",
                  "semantic_ndcg_at_10")
        for bid, values in expected.items():
            for field, value in zip(fields, values):
                with self.subTest(benchmark=bid, field=field):
                    self.assertAlmostEqual(report["per_benchmark"][bid][field], value, places=12)
        self.assertEqual({path: path.read_bytes() for path in sources}, before)
        legacy = PROJECT_ROOT / "eval/runs/matrix_completion_semantic_v1.json"
        self.assertEqual(benchmark_paths(ids[0]).run.read_bytes(), legacy.read_bytes())
        markdown = render_markdown(report)
        self.assertIn("Only 2 frozen benchmarks", markdown)
        self.assertIn("prompt-rubric refinement", markdown)
        self.assertIn("one recorded model/configuration", markdown)
        self.assertIn("No repeated-run variance analysis", markdown)

    def test_observations_describe_decreases_and_unchanged_metrics(self):
        report = self._build()
        first, second = report["per_benchmark"].values()
        first["delta_ndcg_at_10"] = -0.1
        second["delta_ndcg_at_10"] = 0.0
        observations = render_markdown(report).split("## Observations")[1].split("## Limitations")[0]
        self.assertIn("nDCG@10 decreased (delta -0.1000000)", observations)
        self.assertIn("nDCG@10 was unchanged (delta +0.0000000)", observations)


if __name__ == "__main__":
    unittest.main()
