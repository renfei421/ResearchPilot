"""Hand-calculated planner evaluation tests; no APIs or real label changes."""

from copy import deepcopy
from math import log2
import unittest

from eval.query_planner_evaluation import (
    METRICS, build_final_relevance, build_report, evaluate_benchmark,
    macro_metrics, ranking_metrics, relevance_map,
)
from eval.query_planner_report import render_markdown
from eval.retrieval.metrics import ndcg_at_k, precision_at_k, strict_precision_at_k
from test_union_gold import fixture as union_fixture


def fixture():
    manifest, human, _ = union_fixture()
    # Independently specify canonical order, including the merged a records.
    manifest["human_union_ranking"] = ["union:shared", "union:human"]
    manifest["llm_union_ranking"] = ["union:shared", "union:a", "union:b", "union:c"]
    adopted = {
        "benchmark_id": "fixture_v1", "research_question": human["research_question"],
        "annotation_source": "codex_model_assisted", "label_status": "silver", "human_gold": False,
        "judgment_count": 3,
        "judgments": [
            {"union_id": uid, "label": label, "confidence": confidence, "note": "Original annotation note", "annotation_source": "codex_model_assisted"}
            for uid, label, confidence in (("union:a", 0, "high"), ("union:b", 1, "medium"), ("union:c", 2, "low"))
        ],
    }
    return manifest, human, adopted


class FinalRelevanceTests(unittest.TestCase):
    def test_human_mapping_shared_and_human_only_preserve_labels_and_history(self):
        manifest, human, adopted = fixture()
        result = build_final_relevance(manifest, human, adopted)
        self.assertEqual(result["judgment_count"], 5)
        for group, uid in (("h_shared", "union:shared"), ("h_only", "union:human")):
            record = result["judgments"][uid]
            self.assertEqual(record["relevance"], human["judgments"][group]["relevance"])
            self.assertEqual(record["inherited_judgments"], [human["judgments"][group]])
            self.assertEqual(record["provenance"], "inherited_human_gold")
        self.assertEqual(result["source_history"]["human_gold_provenance"], human["provenance"])

    def test_only_llm_only_adopted_labels_fill_union_and_preserve_history(self):
        inputs = fixture()
        result = build_final_relevance(*inputs)
        self.assertEqual(set(result["judgments"]), {w["union_id"] for w in inputs[0]["works"]})
        for entry in inputs[2]["judgments"]:
            record = result["judgments"][entry["union_id"]]
            self.assertEqual(record["relevance"], entry["label"])
            self.assertEqual(record["confidence"], entry["confidence"])
            self.assertEqual(record["note"], entry["note"])
            self.assertEqual(record["original_provenance"], entry)
            self.assertEqual(record["provenance"], "project_owner_adopted")
        self.assertEqual(result["source_history"]["adopted_file_metadata"]["label_status"], "silver")
        self.assertEqual(result["label_status"], "final_adopted_for_query_planner_v1")

    def test_human_precedence_rejects_any_adopted_overwrite(self):
        for uid in ("union:shared", "union:human"):
            for attempted_label in (0, 1, 2):
                with self.subTest(uid=uid, attempted_label=attempted_label):
                    inputs = fixture()
                    original_human = deepcopy(inputs[1])
                    inputs[2]["judgments"].append({"union_id": uid, "label": attempted_label})
                    inputs[2]["judgment_count"] += 1
                    with self.assertRaisesRegex(ValueError, "Cannot overwrite inherited Human"):
                        build_final_relevance(*inputs)
                    self.assertEqual(inputs[1], original_human)

    def test_missing_unknown_duplicate_adopted_labels_fail_clearly(self):
        for change, message in (
            (lambda a: a["judgments"].pop(), "Missing adopted relevance"),
            (lambda a: a["judgments"][0].update(union_id="unknown"), "Unknown adopted union_id"),
            (lambda a: a["judgments"].append(deepcopy(a["judgments"][0])), "Duplicate adopted judgment"),
        ):
            inputs = fixture()
            change(inputs[2])
            inputs[2]["judgment_count"] = len(inputs[2]["judgments"])
            with self.assertRaisesRegex(ValueError, message):
                build_final_relevance(*inputs)

    def test_adopted_label_strict_integer_validation(self):
        for label in (True, False, 2.0, "2", None, -1, 3):
            inputs = fixture()
            inputs[2]["judgments"][0]["label"] = label
            with self.assertRaisesRegex(ValueError, "integer 0, 1, or 2"):
                build_final_relevance(*inputs)

    def test_source_identity_and_count_mismatches_fail(self):
        for change in (
            lambda a: a.update(benchmark_id="other"),
            lambda a: a.update(research_question="other"),
            lambda a: a.update(judgment_count=4),
        ):
            inputs = fixture()
            change(inputs[2])
            with self.assertRaises(ValueError):
                build_final_relevance(*inputs)

    def test_missing_unknown_duplicate_final_judgments_fail(self):
        inputs = fixture()
        base = build_final_relevance(*inputs)
        mutations = (
            (lambda d: d["judgments"].pop("union:a"), "Missing final relevance"),
            (lambda d: d["judgments"].update({"unknown": {"union_id": "unknown", "relevance": 1}}), "Unknown final union_id"),
            (lambda d: d["judgments"].update({"alias": deepcopy(d["judgments"]["union:a"])}), "Duplicate final judgment"),
        )
        for change, message in mutations:
            document = deepcopy(base)
            change(document)
            with self.assertRaisesRegex(ValueError, message):
                relevance_map(inputs[0], document)

    def test_inputs_not_mutated_and_outputs_not_aliased(self):
        inputs = fixture()
        before = deepcopy(inputs)
        result = build_final_relevance(*inputs)
        evaluate_benchmark(inputs[0], result)
        result["judgments"]["union:a"]["original_provenance"]["label"] = 2
        result["judgments"]["union:shared"]["inherited_judgments"][0]["relevance"] = 0
        self.assertEqual(inputs, before)


class PlannerMetricTests(unittest.TestCase):
    def setUp(self):
        self.inputs = fixture()
        self.dataset = build_final_relevance(*self.inputs)
        self.result = evaluate_benchmark(self.inputs[0], self.dataset)

    def test_full_union_recall_formulas_and_common_denominators(self):
        self.assertEqual(self.result["relevance_denominators"], {"relevant": 4, "direct": 2})
        self.assertEqual(self.result["metrics"]["human"]["relevant_recall"], 2/4)
        self.assertEqual(self.result["metrics"]["llm"]["relevant_recall"], 3/4)
        self.assertEqual(self.result["metrics"]["human"]["direct_recall"], 1/2)
        self.assertEqual(self.result["metrics"]["llm"]["direct_recall"], 2/2)
        self.assertEqual(self.result["retrieved_counts"], {"human": {"relevant": 2, "direct": 1}, "llm": {"relevant": 3, "direct": 2}})

    def test_precision_divides_by_ten_even_when_ranking_shorter(self):
        for source, relevant, direct in (("human", 2, 1), ("llm", 3, 2)):
            self.assertEqual(self.result["metrics"][source]["precision_at_10"], relevant/10)
            self.assertEqual(self.result["metrics"][source]["strict_precision_at_10"], direct/10)

    def test_ndcg_full_union_ideal_contains_unretrieved_relevance(self):
        ideal = 3 + 3/log2(3) + 1/log2(4) + 1/log2(5)
        self.assertAlmostEqual(self.result["metrics"]["human"]["ndcg_at_10"], (3 + 1/log2(3))/ideal)
        self.assertAlmostEqual(self.result["metrics"]["llm"]["ndcg_at_10"], (3 + 1/log2(4) + 3/log2(5))/ideal)
        self.assertLess(self.result["metrics"]["human"]["ndcg_at_10"], 1)

    def test_recall_includes_beyond_top_ten_but_precision_does_not(self):
        labels = {str(i): 0 for i in range(10)} | {"direct": 2, "support": 1}
        result = ranking_metrics([str(i) for i in range(10)] + ["direct"], labels)
        self.assertEqual(result["relevant_recall"], 1/2)
        self.assertEqual(result["direct_recall"], 1)
        self.assertEqual(result["precision_at_10"], 0)
        self.assertEqual(result["strict_precision_at_10"], 0)
        self.assertEqual(result["ndcg_at_10"], 0)

    def test_bibliographic_duplicates_count_once(self):
        self.assertEqual(len(self.inputs[0]["works"][2]["members"]), 2)
        self.assertEqual(self.result["rankings"]["llm"].count("union:a"), 1)
        self.assertEqual(self.result["structural_counts"]["llm_retrieved"], 4)
        self.assertEqual(self.result["unique_coverage"]["llm_only"]["total_unique_works"], 3)

    def test_human_and_llm_unique_compositions(self):
        human = self.result["unique_coverage"]["human_only"]
        llm = self.result["unique_coverage"]["llm_only"]
        self.assertEqual((human["label_0"], human["label_1"], human["label_2"]), (0,1,0))
        self.assertEqual((llm["label_0"], llm["label_1"], llm["label_2"]), (1,1,1))
        self.assertEqual(human["relevant_unique_works"], 1)
        self.assertEqual(llm["relevant_unique_works"], 2)
        self.assertEqual(llm["direct_core_unique_works"], 1)

    def test_shared_rank_delta_uses_canonical_positions_not_gapped_raw_ranks(self):
        manifest = deepcopy(self.inputs[0])
        manifest["works"][0]["members"][1]["original_rank"] = 6
        manifest["llm_union_ranking"] = ["union:a", "union:b", "union:c", "union:shared"]
        result = evaluate_benchmark(manifest, self.dataset)
        row = result["shared_work_rank_diagnostics"][0]
        self.assertEqual((row["union_id"], row["human_rank"], row["llm_rank"], row["rank_delta"]), ("union:shared", 1, 4, -3))
        self.assertEqual(row["llm_best_original_rank"], 6)
        self.assertEqual(result["shared_work_rank_summary"], {"shared_count": 1, "llm_higher": 0, "human_higher": 1, "tied": 0})

    def test_confidence_provenance_notes_and_rrf_scores_do_not_affect_metrics(self):
        inputs = deepcopy(self.inputs)
        for entry in inputs[2]["judgments"]:
            entry.update(confidence="low", note="Changed audit text", annotation_source="audit-only", rrf_score=10000)
        changed = build_final_relevance(*inputs)
        for record in changed["judgments"].values():
            record.update(provenance="audit only", confidence="high", note="changed", rrf_score=-999)
        result = evaluate_benchmark(inputs[0], changed)
        self.assertEqual(result["metrics"], self.result["metrics"])
        self.assertEqual(result["unique_coverage"], self.result["unique_coverage"])

    def test_empty_ranking_metrics_zero_with_valid_full_union(self):
        self.assertEqual(ranking_metrics([], {"direct": 2}), {metric: 0.0 for metric in METRICS})

    def test_denominators_must_be_positive(self):
        for labels in ({}, {"a": 0}, {"a": 1}):
            with self.assertRaisesRegex(ValueError, "denominators must both be positive"):
                ranking_metrics(list(labels), labels)

    def test_all_rank_ids_valid_even_after_cutoff(self):
        labels = {str(i): 2 for i in range(11)}
        for ranked, message in ((list(labels) + ["unknown"], "Unknown union_id"), (list(labels) + ["0"], "Duplicate union_id")):
            with self.assertRaisesRegex(ValueError, message):
                ranking_metrics(ranked, labels)

    def test_manifest_rankings_must_match_source_membership_and_order(self):
        for mutation in (
            lambda m: m["llm_union_ranking"].reverse(),
            lambda m: m["llm_union_ranking"].pop(),
            lambda m: m["llm_union_ranking"].append("union:human"),
            lambda m: m["llm_union_ranking"].append("union:a"),
            lambda m: m["summary"].update(intersection=2),
        ):
            manifest = deepcopy(self.inputs[0])
            mutation(manifest)
            with self.assertRaises(ValueError):
                evaluate_benchmark(manifest, self.dataset)

    def test_metric_bounds_structural_invariants_and_deltas(self):
        s = self.result["structural_counts"]
        self.assertEqual(s["human_retrieved"], s["intersection"] + s["human_only"])
        self.assertEqual(s["llm_retrieved"], s["intersection"] + s["llm_only"])
        self.assertEqual(s["union_size"], s["intersection"] + s["human_only"] + s["llm_only"])
        for metric in METRICS:
            h, l = (self.result["metrics"][source][metric] for source in ("human", "llm"))
            self.assertTrue(0 <= h <= 1 and 0 <= l <= 1)
            self.assertEqual(self.result["metrics"]["delta"][metric], l-h)

    def test_existing_metric_helpers_remain_backward_compatible(self):
        for helper in (precision_at_k, strict_precision_at_k, ndcg_at_k):
            self.assertEqual(helper([], {}, 10), 0)
            self.assertEqual(helper(["a", "unknown", "a"], {"a": 2}, 1), 1)
        labels = relevance_map(self.inputs[0], self.dataset)
        for source in ("human", "llm"):
            ranking = self.result["rankings"][source]
            for metric, helper in (("precision_at_10", precision_at_k), ("strict_precision_at_10", strict_precision_at_k), ("ndcg_at_10", ndcg_at_k)):
                self.assertEqual(self.result["metrics"][source][metric], helper(ranking, labels, 10))

    def test_macro_is_arithmetic_mean_not_pool_size_weighted(self):
        results = []
        for i, value in enumerate((0.1,0.4,1.0)):
            result = deepcopy(self.result)
            result["benchmark_id"] = f"fixture_{i}"
            result["structural_counts"]["union_size"] = (5,50,500)[i]
            result["metrics"]["human"] = {key:value for key in METRICS}
            result["metrics"]["llm"] = {key:value/2 for key in METRICS}
            results.append(result)
        result = macro_metrics(results)
        for metric in METRICS:
            self.assertEqual(result["human"][metric], 0.5)
            self.assertEqual(result["llm"][metric], 0.25)
            self.assertEqual(result["delta"][metric], -0.25)

    def test_macro_requires_nonempty_distinct_benchmarks(self):
        for results in ([], [self.result, self.result]):
            with self.assertRaises(ValueError):
                macro_metrics(results)

    def test_markdown_numeric_rows_exactly_match_report(self):
        report = build_report([self.result])
        markdown = render_markdown(report)
        for group in (self.result["metrics"], report["macro_metrics"]):
            for metric in METRICS:
                expected = " | ".join(str(group[source][metric]) for source in ("human","llm","delta"))
                self.assertIn(expected, markdown)
        self.assertIn("shared", markdown)
        self.assertIn("no weighting", markdown)


if __name__ == "__main__":
    unittest.main()
