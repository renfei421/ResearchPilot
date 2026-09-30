"""Offline canonical gold migration, evidence preparation, and final validation."""

from copy import deepcopy
from hashlib import sha256
from itertools import permutations
from unittest.mock import patch
import unittest

from eval.union_gold import (
    build_union_gold, migrate_human_gold, prepare_annotation_tasks,
    validate_annotations, validate_manifest,
)
from eval.union_gold_io import expansion_paths, load_expansion_inputs, read_json
from scripts.label_union_gold import format_task


def fixture():
    """Independent canonical partition; tests never run identity alignment."""
    def member(source, rank, group, *, year=2024):
        return {
            "source": source, "original_rank": rank, "group_id": group,
            "paper_id": f"doi:fixture/{group}", "title": f"Paper {group}",
            "year": year, "version_count": 1,
        }

    def work(uid, members):
        return {
            "union_id": uid, "members": members,
            "present_in_human": any(m["source"] == "human" for m in members),
            "present_in_llm": any(m["source"] == "llm" for m in members),
        }

    manifest = {
        "benchmark_id": "fixture_v1", "version": "v1",
        "identity_review_complete": True, "counts_status": "final",
        "review_candidates": [],
        "summary": {
            "canonical_human_works": 2, "canonical_llm_works": 4,
            "intersection": 1, "human_only": 1, "llm_only": 3,
            "unresolved_title_pairs": 0,
        },
        "works": [
            work("union:shared", [member("human", 1, "h_shared"), member("llm", 1, "l_shared")]),
            work("union:human", [member("human", 2, "h_only")]),
            work("union:a", [member("llm", 2, "l_early"), member("llm", 4, "l_abstract", year=None)]),
            work("union:b", [member("llm", 3, "l_other")]),
            work("union:c", [member("llm", 5, "l_final")]),
        ],
    }
    gold = {
        "schema_version": 1, "benchmark_id": "fixture_v1",
        "research_question": "Which methods address the fixture problem?",
        "relevance_scale": {"0": "Off-target for this benchmark", "1": "Bridge/Supporting", "2": "Direct/Core"},
        "provenance": {"source": "Frozen Human fixture", "frozen_on": "2026-09-17"},
        "judgments": {
            group: {
                "group_id": group, "title": f"Paper {group}",
                "known_paper_ids": [f"doi:fixture/{group}"],
                "relevance": label, "reason": f"Frozen Human reason for {group}.",
            }
            for group, label in (("h_shared", 2), ("h_only", 1))
        },
    }
    snapshots = {
        source: {"benchmark_id": "fixture_v1", "question": gold["research_question"], "candidates": []}
        for source in ("human", "llm")
    }
    for union in manifest["works"]:
        for m in union["members"]:
            record = {key: m[key] for key in ("group_id", "paper_id", "title", "year")}
            record["abstract"] = None if m["group_id"] == "l_early" else f"Evidence for {m['group_id']}."
            record.update({
                "rrf_rank": m["original_rank"], "rrf_score": 0.0123456789,
                "hits": [{"query_id": "SECRET_QUERY_ID", "query_text": "SECRET_QUERY_TEXT", "rank": 1}],
                "query_role": "SECRET_QUERY_ROLE", "query_rationale": "SECRET_RATIONALE",
                "semantic_score": 0.987654321, "category": "SECRET_CATEGORY",
            })
            snapshots[m["source"]]["candidates"].append(record)
    return manifest, gold, snapshots


def complete_fixture():
    manifest, gold, snapshots = fixture()
    annotations = prepare_annotation_tasks(manifest, gold, snapshots)
    # Invented labels apply ONLY to this tiny handcrafted test fixture.
    for index, task in enumerate(annotations["tasks"]):
        task["label"] = index
        task["note"] = f"Human fixture note {index}"
    return manifest, gold, snapshots, annotations


class UnionGoldMigrationTests(unittest.TestCase):
    def test_shared_and_human_only_inherit_exact_labels_and_reasons(self):
        manifest, gold, _ = fixture()
        result = migrate_human_gold(manifest, gold)
        self.assertEqual(set(result), {"union:shared", "union:human"})
        for group, uid in (("h_shared", "union:shared"), ("h_only", "union:human")):
            self.assertEqual(result[uid]["relevance"], gold["judgments"][group]["relevance"])
            self.assertEqual(result[uid]["reason"], gold["judgments"][group]["reason"])
            self.assertEqual(result[uid]["provenance"], "inherited_human_gold")
            self.assertEqual(result[uid]["inherited_judgments"], [gold["judgments"][group]])

    def test_all_three_label_values_preserved_exactly(self):
        for label in (0, 1, 2):
            with self.subTest(label=label):
                manifest, gold, _ = fixture()
                gold["judgments"]["h_shared"]["relevance"] = label
                self.assertEqual(migrate_human_gold(manifest, gold)["union:shared"]["relevance"], label)

    def test_unknown_human_gold_group_fails(self):
        manifest, gold, _ = fixture()
        gold["judgments"]["unknown"] = deepcopy(gold["judgments"]["h_shared"])
        gold["judgments"]["unknown"]["group_id"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unknown Human gold group_id"):
            migrate_human_gold(manifest, gold)

    def test_missing_human_gold_fails(self):
        manifest, gold, _ = fixture()
        del gold["judgments"]["h_only"]
        with self.assertRaisesRegex(ValueError, "Missing Human gold"):
            migrate_human_gold(manifest, gold)

    def test_same_label_collapsing_human_records_preserves_every_judgment(self):
        manifest, gold, _ = fixture()
        second = deepcopy(manifest["works"][0]["members"][0])
        second.update(group_id="h_another", original_rank=3)
        manifest["works"][0]["members"].append(second)
        other_gold = deepcopy(gold["judgments"]["h_shared"])
        other_gold.update(group_id="h_another", reason="Another original Human reason.")
        gold["judgments"]["h_another"] = other_gold
        result = migrate_human_gold(manifest, gold)["union:shared"]
        self.assertEqual(result["relevance"], 2)
        self.assertEqual(len(result["inherited_judgments"]), 2)
        self.assertIn(other_gold, result["inherited_judgments"])
        self.assertIn(other_gold["reason"], result["reason"])

    def test_conflicting_inherited_labels_fail(self):
        manifest, gold, _ = fixture()
        manifest["works"][0]["members"].extend(manifest["works"].pop(1)["members"])
        manifest["summary"].update(canonical_human_works=1, human_only=0)
        with self.assertRaisesRegex(ValueError, "Conflicting inherited"):
            migrate_human_gold(manifest, gold)

    def test_missing_or_ambiguous_manifest_mapping_fails(self):
        manifest, gold, _ = fixture()
        manifest["works"][1]["members"][0]["group_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "cannot map h_only"):
            migrate_human_gold(manifest, gold)
        manifest["works"][1]["members"][0]["group_id"] = "h_shared"
        with self.assertRaisesRegex(ValueError, "Ambiguous canonical mapping"):
            migrate_human_gold(manifest, gold)

    def test_invalid_gold_labels_rejected(self):
        for label in (True, False, -1, 3, 1.0, "2", None):
            with self.subTest(label=label):
                manifest, gold, _ = fixture()
                gold["judgments"]["h_shared"]["relevance"] = label
                with self.assertRaisesRegex(ValueError, "integer 0, 1, or 2"):
                    migrate_human_gold(manifest, gold)

    def test_unfinished_reviews_bad_counts_flags_and_duplicate_union_rejected(self):
        for change in (
            lambda m: m.update(identity_review_complete=False),
            lambda m: m["summary"].update(unresolved_title_pairs=1),
            lambda m: m["summary"].update(llm_only=999),
            lambda m: m["works"][0].update(present_in_human=False),
            lambda m: m["works"].append(deepcopy(m["works"][0])),
        ):
            manifest, _, _ = fixture()
            change(manifest)
            with self.assertRaises(ValueError):
                validate_manifest(manifest)

    def test_formal_count_guard(self):
        manifest, _, _ = fixture()
        manifest["benchmark_id"] = "matrix_completion_v1"
        with self.assertRaisesRegex(ValueError, "must equal"):
            validate_manifest(manifest)


class AnnotationPreparationTests(unittest.TestCase):
    def test_only_llm_only_canonical_works_need_tasks_with_null_labels(self):
        result = prepare_annotation_tasks(*fixture())
        self.assertEqual(result["expected_task_count"], 3)
        self.assertEqual([t["union_id"] for t in result["tasks"]], ["union:a", "union:b", "union:c"])
        self.assertTrue(all(t["label"] is None and t["note"] == "" for t in result["tasks"]))
        self.assertEqual(len(result["tasks"][0]["aliases"]), 2)

    def test_question_and_existing_rubric_preserved_exactly(self):
        manifest, gold, snapshots = fixture()
        tasks = prepare_annotation_tasks(manifest, gold, snapshots)
        self.assertEqual(tasks["research_question"], gold["research_question"])
        self.assertEqual(tasks["relevance_scale"], gold["relevance_scale"])

    def test_nonempty_abstract_preferred_over_year_and_earlier_rank(self):
        task = prepare_annotation_tasks(*fixture())["tasks"][0]
        self.assertEqual(task["title"], "Paper l_abstract")
        self.assertEqual(task["abstract"], "Evidence for l_abstract.")
        self.assertIsNone(task["year"])

    def test_richer_metadata_then_original_rank_break_display_ties(self):
        manifest, gold, snapshots = fixture()
        early = next(c for c in snapshots["llm"]["candidates"] if c["group_id"] == "l_early")
        early["abstract"] = "Another abstract."
        manifest["works"][2]["members"][0]["original_rank"] = 9
        task = prepare_annotation_tasks(manifest, gold, snapshots)["tasks"][0]
        self.assertEqual(task["title"], "Paper l_early")
        # Equal metadata: the first original rank wins, independent of list order.
        early["year"] = None
        manifest["works"][2]["members"][0]["year"] = None
        manifest["works"][2]["members"].reverse()
        self.assertEqual(prepare_annotation_tasks(manifest, gold, snapshots)["tasks"][0]["title"], "Paper l_abstract")

    def test_all_alias_identifiers_and_distinct_abstracts_preserved(self):
        manifest, gold, snapshots = fixture()
        next(c for c in snapshots["llm"]["candidates"] if c["group_id"] == "l_early")["abstract"] = "A different frozen abstract."
        task = prepare_annotation_tasks(manifest, gold, snapshots)["tasks"][0]
        self.assertEqual(set(task["identifiers"]), {"l_early", "l_abstract", "doi:fixture/l_early", "doi:fixture/l_abstract"})
        self.assertEqual({a["abstract"] for a in task["aliases"]}, {"A different frozen abstract.", "Evidence for l_abstract."})
        view = format_task(gold["research_question"], task)
        self.assertIn("A different frozen abstract.", view)
        self.assertIn("Evidence for l_abstract.", view)

    def test_blank_abstract_is_missing(self):
        manifest, gold, snapshots = fixture()
        next(c for c in snapshots["llm"]["candidates"] if c["group_id"] == "l_other")["abstract"] = " \n "
        self.assertIsNone(prepare_annotation_tasks(manifest, gold, snapshots)["tasks"][1]["abstract"])

    def test_preparation_deterministic_despite_member_and_source_order(self):
        manifest, gold, snapshots = fixture()
        expected = prepare_annotation_tasks(manifest, gold, snapshots)
        for ordering in permutations(manifest["works"][:3]):
            permuted = deepcopy(manifest)
            permuted["works"] = list(deepcopy(ordering)) + deepcopy(manifest["works"][3:])
            for work in permuted["works"]:
                work["members"].reverse()
            reversed_snapshots = deepcopy(snapshots)
            for snapshot in reversed_snapshots.values():
                snapshot["candidates"].reverse()
            self.assertEqual(prepare_annotation_tasks(permuted, gold, reversed_snapshots), expected)

    def test_no_ranking_query_or_semantic_metadata_in_task_or_view(self):
        manifest, gold, snapshots = fixture()
        result = prepare_annotation_tasks(manifest, gold, snapshots)
        forbidden = {
            "human_best_rank", "llm_best_rank", "rrf_rank", "rrf_score", "rank", "original_rank",
            "query_id", "query_text", "query_role", "query_rationale", "semantic_score",
            "category", "baseline", "expected_metrics", "hits", "source",
        }

        def assert_safe(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & forbidden)
                for child in value.values():
                    assert_safe(child)
            elif isinstance(value, list):
                for child in value:
                    assert_safe(child)

        assert_safe(result)
        for task in result["tasks"]:
            # Even untrusted additional keys are never printed by the view.
            task.update(query_text="SECRET_QUERY_TEXT", rrf_score="SECRET_RRF_SCORE")
            output = format_task(result["research_question"], task)
            for secret in ("SECRET_", "0.0123456789", "0.987654321"):
                self.assertNotIn(secret, output)

    def test_changed_or_missing_evidence_rejected(self):
        for change in (
            lambda s: s["llm"]["candidates"].pop(),
            lambda s: s["llm"]["candidates"][0].update(paper_id="different"),
            lambda s: s["llm"].update(question="different question"),
            lambda s: s["llm"].update(benchmark_id="different_benchmark"),
            lambda s: s["human"]["candidates"].append(deepcopy(s["human"]["candidates"][0])),
        ):
            manifest, gold, snapshots = fixture()
            change(snapshots)
            with self.assertRaises(ValueError):
                prepare_annotation_tasks(manifest, gold, snapshots)

    def test_sources_not_mutated_or_aliased(self):
        inputs = fixture()
        before = deepcopy(inputs)
        result = prepare_annotation_tasks(*inputs)
        inherited = migrate_human_gold(*inputs[:2])
        result["tasks"][0]["aliases"][0]["title"] = "changed"
        result["relevance_scale"]["2"] = "changed"
        inherited["union:shared"]["inherited_judgments"][0]["relevance"] = 0
        self.assertEqual(inputs, before)


class FinalUnionGoldTests(unittest.TestCase):
    def test_incomplete_tasks_fail(self):
        inputs = fixture()
        with self.assertRaisesRegex(ValueError, "3 required tasks remain unlabeled"):
            build_union_gold(*inputs, prepare_annotation_tasks(*inputs))

    def test_complete_gold_has_exact_union_ids_labels_and_provenance(self):
        inputs = complete_fixture()
        before = deepcopy(inputs)
        result = build_union_gold(*inputs)
        self.assertEqual(set(result["judgments"]), {w["union_id"] for w in inputs[0]["works"]})
        self.assertEqual(len(result["judgments"]), 5)
        self.assertEqual(result["relevance_scale"], inputs[1]["relevance_scale"])
        self.assertEqual(result["provenance"]["inherited_human_gold_provenance"], inputs[1]["provenance"])
        for uid in ("union:shared", "union:human"):
            self.assertEqual(result["judgments"][uid]["provenance"], "inherited_human_gold")
        for task in inputs[3]["tasks"]:
            judgment = result["judgments"][task["union_id"]]
            self.assertEqual(judgment["provenance"], "manual_union_expansion")
            self.assertEqual(judgment["relevance"], task["label"])
            self.assertEqual(judgment["reason"], task["note"])
        self.assertEqual(inputs, before)

    def test_optional_empty_note_does_not_invent_reason(self):
        inputs = complete_fixture()
        inputs[3]["tasks"][0]["note"] = ""
        self.assertEqual(build_union_gold(*inputs)["judgments"]["union:a"]["reason"], "")

    def test_unknown_missing_duplicate_and_shared_task_ids_rejected(self):
        for change in (
            lambda d: d["tasks"][0].update(union_id="union:unknown"),
            lambda d: d["tasks"][0].update(union_id="union:shared"),
            lambda d: d["tasks"].pop(),
            lambda d: d["tasks"].append(deepcopy(d["tasks"][0])),
        ):
            inputs = complete_fixture()
            change(inputs[3])
            with self.assertRaisesRegex(ValueError, "union_id"):
                build_union_gold(*inputs)

    def test_changed_annotation_metadata_and_evidence_rejected(self):
        for change in (
            lambda d: d.update(benchmark_id="other"),
            lambda d: d.update(research_question="other"),
            lambda d: d.update(expected_task_count=999),
            lambda d: d["relevance_scale"].update({"2": "Altered rubric"}),
            lambda d: d["tasks"][0].update(title="other"),
            lambda d: d["tasks"][0]["aliases"][0].update(abstract="invented"),
            lambda d: d["tasks"][0].update(rrf_score=0.5),
        ):
            inputs = complete_fixture()
            change(inputs[3])
            with self.assertRaises(ValueError):
                build_union_gold(*inputs)

    def test_invalid_annotation_label_rejected(self):
        for label in (True, False, 1.0, "2", 3, -1):
            inputs = complete_fixture()
            inputs[3]["tasks"][0]["label"] = label
            with self.assertRaises(ValueError):
                build_union_gold(*inputs)


class FrozenUnionGoldIntegrationTests(unittest.TestCase):
    def test_all_formal_counts_frozen_labels_null_tasks_and_unchanged_inputs(self):
        counts = {
            "matrix_completion_v1": (22, 18, 40),
            "rag_hallucination_v1": (17, 26, 43),
            "cot_reasoning_faithfulness_v1": (22, 35, 57),
        }
        totals = [0, 0, 0]
        # Fail rather than permit any outbound network access during preparation.
        with patch("socket.socket.connect", side_effect=AssertionError("No network allowed")), \
             patch("socket.create_connection", side_effect=AssertionError("No network allowed")):
            for benchmark, expected_counts in counts.items():
                with self.subTest(benchmark=benchmark):
                    paths = expansion_paths(benchmark)
                    protected = (paths.manifest, paths.human_gold, paths.human_candidates, paths.llm_candidates)
                    before = {p: sha256(p.read_bytes()).hexdigest() for p in protected}
                    manifest, gold, snapshots = load_expansion_inputs(benchmark)
                    inherited = migrate_human_gold(manifest, gold)
                    tasks = prepare_annotation_tasks(manifest, gold, snapshots)
                    actual = (len(inherited), len(tasks["tasks"]), len(manifest["works"]))
                    self.assertEqual(actual, expected_counts)
                    self.assertEqual(actual[0] + actual[1], actual[2])
                    totals = [a + b for a, b in zip(totals, actual)]
                    # The checked-in initial task labels are null, but future
                    # Human annotation must not break the offline test suite.
                    saved = read_json(paths.annotations)
                    self.assertEqual(validate_annotations(saved, tasks), saved)
                    self.assertTrue(all(t["label"] is None and t["note"] == "" for t in tasks["tasks"]))
                    new_ids = {w["union_id"] for w in manifest["works"] if not w["present_in_human"] and w["present_in_llm"]}
                    self.assertEqual({t["union_id"] for t in tasks["tasks"]}, new_ids)
                    for work in manifest["works"]:
                        if not work["present_in_human"]:
                            continue
                        for member in work["members"]:
                            if member["source"] == "human":
                                self.assertEqual(inherited[work["union_id"]]["relevance"], gold["judgments"][member["group_id"]]["relevance"])
                    self.assertEqual(before, {p: sha256(p.read_bytes()).hexdigest() for p in protected})
        self.assertEqual(totals, [61, 79, 140])


if __name__ == "__main__":
    unittest.main()
