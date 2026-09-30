"""Manual-merge regressions checked against an independent graph traversal.

The formal fixtures use only frozen candidate pools and approved identity
decisions. No gold labels, retrieval providers or semantic results are used.
"""

from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
from itertools import combinations, permutations
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from eval.identity_alignment import AlignmentMember, IdentityReviews, align_candidates, snapshot_members
from scripts.build_union_alignment import build_union_alignment


def member(source, rank, group, paper, title=None):
    return AlignmentMember(
        source=source, original_rank=rank, group_id=group, paper_id=paper,
        title=title or f"Title for {group}", year=2020, version_count=1,
    )


def review_data(left, right, decision="same_work"):
    return {
        "left_source": left.source, "left_group_id": left.group_id,
        "right_source": right.source, "right_group_id": right.group_id,
        "decision": decision,
    }


def reviews(decisions, benchmark="fixture_v1"):
    return IdentityReviews(benchmark_id=benchmark, version="v1", decisions=decisions)


def record_key(record):
    return (record["source"], record["original_rank"], record["group_id"], record["paper_id"])


def graph_partition(records, decisions):
    """Independent all-pairs adjacency + traversal, without production union-find."""
    neighbors = [set() for _ in records]
    for left, right in combinations(range(len(records)), 2):
        if (records[left]["group_id"] == records[right]["group_id"]
                or records[left]["paper_id"] == records[right]["paper_id"]):
            neighbors[left].add(right)
            neighbors[right].add(left)
    for decision in decisions:
        if decision["decision"] != "same_work":
            continue
        lefts = [index for index, record in enumerate(records)
                 if (record["source"], record["group_id"]) == (decision["left_source"], decision["left_group_id"])]
        rights = [index for index, record in enumerate(records)
                  if (record["source"], record["group_id"]) == (decision["right_source"], decision["right_group_id"])]
        for left in lefts:
            for right in rights:
                neighbors[left].add(right)
                neighbors[right].add(left)
    remaining = set(range(len(records)))
    partition = set()
    while remaining:
        pending = [min(remaining)]
        component = set()
        while pending:
            index = pending.pop()
            if index in component:
                continue
            component.add(index)
            pending.extend(neighbors[index] - component)
        remaining -= component
        partition.add(frozenset(record_key(records[index]) for index in component))
    return partition


class ManualMergeRegressionTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / f"manual-merge-fixture-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self.remove_fixture)
        self.enterContext(redirect_stdout(io.StringIO()))
        for target in ("socket.create_connection", "socket.socket.connect"):
            self.enterContext(patch(target, side_effect=AssertionError("Regression tests must stay offline")))

    def remove_fixture(self):
        target = self.root.resolve()
        if target.parent != Path(__file__).resolve().parent or not target.name.startswith("manual-merge-fixture-"):
            raise AssertionError(f"Unsafe cleanup target: {target}")
        shutil.rmtree(target)

    def assert_manifest(self, manifest, members, decisions=()):
        records = [item.model_dump() for item in members]
        works = manifest["works"]
        flat = [record for work in works for record in work["members"]]
        self.assertEqual(
            Counter(json.dumps(record, sort_keys=True) for record in flat),
            Counter(json.dumps(record, sort_keys=True) for record in records),
        )
        self.assertEqual(
            {frozenset(record_key(record) for record in work["members"]) for work in works},
            graph_partition(records, decisions),
        )
        self.assertEqual(len({work["union_id"] for work in works}), len(works))
        source_ids = {}
        for source in ("human", "llm"):
            ranking = []
            for work in works:
                ranks = [record["original_rank"] for record in work["members"] if record["source"] == source]
                self.assertEqual(work[f"present_in_{source}"], bool(ranks))
                self.assertEqual(work[f"{source}_best_rank"], min(ranks) if ranks else None)
                if ranks:
                    ranking.append((min(ranks), work["union_id"]))
            expected = [union_id for _, union_id in sorted(ranking)]
            self.assertEqual(manifest[f"{source}_union_ranking"], expected)
            self.assertEqual(len(expected), len(set(expected)))
            source_ids[source] = set(expected)
        human, llm = source_ids["human"], source_ids["llm"]
        expected_summary = {
            "raw_human_groups": sum(record["source"] == "human" for record in records),
            "raw_llm_groups": sum(record["source"] == "llm" for record in records),
            "canonical_human_works": len(human), "canonical_llm_works": len(llm),
            "intersection": len(human & llm), "human_only": len(human - llm),
            "llm_only": len(llm - human), "unresolved_title_pairs": len(manifest["review_candidates"]),
        }
        self.assertEqual(manifest["summary"], expected_summary)
        self.assertEqual(len(human), expected_summary["intersection"] + expected_summary["human_only"])
        self.assertEqual(len(llm), expected_summary["intersection"] + expected_summary["llm_only"])
        self.assertEqual(manifest["identity_review_complete"], not manifest["review_candidates"])
        self.assertEqual(manifest["counts_status"], "final" if manifest["identity_review_complete"] else "provisional")
        mapping = {(record["source"], record["group_id"]): work["union_id"]
                   for work in works for record in work["members"]}
        for decision in decisions:
            left = mapping[(decision["left_source"], decision["left_group_id"])]
            right = mapping[(decision["right_source"], decision["right_group_id"])]
            if decision["decision"] == "same_work":
                self.assertEqual(left, right)
                for source in (decision["left_source"], decision["right_source"]):
                    self.assertIn(left, manifest[f"{source}_union_ranking"])
            else:
                self.assertNotEqual(left, right)

    def test_fresh_cross_source_merge_changes_actual_identity_presence_and_both_rankings(self):
        h = member("human", 2, "h", "paper:h", "Same title")
        l = member("llm", 3, "l", "paper:l", "same-title")
        members = [l, member("human", 1, "other-h", "paper:other-h"), h,
                   member("llm", 1, "other-l", "paper:other-l")]
        automatic = align_candidates("fixture_v1", members)
        decisions = [review_data(h, l)]
        final = align_candidates("fixture_v1", members, reviews(decisions))
        self.assert_manifest(automatic, members)
        self.assert_manifest(final, members, decisions)
        self.assertFalse(automatic["identity_review_complete"])
        self.assertTrue(final["identity_review_complete"])
        for field, delta in (("intersection", 1), ("human_only", -1), ("llm_only", -1)):
            self.assertEqual(final["summary"][field], automatic["summary"][field] + delta)
        for field in ("canonical_human_works", "canonical_llm_works"):
            self.assertEqual(final["summary"][field], automatic["summary"][field])
        shared = next(work for work in final["works"] if len(work["members"]) == 2)
        self.assertTrue(shared["present_in_human"] and shared["present_in_llm"])
        for source in ("human", "llm"):
            self.assertNotEqual(final[f"{source}_union_ranking"], automatic[f"{source}_union_ranking"])
            self.assertIn(shared["union_id"], final[f"{source}_union_ranking"])

    def test_cross_source_merge_into_already_shared_work_does_not_add_intersection(self):
        h = member("human", 1, "shared-group", "paper:shared", "Same title")
        copy = member("llm", 5, "shared-group", "paper:shared", "Same title")
        version = member("llm", 2, "other-version", "paper:version", "same-title")
        members = [copy, version, h]
        automatic = align_candidates("fixture_v1", members)
        decisions = [review_data(h, version)]
        final = align_candidates("fixture_v1", members, reviews(decisions))
        self.assert_manifest(final, members, decisions)
        self.assertEqual(automatic["summary"]["intersection"], 1)
        self.assertEqual(final["summary"]["intersection"], 1)
        self.assertEqual(final["summary"]["canonical_llm_works"], 1)
        self.assertEqual(final["summary"]["human_only"], 0)
        self.assertEqual(final["summary"]["llm_only"], 0)
        self.assertEqual(len(final["works"][0]["members"]), 3)
        self.assertEqual(final["works"][0]["llm_best_rank"], 2)
        for source in ("human", "llm"):
            self.assertNotEqual(automatic[f"{source}_union_ranking"], final[f"{source}_union_ranking"])

    def test_within_llm_merge_reduces_work_count_and_keeps_best_original_rank(self):
        first = member("llm", 6, "first", "paper:first", "Same title")
        second = member("llm", 2, "second", "paper:second", "same-title")
        members = [first, member("human", 1, "human-only", "paper:h"), second]
        decisions = [review_data(first, second)]
        final = align_candidates("fixture_v1", members, reviews(decisions))
        self.assert_manifest(final, members, decisions)
        self.assertEqual(final["summary"]["canonical_llm_works"], 1)
        self.assertEqual(final["summary"]["intersection"], 0)
        llm_work = next(work for work in final["works"] if work["present_in_llm"])
        self.assertEqual(llm_work["llm_best_rank"], 2)
        self.assertFalse(llm_work["present_in_human"])
        self.assertEqual([record["original_rank"] for record in llm_work["members"]], [2, 6])

    def test_different_work_resolves_review_without_changing_identity_or_rankings(self):
        h = member("human", 1, "h", "paper:h", "Same title")
        l = member("llm", 1, "l", "paper:l", "same-title")
        automatic = align_candidates("fixture_v1", [h, l])
        decisions = [review_data(h, l, "different_work")]
        final = align_candidates("fixture_v1", [h, l], reviews(decisions))
        self.assert_manifest(final, [h, l], decisions)
        self.assertFalse(automatic["identity_review_complete"])
        self.assertTrue(final["identity_review_complete"])
        for field in ("works", "human_union_ranking", "llm_union_ranking"):
            self.assertEqual(automatic[field], final[field])
        self.assertEqual(final["summary"]["intersection"], 0)

    def test_transitive_manual_graph_and_union_ids_are_independent_of_all_processing_orders(self):
        h = member("human", 2, "h", "paper:h")
        l1 = member("llm", 3, "l1", "paper:l1")
        l2 = member("llm", 1, "l2", "paper:l2")
        members = [h, l1, l2]
        decisions = [review_data(h, l1), review_data(l1, l2)]
        expected = align_candidates("fixture_v1", members, reviews(decisions))
        for ordering in permutations(members):
            for decision_order in permutations(decisions):
                final = align_candidates("fixture_v1", list(ordering), reviews(list(decision_order)))
                self.assert_manifest(final, members, decisions)
                for field in ("works", "summary", "human_union_ranking", "llm_union_ranking"):
                    self.assertEqual(final[field], expected[field])

    def write_fixture(self, benchmark, members, decisions):
        for source, suffix in (("human", "_candidates.json"), ("llm", "_llm_candidates_v1.json")):
            path = self.root / "eval/datasets" / (benchmark + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for item in members:
                if item.source == source:
                    row = item.model_dump(exclude={"source", "original_rank"})
                    rows.append({**row, "rrf_rank": item.original_rank})
            path.write_text(json.dumps({"benchmark_id": benchmark, "question": "Fixture question", "candidates": rows}), encoding="utf-8")
        path = self.root / "eval/alignment" / f"{benchmark}_identity_reviews_v1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(reviews(decisions, benchmark).model_dump_json(), encoding="utf-8")

    def assert_saved_build(self, benchmark, members, decisions):
        result = build_union_alignment(benchmark, root=self.root)
        paths = [self.root / "eval/alignment" / f"{benchmark}_{suffix}_v1.json"
                 for suffix in ("union", "alignment_summary", "review_candidates")]
        saved, summary, pending = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        self.assertEqual(saved, result)
        self.assert_manifest(saved, members, decisions)
        for key, value in saved["summary"].items():
            self.assertEqual(summary[key], value)
        self.assertEqual(pending["review_candidates"], saved["review_candidates"])
        before = {path: path.read_bytes() for path in paths}
        build_union_alignment(benchmark, root=self.root)
        self.assertEqual({path: path.read_bytes() for path in paths}, before)
        return saved

    def test_script_review_json_to_final_partition_summary_and_rankings_is_idempotent(self):
        for already_shared in (False, True):
            with self.subTest(already_shared=already_shared):
                h = member("human", 1, "h", "paper:h", "Same title")
                l = member("llm", 1, "l", "paper:l", "same-title")
                members = [h, l]
                if already_shared:
                    members.append(member("llm", 3, "h", "paper:h", "Same title"))
                decisions = [review_data(h, l)]
                self.write_fixture("fixture_v1", members, decisions)
                saved = self.assert_saved_build("fixture_v1", members, decisions)
                self.assertTrue(saved["identity_review_complete"])
                self.assertEqual(saved["summary"]["intersection"], 1)
                self.assertEqual(len(saved["works"][0]["members"]), len(members))

    def test_all_formal_frozen_pools_match_independent_graph_and_keep_inputs_identical(self):
        repository = Path(__file__).resolve().parents[1]
        # Counts follow the exact/manual graph, including existing cross-source
        # matches before each manual edge; a shared work is counted only once.
        expected = {
            "matrix_completion_v1": (22, 31, 22, 30, 12, 10, 18),
            "rag_hallucination_v1": (17, 38, 17, 38, 12, 5, 26),
            "cot_reasoning_faithfulness_v1": (22, 44, 22, 42, 7, 15, 35),
        }
        fields = ("raw_human_groups", "raw_llm_groups", "canonical_human_works", "canonical_llm_works",
                  "intersection", "human_only", "llm_only")
        for benchmark, counts in expected.items():
            with self.subTest(benchmark=benchmark):
                relative_paths = [Path("eval/datasets") / (benchmark + suffix)
                                  for suffix in ("_candidates.json", "_llm_candidates_v1.json")]
                review_path = Path("eval/alignment") / f"{benchmark}_identity_reviews_v1.json"
                if benchmark != "rag_hallucination_v1":
                    relative_paths.append(review_path)
                before = {}
                for relative in relative_paths:
                    data = (repository / relative).read_bytes()
                    before[relative] = sha256(data).hexdigest()
                    destination = self.root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(data)
                members = []
                for source, relative in zip(("human", "llm"), relative_paths[:2]):
                    members.extend(snapshot_members(json.loads((self.root / relative).read_text(encoding="utf-8")), source))
                decisions = json.loads((self.root / review_path).read_text(encoding="utf-8"))["decisions"] if review_path in relative_paths else []
                saved = self.assert_saved_build(benchmark, members, decisions)
                self.assertEqual(saved["summary"], {**dict(zip(fields, counts)), "unresolved_title_pairs": 0})
                self.assertTrue(saved["identity_review_complete"])
                self.assertEqual(saved["counts_status"], "final")
                for relative in relative_paths:
                    self.assertEqual(sha256((repository / relative).read_bytes()).hexdigest(), before[relative])
                    self.assertEqual(sha256((self.root / relative).read_bytes()).hexdigest(), before[relative])


if __name__ == "__main__":
    unittest.main()
