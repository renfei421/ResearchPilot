"""Pure canonical identity tests using small, independently authored records."""

from copy import deepcopy
from itertools import permutations
import unittest

from pydantic import ValidationError

from eval.identity_alignment import (
    AlignmentMember, IdentityDecision, IdentityReviews, align_candidates,
    canonical_source_ranking, normalize_alignment_title, snapshot_members,
)


def member(source, rank, group, *, paper=None, title=None):
    return AlignmentMember(
        source=source, original_rank=rank, group_id=group,
        paper_id=paper or f"paper:{group}", title=title or f"Title for {group}",
        year=2020, version_count=1,
    )


def decision(left, right, value):
    return IdentityDecision(
        left_source=left.source, left_group_id=left.group_id,
        right_source=right.source, right_group_id=right.group_id, decision=value,
    )


def align(members, decisions=()):
    return align_candidates(
        "fixture_v1", members,
        IdentityReviews(benchmark_id="fixture_v1", version="v1", decisions=list(decisions)),
    )


def work_for(result, group):
    return next(work for work in result["works"] if any(item["group_id"] == group for item in work["members"]))


class IdentityAlignmentTests(unittest.TestCase):
    def test_exact_group_id_matches_different_representative_papers(self):
        result = align([
            member("human", 1, "shared", paper="paper:a", title="Alpha"),
            member("llm", 1, "shared", paper="paper:b", title="Beta"),
        ])
        self.assertEqual(len(result["works"]), 1)
        self.assertEqual(len(result["works"][0]["members"]), 2)
        self.assertEqual(result["summary"]["intersection"], 1)

    def test_exact_paper_id_matches_different_group_ids(self):
        result = align([member("human", 1, "h", paper="shared"), member("llm", 1, "l", paper="shared")])
        self.assertEqual(len(result["works"]), 1)
        self.assertTrue(result["works"][0]["present_in_human"])
        self.assertTrue(result["works"][0]["present_in_llm"])

    def test_exact_edges_are_transitive_including_repeated_group_within_source(self):
        records = [
            member("human", 1, "h", paper="shared"),
            member("llm", 1, "l", paper="shared"),
            member("llm", 2, "l", paper="another"),
        ]
        result = align(records)
        self.assertEqual(len(result["works"]), 1)
        self.assertEqual(len(result["works"][0]["members"]), 3)
        self.assertEqual(len(result["llm_union_ranking"]), 1)

    def test_title_collision_never_auto_merges_any_source_pair(self):
        for sources in (("human", "llm"), ("human", "human"), ("llm", "llm")):
            with self.subTest(sources=sources):
                records = [
                    member(sources[0], 1, "a", title="Cache-Policies!"),
                    member(sources[1], 2, "b", title="cache policies"),
                ]
                result = align(records)
                self.assertEqual(len(result["works"]), 2)
                self.assertFalse(result["identity_review_complete"])
                self.assertEqual(result["counts_status"], "provisional")
                self.assertEqual(result["summary"]["unresolved_title_pairs"], 1)
                pair = result["review_candidates"][0]
                self.assertEqual(pair["benchmark_id"], "fixture_v1")
                self.assertEqual(pair["normalized_title"], "cache policies")
                self.assertEqual(pair["candidate_a"], records[0].model_dump())
                self.assertEqual(pair["candidate_b"], records[1].model_dump())

    def test_title_normalization_nfkc_case_punctuation_separators_and_whitespace(self):
        self.assertEqual(normalize_alignment_title("  ＣＡＣＨＥ—Policies:\tA\u00a0Study!  "), "cache policies a study")
        self.assertEqual(normalize_alignment_title("Cafe\u0301 / Methods"), normalize_alignment_title("CAFÉ—methods"))
        self.assertEqual(normalize_alignment_title("Straße_Models"), "strasse models")
        self.assertEqual(normalize_alignment_title("x+y"), "x+y")

    def test_no_fuzzy_matching_stemming_or_meaningful_word_removal(self):
        for left, right in (("Cache policy", "Cache policies"), ("Cache latency", "Caching latency"),
                            ("Cache latency", "Corrections to Cache latency"), ("Cache latency", "Latency cache")):
            with self.subTest(left=left, right=right):
                result = align([member("human", 1, "h", title=left), member("llm", 1, "l", title=right)])
                self.assertEqual(len(result["works"]), 2)
                self.assertEqual(result["review_candidates"], [])

    def test_empty_normalized_titles_are_not_identity_evidence(self):
        result = align([member("human", 1, "h", title="---"), member("llm", 1, "l", title="...")])
        self.assertEqual(len(result["works"]), 2)
        self.assertEqual(result["review_candidates"], [])

    def test_one_review_pair_per_title_and_component_not_per_member_pair(self):
        records = [member("human", 1, "a", paper="shared", title="Same Title"),
                   member("llm", 1, "b", paper="shared", title="same-title"),
                   member("llm", 2, "c", title="SAME TITLE")]
        result = align(records)
        self.assertEqual(len(result["works"]), 2)
        self.assertEqual(len(result["review_candidates"]), 1)
        self.assertEqual(result["review_candidates"][0]["candidate_a"]["group_id"], "a")

    def test_explicit_same_work_merges_and_completes_title_review(self):
        a, b = member("human", 1, "a", title="Same"), member("llm", 1, "b", title="same")
        result = align([a, b], [decision(a, b, "same_work")])
        self.assertEqual(len(result["works"]), 1)
        self.assertEqual(result["review_candidates"], [])
        self.assertTrue(result["identity_review_complete"])
        self.assertEqual(result["counts_status"], "final")

    def test_explicit_same_work_can_connect_different_titles_without_semantic_checks(self):
        a, b = member("human", 1, "a"), member("llm", 1, "b")
        self.assertEqual(len(align([a, b], [decision(a, b, "same_work")])["works"]), 1)

    def test_explicit_different_work_keeps_separate_components_and_completes_review(self):
        a, b = member("human", 1, "a", title="Same"), member("llm", 1, "b", title="same")
        result = align([a, b], [decision(a, b, "different_work")])
        self.assertEqual(len(result["works"]), 2)
        self.assertEqual(result["review_candidates"], [])
        self.assertTrue(result["identity_review_complete"])
        self.assertEqual(result["summary"]["intersection"], 0)

    def test_unreviewed_pairs_remain_unresolved_after_partial_review(self):
        a, b, c = member("human", 1, "a", title="Same"), member("llm", 1, "b", title="same"), member("llm", 2, "c", title="same")
        result = align([a, b, c], [decision(a, b, "different_work")])
        self.assertFalse(result["identity_review_complete"])
        self.assertEqual(result["summary"]["unresolved_title_pairs"], 2)

    def test_different_work_applies_to_whole_components_after_manual_merges(self):
        a, b, c = member("human", 1, "a", title="Same"), member("llm", 1, "b", title="same"), member("llm", 2, "c", title="same")
        result = align([a, b, c], [decision(a, b, "different_work"), decision(b, c, "same_work")])
        self.assertEqual(len(result["works"]), 2)
        self.assertTrue(result["identity_review_complete"])

    def test_union_ids_and_entire_manifest_are_independent_of_source_processing_order(self):
        records = [member("llm", 3, "b", paper="shared", title="Same"),
                   member("human", 2, "a", paper="shared", title="same"),
                   member("llm", 1, "c", title="Same")]
        expected = align(records)
        for ordering in permutations(records):
            self.assertEqual(align(list(ordering)), expected)
        self.assertRegex(expected["works"][0]["union_id"], r"^union:[a-f0-9]{64}$")

    def test_union_id_depends_on_membership_not_unrelated_records_or_ranks(self):
        a, b = member("human", 2, "a"), member("llm", 2, "b")
        original = work_for(align([a]), "a")["union_id"]
        changed_rank = member("human", 5, "a")
        self.assertEqual(work_for(align([b, changed_rank]), "a")["union_id"], original)
        self.assertNotEqual(work_for(align([a, b], [decision(a, b, "same_work")]), "a")["union_id"], original)

    def test_canonical_rankings_keep_best_rank_for_both_sources_and_summary_is_correct(self):
        records = [member("human", 8, "h8", paper="shared"), member("llm", 9, "l9", paper="shared"),
                   member("human", 2, "h2", paper="shared"), member("llm", 1, "l1", paper="shared"),
                   member("human", 3, "h_only"), member("llm", 4, "l_only")]
        result = align(records)
        shared = work_for(result, "h2")
        self.assertEqual(shared["human_best_rank"], 2)
        self.assertEqual(shared["llm_best_rank"], 1)
        self.assertEqual(len(shared["members"]), 4)
        self.assertEqual(result["human_union_ranking"], [shared["union_id"], work_for(result, "h_only")["union_id"]])
        self.assertEqual(result["llm_union_ranking"], [shared["union_id"], work_for(result, "l_only")["union_id"]])
        for source in ("human", "llm"):
            self.assertEqual(canonical_source_ranking(list(reversed(result["works"])), source), result[f"{source}_union_ranking"])
        self.assertEqual(result["summary"], {
            "raw_human_groups": 3, "raw_llm_groups": 3, "canonical_human_works": 2,
            "canonical_llm_works": 2, "intersection": 1, "human_only": 1, "llm_only": 1,
            "unresolved_title_pairs": 0,
        })
        self.assertIsNone(work_for(result, "l_only")["human_best_rank"])
        self.assertIsNone(work_for(result, "h_only")["llm_best_rank"])

    def test_input_members_reviews_and_nested_output_do_not_mutate_inputs(self):
        records = [member("human", 1, "h"), member("llm", 1, "l")]
        decisions = [decision(*records, "same_work")]
        before = [item.model_dump() for item in records]
        review_before = decisions[0].model_dump()
        result = align(records, decisions)
        result["works"][0]["members"][0]["title"] = "Changed output"
        result["review_decisions"][0]["note"] = "Changed output"
        self.assertEqual([item.model_dump() for item in records], before)
        self.assertEqual(decisions[0].model_dump(), review_before)

    def test_malformed_review_decisions_are_rejected(self):
        base = decision(member("human", 1, "a"), member("llm", 1, "b"), "same_work").model_dump()
        for change in ({"decision": "maybe"}, {"left_source": "unknown"}, {"right_group_id": ""},
                       {"note": 42}, {"unexpected": "field"}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                IdentityDecision.model_validate({**base, **change})
        for data in ({"benchmark_id": "fixture_v1", "version": "v2", "decisions": []},
                     {"benchmark_id": "fixture_v1", "version": "v1", "decisions": None}):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                IdentityReviews.model_validate(data)

    def test_unknown_review_reference_rejected(self):
        a, missing = member("human", 1, "a"), member("llm", 1, "unknown")
        with self.assertRaisesRegex(ValueError, "unknown candidate group"):
            align([a], [decision(a, missing, "same_work")])

    def test_conflicting_reversed_pair_decisions_rejected(self):
        a, b = member("human", 1, "a"), member("llm", 1, "b")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            align([a, b], [decision(a, b, "same_work"), decision(b, a, "different_work")])

    def test_different_work_contradicting_automatic_evidence_rejected(self):
        a, b = member("human", 1, "a", paper="same"), member("llm", 1, "b", paper="same")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            align([a, b], [decision(a, b, "different_work")])

    def test_transitive_manual_conflicts_rejected_in_every_decision_order(self):
        a, b, c = member("human", 1, "a"), member("llm", 1, "b"), member("llm", 2, "c")
        decisions = [decision(a, b, "same_work"), decision(b, c, "same_work"), decision(a, c, "different_work")]
        for ordering in permutations(decisions):
            with self.assertRaisesRegex(ValueError, "contradicts"):
                align([a, b, c], ordering)

    def test_consistent_duplicate_reviews_are_idempotent(self):
        a, b = member("human", 1, "a", title="Same"), member("llm", 1, "b", title="Same")
        result = align([a, b], [decision(a, b, "different_work"), decision(b, a, "different_work")])
        self.assertTrue(result["identity_review_complete"])
        self.assertEqual(len(result["works"]), 2)

    def test_mismatched_review_benchmark_rejected(self):
        reviews = IdentityReviews(benchmark_id="wrong_v1", version="v1", decisions=[])
        with self.assertRaisesRegex(ValueError, "benchmark_id"):
            align_candidates("fixture_v1", [], reviews)

    def test_empty_pool_and_absent_source_rankings(self):
        empty = align([])
        self.assertTrue(empty["identity_review_complete"])
        self.assertEqual(empty["works"], [])
        self.assertTrue(all(value == 0 for value in empty["summary"].values()))
        self.assertEqual(align([member("human", 1, "h")])["llm_union_ranking"], [])
        with self.assertRaises(ValueError):
            canonical_source_ranking([], "unknown")

    def test_snapshot_conversion_ignores_nonidentity_fields_and_preserves_input(self):
        original = member("human", 2, "h").model_dump()
        record = {key: value for key, value in original.items() if key not in ("source", "original_rank")}
        snapshot = {"candidates": [{**record, "rrf_rank": 2, "rrf_score": 0.2, "abstract": "ignored", "relevance": 2}]}
        before = deepcopy(snapshot)
        self.assertEqual(snapshot_members(snapshot, "human")[0].model_dump(), original)
        self.assertEqual(snapshot, before)
        snapshot["candidates"][0]["relevance"] = 0
        self.assertEqual(snapshot_members(snapshot, "human")[0].model_dump(), original)

    def test_malformed_snapshot_records_and_ambiguous_ranks_are_rejected(self):
        for snapshot in ({}, {"candidates": None}, {"candidates": [{}]}, {"candidates": [None]}):
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                snapshot_members(snapshot, "human")
        with self.assertRaisesRegex(ValueError, "unique"):
            align([member("human", 1, "a"), member("human", 1, "b")])
        for field, value in (("original_rank", True), ("original_rank", 0), ("year", True), ("paper_id", " ")):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                AlignmentMember.model_validate({**member("human", 1, "a").model_dump(), field: value})


if __name__ == "__main__":
    unittest.main()
