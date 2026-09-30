"""Handcrafted group-level RRF tests without search clients or network I/O."""

import unittest

from researchpilot.paper import Paper
from researchpilot.paper_candidate import PaperCandidate, QueryHit
from researchpilot.paper_version import PaperVersionGroup
from researchpilot.ranked_paper_group import RankedPaperGroup
from researchpilot.rrf_ranker import RRFRanker


def make_group(
    group_id: str, *versions: list[tuple[str, int]]
) -> PaperVersionGroup:
    members = []
    for index, hits in enumerate(versions, start=1):
        source_id = f"{group_id}-{index}"
        members.append(
            PaperCandidate(
                paper=Paper(
                    paper_id=f"openalex:{source_id}",
                    title=f"Paper {source_id}",
                    abstract=None,
                    authors=["Example Author"],
                    publication_year=2016,
                    doi=None,
                    citation_count=0,
                    source="openalex",
                    source_id=source_id,
                    open_access_url=None,
                ),
                hits=[
                    QueryHit(
                        query_id=query_id, query_text=f"Search {query_id}", rank=rank
                    )
                    for query_id, rank in hits
                ],
            )
        )
    return PaperVersionGroup(group_id=group_id, members=members)


class RRFRankerTests(unittest.TestCase):
    def test_one_group_with_one_hit_uses_default_k_sixty(self) -> None:
        group = make_group("g1", [("q1", 1)])

        ranked = RRFRanker().rank_groups([group])

        self.assertEqual(len(ranked), 1)
        self.assertIsInstance(ranked[0], RankedPaperGroup)
        self.assertIs(ranked[0].group, group)
        self.assertEqual(ranked[0].fused_hits, group.members[0].hits)
        self.assertEqual(ranked[0].rrf_score, 1 / 61)
        self.assertIsInstance(ranked[0].rrf_score, float)

    def test_several_queries_match_hand_calculated_score(self) -> None:
        group = make_group("g1", [("q1", 1), ("q2", 2), ("q3", 4)])

        ranked = RRFRanker(k=0).rank_groups([group])

        # 1/1 + 1/2 + 1/4 = 1.75 exactly in binary floating point.
        self.assertEqual(ranked[0].rrf_score, 1.75)
        self.assertEqual(len(ranked[0].fused_hits), 3)

    def test_versions_share_only_the_best_hit_per_query(self) -> None:
        group = make_group(
            "g1",
            [("q1", 1), ("q2", 1), ("q3", 2)],
            [("q1", 8), ("q2", 5)],
        )

        ranked = RRFRanker().rank_groups([group])[0]

        self.assertEqual(
            [(hit.query_id, hit.rank) for hit in ranked.fused_hits],
            [("q1", 1), ("q2", 1), ("q3", 2)],
        )
        # Two contributions of 1/61 and one of 1/62, not five contributions.
        self.assertAlmostEqual(ranked.rrf_score, 0.04891591750396616, places=15)

    def test_duplicate_versions_do_not_receive_extra_credit(self) -> None:
        original = make_group("original", [("q1", 2), ("q2", 5)])
        duplicated = make_group(
            "duplicated", [("q1", 2), ("q2", 5)], [("q1", 2), ("q2", 5)]
        )

        ranked = RRFRanker().rank_groups([original, duplicated])

        self.assertEqual(ranked[0].rrf_score, ranked[1].rrf_score)
        self.assertEqual([len(item.fused_hits) for item in ranked], [2, 2])
        self.assertEqual(
            [item.group.group_id for item in ranked], ["original", "duplicated"]
        )

    def test_duplicate_query_hits_within_one_member_also_fuse(self) -> None:
        group = make_group("g1", [("q1", 8), ("q2", 4), ("q1", 2)])

        ranked = RRFRanker().rank_groups([group])[0]

        self.assertEqual(
            [(hit.query_id, hit.rank) for hit in ranked.fused_hits],
            [("q1", 2), ("q2", 4)],
        )

    def test_custom_valid_k(self) -> None:
        group = make_group("g1", [("q1", 1)])

        for k, expected in ((0, 1.0), (4, 0.2), (99, 0.01)):
            with self.subTest(k=k):
                self.assertEqual(
                    RRFRanker(k=k).rank_groups([group])[0].rrf_score, expected
                )

    def test_negative_k_is_rejected(self) -> None:
        for k in (-1, -60):
            with self.subTest(k=k):
                with self.assertRaisesRegex(
                    ValueError, "k must be a non-negative integer"
                ):
                    RRFRanker(k=k)

    def test_bool_k_is_rejected(self) -> None:
        for k in (True, False):
            with self.subTest(k=k):
                with self.assertRaisesRegex(ValueError, "bool is not allowed"):
                    RRFRanker(k=k)

    def test_non_integer_k_is_rejected(self) -> None:
        for k in (None, "60", 60.0, [], {}):
            with self.subTest(k=k):
                with self.assertRaisesRegex(
                    ValueError, "k must be a non-negative integer"
                ):
                    RRFRanker(k=k)

    def test_groups_sort_by_descending_score(self) -> None:
        low = make_group("low", [("q1", 5)])
        high = make_group("high", [("q1", 10), ("q2", 10)])
        middle = make_group("middle", [("q1", 1)])

        ranked = RRFRanker().rank_groups([low, high, middle])

        self.assertEqual(
            [item.group.group_id for item in ranked], ["high", "middle", "low"]
        )
        self.assertGreater(ranked[0].rrf_score, ranked[1].rrf_score)
        self.assertGreater(ranked[1].rrf_score, ranked[2].rrf_score)

    def test_equal_scores_preserve_input_order_without_metadata_tiebreaks(self) -> None:
        first = make_group("z-first", [("q1", 1)])
        second = make_group("a-second", [("q2", 1)])
        first.members[0].paper.title = "Z title"
        first.members[0].paper.doi = "10.1000/z"
        first.members[0].paper.publication_year = 2010
        second.members[0].paper.title = "A title"
        second.members[0].paper.doi = "10.1000/a"
        second.members[0].paper.publication_year = 2026
        second.members[0].paper.citation_count = 1000

        for groups in ([first, second], [second, first]):
            with self.subTest(order=[group.group_id for group in groups]):
                ranked = RRFRanker().rank_groups(groups)
                self.assertEqual(ranked[0].rrf_score, ranked[1].rrf_score)
                self.assertEqual([item.group for item in ranked], groups)

    def test_better_hit_replacement_keeps_first_seen_query_order(self) -> None:
        group = make_group(
            "g1",
            [("q2", 9), ("q1", 10), ("q3", 4)],
            [("q4", 3), ("q1", 2), ("q2", 1)],
        )

        fused = RRFRanker().rank_groups([group])[0].fused_hits

        self.assertEqual(
            [(hit.query_id, hit.rank) for hit in fused],
            [("q2", 1), ("q1", 2), ("q3", 4), ("q4", 3)],
        )
        self.assertEqual(len({hit.query_id for hit in fused}), len(fused))
        self.assertIs(fused[0], group.members[1].hits[2])
        self.assertIs(fused[1], group.members[1].hits[1])

    def test_equal_rank_hits_keep_first_occurrence(self) -> None:
        group = make_group("g1", [("q1", 2)], [("q1", 2)])

        fused = RRFRanker().rank_groups([group])[0].fused_hits

        self.assertEqual(len(fused), 1)
        self.assertIs(fused[0], group.members[0].hits[0])

    def test_zero_hit_groups_have_float_zero_score(self) -> None:
        groups = [make_group("empty-members"), make_group("empty-hits", [])]

        ranked = RRFRanker().rank_groups(groups)

        self.assertEqual([item.group for item in ranked], groups)
        for item in ranked:
            self.assertEqual(item.rrf_score, 0.0)
            self.assertIsInstance(item.rrf_score, float)
            self.assertEqual(item.fused_hits, [])

    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(RRFRanker().rank_groups([]), [])

    def test_input_groups_members_and_hits_are_not_mutated(self) -> None:
        low = make_group("low", [("q1", 10)])
        high = make_group("high", [("q1", 8), ("q2", 2)], [("q1", 1)])
        groups = [low, high]
        snapshots = [group.model_dump() for group in groups]
        members = list(high.members)
        original_hit_lists = [member.hits for member in high.members]
        original_hits = [hit for member in high.members for hit in member.hits]

        ranked = RRFRanker().rank_groups(groups)

        self.assertEqual([group.model_dump() for group in groups], snapshots)
        self.assertIs(groups[0], low)
        self.assertIs(groups[1], high)
        self.assertIs(ranked[0].group, high)
        for member, original, hit_list in zip(
            high.members, members, original_hit_lists
        ):
            self.assertIs(member, original)
            self.assertIs(member.hits, hit_list)
            self.assertIsNot(ranked[0].fused_hits, hit_list)
        for hit, original in zip(
            [hit for member in high.members for hit in member.hits], original_hits
        ):
            self.assertIs(hit, original)
        self.assertEqual(high.members[0].hits[0].rank, 8)

    def test_repeated_calls_do_not_accumulate_hits_or_scores(self) -> None:
        group = make_group("g1", [("q1", 2)], [("q1", 1), ("q2", 3)])
        ranker = RRFRanker()

        first = ranker.rank_groups([group])
        second = ranker.rank_groups([group])

        self.assertEqual(first, second)
        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[0].fused_hits, second[0].fused_hits)
