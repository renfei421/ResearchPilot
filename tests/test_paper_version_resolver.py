"""Handcrafted version-grouping fixtures; no network or real-client behavior."""

import unittest

from researchpilot.paper import Paper
from researchpilot.paper_candidate import PaperCandidate, QueryHit
from researchpilot.paper_version import PaperVersionGroup
from researchpilot.paper_version_resolver import (
    PaperVersionResolver,
    normalize_author_signature,
    normalize_title,
)


TITLE = (
    "A Characterization of Deterministic Sampling Patterns "
    "for Low-Rank Matrix Completion"
)


def make_candidate(
    source_id: str,
    *,
    title: str = TITLE,
    authors: list[str] | None = None,
    year: int | None = 2016,
    doi: str | None = None,
) -> PaperCandidate:
    return PaperCandidate(
        paper=Paper(
            paper_id=f"doi:{doi}" if doi else f"openalex:{source_id}",
            title=title,
            abstract=None,
            authors=["Ada Example", "Lin Example"] if authors is None else authors,
            publication_year=year,
            doi=doi,
            citation_count=0,
            source="openalex",
            source_id=source_id,
            open_access_url=None,
        ),
        hits=[
            QueryHit(
                query_id=f"q-{source_id}", query_text="matrix completion", rank=1
            )
        ],
    )


class VersionNormalizationTests(unittest.TestCase):
    def test_title_normalizes_case_and_whitespace(self) -> None:
        self.assertEqual(
            normalize_title("  MATRIX \t completion\n"), "matrix completion"
        )

    def test_title_normalizes_punctuation_and_dash_variants(self) -> None:
        for title in (
            "Low-rank: matrix completion.",
            "Low–rank, matrix completion!",
            "“Low—rank” (matrix completion)",
            "Low‑rank; matrix completion",
            "low rank matrix completion",
        ):
            with self.subTest(title=title):
                self.assertEqual(normalize_title(title), "low rank matrix completion")

    def test_unicode_compatibility_and_combining_characters(self) -> None:
        self.assertEqual(normalize_title("Ｃａｆé—Ｍａｔｒｉｘ"), "café matrix")
        self.assertEqual(normalize_title("Cafe\u0301 matrix"), "café matrix")

    def test_title_retains_meaningful_words_and_math_symbols(self) -> None:
        self.assertNotEqual(
            normalize_title(f"Corrections to {TITLE}"), normalize_title(TITLE)
        )
        self.assertNotEqual(
            normalize_title("C++ methods"), normalize_title("C methods")
        )

    def test_author_signature_normalizes_unicode_case_and_whitespace(self) -> None:
        self.assertEqual(
            normalize_author_signature(["  Jose\u0301  GARCÍA ", "Ａｄａ\tExample"]),
            ("josé garcía", "ada example"),
        )

    def test_author_signature_preserves_order_punctuation_and_full_names(self) -> None:
        signature = normalize_author_signature(["J. Example", "Lin Example"])
        for authors in (
            ["Lin Example", "J. Example"],
            ["J Example", "Lin Example"],
            ["John Example", "Lin Example"],
        ):
            with self.subTest(authors=authors):
                self.assertNotEqual(signature, normalize_author_signature(authors))


class PaperVersionResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = PaperVersionResolver()

    def test_empty_input_returns_no_groups(self) -> None:
        self.assertEqual(self.resolver.group_versions([]), [])

    def test_single_candidate(self) -> None:
        candidate = make_candidate("W1")

        groups = self.resolver.group_versions([candidate])

        self.assertEqual(len(groups), 1)
        self.assertIsInstance(groups[0], PaperVersionGroup)
        self.assertIs(groups[0].members[0], candidate)
        self.assertTrue(groups[0].group_id)

    def test_unrelated_titles_stay_separate(self) -> None:
        first = make_candidate("W1")
        second = make_candidate("W2", title="A different research problem")

        groups = self.resolver.group_versions([first, second])

        self.assertEqual([group.members for group in groups], [[first], [second]])

    def test_case_variants_with_distinct_dois_and_nearby_years_group(self) -> None:
        journal = make_candidate("W1", doi="10.1109/jstsp.2016.2537145", year=2016)
        conference = make_candidate(
            "W2",
            title=TITLE.lower(),
            doi="10.1109/allerton.2015.7447128",
            year=2015,
        )

        groups = self.resolver.group_versions([journal, conference])

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].members, [journal, conference])
        self.assertNotEqual(journal.paper.doi, conference.paper.doi)

    def test_whitespace_punctuation_and_hyphen_variants_group(self) -> None:
        first = make_candidate("W1", title="Low-rank matrix completion")
        second = make_candidate("W2", title="  “LOW—RANK”: matrix\t completion! ")

        groups = self.resolver.group_versions([first, second])

        self.assertEqual([group.members for group in groups], [[first, second]])

    def test_different_authors_stay_separate(self) -> None:
        first = make_candidate("W1")
        second = make_candidate("W2", authors=["Different Author", "Lin Example"])

        self.assertEqual(len(self.resolver.group_versions([first, second])), 2)

    def test_reordered_authors_stay_separate(self) -> None:
        first = make_candidate("W1")
        second = make_candidate("W2", authors=["Lin Example", "Ada Example"])

        self.assertEqual(len(self.resolver.group_versions([first, second])), 2)

    def test_year_difference_boundary_is_inclusive_and_symmetric(self) -> None:
        for year in (2014, 2016, 2018):
            with self.subTest(year=year):
                first = make_candidate("W1", year=2016)
                second = make_candidate("W2", year=year)
                self.assertEqual(len(self.resolver.group_versions([first, second])), 1)

    def test_years_more_than_two_apart_stay_separate(self) -> None:
        for year in (2013, 2019):
            with self.subTest(year=year):
                first = make_candidate("W1", year=2016)
                second = make_candidate("W2", year=year)
                self.assertEqual(len(self.resolver.group_versions([first, second])), 2)

    def test_intermediate_year_does_not_bridge_incompatible_versions(self) -> None:
        middle = make_candidate("W1", year=2017)
        earlier = make_candidate("W2", year=2015)
        later = make_candidate("W3", year=2019)

        groups = self.resolver.group_versions([middle, earlier, later])

        self.assertEqual(
            [group.members for group in groups], [[middle, earlier], [later]]
        )

    def test_joins_first_compatible_group_without_combining_existing_groups(self) -> None:
        earlier = make_candidate("W1", year=2015)
        later = make_candidate("W2", year=2018)
        middle = make_candidate("W3", year=2016)

        groups = self.resolver.group_versions([earlier, later, middle])

        self.assertEqual(
            [group.members for group in groups], [[earlier, middle], [later]]
        )

    def test_correction_title_stays_separate(self) -> None:
        original = make_candidate("W1")
        correction = make_candidate("W2", title=f"Corrections to “{TITLE}”")

        groups = self.resolver.group_versions([original, correction])

        self.assertEqual(
            [group.members for group in groups], [[original], [correction]]
        )

    def test_missing_year_prevents_grouping_in_either_direction(self) -> None:
        for first_year, second_year in ((None, 2016), (2016, None), (None, None)):
            with self.subTest(first_year=first_year, second_year=second_year):
                first = make_candidate("W1", year=first_year)
                second = make_candidate("W2", year=second_year)
                groups = self.resolver.group_versions([first, second])
                self.assertEqual(
                    [group.members for group in groups], [[first], [second]]
                )
                self.assertNotEqual(groups[0].group_id, groups[1].group_id)

    def test_missing_or_blank_author_information_remains_separate(self) -> None:
        for authors in ([], [" "], ["Ada Example", ""]):
            with self.subTest(authors=authors):
                first = make_candidate("W1", authors=authors)
                second = make_candidate("W2", authors=authors)
                self.assertEqual(len(self.resolver.group_versions([first, second])), 2)

    def test_empty_normalized_titles_remain_separate(self) -> None:
        first = make_candidate("W1", title="...")
        second = make_candidate("W2", title=" ")

        self.assertEqual(len(self.resolver.group_versions([first, second])), 2)

    def test_unicode_title_and_author_variants_group(self) -> None:
        first = make_candidate("W1", title="Ｃａｆé—Ｍａｔｒｉｘ", authors=["José García"])
        second = make_candidate(
            "W2", title="Cafe\u0301 matrix", authors=["  JOSE\u0301  GARCÍA "]
        )

        groups = self.resolver.group_versions([first, second])

        self.assertEqual([group.members for group in groups], [[first, second]])

    def test_all_original_candidates_and_hits_remain_available(self) -> None:
        first = make_candidate("W1")
        unrelated = make_candidate("W2", title="Unrelated work")
        version = make_candidate("W3", year=2015)
        candidates = [first, unrelated, version]
        snapshots = [candidate.model_dump() for candidate in candidates]

        groups = self.resolver.group_versions(candidates)

        members = [member for group in groups for member in group.members]
        self.assertCountEqual(
            [id(member) for member in members], [id(c) for c in candidates]
        )
        self.assertEqual(
            [candidate.model_dump() for candidate in candidates], snapshots
        )
        self.assertEqual(
            [id(c) for c in candidates], [id(first), id(unrelated), id(version)]
        )
        self.assertIs(groups[0].members[0].hits[0], first.hits[0])
        self.assertIsNot(groups[0].members, candidates)

    def test_first_seen_group_and_member_order(self) -> None:
        first = make_candidate("W1", title="First work", year=2016)
        second = make_candidate("W2", title="Second work", year=2014)
        second_version = make_candidate("W3", title="Second work", year=2015)
        first_version = make_candidate("W4", title="First work", year=2015)

        groups = self.resolver.group_versions(
            [first, second, second_version, first_version]
        )

        self.assertEqual(
            [group.members for group in groups],
            [[first, first_version], [second, second_version]],
        )

    def test_group_ids_are_deterministic_for_rebuilt_objects(self) -> None:
        candidates = [make_candidate("W1"), make_candidate("W2", year=2015)]
        rebuilt = [PaperCandidate.model_validate(c.model_dump()) for c in candidates]

        first = self.resolver.group_versions(candidates)
        second = PaperVersionResolver().group_versions(rebuilt)

        self.assertEqual(first[0].group_id, second[0].group_id)

    def test_group_id_is_stable_when_same_members_are_reordered(self) -> None:
        candidates = [make_candidate("W1"), make_candidate("W2", year=2015)]

        forward = self.resolver.group_versions(candidates)
        reverse = self.resolver.group_versions(list(reversed(candidates)))

        self.assertEqual(forward[0].group_id, reverse[0].group_id)
        self.assertEqual(reverse[0].members, list(reversed(candidates)))

    def test_year_separated_groups_have_distinct_deterministic_ids(self) -> None:
        candidates = [make_candidate("W1", year=2010), make_candidate("W2", year=2020)]

        first = self.resolver.group_versions(candidates)
        repeated = self.resolver.group_versions(candidates)

        self.assertNotEqual(first[0].group_id, first[1].group_id)
        self.assertEqual(
            [group.group_id for group in first], [group.group_id for group in repeated]
        )
