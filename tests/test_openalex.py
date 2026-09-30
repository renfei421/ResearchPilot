"""Offline tests using small, handcrafted OpenAlex Work fixtures."""

from copy import deepcopy
import unittest

from researchpilot.openalex import (
    normalize_doi,
    openalex_work_to_paper,
    reconstruct_abstract,
)
from researchpilot.paper import Paper


class NormalizeDoiTests(unittest.TestCase):
    def test_normalizes_prefixes_case_and_whitespace(self) -> None:
        for doi in (
            "https://doi.org/10.1007/s10208-009-9045-5",
            "doi:10.1007/s10208-009-9045-5",
            "10.1007/S10208-009-9045-5",
            "  HTTPS://DOI.ORG/10.1007/S10208-009-9045-5  ",
            " DOI: 10.1007/S10208-009-9045-5 ",
            "http://doi.org/10.1007/S10208-009-9045-5",
            "https://dx.doi.org/10.1007/S10208-009-9045-5",
            "http://dx.doi.org/10.1007/S10208-009-9045-5",
        ):
            with self.subTest(doi=doi):
                self.assertEqual(normalize_doi(doi), "10.1007/s10208-009-9045-5")

    def test_missing_or_blank_doi(self) -> None:
        for doi in (None, "", "   ", "doi:", "https://doi.org/"):
            with self.subTest(doi=doi):
                self.assertIsNone(normalize_doi(doi))


class ReconstructAbstractTests(unittest.TestCase):
    def test_orders_words_by_position(self) -> None:
        index = {"completion.": [2], "Deterministic": [0], "matrix": [1]}

        self.assertEqual(
            reconstruct_abstract(index), "Deterministic matrix completion."
        )

    def test_preserves_repeated_words_and_unsorted_positions(self) -> None:
        index = {
            "evidence": [3, 0],
            "supports": [4, 1],
            "claims;": [2],
            "research.": [5],
        }

        self.assertEqual(
            reconstruct_abstract(index),
            "evidence supports claims; evidence supports research.",
        )

    def test_missing_or_empty_abstract(self) -> None:
        for index in (None, {}, {"unused": []}):
            with self.subTest(index=index):
                self.assertIsNone(reconstruct_abstract(index))


class OpenAlexWorkToPaperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = {
            "id": "https://openalex.org/W2611328865",
            "title": "Deterministic matrix completion",
            "doi": "https://doi.org/10.1007/S10208-009-9045-5",
            "publication_year": 2015,
            "cited_by_count": 12,
            "authorships": [
                {"author": {"display_name": "Ada Example"}},
                {"author": {"display_name": "Lin Example"}},
            ],
            "abstract_inverted_index": {"matrices.": [2], "We": [0], "recover": [1]},
            "open_access": {
                "is_oa": True,
                "oa_url": "https://example.org/paper.pdf",
            },
        }

    def test_maps_metadata_to_paper(self) -> None:
        paper = openalex_work_to_paper(self.work)

        self.assertIsInstance(paper, Paper)
        self.assertEqual(paper.source_id, "W2611328865")
        self.assertEqual(paper.source, "openalex")
        self.assertEqual(paper.title, "Deterministic matrix completion")
        self.assertEqual(paper.publication_year, 2015)
        self.assertEqual(paper.citation_count, 12)
        self.assertEqual(paper.abstract, "We recover matrices.")
        self.assertEqual(paper.doi, "10.1007/s10208-009-9045-5")
        self.assertIsNone(paper.relevance_score)

    def test_extracts_authors_in_source_order(self) -> None:
        paper = openalex_work_to_paper(self.work)

        self.assertEqual(paper.authors, ["Ada Example", "Lin Example"])

    def test_skips_author_entries_without_names(self) -> None:
        self.work["authorships"].extend(
            [{}, {"author": None}, {"author": {}}, {"author": {"display_name": None}}]
        )

        self.assertEqual(
            openalex_work_to_paper(self.work).authors, ["Ada Example", "Lin Example"]
        )

    def test_extracts_open_access_url(self) -> None:
        self.assertEqual(
            openalex_work_to_paper(self.work).open_access_url,
            "https://example.org/paper.pdf",
        )

    def test_closed_access_paper_has_no_open_access_url(self) -> None:
        self.work["open_access"] = {"is_oa": False, "oa_url": None}

        self.assertIsNone(openalex_work_to_paper(self.work).open_access_url)

    def test_doi_paper_id_is_deterministic_across_doi_formats_and_source_ids(self) -> None:
        for source_id, doi in (
            ("W2611328865", "https://doi.org/10.1007/S10208-009-9045-5"),
            ("W123", "doi:10.1007/s10208-009-9045-5"),
            ("W456", "10.1007/S10208-009-9045-5"),
        ):
            with self.subTest(source_id=source_id, doi=doi):
                work = {
                    **self.work,
                    "id": f"https://openalex.org/{source_id}",
                    "doi": doi,
                }
                self.assertEqual(
                    openalex_work_to_paper(work).paper_id,
                    "doi:10.1007/s10208-009-9045-5",
                )

    def test_falls_back_to_openalex_id_without_doi(self) -> None:
        for doi in (None, "", "   "):
            with self.subTest(doi=doi):
                paper = openalex_work_to_paper({**self.work, "doi": doi})
                self.assertEqual(paper.paper_id, "openalex:W2611328865")
                self.assertIsNone(paper.doi)

        del self.work["doi"]
        self.assertEqual(
            openalex_work_to_paper(self.work).paper_id, "openalex:W2611328865"
        )

    def test_missing_optional_metadata_uses_empty_defaults(self) -> None:
        paper = openalex_work_to_paper(
            {"id": "https://openalex.org/W123", "title": "Example"}
        )

        self.assertEqual(paper.authors, [])
        self.assertEqual(paper.citation_count, 0)
        self.assertIsNone(paper.abstract)
        self.assertIsNone(paper.publication_year)
        self.assertIsNone(paper.open_access_url)

    def test_null_optional_metadata_uses_empty_defaults(self) -> None:
        paper = openalex_work_to_paper(
            {
                **self.work,
                "authorships": None,
                "abstract_inverted_index": None,
                "publication_year": None,
                "cited_by_count": None,
                "open_access": None,
            }
        )

        self.assertEqual(paper.authors, [])
        self.assertEqual(paper.citation_count, 0)
        self.assertIsNone(paper.abstract)
        self.assertIsNone(paper.publication_year)
        self.assertIsNone(paper.open_access_url)

    def test_requires_a_nonempty_source_id(self) -> None:
        for source_id in (None, ""):
            with self.subTest(source_id=source_id):
                with self.assertRaisesRegex(ValueError, "non-empty id"):
                    openalex_work_to_paper({**self.work, "id": source_id})

        del self.work["id"]
        with self.assertRaisesRegex(ValueError, "non-empty id"):
            openalex_work_to_paper(self.work)

    def test_missing_title_raises_clear_value_error(self) -> None:
        del self.work["title"]

        with self.assertRaisesRegex(ValueError, "title must be a non-empty string"):
            openalex_work_to_paper(self.work)

    def test_null_non_string_and_blank_titles_raise_clear_value_error(self) -> None:
        for title in (None, 123, False, b"Title", [], {}, "", " \t\n "):
            with self.subTest(title=title):
                with self.assertRaisesRegex(
                    ValueError, "title must be a non-empty string"
                ):
                    openalex_work_to_paper({**self.work, "title": title})

    def test_valid_title_is_preserved(self) -> None:
        title = "  Deterministic matrix completion  "

        self.assertEqual(
            openalex_work_to_paper({**self.work, "title": title}).title, title
        )

    def test_does_not_mutate_input_and_is_repeatable(self) -> None:
        original = deepcopy(self.work)

        first = openalex_work_to_paper(self.work)
        second = openalex_work_to_paper(self.work)

        self.assertEqual(self.work, original)
        self.assertEqual(first, second)
