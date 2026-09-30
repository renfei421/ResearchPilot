"""Focused tests for the normalized Paper data contract."""

import unittest

from pydantic import ValidationError

from researchpilot.paper import Paper


class PaperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = {
            "paper_id": "openalex:W123",
            "title": "A small example",
            "abstract": None,
            "authors": [],
            "publication_year": None,
            "doi": None,
            "citation_count": 0,
            "source": "openalex",
            "source_id": "W123",
            "open_access_url": None,
        }

    def test_nullable_metadata_and_default_relevance_score(self) -> None:
        paper = Paper(**self.metadata)

        self.assertEqual(
            paper.model_dump(), {**self.metadata, "relevance_score": None}
        )

    def test_invalid_citation_count_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Paper(**{**self.metadata, "citation_count": "many"})

    def test_nonnegative_citation_counts_are_accepted(self) -> None:
        for citation_count in (0, 12):
            with self.subTest(citation_count=citation_count):
                paper = Paper(**{**self.metadata, "citation_count": citation_count})
                self.assertEqual(paper.citation_count, citation_count)

    def test_negative_citation_count_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Paper(**{**self.metadata, "citation_count": -1})

    def test_relevance_score_accepts_none_and_inclusive_unit_interval(self) -> None:
        for score in (None, 0.0, 0.5, 1.0):
            with self.subTest(score=score):
                paper = Paper(**{**self.metadata, "relevance_score": score})
                self.assertEqual(paper.relevance_score, score)

    def test_out_of_range_relevance_score_is_rejected(self) -> None:
        for score in (-0.01, 1.01):
            with self.subTest(score=score):
                with self.assertRaises(ValidationError):
                    Paper(**{**self.metadata, "relevance_score": score})
