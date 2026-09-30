"""Validation tests for semantic assessment domain models."""

import unittest

from pydantic import ValidationError

from researchpilot.relevance import RelevanceAssessment


class RelevanceAssessmentTests(unittest.TestCase):
    def test_score_boundaries_are_valid(self) -> None:
        for score in (0.0, 0.5, 1.0):
            with self.subTest(score=score):
                assessment = RelevanceAssessment(
                    category="direct", score=score, reason="Studies the core problem."
                )
                self.assertEqual(assessment.score, score)
                self.assertIsInstance(assessment.score, float)

    def test_invalid_scores_are_rejected(self) -> None:
        for score in (-0.001, 1.001, float("nan"), float("inf"), -float("inf"), None):
            with self.subTest(score=score):
                with self.assertRaises(ValidationError):
                    RelevanceAssessment(category="direct", score=score, reason="Reason")

    def test_empty_or_blank_reason_is_rejected(self) -> None:
        for reason in ("", " ", "\t\n"):
            with self.subTest(reason=reason):
                with self.assertRaises(ValidationError):
                    RelevanceAssessment(category="direct", score=0.8, reason=reason)

    def test_reason_must_be_a_string(self) -> None:
        for reason in (None, 123, False, []):
            with self.subTest(reason=reason):
                with self.assertRaises(ValidationError):
                    RelevanceAssessment(category="direct", score=0.8, reason=reason)

    def test_all_three_categories_are_accepted(self) -> None:
        for category in ("direct", "supporting", "off_target"):
            with self.subTest(category=category):
                assessment = RelevanceAssessment(
                    category=category, score=0.5, reason="Concise evidence."
                )
                self.assertEqual(assessment.category, category)

    def test_unknown_category_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RelevanceAssessment(category="relevant", score=0.5, reason="Reason")

    def test_extra_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RelevanceAssessment(
                category="direct", score=0.8, reason="Reason", rank=1
            )


if __name__ == "__main__":
    unittest.main()
