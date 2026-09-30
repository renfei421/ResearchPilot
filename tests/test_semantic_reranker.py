"""Offline semantic reranking tests using handcrafted groups and a fake client."""

import unittest

from researchpilot.paper import Paper
from researchpilot.paper_candidate import PaperCandidate, QueryHit
from researchpilot.paper_version import PaperVersionGroup
from researchpilot.ranked_paper_group import RankedPaperGroup
from researchpilot.relevance import RelevanceAssessment, SemanticallyRankedGroup
from researchpilot.semantic_reranker import SemanticReranker


def make_group(
    group_id: str,
    *,
    title: str = "Deterministic matrix completion",
    abstract: str | None = "We study fixed sampling patterns.",
    rrf_score: float = 1 / 61,
) -> RankedPaperGroup:
    hit = QueryHit(query_id="q1", query_text="matrix completion", rank=1)
    candidate = PaperCandidate(
        paper=Paper(
            paper_id=f"openalex:{group_id}",
            title=title,
            abstract=abstract,
            authors=["Example Author"],
            publication_year=2016,
            doi=None,
            citation_count=100,
            source="openalex",
            source_id=group_id,
            open_access_url=None,
        ),
        hits=[hit],
    )
    return RankedPaperGroup(
        group=PaperVersionGroup(group_id=group_id, members=[candidate]),
        fused_hits=[hit],
        rrf_score=rrf_score,
    )


def assessment(score: float, category: str = "direct") -> RelevanceAssessment:
    return RelevanceAssessment(
        category=category, score=score, reason="Evidence in the title and abstract."
    )


class FakeRelevanceClient:
    def __init__(self, outcomes: list[RelevanceAssessment | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, str, str | None]] = []

    def assess(
        self, research_question: str, title: str, abstract: str | None
    ) -> RelevanceAssessment:
        outcome = self.outcomes[len(self.calls)]
        self.calls.append((research_question, title, abstract))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class SemanticRerankerTests(unittest.TestCase):
    def test_one_group_retains_retrieval_result_and_assessment(self) -> None:
        group = make_group("g1")
        result = assessment(0.9)
        client = FakeRelevanceClient([result])

        ranked = SemanticReranker(client).rerank("Research question", [group])

        self.assertEqual(len(ranked), 1)
        self.assertIsInstance(ranked[0], SemanticallyRankedGroup)
        self.assertIs(ranked[0].ranked_group, group)
        self.assertIs(ranked[0].assessment, result)
        self.assertEqual(set(ranked[0].model_dump()), {"ranked_group", "assessment"})
        self.assertEqual(len(client.calls), 1)

    def test_sort_uses_semantic_score_without_numerical_rrf_fusion(self) -> None:
        groups = [
            make_group("low", rrf_score=100.0),
            make_group("high", rrf_score=0.01),
            make_group("middle", rrf_score=0.001),
        ]
        client = FakeRelevanceClient([assessment(0.1), assessment(0.9), assessment(0.5)])

        ranked = SemanticReranker(client).rerank("Question", groups)

        self.assertEqual(
            [item.ranked_group.group.group_id for item in ranked],
            ["high", "middle", "low"],
        )
        self.assertEqual([item.assessment.score for item in ranked], [0.9, 0.5, 0.1])

    def test_semantic_ties_preserve_incoming_order_without_hidden_tiebreaks(self) -> None:
        first = make_group("z-first", title="Z title", rrf_score=0.001)
        second = make_group("a-second", title="A title", rrf_score=100.0)
        second.group.members[0].paper.citation_count = 10000
        second.group.members[0].paper.publication_year = 2026
        for groups in ([first, second], [second, first]):
            with self.subTest(order=[item.group.group_id for item in groups]):
                client = FakeRelevanceClient(
                    [assessment(0.5, "supporting"), assessment(0.5, "direct")]
                )

                ranked = SemanticReranker(client).rerank("Question", groups)

                self.assertEqual([item.ranked_group for item in ranked], groups)

    def test_first_member_is_the_only_representative_even_without_abstract(self) -> None:
        group = make_group("first", title="First version", abstract=None)
        later = make_group("later", title="Later version", abstract="Richer abstract")
        group.group.members.append(later.group.members[0])
        client = FakeRelevanceClient([assessment(0.7)])

        ranked = SemanticReranker(client).rerank("Question", [group])

        self.assertEqual(client.calls, [("Question", "First version", None)])
        self.assertEqual(len(ranked[0].ranked_group.group.members), 2)

    def test_question_title_and_abstract_are_passed_unchanged(self) -> None:
        question = "  How does spectral expansion affect completion?\n"
        title = "  Fixed patterns: α-expansion\n"
        abstract = "  First line.\nSecond line.\t"
        group = make_group("g1", title=title, abstract=abstract)
        client = FakeRelevanceClient([assessment(0.8)])

        SemanticReranker(client).rerank(question, [group])

        self.assertEqual(client.calls, [(question, title, abstract)])

    def test_none_abstract_is_supported(self) -> None:
        group = make_group("g1", abstract=None)
        client = FakeRelevanceClient([assessment(0.4)])

        ranked = SemanticReranker(client).rerank("Question", [group])

        self.assertIsNone(client.calls[0][2])
        self.assertEqual(ranked[0].assessment.score, 0.4)

    def test_input_list_and_nested_objects_are_not_mutated(self) -> None:
        first, second = make_group("first"), make_group("second")
        groups = [first, second]
        before = [item.model_dump() for item in groups]
        members = [item.group.members[0] for item in groups]
        hits = [item.fused_hits[0] for item in groups]
        client = FakeRelevanceClient([assessment(0.1), assessment(0.9)])

        ranked = SemanticReranker(client).rerank("Question", groups)

        self.assertEqual([item.model_dump() for item in groups], before)
        self.assertIs(groups[0], first)
        self.assertIs(groups[1], second)
        self.assertIs(ranked[0].ranked_group, second)
        for index, group in enumerate(groups):
            self.assertIs(group.group.members[0], members[index])
            self.assertIs(group.fused_hits[0], hits[index])
            self.assertIsNone(group.group.members[0].paper.relevance_score)

    def test_first_client_exception_propagates_unchanged(self) -> None:
        failure = RuntimeError("Assessment failed")
        client = FakeRelevanceClient([failure])

        with self.assertRaises(RuntimeError) as caught:
            SemanticReranker(client).rerank("Question", [make_group("g1")])

        self.assertIs(caught.exception, failure)
        self.assertEqual(len(client.calls), 1)

    def test_later_failure_returns_no_partial_results_and_stops_assessing(self) -> None:
        failure = RuntimeError("Second assessment failed")
        client = FakeRelevanceClient([assessment(0.9), failure, assessment(0.5)])
        groups = [make_group("g1"), make_group("g2"), make_group("g3")]
        before = [item.model_dump() for item in groups]
        result = None

        with self.assertRaises(RuntimeError) as caught:
            result = SemanticReranker(client).rerank("Question", groups)

        self.assertIs(caught.exception, failure)
        self.assertIsNone(result)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual([item.model_dump() for item in groups], before)

    def test_empty_input_returns_empty_list_without_assessment(self) -> None:
        client = FakeRelevanceClient([])

        self.assertEqual(SemanticReranker(client).rerank("Question", []), [])
        self.assertEqual(client.calls, [])

    def test_empty_group_is_rejected_before_any_assessment(self) -> None:
        empty = RankedPaperGroup(
            group=PaperVersionGroup(group_id="empty", members=[]),
            fused_hits=[],
            rrf_score=0.0,
        )
        client = FakeRelevanceClient([])

        with self.assertRaisesRegex(ValueError, "at least one member"):
            SemanticReranker(client).rerank("Question", [make_group("valid"), empty])

        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
