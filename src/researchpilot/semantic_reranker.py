"""Semantic reranking without combining semantic and retrieval scores."""

from researchpilot.ranked_paper_group import RankedPaperGroup
from researchpilot.relevance import RelevanceClient, SemanticallyRankedGroup


class SemanticReranker:
    """Assess the first member of each group and sort by semantic score only."""

    def __init__(self, client: RelevanceClient) -> None:
        self._client = client

    def rerank(
        self,
        question: str,
        ranked_groups: list[RankedPaperGroup],
    ) -> list[SemanticallyRankedGroup]:
        """Preserve incoming RRF order for ties and retain original objects.

        Empty input returns an empty list. Every group must have a member;
        this is checked before assessment starts. Assessment errors propagate
        immediately, and results are returned only after all groups succeed.
        The question, title, and abstract are passed through unchanged.
        """
        for ranked_group in ranked_groups:
            if not ranked_group.group.members:
                raise ValueError("Each PaperVersionGroup must have at least one member.")

        assessed = []
        for ranked_group in ranked_groups:
            paper = ranked_group.group.members[0].paper
            assessment = self._client.assess(
                research_question=question,
                title=paper.title,
                abstract=paper.abstract,
            )
            assessed.append(
                SemanticallyRankedGroup(
                    ranked_group=ranked_group,
                    assessment=assessment,
                )
            )

        # Python's stable sort preserves the incoming order for equal scores.
        return sorted(assessed, key=lambda item: item.assessment.score, reverse=True)
