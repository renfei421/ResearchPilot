"""Unweighted group-level Reciprocal Rank Fusion of original query ranks."""

from researchpilot.paper_candidate import QueryHit
from researchpilot.paper_version import PaperVersionGroup
from researchpilot.ranked_paper_group import RankedPaperGroup


class RRFRanker:
    """Fuse retrieval positions; the score does not judge semantic relevance."""

    def __init__(self, k: int = 60) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative integer; bool is not allowed.")
        self._k = k

    def rank_groups(self, groups: list[PaperVersionGroup]) -> list[RankedPaperGroup]:
        """Score groups using one best hit per query, then sort stably descending.

        Fused queries retain first-seen order. Equal-rank hits retain the first
        occurrence. Input groups, members, and hit objects are never modified.
        """
        ranked_groups = []
        for group in groups:
            best_hits: dict[str, QueryHit] = {}
            for member in group.members:
                for hit in member.hits:
                    best = best_hits.get(hit.query_id)
                    if best is None or hit.rank < best.rank:
                        # Replacing a value preserves the query's dictionary position.
                        best_hits[hit.query_id] = hit

            fused_hits = list(best_hits.values())
            score = sum((1 / (self._k + hit.rank) for hit in fused_hits), 0.0)
            ranked_groups.append(
                RankedPaperGroup(group=group, fused_hits=fused_hits, rrf_score=score)
            )

        # Python's stable sort preserves input order for exactly equal scores.
        return sorted(ranked_groups, key=lambda item: item.rrf_score, reverse=True)
