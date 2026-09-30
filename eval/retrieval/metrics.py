"""Pure metrics for ranked item identifiers and a fixed judgment pool.

Judgments map stable item ids to integer grades: 2 (Direct/Core),
1 (Bridge/Supporting), or 0 (Off-target). k must be a positive integer,
excluding bool. Missing result slots count as non-relevant for precision.
Unjudged or duplicate ids in the evaluated top k raise ValueError.

nDCG uses all supplied judgments for its ideal ranking, including judged items
that were not retrieved. Thus returning fewer items cannot improve the ideal
denominator. Empty rankings and all-zero judgments produce zero scores.
"""

from collections.abc import Mapping, Sequence
from math import log2


def _top_k_relevances(
    ranked_ids: Sequence[str], judgments: Mapping[str, int], k: int
) -> list[int]:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer; bool is not allowed.")
    for relevance in judgments.values():
        if (
            isinstance(relevance, bool)
            or not isinstance(relevance, int)
            or relevance not in (0, 1, 2)
        ):
            raise ValueError("Relevance judgments must be integers 0, 1, or 2.")

    top_ids = ranked_ids[:k]
    if len(set(top_ids)) != len(top_ids):
        raise ValueError("Ranked item ids must be unique within the evaluated top k.")
    for item_id in top_ids:
        if item_id not in judgments:
            raise ValueError(f"Missing relevance judgment for item {item_id!r}.")
    return [judgments[item_id] for item_id in top_ids]


def precision_at_k(
    ranked_ids: Sequence[str], judgments: Mapping[str, int], k: int
) -> float:
    """Count top-k grades >= 1 and divide by k, even for a shorter ranking."""
    relevances = _top_k_relevances(ranked_ids, judgments, k)
    return sum(relevance >= 1 for relevance in relevances) / k


def strict_precision_at_k(
    ranked_ids: Sequence[str], judgments: Mapping[str, int], k: int
) -> float:
    """Count top-k grades == 2 and divide by k, even for a shorter ranking."""
    relevances = _top_k_relevances(ranked_ids, judgments, k)
    return sum(relevance == 2 for relevance in relevances) / k


def _dcg(relevances: Sequence[int]) -> float:
    return sum(
        (
            (2**relevance - 1) / log2(rank + 1)
            for rank, relevance in enumerate(relevances, start=1)
        ),
        0.0,
    )


def ndcg_at_k(
    ranked_ids: Sequence[str], judgments: Mapping[str, int], k: int
) -> float:
    """Use graded gain 2**relevance - 1 and the full judgment pool for IDCG."""
    relevances = _top_k_relevances(ranked_ids, judgments, k)
    ideal_relevances = sorted(judgments.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal_relevances)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(relevances) / ideal_dcg
