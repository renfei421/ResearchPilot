"""Sequential multi-query candidate aggregation, separate from HTTP behavior."""

from typing import Protocol

from researchpilot.paper import Paper
from researchpilot.paper_candidate import PaperCandidate, QueryHit, SearchQuery


class PaperSearchClient(Protocol):
    """The synchronous search interface required from an injected client."""

    def search_works(
        self,
        query: str,
        per_page: int = 5,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[Paper]: ...


class PaperSearchService:
    """Combine query results by exact paper_id while retaining provenance."""

    def __init__(self, client: PaperSearchClient) -> None:
        self._client = client

    def search_candidates(
        self,
        queries: list[SearchQuery],
        per_query: int = 10,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[PaperCandidate]:
        """Return candidates in first-seen order, preserving original ranks.

        Queries must be validated SearchQuery instances with unique ids. Year
        validation remains with the client. Search errors propagate immediately;
        the local candidate pool is returned only after every search succeeds.
        """
        if not queries:
            raise ValueError("queries must not be empty.")
        if (
            isinstance(per_query, bool)
            or not isinstance(per_query, int)
            or not 1 <= per_query <= 100
        ):
            raise ValueError("per_query must be an integer between 1 and 100.")

        query_ids = [query.query_id for query in queries]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("query_id values must be unique across the request.")

        candidates: dict[str, PaperCandidate] = {}
        for query in queries:
            papers = self._client.search_works(
                query=query.text,
                per_page=per_query,
                year_from=year_from,
                year_to=year_to,
            )
            seen_in_query: set[str] = set()
            for rank, paper in enumerate(papers, start=1):
                # Results are visited in rank order, so the first hit is best.
                if paper.paper_id in seen_in_query:
                    continue
                seen_in_query.add(paper.paper_id)

                if paper.paper_id not in candidates:
                    candidates[paper.paper_id] = PaperCandidate(paper=paper, hits=[])
                candidates[paper.paper_id].hits.append(
                    QueryHit(
                        query_id=query.query_id,
                        query_text=query.text,
                        rank=rank,
                    )
                )

        # Dictionary insertion order preserves each paper's first appearance.
        return list(candidates.values())
