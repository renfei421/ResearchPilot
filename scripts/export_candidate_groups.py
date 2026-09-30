import json

from researchpilot.openalex_client import OpenAlexClient
from researchpilot.paper_candidate import SearchQuery
from researchpilot.paper_search_service import PaperSearchService
from researchpilot.paper_version_resolver import PaperVersionResolver
from researchpilot.rrf_ranker import RRFRanker


QUESTION = (
    "How does spectral expansion affect deterministic matrix completion "
    "under fixed observation patterns?"
)

QUERIES = [
    SearchQuery(
        query_id="q1",
        text='"matrix completion" AND deterministic',
    ),
    SearchQuery(
        query_id="q2",
        text='"matrix completion" AND "sampling pattern"',
    ),
    SearchQuery(
        query_id="q3",
        text='"matrix completion" AND "spectral gap"',
    ),
]


candidates = PaperSearchService(
    OpenAlexClient()
).search_candidates(
    queries=QUERIES,
    per_query=10,
    year_from=2015,
    year_to=2026,
)

groups = PaperVersionResolver().group_versions(candidates)

ranked = RRFRanker(k=60).rank_groups(groups)


records = []

for rank, item in enumerate(ranked, start=1):
    representative = item.group.members[0].paper

    records.append(
        {
            "rrf_rank": rank,
            "group_id": item.group.group_id,
            "rrf_score": item.rrf_score,
            "title": representative.title,
            "paper_id": representative.paper_id,
            "year": representative.publication_year,
            "abstract": representative.abstract,
            "version_count": len(item.group.members),
            "hits": [
                {
                    "query_id": hit.query_id,
                    "rank": hit.rank,
                }
                for hit in item.fused_hits
            ],
        }
    )


with open(
    "eval/datasets/matrix_completion_candidates_v1.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {
            "question": QUESTION,
            "candidates": records,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )


print(f"Exported {len(records)} groups.")
print("eval/datasets/matrix_completion_candidates_v1.json")