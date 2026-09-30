from researchpilot.openalex_client import OpenAlexClient
from researchpilot.openai_relevance_client import OpenAIRelevanceClient
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

TARGETS = [
    (
        "EXPECTED: direct",
        "A Characterization of Deterministic Sampling Patterns "
        "for Low-Rank Matrix Completion",
    ),
    (
        "EXPECTED: off_target",
        "A survey of matrix completion methods for recommendation systems",
    ),
    (
        "EXPECTED: supporting",
        "Spectral gap in random bipartite biregular graphs and applications",
    ),
]


# --------------------------------
# Existing retrieval pipeline
# --------------------------------

search_service = PaperSearchService(OpenAlexClient())

candidates = search_service.search_candidates(
    queries=QUERIES,
    per_query=10,
    year_from=2015,
    year_to=2026,
)

groups = PaperVersionResolver().group_versions(candidates)

ranked_groups = RRFRanker(k=60).rank_groups(groups)


# --------------------------------
# New LLM component
# --------------------------------

relevance_client = OpenAIRelevanceClient(
    model="gpt-5.6-terra"
)


for expected, target_title in TARGETS:

    ranked_group = next(
        (
            item
            for item in ranked_groups
            if item.group.members[0].paper.title == target_title
        ),
        None,
    )

    if ranked_group is None:
        raise RuntimeError(
            f"Target paper not found: {target_title}"
        )

    paper = ranked_group.group.members[0].paper

    assessment = relevance_client.assess(
        research_question=QUESTION,
        title=paper.title,
        abstract=paper.abstract,
    )

    print()
    print("=" * 80)
    print(expected)
    print("TITLE:", paper.title)
    print("CATEGORY:", assessment.category)
    print("SCORE:", assessment.score)
    print("REASON:", assessment.reason)