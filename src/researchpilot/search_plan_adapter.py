"""Pure conversion from validated plans to executable search queries."""

from researchpilot.paper_candidate import SearchQuery
from researchpilot.query_plan import SearchPlan


def search_plan_to_queries(plan: SearchPlan) -> list[SearchQuery]:
    """Copy ids and text in plan order, without rewriting or planning metadata."""
    return [
        SearchQuery(query_id=planned.query_id, text=planned.text)
        for planned in plan.queries
    ]
