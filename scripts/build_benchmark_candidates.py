"""Build a frozen candidate pool using the existing retrieval pipeline.

Run from the repository root with --benchmark <benchmark_id>.
Existing snapshots require --overwrite before any search is performed.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from eval.benchmark import PROJECT_ROOT, benchmark_paths, load_benchmark
from researchpilot.openalex_client import OpenAlexClient
from researchpilot.paper_search_service import PaperSearchClient, PaperSearchService
from researchpilot.paper_version_resolver import PaperVersionResolver
from researchpilot.rrf_ranker import RRFRanker


def build_candidates(
    benchmark_id: str,
    *,
    client: PaperSearchClient | None = None,
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build from a definition; inject a fake search client for offline tests."""
    definition = load_benchmark(benchmark_id, root=root)
    output_path = benchmark_paths(benchmark_id, root=root).candidates
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Frozen snapshot already exists: {output_path}. Use --overwrite to replace it."
        )
    if client is None:
        client = OpenAlexClient()
    candidates = PaperSearchService(client).search_candidates(
        queries=definition.queries,
        per_query=definition.per_query,
        year_from=definition.year_from,
        year_to=definition.year_to,
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
                "hits": [hit.model_dump() for hit in item.fused_hits],
            }
        )
    snapshot = {
        "benchmark_id": definition.benchmark_id,
        "question": definition.question,
        "candidates": records,
    }
    serialized = json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents overwriting a file created during search.
    with output_path.open("w" if overwrite else "x", encoding="utf-8") as output:
        output.write(serialized)
    print(f"Exported {len(records)} groups to {output_path}")
    return snapshot


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="Benchmark definition id")
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace an existing snapshot"
    )
    args = parser.parse_args(argv)
    build_candidates(args.benchmark, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
