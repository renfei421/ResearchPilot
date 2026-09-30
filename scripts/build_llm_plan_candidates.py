"""Execute frozen LLM plans through the unchanged retrieval pipeline.

Run from the repository root with --benchmarks <id> [<id> ...].
This script never generates plans or evaluates candidates.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from eval.benchmark import PROJECT_ROOT, benchmark_paths
from researchpilot.openalex_client import OpenAlexClient
from researchpilot.paper_search_service import PaperSearchClient, PaperSearchService
from researchpilot.paper_version_resolver import PaperVersionResolver
from researchpilot.query_plan import SearchPlan
from researchpilot.query_planner import PlannerRequest
from researchpilot.rrf_ranker import RRFRanker
from researchpilot.search_plan_adapter import search_plan_to_queries


class FrozenQueryPlan(BaseModel):
    """Validate saved V1 metadata and contracts without invoking a planner."""

    benchmark_id: str = Field(strict=True, min_length=1)
    planner_version: Literal["v1"]
    model: str = Field(strict=True, min_length=1)
    generated_at: str = Field(strict=True, min_length=1)
    request: PlannerRequest
    plan: SearchPlan

    @model_validator(mode="after")
    def validate_request_consistency(self) -> "FrozenQueryPlan":
        if self.request.research_question != self.plan.research_question:
            raise ValueError("Frozen request and plan research_question must match exactly.")
        if len(self.plan.queries) > self.request.max_queries:
            raise ValueError("Frozen plan exceeds request.max_queries.")
        return self


def load_frozen_plan(
    benchmark_id: str, *, root: Path = PROJECT_ROOT
) -> FrozenQueryPlan:
    """Load only the requested plan; existing contracts validate queries and years."""
    benchmark_paths(benchmark_id, root=root)  # Validate the id before forming a path.
    path = root / "eval/plans" / f"{benchmark_id}_llm_plan_v1.json"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen LLM plan not found for {benchmark_id!r}: {path}.")
    frozen = FrozenQueryPlan.model_validate_json(path.read_text(encoding="utf-8"))
    if frozen.benchmark_id != benchmark_id:
        raise ValueError("Frozen plan benchmark_id does not match the requested benchmark.")
    return frozen


def load_per_query(benchmark_id: str, *, root: Path = PROJECT_ROOT) -> int:
    """Use only definition identity and per_query, never its handcrafted queries.

    Do not use load_benchmark here: validating BenchmarkDefinition would consume
    its human queries. Question and year bounds come from the frozen plan.
    """
    path = benchmark_paths(benchmark_id, root=root).definition
    if not path.is_file():
        raise ValueError(f"Unknown benchmark id {benchmark_id!r}: no definition at {path}.")
    definition = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(definition, dict):
        raise ValueError("Benchmark definition must be a JSON object.")
    if definition.get("benchmark_id") != benchmark_id:
        raise ValueError("Definition benchmark_id does not match the requested benchmark.")
    per_query = definition.get("per_query")
    if isinstance(per_query, bool) or not isinstance(per_query, int) or not 1 <= per_query <= 100:
        raise ValueError("per_query must be an integer between 1 and 100.")
    return per_query


def build_llm_plan_candidates(
    benchmark_ids: list[str],
    *,
    client: PaperSearchClient | None = None,
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> list[Path]:
    """Build in supplied order, stopping on failure and retaining earlier outputs.

    Inject a fake PaperSearchClient for offline execution. Only complete results
    are saved; human candidate snapshots are never opened or written.
    """
    if not benchmark_ids:
        raise ValueError("At least one benchmark id is required.")
    if len(set(benchmark_ids)) != len(benchmark_ids):
        raise ValueError("Benchmark ids must be unique.")

    outputs = []
    for index, benchmark_id in enumerate(benchmark_ids, start=1):
        print(f"[{index}/{len(benchmark_ids)}] building LLM candidates for {benchmark_id}", flush=True)
        try:
            frozen = load_frozen_plan(benchmark_id, root=root)
            per_query = load_per_query(benchmark_id, root=root)
            output_path = root / "eval/datasets" / f"{benchmark_id}_llm_candidates_v1.json"
            if output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Frozen LLM snapshot already exists: {output_path}. Use --overwrite to replace it."
                )
            queries = search_plan_to_queries(frozen.plan)
            if client is None:
                client = OpenAlexClient()
            candidates = PaperSearchService(client).search_candidates(
                queries=queries,
                per_query=per_query,
                year_from=frozen.request.year_from,
                year_to=frozen.request.year_to,
            )
            groups = PaperVersionResolver().group_versions(candidates)
            ranked = RRFRanker(k=60).rank_groups(groups)
            records = []
            for rank, item in enumerate(ranked, start=1):
                representative = item.group.members[0].paper
                records.append({
                    "rrf_rank": rank,
                    "group_id": item.group.group_id,
                    "rrf_score": item.rrf_score,
                    "title": representative.title,
                    "paper_id": representative.paper_id,
                    "year": representative.publication_year,
                    "abstract": representative.abstract,
                    "version_count": len(item.group.members),
                    "hits": [hit.model_dump() for hit in item.fused_hits],
                })
            snapshot = {
                "benchmark_id": benchmark_id,
                "source": "llm_query_plan_v1",
                "planner_version": frozen.planner_version,
                "planner_model": frozen.model,
                "question": frozen.plan.research_question,
                "plan_path": f"eval/plans/{benchmark_id}_llm_plan_v1.json",
                "candidates": records,
            }
            serialized = json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Serialize the complete snapshot before replacing any existing file.
            # Exclusive creation also protects a file created while search ran.
            with output_path.open("w" if overwrite else "x", encoding="utf-8") as output:
                output.write(serialized)
            outputs.append(output_path)
        except Exception:
            print(f"LLM candidate execution failed for benchmark {benchmark_id!r}.", file=sys.stderr)
            raise
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", required=True, help="Benchmark ids in run order")
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace existing LLM candidate snapshots"
    )
    args = parser.parse_args(argv)
    build_llm_plan_candidates(args.benchmarks, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
