"""Freeze Query Planner V1 outputs without executing or evaluating searches.

Run from the repository root with --benchmarks <id> [<id> ...].
Existing frozen plans require an explicit --overwrite.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from eval.benchmark import PROJECT_ROOT, benchmark_paths
from researchpilot.openai_query_planner import OpenAIQueryPlanner
from researchpilot.query_planner import PlannerRequest, QueryPlanner


DEFAULT_MODEL = "gpt-5.6-terra"


def load_planner_request(
    benchmark_id: str, *, root: Path = PROJECT_ROOT
) -> PlannerRequest:
    """Read a definition, using only identity, question and year bounds.

    Deliberately do not load BenchmarkDefinition: its handcrafted queries and
    per_query setting are irrelevant to independent LLM planning.
    """
    path = benchmark_paths(benchmark_id, root=root).definition
    if not path.is_file():
        raise ValueError(f"Unknown benchmark id {benchmark_id!r}: no definition at {path}.")
    definition = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(definition, dict):
        raise ValueError("Benchmark definition must be a JSON object.")
    required = ("benchmark_id", "question", "year_from", "year_to")
    missing = [field for field in required if field not in definition]
    if missing:
        raise ValueError(f"Benchmark definition is missing fields: {', '.join(missing)}.")
    if definition["benchmark_id"] != benchmark_id:
        raise ValueError("Definition benchmark_id does not match the requested benchmark.")
    return PlannerRequest(
        research_question=definition["question"],
        year_from=definition["year_from"],
        year_to=definition["year_to"],
        max_queries=5,
    )


def generate_query_plans(
    benchmark_ids: list[str],
    *,
    model: str = DEFAULT_MODEL,
    planner: QueryPlanner | None = None,
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> list[Path]:
    """Generate in supplied order; inject a QueryPlanner for offline use.

    Stop on the first error, retaining earlier completed files. The recorded
    model is the supplied model name, also used to construct the live planner.
    """
    if not benchmark_ids:
        raise ValueError("At least one benchmark id is required.")
    if len(set(benchmark_ids)) != len(benchmark_ids):
        raise ValueError("Benchmark ids must be unique.")

    outputs = []
    for index, benchmark_id in enumerate(benchmark_ids, start=1):
        print(f"[{index}/{len(benchmark_ids)}] planning {benchmark_id}", flush=True)
        try:
            request = load_planner_request(benchmark_id, root=root)
            output_path = root / "eval/plans" / f"{benchmark_id}_llm_plan_v1.json"
            if output_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Frozen plan already exists: {output_path}. Use --overwrite to replace it."
                )
            if planner is None:
                planner = OpenAIQueryPlanner(model=model)
            request_record = request.model_dump(mode="json")
            plan = planner.plan(request)
            record = {
                "benchmark_id": benchmark_id,
                "planner_version": "v1",
                "model": model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "request": request_record,
                "plan": plan.model_dump(mode="json"),
            }
            serialized = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Check again through exclusive creation if another run wrote this
            # plan while the model was working. Serialize before any overwrite.
            with output_path.open("w" if overwrite else "x", encoding="utf-8") as output:
                output.write(serialized)
            outputs.append(output_path)
        except Exception:
            print(f"Planning failed for benchmark {benchmark_id!r}.", file=sys.stderr)
            raise
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", required=True, help="Benchmark ids in run order")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI planner model")
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace existing frozen plans"
    )
    args = parser.parse_args(argv)
    generate_query_plans(args.benchmarks, model=args.model, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
