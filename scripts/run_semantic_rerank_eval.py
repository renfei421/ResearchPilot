"""Evaluate the frozen pool without retrieving candidates again.

From the repository root:
python -m scripts.run_semantic_rerank_eval --benchmark <benchmark_id>
The command assesses each frozen candidate; importing it performs no I/O.
Gold is read only after every assessment and the semantic ranking are complete.
"""

import argparse
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from typing import Any

from eval.benchmark import PROJECT_ROOT, benchmark_paths, load_benchmark
from eval.retrieval.metrics import ndcg_at_k, precision_at_k, strict_precision_at_k
from researchpilot.openai_relevance_client import OpenAIRelevanceClient
from researchpilot.relevance import RelevanceClient


DEFAULT_MODEL = "gpt-5.6-terra"


def _load_candidates(
    path: Path, benchmark_id: str
) -> tuple[str, list[dict[str, Any]]]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise ValueError("Candidate snapshot must be an object.")
    # The first frozen snapshot predates benchmark_id; preserve it unchanged.
    if "benchmark_id" in snapshot and snapshot["benchmark_id"] != benchmark_id:
        raise ValueError("Candidate snapshot benchmark_id does not match the benchmark.")
    question = snapshot.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Candidate snapshot must have a non-empty question.")
    candidates = snapshot.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("Candidate snapshot must contain a candidates list.")
    candidate_count = len(candidates)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Each candidate must be an object.")
        for field in ("group_id", "title"):
            value = candidate.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Each candidate must have a non-empty {field}.")
        abstract = candidate.get("abstract")
        if abstract is not None and not isinstance(abstract, str):
            raise ValueError("Candidate abstract must be a string or None.")
        rank = candidate.get("rrf_rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise ValueError("Candidate rrf_rank must be an integer.")
        score = candidate.get("rrf_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not isfinite(score)
            or score < 0
        ):
            raise ValueError("Candidate rrf_score must be finite and non-negative.")

    if len({item["group_id"] for item in candidates}) != candidate_count:
        raise ValueError("Candidate group_id values must be unique.")
    if sorted(item["rrf_rank"] for item in candidates) != list(range(1, candidate_count + 1)):
        raise ValueError("Candidate rrf_rank values must be consecutive starting at 1.")
    return question, sorted(candidates, key=lambda item: item["rrf_rank"])


def _load_gold(
    path: Path, question: str, rrf_ids: list[str], benchmark_id: str
) -> dict[str, int]:
    """Load and align the complete gold pool, only after semantic ranking."""
    gold = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(gold, dict) or gold.get("benchmark_id") != benchmark_id:
        raise ValueError(f"Gold benchmark_id must be {benchmark_id}.")
    if gold.get("research_question") != question:
        raise ValueError("Gold research question does not match the candidate snapshot.")
    judgments = gold.get("judgments")
    if not isinstance(judgments, dict):
        raise ValueError("Gold must contain judgments keyed by group_id.")
    missing = set(rrf_ids) - set(judgments)
    if missing:
        raise ValueError(f"Missing gold judgment for group_id(s): {sorted(missing)}")
    if set(judgments) - set(rrf_ids):
        raise ValueError("Gold judgments must match the frozen candidate pool.")
    if gold.get("baseline", {}).get("ranked_item_ids") != rrf_ids:
        raise ValueError("Gold baseline order does not match the candidate RRF order.")

    grades = {}
    for group_id, judgment in judgments.items():
        if not isinstance(judgment, dict) or "relevance" not in judgment:
            raise ValueError(f"Missing relevance grade for group_id {group_id!r}.")
        grades[group_id] = judgment["relevance"]
    # The existing metrics validate the grade types and allowed values.
    return grades


def _metrics(ranked_ids: list[str], judgments: dict[str, int]) -> dict[str, float]:
    return {
        "precision_at_10": precision_at_k(ranked_ids, judgments, 10),
        "strict_precision_at_10": strict_precision_at_k(ranked_ids, judgments, 10),
        "ndcg_at_10": ndcg_at_k(ranked_ids, judgments, 10),
    }


def run_evaluation(
    benchmark_id: str,
    *,
    model: str = DEFAULT_MODEL,
    client: RelevanceClient | None = None,
    root: Path = PROJECT_ROOT,
    candidate_path: Path | None = None,
    gold_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Assess the snapshot, evaluate against gold, and save only a complete run.

    Inject a fake relevance client and temporary paths for offline tests.
    Exceptions propagate; an assessment failure leaves any prior run untouched.
    """
    definition = load_benchmark(benchmark_id, root=root)
    paths = benchmark_paths(benchmark_id, root=root)
    candidate_path = candidate_path if candidate_path is not None else paths.candidates
    gold_path = gold_path if gold_path is not None else paths.gold
    output_path = output_path if output_path is not None else paths.run
    if output_path.resolve() in {
        candidate_path.resolve(), gold_path.resolve(), paths.definition.resolve()
    }:
        raise ValueError("Run output must not overwrite the candidate or gold file.")
    question, candidates = _load_candidates(candidate_path, benchmark_id)
    if question != definition.question:
        raise ValueError("Candidate question does not match the benchmark definition.")
    if client is None:
        client = OpenAIRelevanceClient(model=model)

    # Phase 1: gold is not loaded or passed to the assessment client.
    assessments = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] assessing {candidate['title']}", flush=True)
        assessment = client.assess(
            research_question=question,
            title=candidate["title"],
            abstract=candidate.get("abstract"),
        )
        assessments.append(
            {
                "group_id": candidate["group_id"],
                "original_rrf_rank": candidate["rrf_rank"],
                "original_rrf_score": candidate["rrf_score"],
                "title": candidate["title"],
                "category": assessment.category,
                "semantic_score": assessment.score,
                "reason": assessment.reason,
            }
        )
    # Stable sorting keeps original RRF order for equal semantic scores.
    semantic_order = sorted(
        assessments, key=lambda item: item["semantic_score"], reverse=True
    )
    semantic_ids = [item["group_id"] for item in semantic_order]

    # Phase 2: all assessments and the ranking are finished before reading gold.
    rrf_ids = [candidate["group_id"] for candidate in candidates]
    judgments = _load_gold(gold_path, question, rrf_ids, benchmark_id)
    baseline_metrics = _metrics(rrf_ids, judgments)
    semantic_metrics = _metrics(semantic_ids, judgments)
    run = {
        "benchmark_id": benchmark_id,
        "question": question,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "assessments": assessments,
        "semantic_ranking": semantic_ids,
        "baseline_metrics": baseline_metrics,
        "semantic_metrics": semantic_metrics,
    }
    serialized = json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")

    print(f"\n{'Metric':<23} {'RRF baseline':>15} {'Semantic reranker':>19}")
    for label, key in (
        ("Precision@10", "precision_at_10"),
        ("Strict Precision@10", "strict_precision_at_10"),
        ("nDCG@10", "ndcg_at_10"),
    ):
        print(f"{label:<23} {baseline_metrics[key]:>15.7f} {semantic_metrics[key]:>19.7f}")
    print("\nSemantic Top 10:")
    for rank, item in enumerate(semantic_order[:10], start=1):
        print(
            f"{rank}. {item['title']} | {item['category']} | "
            f"score={item['semantic_score']:.6f} | "
            f"original RRF rank={item['original_rrf_rank']}"
        )
    print(f"\nSaved completed run: {output_path}")
    return run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="Benchmark definition id")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name")
    args = parser.parse_args(argv)
    run_evaluation(args.benchmark, model=args.model)


if __name__ == "__main__":
    main()
