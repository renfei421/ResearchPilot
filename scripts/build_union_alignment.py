"""Rebuild evaluation-only work alignment from frozen Human and LLM snapshots.

Generated manifests/review candidates/summaries are replaceable. Human review
decisions are optional inputs and are never created, changed, or overwritten.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from eval.benchmark import PROJECT_ROOT, benchmark_paths
from eval.identity_alignment import IdentityReviews, align_candidates, snapshot_members


def _load_snapshot(path: Path, benchmark_id: str, *, legacy_human: bool = False) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Snapshot must be a JSON object: {path}.")
    # The original Human matrix snapshot predates the top-level benchmark_id.
    # Its conventional filename and matching question identify the legacy input.
    if "benchmark_id" in data or not legacy_human:
        if data.get("benchmark_id") != benchmark_id:
            raise ValueError(f"Snapshot benchmark_id does not match {benchmark_id!r}: {path}.")
    if not isinstance(data.get("question"), str) or not data["question"].strip():
        raise ValueError(f"Snapshot question must be a non-empty string: {path}.")
    return data


def build_union_alignment(benchmark_id: str, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Read only two candidate snapshots and optional explicit identity reviews."""
    human_path = benchmark_paths(benchmark_id, root=root).candidates
    llm_path = root / "eval/datasets" / f"{benchmark_id}_llm_candidates_v1.json"
    directory = root / "eval/alignment"
    decisions_path = directory / f"{benchmark_id}_identity_reviews_v1.json"
    human = _load_snapshot(human_path, benchmark_id, legacy_human=True)
    llm = _load_snapshot(llm_path, benchmark_id)
    if human["question"] != llm["question"]:
        raise ValueError("Human and LLM snapshot questions must match.")
    members = snapshot_members(human, "human") + snapshot_members(llm, "llm")
    reviews = None
    if decisions_path.exists():
        reviews = IdentityReviews.model_validate_json(decisions_path.read_text(encoding="utf-8"))
    manifest = align_candidates(benchmark_id, members, reviews)

    metadata = {key: manifest[key] for key in (
        "benchmark_id", "version", "identity_review_complete", "counts_status",
    )}
    artifacts = {
        directory / f"{benchmark_id}_union_v1.json": manifest,
        directory / f"{benchmark_id}_review_candidates_v1.json": {
            **metadata, "review_candidates": manifest["review_candidates"],
        },
        directory / f"{benchmark_id}_alignment_summary_v1.json": {
            **metadata, **manifest["summary"],
        },
    }
    protected = {path.resolve() for path in (human_path, llm_path, decisions_path)}
    for path in artifacts:
        if path.is_symlink() or path.resolve() in protected:
            raise ValueError(f"Generated output must not alias a protected input: {path}.")
    # Validate and serialize the entire result before replacing generated files.
    serialized = {
        path: json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        for path, data in artifacts.items()
    }
    directory.mkdir(parents=True, exist_ok=True)
    for path, text in serialized.items():
        path.write_text(text, encoding="utf-8")

    status = "identity review complete" if manifest["identity_review_complete"] else "PROVISIONAL: identity review incomplete"
    print(f"{benchmark_id}: {status}")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    if not manifest["identity_review_complete"]:
        print("Human-only/LLM-only counts are provisional; resolve identity reviews before using them for gold construction.")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="Benchmark id to align")
    args = parser.parse_args(argv)
    build_union_alignment(args.benchmark)


if __name__ == "__main__":
    main()
