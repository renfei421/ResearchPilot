"""Evaluate frozen Human/LLM query plans offline using final adopted labels."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from eval.benchmark import PROJECT_ROOT
from eval.query_planner_evaluation import BENCHMARKS, build_final_relevance, build_report, evaluate_benchmark
from eval.query_planner_report import render_markdown
from eval.union_gold_io import save_json_atomic


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key/judgment: {key}.")
        result[key] = value
    return result


def read_document(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    data = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}.")
    return data, sha256(raw).hexdigest()


def input_paths(benchmark: str, root: Path) -> dict[str, Path]:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unsupported planner benchmark: {benchmark}.")
    return {
        "manifest": root / "eval/alignment" / f"{benchmark}_union_v1.json",
        "human_gold": root / "eval/datasets" / f"{benchmark}.json",
        "adopted_labels": root / "eval/annotations/model_assisted" / f"{benchmark}_silver_labels_v1.json",
    }


def compute_evaluation(*, root: Path = PROJECT_ROOT) -> tuple[dict, dict[str, dict]]:
    """Read exactly nine source files; never resolve historical task-file paths."""
    results, datasets = [], {}
    for benchmark in BENCHMARKS:
        documents, sources = {}, {}
        for role, path in input_paths(benchmark, root).items():
            documents[role], digest = read_document(path)
            sources[role] = {"path": path.relative_to(root).as_posix(), "sha256": digest}
        if documents["manifest"].get("benchmark_id") != benchmark:
            raise ValueError("Manifest benchmark_id does not match the requested benchmark path.")
        dataset = build_final_relevance(documents["manifest"], documents["human_gold"], documents["adopted_labels"])
        dataset["input_artifacts"] = sources
        result = evaluate_benchmark(documents["manifest"], dataset)
        result["input_artifacts"] = sources
        result["final_relevance_artifact"] = f"eval/datasets/{benchmark}_union_relevance_v1.json"
        datasets[benchmark] = dataset
        results.append(result)
    report = build_report(results)
    if report["totals"]["judgments"] != 140 or report["totals"]["label_counts"] != {"label_0": 52, "label_1": 56, "label_2": 32}:
        raise ValueError("Actual final union totals disagree with the accepted 140 / 52,56,32 expectations.")
    return report, datasets


def run_evaluation(*, root: Path = PROJECT_ROOT, overwrite: bool = False) -> dict:
    report, datasets = compute_evaluation(root=root)
    json_outputs = {root / "eval/datasets" / f"{b}_union_relevance_v1.json": data for b, data in datasets.items()}
    json_outputs[root / "eval/reports/query_planner_v1.json"] = report
    markdown_path = root / "eval/reports/query_planner_v1.md"
    content = {
        path: (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        for path, data in json_outputs.items()
    }
    content[markdown_path] = render_markdown(report).encode("utf-8")
    protected = {p.resolve() for b in BENCHMARKS for p in input_paths(b, root).values()}
    previous = {}
    # Validate every output before writing any derived artifact.
    for path, target_bytes in content.items():
        if path.is_symlink() or path.resolve() in protected:
            raise ValueError(f"Output must not alias a frozen source: {path}.")
        previous[path] = path.read_bytes() if path.exists() else None
        if previous[path] is not None and previous[path] != target_bytes and not overwrite:
            raise FileExistsError(f"Derived output differs: {path}. Use --overwrite to rebuild derived artifacts.")
    # Ensure the same frozen inputs remained in place during computation.
    for result in report["benchmarks"].values():
        for source in result["input_artifacts"].values():
            if sha256((root / source["path"]).read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("Frozen evaluation input changed during computation.")
    for path, data in json_outputs.items():
        if previous[path] != content[path]:
            save_json_atomic(path, data, expected_bytes=previous[path])
    if previous[markdown_path] != content[markdown_path]:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_bytes(content[markdown_path])
    print("Query Planner V1: 140 final union judgments; all three benchmarks evaluated offline.")
    print(f"Reports: {root / 'eval/reports/query_planner_v1.json'} and {markdown_path}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace differing derived relevance/report artifacts only")
    args = parser.parse_args(argv)
    run_evaluation(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
