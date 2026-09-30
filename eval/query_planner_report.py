"""Deterministic Markdown rendering of the machine-readable planner results."""

from eval.query_planner_evaluation import METRICS


METRIC_NAMES = {
    "relevant_recall": "Relevant Recall", "direct_recall": "Direct Recall",
    "precision_at_10": "Precision@10", "strict_precision_at_10": "Strict Precision@10",
    "ndcg_at_10": "nDCG@10",
}


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _table(headers, rows) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows),
        "",
    ]


def metric_table(metrics: dict) -> list[str]:
    # Use Python float representations, preserving the JSON numeric values.
    return _table(
        ("Metric", "Human", "LLM", "Delta (LLM - Human)"),
        [(METRIC_NAMES[m], metrics["human"][m], metrics["llm"][m], metrics["delta"][m]) for m in METRICS],
    )


def render_markdown(report: dict) -> str:
    results = list(report["benchmarks"].values())
    lines = [
        "# Query Planner V1: frozen canonical retrieval evaluation", "",
        "## 1. Experiment definition", "",
        "Human handcrafted queries are compared with frozen LLM Query Planner V1 using the existing canonical RRF orders. No retrieval, API calls, candidate generation, or semantic reranking is performed.", "",
        "Recall uses all retrieved canonical works and the full union denominator. Precision@10 divides by 10. nDCG@10 uses gain `2**relevance - 1`, discount `log2(rank + 1)`, and an ideal order drawn from the full union. Relevance >= 1 is relevant; relevance == 2 is Direct/Core.", "",
        "IDCG now uses the expanded union, so these nDCG values are not directly comparable with earlier baselines whose ideal pools contained only Human-retrieved works.", "",
        "## 2. Final relevance-set construction", "",
        "The final union relevance set combines 61 frozen inherited Human judgments and 79 project-owner-adopted judgments. Human judgments take precedence; adopted labels fill only LLM-only works. Historical model-assisted provenance and confidence are retained for audit, with no weighting, exclusion, or further review requirement. Manual task files and semantic assessment files are not evaluation inputs.", "",
    ]
    lines += _table(
        ("Benchmark", "Inherited", "Adopted", "Final", "Label 0", "Label 1", "Label 2"),
        [(r["benchmark_id"], sum(r["label_counts"]["human"].values()), sum(r["label_counts"]["adopted"].values()), r["structural_counts"]["union_size"], *r["label_counts"]["final"].values()) for r in results],
    )
    lines += [f"Total: {report['totals']['judgments']} benchmark-work judgments; labels 0/1/2 = " + "/".join(str(report["totals"]["label_counts"][f"label_{i}"]) for i in range(3)) + ".", "", "## 3. Canonical pool statistics", ""]
    lines += _table(
        ("Benchmark", "Union", "Human", "LLM", "Intersection", "Human-only", "LLM-only"),
        [(r["benchmark_id"], *(r["structural_counts"][key] for key in ("union_size", "human_retrieved", "llm_retrieved", "intersection", "human_only", "llm_only"))) for r in results],
    )
    lines += ["## 4. Per-benchmark metrics", ""]
    for result in results:
        lines += [f"### {result['benchmark_id']}", "", result["research_question"], "", f"Shared recall denominators: relevant = {result['relevance_denominators']['relevant']}; Direct/Core = {result['relevance_denominators']['direct']}.", ""]
        lines += metric_table(result["metrics"])
    lines += ["## 5. Macro metrics", "", "Equal-weight arithmetic mean across the three benchmarks; no weighting by candidate-pool size.", ""]
    lines += metric_table(report["macro_metrics"])
    lines += ["## 6. Unique relevance composition", ""]
    rows = []
    for result in results:
        for source, composition in result["unique_coverage"].items():
            rows.append((result["benchmark_id"], source, *(composition[key] for key in ("total_unique_works", "label_0", "label_1", "label_2", "relevant_unique_works"))))
    for source, composition in report["totals"]["unique_coverage"].items():
        rows.append(("Total benchmark-work pairs", source, *(composition[key] for key in ("total_unique_works", "label_0", "label_1", "label_2", "relevant_unique_works"))))
    lines += _table(("Benchmark", "Source-only", "Total", "Off-target", "Supporting", "Direct/Core", "Relevant"), rows)
    lines += ["## 7. Shared-work rank diagnostics", "", "Ranks are one-based positions after canonical deduplication. Positive delta (`Human - LLM`) means LLM placed the shared work higher. These are diagnostics, not primary quality metrics. Original best source ranks are retained in JSON.", ""]
    lines += _table(
        ("Benchmark", "Shared", "LLM higher", "Human higher", "Tied"),
        [(r["benchmark_id"], *(r["shared_work_rank_summary"][key] for key in ("shared_count", "llm_higher", "human_higher", "tied"))) for r in results],
    )
    for result in results:
        lines += [f"### Shared works: {result['benchmark_id']}", ""]
        lines += _table(
            ("union_id", "Title", "Relevance", "Human rank", "LLM rank", "Delta"),
            [(d["union_id"], d["title"], d["relevance"], d["human_rank"], d["llm_rank"], d["rank_delta"]) for d in result["shared_work_rank_diagnostics"]],
        )
    lines += ["## 8. Main empirical observations", ""]
    for metric in METRICS:
        higher = sum(r["metrics"]["delta"][metric] > 0 for r in results)
        lower = sum(r["metrics"]["delta"][metric] < 0 for r in results)
        tied = len(results) - higher - lower
        lines += [f"- {METRIC_NAMES[metric]}: LLM is higher on {higher} benchmarks, lower on {lower}, and equal on {tied}."]
    lines += [""]
    for source, counts in report["totals"]["unique_coverage"].items():
        lines += [f"- {source}: {counts['relevant_unique_works']} relevant discoveries, including {counts['direct_core_unique_works']} Direct/Core works; {counts['off_target_unique_works']} off-target works (summed benchmark-work pairs)."]
    lines += ["", "## 9. Limitations", "",
        "- Three fixed research questions do not establish general superiority or statistical significance.",
        "- Recall is relative to the pooled retrieved union, not all relevant literature. Unretrieved works are not judged.",
        "- Source pools differ in size; this is the frozen configuration comparison, not a cost- or query-budget-normalized result.",
        "- Adopted labels retain their model-assisted history and title/abstract evidence limitations; project acceptance does not imply independently collected human annotations.",
        "- The experiment evaluates retrieval from frozen plans and existing RRF only. It does not evaluate semantic reranking or downstream research-answer quality.", "",
        "Reproduce from the repository root: `python -B -m scripts.run_query_planner_eval`. JSON retains raw floats, complete canonical rankings, source-file hashes, and rank diagnostics. Existing identical derived artifacts are left unchanged; `--overwrite` permits replacing only these derived outputs.", "",
    ]
    return "\n".join(lines)
