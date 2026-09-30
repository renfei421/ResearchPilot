"""Offline validation and reporting of completed semantic retrieval experiments.

Only explicitly requested benchmark paths are read. This module has no retrieval
or inference dependency and recomputes metrics against the complete gold pool.
"""

from datetime import datetime, timezone
import json
from math import isclose, isfinite
from pathlib import Path
import re
from statistics import fmean
from typing import Any

from eval.benchmark import PROJECT_ROOT, benchmark_paths, load_benchmark
from eval.retrieval.metrics import ndcg_at_k, precision_at_k, strict_precision_at_k


LABELS = ("direct", "supporting", "off_target")
GOLD_LABELS = {2: "direct", 1: "supporting", 0: "off_target"}
METRICS = {
    "precision_at_10": ("Precision@10", precision_at_k),
    "strict_precision_at_10": ("Strict Precision@10", strict_precision_at_k),
    "ndcg_at_10": ("nDCG@10", ndcg_at_k),
}


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _list(value: Any, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list.")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value (not bool).")
    return float(value)


def _read(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Missing required artifact: {path}")
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), str(path))
    except ValueError as error:
        raise ValueError(f"Invalid artifact {path}: {error}") from error


def _unique_ids(values: Any, name: str) -> list[str]:
    ids = [_text(value, f"{name} group_id") for value in _list(values, name)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} group_ids must be unique.")
    return ids


def _same_pool(ids: list[str], expected: list[str], name: str) -> None:
    if set(ids) != set(expected):
        raise ValueError(f"{name} group_ids must exactly match the candidate pool.")


def _metrics(ids: list[str], grades: dict[str, int]) -> dict[str, float]:
    return {name: function(ids, grades, 10) for name, (_, function) in METRICS.items()}


def _check_metrics(saved: Any, computed: dict[str, float], name: str) -> None:
    saved = _object(saved, name)
    # Validate every stored metric, including any additional fields.
    for key, value in saved.items():
        _finite(value, f"{name}.{key}")
    for key, actual in computed.items():
        expected = _finite(saved.get(key), f"{name}.{key}")
        if not isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(
                f"{name}.{key} mismatch: saved={expected}, recomputed={actual}."
            )


def summarize_benchmark(benchmark_id: str, *, root: Path = PROJECT_ROOT) -> dict:
    """Validate one frozen experiment, then join rankings and gold by group_id."""
    paths = benchmark_paths(benchmark_id, root=root)
    definition = load_benchmark(benchmark_id, root=root)
    snapshot, gold, run = (_read(path) for path in (paths.candidates, paths.gold, paths.run))
    notes = []
    # Legacy snapshots predate this metadata field. Identity is still checked
    # via the requested standard path, exact question and complete group-id set.
    if "benchmark_id" not in snapshot:
        notes.append(
            "Candidate snapshot predates benchmark_id; identity validated by "
            "the standard path, matching question and complete group_id sets."
        )
    for name, artifact, question_key in (
        ("Candidate snapshot", snapshot, "question"),
        ("Gold dataset", gold, "research_question"),
        ("Semantic run", run, "question"),
    ):
        if name != "Candidate snapshot" or "benchmark_id" in artifact:
            if artifact.get("benchmark_id") != benchmark_id:
                raise ValueError(f"{name} benchmark_id does not match {benchmark_id}.")
        if artifact.get(question_key) != definition.question:
            raise ValueError(f"{name} question does not match the benchmark definition.")

    candidates = _list(snapshot.get("candidates"), "Candidates")
    if not candidates:
        raise ValueError("Cannot summarize an empty candidate pool.")
    for item in candidates:
        _object(item, "Candidate")
        _text(item.get("title"), "Candidate title")
        if type(item.get("rrf_rank")) is not int:
            raise ValueError("Candidate rrf_rank must be an integer.")
        if _finite(item.get("rrf_score"), "Candidate rrf_score") < 0:
            raise ValueError("Candidate rrf_score must be non-negative.")
    _unique_ids([item.get("group_id") for item in candidates], "Candidates")
    candidates = sorted(candidates, key=lambda item: item["rrf_rank"])
    if [item["rrf_rank"] for item in candidates] != list(range(1, len(candidates) + 1)):
        raise ValueError("Candidate RRF ranks must be consecutive starting at 1.")
    rrf_ids = [item["group_id"] for item in candidates]

    judgments = _object(gold.get("judgments"), "Gold judgments")
    _same_pool(list(judgments), rrf_ids, "Gold judgment")
    grades = {}
    for group_id, judgment in judgments.items():
        judgment = _object(judgment, f"Gold judgment {group_id}")
        if judgment.get("group_id") != group_id:
            raise ValueError(f"Gold judgment group_id does not match its key: {group_id}.")
        grade = judgment.get("relevance")
        if type(grade) is not int or grade not in GOLD_LABELS:
            raise ValueError("Gold relevance must be an integer 0, 1, or 2.")
        grades[group_id] = grade
    baseline = _object(gold.get("baseline"), "Gold baseline")
    if baseline.get("ranked_item_ids") != rrf_ids:
        raise ValueError("Gold baseline ranking must match frozen RRF order.")

    assessments = _list(run.get("assessments"), "Run assessments")
    for item in assessments:
        _object(item, "Run assessment")
    assessment_ids = _unique_ids([item.get("group_id") for item in assessments], "Assessments")
    _same_pool(assessment_ids, rrf_ids, "Assessment")
    by_id = {item["group_id"]: item for item in assessments}
    for candidate in candidates:
        item = by_id[candidate["group_id"]]
        if type(item.get("original_rrf_rank")) is not int or item["original_rrf_rank"] != candidate["rrf_rank"]:
            raise ValueError("Assessment original_rrf_rank must match frozen RRF order.")
        if item.get("title") != candidate["title"]:
            raise ValueError("Assessment title must match the frozen candidate title.")
        score = _finite(item.get("original_rrf_score"), "Assessment original_rrf_score")
        if not isclose(score, candidate["rrf_score"], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Assessment original_rrf_score must match the frozen candidate.")
        score = _finite(item.get("semantic_score"), "Assessment semantic_score")
        if not 0 <= score <= 1:
            raise ValueError("Assessment semantic_score must be between 0 and 1.")
        if item.get("category") not in LABELS:
            raise ValueError(f"Assessment category must be one of {LABELS}.")

    semantic_ids = _unique_ids(run.get("semantic_ranking"), "Semantic ranking")
    _same_pool(semantic_ids, rrf_ids, "Semantic ranking")
    expected_order = sorted(rrf_ids, key=lambda gid: by_id[gid]["semantic_score"], reverse=True)
    if semantic_ids != expected_order:
        raise ValueError("Semantic ranking must follow descending score with stable RRF ties.")
    model = _text(run.get("model"), "Run model")
    timestamp = _text(run.get("timestamp"), "Run timestamp")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError("Run timestamp must be ISO-8601 with a timezone.") from error
    if parsed_timestamp.tzinfo is None:
        raise ValueError("Run timestamp must include a timezone.")

    rrf_metrics = _metrics(rrf_ids, grades)
    semantic_metrics = _metrics(semantic_ids, grades)
    _check_metrics(run.get("baseline_metrics"), rrf_metrics, "Run baseline_metrics")
    _check_metrics(run.get("semantic_metrics"), semantic_metrics, "Run semantic_metrics")
    _check_metrics(baseline.get("expected_metrics"), rrf_metrics, "Gold baseline expected_metrics")

    semantic_ranks = {group_id: rank for rank, group_id in enumerate(semantic_ids, 1)}
    details = []
    confusion = [[0 for _ in LABELS] for _ in LABELS]
    for group_id in rrf_ids:
        item = by_id[group_id]
        gold_label = GOLD_LABELS[grades[group_id]]
        confusion[LABELS.index(gold_label)][LABELS.index(item["category"])] += 1
        details.append({
            "group_id": group_id,
            "title": item["title"],
            "original_rrf_rank": item["original_rrf_rank"],
            "semantic_rank": semantic_ranks[group_id],
            "rank_change": item["original_rrf_rank"] - semantic_ranks[group_id],
            "semantic_score": item["semantic_score"],
            "model_category": item["category"],
            "gold_relevance": grades[group_id],
            "gold_label_name": gold_label,
        })
    # Starting in RRF order makes equal-sized movements deterministic.
    upward = sorted((item for item in details if item["rank_change"] > 0),
                    key=lambda item: item["rank_change"], reverse=True)[:5]
    downward = sorted((item for item in details if item["rank_change"] < 0),
                      key=lambda item: item["rank_change"])[:5]
    summary = {
        "benchmark_id": benchmark_id,
        "research_question": definition.question,
        "candidate_count": len(candidates),
        "model": model,
        "semantic_run_timestamp": timestamp,
        "direct_count": sum(grade == 2 for grade in grades.values()),
        "supporting_count": sum(grade == 1 for grade in grades.values()),
        "off_target_count": sum(grade == 0 for grade in grades.values()),
        "validation_notes": notes,
        "semantic_top_10": sorted(details, key=lambda item: item["semantic_rank"])[:10],
        "rank_movements": details,
        "top_5_upward_movements": upward,
        "top_5_downward_movements": downward,
        "category_accuracy": sum(confusion[i][i] for i in range(3)) / len(candidates),
        "confusion_matrix": {"row_labels": list(LABELS), "column_labels": list(LABELS),
                             "counts": confusion},
    }
    for name in METRICS:
        summary[f"rrf_{name}"] = rrf_metrics[name]
        summary[f"semantic_{name}"] = semantic_metrics[name]
        summary[f"delta_{name}"] = semantic_metrics[name] - rrf_metrics[name]
    return summary


def build_report(report_id: str, benchmark_ids: list[str], *, root: Path = PROJECT_ROOT) -> dict:
    """Validate all requested artifacts before producing any report output."""
    if not isinstance(report_id, str) or re.fullmatch(r"[a-z][a-z0-9_]*", report_id) is None:
        raise ValueError("report_id must start with a-z and contain only a-z, 0-9, or _.")
    if not benchmark_ids:
        raise ValueError("At least one benchmark id is required.")
    ids = _unique_ids(benchmark_ids, "Requested benchmark")
    summaries = {}
    for benchmark_id in ids:
        try:
            summaries[benchmark_id] = summarize_benchmark(benchmark_id, root=root)
        except ValueError as error:
            raise ValueError(f"{benchmark_id}: {error}") from error
    aggregate = {"number_of_benchmarks": len(ids)}
    for name in METRICS:
        aggregate[name] = {
            f"mean_{label}": fmean(summary[f"{label}_{name}"] for summary in summaries.values())
            for label in ("rrf", "semantic", "delta")
        }
    return {
        "report_id": report_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_ids": ids,
        "benchmark_count": len(ids),
        "per_benchmark": summaries,
        "aggregate": aggregate,
    }


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    )


def render_markdown(report: dict) -> str:
    """Render descriptive comparisons only; no significance or generalization claim."""
    summaries = list(report["per_benchmark"].values())
    models = list(dict.fromkeys(summary["model"] for summary in summaries))
    sections = [
        "# Semantic Reranking Evaluation Report",
        f"Report: `{report['report_id']}` · Generated (UTC): {report['generated_at']}",
        "## Experimental Setup",
        f"This offline report covers {report['benchmark_count']} frozen benchmarks. "
        "Handcrafted multi-query retrieval and version-aware grouping produced the "
        "fixed candidate pools; group-level RRF supplies the baseline order. "
        "The saved semantic reranker assessed only the research question, title and "
        "abstract, without gold labels or judgment reasons. Complete human gold "
        "judgments are joined by group_id only for offline evaluation.",
        f"Recorded model(s): {', '.join(f'`{model}`' for model in models)}. "
        "Semantic ranking uses score alone, retaining RRF order on exact ties. "
        "This report reads completed runs and makes no API requests.",
        "All metrics are recomputed and checked against saved results "
        "(relative tolerance 1e-9; absolute tolerance 1e-12). Precision counts "
        "gold grades >= 1; strict precision counts grade 2; both divide by 10. "
        "nDCG uses gain 2**relevance - 1, logarithmic rank discount, and all gold "
        "judgments for the ideal ranking. Deltas are semantic minus RRF.",
        "## Overall Results",
        _table(
            ["Benchmark", "Candidates", "RRF P@10", "Semantic P@10", "RRF Strict P@10",
             "Semantic Strict P@10", "RRF nDCG@10", "Semantic nDCG@10", "Δ nDCG"],
            [[s["benchmark_id"], s["candidate_count"]]
             + [f"{s[key]:.7f}" for key in (
                 "rrf_precision_at_10", "semantic_precision_at_10",
                 "rrf_strict_precision_at_10", "semantic_strict_precision_at_10",
                 "rrf_ndcg_at_10", "semantic_ndcg_at_10", "delta_ndcg_at_10")]
             for s in summaries],
        ),
        "## Aggregate Results",
        f"Unweighted macro averages across {report['benchmark_count']} benchmarks; "
        "each benchmark contributes equally regardless of pool size.",
        _table(["Metric", "Mean RRF", "Mean semantic", "Mean delta"],
               [[label] + [f"{report['aggregate'][key][field]:.7f}" for field in
                            ("mean_rrf", "mean_semantic", "mean_delta")]
                for key, (label, _) in METRICS.items()]),
    ]
    for s in summaries:
        sections.extend([
            f"## Benchmark: {s['benchmark_id']}",
            f"**Question:** {_cell(s['research_question'])}",
            f"Candidates: {s['candidate_count']}; gold: {s['direct_count']} direct, "
            f"{s['supporting_count']} supporting, {s['off_target_count']} off_target. "
            f"Model: `{s['model']}`; semantic run timestamp: {s['semantic_run_timestamp']}.",
            "### Metric Comparison",
            _table(["Metric", "RRF", "Semantic", "Delta"],
                   [[label] + [f"{s[f'{prefix}_{key}']:.7f}" for prefix in
                                ("rrf", "semantic", "delta")]
                    for key, (label, _) in METRICS.items()]),
            "### Semantic Top 10",
            _table(["Semantic rank", "Group ID", "Title", "RRF rank", "Score",
                    "Model category", "Gold relevance", "Gold label"],
                   [[item["semantic_rank"], item["group_id"], item["title"],
                     item["original_rrf_rank"], f"{item['semantic_score']:.4f}",
                     item["model_category"], item["gold_relevance"], item["gold_label_name"]]
                    for item in s["semantic_top_10"]]),
            "### Rank Movements",
            "Rank change = original RRF rank - semantic rank. Positive is upward; "
            "negative is downward. Equal changes retain original RRF order.",
        ])
        for heading, key in (("Top 5 upward movements", "top_5_upward_movements"),
                             ("Top 5 downward movements", "top_5_downward_movements")):
            sections.extend([
                f"#### {heading}",
                _table(["Title", "Gold relevance", "RRF rank", "Semantic rank", "Change", "Score"],
                       [[item["title"], item["gold_relevance"], item["original_rrf_rank"],
                         item["semantic_rank"], f"{item['rank_change']:+d}",
                         f"{item['semantic_score']:.4f}"] for item in s[key]])
                if s[key] else "No movements in this direction.",
            ])
        sections.extend([
            "### Category Calibration",
            f"Category accuracy over all candidates: {s['category_accuracy']:.7f}. "
            "Rows are gold labels; columns are model categories. This is a diagnostic "
            "comparison and does not affect ranking.",
            _table(["Gold / model", *LABELS],
                   [[label, *row] for label, row in zip(LABELS, s["confusion_matrix"]["counts"])]),
        ])
        if s["validation_notes"]:
            sections.append("**Artifact compatibility:** " + " ".join(s["validation_notes"]))
    sections.append("## Observations")
    for s in summaries:
        delta = s["delta_ndcg_at_10"]
        direction = "increased" if delta > 0 else "decreased" if delta < 0 else "was unchanged"
        sections.append(
            f"- `{s['benchmark_id']}`: nDCG@10 {direction} "
            f"(delta {delta:+.7f}); Precision@10 changed by {s['delta_precision_at_10']:+.7f}, "
            f"and Strict Precision@10 by {s['delta_strict_precision_at_10']:+.7f}."
        )
    sections.extend([
        "## Limitations",
        f"- Only {report['benchmark_count']} frozen benchmarks are included; candidate pools "
        f"are small ({', '.join(str(s['candidate_count']) for s in summaries)} groups).\n"
        "- Human relevance judgments are manually constructed.\n"
        + ("- One included benchmark, matrix_completion_v1, was involved in prompt-rubric "
           "refinement and is not an independent held-out test of that rubric.\n"
           if "matrix_completion_v1" in report["benchmark_ids"] else "")
        + ("- Semantic scores come from one recorded model/configuration; no model comparison "
           "is available here.\n" if len(models) == 1 else
           "- Recorded models differ across benchmarks; model effects are not isolated.\n")
        + "- No repeated-run variance analysis has been performed.\n"
        "- Evidence is limited to the included frozen benchmarks. These descriptive "
        "results do not establish statistical significance or universal generalization.",
    ])
    return "\n\n".join(sections) + "\n"


def generate_report(
    report_id: str, benchmark_ids: list[str], *, root: Path = PROJECT_ROOT,
    overwrite: bool = False,
) -> dict:
    """Write JSON and Markdown only after every benchmark passes validation."""
    report = build_report(report_id, benchmark_ids, root=root)
    outputs = [root / "eval/reports" / f"{report_id}{suffix}" for suffix in (".json", ".md")]
    for path in outputs:
        if path.is_symlink():
            raise ValueError(f"Report output must not be a symlink: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Report already exists: {path}; use --overwrite explicitly.")
    contents = [json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                render_markdown(report)]
    outputs[0].parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, content in zip(outputs, contents):
            with path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as handle:
                if not overwrite:
                    created.append(path)
                handle.write(content)
    except OSError:
        # An exclusive-create collision must not leave half a new report pair.
        for path in created:
            path.unlink()
        raise
    return report
