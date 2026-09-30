"""Offline query-planning evaluation against final adopted union relevance.

Canonical identity and order come from the frozen alignment manifests. Labels
are never inferred from scores, query text, confidence, or provenance.
"""

from collections import Counter
from copy import deepcopy
from math import isfinite
from typing import Any

from eval.identity_alignment import canonical_source_ranking
from eval.retrieval.metrics import ndcg_at_k, precision_at_k, strict_precision_at_k
from eval.union_gold import migrate_human_gold, validate_manifest


BENCHMARKS = (
    "matrix_completion_v1", "rag_hallucination_v1", "cot_reasoning_faithfulness_v1",
)
METRICS = (
    "relevant_recall", "direct_recall", "precision_at_10",
    "strict_precision_at_10", "ndcg_at_10",
)
# Acceptance guards only; metric values are always calculated from actual labels.
EXPECTED_LABELS = {
    BENCHMARKS[0]: {"human": (10, 5, 7), "adopted": (9, 4, 5), "final": (19, 9, 12)},
    BENCHMARKS[1]: {"human": (1, 12, 4), "adopted": (6, 12, 8), "final": (7, 24, 12)},
    BENCHMARKS[2]: {"human": (5, 11, 6), "adopted": (21, 12, 2), "final": (26, 23, 8)},
}
EXPECTED_STRUCTURE = {
    BENCHMARKS[0]: (22, 30, 12, 10, 18, 40),
    BENCHMARKS[1]: (17, 38, 12, 5, 26, 43),
    BENCHMARKS[2]: (22, 42, 7, 15, 35, 57),
}


def _label(value: Any) -> int:
    if type(value) is not int or value not in (0, 1, 2):
        raise ValueError("Relevance must be exactly an integer 0, 1, or 2.")
    return value


def label_counts(values) -> dict[str, int]:
    counts = Counter(_label(value) for value in values)
    return {f"label_{label}": counts[label] for label in (0, 1, 2)}


def _check_expected_labels(benchmark: str, source: str, values) -> dict[str, int]:
    counts = label_counts(values)
    expected = EXPECTED_LABELS.get(benchmark, {}).get(source)
    if expected is not None and tuple(counts.values()) != expected:
        raise ValueError(f"{benchmark} {source} label distribution {counts} disagrees with {expected}; inspect source data.")
    return counts


def build_final_relevance(manifest: dict, human_gold: dict, adopted: dict) -> dict:
    """Inherit Human labels and fill only canonical LLM-only works.

An adopted record targeting a Human work is rejected, even if its label agrees;
it must never override or replace the frozen Human judgment.
"""
    inherited = migrate_human_gold(manifest, human_gold)
    benchmark = manifest["benchmark_id"]
    works = {work["union_id"]: work for work in manifest["works"]}
    if adopted.get("benchmark_id") != benchmark:
        raise ValueError("Adopted benchmark_id must match the manifest.")
    if adopted.get("research_question") != human_gold["research_question"]:
        raise ValueError("Adopted research_question must match Human gold.")
    entries = adopted.get("judgments")
    if not isinstance(entries, list):
        raise ValueError("Adopted judgments must be a list.")
    if "judgment_count" in adopted and adopted["judgment_count"] != len(entries):
        raise ValueError("Adopted judgment_count disagrees with its records.")
    adopted_by_id = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("union_id"), str):
            raise ValueError("Every adopted judgment needs a union_id.")
        uid = entry["union_id"]
        if uid in adopted_by_id:
            raise ValueError(f"Duplicate adopted judgment: {uid}.")
        if uid not in works:
            raise ValueError(f"Unknown adopted union_id: {uid}.")
        if uid in inherited:
            raise ValueError(f"Cannot overwrite inherited Human judgment {uid}; adopted labels must be LLM-only.")
        _label(entry.get("label"))
        adopted_by_id[uid] = entry
    required = set(works) - set(inherited)
    if set(adopted_by_id) != required:
        raise ValueError(f"Missing adopted relevance labels: {sorted(required - set(adopted_by_id))}.")

    counts = {
        "human": _check_expected_labels(benchmark, "human", (j["relevance"] for j in human_gold["judgments"].values())),
        "adopted": _check_expected_labels(benchmark, "adopted", (j["label"] for j in entries)),
    }
    judgments = deepcopy(inherited)
    for uid, entry in adopted_by_id.items():
        representative = min(works[uid]["members"], key=lambda m: (m["original_rank"], m["group_id"]))
        judgments[uid] = {
            "union_id": uid,
            "relevance": entry["label"],
            "title": representative["title"],
            "provenance": "project_owner_adopted",
            "original_provenance": deepcopy(entry),
            **{key: deepcopy(entry[key]) for key in ("confidence", "note", "annotation_source") if key in entry},
        }
    counts["final"] = _check_expected_labels(benchmark, "final", (j["relevance"] for j in judgments.values()))
    return {
        "schema_version": 1,
        "benchmark_id": benchmark,
        "research_question": human_gold["research_question"],
        "relevance_scale": deepcopy(human_gold["relevance_scale"]),
        "label_status": "final_adopted_for_query_planner_v1",
        "adoption_policy": "Project owner accepted all 79 new labels for Step 4.9. All inherited and adopted labels have equal metric weight; no further review is required.",
        "source_history": {
            "human_gold_provenance": deepcopy(human_gold.get("provenance", {})),
            "adopted_file_metadata": deepcopy({key: value for key, value in adopted.items() if key != "judgments"}),
        },
        "label_counts": counts,
        "judgment_count": len(judgments),
        "judgments": {uid: judgments[uid] for uid in sorted(judgments)},
    }


def relevance_map(manifest: dict, dataset: dict) -> dict[str, int]:
    """Extract only IDs/grades, validating exact canonical coverage."""
    known = {work["union_id"] for work in manifest["works"]}
    if dataset.get("benchmark_id") != manifest["benchmark_id"]:
        raise ValueError("Final relevance benchmark_id does not match manifest.")
    records = dataset.get("judgments")
    if not isinstance(records, dict):
        raise ValueError("Final judgments must be keyed by union_id.")
    result = {}
    for key, record in records.items():
        if not isinstance(record, dict) or not isinstance(record.get("union_id"), str):
            raise ValueError("Every final judgment needs a union_id.")
        uid = record["union_id"]
        if uid in result:
            raise ValueError(f"Duplicate final judgment: {uid}.")
        if uid not in known:
            raise ValueError(f"Unknown final union_id: {uid}.")
        if key != uid:
            raise ValueError("Final judgment key must match union_id.")
        result[uid] = _label(record.get("relevance"))
    if set(result) != known:
        raise ValueError(f"Missing final relevance labels: {sorted(known - set(result))}.")
    if dataset.get("judgment_count") != len(result):
        raise ValueError("Final judgment_count disagrees with its records.")
    _check_expected_labels(manifest["benchmark_id"], "final", result.values())
    return result


def _validate_ranking(ranked: list[str], labels: dict[str, int]) -> None:
    if not isinstance(ranked, list) or any(not isinstance(uid, str) for uid in ranked):
        raise ValueError("Canonical ranking must be a list of union_ids.")
    if len(set(ranked)) != len(ranked):
        raise ValueError("Duplicate union_id in canonical ranking.")
    unknown = set(ranked) - set(labels)
    if unknown:
        raise ValueError(f"Unknown union_id in canonical ranking: {sorted(unknown)}.")


def ranking_metrics(ranked: list[str], labels: dict[str, int]) -> dict[str, float]:
    """Recall over the whole source ranking; top-10 metrics share full-union labels."""
    for value in labels.values():
        _label(value)
    _validate_ranking(ranked, labels)
    relevant = sum(value >= 1 for value in labels.values())
    direct = sum(value == 2 for value in labels.values())
    if relevant == 0 or direct == 0:
        raise ValueError("Full union relevant and direct denominators must both be positive.")
    result = {
        "relevant_recall": sum(labels[uid] >= 1 for uid in ranked) / relevant,
        "direct_recall": sum(labels[uid] == 2 for uid in ranked) / direct,
        "precision_at_10": precision_at_k(ranked, labels, 10),
        "strict_precision_at_10": strict_precision_at_k(ranked, labels, 10),
        "ndcg_at_10": ndcg_at_k(ranked, labels, 10),
    }
    if any(not isfinite(value) or not 0 <= value <= 1 for value in result.values()):
        raise ValueError("Every metric must be finite and between 0 and 1.")
    return result


def composition(ids, labels: dict[str, int]) -> dict[str, int]:
    counts = label_counts(labels[uid] for uid in ids)
    return {
        "total_unique_works": sum(counts.values()),
        **counts,
        "off_target_unique_works": counts["label_0"],
        "supporting_unique_works": counts["label_1"],
        "direct_core_unique_works": counts["label_2"],
        "relevant_unique_works": counts["label_1"] + counts["label_2"],
    }


def evaluate_benchmark(manifest: dict, dataset: dict) -> dict:
    works = validate_manifest(manifest)
    labels = relevance_map(manifest, dataset)
    rankings = {}
    for source in ("human", "llm"):
        ranked = manifest.get(f"{source}_union_ranking")
        _validate_ranking(ranked, labels)
        expected_ids = {uid for uid, work in works.items() if work[f"present_in_{source}"]}
        if set(ranked) != expected_ids:
            raise ValueError(f"{source} ranking must cover exactly its canonical source works.")
        # Cross-check the stored order without regenerating retrieval or identity.
        if ranked != canonical_source_ranking(manifest["works"], source):
            raise ValueError(f"{source} ranking disagrees with best original source ranks.")
        rankings[source] = list(ranked)
    human, llm = set(rankings["human"]), set(rankings["llm"])
    shared, human_only, llm_only = human & llm, human - llm, llm - human
    structure = {
        "human_retrieved": len(human), "llm_retrieved": len(llm),
        "intersection": len(shared), "human_only": len(human_only),
        "llm_only": len(llm_only), "union_size": len(labels),
    }
    if human | llm != set(labels) or (
        len(human) != len(shared) + len(human_only)
        or len(llm) != len(shared) + len(llm_only)
        or len(labels) != len(shared) + len(human_only) + len(llm_only)
    ):
        raise ValueError("Canonical source sets violate union/intersection invariants.")
    expected = EXPECTED_STRUCTURE.get(manifest["benchmark_id"])
    if expected is not None and tuple(structure.values()) != expected:
        raise ValueError(f"Canonical structure {structure} disagrees with expected {expected}.")
    metrics = {source: ranking_metrics(ranking, labels) for source, ranking in rankings.items()}
    metrics["delta"] = {metric: metrics["llm"][metric] - metrics["human"][metric] for metric in METRICS}
    positions = {source: {uid: index for index, uid in enumerate(ranking, 1)} for source, ranking in rankings.items()}
    diagnostics = []
    for uid in rankings["human"]:
        if uid not in shared:
            continue
        h, l = positions["human"][uid], positions["llm"][uid]
        diagnostics.append({
            "union_id": uid, "title": dataset["judgments"][uid]["title"],
            "relevance": labels[uid], "human_rank": h, "llm_rank": l,
            "rank_delta": h - l,
            **{f"{source}_best_original_rank": min(m["original_rank"] for m in works[uid]["members"] if m["source"] == source) for source in ("human", "llm")},
        })
    return {
        "benchmark_id": manifest["benchmark_id"], "research_question": dataset["research_question"],
        "k": 10, "structural_counts": structure,
        "label_counts": deepcopy(dataset["label_counts"]),
        "relevance_denominators": {"relevant": sum(v >= 1 for v in labels.values()), "direct": sum(v == 2 for v in labels.values())},
        "rankings": rankings,
        "retrieved_counts": {source: {"relevant": sum(labels[uid] >= 1 for uid in ranking), "direct": sum(labels[uid] == 2 for uid in ranking)} for source, ranking in rankings.items()},
        "metrics": metrics,
        "unique_coverage": {"human_only": composition(human_only, labels), "llm_only": composition(llm_only, labels)},
        "shared_work_rank_diagnostics": diagnostics,
        "shared_work_rank_summary": {
            "shared_count": len(diagnostics),
            "llm_higher": sum(d["rank_delta"] > 0 for d in diagnostics),
            "human_higher": sum(d["rank_delta"] < 0 for d in diagnostics),
            "tied": sum(d["rank_delta"] == 0 for d in diagnostics),
        },
    }


def macro_metrics(results: list[dict]) -> dict:
    if not results:
        raise ValueError("Macro averaging requires at least one benchmark.")
    if len({r["benchmark_id"] for r in results}) != len(results):
        raise ValueError("Macro averaging requires distinct benchmark ids.")
    macro = {
        source: {metric: sum(r["metrics"][source][metric] for r in results) / len(results) for metric in METRICS}
        for source in ("human", "llm")
    }
    macro["delta"] = {metric: macro["llm"][metric] - macro["human"][metric] for metric in METRICS}
    return macro


def build_report(results: list[dict]) -> dict:
    macro = macro_metrics(results)
    return {
        "schema_version": 1, "experiment_id": "query_planner_v1",
        "experiment": {
            "comparison": "Human handcrafted queries versus frozen LLM Query Planner V1",
            "relevance_set": "Final adopted canonical union relevance; all labels weighted equally",
            "semantic_reranking": False, "retrieval_rerun": False,
            "recall_scope": "Full frozen canonical union for each benchmark, not the research literature",
            "precision_denominator": 10,
            "ndcg_gain": "2**relevance - 1", "ndcg_discount": "log2(one_based_rank + 1)",
            "ndcg_ideal_pool": "Full canonical union relevance set",
            "macro_weighting": "Arithmetic mean, one equal weight per benchmark",
            "shared_rank_definition": "One-based positions in canonical rankings after deduplication; original best ranks retained as audit fields",
            "rank_delta_definition": "human_rank - llm_rank; positive means LLM ranked higher; diagnostic only",
            "provenance_policy": "History retained; confidence, provenance, notes, and review completion do not affect metrics",
        },
        "benchmarks": {r["benchmark_id"]: deepcopy(r) for r in results},
        "macro_metrics": macro,
        "totals": {
            "judgments": sum(r["structural_counts"]["union_size"] for r in results),
            "label_counts": {f"label_{label}": sum(r["label_counts"]["final"][f"label_{label}"] for r in results) for label in (0, 1, 2)},
            "unique_coverage": {source: {key: sum(r["unique_coverage"][source][key] for r in results) for key in results[0]["unique_coverage"][source]} for source in ("human_only", "llm_only")},
            "counting_unit": "Benchmark-work pairs; no cross-question deduplication",
        },
    }
