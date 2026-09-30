"""Summarize explicitly selected completed benchmarks without network requests.

python -m scripts.summarize_semantic_benchmarks --report-id semantic_reranking_v1 \
    --benchmarks matrix_completion_v1 rag_hallucination_v1
"""

import argparse

from eval.benchmark import PROJECT_ROOT
from eval.semantic_summary import generate_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--benchmarks", required=True, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = generate_report(args.report_id, args.benchmarks, overwrite=args.overwrite)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Report generation failed: {error}\n")
    print(f"Validated {report['benchmark_count']} frozen benchmarks; no API requests.")
    for suffix in (".json", ".md"):
        print(PROJECT_ROOT / "eval/reports" / f"{args.report_id}{suffix}")


if __name__ == "__main__":
    main()
