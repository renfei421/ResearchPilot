"""Resume offline Human annotation; display only question and paper evidence."""

import argparse
from collections.abc import Callable
from pathlib import Path

from eval.benchmark import PROJECT_ROOT
from eval.union_gold_io import load_annotation_state, save_json_atomic


def format_task(question: str, task: dict) -> str:
    """Explicit display whitelist, including every frozen member abstract."""
    lines = [
        f"Research Question: {question}",
        f"Union ID: {task['union_id']}",
        f"Title: {task['title']}",
        f"Year: {task['year'] if task['year'] is not None else 'unknown'}",
        f"Abstract: {task['abstract'] or '(not available)'}",
        "Identifiers: " + ", ".join(task["identifiers"]),
        "Bibliographic aliases:",
    ]
    for alias in task["aliases"]:
        lines.append(f"- {alias['title']} | year={alias['year']} | {alias['paper_id']} | {alias['group_id']}")
        if alias["abstract"] and alias["abstract"] != task["abstract"]:
            lines.append(f"  Additional abstract: {alias['abstract']}")
    return "\n".join(lines)


def label_benchmark(
    benchmark_id: str, *, root: Path = PROJECT_ROOT, list_unlabeled: bool = False,
    input_fn: Callable[[str], str] = input, output_fn: Callable[[str], None] = print,
) -> dict:
    path, document, revision = load_annotation_state(benchmark_id, root=root)
    tasks = document["tasks"]

    def progress() -> None:
        completed = sum(task["label"] is not None for task in tasks)
        output_fn(f"{completed} / {len(tasks)} completed")

    def save() -> None:
        nonlocal revision
        revision = save_json_atomic(path, document, expected_bytes=revision)

    output_fn(f"Benchmark: {document['benchmark_id']}")
    for label in ("0", "1", "2"):
        output_fn(f"{label} = {document['relevance_scale'][label]}")
    progress()
    try:
        for task in tasks:
            if task["label"] is not None:
                continue
            output_fn(format_task(document["research_question"], task))
            if list_unlabeled:
                continue
            while True:
                answer = input_fn("Relevance [0/1/2/s/q]: ").strip().lower()
                if answer == "q":
                    output_fn("Progress saved. Resume with the same command.")
                    progress()
                    return document
                if answer == "s":
                    break
                if answer not in ("0", "1", "2"):
                    output_fn("Enter exactly 0, 1, 2, s (skip), or q (save and quit).")
                    continue
                task["label"] = int(answer)
                # Save the judgment before asking for a note, even if input ends.
                save()
                task["note"] = input_fn("Note (optional): ")
                save()
                progress()
                break
    except (EOFError, KeyboardInterrupt):
        output_fn("Input ended. Previously saved judgments are preserved.")
    progress()
    return document


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--list-unlabeled", action="store_true")
    args = parser.parse_args(argv)
    label_benchmark(args.benchmark, list_unlabeled=args.list_unlabeled)


if __name__ == "__main__":
    main()
