"""Prepare offline, unlabeled canonical-work tasks without overwriting progress."""

import argparse
from pathlib import Path

from eval.benchmark import PROJECT_ROOT
from eval.union_gold import migrate_human_gold, prepare_annotation_tasks
from eval.union_gold_io import expansion_paths, load_expansion_inputs, save_json_atomic


def prepare_tasks(benchmark_id: str, *, root: Path = PROJECT_ROOT) -> dict:
    manifest, gold, snapshots = load_expansion_inputs(benchmark_id, root=root)
    tasks = prepare_annotation_tasks(manifest, gold, snapshots)
    path = expansion_paths(benchmark_id, root=root).annotations
    save_json_atomic(path, tasks)
    inherited = len(migrate_human_gold(manifest, gold))
    print(f"{benchmark_id}: inherited={inherited}, new={len(tasks['tasks'])}, union={len(manifest['works'])}")
    print(f"Unlabeled tasks saved to {path}")
    return tasks


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    args = parser.parse_args(argv)
    prepare_tasks(args.benchmark)


if __name__ == "__main__":
    main()
