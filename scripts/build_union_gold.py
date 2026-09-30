"""Build final canonical union gold only after all manual tasks are complete."""

import argparse
from pathlib import Path

from eval.benchmark import PROJECT_ROOT
from eval.union_gold import build_union_gold
from eval.union_gold_io import expansion_paths, load_expansion_inputs, read_json, save_json_atomic


def build_gold(benchmark_id: str, *, root: Path = PROJECT_ROOT) -> dict:
    paths = expansion_paths(benchmark_id, root=root)
    result = build_union_gold(
        *load_expansion_inputs(benchmark_id, root=root), read_json(paths.annotations)
    )
    save_json_atomic(paths.union_gold, result)
    print(f"{benchmark_id}: {len(result['judgments'])} complete union judgments saved to {paths.union_gold}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    args = parser.parse_args(argv)
    build_gold(args.benchmark)


if __name__ == "__main__":
    main()
