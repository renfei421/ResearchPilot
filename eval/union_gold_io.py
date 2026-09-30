"""Offline artifact loading and atomic saves for Human union-gold expansion."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from eval.benchmark import PROJECT_ROOT, benchmark_paths
from eval.union_gold import prepare_annotation_tasks, validate_annotations


@dataclass(frozen=True)
class ExpansionPaths:
    manifest: Path
    human_gold: Path
    human_candidates: Path
    llm_candidates: Path
    annotations: Path
    union_gold: Path


def expansion_paths(benchmark_id: str, *, root: Path = PROJECT_ROOT) -> ExpansionPaths:
    paths = benchmark_paths(benchmark_id, root=root)
    return ExpansionPaths(
        manifest=root / "eval/alignment" / f"{benchmark_id}_union_v1.json",
        human_gold=paths.gold,
        human_candidates=paths.candidates,
        llm_candidates=root / "eval/datasets" / f"{benchmark_id}_llm_candidates_v1.json",
        annotations=root / "eval/annotations" / f"{benchmark_id}_new_gold_tasks_v1.json",
        union_gold=root / "eval/datasets" / f"{benchmark_id}_union_gold_v1.json",
    )


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}.")
    return data


def load_expansion_inputs(benchmark_id: str, *, root: Path = PROJECT_ROOT) -> tuple:
    paths = expansion_paths(benchmark_id, root=root)
    manifest = read_json(paths.manifest)
    if manifest.get("benchmark_id") != benchmark_id:
        raise ValueError("Manifest benchmark_id does not match the requested benchmark.")
    return manifest, read_json(paths.human_gold), {
        "human": read_json(paths.human_candidates),
        "llm": read_json(paths.llm_candidates),
    }


def load_annotation_state(benchmark_id: str, *, root: Path = PROJECT_ROOT) -> tuple:
    paths = expansion_paths(benchmark_id, root=root)
    expected = prepare_annotation_tasks(*load_expansion_inputs(benchmark_id, root=root))
    original = paths.annotations.read_bytes()
    document = validate_annotations(json.loads(original), expected)
    return paths.annotations, document, original


def save_json_atomic(
    path: Path, document: dict[str, Any], *, expected_bytes: bytes | None = None
) -> bytes:
    """Create a new artifact or replace the exact annotation revision we loaded.

Refuse existing outputs by default. Write in the same directory, flush, then
replace atomically so interruption cannot truncate the previous judgments.
"""
    content = (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")

    def check_destination() -> None:
        if path.is_symlink():
            raise ValueError(f"Output must not be a symbolic link: {path}.")
        if expected_bytes is None:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing artifact: {path}.")
        elif not path.is_file() or path.read_bytes() != expected_bytes:
            raise ValueError("Annotation file changed since loading; reload before saving.")

    check_destination()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        check_destination()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return content
