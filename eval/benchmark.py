"""Gold-free benchmark definitions and conventional artifact paths."""

from dataclasses import dataclass
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from researchpilot.paper_candidate import SearchQuery


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ID_PATTERN = r"^[a-z][a-z0-9_]*$"


class BenchmarkDefinition(BaseModel):
    """Search configuration only; gold judgments belong in a separate dataset."""

    model_config = ConfigDict(extra="forbid")

    benchmark_id: str = Field(strict=True, pattern=_ID_PATTERN)
    question: str = Field(strict=True, min_length=1)
    year_from: int | None = Field(strict=True)
    year_to: int | None = Field(strict=True)
    per_query: int = Field(strict=True, ge=1, le=100)
    queries: list[SearchQuery] = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def require_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must be a non-empty string.")
        return value

    @model_validator(mode="after")
    def validate_search_configuration(self) -> "BenchmarkDefinition":
        query_ids = [query.query_id for query in self.queries]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("query_id values must be unique.")
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must be less than or equal to year_to.")
        return self


@dataclass(frozen=True)
class BenchmarkPaths:
    definition: Path
    candidates: Path
    gold: Path
    run: Path


def benchmark_paths(
    benchmark_id: str, *, root: Path = PROJECT_ROOT
) -> BenchmarkPaths:
    """Resolve paths without reading datasets or accepting directory traversal."""
    if not isinstance(benchmark_id, str) or re.fullmatch(_ID_PATTERN, benchmark_id) is None:
        raise ValueError("benchmark_id must start with a-z and contain only a-z, 0-9, or _.")
    return BenchmarkPaths(
        definition=root / "eval/benchmarks" / f"{benchmark_id}.json",
        candidates=root / "eval/datasets" / f"{benchmark_id}_candidates.json",
        gold=root / "eval/datasets" / f"{benchmark_id}.json",
        run=root / "eval/runs" / f"{benchmark_id}_semantic_v1.json",
    )


def load_benchmark(
    benchmark_id: str, *, root: Path = PROJECT_ROOT
) -> BenchmarkDefinition:
    path = benchmark_paths(benchmark_id, root=root).definition
    if not path.is_file():
        raise ValueError(f"Unknown benchmark id {benchmark_id!r}: no definition at {path}.")
    definition = BenchmarkDefinition.model_validate_json(path.read_text(encoding="utf-8"))
    if definition.benchmark_id != benchmark_id:
        raise ValueError("Definition benchmark_id does not match the requested benchmark.")
    return definition
