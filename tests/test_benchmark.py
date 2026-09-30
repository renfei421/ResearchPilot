"""Offline benchmark definition, path, and candidate-builder tests."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, call, patch
from uuid import uuid4

from pydantic import ValidationError

from eval.benchmark import (
    PROJECT_ROOT, BenchmarkDefinition, benchmark_paths, load_benchmark,
)
from researchpilot.paper import Paper
from researchpilot.paper_candidate import SearchQuery
from researchpilot.paper_search_service import PaperSearchClient
from scripts import build_benchmark_candidates as builder


def definition_data() -> dict:
    return {
        "benchmark_id": "test_topic_v1",
        "question": "How do fixed sampling patterns affect recovery?",
        "year_from": 2015,
        "year_to": 2026,
        "per_query": 7,
        "queries": [
            {"query_id": "q1", "text": '"matrix completion" AND deterministic'},
            {"query_id": "q2", "text": '"matrix completion" AND "sampling pattern"'},
        ],
    }


class BenchmarkDefinitionTests(unittest.TestCase):
    def test_matrix_definition_preserves_question_queries_and_search_parameters(self):
        definition = load_benchmark("matrix_completion_v1")
        gold_path = benchmark_paths(definition.benchmark_id).gold
        gold = json.loads(gold_path.read_text(encoding="utf-8"))

        self.assertEqual(definition.question, gold["research_question"])
        self.assertEqual([query.model_dump() for query in definition.queries], gold["search_queries"])
        self.assertTrue(all(isinstance(query, SearchQuery) for query in definition.queries))
        self.assertEqual((definition.year_from, definition.year_to, definition.per_query), (2015, 2026, 10))
        self.assertEqual(
            set(definition.model_dump()),
            {"benchmark_id", "question", "year_from", "year_to", "per_query", "queries"},
        )

    def test_second_definition_has_the_supplied_question_and_queries(self):
        definition = load_benchmark("rag_hallucination_v1")

        self.assertEqual(
            definition.question,
            "How does retrieval-augmented generation reduce hallucination and "
            "improve factuality in large language models?",
        )
        self.assertEqual((definition.year_from, definition.year_to, definition.per_query), (2020, 2026, 10))
        self.assertEqual(
            [query.model_dump() for query in definition.queries],
            [
                {"query_id": "q1", "text": '"retrieval augmented generation" AND hallucination'},
                {"query_id": "q2", "text": '"retrieval augmented generation" AND factuality'},
                {"query_id": "q3", "text": '"retrieval augmented generation" AND grounding'},
            ],
        )

    def test_cot_definition_loads_with_exact_question_and_bounds(self):
        definition = load_benchmark("cot_reasoning_faithfulness_v1")

        self.assertIsInstance(definition, BenchmarkDefinition)
        self.assertEqual(definition.benchmark_id, "cot_reasoning_faithfulness_v1")
        self.assertEqual(
            definition.question,
            "How does chain-of-thought prompting affect reasoning accuracy and "
            "faithfulness in large language models?",
        )
        self.assertEqual(
            (definition.year_from, definition.year_to, definition.per_query),
            (2022, 2026, 10),
        )

    def test_cot_definition_preserves_exact_query_ids_texts_and_order(self):
        definition = load_benchmark("cot_reasoning_faithfulness_v1")

        self.assertEqual(
            [query.model_dump() for query in definition.queries],
            [
                {"query_id": "q1", "text": '"chain of thought" AND reasoning'},
                {"query_id": "q2", "text": '"chain of thought" AND faithfulness'},
                {"query_id": "q3", "text": '"chain of thought" AND accuracy'},
            ],
        )

    def test_invalid_definitions_are_rejected(self):
        invalid_updates = [
            {"benchmark_id": "../outside"},
            {"question": " \n"},
            {"question": 123},
            {"per_query": True},
            {"per_query": "10"},
            {"per_query": 0},
            {"per_query": 101},
            {"year_from": True},
            {"year_to": "2026"},
            {"year_from": 2027, "year_to": 2026},
            {"queries": []},
            {"queries": [{"query_id": " ", "text": "query"}]},
            {"queries": [{"query_id": "q1", "text": " "}]},
            {"queries": [{"query_id": "q1", "text": "one"}, {"query_id": "q1", "text": "two"}]},
            {"judgments": {"paper": 2}},
        ]
        for update in invalid_updates:
            with self.subTest(update=update), self.assertRaises(ValidationError):
                BenchmarkDefinition.model_validate(definition_data() | update)
        for field in definition_data():
            data = definition_data()
            del data[field]
            with self.subTest(missing=field), self.assertRaises(ValidationError):
                BenchmarkDefinition.model_validate(data)

    def test_nullable_years_and_exact_question_are_supported(self):
        question = "  Question with significant original formatting\n"
        definition = BenchmarkDefinition.model_validate(
            definition_data() | {"question": question, "year_from": None, "year_to": None}
        )
        self.assertEqual(definition.question, question)
        self.assertIsNone(definition.year_from)
        self.assertIsNone(definition.year_to)

    def test_generic_paths(self):
        for benchmark_id in ("matrix_completion_v1", "rag_hallucination_v1", "new_topic_v2"):
            with self.subTest(benchmark_id=benchmark_id):
                paths = benchmark_paths(benchmark_id)
                self.assertEqual(paths.definition, PROJECT_ROOT / "eval/benchmarks" / f"{benchmark_id}.json")
                self.assertEqual(paths.candidates, PROJECT_ROOT / "eval/datasets" / f"{benchmark_id}_candidates.json")
                self.assertEqual(paths.gold, PROJECT_ROOT / "eval/datasets" / f"{benchmark_id}.json")
                self.assertEqual(paths.run, PROJECT_ROOT / "eval/runs" / f"{benchmark_id}_semantic_v1.json")

    def test_unsafe_benchmark_ids_are_rejected(self):
        for benchmark_id in ("", "../outside", "..\\outside", "/absolute", "with space", None):
            with self.subTest(benchmark_id=benchmark_id), self.assertRaises(ValueError):
                benchmark_paths(benchmark_id)

    def test_unknown_benchmark_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown benchmark id 'unknown_topic_v1'"):
            load_benchmark("unknown_topic_v1")

    def test_definition_id_must_match_requested_id(self):
        with patch.object(Path, "is_file", return_value=True):
            with patch.object(Path, "read_text", return_value=json.dumps(definition_data())):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    load_benchmark("different_topic_v1")

    def test_matrix_canonical_snapshot_is_an_exact_copy_of_the_frozen_original(self):
        legacy = PROJECT_ROOT / "eval/datasets/matrix_completion_candidates_v1.json"
        canonical = benchmark_paths("matrix_completion_v1").candidates
        self.assertEqual(canonical.read_bytes(), legacy.read_bytes())


def make_paper(source_id: str, title: str, year: int) -> Paper:
    return Paper(
        paper_id=f"openalex:{source_id}", title=title, abstract="Handcrafted abstract.",
        authors=["Example Author"], publication_year=year, doi=None, citation_count=0,
        source="openalex", source_id=source_id, open_access_url=None,
    )


class BenchmarkCandidateBuilderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / f"benchmark-build-{uuid4().hex}"
        self.root.mkdir()
        self.definition = definition_data()
        self.benchmark_id = self.definition["benchmark_id"]
        self.paths = benchmark_paths(self.benchmark_id, root=self.root)
        self.paths.definition.parent.mkdir(parents=True)
        self.paths.definition.write_text(json.dumps(self.definition), encoding="utf-8")
        self.addCleanup(self._cleanup)
        self.first = make_paper("first", "Fixed patterns", 2015)
        self.version = make_paper("version", "FIXED PATTERNS", 2016)
        self.other = make_paper("other", "Different topic", 2015)
        self.client = Mock(spec=PaperSearchClient)
        self.client.search_works.side_effect = [
            [self.first, self.other], [self.version, self.first]
        ]
        constructor = patch.object(builder, "OpenAlexClient", side_effect=AssertionError("No API"))
        constructor.start()
        self.addCleanup(constructor.stop)

    def _cleanup(self):
        self.paths.candidates.unlink(missing_ok=True)
        self.paths.definition.unlink(missing_ok=True)
        for directory in (self.paths.candidates.parent, self.paths.definition.parent):
            if directory.exists():
                directory.rmdir()
        (self.root / "eval").rmdir()
        self.root.rmdir()

    def _build(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return builder.build_candidates(
                self.benchmark_id, client=self.client, root=self.root, **kwargs
            )

    def test_builder_uses_definition_and_preserves_group_metadata_and_fused_hits(self):
        before = self.paths.definition.read_bytes()

        snapshot = self._build()

        self.assertEqual(
            self.client.search_works.call_args_list,
            [call(query=query["text"], per_page=7, year_from=2015, year_to=2026)
             for query in self.definition["queries"]],
        )
        self.assertEqual(snapshot["benchmark_id"], self.benchmark_id)
        self.assertEqual(snapshot["question"], self.definition["question"])
        self.assertEqual(len(snapshot["candidates"]), 2)
        first = snapshot["candidates"][0]
        self.assertEqual(first["rrf_rank"], 1)
        self.assertTrue(first["group_id"].startswith("version:"))
        self.assertEqual(first["rrf_score"], 2 / 61)
        self.assertEqual(first["title"], self.first.title)
        self.assertEqual(first["paper_id"], self.first.paper_id)
        self.assertEqual(first["year"], 2015)
        self.assertEqual(first["abstract"], self.first.abstract)
        self.assertEqual(first["version_count"], 2)
        self.assertEqual(
            first["hits"],
            [{"query_id": query["query_id"], "query_text": query["text"], "rank": 1}
             for query in self.definition["queries"]],
        )
        self.assertEqual(json.loads(self.paths.candidates.read_text(encoding="utf-8")), snapshot)
        self.assertEqual(self.paths.definition.read_bytes(), before)
        self.assertFalse(self.paths.gold.exists())

    def test_snapshot_requires_explicit_overwrite_before_any_search(self):
        self.paths.candidates.parent.mkdir()
        self.paths.candidates.write_text("existing frozen snapshot", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "--overwrite"):
            self._build()

        self.client.search_works.assert_not_called()
        self.assertEqual(self.paths.candidates.read_text(encoding="utf-8"), "existing frozen snapshot")
        snapshot = self._build(overwrite=True)
        self.assertEqual(json.loads(self.paths.candidates.read_text(encoding="utf-8")), snapshot)

    def test_search_failure_does_not_overwrite_existing_snapshot(self):
        self.paths.candidates.parent.mkdir()
        self.paths.candidates.write_text("existing frozen snapshot", encoding="utf-8")
        failure = RuntimeError("Search failed")
        self.client.search_works.side_effect = failure
        with self.assertRaises(RuntimeError) as caught:
            self._build(overwrite=True)
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.paths.candidates.read_text(encoding="utf-8"), "existing frozen snapshot")

    def test_unknown_benchmark_fails_before_any_search(self):
        with self.assertRaisesRegex(ValueError, "Unknown benchmark id"):
            builder.build_candidates("unknown_topic_v1", client=self.client, root=self.root)
        self.client.search_works.assert_not_called()

    def test_builder_cli_passes_benchmark_and_overwrite_flag(self):
        with patch.object(builder, "build_candidates") as build:
            builder.main(["--benchmark", self.benchmark_id])
            build.assert_called_once_with(self.benchmark_id, overwrite=False)
            build.reset_mock()
            builder.main(["--benchmark", self.benchmark_id, "--overwrite"])
            build.assert_called_once_with(self.benchmark_id, overwrite=True)


if __name__ == "__main__":
    unittest.main()
