"""Offline frozen-plan execution using handcrafted papers and isolated files."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from researchpilot.paper import Paper
from scripts.build_llm_plan_candidates import build_llm_plan_candidates, load_frozen_plan, main


def frozen_plan_data(benchmark_id="cache_v1"):
    question = f"How do shared caches affect latency ({benchmark_id})?"
    return {
        "benchmark_id": benchmark_id,
        "planner_version": "v1",
        "model": "gpt-5.6-terra",
        "generated_at": "2026-09-17T00:00:00+00:00",
        "request": {
            "research_question": question, "year_from": 2019,
            "year_to": 2024, "max_queries": 5,
        },
        "plan": {
            "research_question": question,
            "concepts": ["shared caches", "latency"],
            "queries": [
                {"query_id": "Q-2", "text": '"shared cache"  AND Latency',
                 "role": "core", "rationale": "Study cache latency."},
                {"query_id": "q_a", "text": '"cache contention" AND throughput',
                 "role": "facet", "rationale": "Study contention effects."},
                {"query_id": "bridge:α", "text": '"request scheduling" AND caching',
                 "role": "bridge", "rationale": "Find useful scheduling methods."},
            ],
        },
    }


def paper(identifier, title, year, authors):
    return Paper(
        paper_id=f"doi:10.1000/{identifier}", title=title,
        abstract=f"Handcrafted abstract for {identifier}.", authors=authors,
        publication_year=year, doi=f"10.1000/{identifier}", citation_count=0,
        source="openalex", source_id=identifier, open_access_url=None,
    )


class FakeSearchClient:
    def __init__(self):
        self.first_version = paper("a", "Shared Cache Latency", 2020, ["A. Researcher"])
        self.other_version = paper("b", "shared-cache latency", 2021, ["A. Researcher"])
        self.unrelated = paper("u", "Queue stability", 2020, ["B. Researcher"])
        # First-seen group order differs from RRF order. Both versions have
        # overlapping query hits, so grouping and group-level fusion matter.
        self.responses = [
            [self.unrelated, self.first_version, self.other_version],
            [self.other_version, self.first_version],
            [self.first_version],
        ]
        self.calls = []
        self.fail_on = None
        self.error = RuntimeError("Offline retrieval fixture failure")

    def search_works(self, query, per_page=5, year_from=None, year_to=None):
        self.calls.append({
            "query": query, "per_page": per_page,
            "year_from": year_from, "year_to": year_to,
        })
        if len(self.calls) == self.fail_on:
            raise self.error
        return self.responses[(len(self.calls) - 1) % len(self.responses)]


class BuildLLMPlanCandidatesTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / f"llm-candidate-fixture-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self.remove_fixture)
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.enterContext(redirect_stdout(self.stdout))
        self.enterContext(redirect_stderr(self.stderr))
        for target in ("socket.create_connection", "socket.socket.connect",
                       "scripts.build_llm_plan_candidates.OpenAlexClient",
                       "researchpilot.openai_query_planner.OpenAIQueryPlanner.plan"):
            self.enterContext(patch(target, side_effect=AssertionError("No live providers in tests")))
        self.client = FakeSearchClient()
        self.frozen = self.write_fixture("cache_v1")

    def remove_fixture(self):
        target = self.root.resolve()
        if target.parent != Path(__file__).resolve().parent or not target.name.startswith("llm-candidate-fixture-"):
            raise AssertionError(f"Unsafe fixture cleanup path: {target}")
        shutil.rmtree(target)

    def write_json(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def plan_path(self, benchmark_id="cache_v1"):
        return self.root / "eval/plans" / f"{benchmark_id}_llm_plan_v1.json"

    def definition_path(self, benchmark_id="cache_v1"):
        return self.root / "eval/benchmarks" / f"{benchmark_id}.json"

    def output_path(self, benchmark_id="cache_v1"):
        return self.root / "eval/datasets" / f"{benchmark_id}_llm_candidates_v1.json"

    def write_fixture(self, benchmark_id):
        frozen = frozen_plan_data(benchmark_id)
        self.write_json(self.plan_path(benchmark_id), frozen)
        self.write_json(self.definition_path(benchmark_id), {
            "benchmark_id": benchmark_id, "per_query": 7,
            "question": "DEFINITION_QUESTION_SENTINEL",
            "year_from": 1900, "year_to": 1901,
            "queries": [{"query_id": "human_q1", "text": "HUMAN_QUERY_SENTINEL"}],
        })
        return frozen

    def build(self, ids=None, **kwargs):
        return build_llm_plan_candidates(
            ["cache_v1"] if ids is None else ids, client=self.client, root=self.root, **kwargs,
        )

    def read_snapshot(self, benchmark_id="cache_v1"):
        return json.loads(self.output_path(benchmark_id).read_text(encoding="utf-8"))

    def test_loads_matching_frozen_plan_using_existing_contracts(self):
        frozen = load_frozen_plan("cache_v1", root=self.root)
        self.assertEqual(frozen.model_dump(mode="json"), self.frozen)

    def test_rejects_mismatched_benchmark_version_or_question_before_search(self):
        for field, value in (("benchmark_id", "other_v1"), ("planner_version", "v2"),
                             ("research_question", "Different question")):
            with self.subTest(field=field):
                frozen = deepcopy(self.frozen)
                if field == "research_question":
                    frozen["plan"][field] = value
                else:
                    frozen[field] = value
                self.write_json(self.plan_path(), frozen)
                with self.assertRaises(ValueError):
                    self.build()
        self.assertEqual(self.client.calls, [])
        self.assertFalse(self.output_path().exists())

    def test_malformed_plans_and_query_counts_are_rejected_before_search(self):
        for plan in (None, "invalid", {**self.frozen["plan"], "queries": []},
                     {**self.frozen["plan"], "queries": self.frozen["plan"]["queries"][:2]},
                     {**self.frozen["plan"], "queries": self.frozen["plan"]["queries"] * 2}):
            with self.subTest(plan=plan):
                self.write_json(self.plan_path(), {**self.frozen, "plan": plan})
                with self.assertRaises(ValueError):
                    self.build()
        self.assertEqual(self.client.calls, [])

    def test_invalid_year_bounds_are_rejected_before_search(self):
        for bounds in ({"year_from": True}, {"year_to": False}, {"year_from": "2020"},
                       {"year_to": 2024.0}, {"year_from": 2025, "year_to": 2024}):
            with self.subTest(bounds=bounds):
                frozen = deepcopy(self.frozen)
                frozen["request"].update(bounds)
                self.write_json(self.plan_path(), frozen)
                with self.assertRaises(ValueError):
                    self.build()
        self.assertEqual(self.client.calls, [])

    def test_frozen_queries_years_and_definition_per_query_are_used_exactly(self):
        paths = self.build()
        self.assertEqual(paths, [self.output_path()])
        self.assertEqual(self.client.calls, [
            {"query": query["text"], "per_page": 7, "year_from": 2019, "year_to": 2024}
            for query in self.frozen["plan"]["queries"]
        ])

    def test_nullable_plan_years_override_definition_years(self):
        self.frozen["request"].update(year_from=None, year_to=None)
        self.write_json(self.plan_path(), self.frozen)
        self.build()
        self.assertTrue(all(call["year_from"] is None and call["year_to"] is None for call in self.client.calls))

    def test_existing_version_grouping_and_rrf_run_and_preserve_candidate_fields(self):
        self.build()
        records = self.read_snapshot()["candidates"]
        self.assertEqual(len(records), 2)
        grouped, unrelated = records
        representative = self.client.first_version
        self.assertEqual([row["rrf_rank"] for row in records], [1, 2])
        self.assertEqual([row["paper_id"] for row in records], [representative.paper_id, self.client.unrelated.paper_id])
        self.assertEqual(grouped["version_count"], 2)
        self.assertEqual(unrelated["version_count"], 1)
        self.assertEqual(grouped["title"], representative.title)
        self.assertEqual(grouped["year"], representative.publication_year)
        self.assertEqual(grouped["abstract"], representative.abstract)
        self.assertTrue(all(row["group_id"].startswith("version:") for row in records))
        self.assertNotEqual(grouped["group_id"], unrelated["group_id"])
        self.assertAlmostEqual(grouped["rrf_score"], 1 / 62 + 1 / 61 + 1 / 61)
        self.assertAlmostEqual(unrelated["rrf_score"], 1 / 61)
        self.assertEqual(grouped["hits"], [
            {"query_id": query["query_id"], "query_text": query["text"], "rank": rank}
            for query, rank in zip(self.frozen["plan"]["queries"], [2, 1, 1])
        ])
        self.assertEqual(set(grouped), {
            "rrf_rank", "group_id", "rrf_score", "title", "paper_id", "year", "abstract", "version_count", "hits",
        })

    def test_snapshot_records_llm_source_and_original_plan_metadata(self):
        self.build()
        saved = self.read_snapshot()
        self.assertEqual(set(saved), {
            "benchmark_id", "source", "planner_version", "planner_model", "question", "plan_path", "candidates",
        })
        self.assertEqual(saved["benchmark_id"], "cache_v1")
        self.assertEqual(saved["source"], "llm_query_plan_v1")
        self.assertEqual(saved["planner_version"], "v1")
        self.assertEqual(saved["planner_model"], self.frozen["model"])
        self.assertEqual(saved["question"], self.frozen["plan"]["research_question"])
        self.assertEqual(saved["plan_path"], "eval/plans/cache_v1_llm_plan_v1.json")

    def test_only_allowlisted_definition_fields_are_consumed(self):
        class ConfigurationOnly(dict):
            def get(self, key, default=None):
                if key not in {"benchmark_id", "per_query"}:
                    raise AssertionError(f"Forbidden definition field: {key}")
                return super().get(key, default)

            def __getitem__(self, key):
                if key not in {"benchmark_id", "per_query"}:
                    raise AssertionError(f"Forbidden definition field: {key}")
                return super().__getitem__(key)

        # Also make human query data invalid for BenchmarkDefinition validation.
        definition = ConfigurationOnly(benchmark_id="cache_v1", per_query=7, queries="FORBIDDEN_QUERY_SENTINEL")
        with patch("scripts.build_llm_plan_candidates.json.loads", return_value=definition):
            self.build()
        self.assertEqual(len(self.client.calls), 3)
        serialized = self.output_path().read_text(encoding="utf-8")
        for sentinel in ("FORBIDDEN_QUERY_SENTINEL", "HUMAN_QUERY_SENTINEL", "DEFINITION_QUESTION_SENTINEL"):
            self.assertNotIn(sentinel, serialized)

    def test_only_plan_and_definition_are_read_and_human_artifacts_stay_unchanged(self):
        forbidden = [self.root / "eval" / relative for relative in (
            "datasets/cache_v1.json", "datasets/cache_v1_candidates.json",
            "runs/cache_v1_semantic_v1.json", "reports/previous.json",
        )]
        for path in forbidden:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"FORBIDDEN_ARTIFACT_SENTINEL")
        sources = [self.plan_path(), self.definition_path(), *forbidden]
        before = {path: path.read_bytes() for path in sources}
        allowed_reads = {self.plan_path().resolve(), self.definition_path().resolve()}
        reads = []
        real_open = io.open

        def guarded_open(file, mode="r", *args, **kwargs):
            path = Path(file).resolve()
            if "r" in mode or "+" in mode:
                self.assertIn(path, allowed_reads)
                reads.append(path)
            else:
                self.assertEqual(path, self.output_path().resolve())
            return real_open(file, mode, *args, **kwargs)

        with patch("io.open", side_effect=guarded_open), patch("builtins.open", side_effect=guarded_open), \
             patch("os.scandir", side_effect=AssertionError("No artifact directory scans")), \
             patch("os.listdir", side_effect=AssertionError("No artifact directory scans")):
            self.build()
        self.assertEqual(reads, [self.plan_path().resolve(), self.definition_path().resolve()])
        self.assertEqual({path: path.read_bytes() for path in sources}, before)
        self.assertNotIn("FORBIDDEN_ARTIFACT_SENTINEL", self.output_path().read_text(encoding="utf-8"))

    def test_api_keys_environment_and_extra_plan_metadata_are_not_saved(self):
        self.frozen["api_key"] = "FROZEN_EXTRA_SECRET_SENTINEL"
        self.write_json(self.plan_path(), self.frozen)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "OPENAI_SECRET_SENTINEL", "OPENALEX_API_KEY": "OPENALEX_SECRET_SENTINEL"}):
            self.build()
        serialized = self.output_path().read_text(encoding="utf-8")
        for secret in ("api_key", "FROZEN_EXTRA_SECRET_SENTINEL", "OPENAI_SECRET_SENTINEL", "OPENALEX_SECRET_SENTINEL"):
            self.assertNotIn(secret, serialized)

    def test_existing_llm_snapshot_is_protected_before_search(self):
        self.output_path().parent.mkdir(parents=True)
        self.output_path().write_bytes(b"previous frozen snapshot")
        with self.assertRaisesRegex(FileExistsError, "--overwrite"):
            self.build()
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.output_path().read_bytes(), b"previous frozen snapshot")

    def test_explicit_overwrite_changes_only_llm_snapshot(self):
        self.output_path().parent.mkdir(parents=True)
        self.output_path().write_bytes(b"previous frozen LLM snapshot")
        human = self.root / "eval/datasets/cache_v1_candidates.json"
        human.write_bytes(b"frozen human snapshot")
        self.build(overwrite=True)
        self.assertEqual(self.read_snapshot()["source"], "llm_query_plan_v1")
        self.assertEqual(human.read_bytes(), b"frozen human snapshot")

    def test_retrieval_failure_propagates_without_partial_snapshot_or_skipped_query(self):
        before = self.plan_path().read_bytes()
        self.client.fail_on = 2
        with self.assertRaises(RuntimeError) as raised:
            self.build()
        self.assertIs(raised.exception, self.client.error)
        self.assertEqual(len(self.client.calls), 2)
        self.assertFalse(self.output_path().exists())
        self.assertEqual(self.plan_path().read_bytes(), before)
        self.assertIn("cache_v1", self.stderr.getvalue())

    def test_failed_overwrite_preserves_existing_snapshot(self):
        self.output_path().parent.mkdir(parents=True)
        self.output_path().write_bytes(b"previous completed snapshot")
        self.client.fail_on = 2
        with self.assertRaises(RuntimeError):
            self.build(overwrite=True)
        self.assertEqual(self.output_path().read_bytes(), b"previous completed snapshot")

    def test_multiple_benchmarks_preserve_supplied_order(self):
        other = self.write_fixture("other_v1")
        other["plan"]["queries"][0]["text"] = '"shared buffering" AND latency'
        self.write_json(self.plan_path("other_v1"), other)
        paths = self.build(["other_v1", "cache_v1"])
        self.assertEqual(paths, [self.output_path("other_v1"), self.output_path()])
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertEqual([call["query"] for call in self.client.calls], [
            query["text"] for frozen in (other, self.frozen) for query in frozen["plan"]["queries"]
        ])
        self.assertEqual(self.stdout.getvalue().splitlines(), [
            "[1/2] building LLM candidates for other_v1", "[2/2] building LLM candidates for cache_v1",
        ])

    def test_multi_benchmark_failure_keeps_completed_outputs_and_stops(self):
        self.write_fixture("second_v1")
        self.write_fixture("third_v1")
        self.client.fail_on = 5
        with self.assertRaises(RuntimeError) as raised:
            self.build(["cache_v1", "second_v1", "third_v1"])
        self.assertIs(raised.exception, self.client.error)
        self.assertEqual(len(self.read_snapshot()["candidates"]), 2)
        self.assertFalse(self.output_path("second_v1").exists())
        self.assertFalse(self.output_path("third_v1").exists())
        self.assertEqual(len(self.client.calls), 5)
        self.assertIn("second_v1", self.stderr.getvalue())
        self.assertNotIn("third_v1", self.stdout.getvalue())

    def test_empty_results_produce_a_valid_empty_snapshot(self):
        self.client.responses = [[], [], []]
        self.build()
        self.assertEqual(self.read_snapshot()["candidates"], [])
        self.assertEqual(len(self.client.calls), 3)

    def test_missing_plan_or_definition_fails_before_search(self):
        with self.assertRaisesRegex(FileNotFoundError, "Frozen LLM plan not found"):
            self.build(["unknown_v1"])
        self.definition_path().unlink()
        with self.assertRaisesRegex(ValueError, "Unknown benchmark id"):
            self.build()
        self.assertEqual(self.client.calls, [])

    def test_invalid_per_query_and_definition_id_fail_before_search(self):
        for per_query in (None, True, 0, 101, 7.0, "7"):
            with self.subTest(per_query=per_query):
                self.write_json(self.definition_path(), {"benchmark_id": "cache_v1", "per_query": per_query})
                with self.assertRaisesRegex(ValueError, "per_query"):
                    self.build()
        self.write_json(self.definition_path(), {"benchmark_id": "other_v1", "per_query": 7})
        with self.assertRaisesRegex(ValueError, "benchmark_id"):
            self.build()
        self.assertEqual(self.client.calls, [])

    def test_empty_duplicate_and_unsafe_ids_fail_without_search(self):
        for ids in ([], ["cache_v1", "cache_v1"], ["../outside"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.build(ids)
        self.assertEqual(self.client.calls, [])

    def test_default_openalex_dependency_is_constructed_only_for_valid_execution(self):
        with patch("scripts.build_llm_plan_candidates.OpenAlexClient", return_value=self.client) as factory:
            build_llm_plan_candidates(["cache_v1"], root=self.root)
        factory.assert_called_once_with()
        self.assertEqual(len(self.client.calls), 3)

    def test_cli_passes_order_and_only_explicit_overwrite(self):
        ids = ["matrix_completion_v1", "rag_hallucination_v1", "cot_reasoning_faithfulness_v1"]
        for overwrite in (False, True):
            with self.subTest(overwrite=overwrite), patch("scripts.build_llm_plan_candidates.build_llm_plan_candidates") as build:
                main(["--benchmarks", *ids, *(["--overwrite"] if overwrite else [])])
                build.assert_called_once_with(ids, overwrite=overwrite)


if __name__ == "__main__":
    unittest.main()
