"""Offline freezing tests with independent definitions and a fake planner."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from researchpilot.query_plan import SearchPlan
from researchpilot.query_planner import PlannerRequest
from scripts.generate_query_plans import generate_query_plans, main


class FakePlanner:
    def __init__(self, *, fail_on: int | None = None):
        self.requests: list[PlannerRequest] = []
        self.plans: list[SearchPlan] = []
        self.fail_on = fail_on
        self.error = RuntimeError("Offline fixture planning failure")

    def plan(self, request: PlannerRequest) -> SearchPlan:
        self.requests.append(request)
        if len(self.requests) == self.fail_on:
            raise self.error
        result = SearchPlan(
            research_question=request.research_question,
            concepts=["shared caches", "latency"],
            queries=[
                {"query_id": "q1", "text": '"shared cache" AND latency',
                 "role": "core", "rationale": "Study cache latency."},
                {"query_id": "q2", "text": '"cache contention" AND throughput',
                 "role": "facet", "rationale": "Study contention effects."},
                {"query_id": "q3", "text": '"request scheduling" AND caching',
                 "role": "bridge", "rationale": "Find relevant scheduling methods."},
            ],
        )
        self.plans.append(result)
        return result


class GenerateQueryPlansTests(unittest.TestCase):
    def setUp(self):
        # Default mkdir permissions keep fixtures readable on Windows too.
        self.root = Path(__file__).resolve().parent / f"plan-fixture-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self.remove_fixture)
        self.planner = FakePlanner()
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.enterContext(redirect_stdout(self.stdout))
        self.enterContext(redirect_stderr(self.stderr))
        for target in ("socket.create_connection", "socket.socket.connect",
                       "scripts.generate_query_plans.OpenAIQueryPlanner"):
            self.enterContext(patch(target, side_effect=AssertionError("Tests must remain offline")))
        self.definition = self.write_definition("cache_v1")

    def remove_fixture(self):
        target = self.root.resolve()
        tests_root = Path(__file__).resolve().parent
        if target.parent != tests_root or not target.name.startswith("plan-fixture-"):
            raise AssertionError(f"Unsafe fixture cleanup path: {target}")
        shutil.rmtree(target)

    def write_definition(self, benchmark_id, **changes):
        definition = {
            "benchmark_id": benchmark_id,
            "question": f"How do shared caches affect latency at α scale ({benchmark_id})?",
            "year_from": 2020,
            "year_to": 2026,
            "per_query": 10,
            "queries": [{"query_id": "human_q1", "text": "HANDCRAFTED_BASELINE_SENTINEL"}],
        }
        definition.update(changes)
        path = self.root / "eval/benchmarks" / f"{benchmark_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(definition, ensure_ascii=False), encoding="utf-8")
        return definition

    def output_path(self, benchmark_id="cache_v1"):
        return self.root / "eval/plans" / f"{benchmark_id}_llm_plan_v1.json"

    def generate(self, ids=None, **kwargs):
        return generate_query_plans(
            ["cache_v1"] if ids is None else ids,
            root=self.root, planner=self.planner, **kwargs,
        )

    def test_single_plan_records_request_plan_and_metadata(self):
        before = datetime.now(timezone.utc)
        paths = self.generate()
        after = datetime.now(timezone.utc)
        self.assertEqual(paths, [self.output_path()])
        saved = json.loads(paths[0].read_text(encoding="utf-8"))
        self.assertEqual(set(saved), {
            "benchmark_id", "planner_version", "model", "generated_at", "request", "plan",
        })
        self.assertEqual(saved["benchmark_id"], "cache_v1")
        self.assertEqual(saved["planner_version"], "v1")
        self.assertEqual(saved["model"], "gpt-5.6-terra")
        timestamp = datetime.fromisoformat(saved["generated_at"])
        self.assertEqual(timestamp.utcoffset(), timedelta(0))
        self.assertLessEqual(before, timestamp)
        self.assertLessEqual(timestamp, after)
        expected_request = {
            "research_question": self.definition["question"],
            "year_from": 2020, "year_to": 2026, "max_queries": 5,
        }
        self.assertEqual(len(self.planner.requests), 1)
        self.assertEqual(self.planner.requests[0].model_dump(), expected_request)
        self.assertEqual(saved["request"], expected_request)
        self.assertEqual(saved["plan"], self.planner.plans[0].model_dump(mode="json"))
        self.assertEqual(SearchPlan.model_validate(saved["plan"]), self.planner.plans[0])

    def test_multiple_benchmarks_preserve_supplied_order_and_print_progress(self):
        other = self.write_definition("other_v1", year_from=None, year_to=None)
        paths = self.generate(["other_v1", "cache_v1"])
        self.assertEqual(paths, [self.output_path("other_v1"), self.output_path()])
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertEqual(
            [request.research_question for request in self.planner.requests],
            [other["question"], self.definition["question"]],
        )
        self.assertIsNone(self.planner.requests[0].year_from)
        self.assertIsNone(self.planner.requests[0].year_to)
        self.assertEqual([request.max_queries for request in self.planner.requests], [5, 5])
        self.assertEqual(self.stdout.getvalue().splitlines(), [
            "[1/2] planning other_v1", "[2/2] planning cache_v1",
        ])

    def test_ignores_handcrafted_queries_and_every_non_allowlisted_field(self):
        self.write_definition(
            "cache_v1", queries={"invalid_query_schema": "HUMAN_QUERY_SENTINEL"},
            per_query="PER_QUERY_SENTINEL", max_queries=3,
            notes="UNRELATED_METADATA_SENTINEL",
        )
        self.generate()
        self.assertEqual(self.planner.requests[0].model_dump(), {
            "research_question": self.definition["question"],
            "year_from": 2020, "year_to": 2026, "max_queries": 5,
        })
        serialized = self.output_path().read_text(encoding="utf-8")
        for sentinel in ("HUMAN_QUERY_SENTINEL", "PER_QUERY_SENTINEL", "UNRELATED_METADATA_SENTINEL"):
            self.assertNotIn(sentinel, serialized)

    def test_reads_only_definition_and_never_gold_candidates_runs_or_reports(self):
        # Poison artifacts are unrelated local fixtures, never formal benchmark data.
        for relative in ("datasets/cache_v1.json", "datasets/cache_v1_candidates.json",
                         "runs/previous.json", "reports/previous.json"):
            path = self.root / "eval" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("FORBIDDEN_ARTIFACT_SENTINEL", encoding="utf-8")
        definition_path = self.root / "eval/benchmarks/cache_v1.json"
        reads = []
        real_open = io.open

        def guarded_open(file, mode="r", *args, **kwargs):
            if "r" in mode or "+" in mode:
                self.assertEqual(Path(file).resolve(), definition_path.resolve())
                reads.append(Path(file).resolve())
            return real_open(file, mode, *args, **kwargs)

        with patch("io.open", side_effect=guarded_open), patch("builtins.open", side_effect=guarded_open), \
             patch("os.scandir", side_effect=AssertionError("Do not scan artifact directories")), \
             patch("os.listdir", side_effect=AssertionError("Do not scan artifact directories")):
            self.generate()
        self.assertEqual(reads, [definition_path.resolve()])
        self.assertNotIn("FORBIDDEN_ARTIFACT_SENTINEL", self.output_path().read_text(encoding="utf-8"))

    def test_api_key_and_environment_are_not_stored(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline_secret_sentinel",
                                     "UNRELATED_SETTING": "offline_env_sentinel"}):
            self.generate()
        serialized = self.output_path().read_text(encoding="utf-8")
        for value in ("OPENAI_API_KEY", "offline_secret_sentinel", "UNRELATED_SETTING", "offline_env_sentinel"):
            self.assertNotIn(value, serialized)

    def test_source_definitions_remain_byte_for_byte_unchanged(self):
        self.write_definition("other_v1")
        paths = list((self.root / "eval/benchmarks").glob("*.json"))
        before = {path: path.read_bytes() for path in paths}
        self.generate(["cache_v1", "other_v1"])
        self.assertEqual({path: path.read_bytes() for path in paths}, before)

    def test_existing_plan_is_protected_before_planner_call(self):
        self.output_path().parent.mkdir(parents=True)
        self.output_path().write_bytes(b"existing frozen plan")
        with self.assertRaisesRegex(FileExistsError, "--overwrite"):
            self.generate()
        self.assertEqual(self.output_path().read_bytes(), b"existing frozen plan")
        self.assertEqual(self.planner.requests, [])
        self.assertIn("cache_v1", self.stderr.getvalue())

    def test_explicit_overwrite_replaces_existing_plan(self):
        self.output_path().parent.mkdir(parents=True)
        self.output_path().write_bytes(b"existing frozen plan")
        self.generate(overwrite=True)
        saved = json.loads(self.output_path().read_text(encoding="utf-8"))
        self.assertEqual(saved["plan"], self.planner.plans[0].model_dump(mode="json"))

    def test_concurrent_output_creation_is_also_protected(self):
        plan = self.planner.plan

        def concurrent_plan(request):
            self.output_path().parent.mkdir(parents=True)
            self.output_path().write_bytes(b"another completed run")
            return plan(request)

        with patch.object(self.planner, "plan", side_effect=concurrent_plan):
            with self.assertRaises(FileExistsError):
                self.generate()
        self.assertEqual(self.output_path().read_bytes(), b"another completed run")

    def test_unknown_benchmark_fails_clearly_without_planning(self):
        with self.assertRaisesRegex(ValueError, "Unknown benchmark id 'unknown_v1'"):
            self.generate(["unknown_v1"])
        self.assertEqual(self.planner.requests, [])
        self.assertFalse(self.output_path("unknown_v1").exists())
        self.assertIn("unknown_v1", self.stderr.getvalue())

    def test_definition_id_mismatch_is_rejected(self):
        path = self.root / "eval/benchmarks/cache_v1.json"
        path.write_text(json.dumps({**self.definition, "benchmark_id": "other_v1"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.generate()
        self.assertEqual(self.planner.requests, [])

    def test_malformed_or_incomplete_definition_is_rejected_before_planning(self):
        path = self.root / "eval/benchmarks/cache_v1.json"
        for data in ([], {key: value for key, value in self.definition.items() if key != "question"},
                     {**self.definition, "question": " "}, {**self.definition, "year_from": True}):
            with self.subTest(data=data):
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.generate()
        self.assertEqual(self.planner.requests, [])
        self.assertFalse(self.output_path().exists())

    def test_unsafe_benchmark_id_is_rejected_without_planning(self):
        with self.assertRaisesRegex(ValueError, "benchmark_id"):
            self.generate(["../outside"])
        self.assertEqual(self.planner.requests, [])

    def test_empty_or_duplicate_benchmark_ids_are_rejected_without_planning(self):
        for ids in ([], ["cache_v1", "cache_v1"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.generate(ids)
        self.assertEqual(self.planner.requests, [])

    def test_planner_error_propagates_without_fallback_file(self):
        self.planner.fail_on = 1
        with self.assertRaises(RuntimeError) as raised:
            self.generate()
        self.assertIs(raised.exception, self.planner.error)
        self.assertFalse(self.output_path().exists())
        self.assertEqual(len(self.planner.requests), 1)
        self.assertIn("cache_v1", self.stderr.getvalue())

    def test_failure_preserves_completed_files_and_stops_later_planning(self):
        self.write_definition("second_v1")
        self.write_definition("third_v1")
        self.planner.fail_on = 2
        with self.assertRaises(RuntimeError) as raised:
            self.generate(["cache_v1", "second_v1", "third_v1"])
        self.assertIs(raised.exception, self.planner.error)
        saved = json.loads(self.output_path().read_text(encoding="utf-8"))
        self.assertEqual(saved["plan"], self.planner.plans[0].model_dump(mode="json"))
        self.assertFalse(self.output_path("second_v1").exists())
        self.assertFalse(self.output_path("third_v1").exists())
        self.assertEqual(len(self.planner.requests), 2)
        self.assertIn("second_v1", self.stderr.getvalue())
        self.assertNotIn("third_v1", self.stdout.getvalue())

    def test_failed_overwrite_preserves_previous_file(self):
        self.output_path().parent.mkdir(parents=True)
        self.output_path().write_bytes(b"previous completed plan")
        self.planner.fail_on = 1
        with self.assertRaises(RuntimeError):
            self.generate(overwrite=True)
        self.assertEqual(self.output_path().read_bytes(), b"previous completed plan")

    def test_default_adapter_receives_configured_model(self):
        with patch("scripts.generate_query_plans.OpenAIQueryPlanner", return_value=self.planner) as factory:
            generate_query_plans(["cache_v1"], root=self.root, model="fixture-model")
        factory.assert_called_once_with(model="fixture-model")
        saved = json.loads(self.output_path().read_text(encoding="utf-8"))
        self.assertEqual(saved["model"], "fixture-model")

    def test_cli_preserves_order_and_defaults(self):
        ids = ["matrix_completion_v1", "rag_hallucination_v1", "cot_reasoning_faithfulness_v1"]
        with patch("scripts.generate_query_plans.generate_query_plans") as generate:
            main(["--benchmarks", *ids])
        generate.assert_called_once_with(ids, model="gpt-5.6-terra", overwrite=False)

    def test_cli_requires_explicit_overwrite_and_passes_model(self):
        with patch("scripts.generate_query_plans.generate_query_plans") as generate:
            main(["--benchmarks", "cache_v1", "--model", "fixture-model", "--overwrite"])
        generate.assert_called_once_with(["cache_v1"], model="fixture-model", overwrite=True)


if __name__ == "__main__":
    unittest.main()
