"""Offline alignment artifact tests, independent of formal benchmark labels."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from scripts.build_union_alignment import build_union_alignment, main


def candidate(rank, group, paper, title):
    return {
        "rrf_rank": rank, "group_id": group, "paper_id": paper, "title": title,
        "year": 2020, "version_count": 1, "abstract": "Unused fixture abstract",
        "rrf_score": 0.01, "hits": [],
    }


class BuildUnionAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / f"alignment-fixture-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self.remove_fixture)
        self.stdout = io.StringIO()
        self.enterContext(redirect_stdout(self.stdout))
        for target in ("socket.create_connection", "socket.socket.connect"):
            self.enterContext(patch(target, side_effect=AssertionError("Alignment tests must remain offline")))
        self.human_path = self.root / "eval/datasets/fixture_v1_candidates.json"
        self.llm_path = self.root / "eval/datasets/fixture_v1_llm_candidates_v1.json"
        self.directory = self.root / "eval/alignment"
        self.decisions_path = self.directory / "fixture_v1_identity_reviews_v1.json"
        self.outputs = [self.directory / f"fixture_v1_{suffix}_v1.json" for suffix in (
            "union", "review_candidates", "alignment_summary",
        )]
        envelope = {"benchmark_id": "fixture_v1", "question": "How do shared caches affect latency?"}
        self.human = {**envelope, "candidates": [
            candidate(1, "h1", "common", "Alpha"), candidate(2, "h2", "human_only", "Ambiguous Work"),
        ]}
        self.llm = {**envelope, "candidates": [
            candidate(1, "l1", "common", "Alpha"), candidate(2, "l2", "llm_only", "ambiguous-work"),
        ]}
        self.write_json(self.human_path, self.human)
        self.write_json(self.llm_path, self.llm)

    def remove_fixture(self):
        target = self.root.resolve()
        if target.parent != Path(__file__).resolve().parent or not target.name.startswith("alignment-fixture-"):
            raise AssertionError(f"Unsafe fixture cleanup: {target}")
        shutil.rmtree(target)

    def write_json(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def write_decisions(self, *, value="different_work", left="h2", right="l2"):
        self.write_json(self.decisions_path, {
            "benchmark_id": "fixture_v1", "version": "v1", "decisions": [{
                "left_source": "human", "left_group_id": left,
                "right_source": "llm", "right_group_id": right,
                "decision": value, "note": "Explicit fixture identity decision.",
            }],
        })

    def build(self):
        return build_union_alignment("fixture_v1", root=self.root)

    def test_writes_manifest_review_candidates_and_provisional_summary_without_decisions(self):
        result = self.build()
        manifest, review, summary = [json.loads(path.read_text(encoding="utf-8")) for path in self.outputs]
        self.assertEqual(manifest, result)
        self.assertEqual(review["review_candidates"], result["review_candidates"])
        self.assertEqual(len(review["review_candidates"]), 1)
        self.assertEqual(result["summary"], {
            "raw_human_groups": 2, "raw_llm_groups": 2,
            "canonical_human_works": 2, "canonical_llm_works": 2,
            "intersection": 1, "human_only": 1, "llm_only": 1, "unresolved_title_pairs": 1,
        })
        self.assertEqual(set(summary), {*result["summary"], "benchmark_id", "version", "identity_review_complete", "counts_status"})
        for data in (manifest, review, summary):
            self.assertFalse(data["identity_review_complete"])
            self.assertEqual(data["counts_status"], "provisional")
        self.assertIn("PROVISIONAL", self.stdout.getvalue())
        self.assertIn("before using them for gold construction", self.stdout.getvalue())
        self.assertFalse(self.decisions_path.exists())

    def test_reviews_are_read_only_and_generated_artifacts_update_after_review(self):
        self.build()
        self.write_decisions()
        protected = [self.human_path, self.llm_path, self.decisions_path]
        before = {path: path.read_bytes() for path in protected}
        result = self.build()
        self.assertTrue(result["identity_review_complete"])
        self.assertEqual(result["counts_status"], "final")
        self.assertEqual(result["review_candidates"], [])
        self.assertEqual(result["summary"]["intersection"], 1)
        self.assertEqual({path: path.read_bytes() for path in protected}, before)
        self.assertEqual(json.loads(self.outputs[1].read_text(encoding="utf-8"))["review_candidates"], [])

    def test_explicit_same_work_decision_is_loaded_without_hardcoded_benchmark_logic(self):
        self.write_decisions(value="same_work")
        result = self.build()
        self.assertTrue(result["identity_review_complete"])
        self.assertEqual(result["summary"]["intersection"], 2)
        self.assertEqual(result["summary"]["human_only"], 0)
        self.assertEqual(result["summary"]["llm_only"], 0)

    def test_repeated_build_is_byte_deterministic(self):
        self.build()
        before = {path: path.read_bytes() for path in self.outputs}
        self.build()
        self.assertEqual({path: path.read_bytes() for path in self.outputs}, before)

    def test_reads_only_snapshots_and_optional_reviews_and_never_modifies_protected_inputs(self):
        forbidden = [self.root / "eval" / relative for relative in (
            "datasets/fixture_v1.json", "runs/fixture_v1_semantic_v1.json", "reports/previous.json",
            "benchmarks/fixture_v1.json", "plans/fixture_v1_llm_plan_v1.json",
        )]
        for path in forbidden:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"FORBIDDEN_ARTIFACT_SENTINEL")
        self.write_decisions()
        allowed = {path.resolve() for path in (self.human_path, self.llm_path, self.decisions_path)}
        protected = [self.human_path, self.llm_path, self.decisions_path, *forbidden]
        before = {path: path.read_bytes() for path in protected}
        reads = []
        real_open = io.open

        def guarded_open(file, mode="r", *args, **kwargs):
            path = Path(file).resolve()
            if "r" in mode or "+" in mode:
                self.assertIn(path, allowed)
                reads.append(path)
            else:
                self.assertIn(path, {output.resolve() for output in self.outputs})
            return real_open(file, mode, *args, **kwargs)

        with patch("io.open", side_effect=guarded_open), patch("builtins.open", side_effect=guarded_open), \
             patch("os.scandir", side_effect=AssertionError("No directory scans")), \
             patch("os.listdir", side_effect=AssertionError("No directory scans")):
            self.build()
        self.assertEqual(reads, [self.human_path.resolve(), self.llm_path.resolve(), self.decisions_path.resolve()])
        self.assertEqual({path: path.read_bytes() for path in protected}, before)
        self.assertNotIn("FORBIDDEN_ARTIFACT_SENTINEL", self.outputs[0].read_text(encoding="utf-8"))

    def test_legacy_human_snapshot_without_id_is_supported(self):
        self.human.pop("benchmark_id")
        self.write_json(self.human_path, self.human)
        self.assertEqual(self.build()["summary"]["raw_human_groups"], 2)

    def test_snapshot_id_and_question_mismatches_rejected_without_outputs(self):
        for field, value in (("benchmark_id", "wrong_v1"), ("question", "Different question")):
            with self.subTest(field=field):
                self.write_json(self.llm_path, {**self.llm, field: value})
                with self.assertRaises(ValueError):
                    self.build()
        self.assertFalse(any(path.exists() for path in self.outputs))

    def test_missing_snapshot_is_not_treated_as_empty_pool(self):
        self.llm_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()
        self.assertFalse(any(path.exists() for path in self.outputs))

    def test_malformed_or_mismatched_reviews_are_rejected_without_outputs(self):
        for data in (None, {}, {"benchmark_id": "fixture_v1", "version": "v2", "decisions": []},
                     {"benchmark_id": "other_v1", "version": "v1", "decisions": []}):
            with self.subTest(data=data):
                self.write_json(self.decisions_path, data)
                with self.assertRaises(ValueError):
                    self.build()
        self.assertFalse(any(path.exists() for path in self.outputs))

    def test_unknown_review_group_rejected_without_outputs(self):
        self.write_decisions(right="missing")
        with self.assertRaisesRegex(ValueError, "unknown candidate group"):
            self.build()
        self.assertFalse(any(path.exists() for path in self.outputs))

    def test_conflicting_reviews_do_not_replace_previous_generated_files(self):
        self.build()
        before = {path: path.read_bytes() for path in self.outputs}
        self.write_decisions(left="h1", right="l1")  # Exact shared paper_id.
        decision_before = self.decisions_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.build()
        self.assertEqual({path: path.read_bytes() for path in self.outputs}, before)
        self.assertEqual(self.decisions_path.read_bytes(), decision_before)

    def test_unsafe_benchmark_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "benchmark_id"):
            build_union_alignment("../outside", root=self.root)

    def test_cli_passes_requested_benchmark(self):
        with patch("scripts.build_union_alignment.build_union_alignment") as build:
            main(["--benchmark", "cot_reasoning_faithfulness_v1"])
        build.assert_called_once_with("cot_reasoning_faithfulness_v1")


if __name__ == "__main__":
    unittest.main()
