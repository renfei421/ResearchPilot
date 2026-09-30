"""Offline CLI interaction, resumability, and safe file persistence."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from eval.union_gold_io import expansion_paths, read_json, save_json_atomic
from scripts.build_union_gold import build_gold
from scripts.label_union_gold import label_benchmark
from scripts.prepare_union_gold_tasks import prepare_tasks
from test_union_gold import fixture


class UnionGoldCLITests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / f"union-gold-fixture-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self.cleanup_root)
        self.paths = expansion_paths("fixture_v1", root=self.root)
        manifest, gold, snapshots = fixture()
        self.protected = {
            self.paths.manifest: manifest, self.paths.human_gold: gold,
            self.paths.human_candidates: snapshots["human"],
            self.paths.llm_candidates: snapshots["llm"],
        }
        for path, data in self.protected.items():
            save_json_atomic(path, data)
        self.original_bytes = {p: p.read_bytes() for p in self.protected}
        with redirect_stdout(io.StringIO()):
            prepare_tasks("fixture_v1", root=self.root)

    def cleanup_root(self):
        resolved = self.root.resolve()
        if resolved.parent != Path(__file__).resolve().parent or not resolved.name.startswith("union-gold-fixture-"):
            raise AssertionError("Unsafe test cleanup path")
        shutil.rmtree(resolved)

    def interact(self, answers, **kwargs):
        iterator = iter(answers)
        output = []

        def get_input(prompt):
            try:
                answer = next(iterator)
            except StopIteration:
                raise AssertionError(f"Unexpected input prompt: {prompt}") from None
            if isinstance(answer, BaseException):
                raise answer
            return answer

        with patch("socket.socket.connect", side_effect=AssertionError("No network allowed")):
            result = label_benchmark(
                "fixture_v1", root=self.root, input_fn=get_input, output_fn=output.append, **kwargs
            )
        return result, "\n".join(output)

    def test_cli_accepts_0_1_2_and_optional_notes(self):
        result, output = self.interact(["0", "", "1", "Bridge evidence", "2", "Core evidence"])
        self.assertEqual([t["label"] for t in result["tasks"]], [0, 1, 2])
        self.assertEqual(result, read_json(self.paths.annotations))
        self.assertIn("3 / 3 completed", output)
        self.assertIn("1 = Bridge/Supporting", output)
        self.assertNotIn("SECRET_", output)

    def test_rejects_other_labels_without_saving_them(self):
        result, output = self.interact(["", "true", "-1", "3", "1.0", "yes", "2", "", "q"])
        self.assertEqual([t["label"] for t in result["tasks"]], [2, None, None])
        self.assertEqual(output.count("Enter exactly"), 6)

    def test_quit_and_resume_preserve_previous_label_and_note(self):
        first, output = self.interact(["0", "Retain this note", "q"])
        self.assertEqual(first["tasks"][0]["label"], 0)
        self.assertIn("1 / 3 completed", output)
        second, output = self.interact(["2", "New note", "q"])
        self.assertEqual(second["tasks"][0], first["tasks"][0])
        self.assertEqual([t["label"] for t in second["tasks"]], [0, 2, None])
        self.assertNotIn("Union ID: union:a", output)

    def test_skip_leaves_null_and_returns_on_next_session(self):
        first, _ = self.interact(["s", "1", "", "q"])
        self.assertEqual([t["label"] for t in first["tasks"]], [None, 1, None])
        second, output = self.interact(["2", "", "q"])
        self.assertIn("Union ID: union:a", output)
        self.assertEqual([t["label"] for t in second["tasks"]], [2, 1, None])

    def test_quit_immediately_and_list_unlabeled_do_not_write(self):
        before = self.paths.annotations.read_bytes()
        self.interact(["q"])
        self.assertEqual(self.paths.annotations.read_bytes(), before)
        _, output = self.interact([], list_unlabeled=True)
        self.assertEqual(output.count("Research Question:"), 3)
        self.assertNotIn("SECRET_", output)
        self.assertEqual(self.paths.annotations.read_bytes(), before)

    def test_each_judgment_is_saved_before_next_note_or_task(self):
        prompts = []

        def answer(prompt):
            prompts.append(prompt)
            if len(prompts) == 1:
                return "1"
            self.assertEqual(read_json(self.paths.annotations)["tasks"][0]["label"], 1)
            if len(prompts) == 2:
                return "Saved incrementally"
            self.assertEqual(read_json(self.paths.annotations)["tasks"][0]["note"], "Saved incrementally")
            return "q"

        label_benchmark("fixture_v1", root=self.root, input_fn=answer, output_fn=lambda _: None)
        self.assertEqual(len(prompts), 3)

    def test_eof_and_interrupt_preserve_label_even_during_optional_note(self):
        for exception in (EOFError(), KeyboardInterrupt()):
            with self.subTest(exception=type(exception).__name__):
                result, output = self.interact(["1", exception])
                self.assertEqual(result, read_json(self.paths.annotations))
                self.assertIn("Previously saved judgments are preserved", output)
        self.assertEqual([t["label"] for t in result["tasks"]], [1, 1, None])

    def test_generation_refuses_overwriting_annotation_progress(self):
        self.interact(["2", "Keep", "q"])
        before = self.paths.annotations.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_tasks("fixture_v1", root=self.root)
        self.assertEqual(self.paths.annotations.read_bytes(), before)

    def test_atomic_replace_failure_keeps_prior_file_and_cleans_temp(self):
        before = self.paths.annotations.read_bytes()
        changed = read_json(self.paths.annotations)
        changed["tasks"][0]["label"] = 2
        with patch("eval.union_gold_io.os.replace", side_effect=OSError("simulated failure")):
            with self.assertRaisesRegex(OSError, "simulated failure"):
                save_json_atomic(self.paths.annotations, changed, expected_bytes=before)
        self.assertEqual(self.paths.annotations.read_bytes(), before)
        self.assertEqual(list(self.paths.annotations.parent.glob("*.tmp")), [])

    def test_stale_session_cannot_overwrite_newer_progress(self):
        before = self.paths.annotations.read_bytes()
        self.interact(["1", "Another session", "q"])
        newer = self.paths.annotations.read_bytes()
        with self.assertRaisesRegex(ValueError, "changed since loading"):
            save_json_atomic(self.paths.annotations, json.loads(before), expected_bytes=before)
        self.assertEqual(self.paths.annotations.read_bytes(), newer)

    def test_final_builder_refuses_incomplete_without_output(self):
        with self.assertRaisesRegex(ValueError, "Incomplete union gold"):
            build_gold("fixture_v1", root=self.root)
        self.assertFalse(self.paths.union_gold.exists())

    def test_final_builder_saves_complete_fixture_once_and_preserves_inputs(self):
        self.interact(["0", "", "1", "", "2", ""])
        with redirect_stdout(io.StringIO()):
            result = build_gold("fixture_v1", root=self.root)
        self.assertEqual(result, read_json(self.paths.union_gold))
        self.assertEqual(len(result["judgments"]), 5)
        self.assertEqual(self.original_bytes, {p: p.read_bytes() for p in self.protected})
        with self.assertRaises(FileExistsError):
            build_gold("fixture_v1", root=self.root)

    def test_invalid_saved_labels_and_metadata_rejected_before_prompt(self):
        for change in (
            lambda d: d["tasks"][0].update(label=True),
            lambda d: d["tasks"][0].update(label=3),
            lambda d: d["tasks"][0].update(query_text="SECRET"),
            lambda d: d["tasks"][0]["aliases"][0].update(semantic_score=0.9),
        ):
            original = self.paths.annotations.read_bytes()
            document = json.loads(original)
            change(document)
            save_json_atomic(self.paths.annotations, document, expected_bytes=original)
            with self.assertRaises(ValueError):
                self.interact([])
            save_json_atomic(self.paths.annotations, json.loads(original), expected_bytes=self.paths.annotations.read_bytes())


if __name__ == "__main__":
    unittest.main()
