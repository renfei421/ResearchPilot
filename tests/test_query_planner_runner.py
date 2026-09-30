"""Real frozen inputs, independent metric calculations, and restricted I/O."""

from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from math import log2
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from eval.benchmark import PROJECT_ROOT
from eval.query_planner_evaluation import BENCHMARKS, METRICS
from eval.query_planner_report import render_markdown
from scripts.run_query_planner_eval import compute_evaluation, input_paths, read_document, run_evaluation


def independent_metrics(ranked, labels):
    """Test-only arithmetic reference; no production evaluation helpers."""
    relevant_total = sum(v >= 1 for v in labels.values())
    direct_total = sum(v == 2 for v in labels.values())
    top = [labels[uid] for uid in ranked[:10]]
    ideal = sorted(labels.values(), reverse=True)[:10]
    dcg = sum((2**value-1)/log2(index+2) for index,value in enumerate(top))
    idcg = sum((2**value-1)/log2(index+2) for index,value in enumerate(ideal))
    return {
        "relevant_recall": sum(labels[uid] >= 1 for uid in ranked)/relevant_total,
        "direct_recall": sum(labels[uid] == 2 for uid in ranked)/direct_total,
        "precision_at_10": sum(v >= 1 for v in top)/10,
        "strict_precision_at_10": sum(v == 2 for v in top)/10,
        "ndcg_at_10": dcg/idcg,
    }


class PlannerDatasetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")), patch("socket.create_connection", side_effect=AssertionError("Network forbidden")):
            cls.report, cls.datasets = compute_evaluation()

    def test_all_actual_counts_label_distributions_and_rank_sizes(self):
        counts = ((40,(19,9,12),22,30,12,10,18), (43,(7,24,12),17,38,12,5,26), (57,(26,23,8),22,42,7,15,35))
        for benchmark, (size, histogram, h, l, shared, h_only, l_only) in zip(BENCHMARKS,counts):
            with self.subTest(benchmark=benchmark):
                data = self.datasets[benchmark]
                self.assertEqual(data["judgment_count"], size)
                self.assertEqual(tuple(data["label_counts"]["final"].values()), histogram)
                result = self.report["benchmarks"][benchmark]
                self.assertEqual(result["structural_counts"], {"human_retrieved":h,"llm_retrieved":l,"intersection":shared,"human_only":h_only,"llm_only":l_only,"union_size":size})
                self.assertEqual(len(result["rankings"]["human"]),h)
                self.assertEqual(len(result["rankings"]["llm"]),l)
                for ranked in result["rankings"].values():
                    self.assertEqual(len(set(ranked)),len(ranked))
        self.assertEqual(self.report["totals"]["judgments"],140)
        self.assertEqual(self.report["totals"]["label_counts"],{"label_0":52,"label_1":56,"label_2":32})

    def test_independent_source_mapping_original_rank_order_metrics_and_unique_composition(self):
        for benchmark in BENCHMARKS:
            with self.subTest(benchmark=benchmark):
                files = {key:json.loads(path.read_bytes()) for key,path in input_paths(benchmark, PROJECT_ROOT).items()}
                adopted = {j["union_id"]:j["label"] for j in files["adopted_labels"]["judgments"]}
                labels, original_ranks = {}, {"human":{},"llm":{}}
                for work in files["manifest"]["works"]:
                    uid = work["union_id"]
                    human = [m for m in work["members"] if m["source"] == "human"]
                    labels[uid] = files["human_gold"]["judgments"][human[0]["group_id"]]["relevance"] if human else adopted[uid]
                    for member in work["members"]:
                        source = member["source"]
                        original_ranks[source][uid] = min(original_ranks[source].get(uid,member["original_rank"]),member["original_rank"])
                result = self.report["benchmarks"][benchmark]
                self.assertEqual(labels,{uid:j["relevance"] for uid,j in self.datasets[benchmark]["judgments"].items()})
                for source in ("human","llm"):
                    ordered = sorted(original_ranks[source],key=lambda uid:(original_ranks[source][uid],uid))
                    self.assertEqual(ordered,result["rankings"][source])
                    for name,value in independent_metrics(ordered,labels).items():
                        self.assertAlmostEqual(result["metrics"][source][name],value,places=14)
                    own = set(ordered) - set(result["rankings"]["llm" if source == "human" else "human"])
                    comp = result["unique_coverage"][f"{source}_only"]
                    self.assertEqual([comp[f"label_{i}"] for i in range(3)], [sum(labels[uid] == i for uid in own) for i in range(3)])
                for row in result["shared_work_rank_diagnostics"]:
                    uid = row["union_id"]
                    h,l = (result["rankings"][s].index(uid)+1 for s in ("human","llm"))
                    self.assertEqual((row["human_rank"],row["llm_rank"],row["rank_delta"]),(h,l,h-l))

    def test_actual_adopted_distribution_and_history_unchanged(self):
        for benchmark, expected in zip(BENCHMARKS,((9,4,5),(6,12,8),(21,12,2))):
            data = self.datasets[benchmark]
            self.assertEqual(tuple(data["label_counts"]["adopted"].values()), expected)
            original = json.loads(input_paths(benchmark,PROJECT_ROOT)["adopted_labels"].read_bytes())
            for entry in original["judgments"]:
                final = data["judgments"][entry["union_id"]]
                self.assertEqual(final["original_provenance"],entry)
                self.assertEqual(final["relevance"],entry["label"])

    def test_macro_and_markdown_report_match_machine_readable_values(self):
        text = render_markdown(self.report)
        for source in ("human","llm"):
            for name in METRICS:
                mean = sum(r["metrics"][source][name] for r in self.report["benchmarks"].values())/3
                self.assertEqual(self.report["macro_metrics"][source][name],mean)
                self.assertIn(str(mean),text)
        for result in self.report["benchmarks"].values():
            for metric in METRICS:
                row = " | ".join(str(result["metrics"][s][metric]) for s in ("human","llm","delta"))
                self.assertIn(row,text)
            for d in result["shared_work_rank_diagnostics"]:
                self.assertIn(d["union_id"],text)


class PlannerRunnerTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / f"planner-eval-fixture-{uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(self.cleanup_root)
        self.allowed = set()
        for benchmark in BENCHMARKS:
            source = input_paths(benchmark,PROJECT_ROOT)
            for role,path in input_paths(benchmark,self.root).items():
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(source[role].read_bytes())
                self.allowed.add(path)
        # Poison files demonstrate the runner never consumes these workflows.
        self.poison = []
        for relative in (
            'eval/annotations/matrix_completion_v1_new_gold_tasks_v1.json',
            'eval/annotations/model_assisted/manual_review_queue_v1.json',
            'eval/runs/fixture_semantic_v1.json', 'eval/reports/semantic_fixture.json',
            'eval/datasets/matrix_completion_v1_candidates.json',
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('DO NOT READ OR CHANGE',encoding='utf-8')
            self.poison.append(path)

    def cleanup_root(self):
        target = self.root.resolve()
        if target.parent != Path(__file__).resolve().parent or not target.name.startswith('planner-eval-fixture-'):
            raise AssertionError('Unsafe cleanup path')
        shutil.rmtree(target)

    def run_offline(self, **kwargs):
        with patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')), \
             patch('socket.create_connection',side_effect=AssertionError('Network forbidden')), \
             redirect_stdout(io.StringIO()):
            return run_evaluation(root=self.root,**kwargs)

    def test_runner_reads_only_nine_inputs_and_preserves_every_input(self):
        before = {p:sha256(p.read_bytes()).hexdigest() for p in self.allowed | set(self.poison)}
        calls = []

        def guarded(path):
            self.assertIn(path,self.allowed)
            calls.append(path)
            return read_document(path)

        with patch('scripts.run_query_planner_eval.read_document',side_effect=guarded):
            report = self.run_offline()
        self.assertEqual(len(calls),9)
        self.assertEqual(set(calls),self.allowed)
        self.assertEqual(before,{p:sha256(p.read_bytes()).hexdigest() for p in before})
        self.assertEqual(json.loads((self.root/'eval/reports/query_planner_v1.json').read_bytes()),report)
        self.assertEqual((self.root/'eval/reports/query_planner_v1.md').read_text(encoding='utf-8'),render_markdown(report))
        for benchmark in BENCHMARKS:
            final = json.loads((self.root/f'eval/datasets/{benchmark}_union_relevance_v1.json').read_bytes())
            self.assertEqual(final['judgment_count'],report['benchmarks'][benchmark]['structural_counts']['union_size'])

    def test_identical_rerun_is_idempotent(self):
        first = self.run_offline()
        path = self.root/'eval/reports/query_planner_v1.json'
        before = (path.read_bytes(),path.stat().st_mtime_ns)
        self.assertEqual(self.run_offline(),first)
        self.assertEqual((path.read_bytes(),path.stat().st_mtime_ns),before)

    def test_bad_source_distribution_fails_before_any_artifact_written(self):
        path = input_paths(BENCHMARKS[0],self.root)['adopted_labels']
        data = json.loads(path.read_bytes())
        data['judgments'][0]['label'] = 2
        path.write_text(json.dumps(data),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'label distribution'):
            self.run_offline()
        self.assertFalse((self.root/'eval/reports/query_planner_v1.json').exists())
        self.assertEqual(list((self.root/'eval/datasets').glob('*union_relevance*')),[])

    def test_manifest_id_must_match_requested_benchmark_path(self):
        path = input_paths(BENCHMARKS[0],self.root)['manifest']
        data = json.loads(path.read_bytes())
        data['benchmark_id'] = 'wrong_benchmark'
        path.write_text(json.dumps(data),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'requested benchmark path'):
            self.run_offline()

    def test_no_api_retrieval_or_semantic_reranker_entrypoint_is_called(self):
        forbidden = AssertionError('Planner evaluation must stay offline and retrieval-only')
        with patch('researchpilot.openalex_client.OpenAlexClient.search_works',side_effect=forbidden), \
             patch('researchpilot.paper_search_service.PaperSearchService.search_candidates',side_effect=forbidden), \
             patch('researchpilot.semantic_reranker.SemanticReranker.rerank',side_effect=forbidden), \
             patch('researchpilot.openai_relevance_client.OpenAIRelevanceClient.assess',side_effect=forbidden):
            self.assertEqual(self.run_offline()['totals']['judgments'],140)

    def test_duplicate_json_judgment_key_is_rejected_not_silently_dropped(self):
        path = input_paths(BENCHMARKS[0],self.root)['human_gold']
        path.write_text('{"judgments":{"union:a":1,"union:a":2}}',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Duplicate JSON key/judgment'):
            read_document(path)

    def test_overwrite_is_explicit_and_only_for_derived_outputs(self):
        path = self.root/'eval/reports/query_planner_v1.json'
        path.write_text('previous derived report',encoding='utf-8')
        with self.assertRaises(FileExistsError):
            self.run_offline()
        self.assertEqual(list((self.root/'eval/datasets').glob('*union_relevance*')),[])
        self.run_offline(overwrite=True)
        self.assertEqual(json.loads(path.read_bytes())['totals']['judgments'],140)
        for poison in self.poison:
            self.assertEqual(poison.read_text(encoding='utf-8'),'DO NOT READ OR CHANGE')


if __name__ == '__main__':
    unittest.main()
