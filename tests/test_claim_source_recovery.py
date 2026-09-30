"""Offline supplementary-recall tests with the real evidence and relation pipeline."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, MagicMock, patch

import httpx
from pydantic import ValidationError

from claim_check_helpers import fixture, assertion, relation, CLAIM
from phase5_helpers import OfflineTest, paper
from researchpilot.claim_check import ClaimCheckRequest, ClaimCheckResult, ClaimDecomposition, RelationBatch, CAVEAT
from researchpilot.claim_check_views import claim_markdown
from researchpilot.claim_recovery import RecoveryPlan, RecoveryQuery, recovery_query_key
from researchpilot.evidence import AcquiredDocument, DocumentPage, EvidencePassage
from researchpilot.openai_claim_recovery import OpenAIClaimRecoveryPlanner, _RECOVERY
from researchpilot.project_context import ProjectContext, QueryHint, project_usage
from researchpilot.project_models import ProjectPaper, ProjectEvidence, EvidenceAttestation, ProjectCreate
from researchpilot.project_reuse import ProjectSearchSession
from researchpilot.run_store import RunStore


def source(key):
    return paper(key, title="Fixed graph weights study " + key,
        abstract="Fixed graph weights control a spectral ratio and a curvature bound under scoped assumptions.")


def recovery_fixture(*, initial_label="ADJACENT", initial_count=1, new_label="SUPPORTING", confidence="high"):
    f = fixture()
    initial = [source("initial"+str(i)) for i in range(initial_count)]
    f.decomposer.decompose = lambda req: ClaimDecomposition(main_claim=req.claim,
        assertions=[assertion(text="Fixed graph weights reduce a spectral ratio under scoped assumptions.")])
    f.recovery_calls, f.searches = [], []
    def plan(req, target, dependencies, previous, gaps, context):
        f.recovery_calls.append((target, dependencies, list(previous), gaps, context))
        n = len(f.recovery_calls)
        return RecoveryPlan(claim_id=target.claim_id, queries=[RecoveryQuery(
            text=f"fixed topology weights spectral deviation {n} {i}", role=role)
            for i, role in enumerate(["direct", "terminology", "bridge"])])
    f.agent.recovery_planner = SimpleNamespace(plan=plan)
    def search(**kw):
        f.searches.append(kw)
        if kw["query"].startswith("fixed topology"):
            return [source("new" + kw["query"].split()[-2] + kw["query"].split()[-1])]
        return initial
    f.client.search_works = search
    def acquire(p):
        f.fetcher.calls.append(p.paper_id)
        return AcquiredDocument(pages=[DocumentPage(paper_id=p.paper_id, title=p.title,
            page_number=1, source_type="pdf", text=p.abstract)])
    f.fetcher.acquire = acquire
    def analyze(assertions, pid, evidence, bundles):
        f.analyzer.calls.append((assertions, pid, evidence, bundles))
        label = initial_label if "initial" in pid else new_label
        return RelationBatch(relations=[relation(a.claim_id, pid, label,
            [evidence[0].passage_id], confidence=confidence) for a in assertions])
    f.analyzer.analyze = analyze
    f.request = ClaimCheckRequest(claim=CLAIM, max_search_rounds=1)
    return f


class SourceRecoveryTests(OfflineTest):
    def test_weak_core_triggers_before_novelty_boundary(self):
        f = recovery_fixture()
        result = f.agent.run(f.request)
        self.assertEqual(len(f.recovery_calls), 1)
        d = result.source_recovery[0]
        self.assertEqual((d.coverage_before, d.coverage_after), ("weak", "moderate"))
        self.assertTrue(any("new" in p for item in result.novelty_boundary.established_components for p in item.paper_ids))

    def test_moderate_core_unchanged(self):
        f = recovery_fixture(initial_label="SUPPORTING", initial_count=2)
        result = f.agent.run(f.request)
        self.assertEqual(result.claim_maps[0].coverage_status, "moderate")
        self.assertEqual(result.source_recovery, [])
        self.assertEqual(f.recovery_calls, [])
        self.assertEqual(len(f.searches), 3)

    def test_strong_core_unchanged(self):
        f = recovery_fixture(initial_label="DIRECT_OVERLAP")
        result = f.agent.run(f.request)
        self.assertEqual(result.claim_maps[0].coverage_status, "strong")
        self.assertEqual(result.source_recovery, [])

    def test_weak_supporting_assertion_never_triggers(self):
        f = recovery_fixture(initial_label="DIRECT_OVERLAP")
        f.decomposer.decompose = lambda req: ClaimDecomposition(main_claim=req.claim,
            assertions=[assertion(), assertion("C2").model_copy(update={"importance":"supporting"})])
        analyze = f.analyzer.analyze
        f.analyzer.analyze = lambda assertions, pid, evidence, bundles: RelationBatch(relations=[
            r if r.claim_id == "C1" else relation("C2", pid, "ADJACENT")
            for r in analyze(assertions,pid,evidence,bundles).relations])
        result = f.agent.run(f.request)
        self.assertEqual(result.claim_maps[1].coverage_status, "weak")
        self.assertFalse(f.recovery_calls)

    def test_two_round_six_query_budget_and_weak_remains_weak(self):
        f = recovery_fixture(confidence="low")
        result = f.agent.run(f.request)
        d = result.source_recovery[0]
        self.assertEqual(d.rounds_used, 2)
        self.assertEqual(len(d.recovery_queries), 6)
        self.assertEqual((d.coverage_after, d.stop_reason), ("weak", "budget_exhausted"))
        self.assertIn("Coverage remains weak", claim_markdown(result))

    def test_query_plan_exceeding_three_rejected_without_search(self):
        f = recovery_fixture()
        good = f.agent.recovery_planner.plan
        def bad(*args):
            p = good(*args);p.queries.append(RecoveryQuery(text="extra technical relationship",role="bridge"));return p
        f.agent.recovery_planner.plan = bad
        r = f.agent.run(f.request)
        self.assertEqual(len(f.searches), 3)
        self.assertTrue(r.source_recovery[0].incomplete)

    def test_trivial_initial_duplicates_are_skipped(self):
        f = recovery_fixture()
        f.agent.recovery_planner.plan = lambda *args: RecoveryPlan(claim_id="C1", queries=[
            RecoveryQuery(text='  “ＧＲＡＰＨ—weights” ',role="direct"),
            RecoveryQuery(text="spectral   GEOMETRY",role="terminology")])
        r = f.agent.run(f.request)
        self.assertEqual(len(f.searches),3)
        self.assertEqual(r.source_recovery[0].stop_reason,"duplicate_queries")

    def test_recovery_duplicates_not_resubmitted(self):
        f = recovery_fixture(confidence="low")
        first = f.agent.recovery_planner.plan(f.request, assertion(), [], [], [], None)
        f.agent.recovery_planner.plan = lambda *args: first
        r = f.agent.run(f.request)
        self.assertEqual(len(r.source_recovery[0].recovery_queries),3)
        self.assertEqual(r.source_recovery[0].stop_reason,"duplicate_queries")

    def test_dedup_does_not_remove_distinct_technical_queries(self):
        self.assertNotEqual(recovery_query_key("fixed weights spectral deviation"),
                            recovery_query_key("fixed weights spectral curvature"))
        self.assertNotEqual(recovery_query_key("a AND b"),recovery_query_key("a OR b"))

    def test_duplicate_recovery_papers_added_once_and_old_preserved(self):
        f = recovery_fixture()
        original = f.client.search_works
        f.client.search_works = lambda **kw: [source("new")] * 2 if kw["query"].startswith("fixed topology") else original(**kw)
        r = f.agent.run(f.request)
        self.assertEqual([s.paper.paper_id for s in r.sources],[source("initial0").paper_id,source("new").paper_id])
        self.assertEqual(f.fetcher.calls.count(source("new").paper_id),1)

    def test_new_version_of_existing_source_not_acquired_again(self):
        f = recovery_fixture()
        original = f.client.search_works
        version = source("initial0").model_copy(update={"paper_id":"other:version", "publication_year":2025})
        f.client.search_works = lambda **kw: [version] if kw["query"].startswith("fixed topology") else original(**kw)
        r = f.agent.run(f.request)
        self.assertEqual(len(r.sources),1)
        self.assertEqual(r.source_recovery[0].stop_reason,"no_new_papers")

    def test_normal_fulltext_pipeline_used_and_initial_relation_preserved(self):
        f = recovery_fixture()
        with patch("researchpilot.claim_novelty_agent.assemble_selected", wraps=__import__(
                "researchpilot.bundle_runtime",fromlist=["assemble_selected"]).assemble_selected) as assemble:
            r = f.agent.run(f.request)
        self.assertGreaterEqual(assemble.call_count,2)
        self.assertTrue(all(s.source_type=="pdf" for s in r.sources))
        self.assertEqual(sum(pid==source("initial0").paper_id for _,pid,_,_ in f.analyzer.calls),1)
        self.assertEqual(len(r.source_recovery[0].new_selected_papers),3)
        self.assertEqual(r.source_recovery[0].new_evidence_count,3)

    def test_strong_coverage_stops_first_round(self):
        f = recovery_fixture(new_label="DIRECT_OVERLAP")
        r = f.agent.run(f.request)
        self.assertEqual((r.source_recovery[0].coverage_after,r.source_recovery[0].rounds_used),("strong",1))
        self.assertEqual(r.source_recovery[0].stop_reason,"coverage_sufficient")

    def test_no_new_paper_stops(self):
        f = recovery_fixture()
        f.client.search_works = lambda **kw: [source("initial0")]
        r = f.agent.run(f.request)
        self.assertEqual(r.source_recovery[0].stop_reason,"no_new_papers")
        self.assertEqual(r.source_recovery[0].rounds_used,1)

    def test_no_new_substantive_evidence_stops(self):
        f = recovery_fixture(new_label="ADJACENT")
        r = f.agent.run(f.request)
        self.assertEqual(r.source_recovery[0].stop_reason,"no_new_substantive_evidence")
        self.assertEqual(r.source_recovery[0].new_evidence_count,0)

    def test_partial_external_query_failure_preserves_initial_results(self):
        f = recovery_fixture(initial_label="SUPPORTING")
        original = f.client.search_works
        def search(**kw):
            if kw['query'].startswith('fixed topology') and kw['query'].endswith(' 0'):
                raise httpx.ReadTimeout("unavailable")
            return original(**kw)
        f.client.search_works = search
        r=f.agent.run(f.request)
        self.assertTrue(r.source_recovery[0].incomplete)
        self.assertTrue(any(t.status=="failed" for t in r.search_trace))
        self.assertTrue(any(x.paper_id==source("initial0").paper_id and x.relation=="SUPPORTING" for x in r.relations))
        self.assertTrue(r.source_recovery[0].successful_queries)

    def test_all_recovery_queries_fail_keeps_original_weak_result(self):
        f=recovery_fixture(initial_label="SUPPORTING")
        original=f.client.search_works
        def search(**kw):
            if kw['query'].startswith('fixed topology'):raise httpx.ConnectError("unavailable")
            return original(**kw)
        f.client.search_works=search
        r=f.agent.run(f.request)
        self.assertEqual(r.source_recovery[0].stop_reason,"recovery_unavailable")
        self.assertEqual(len(r.sources),1)
        self.assertEqual(r.claim_maps[0].coverage_status,"weak")
        self.assertIn("recovery was incomplete",claim_markdown(r))

    def test_paper_budget_stops_at_eight(self):
        f=recovery_fixture(confidence="low")
        original=f.client.search_works
        def search(**kw):
            if kw['query'].startswith('fixed topology'):
                return [source(kw['query'].split()[-1]+str(i)) for i in range(10)]
            return original(**kw)
        f.client.search_works=search
        r=f.agent.run(f.request)
        self.assertEqual(len(r.source_recovery[0].new_selected_papers),8)
        self.assertEqual(r.source_recovery[0].stop_reason,"budget_exhausted")

    def test_unrelated_core_not_reassessed_by_recovery(self):
        f=recovery_fixture()
        f.decomposer.decompose=lambda req:ClaimDecomposition(main_claim=req.claim,assertions=[assertion(),assertion("C2")])
        analyze=f.analyzer.analyze
        def check(assertions,pid,evidence,bundles):
            batch=analyze(assertions,pid,evidence,bundles)
            return RelationBatch(relations=[r if r.claim_id=="C1" else relation("C2",pid,"DIRECT_OVERLAP",[evidence[0].passage_id]) for r in batch.relations])
        f.analyzer.analyze=check
        r=f.agent.run(f.request)
        self.assertEqual([d.claim_id for d in r.source_recovery],["C1"])
        self.assertTrue(all([a.claim_id for a in aa]==["C1"] for aa,pid,_,_ in f.analyzer.calls if "new" in pid))

    def test_caveat_and_diagnostics_roundtrip_without_prompts(self):
        r=recovery_fixture().agent.run(ClaimCheckRequest(claim=CLAIM,max_search_rounds=1))
        saved=ClaimCheckResult.model_validate_json(r.model_dump_json())
        self.assertEqual(saved.source_recovery,r.source_recovery)
        self.assertEqual(saved.novelty_boundary.caveat,CAVEAT)
        for key in ('prompt','chain_of_thought','instructions','api_key'):
            self.assertNotIn(key,json.dumps(saved.source_recovery[0].model_dump()))

    def test_recovery_coverage_cannot_be_fabricated(self):
        r=recovery_fixture().agent.run(ClaimCheckRequest(claim=CLAIM,max_search_rounds=1))
        data=r.model_dump();data['source_recovery'][0]['coverage_after']='strong'
        with self.assertRaisesRegex(ValueError,'Recovery coverage'):
            ClaimCheckResult.model_validate(data)

    def test_project_sources_reused_and_evidence_rechecked(self):
        f=recovery_fixture()
        p=source('project');e=EvidencePassage(passage_id='old-evidence',paper_id=p.paper_id,title=p.title,
            text=p.abstract,source_type='pdf',page_number=1)
        f.agent.project_context=ProjectContext(project_id='project',papers=[ProjectPaper(project_id='project',
            paper_id=p.paper_id,paper=p,version_group_id='v1',first_seen_run_id='old')], evidence=[ProjectEvidence(
            project_id='project',evidence_id='e1',passage=e,verification_status='supported',originating_run_id='old',
            attestations=[EvidenceAttestation(run_id='old',claim_id='old-claim',kind='atomic_verification',status='supported',reason='Previous claim only.')])])
        # Historical support is not proof of this new claim.
        analyze=f.analyzer.analyze
        def current(assertions,pid,evidence,bundles):
            batch=analyze(assertions,pid,evidence,bundles)
            return RelationBatch(relations=[relation(a.claim_id,pid,'ADJACENT') for a in assertions]) if pid==p.paper_id else batch
        f.analyzer.analyze=current
        r=f.agent.run(f.request.model_copy(update={'project_id':'project'}))
        self.assertIn(p.paper_id,[s.paper.paper_id for s in r.sources])
        self.assertTrue(any(pid==p.paper_id for _,pid,_,_ in f.analyzer.calls))
        self.assertTrue(all(x.relation=='ADJACENT' for x in r.relations if x.paper_id==p.paper_id))
        self.assertTrue(f.recovery_calls[0][4]['evidence'])

    def test_relevant_project_query_not_submitted(self):
        f=recovery_fixture()
        f.agent.project_context=ProjectContext(project_id='project',previous_queries=[QueryHint(
            query_id='old',query='fixed topology weights spectral deviation 1 0',originating_run_id='old')])
        r=f.agent.run(f.request.model_copy(update={'project_id':'project'}))
        self.assertEqual(len(r.source_recovery[0].recovery_queries),2)
        self.assertIn('fixed topology weights spectral deviation 1 0',r.source_recovery[0].skipped_duplicate_queries)

    def test_persisted_diagnostics_queries_and_safe_project_ingestion(self):
        f=recovery_fixture();original=f.analyzer.analyze
        def forged(assertions,pid,evidence,bundles):
            batch=original(assertions,pid,evidence,bundles)
            if 'new' in pid:
                batch.relations=[relation(a.claim_id,pid,'SUPPORTING',['invented']) for a in assertions]
            return batch
        f.analyzer.analyze=forged
        root=self.temporary_directory()
        store=RunStore(root/'runs.sqlite3')
        p=store.projects.create(ProjectCreate(title='Fixed graph weights'))
        req=f.request.model_copy(update={'project_id':p.project_id})
        run=store.create(req);store.start(run.run_id)
        req=store.get(run.run_id).request
        session=ProjectSearchSession(f.services.paper_search,store.projects,req,run.run_id)
        f.services.paper_search=session
        result=f.agent.run(req)
        store.complete(run.run_id,result,queries=list(session.records.values()))
        stored=RunStore(root/'runs.sqlite3').get(run.run_id).result
        self.assertEqual(stored.source_recovery,result.source_recovery)
        self.assertTrue(stored.source_recovery[0].incomplete)
        self.assertEqual(store.projects.list_records(p.project_id,'findings'),[])
        self.assertEqual(store.projects.list_records(p.project_id,'evidence'),[])
        self.assertEqual(len(store.projects.list_records(p.project_id,'queries')),6)
        self.assertTrue(any('fixed topology' in q for q in session.previous_query_texts()))

    def test_global_document_budget_shared_by_weak_claims(self):
        f=recovery_fixture(confidence='low')
        f.decomposer.decompose=lambda req:ClaimDecomposition(main_claim=req.claim,
            assertions=[assertion('C1'),assertion('C2')])
        r=f.agent.run(f.request)
        self.assertLessEqual(sum(len(d.new_selected_papers) for d in r.source_recovery),8)
        self.assertLessEqual(len(r.sources),9)
        self.assertEqual(len(r.source_recovery),2)
        self.assertTrue(all(d.recovery_queries and d.new_selected_papers for d in r.source_recovery))

    def test_lower_requested_paper_budget_is_respected(self):
        f=recovery_fixture(confidence='low')
        r=f.agent.run(f.request.model_copy(update={'max_papers_per_claim':1}))
        self.assertEqual(len(r.source_recovery[0].new_selected_papers),1)

    def test_recovery_plan_failure_preserves_successful_initial_relations(self):
        f=recovery_fixture(initial_label='SUPPORTING')
        f.agent.recovery_planner.plan=Mock(side_effect=RuntimeError('unavailable'))
        r=f.agent.run(f.request)
        self.assertEqual(r.relations[0].relation,'SUPPORTING')
        self.assertEqual(r.source_recovery[0].stop_reason,'recovery_unavailable')
        self.assertTrue(r.source_recovery[0].incomplete)

    def test_existing_paper_reassessed_when_new_passage_is_available(self):
        f=recovery_fixture(initial_label='SUPPORTING')
        original=f.services.passage_retriever.search_views
        def retrieve(*args,**kwargs):
            batch=original(*args,**kwargs)
            if f.recovery_calls and batch.passages:
                p=batch.passages[0].passage
                batch.passages[0]=batch.passages[0].model_copy(update={'passage':p.model_copy(update={
                    'passage_id':p.passage_id+'-additional','text':p.text+' A further scoped result.'})})
            return batch
        f.services.passage_retriever.search_views=retrieve
        r=f.agent.run(f.request)
        old=[call for call in f.analyzer.calls if call[1]==source('initial0').paper_id]
        self.assertEqual(len(old),2)
        self.assertGreater(len(old[1][2]),len(old[0][2]))
        self.assertTrue(any(x.paper_id==source('initial0').paper_id for x in r.relations))

    def test_project_query_guard_can_check_rows_outside_prompt_budget(self):
        f=recovery_fixture()
        f.agent.project_context=ProjectContext(project_id='project')
        f.services.paper_search.previous_query_texts=lambda:['fixed topology weights spectral deviation 1 0']
        r=f.agent.run(f.request.model_copy(update={'project_id':'project'}))
        self.assertEqual(len(r.source_recovery[0].recovery_queries),2)
        self.assertEqual(f.recovery_calls[0][4]['previous_queries'],[])

    def test_research_mode_has_no_recovery_dependency(self):
        from phase12_helpers import research_fixture, QUESTION
        from researchpilot.research_models import ResearchRequest
        f=research_fixture()
        with patch('researchpilot.openai_claim_recovery.OpenAIClaimRecoveryPlanner',side_effect=AssertionError('Idea only')):
            result=f.agent.run(ResearchRequest(question=QUESTION,max_search_rounds=1))
        self.assertNotIn('source_recovery',result.model_dump())


class RecoveryAdapterTests(OfflineTest):
    def test_structured_bounded_planner_payload(self):
        sdk=MagicMock();sdk.with_options.return_value=sdk
        sdk.responses.parse.return_value=SimpleNamespace(status='completed',output=[],
            output_parsed=RecoveryPlan(claim_id='C1',queries=[RecoveryQuery(text='fixed weights spectral deviation',role='direct')]))
        adapter=OpenAIClaimRecoveryPlanner(client=sdk)
        adapter.plan(ClaimCheckRequest(claim=CLAIM),assertion(),[],['old query'],['missing edge'],{'open_gaps':['gap']})
        kw=sdk.responses.parse.call_args.kwargs
        self.assertIs(kw['text_format'],RecoveryPlan)
        self.assertFalse(kw['store'])
        self.assertEqual(kw['model'],'gpt-5.6-terra')
        payload=json.loads(kw['input'])
        self.assertEqual(payload['previous_queries'],['old query'])
        self.assertEqual(payload['project_context'],{'open_gaps':['gap']})
        self.assertNotIn('reason',RecoveryQuery.model_fields)

    def test_recovery_prompt_limits_topic_drift_and_novelty(self):
        for text in ['ONE supplied weak','at most three','method_neighbor','not recovery queries',
                     'fixed support','normalized spectral','Never predict results','chain-of-thought']:
            self.assertIn(text,_RECOVERY)

    def test_wrong_claim_plan_rejected(self):
        adapter=OpenAIClaimRecoveryPlanner()
        with patch.object(adapter,'_parse',return_value=RecoveryPlan(claim_id='wrong',queries=[])):
            with self.assertRaisesRegex(ValueError,'target claim'):
                adapter.plan(ClaimCheckRequest(claim=CLAIM),assertion(),[],[],[],None)
