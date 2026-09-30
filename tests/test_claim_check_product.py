"""Offline structured-adapter, SQLite migration, API/history and export checks."""

import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from claim_check_helpers import *
from phase5_helpers import OfflineTest
from test_product import fake_factory, wait_for
from researchpilot.api import create_app
from researchpilot.config import Settings
from researchpilot.openai_claim_check import OpenAIClaimDecomposer, OpenAIClaimSearchPlanner, OpenAIClaimRelationAnalyzer
from researchpilot.openai_claim_check import _RelationJudgments
from researchpilot.run_store import RunStore


class ClaimAdapterTests(OfflineTest):
    def sdk(self, parsed):
        sdk=MagicMock()
        sdk.with_options.return_value=sdk
        if isinstance(parsed, RelationBatch):
            parsed = _RelationJudgments.model_validate({"relations": [
                r.model_dump(exclude={"paper_id", "evidence_ids"}) for r in parsed.relations]})
        sdk.responses.parse.return_value=SimpleNamespace(status="completed",output=[],output_parsed=parsed)
        return sdk

    def test_decomposer_uses_structured_output_without_storage(self):
        sdk=self.sdk(decomposition())
        result=OpenAIClaimDecomposer(client=sdk).decompose(ClaimCheckRequest(claim=CLAIM))
        kw=sdk.responses.parse.call_args.kwargs
        self.assertIs(kw["text_format"],ClaimDecomposition)
        self.assertFalse(kw["store"])
        self.assertEqual(kw["model"],"gpt-5.6-terra")
        self.assertEqual(result.main_claim,CLAIM)
        self.assertEqual(set(json.loads(kw["input"])), {"claim","field","context"})

    def test_decomposer_rejects_changed_original(self):
        with self.assertRaises(ValueError):
            OpenAIClaimDecomposer(client=self.sdk(decomposition("changed"))).decompose(ClaimCheckRequest(claim=CLAIM))

    def test_planner_receives_memory_and_missing_relations(self):
        request=ClaimCheckRequest(claim=CLAIM)
        plan=fixture().planner.plan(request,decomposition(),["C3"],[],[])
        sdk=self.sdk(plan)
        OpenAIClaimSearchPlanner(client=sdk).plan(request,decomposition(),["C3"],["old query"],["missing connection"])
        payload=json.loads(sdk.responses.parse.call_args.kwargs["input"])
        self.assertEqual(payload["previous_queries"],["old query"])
        self.assertEqual(payload["missing_relations"],["missing connection"])
        self.assertEqual(payload["target_ids"],["C3"])

    def test_relation_adapter_only_supplied_assertions_and_evidence(self):
        result=fixture().agent.run(ClaimCheckRequest(claim=CLAIM))
        evidence=[p for p in result.evidence if p.paper_id=="A"]
        rows=[relation(a.claim_id,"P1","SUPPORTING",["E1"]) for a in result.decomposition.assertions]
        sdk=self.sdk(RelationBatch(relations=rows))
        decoded=OpenAIClaimRelationAnalyzer(client=sdk).analyze(result.decomposition.assertions,"A",evidence,[])
        payload=json.loads(sdk.responses.parse.call_args.kwargs["input"])
        self.assertEqual(set(payload), {"assertions","paper_id","passages","bundles"})
        self.assertNotIn("rrf_score",str(payload))
        self.assertNotIn("citation_count",str(payload))
        self.assertNotIn("gold",str(payload))
        self.assertEqual(payload["paper_id"],"P1")
        self.assertEqual(payload["passages"][0]["passage_id"],"E1")
        self.assertEqual(decoded.relations[0].evidence_ids,[evidence[0].passage_id])
        self.assertEqual(decoded.relations[0].paper_id,"A")

    def test_relation_adapter_rejects_unknown_evidence_ids(self):
        sdk=self.sdk(RelationBatch(relations=[relation(pid="P1",ids=["E99"])]))
        with self.assertRaises(ValueError):
            OpenAIClaimRelationAnalyzer(client=sdk).analyze([assertion()],"A",[],[])

    def test_short_handles_roundtrip_long_original_hashes(self):
        from researchpilot.evidence import EvidencePassage
        evidence=[EvidencePassage(passage_id="passage:0123456789abcdef01234567",paper_id="original-paper",
            title="Generic source",text="A substantive finding under stated conditions.",source_type="pdf",page_number=7)]
        sdk=self.sdk(RelationBatch(relations=[relation(pid="P1",ids=["E1"])]))
        result=OpenAIClaimRelationAnalyzer(client=sdk).analyze([assertion()],"original-paper",evidence,[])
        self.assertEqual(result.relations[0].evidence_ids,[evidence[0].passage_id])
        self.assertTrue(all(d.evidence_ids==[evidence[0].passage_id] for d in result.relations[0].comparisons))
        payload=json.loads(sdk.responses.parse.call_args.kwargs["input"])
        self.assertNotIn(evidence[0].passage_id,str(payload))
        self.assertEqual(payload["passages"][0]["text"],evidence[0].text)
        self.assertEqual(payload["passages"][0]["page_number"],7)

    def test_malformed_handle_is_never_fuzzily_corrected(self):
        from researchpilot.evidence import EvidencePassage
        p=EvidencePassage(passage_id="passage:0123456789abcdef01234567",paper_id="A",title="A",
            text="Substantive original passage.",source_type="abstract",page_number=None)
        for handle in ["E01","E2","passage:0123456789abcdef0123456"]:
            sdk=self.sdk(RelationBatch(relations=[relation(pid="P1",ids=[handle])]))
            with self.subTest(handle=handle), self.assertRaisesRegex(ValueError,"Unknown evidence handle"):
                OpenAIClaimRelationAnalyzer(client=sdk).analyze([assertion()],"A",[p],[])

    def test_model_cannot_supply_or_change_known_paper_identity(self):
        row=relation(pid="P2",label="ADJACENT").model_dump(exclude={"evidence_ids"})
        with self.assertRaisesRegex(ValueError,"paper_id"):
            _RelationJudgments.model_validate({"relations":[row]})

    def test_bundle_references_use_same_short_handle_namespace(self):
        from researchpilot.evidence import EvidencePassage
        from researchpilot.evidence_assembly import EvidenceAssembler, finish_bundle
        from researchpilot.evidence_bundle import AssemblyLimits
        p=EvidencePassage(passage_id="passage:originalhash",paper_id="A",title="Generic theory",
            text="Proposition 1. Under Assumption 2, graph curvature is bounded. Assumption 2. The graph is regular.",
            source_type="pdf",page_number=1)
        bundle=finish_bundle(EvidenceAssembler().assemble("graph curvature guarantee",p,[p],AssemblyLimits()))
        sdk=self.sdk(RelationBatch(relations=[relation(pid="P1",ids=["E1"])]))
        original=bundle.model_dump()
        OpenAIClaimRelationAnalyzer(client=sdk).analyze([assertion()],"A",[p],[bundle])
        payload=json.loads(sdk.responses.parse.call_args.kwargs["input"])
        self.assertEqual(payload["bundles"][0]["anchor_evidence_id"],"E1")
        self.assertEqual(payload["bundles"][0]["members"][0]["passage_id"],"E1")
        self.assertNotIn(p.passage_id,str(payload))
        self.assertEqual(bundle.model_dump(),original)

    def test_relation_adapter_rejects_omitted_claim(self):
        sdk=self.sdk(RelationBatch(relations=[]))
        with self.assertRaises(ValueError):
            OpenAIClaimRelationAnalyzer(client=sdk).analyze([assertion()],"A",[],[])

    def test_model_failure_propagates_without_retry(self):
        sdk=self.sdk(None)
        sdk.responses.parse.side_effect=RuntimeError("unavailable")
        with self.assertRaises(RuntimeError):
            OpenAIClaimDecomposer(client=sdk).decompose(ClaimCheckRequest(claim=CLAIM))
        self.assertEqual(sdk.responses.parse.call_count,1)
        self.assertEqual(sdk.with_options.call_args.kwargs["max_retries"],0)

    def test_validation_diagnostics_omit_submitted_values(self):
        from pydantic import ValidationError
        from researchpilot.run_manager import log_event
        from researchpilot.progress import ResearchProgress
        try:
            ClaimCheckRequest(claim="sensitive unpublished claim",max_search_rounds=99)
        except ValidationError as error:
            with patch("researchpilot.run_manager.logger.log") as logger:
                log_event("test",ResearchProgress(),error)
            body=logger.call_args.args[1]
        self.assertNotIn("sensitive unpublished claim",body)
        self.assertIn("max_search_rounds",body)
        self.assertIn("validation_errors",body)


def claim_factory(callback):
    f=fixture()
    f.agent.on_progress=callback
    return f.agent


class ClaimProductTests(OfflineTest):
    def setUp(self):
        for name in ("httpx.HTTPTransport.handle_request","httpx.AsyncHTTPTransport.handle_async_request","socket.create_connection"):
            guard=patch(name,side_effect=AssertionError("Offline product test"))
            guard.start(); self.addCleanup(guard.stop)
        self.root=self.temporary_directory()
        self.settings=Settings(cache_dir=self.root/"cache",run_db=self.root/"runs.sqlite3")

    def client(self, factory=claim_factory):
        return TestClient(create_app(settings=self.settings,agent_factory=fake_factory(),claim_agent_factory=factory))

    def complete(self,client):
        response=client.post("/claim-check",json={"claim":CLAIM})
        self.assertEqual(response.status_code,202,response.text)
        run=wait_for(client,response.json()["run_id"])
        self.assertEqual(run["status"],"completed",run.get("error"))
        return run

    def test_post_get_claim_check(self):
        with self.client() as client:
            run=self.complete(client)
            read=client.get("/claim-check/"+run["run_id"]).json()
            self.assertEqual(read["mode"],"claim_check")
            self.assertEqual(read["result"]["request"]["claim"],CLAIM)

    def test_unified_history_preserves_both_modes(self):
        with self.client() as client:
            self.complete(client)
            research=client.post("/research",json={"question":"A research question"}).json()
            self.assertEqual(wait_for(client,research["run_id"])["status"],"completed")
            self.assertEqual({r["mode"] for r in client.get("/research").json()},{"research","claim_check"})

    def test_exports_share_persisted_result(self):
        with self.client() as client:
            run=self.complete(client); base="/claim-check/"+run["run_id"]
            self.assertEqual(client.get(base+"/export/json").json(),run["result"])
            self.assertIn(CAVEAT,client.get(base+"/export/markdown").text)
            self.assertIn("Claim-by-Claim Literature Map",client.get(base+"/view").text)

    def test_reopen_after_server_restart(self):
        with self.client() as client:
            run=self.complete(client)
        with self.client() as client:
            reopened=client.get("/claim-check/"+run["run_id"]).json()
            self.assertEqual(reopened["result"],run["result"])
            self.assertEqual(reopened["status"],"completed")

    def test_invalid_request_creates_no_run(self):
        with self.client() as client:
            for body in [{"claim":" "},{"claim":CLAIM,"year_from":True},{"claim":CLAIM,"max_search_rounds":3}]:
                self.assertEqual(client.post("/claim-check",json=body).status_code,422)
            self.assertEqual(client.get("/research").json(),[])

    def test_failure_stores_no_partial_result(self):
        def failed(callback):
            return SimpleNamespace(run=MagicMock(side_effect=RuntimeError("private failure text")))
        with self.client(failed) as client:
            created=client.post("/claim-check",json={"claim":CLAIM}).json()
            run=wait_for(client,created["run_id"])
            self.assertEqual(run["status"],"failed")
            self.assertIsNone(run["result"])
            self.assertNotIn("private failure text",run["error"])
            self.assertEqual(client.get("/claim-check/"+run["run_id"]+"/export/json").status_code,409)

    def test_ui_mode_selector_and_claim_fields(self):
        with self.client() as client:
            html=client.get("/").text
            for text in ["Check My Idea","Research Question",'id="claim-field"','id="claim-context"']:
                self.assertIn(text,html)
            js=client.get("/assets/app.js").text
            self.assertIn("Check Prior Work",js)
            self.assertIn('"/claim-check"',js)

    def test_old_sqlite_schema_migrates_without_loss(self):
        with closing(sqlite3.connect(self.settings.run_db)) as db, db:
            db.execute("""CREATE TABLE runs (run_id TEXT PRIMARY KEY, question TEXT NOT NULL,
                request_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                status TEXT NOT NULL, progress_json TEXT NOT NULL, termination_reason TEXT,
                result_json TEXT, error TEXT)""")
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("legacy","Old research",'{"question":"Old research"}',"2025","2025","failed","{}",None,None,"Interrupted"))
        store=RunStore(self.settings.run_db)
        self.assertEqual(store.get("legacy").mode,"research")
        self.assertEqual(store.get("legacy").question,"Old research")
        RunStore(self.settings.run_db)  # migration is idempotent
        self.assertEqual(len(store.history()),1)
