"""Local project API and escaped workspace views with the real fake-agent lifecycle."""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

from fastapi.testclient import TestClient

from phase5_helpers import OfflineTest
from phase12_helpers import research_fixture, QUESTION
from claim_check_helpers import fixture as idea_fixture, CLAIM
from test_product import wait_for
from researchpilot.api import create_app
from researchpilot.config import Settings
from researchpilot.run_store import RunStore
from researchpilot.project_views import project_export_data, project_markdown
from researchpilot.project_models import ProjectCreate
from researchpilot.research_models import ResearchRequest
from researchpilot.run_manager import RunManager


class ProjectProductTests(OfflineTest):
    def setUp(self):
        for name in ("httpx.HTTPTransport.handle_request", "httpx.AsyncHTTPTransport.handle_async_request", "socket.create_connection"):
            guard = patch(name, side_effect=AssertionError("Offline project test"))
            guard.start(); self.addCleanup(guard.stop)
        self.root = self.temporary_directory()
        self.settings = Settings(cache_dir=self.root / "cache", run_db=self.root / "runs.db")

    @contextmanager
    def client(self):
        def factory(callback):
            f = research_fixture(); f.agent._on_progress = callback
            return f.agent
        def idea(callback):
            f = idea_fixture(); f.agent.on_progress = callback
            return f.agent
        with TestClient(create_app(settings=self.settings, agent_factory=factory, claim_agent_factory=idea)) as client:
            yield client

    def create(self, client, **changes):
        response = client.post("/projects", json=dict(title="Low-Rank Recovery", description="Spectral recovery", **changes))
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["project_id"]

    def research(self, client, pid=None):
        response = client.post("/research", json=dict(question=QUESTION, project_id=pid, max_search_rounds=1))
        self.assertEqual(response.status_code, 202, response.text)
        run = wait_for(client, response.json()["run_id"])
        self.assertEqual(run["status"], "completed", run)
        return run

    def test_project_create(self):
        with self.client() as c:
            self.assertTrue(self.create(c))

    def test_project_list(self):
        with self.client() as c:
            pid = self.create(c)
            self.assertEqual(c.get("/projects").json()[0]["project_id"], pid)

    def test_project_detail(self):
        with self.client() as c:
            pid = self.create(c)
            self.assertEqual(c.get(f"/projects/{pid}").json()["project"]["title"], "Low-Rank Recovery")

    def test_project_update_and_archive(self):
        with self.client() as c:
            pid = self.create(c)
            self.assertEqual(c.patch(f"/projects/{pid}", json={"title": "Changed", "status": "archived"}).status_code, 200)
            self.assertEqual(c.post("/research", json={"question": QUESTION, "project_id": pid}).status_code, 409)
            self.assertEqual(c.get(f"/projects/{pid}").json()["project"]["title"], "Changed")

    def test_questions_endpoint(self):
        with self.client() as c:
            pid = self.create(c)
            q = c.post(f"/projects/{pid}/questions", json={"text": QUESTION}).json()
            self.assertEqual(c.get(f"/projects/{pid}/questions").json()[0], q)

    def test_claims_endpoint(self):
        with self.client() as c:
            pid = self.create(c)
            a = c.post(f"/projects/{pid}/claims", json={"text": CLAIM}).json()
            self.assertEqual(c.get(f"/projects/{pid}/claims").json()[0], a)
            self.assertEqual(a["status"], "proposed")

    def test_notes_endpoint(self):
        with self.client() as c:
            pid = self.create(c)
            self.assertEqual(c.post(f"/projects/{pid}/notes", json={"text": "User thought"}).status_code, 201)
            self.assertEqual(c.get(f"/projects/{pid}/evidence").json(), [])

    def test_archive_question_and_claim(self):
        with self.client() as c:
            pid = self.create(c)
            for resource, key in [("questions", "question_id"), ("claims", "claim_id")]:
                row = c.post(f"/projects/{pid}/{resource}", json={"text": QUESTION}).json()
                self.assertEqual(c.patch(f"/projects/{pid}/{resource}/{row[key]}", json={"status": "archived"}).json()["status"], "archived")

    def test_paper_and_evidence_endpoints(self):
        with self.client() as c:
            pid = self.create(c); self.research(c, pid)
            self.assertEqual(len(c.get(f"/projects/{pid}/papers").json()), 1)
            self.assertEqual(len(c.get(f"/projects/{pid}/evidence").json()), 1)

    def test_gap_endpoint(self):
        with self.client() as c:
            pid = self.create(c); self.research(c, pid)
            self.assertEqual(c.get(f"/projects/{pid}/gaps").json()[0]["status"], "unresolved")

    def test_research_carries_project_context(self):
        with self.client() as c:
            pid = self.create(c); self.research(c, pid)
            second = self.research(c, pid)
            self.assertEqual(second["project_id"], pid)
            self.assertTrue(second["project_context"]["evidence"])
            self.assertEqual(second["result"]["project_usage"]["queries_reused"], 3)
            self.assertEqual(len(c.get(f"/projects/{pid}/history").json()), 2)

    def test_idea_carries_project_context(self):
        with self.client() as c:
            pid = self.create(c); self.research(c, pid)
            r = c.post("/claim-check", json={"claim": CLAIM + " Investigate spectral recovery.", "project_id": pid, "max_search_rounds": 1})
            self.assertEqual(r.status_code, 202, r.text)
            run = wait_for(c, r.json()["run_id"])
            self.assertEqual(run["status"], "completed", run)
            self.assertTrue(run["result"]["project_usage"]["reused_paper_ids"])
            self.assertTrue(run["project_context"]["gaps"])

    def test_invalid_project_rejected(self):
        with self.client() as c:
            for route, body in [("/research", {"question": QUESTION}), ("/claim-check", {"claim": CLAIM})]:
                self.assertEqual(c.post(route, json=body | {"project_id": "not-found"}).status_code, 404)

    def test_standalone_requests_accepted(self):
        with self.client() as c:
            self.assertIsNone(self.research(c)["project_id"])

    def test_continue_launches_single_normal_run(self):
        with self.client() as c:
            pid = self.create(c); self.research(c, pid)
            gap = c.get(f"/projects/{pid}/gaps").json()[0]
            response = c.post(f"/projects/{pid}/continue", json={"target_type": "gap", "target_id": gap["gap_id"]})
            self.assertEqual(response.status_code, 202, response.text)
            run = wait_for(c, response.json()["run_id"])
            self.assertEqual(run["status"], "completed", run)
            self.assertEqual(run["request"]["target_gap_id"], gap["gap_id"])
            self.assertEqual(len(c.get(f"/projects/{pid}/history").json()), 2)

    def test_continue_foreign_target_rejected(self):
        with self.client() as c:
            pid, other = self.create(c), self.create(c)
            q = c.post(f"/projects/{other}/questions", json={"text": QUESTION}).json()
            self.assertEqual(c.post(f"/projects/{pid}/continue", json={"target_type": "question", "target_id": q["question_id"]}).status_code, 404)

    def test_export_complete_json(self):
        with self.client() as c:
            pid = self.create(c); self.research(c, pid)
            response = c.get(f"/projects/{pid}/export/json")
            data = response.json()
            self.assertTrue(data["runs"][0]["result"])
            self.assertTrue(data["runs"][0]["project_context"])
            self.assertEqual(len(data["queries"]), 3)
            self.assertTrue(data["evidence"][0]["attestations"])
            self.assertIn("attachment", response.headers["Content-Disposition"])

    def test_export_markdown(self):
        with self.client() as c:
            pid = self.create(c); self.research(c, pid)
            text = c.get(f"/projects/{pid}/export/markdown").text
            for heading in ["Research Questions", "Claims", "Evidence-grounded Findings", "Research Gaps", "Papers", "User Notes (unverified)"]:
                self.assertIn("## " + heading, text)

    def test_json_export_excludes_cached_candidates_but_preserves_query_audit_and_cache(self):
        with self.client() as c:
            pid = self.create(c); self.research(c, pid)
            store = RunStore(self.settings.run_db)
            before = store.projects.list_records(pid, "queries")
            self.assertTrue(any(q.candidates for q in before))
            data = c.get(f"/projects/{pid}/export/json").json()
            self.assertEqual([q["query_id"] for q in data["queries"]], [q.query_id for q in before])
            self.assertTrue(all("candidates" not in q and q["query"] for q in data["queries"]))
            self.assertEqual(store.projects.list_records(pid, "queries"), before)

    def test_json_export_excludes_preselection_passages_but_preserves_verified_result(self):
        with self.client() as c:
            pid = self.create(c); run = self.research(c, pid)
            original = RunStore(self.settings.run_db).get(run["run_id"]).result
            self.assertTrue(original.retrieved_passages)
            data = c.get(f"/projects/{pid}/export/json").json()
            result = data["runs"][0]["result"]
            self.assertNotIn("retrieved_passages", result)
            self.assertEqual(result["answer"], original.answer)
            self.assertEqual(result["claims"], original.model_dump(mode="json")["claims"])
            self.assertEqual(result["evidence"], original.model_dump(mode="json")["evidence"])
            self.assertEqual(RunStore(self.settings.run_db).get(run["run_id"]).result, original)

    def test_export_projection_removes_unselected_noise_without_mutating_source(self):
        from copy import deepcopy
        data = {"queries": [{"query": "spectral recovery", "candidates": [{"title": "UNSELECTED_NOISE"}]}],
                "runs": [{"status": "failed", "result": None}, {"status": "completed", "result": {
                    "answer": "Verified finding", "retrieved_passages": [{"text": "UNSELECTED_NOISE"}]}}],
                "evidence": [{"text": "Verified passage"}], "notes": [{"text": "Unverified user note"}]}
        original = deepcopy(data)
        exported = project_export_data(data)
        self.assertNotIn("UNSELECTED_NOISE", json.dumps(exported))
        self.assertEqual(exported["evidence"], original["evidence"])
        self.assertEqual(exported["notes"], original["notes"])
        self.assertIsNone(exported["runs"][0]["result"])
        self.assertEqual(data, original)

    def test_state_survives_app_restart(self):
        with self.client() as c:
            pid = self.create(c); self.research(c, pid)
            before = c.get(f"/projects/{pid}/export/json").json()
        with self.client() as c:
            self.assertEqual(c.get(f"/projects/{pid}/export/json").json(), before)

    def test_cross_origin_patch_rejected(self):
        with self.client() as c:
            pid = self.create(c)
            self.assertEqual(c.patch(f"/projects/{pid}", json={"title": "attack"}, headers={"Origin": "https://untrusted.example"}).status_code, 403)

    def test_blank_project_rejected(self):
        with self.client() as c:
            self.assertEqual(c.post("/projects", json={"title": "  "}).status_code, 422)

    def test_user_cannot_submit_fake_verified_evidence(self):
        with self.client() as c:
            pid = self.create(c)
            self.assertEqual(c.post(f"/projects/{pid}/evidence", json={"text": "claim proof"}).status_code, 405)

    def rendered(self, c):
        pid = self.create(c); self.research(c, pid)
        return c.get(f"/projects/{pid}/view").text

    def test_ui_project_list_and_create_form_loads(self):
        with self.client() as c:
            html = c.get("/").text
            self.assertIn('id="project-list"', html)
            self.assertIn('id="project-create"', html)
            self.assertIn('/assets/projects.js', html)

    def test_ui_project_detail_renders(self):
        with self.client() as c:
            self.assertIn("Low-Rank Recovery", self.rendered(c))

    def test_ui_questions_render(self):
        with self.client() as c:
            self.assertIn(QUESTION, self.rendered(c))

    def test_ui_claims_render(self):
        with self.client() as c:
            html = self.rendered(c)
            self.assertIn('id="project-claims"', html)
            self.assertIn('data-continue="claim"', html)

    def test_ui_papers_render(self):
        with self.client() as c:
            html = self.rendered(c)
            self.assertIn("Spectral recovery A", html)
            self.assertIn("openalex", html)

    def test_ui_evidence_renders(self):
        with self.client() as c:
            html = self.rendered(c)
            self.assertIn('class="passage-text"', html)
            self.assertIn('data-evidence-link', html)

    def test_ui_open_gaps_render(self):
        with self.client() as c:
            self.assertIn('data-continue="gap"', self.rendered(c))

    def test_ui_escapes_user_text(self):
        with self.client() as c:
            pid = self.create(c)
            c.post(f"/projects/{pid}/notes", json={"text": '<img src=x onerror="alert(1)">'})
            text = c.get(f"/projects/{pid}/view").text
            self.assertNotIn("<img", text)
            self.assertIn("&lt;img", text)

    def test_ui_create_and_run_actions_in_dom_harness(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Optional Node runtime unavailable; server-rendered view contracts remain tested")
        result = subprocess.run([node, str(Path(__file__).with_name("project_ui_harness.cjs"))], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("create, research, idea, continue, archive: passed", result.stdout)

    def test_reused_dependency_does_not_leak_context_to_standalone_run(self):
        store = RunStore(self.settings.run_db)
        pid = store.projects.create(ProjectCreate(title="Project")).project_id
        f = research_fixture()
        original_search = f.agent.paper_search
        manager = RunManager(store, self.settings, agent_factory=lambda callback: f.agent)
        self.addCleanup(manager.close)
        for project_id in (pid, None):
            created = store.create(ResearchRequest(question=QUESTION, project_id=project_id, max_search_rounds=1))
            manager._execute(created.run_id, store.get(created.run_id).request)
            self.assertEqual(store.get(created.run_id).status, "completed")
        self.assertIsNone(f.planner.plan.call_args.args[0].project_context)
        self.assertIs(f.agent.paper_search, original_search)
