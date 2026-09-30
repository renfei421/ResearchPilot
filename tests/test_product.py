"""Offline product contracts using the real graph with synthetic providers."""

from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import re
import httpx
from threading import Event
from time import monotonic, sleep
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from pydantic import SecretStr

from phase5_helpers import OfflineTest, QUESTION
from test_research_graph import A_TEXT, B_TEXT, loop_fixture, missing_b
from researchpilot.api import create_app
from researchpilot.config import Settings
from researchpilot.product_views import markdown_export, result_html
from researchpilot.progress import ResearchProgress
from researchpilot.research_agent import ResearchAgent
from researchpilot.research_models import ResearchRequest, ResearchResult
from researchpilot.run_manager import QueueFullError, RunManager
from researchpilot.run_store import RunStore


def fake_factory(events=None, started=None, release=None, error=None):
    def factory(callback):
        fixture = loop_fixture()
        def progress(event):
            if events is not None:
                events.append(event)
            callback(event)
        fixture.agent._on_progress = progress
        class FakeAgent:
            def run(self, request):
                if started is not None:
                    started.set()
                if release is not None and not release.wait(5):
                    raise RuntimeError("Offline test gate timed out")
                if error is not None:
                    raise error
                fixture.planner.plan.return_value = fixture.planner.plan.return_value.model_copy(
                    update={"research_question": request.question})
                return fixture.agent.run(request)
        return FakeAgent()
    return factory


def wait_for(client, run_id, statuses=("completed", "failed")):
    deadline = monotonic() + 5
    while monotonic() < deadline:
        response = client.get(f"/research/{run_id}")
        if response.status_code != 200:
            raise AssertionError(response.text)
        run = response.json()
        if run["status"] in statuses:
            return run
        sleep(0.01)
    raise AssertionError(f"Run did not reach {statuses}")


class ProductTests(OfflineTest):
    def setUp(self):
        # Windows asyncio uses a loopback socketpair internally. Block outbound
        # transports instead of socket.connect so TestClient's event loop works.
        for name in ("httpx.HTTPTransport.handle_request", "httpx.AsyncHTTPTransport.handle_async_request",
                     "socket.create_connection"):
            guard = patch(name, side_effect=AssertionError("Product tests must not call external services"))
            guard.start()
            self.addCleanup(guard.stop)
        self.root = self.temporary_directory()
        self.settings = Settings(cache_dir=self.root / "cache", run_db=self.root / "runs.sqlite3")

    @contextmanager
    def client(self, factory=None, settings=None):
        app = create_app(settings=settings or self.settings, agent_factory=factory or fake_factory())
        with TestClient(app) as client:
            yield client

    def completed(self, client, **fields):
        response = client.post("/research", json={"question": QUESTION, "max_queries": 3} | fields)
        self.assertEqual(response.status_code, 202, response.text)
        run = wait_for(client, response.json()["run_id"])
        self.assertEqual(run["status"], "completed", run)
        return run

    def test_health_does_not_construct_or_call_openai(self):
        with patch("researchpilot.run_manager.OpenAI") as sdk, self.client() as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok", "openai_configured": False})
            sdk.assert_not_called()

    def test_configured_health_exposes_only_boolean(self):
        settings = Settings(openai_api_key=SecretStr("test-key-do-not-expose"), run_db=self.settings.run_db)
        with self.client(settings=settings) as client:
            response = client.get("/health")
            self.assertEqual(response.json(), {"status": "ok", "openai_configured": True})
            self.assertNotIn("test-key-do-not-expose", response.text)

    def test_post_is_nonblocking_and_converts_request(self):
        started, release = Event(), Event()
        self.addCleanup(release.set)
        with self.client(fake_factory(started=started, release=release)) as client:
            try:
                response = client.post("/research", json={"question": "  " + QUESTION + "  ",
                    "year_from": 2020, "year_to": 2026, "max_papers": 5, "max_search_rounds": 2})
                self.assertEqual(response.status_code, 202)
                created = response.json()
                self.assertEqual(created["status"], "queued")
                self.assertTrue(started.wait(2))
                self.assertFalse(release.is_set())
                run = client.get(response.headers["Location"]).json()
                self.assertEqual(run["status"], "running")
                self.assertEqual(run["request"]["question"], QUESTION)
                self.assertEqual(run["request"]["year_from"], 2020)
                self.assertEqual(run["request"]["year_to"], 2026)
                self.assertEqual(run["request"]["max_papers"], 5)
                self.assertIsNone(run["result"])
            finally:
                release.set()
            self.assertEqual(wait_for(client, created["run_id"])["status"], "completed")

    def test_invalid_requests_create_no_run_and_omit_input(self):
        with self.client() as client:
            for fields in ({"question": " "}, {"question": 123}, {"max_papers": 0},
                           {"max_search_rounds": 4}, {"year_from": True},
                           {"year_from": 2025, "year_to": 2020}, {"openai_api_key": "secret-sent-value"}):
                with self.subTest(fields=fields):
                    response = client.post("/research", json={"question": QUESTION} | fields)
                    self.assertEqual(response.status_code, 422)
                    self.assertNotIn("secret-sent-value", response.text)
            self.assertEqual(client.get("/research").json(), [])

    def test_completed_result_serialization_and_progress(self):
        events = []
        with self.client(fake_factory(events=events)) as client:
            run = self.completed(client)
            result = ResearchResult.model_validate(run["result"])
            self.assertEqual(result.question, QUESTION)
            self.assertEqual(run["current_stage"], "completed")
            self.assertEqual(run["current_round"], 2)
            self.assertEqual(run["selected_papers"], 2)
            self.assertEqual(run["evidence_collected"], 2)
            self.assertEqual(run["termination_reason"], "sufficient_evidence")
            self.assertEqual(result, ResearchResult.model_validate_json(result.model_dump_json()))
        stages = [event.current_stage for event in events]
        for stage in ("planning", "searching", "reranking", "acquiring", "retrieving", "selecting",
                      "synthesizing", "verifying", "assessing", "follow_up", "finalizing"):
            self.assertIn(stage, stages)
        self.assertTrue(any(e.current_round == 2 and e.evidence_collected == 2 for e in events))
        self.assertFalse(any(A_TEXT in e.model_dump_json() for e in events))

    def test_running_progress_can_be_polled_before_result_is_ready(self):
        ready, release = Event(), Event()
        def factory(callback):
            class Agent:
                def run(self, request):
                    callback(ResearchProgress(current_stage="verifying", current_round=2,
                        selected_papers=3, evidence_collected=4, warnings=["Abstract fallback used."]))
                    ready.set()
                    if not release.wait(5):
                        raise RuntimeError("Test gate timed out")
                    return loop_fixture().agent.run(request)
            return Agent()
        with self.client(factory) as client:
            try:
                run_id = client.post("/research", json={"question": QUESTION}).json()["run_id"]
                self.assertTrue(ready.wait(2))
                run = client.get(f"/research/{run_id}").json()
                self.assertEqual((run["current_stage"], run["current_round"]), ("verifying", 2))
                self.assertEqual((run["selected_papers"], run["evidence_collected"]), (3, 4))
                self.assertEqual(run["warnings"], ["Abstract fallback used."])
                for suffix in ("view", "export/json", "export/markdown"):
                    self.assertEqual(client.get(f"/research/{run_id}/{suffix}").status_code, 409)
            finally:
                release.set()
            self.assertEqual(wait_for(client, run_id)["status"], "completed")

    def test_empty_retrieval_publishes_explicit_insufficiency(self):
        def factory(callback):
            fixture = loop_fixture()
            fixture.initial_records = []
            fixture.agent._on_progress = callback
            return fixture.agent
        with self.client(factory) as client:
            run = self.completed(client, max_search_rounds=1)
            self.assertEqual(run["selected_papers"], 0)
            self.assertEqual(run["evidence_collected"], 0)
            self.assertIn("Insufficient evidence", run["result"]["answer"])
            self.assertTrue(run["warnings"])
            html = client.get(f"/research/{run['run_id']}/view").text
            self.assertIn("Insufficient evidence", html)
            self.assertNotIn('class="citation"', html)
            self.assertEqual(run["termination_reason"], "no_literature_found")
            self.assertIsNone(run["error"])

    def test_retrieval_failure_has_durable_secret_safe_query_diagnostics(self):
        secret = "test-openalex-private-key"
        settings = Settings(openalex_api_key=SecretStr(secret), cache_dir=self.root / "cache",
                            run_db=self.settings.run_db)
        def factory(callback):
            fixture = loop_fixture()
            def search(**kwargs):
                request = httpx.Request("GET", f"https://api.openalex.org/works?api_key={secret}")
                httpx.Response(400, request=request).raise_for_status()
            fixture.raw.search_works.side_effect = search
            fixture.agent._on_progress = callback
            return fixture.agent
        with self.client(factory, settings) as client:
            run_id = client.post("/research", json={"question": QUESTION, "year_from": 2010,
                                                    "year_to": 2020}).json()["run_id"]
            run = wait_for(client, run_id)
            self.assertEqual(run["status"], "failed")
            self.assertIn("Invalid/generated search query", run["error"])
            self.assertIsNone(run["result"])
            # FileHandler flushes diagnostics before process shutdown.
            text = (settings.cache_dir / "researchpilot.log").read_text(encoding="utf-8")
            records = [json.loads(line) for line in text.splitlines()]
            failure = next(x for x in records if x.get("event") == "literature_query_failed")
            self.assertEqual(failure["run_id"], run_id)
            self.assertEqual(failure["http_status"], 400)
            self.assertEqual(failure["query_id"], "q1")
            self.assertEqual((failure["year_from"], failure["year_to"]), (2010, 2020))
            self.assertTrue(failure["traceback"])
            self.assertNotIn(secret, text + json.dumps(run))

    def test_partial_retrieval_warning_survives_http_persistence_and_export(self):
        def factory(callback):
            fixture = loop_fixture()
            original = fixture.raw.search_works.side_effect
            def search(query, **kwargs):
                if query == fixture.planner.plan.return_value.queries[1].text:
                    raise httpx.ReadTimeout("temporary read failure")
                return original(query, **kwargs)
            fixture.raw.search_works.side_effect = search
            fixture.agent._on_progress = callback
            return fixture.agent
        with self.client(factory) as client:
            run = self.completed(client, max_search_rounds=1)
            self.assertGreater(run["selected_papers"], 0)
            self.assertTrue(any("q2" in w and "successful queries only" in w for w in run["warnings"]))
            self.assertEqual(client.get(f"/research/{run['run_id']}/export/json").json(), run["result"])
            self.assertIn("successful queries only", client.get(f"/research/{run['run_id']}/view").text)

    def test_total_network_failure_is_failed_not_empty_completed_result(self):
        def factory(callback):
            fixture = loop_fixture()
            fixture.raw.search_works.side_effect = httpx.ConnectError("temporary network failure")
            fixture.agent._on_progress = callback
            return fixture.agent
        with self.client(factory) as client:
            run_id = client.post("/research", json={"question": QUESTION}).json()["run_id"]
            run = wait_for(client, run_id)
            self.assertEqual(run["status"], "failed")
            self.assertIn("temporarily unavailable", run["error"])
            self.assertIsNone(run["result"])

    def test_mismatched_result_question_is_not_published(self):
        factory = lambda callback: Mock(run=Mock(return_value=loop_fixture().agent.run(ResearchRequest(question=QUESTION))))
        with self.client(factory) as client:
            run_id = client.post("/research", json={"question": "A different question"}).json()["run_id"]
            run = wait_for(client, run_id)
            self.assertEqual(run["status"], "failed")
            self.assertIn("failed validation", run["error"])
            self.assertIsNone(run["result"])

    def test_queue_runs_in_order_and_runs_remain_isolated(self):
        started, release = Event(), Event()
        with self.client(fake_factory(started=started, release=release)) as client:
            try:
                first = client.post("/research", json={"question": QUESTION}).json()["run_id"]
                self.assertTrue(started.wait(2))
                second_question = "How do cache capacity constraints affect latency?"
                second = client.post("/research", json={"question": second_question}).json()["run_id"]
                self.assertNotEqual(first, second)
                self.assertEqual(client.get(f"/research/{first}").json()["status"], "running")
                self.assertEqual(client.get(f"/research/{second}").json()["status"], "queued")
            finally:
                release.set()
            a, b = wait_for(client, first), wait_for(client, second)
            self.assertEqual(a["result"]["question"], QUESTION)
            self.assertEqual(b["result"]["question"], second_question)
            self.assertEqual(len(a["result"]["round_trace"]), 2)
            self.assertEqual(len(b["result"]["round_trace"]), 2)

    def test_failure_is_persistent_and_secret_safe_in_logs_and_browser(self):
        secret = "fake-secret-that-must-not-be-saved"
        settings = Settings(openai_api_key=SecretStr(secret), run_db=self.settings.run_db)
        with self.assertLogs("researchpilot.runs", level="INFO") as logs:
            with self.client(fake_factory(error=RuntimeError(f"secret={secret}; hidden prompt text")), settings) as client:
                created = client.post("/research", json={"question": QUESTION}).json()
                run = wait_for(client, created["run_id"])
                self.assertEqual(run["status"], "failed")
                self.assertEqual(run["current_stage"], "failed")
                self.assertIsNone(run["result"])
                self.assertNotIn(secret, json.dumps(run))
                self.assertNotIn("hidden prompt", json.dumps(run))
                self.assertIn("could not finish", run["error"])
                self.assertEqual(client.get(f"/research/{created['run_id']}/export/json").status_code, 409)
        output = "\n".join(logs.output)
        self.assertIn('"traceback"', output)
        self.assertIn("RuntimeError", output)
        self.assertNotIn(secret, output)
        self.assertNotIn("hidden prompt", output)
        self.assertNotIn(secret.encode(), self.settings.run_db.read_bytes())
        self.assertEqual(RunStore(self.settings.run_db).get(created["run_id"]).status, "failed")

    def test_missing_credentials_fails_safely_without_sdk(self):
        app = create_app(settings=self.settings)
        with patch("researchpilot.run_manager.OpenAI") as sdk, TestClient(app) as client:
            run_id = client.post("/research", json={"question": QUESTION}).json()["run_id"]
            run = wait_for(client, run_id)
            self.assertEqual(run["status"], "failed")
            self.assertIn("API credentials unavailable", run["error"])
            sdk.assert_not_called()

    def test_unexpected_http_error_does_not_escape_as_raw_traceback(self):
        with self.client() as client, self.assertLogs("researchpilot.runs", level="ERROR") as logs:
            with patch.object(client.app.state.store, "history", side_effect=RuntimeError("private-key-and-prompt")):
                # TestClient normally re-raises unhandled exceptions, as Uvicorn
                # would log them. A returned 500 proves the safe boundary caught it.
                response = client.get("/research")
            self.assertEqual(response.status_code, 500)
            self.assertNotIn("private-key-and-prompt", response.text)
        self.assertNotIn("private-key-and-prompt", "\n".join(logs.output))
        self.assertIn("traceback", "\n".join(logs.output))

    def test_history_pagination_and_missing_runs(self):
        with self.client() as client:
            a = self.completed(client)
            b = self.completed(client)
            recent = client.get("/research?limit=1").json()
            older = client.get("/research?limit=1&offset=1").json()
            self.assertEqual(recent[0]["run_id"], b["run_id"])
            self.assertEqual(older[0]["run_id"], a["run_id"])
            self.assertNotIn("result", recent[0])
            for path in ("/research/missing", "/research/missing/view", "/research/missing/export/json",
                         "/research/missing/export/markdown"):
                self.assertEqual(client.get(path).status_code, 404)
            for query in ("limit=0", "limit=101", "offset=-1"):
                self.assertEqual(client.get("/research?" + query).status_code, 422)

    def test_store_and_app_recreation_preserve_completed_results(self):
        with self.client() as client:
            run = self.completed(client)
        store = RunStore(self.settings.run_db)
        self.assertEqual(store.get(run["run_id"]).result.model_dump(mode="json"), run["result"])
        with self.client() as restarted:
            self.assertEqual(restarted.get(f"/research/{run['run_id']}").json(), run)
            self.assertIn(A_TEXT, restarted.get(f"/research/{run['run_id']}/view").text)

    def test_startup_marks_only_interrupted_runs_failed(self):
        store = RunStore(self.settings.run_db)
        queued = store.create(ResearchRequest(question=QUESTION))
        active = store.create(ResearchRequest(question=QUESTION))
        store.start(active.run_id)
        with self.client() as client:
            for run_id in (queued.run_id, active.run_id):
                run = client.get(f"/research/{run_id}").json()
                self.assertEqual(run["status"], "failed")
                self.assertIn("restarted", run["error"])

    def test_exports_preserve_citations_question_sources_uncertainty(self):
        with self.client() as client:
            run = self.completed(client)
            root = f"/research/{run['run_id']}/export"
            raw = client.get(root + "/json")
            md = client.get(root + "/markdown")
            self.assertEqual(raw.json(), run["result"])
            self.assertIn("attachment", raw.headers["content-disposition"])
            self.assertIn(".md", md.headers["content-disposition"])
            self.assertIn(QUESTION, md.text)
            for text in ("[P1, p. 3]", "[P2, abstract]", "## Sources", "## Remaining uncertainty",
                         "Repeated-read shared caches", "Contention-aware cache allocation"):
                self.assertIn(text, md.text)

    def test_frontend_loads_with_local_assets_and_result_citations_resolve(self):
        with self.client() as client:
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn('id="research-form"', page.text)
            self.assertIn('lang="zh-CN"', page.text)
            self.assertIn("frame-ancestors 'none'", page.headers["content-security-policy"])
            for path in ("/assets/app.js", "/assets/app.css"):
                self.assertEqual(client.get(path).status_code, 200)
            run = self.completed(client)
            html = client.get(f"/research/{run['run_id']}/view").text
            for expected in (A_TEXT, B_TEXT, "来源文献", "Repeated-read shared caches", "[P1, p. 3]", "[P2, abstract]"):
                self.assertIn(expected, html)
            ids = re.findall(r'class="citation" href="#([^"]+)"', html)
            self.assertEqual(len(ids), 2)
            for anchor in ids:
                self.assertIn(f'id="{anchor}"', html)
            self.assertIn('<details class="trace">', html)
            self.assertNotIn('class="trace" open', html)
            self.assertIn("https://papers.example/study.pdf", html)

    def test_cross_origin_and_untrusted_hosts_are_rejected(self):
        with self.client() as client:
            self.assertEqual(client.post("/research", json={"question": QUESTION},
                headers={"Origin": "https://unrelated.example"}).status_code, 403)
            self.assertEqual(client.get("/health", headers={"Host": "unrelated.example"}).status_code, 400)

    def test_store_redacts_credentials_even_if_provider_warning_echoes_them(self):
        secret = "test-provider-key-should-never-persist"
        settings = Settings(openai_api_key=SecretStr(secret), run_db=self.settings.run_db)
        store = RunStore(settings.run_db, redact=settings.redact)
        created = store.create(ResearchRequest(question=QUESTION + secret))
        store.start(created.run_id)
        store.progress(created.run_id, ResearchProgress(current_stage="searching", warnings=[secret]))
        result = loop_fixture().agent.run(ResearchRequest(question=QUESTION))
        result.warnings.append(secret)
        store.complete(created.run_id, result)
        self.assertNotIn(secret, store.get(created.run_id).model_dump_json())
        self.assertNotIn(secret.encode(), settings.run_db.read_bytes())

    def test_progress_cannot_reopen_completed_run(self):
        store = RunStore(self.settings.run_db)
        created = store.create(ResearchRequest(question=QUESTION))
        store.start(created.run_id)
        store.complete(created.run_id, loop_fixture().agent.run(ResearchRequest(question=QUESTION)))
        store.progress(created.run_id, ResearchProgress(current_stage="verifying"))
        store.fail(created.run_id, "late error")
        self.assertEqual(store.get(created.run_id).status, "completed")
        self.assertEqual(store.get(created.run_id).current_stage, "completed")

    def test_queue_capacity_rejects_without_creating_extra_run(self):
        store = RunStore(self.settings.run_db)
        started, release = Event(), Event()
        manager = RunManager(store, self.settings, fake_factory(started=started, release=release), capacity=1)
        try:
            manager.submit(ResearchRequest(question=QUESTION))
            self.assertTrue(started.wait(2))
            with self.assertRaises(QueueFullError):
                manager.submit(ResearchRequest(question="Another research question"))
            self.assertEqual(len(store.history()), 1)
        finally:
            release.set()
            manager.close()

    def test_default_worker_reuses_research_agent_and_closes_sdk(self):
        settings = Settings(openai_api_key=SecretStr("fake-sdk-key"), run_db=self.settings.run_db)
        expected = loop_fixture().agent.run(ResearchRequest(question=QUESTION))
        with patch("researchpilot.run_manager.OpenAI") as sdk, patch("researchpilot.run_manager.ResearchAgent") as agent:
            agent.return_value.run.return_value = expected
            with TestClient(create_app(settings=settings)) as client:
                self.completed(client)
            sdk.assert_called_once_with(api_key="fake-sdk-key", max_retries=0, timeout=60.0)
            self.assertIs(agent.call_args.kwargs["openai_client"], sdk.return_value.__enter__.return_value)
            self.assertIs(agent.call_args.kwargs["settings"], settings)
            sdk.return_value.__exit__.assert_called_once()

    def test_production_run_never_reads_eval_files(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            self.assertNotIn("eval", path.parts)
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded), self.client() as client:
            self.completed(client)
        for module in ("api", "run_manager", "run_store", "product_views"):
            source = (Path(__file__).parents[1] / "src" / "researchpilot" / f"{module}.py").read_text(encoding="utf-8")
            self.assertNotRegex(source, r"(?:from|import) eval\b")


class ProductPresentationTests(OfflineTest):
    def test_html_escapes_model_prose_and_source_title(self):
        f = loop_fixture()
        f.a.title = '<script>alert("title")</script>'
        f.a.abstract = 'Shared caches reduce latency <img src=x onerror="alert(1)">.'
        result = f.agent.run(ResearchRequest(question=QUESTION))
        html = result_html(result)
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;img", html)

    def test_unsupported_claim_is_absent_from_web_and_markdown(self):
        f = loop_fixture()
        f.statuses[f.b.paper_id] = "unsupported"
        f.assessor.assess.side_effect = lambda *args, prior_gaps=(): missing_b(prior_gaps)
        result = f.agent.run(ResearchRequest(question=QUESTION))
        self.assertNotIn(B_TEXT, result_html(result))
        self.assertNotIn(B_TEXT, markdown_export(result))
        self.assertIn(A_TEXT, result_html(result))

    def test_structured_progress_does_not_change_graph_result(self):
        request = ResearchRequest(question=QUESTION)
        # Wall-clock diagnostics vary independently of progress callbacks.
        with patch("researchpilot.hybrid_retrieval.monotonic", return_value=0):
            plain = loop_fixture().agent.run(request)
            f, events = loop_fixture(), []
            f.agent._on_progress = events.append
            observed = f.agent.run(request)
        for result in (plain, observed):
            result.run_stats.elapsed_seconds = 0
        self.assertEqual(plain, observed)
        self.assertGreater(len(events), 10)

    def test_environment_configuration_is_centralized_and_hides_secrets(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "private-openai", "OPENALEX_API_KEY": "private-openalex",
                "RESEARCHPILOT_CACHE_DIR": "custom-cache", "RESEARCHPILOT_RUN_DB": "custom-history.sqlite3"}, clear=True):
            settings = Settings.from_env()
        self.assertTrue(settings.openai_configured)
        self.assertEqual(settings.run_db, Path("custom-history.sqlite3"))
        self.assertEqual(settings.cache_dir, Path("custom-cache"))
        self.assertNotIn("private", repr(settings))
        self.assertEqual(settings.redact("private-openai/private-openalex"), "[REDACTED]/[REDACTED]")
        self.assertNotIn("os.environ", inspect.getsource(ResearchAgent))
