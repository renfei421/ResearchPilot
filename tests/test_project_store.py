"""Persistence and atomic ingestion contracts; all synthetic and offline."""

import json
import sqlite3
from unittest.mock import patch

from phase5_helpers import OfflineTest
from phase12_helpers import project, finish, research_fixture, QUESTION
from claim_check_helpers import fixture as idea_fixture, CLAIM
from researchpilot.project_models import ProjectCreate, ProjectUpdate, ProjectGap, ProjectClaim, ProjectEvidence
from researchpilot.project_store import ProjectNotFound, ProjectArchived
from researchpilot.research_models import ResearchRequest
from researchpilot.claim_check import ClaimCheckRequest
from researchpilot.run_store import RunStore


class ProjectStoreTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.root = self.temporary_directory()
        self.store = RunStore(self.root / "runs.sqlite3")
        self.ps = self.store.projects
        self.p = project(self.store)

    def rows(self, resource):
        return self.ps.list_records(self.p.project_id, resource)

    def run_first(self, **kw):
        return finish(self.store, self.p.project_id, **kw)

    def test_create(self):
        self.assertEqual(self.p.title, "Low-Rank Recovery")
        self.assertEqual(self.p.status, "active")
        self.assertTrue(self.p.created_at)

    def test_list(self):
        self.assertEqual([p.project_id for p in self.ps.list_projects()], [self.p.project_id])

    def test_reopen(self):
        self.assertEqual(RunStore(self.store.path).projects.get(self.p.project_id), self.p)

    def test_update(self):
        updated = self.ps.update(self.p.project_id, ProjectUpdate(title="Updated", description="Edited", field="Math"))
        self.assertEqual((updated.title, updated.description, updated.field), ("Updated", "Edited", "Math"))

    def test_archive_retains_metadata(self):
        self.ps.update(self.p.project_id, ProjectUpdate(status="archived"))
        self.assertEqual(self.ps.get(self.p.project_id).status, "archived")
        self.assertEqual(self.ps.list_projects(include_archived=False), [])
        with self.assertRaises(ProjectArchived):
            self.ps.add_note(self.p.project_id, "No new work")

    def test_reactivate(self):
        self.ps.update(self.p.project_id, ProjectUpdate(status="archived"))
        self.ps.update(self.p.project_id, ProjectUpdate(status="active"))
        self.assertEqual(self.ps.add_note(self.p.project_id, "Note").text, "Note")

    def test_add_question(self):
        self.assertEqual(self.ps.add_question(self.p.project_id, QUESTION).status, "open")

    def test_add_claim_and_parent(self):
        a = self.ps.add_claim(self.p.project_id, "Spectral hypothesis")
        b = self.ps.add_claim(self.p.project_id, "Refinement", parent_claim_id=a.claim_id)
        self.assertEqual((b.parent_claim_id, b.status), (a.claim_id, "proposed"))

    def test_cross_project_parent_rejected(self):
        other = project(self.store)
        a = self.ps.add_claim(other.project_id, "Foreign")
        with self.assertRaises(ValueError):
            self.ps.add_claim(self.p.project_id, "Child", parent_claim_id=a.claim_id)

    def test_circular_parent_rejected(self):
        a = self.ps.add_claim(self.p.project_id, "Parent")
        b = self.ps.add_claim(self.p.project_id, "Child", parent_claim_id=a.claim_id)
        with self.assertRaises(ValueError):
            self.ps.put("claims", a.model_copy(update={"parent_claim_id": b.claim_id}))

    def test_notes_never_become_evidence(self):
        self.ps.add_note(self.p.project_id, "I think the theorem holds.")
        self.assertEqual(len(self.rows("notes")), 1)
        self.assertEqual(self.rows("evidence"), [])
        self.assertEqual(self.rows("findings"), [])

    def test_archive_question(self):
        q = self.ps.add_question(self.p.project_id, QUESTION)
        self.assertEqual(self.ps.archive_record(self.p.project_id, "questions", q.question_id).status, "archived")

    def test_archive_claim(self):
        c = self.ps.add_claim(self.p.project_id, "Spectral claim")
        self.assertEqual(self.ps.archive_record(self.p.project_id, "claims", c.claim_id).status, "archived")

    def test_standalone_unchanged(self):
        r = finish(self.store, None).run
        self.assertEqual(r.status, "completed")
        self.assertIsNone(r.project_id)
        self.assertEqual(self.rows("papers"), [])

    def test_migration_preserves_old_research_and_idea_rows(self):
        old = self.root / "legacy.sqlite3"
        with sqlite3.connect(old) as db:
            db.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, question TEXT, request_json TEXT, created_at TEXT, updated_at TEXT, status TEXT, progress_json TEXT, termination_reason TEXT, result_json TEXT, error TEXT, mode TEXT)")
            for mode, request in [("research", ResearchRequest(question=QUESTION)), ("claim_check", ClaimCheckRequest(claim=CLAIM))]:
                raw = request.model_dump(exclude={"project_id", "target_claim_id", "target_question_id", "target_gap_id", "refresh_search"})
                db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?)", (mode, QUESTION, json.dumps(raw), "2026", "2026", "failed", "{}", None, None, "historical", mode))
        db.close()
        upgraded = RunStore(old)
        self.assertEqual(len(upgraded.history()), 2)
        self.assertIsInstance(upgraded.get("claim_check").request, ClaimCheckRequest)
        self.assertIsNone(upgraded.get("research").project_id)
        self.assertEqual(RunStore(old).get("research").error, "historical")

    def test_complete_ingests_paper(self):
        r = self.run_first().run
        self.assertEqual(len(self.rows("papers")), 1)
        self.assertEqual(self.rows("papers")[0].first_seen_run_id, r.run_id)

    def test_verified_evidence_provenance(self):
        r = self.run_first().run
        e = self.rows("evidence")[0]
        self.assertEqual(e.passage, r.result.evidence[0])
        self.assertEqual(e.originating_run_id, r.run_id)
        self.assertEqual(e.attestations[0].kind, "atomic_verification")

    def test_unsupported_not_trusted(self):
        self.run_first(fixture=research_fixture(supported=False))
        self.assertEqual(self.rows("findings"), [])
        self.assertEqual(self.rows("evidence"), [])
        self.assertEqual(self.rows("claims")[0].status, "unresolved")

    def test_question_never_auto_answered(self):
        self.run_first()
        self.assertEqual(self.rows("questions")[0].status, "partially_answered")

    def test_paper_dedup_first_seen_preserved(self):
        first = self.run_first().run
        self.run_first()
        self.assertEqual(len(self.rows("papers")), 1)
        self.assertEqual(self.rows("papers")[0].first_seen_run_id, first.run_id)

    def test_evidence_dedup_retains_attestations(self):
        self.run_first()
        self.run_first()
        self.assertEqual(len(self.rows("evidence")), 1)
        self.assertEqual(len(self.rows("evidence")[0].attestations), 2)

    def test_gap_persists_across_restart(self):
        self.run_first()
        self.assertEqual(RunStore(self.store.path).projects.list_records(self.p.project_id, "gaps"), self.rows("gaps"))

    def test_absent_gap_not_resolved(self):
        self.run_first()
        f = research_fixture(second=True)
        # Unrelated request cannot receive or implicitly clear the previous gap.
        r = self.store.create(ResearchRequest(question="Distant topic", project_id=self.p.project_id, max_search_rounds=1))
        self.store.start(r.run_id)
        result = f.agent.run(self.store.get(r.run_id).request)
        self.store.complete(r.run_id, result)
        self.assertEqual(self.rows("gaps")[0].status, "unresolved")

    def test_duplicate_ingestion_idempotent(self):
        r = self.run_first().run
        before = self.ps.overview(self.p.project_id, export=True)
        self.ps.ingest_completed_run(r.run_id)
        self.assertEqual(self.ps.overview(self.p.project_id, export=True), before)

    def test_incomplete_run_rejected(self):
        r = self.store.create(ResearchRequest(question=QUESTION, project_id=self.p.project_id))
        with self.assertRaises(ValueError):
            self.ps.ingest_completed_run(r.run_id)
        self.assertEqual(self.rows("papers"), [])

    def test_corrupt_result_rolls_back_completion(self):
        f = research_fixture()
        r = self.store.create(ResearchRequest(question=QUESTION, project_id=self.p.project_id, max_search_rounds=1))
        self.store.start(r.run_id)
        result = f.agent.run(self.store.get(r.run_id).request)
        result.claims[0].claim.evidence_ids.append("missing")
        with self.assertRaises(ValueError):
            self.store.complete(r.run_id, result)
        self.assertEqual(self.store.get(r.run_id).status, "running")
        self.assertEqual(self.rows("papers"), [])

    def test_mid_ingestion_error_rolls_back(self):
        f = research_fixture()
        r = self.store.create(ResearchRequest(question=QUESTION, project_id=self.p.project_id, max_search_rounds=1))
        self.store.start(r.run_id)
        result = f.agent.run(self.store.get(r.run_id).request)
        original = self.ps._put
        def fail(db, resource, record):
            if resource == "findings":
                raise RuntimeError("simulated disk failure")
            return original(db, resource, record)
        with patch.object(self.ps, "_put", side_effect=fail), self.assertRaises(RuntimeError):
            self.store.complete(r.run_id, result)
        self.assertEqual(self.rows("papers"), [])
        self.assertEqual(self.rows("evidence"), [])

    def test_idea_ingestion_retains_scoped_relations_not_support(self):
        r = self.store.create(ClaimCheckRequest(claim=CLAIM, project_id=self.p.project_id, max_search_rounds=1))
        self.store.start(r.run_id)
        result = idea_fixture().agent.run(self.store.get(r.run_id).request)
        self.store.complete(r.run_id, result)
        self.assertTrue(self.rows("papers"))
        self.assertTrue(self.rows("evidence"))
        self.assertTrue(self.rows("findings"))
        self.assertTrue(all(c.status == "proposed" for c in self.rows("claims")))
        self.assertTrue(all(f.status == "relation_assessed" and "Scope:" in f.text for f in self.rows("findings")))

    def test_invalid_project_rejected_before_run_created(self):
        with self.assertRaises(ProjectNotFound):
            self.store.create(ResearchRequest(question=QUESTION, project_id="missing"))
        self.assertEqual(self.store.history(), [])

    def test_foreign_target_rejected(self):
        other = project(self.store)
        q = self.ps.add_question(other.project_id, QUESTION)
        with self.assertRaises(ProjectNotFound):
            self.store.create(ResearchRequest(question=QUESTION, project_id=self.p.project_id, target_question_id=q.question_id))

    def test_support_without_evidence_rejected(self):
        with self.assertRaises(ValueError):
            self.ps.put("claims", ProjectClaim(project_id=self.p.project_id, text="Unsupported", status="supported"))

    def test_resolved_gap_without_evidence_rejected(self):
        with self.assertRaises(ValueError):
            self.ps.put("gaps", ProjectGap(project_id=self.p.project_id, gap_id="g", description="Gap", search_focus="Gap",
                status="resolved", first_seen_run_id="r", last_updated_run_id="r"))

    def test_full_fake_lifecycle_resolves_gap_and_survives_restart(self):
        question = self.ps.add_question(self.p.project_id, QUESTION)
        first = self.run_first(target_question_id=question.question_id)
        self.assertEqual((len(self.rows("papers")), len(self.rows("evidence")), len(self.rows("findings")), len(self.rows("gaps"))), (1, 1, 1, 1))
        gap = self.rows("gaps")[0]
        second = self.run_first(fixture=research_fixture(second=True), target_gap_id=gap.gap_id, refresh_search=True)
        self.assertEqual(len(second.context.papers), 1)
        self.assertEqual(len(second.context.evidence), 1)
        self.assertEqual(second.context.gaps[0].gap_id, gap.gap_id)
        self.assertEqual((len(self.rows("papers")), len(self.rows("evidence")), len(self.rows("findings"))), (2, 2, 2))
        self.assertEqual(self.rows("gaps")[0].status, "resolved")
        self.assertTrue(self.rows("gaps")[0].evidence_ids)
        self.assertEqual(second.run.result.project_usage.gaps_resolved, 1)
        before = self.ps.overview(self.p.project_id, export=True)
        self.assertEqual(RunStore(self.store.path).projects.overview(self.p.project_id, export=True), before)

    def test_redacts_configured_secrets_from_persistent_data(self):
        store = RunStore(self.root / "redacted.db", redact=lambda text: text.replace("secret-token", "[redacted]"))
        p = store.projects.create(ProjectCreate(title="secret-token"))
        store.projects.add_note(p.project_id, "secret-token")
        self.assertNotIn("secret-token", json.dumps(store.projects.overview(p.project_id, export=True)))
