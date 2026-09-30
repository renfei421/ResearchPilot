"""Transactional local run history. Each operation owns its SQLite connection."""

from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from researchpilot.progress import ResearchProgress
from researchpilot.research_models import ResearchRequest, ResearchResult, TerminationReason
from researchpilot.claim_check import ClaimCheckRequest, ClaimCheckResult
from researchpilot.project_store import ProjectStore
from researchpilot.project_context import ProjectContext


RunStatus = Literal["queued", "running", "completed", "failed"]


class RunSummary(ResearchProgress):
    project_id: str | None = None
    run_id: str
    question: str
    status: RunStatus
    created_at: str
    updated_at: str
    mode: Literal["research", "claim_check"] = "research"
    termination_reason: str | None = None
    error: str | None = None


class RunDetail(RunSummary):
    project_context: ProjectContext | None = None
    request: ResearchRequest | ClaimCheckRequest
    result: ResearchResult | ClaimCheckResult | None = None


class RunCreated(BaseModel):
    run_id: str
    status: RunStatus


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, path: Path, *, redact: Callable[[str], str] = lambda text: text):
        self.path = Path(path)
        self.redact = redact
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.projects = ProjectStore(self.path, redact=redact)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, question TEXT NOT NULL, request_json TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
                progress_json TEXT NOT NULL, termination_reason TEXT,
                result_json TEXT, error TEXT
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS runs_created ON runs(created_at DESC)")
            # Backward-compatible migration in the same transaction/database.
            if "mode" not in {row[1] for row in db.execute("PRAGMA table_info(runs)")}:
                db.execute("ALTER TABLE runs ADD COLUMN mode TEXT NOT NULL DEFAULT 'research'")
            columns = {row[1] for row in db.execute("PRAGMA table_info(runs)")}
            if "project_id" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN project_id TEXT REFERENCES projects(project_id)")
            if "project_context_json" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN project_context_json TEXT")
            db.execute("CREATE INDEX IF NOT EXISTS runs_project ON runs(project_id,created_at DESC)")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def _dump(self, model: BaseModel) -> str:
        return self.redact(model.model_dump_json())

    def create(self, request: ResearchRequest | ClaimCheckRequest) -> RunCreated:
        run_id, now = uuid4().hex, _now()
        idea = isinstance(request, ClaimCheckRequest)
        with self._connect() as db:
            request = self.projects.bind_request(db, request)
            db.execute("""INSERT INTO runs (run_id, question, request_json, created_at, updated_at,
                       status, progress_json, mode, project_id) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)""",
                       (run_id, self.redact(request.claim if idea else request.question), self._dump(request), now, now,
                        self._dump(ResearchProgress()), "claim_check" if idea else "research", request.project_id))
        return RunCreated(run_id=run_id, status="queued")

    def start(self, run_id: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE runs SET status='running', updated_at=? WHERE run_id=? AND status='queued'",
                       (_now(), run_id))

    def progress(self, run_id: str, event: ResearchProgress) -> None:
        with self._connect() as db:
            db.execute("UPDATE runs SET progress_json=?, updated_at=? WHERE run_id=? AND status='running'",
                       (self._dump(event), _now(), run_id))

    def set_project_context(self, run_id: str, context: ProjectContext) -> None:
        with self._connect() as db:
            db.execute("UPDATE runs SET project_context_json=? WHERE run_id=? AND project_id=?",
                       (self._dump(context), run_id, context.project_id))

    def complete(self, run_id: str, result: ResearchResult | ClaimCheckResult, *, queries=(), document_refs=None) -> None:
        progress = ResearchProgress(current_stage="completed", current_round=result.run_stats.search_rounds,
            selected_papers=len(result.sources if isinstance(result, ClaimCheckResult) else result.papers), evidence_collected=len(result.evidence),
            elapsed_seconds=result.run_stats.elapsed_seconds, warnings=result.warnings)
        # Result and completed status are committed together: never a dangling artifact.
        with self._connect() as db:
            row = db.execute("SELECT request_json,mode,project_id FROM runs WHERE run_id=? AND status='running'", (run_id,)).fetchone()
            if row is None:
                return
            if row["project_id"]:
                request = (ClaimCheckRequest if row["mode"] == "claim_check" else ResearchRequest).model_validate_json(row["request_json"])
                counts = self.projects.ingest_result(db, run_id, request, result, queries=queries, document_refs=document_refs)
                result = result.model_copy(update={"project_usage": result.project_usage.model_copy(update=counts | {"project_id": request.project_id})})
            db.execute("""UPDATE runs SET status='completed', result_json=?, progress_json=?,
                       termination_reason=?, updated_at=?, error=NULL WHERE run_id=? AND status='running'""",
                       (self._dump(result), self._dump(progress), result.termination_reason, _now(), run_id))

    def fail(self, run_id: str, message: str) -> None:
        with self._connect() as db:
            row = db.execute("SELECT progress_json FROM runs WHERE run_id=? AND status IN ('queued','running')",
                             (run_id,)).fetchone()
            if row is None:
                return
            event = ResearchProgress.model_validate_json(row[0])
            event.current_stage = "failed"
            db.execute("""UPDATE runs SET status='failed', error=?, progress_json=?, updated_at=?
                       WHERE run_id=? AND status IN ('queued','running')""",
                       (self.redact(message), self._dump(event), _now(), run_id))

    def recover_interrupted(self) -> None:
        # This local MVP has a single server process; interrupted work is not resumed.
        with self._connect() as db:
            ids = [row[0] for row in db.execute("SELECT run_id FROM runs WHERE status IN ('queued','running')")]
        for run_id in ids:
            self.fail(run_id, "Server restarted before this run finished. Submit the question again.")

    @staticmethod
    def _summary(row) -> dict:
        return {key: row[key] for key in ("run_id", "question", "status", "created_at", "updated_at",
                                          "termination_reason", "error", "mode", "project_id")} | json.loads(row["progress_json"])

    def get(self, run_id: str) -> RunDetail | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        return RunDetail(**self._summary(row), request=json.loads(row["request_json"]),
                         project_context=json.loads(row["project_context_json"]) if row["project_context_json"] else None,
                         result=json.loads(row["result_json"]) if row["result_json"] else None)

    def history(self, limit: int = 20, offset: int = 0, project_id: str | None = None) -> list[RunSummary]:
        with self._connect() as db:
            rows = db.execute("""SELECT run_id, question, status, created_at, updated_at,
                termination_reason, error, progress_json, mode, project_id FROM runs """ +
                ("WHERE project_id=? " if project_id else "") +
                "ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
                ((project_id,) if project_id else ()) + (limit, offset)).fetchall()
        return [RunSummary(**self._summary(row)) for row in rows]
