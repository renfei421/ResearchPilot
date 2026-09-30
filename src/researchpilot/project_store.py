"""Forward-only SQLite project storage in the existing run database."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from researchpilot.project_models import (
    ProjectClaim, ProjectCreate, ProjectEvidence, ProjectFinding, ProjectGap,
    ProjectNote, ProjectPaper, ProjectQuery, ProjectQuestion, ProjectUpdate,
    ResearchProject, now, text_key,
)


RESOURCES = {
    "questions": (ProjectQuestion, "question_id"), "claims": (ProjectClaim, "claim_id"),
    "notes": (ProjectNote, "note_id"), "papers": (ProjectPaper, "paper_id"),
    "evidence": (ProjectEvidence, "evidence_id"), "gaps": (ProjectGap, "gap_id"),
    "findings": (ProjectFinding, "finding_id"), "queries": (ProjectQuery, "query_id"),
}


class ProjectNotFound(ValueError):
    pass


class ProjectArchived(ValueError):
    pass


class ProjectStore:
    def __init__(self, path: Path, *, redact=lambda text: text):
        self.path, self.redact = Path(path), redact
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY, data_json TEXT NOT NULL,
                updated_at TEXT NOT NULL, status TEXT NOT NULL)""")
            for resource in RESOURCES:
                db.execute(f"""CREATE TABLE IF NOT EXISTS project_{resource} (
                    project_id TEXT NOT NULL REFERENCES projects(project_id),
                    record_id TEXT NOT NULL, data_json TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, record_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS project_ingestions (
                project_id TEXT NOT NULL REFERENCES projects(project_id), run_id TEXT NOT NULL,
                ingested_at TEXT NOT NULL, counts_json TEXT NOT NULL,
                PRIMARY KEY(project_id, run_id))""")
            db.execute("CREATE TABLE IF NOT EXISTS project_schema (version INTEGER PRIMARY KEY)")
            db.execute("INSERT OR IGNORE INTO project_schema(version) VALUES (1)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def _dump(self, value):
        return self.redact(value.model_dump_json())

    def _project(self, db, project_id, *, active=False):
        row = db.execute("SELECT data_json FROM projects WHERE project_id=?", (project_id,)).fetchone()
        if row is None:
            raise ProjectNotFound("Project not found.")
        project = ResearchProject.model_validate_json(row[0])
        if active and project.status != "active":
            raise ProjectArchived("Project is archived. Reactivate it before adding research.")
        return project

    def get(self, project_id):
        with self.connect() as db:
            return self._project(db, project_id)

    def create(self, data: ProjectCreate):
        project = ResearchProject(**ProjectCreate.model_validate(data).model_dump())
        with self.connect() as db:
            db.execute("INSERT INTO projects VALUES (?,?,?,?)",
                       (project.project_id, self._dump(project), project.updated_at, project.status))
        return self.get(project.project_id)

    def list_projects(self, *, include_archived=True):
        with self.connect() as db:
            rows = db.execute("SELECT data_json FROM projects " +
                ("" if include_archived else "WHERE status='active' ") +
                "ORDER BY updated_at DESC, project_id").fetchall()
        return [ResearchProject.model_validate_json(row[0]) for row in rows]

    def update(self, project_id, changes: ProjectUpdate):
        changes = ProjectUpdate.model_validate(changes)
        fields = changes.model_dump(exclude_unset=True)
        if any(fields.get(k, "") is None for k in ("title", "description", "status")):
            raise ValueError("Title, description and status cannot be null.")
        with self.connect() as db:
            project = ResearchProject.model_validate(self._project(db, project_id).model_dump() |
                                                     fields | {"updated_at": now()})
            self._save_project(db, project)
        return self.get(project_id)

    def _save_project(self, db, project):
        db.execute("UPDATE projects SET data_json=?,updated_at=?,status=? WHERE project_id=?",
                   (self._dump(project), project.updated_at, project.status, project.project_id))

    def _list(self, db, project_id, resource):
        model, _ = RESOURCES[resource]
        return [model.model_validate_json(row[0]) for row in db.execute(
            f"SELECT data_json FROM project_{resource} WHERE project_id=? ORDER BY rowid", (project_id,))]

    def list_records(self, project_id, resource):
        with self.connect() as db:
            self._project(db, project_id)
            return self._list(db, project_id, resource)

    def _get(self, db, project_id, resource, record_id):
        model, _ = RESOURCES[resource]
        row = db.execute(f"SELECT data_json FROM project_{resource} WHERE project_id=? AND record_id=?",
                         (project_id, record_id)).fetchone()
        return model.model_validate_json(row[0]) if row else None

    def get_record(self, project_id, resource, record_id):
        with self.connect() as db:
            self._project(db, project_id)
            value = self._get(db, project_id, resource, record_id)
            if value is None:
                raise ProjectNotFound("Project record not found.")
            return value

    def _links(self, db, record):
        project_id = record.project_id
        fields = [("related_claim_ids", "claims"), ("claim_ids", "claims"),
                  ("related_question_ids", "questions"), ("evidence_ids", "evidence")]
        for field, resource in fields:
            for key in getattr(record, field, []):
                if self._get(db, project_id, resource, key) is None:
                    raise ValueError(f"Unknown project {resource} reference.")
        if isinstance(record, ProjectClaim) and record.parent_claim_id:
            parent = self._get(db, project_id, "claims", record.parent_claim_id)
            if parent is None or record.parent_claim_id == record.claim_id:
                raise ValueError("Unknown or circular parent claim.")
            ancestors = {record.claim_id}
            while parent is not None:
                if parent.claim_id in ancestors:
                    raise ValueError("Claim ancestry must be acyclic.")
                ancestors.add(parent.claim_id)
                parent = self._get(db, project_id, "claims", parent.parent_claim_id) if parent.parent_claim_id else None
        if isinstance(record, ProjectEvidence):
            paper = self._get(db, project_id, "papers", record.passage.paper_id)
            if paper is None:
                raise ValueError("Project evidence must match a stored paper.")
        if isinstance(record, ProjectGap) and record.status in ("resolved", "partially_resolved") and not record.evidence_ids:
            raise ValueError("Gap resolution requires verified project evidence.")
        if isinstance(record, ProjectClaim) and record.status in ("supported", "partially_supported", "contradicted") and not record.evidence_ids:
            raise ValueError("Claim support requires verified project evidence.")

    def _put(self, db, resource, record):
        model, field = RESOURCES[resource]
        # Revalidate even model_copy/model_construct inputs before durable writes.
        record = model.model_validate(record.model_dump())
        self._links(db, record)
        db.execute(f"""INSERT INTO project_{resource} VALUES (?,?,?,?)
            ON CONFLICT(project_id,record_id) DO UPDATE SET data_json=excluded.data_json,
            updated_at=excluded.updated_at""", (record.project_id, getattr(record, field), self._dump(record), record.updated_at))

    def put(self, resource, record):
        with self.connect() as db:
            project = self._project(db, record.project_id, active=True)
            self._put(db, resource, record)
            self._save_project(db, project.model_copy(update={"updated_at": now()}))
        return self.get_record(record.project_id, resource, getattr(record, RESOURCES[resource][1]))

    def add_question(self, project_id, text):
        return self.put("questions", ProjectQuestion(project_id=project_id, text=text))

    def add_claim(self, project_id, text, claim_type="hypothesis", parent_claim_id=None):
        return self.put("claims", ProjectClaim(project_id=project_id, text=text,
            claim_type=claim_type, parent_claim_id=parent_claim_id))

    def add_note(self, project_id, text):
        return self.put("notes", ProjectNote(project_id=project_id, text=text))

    def archive_record(self, project_id, resource, record_id):
        if resource not in ("questions", "claims", "gaps"):
            raise ValueError("Only questions, claims and gaps can be archived.")
        value = self.get_record(project_id, resource, record_id)
        return self.put(resource, value.model_copy(update={"status": "archived", "updated_at": now()}))

    def bind_request(self, db, request):
        """Validate scoped targets; save user questions/ideas independently of run success."""
        if request.project_id is None:
            return request
        self._project(db, request.project_id, active=True)
        for kind, resource in (("question", "questions"), ("claim", "claims"), ("gap", "gaps")):
            key = getattr(request, f"target_{kind}_id")
            if key:
                target = self._get(db, request.project_id, resource, key)
                if target is None or target.status == "archived":
                    raise ProjectNotFound("Active target not found in this project.")
                return request
        idea = hasattr(request, "claim")
        resource, field = ("claims", "claim_id") if idea else ("questions", "question_id")
        text = request.claim if idea else request.question
        prior = next((v for v in self._list(db, request.project_id, resource)
                      if text_key(v.text) == text_key(text) and v.status != "archived"), None)
        if prior is None:
            prior = RESOURCES[resource][0](project_id=request.project_id, text=text)
            self._put(db, resource, prior)
        return type(request).model_validate(request.model_dump() | {"target_" + field: getattr(prior, field)})

    def overview(self, project_id, *, export=False):
        with self.connect() as db:
            project = self._project(db, project_id)
            resources = {name: [v.model_dump(mode="json") for v in self._list(db, project_id, name)]
                         for name in RESOURCES if export or name != "queries"}
            has_runs = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'").fetchone()
            runs = []
            if has_runs and "project_id" in {r[1] for r in db.execute("PRAGMA table_info(runs)")}:
                for row in db.execute("SELECT * FROM runs WHERE project_id=? ORDER BY created_at DESC, rowid DESC" +
                                      ("" if export else " LIMIT 20"), (project_id,)):
                    record = {k: row[k] for k in ("run_id", "question", "mode", "status", "created_at", "updated_at", "termination_reason", "error")}
                    if export:
                        record.update(request=json.loads(row["request_json"]),
                                      result=json.loads(row["result_json"]) if row["result_json"] else None,
                                      project_context=json.loads(row["project_context_json"]) if row["project_context_json"] else None)
                    runs.append(record)
        return {"schema_version": 1, "project": project.model_dump(mode="json"), **resources, "runs": runs}

    def ingest_completed_run(self, run_id):
        from researchpilot.claim_check import ClaimCheckRequest, ClaimCheckResult
        from researchpilot.research_models import ResearchRequest, ResearchResult
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None or row["status"] != "completed" or not row["result_json"]:
                raise ValueError("Only validated completed runs can be ingested.")
            idea = row["mode"] == "claim_check"
            request = (ClaimCheckRequest if idea else ResearchRequest).model_validate_json(row["request_json"])
            result = (ClaimCheckResult if idea else ResearchResult).model_validate_json(row["result_json"])
            return self.ingest_result(db, run_id, request, result)

    def ingest_result(self, db, run_id, request, result, *, queries=(), document_refs=None):
        from researchpilot.project_ingestion import ingest_result
        return ingest_result(self, db, run_id, request, result, queries=queries, document_refs=document_refs)
