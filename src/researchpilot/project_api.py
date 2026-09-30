"""Local project resources and bounded continuation through the existing run manager."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from researchpilot.evidence import EvidenceModel, Text
from researchpilot.project_models import ProjectCreate, ProjectUpdate, ProjectContinue, ShortText
from researchpilot.project_store import ProjectArchived, ProjectNotFound
from researchpilot.project_views import project_export_data, project_html, project_markdown
from researchpilot.research_models import ResearchRequest
from researchpilot.claim_check import ClaimCheckRequest
from researchpilot.run_store import RunCreated
from researchpilot.run_manager import QueueFullError


class UserText(EvidenceModel):
    text: ShortText


class UserClaim(UserText):
    claim_type: Text = "hypothesis"
    parent_claim_id: Text | None = None


class ArchiveRecord(EvidenceModel):
    status: Literal["archived"]


def continuation_request(store, project_id, body: ProjectContinue):
    resource = {"question": "questions", "claim": "claims", "gap": "gaps"}[body.target_type]
    target = store.get_record(project_id, resource, body.target_id)
    if store.get(project_id).status != "active" or target.status == "archived":
        raise ProjectArchived("Reactivate the project and choose an active target.")
    text = target.description + "\nSearch focus: " + target.search_focus if body.target_type == "gap" else target.text
    if body.instruction.strip():
        text += "\nFollow-up: " + body.instruction
    fields = dict(project_id=project_id, **{f"target_{body.target_type}_id": body.target_id},
                  year_from=body.year_from, year_to=body.year_to, refresh_search=body.refresh_search)
    if body.mode == "claim_check":
        if body.target_type != "claim":
            raise ValueError("Idea Check continuation requires a claim target.")
        # User instructions are context, not a silent rewrite of the hypothesis.
        return ClaimCheckRequest(claim=target.text, context=body.instruction or None,
                                 field=store.get(project_id).field, **fields)
    return ResearchRequest(question=text, **fields)


def install_project_api(app):
    router = APIRouter(prefix="/projects")

    @app.exception_handler(ProjectNotFound)
    async def missing(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ProjectArchived)
    async def archived(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    def store(request):
        return request.app.state.store.projects

    @router.post("", status_code=201)
    def create(body: ProjectCreate, request: Request):
        return store(request).create(body)

    @router.get("")
    def projects(request: Request, include_archived: bool = True):
        return store(request).list_projects(include_archived=include_archived)

    @router.get("/{project_id}")
    def overview(project_id: str, request: Request):
        return store(request).overview(project_id)

    @router.patch("/{project_id}")
    def update(project_id: str, body: ProjectUpdate, request: Request):
        try:
            return store(request).update(project_id, body)
        except (ProjectNotFound, ProjectArchived):
            raise
        except ValueError:
            raise HTTPException(422, "Invalid project update.") from None

    @router.get("/{project_id}/view", response_class=HTMLResponse, include_in_schema=False)
    def view(project_id: str, request: Request):
        return HTMLResponse(project_html(store(request).overview(project_id)))

    @router.get("/{project_id}/export/json")
    def export_json(project_id: str, request: Request):
        data = project_export_data(store(request).overview(project_id, export=True))
        return JSONResponse(data, headers={"Content-Disposition": f'attachment; filename="project-{data["project"]["project_id"]}.json"'})

    @router.get("/{project_id}/export/markdown")
    def export_markdown(project_id: str, request: Request):
        data = store(request).overview(project_id, export=True)
        return Response(project_markdown(data), media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="project-{data["project"]["project_id"]}.md"'})

    @router.post("/{project_id}/questions", status_code=201)
    def add_question(project_id: str, body: UserText, request: Request):
        return store(request).add_question(project_id, body.text)

    @router.post("/{project_id}/claims", status_code=201)
    def add_claim(project_id: str, body: UserClaim, request: Request):
        try:
            return store(request).add_claim(project_id, **body.model_dump())
        except (ProjectNotFound, ProjectArchived):
            raise
        except ValueError:
            raise HTTPException(422, "Parent claim must belong to this project.") from None

    @router.post("/{project_id}/notes", status_code=201)
    def add_note(project_id: str, body: UserText, request: Request):
        return store(request).add_note(project_id, body.text)

    @router.post("/{project_id}/continue", response_model=RunCreated, status_code=202)
    def continue_research(project_id: str, body: ProjectContinue, request: Request, response: Response):
        try:
            run_request = continuation_request(store(request), project_id, body)
            created = request.app.state.manager.submit(run_request)
        except (ProjectNotFound, ProjectArchived):
            raise
        except ValueError:
            raise HTTPException(422, "Invalid continuation target, instructions or year bounds.") from None
        except QueueFullError as exc:
            raise HTTPException(503, str(exc)) from None
        response.headers["Location"] = f"/research/{created.run_id}"
        return created

    @router.get("/{project_id}/{resource}")
    def records(project_id: str, resource: Literal["questions", "claims", "papers", "evidence", "gaps", "findings", "notes", "history"], request: Request):
        if resource == "history":
            store(request).get(project_id)
            return request.app.state.store.history(limit=100, project_id=project_id)
        return store(request).list_records(project_id, resource)

    @router.patch("/{project_id}/{resource}/{record_id}")
    def archive_record(project_id: str, resource: Literal["questions", "claims", "gaps"], record_id: str,
                       body: ArchiveRecord, request: Request):
        return store(request).archive_record(project_id, resource, record_id)

    app.include_router(router)
