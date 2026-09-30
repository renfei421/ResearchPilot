"""Local single-process HTTP product; research is delegated to ResearchAgent."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from researchpilot.config import Settings
from researchpilot.product_views import markdown_export, result_html
from researchpilot.progress import ResearchProgress
from researchpilot.research_models import ResearchRequest, ResearchResult
from researchpilot.claim_check import ClaimCheckRequest, ClaimCheckResult
from researchpilot.claim_check_views import claim_markdown, claim_html
from researchpilot.run_manager import AgentFactory, QueueFullError, RunManager, configure_logging, log_event, logger
from researchpilot.run_store import RunCreated, RunDetail, RunStore, RunSummary
from researchpilot.project_api import install_project_api


class Health(BaseModel):
    status: str = "ok"
    openai_configured: bool


def create_app(*, settings: Settings | None = None, agent_factory: AgentFactory | None = None, claim_agent_factory=None) -> FastAPI:
    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log_handler = configure_logging(config.cache_dir / "researchpilot.log")
        store = RunStore(config.run_db, redact=config.redact)
        store.recover_interrupted()
        app.state.store = store
        app.state.manager = RunManager(store, config, agent_factory, claim_agent_factory=claim_agent_factory)
        try:
            yield
        finally:
            app.state.manager.close()
            if log_handler is not None:
                logger.removeHandler(log_handler)
                log_handler.close()

    app = FastAPI(title="ResearchPilot", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
    web = Path(__file__).with_name("web")
    app.mount("/assets", StaticFiles(directory=web), name="assets")

    @app.middleware("http")
    async def local_headers(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method in ("POST", "PATCH", "PUT", "DELETE") and origin and origin != f"{request.url.scheme}://{request.url.netloc}":
            return JSONResponse({"detail": "Cross-origin research requests are not allowed."}, status_code=403)
        try:
            response = await call_next(request)
        except Exception as exc:
            # Catch before Starlette's outer error middleware re-raises to Uvicorn,
            # which would otherwise log the raw (possibly secret-bearing) message.
            log_event("http", ResearchProgress(current_stage="failed"), exc)
            response = JSONResponse({"detail": "Local server error. See the server log for details."}, status_code=500)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        if request.url.path == "/" or request.url.path.endswith("/view"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; "
                "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # FastAPI's default includes submitted input. Return field paths and types only.
        issues = [{"field": config.redact(".".join(map(str, error["loc"]))), "type": error["type"]}
                  for error in exc.errors()]
        return JSONResponse({"detail": "Invalid research request. Check fields and year bounds.", "issues": issues}, status_code=422)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(web / "index.html")

    @app.get("/health", response_model=Health)
    def health():
        return Health(openai_configured=config.openai_configured)

    @app.post("/research", response_model=RunCreated, status_code=202)
    def create_research(body: ResearchRequest, response: Response):
        try:
            created = app.state.manager.submit(body)
        except QueueFullError as exc:
            raise HTTPException(503, str(exc)) from None
        response.headers["Location"] = f"/research/{created.run_id}"
        return created

    @app.get("/research", response_model=list[RunSummary])
    def history(limit: int = Query(default=20, ge=1, le=100), offset: int = Query(default=0, ge=0)):
        return app.state.store.history(limit, offset)

    @app.post("/claim-check", response_model=RunCreated, status_code=202)
    def create_claim_check(body: ClaimCheckRequest, response: Response):
        try:
            created = app.state.manager.submit(body)
        except QueueFullError as exc:
            raise HTTPException(503, str(exc)) from None
        response.headers["Location"] = f"/claim-check/{created.run_id}"
        return created

    def find(run_id: str) -> RunDetail:
        run = app.state.store.get(run_id)
        if run is None:
            raise HTTPException(404, "Research run not found.")
        return run

    def completed(run_id: str) -> ResearchResult:
        run = find(run_id)
        if run.status != "completed" or run.result is None:
            raise HTTPException(409, "Research result is not complete.")
        return run.result

    @app.get("/research/{run_id}", response_model=RunDetail)
    def get_research(run_id: str):
        return find(run_id)

    @app.get("/claim-check/{run_id}", response_model=RunDetail)
    def get_claim_check(run_id: str):
        run = find(run_id)
        if run.mode != "claim_check":
            raise HTTPException(404, "Idea Check run not found.")
        return run

    @app.get("/claim-check/{run_id}/view", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/research/{run_id}/view", response_class=HTMLResponse, include_in_schema=False)
    def view_result(run_id: str):
        result = completed(run_id)
        return HTMLResponse(claim_html(result) if isinstance(result, ClaimCheckResult) else result_html(result))

    @app.get("/claim-check/{run_id}/export/json")
    @app.get("/research/{run_id}/export/json")
    def export_json(run_id: str):
        result = completed(run_id)
        return Response(result.model_dump_json(indent=2), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="research-{run_id}.json"'})

    @app.get("/claim-check/{run_id}/export/markdown")
    @app.get("/research/{run_id}/export/markdown")
    def export_markdown(run_id: str):
        result = completed(run_id)
        return Response(claim_markdown(result) if isinstance(result, ClaimCheckResult) else markdown_export(result), media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="research-{run_id}.md"'})

    install_project_api(app)
    return app


app = create_app()
