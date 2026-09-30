"""Bounded in-process background execution of the existing ResearchAgent."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
from threading import BoundedSemaphore, Lock
import traceback
from typing import Protocol

import httpx
from openai import AuthenticationError, OpenAI, OpenAIError
from pydantic import ValidationError

from researchpilot.config import Settings
from researchpilot.progress import ResearchProgress
from researchpilot.research_agent import ResearchAgent
from researchpilot.research_models import ResearchRequest, ResearchResult
from researchpilot.claim_check import ClaimCheckRequest, ClaimCheckResult
from researchpilot.claim_novelty_agent import ClaimNoveltyAgent
from researchpilot.run_store import RunCreated, RunStore
from researchpilot.project_context import ProjectContextBuilder, project_usage
from researchpilot.project_reuse import ProjectSearchSession
from researchpilot.retrieval_diagnostics import error_details, is_transient, retrieval_context


logger = logging.getLogger("researchpilot.runs")


def configure_logging(log_path: Path | None = None) -> logging.Handler | None:
    # Configure only our logger. Enabling the HTTPX root logger could disclose
    # an OpenAlex API key embedded in a request URL.
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        return handler
    return None


class Agent(Protocol):
    def run(self, request: ResearchRequest) -> ResearchResult: ...


AgentFactory = Callable[[Callable[[ResearchProgress], None]], Agent]


class QueueFullError(RuntimeError):
    pass


class MissingCredentialsError(RuntimeError):
    pass


def log_event(run_id: str, event: ResearchProgress, error: Exception | None = None) -> None:
    fields = dict(run_id=run_id, stage=event.current_stage, round=event.current_round,
                  elapsed_seconds=event.elapsed_seconds, selected_papers=event.selected_papers,
                  evidence_collected=event.evidence_collected)
    if error is not None:
        # Keep stack locations for debugging, excluding exception text, locals and
        # source lines: HTTP/SDK error strings can contain credentials or prompts.
        fields["error_type"] = type(error).__name__
        fields["traceback"] = [dict(file=f.filename, line=f.lineno, function=f.name)
                               for f in traceback.extract_tb(error.__traceback__)]
        if isinstance(error, httpx.HTTPError):
            fields.update(error_details(error))
        if isinstance(error, ValidationError):
            # No submitted values, model output or exception context. Cross-field
            # rule names make structured-output failures diagnosable offline.
            fields["validation_errors"] = [
                {"field": ".".join(map(str, item["loc"])), "type": item["type"], "rule": item["msg"][:240]}
                for item in error.errors(include_input=False, include_context=False, include_url=False)[:12]
            ]
    _, redact = retrieval_context.get()
    logger.log(logging.ERROR if error is not None else logging.INFO, redact(json.dumps(fields)))


def safe_error(error: Exception, stage: str) -> str:
    if isinstance(error, MissingCredentialsError):
        return "API credentials unavailable. Set OPENAI_API_KEY and restart the local server."
    if isinstance(error, AuthenticationError):
        return "API credentials were rejected. Check OPENAI_API_KEY and restart the local server."
    if stage == "searching":
        if is_transient(error):
            return "Literature service temporarily unavailable. Try again later."
        if isinstance(error, httpx.HTTPStatusError) and error.response.status_code in (400, 422):
            return "Invalid/generated search query or search constraints. See the local server log."
        if isinstance(error, httpx.HTTPStatusError) and error.response.status_code in (401, 403):
            return "Literature service rejected access. Check OPENALEX_API_KEY and service permissions."
        return "Internal research pipeline error during literature retrieval. See the local server log."
    if isinstance(error, (OpenAIError, httpx.HTTPError)):
        return "A research service is unavailable. Try again later."
    if isinstance(error, (ValidationError, ValueError)):
        return "Research output failed validation. No unverified answer was published."
    return "Research could not finish. See the local server log for the error type and stack location."


class RunManager:
    def __init__(self, store: RunStore, settings: Settings, agent_factory: AgentFactory | None = None,
                 *, capacity: int = 16, claim_agent_factory=None):
        self.store, self.settings, self.agent_factory = store, settings, agent_factory
        self.claim_agent_factory = claim_agent_factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="researchpilot")
        self._slots = BoundedSemaphore(capacity)
        self._lock = Lock()
        self._closed = False

    def submit(self, request: ResearchRequest | ClaimCheckRequest) -> RunCreated:
        with self._lock:
            if self._closed or not self._slots.acquire(blocking=False):
                raise QueueFullError("Research queue is full or shutting down. Try again later.")
            created = None
            try:
                created = self.store.create(request)
                request = self.store.get(created.run_id).request
                log_event(created.run_id, ResearchProgress())
                future = self._executor.submit(self._execute, created.run_id, request)
                def finished(task):
                    try:
                        if task.cancelled():
                            self.store.fail(created.run_id, "Server stopped before this queued run started.")
                    finally:
                        self._slots.release()
                future.add_done_callback(finished)
                return created
            except Exception:
                self._slots.release()
                if created:
                    self.store.fail(created.run_id, "Could not schedule this research run.")
                raise

    def _execute(self, run_id: str, request: ResearchRequest | ClaimCheckRequest) -> None:
        context_token = retrieval_context.set((run_id, self.settings.redact))
        latest = ResearchProgress(current_stage="planning", current_round=1)
        def progress(event: ResearchProgress) -> None:
            nonlocal latest
            latest = event
            self.store.progress(run_id, event)
            log_event(run_id, event)
        try:
            self.store.start(run_id)
            progress(latest)
            idea = isinstance(request, ClaimCheckRequest)
            factory = self.claim_agent_factory if idea else self.agent_factory
            context = ProjectContextBuilder(self.store.projects).build(request.project_id, request) if request.project_id else None
            if context is not None:
                self.store.set_project_context(run_id, context)
            session, document_refs = None, {}
            def execute(agent):
                nonlocal session
                services = getattr(agent, "services", agent)
                prior_context = getattr(agent, "project_context", None)
                prior_search = getattr(services, "paper_search", None)
                agent.project_context = context
                if context is not None:
                    if hasattr(services, "paper_search"):
                        session = ProjectSearchSession(services.paper_search, self.store.projects, request, run_id)
                        services.paper_search = session
                try:
                    result = agent.run(request)
                    if context is not None:
                        from researchpilot.document_acquisition import DocumentAcquirer
                        fetcher = getattr(services, "document_fetcher", None)
                        if isinstance(fetcher, DocumentAcquirer):
                            for source in result.sources if idea else result.papers:
                                path = fetcher.cached_pdf(source.paper)
                                if path is not None:
                                    document_refs[source.paper.paper_id] = str(path.resolve())
                    return result
                finally:
                    agent.project_context = prior_context
                    if session is not None:
                        services.paper_search = prior_search
            if factory is not None:
                result = execute(factory(progress))
            else:
                if not self.settings.openai_configured:
                    raise MissingCredentialsError()
                with OpenAI(api_key=self.settings.openai_api_key.get_secret_value(),
                            max_retries=0, timeout=60.0) as client:
                    agent_type = ClaimNoveltyAgent if idea else ResearchAgent
                    result = execute(agent_type(settings=self.settings, openai_client=client,
                                           on_progress=progress))
            result = (ClaimCheckResult if idea else ResearchResult).model_validate(result)
            if (result.request != request if idea else result.question != request.question):
                raise ValueError("Result question does not match the run request.")
            if context is not None:
                result.project_usage = project_usage(context, result.sources if idea else result.papers, result.evidence,
                    queries_reused=session.reused if session else 0)
            self.store.complete(run_id, result, queries=list(session.records.values()) if session else (), document_refs=document_refs)
            if context is not None:
                usage = self.store.get(run_id).result.project_usage
                logger.info(self.settings.redact(json.dumps(dict(event="project_ingested", run_id=run_id,
                    project_id=context.project_id, context_records=len(usage.context_record_ids),
                    reused_papers=len(usage.reused_paper_ids),
                    reused_evidence=sum(v == "reused_project_evidence" for v in usage.evidence_origins.values()),
                    new_papers=usage.new_papers, new_evidence=usage.new_evidence,
                    gaps_carried_in=usage.gaps_carried_in, gaps_resolved=usage.gaps_resolved,
                    queries_reused=usage.queries_reused))))
            latest = latest.model_copy(update={"current_stage": "completed",
                "current_round": result.run_stats.search_rounds,
                "selected_papers": len(result.sources if idea else result.papers), "evidence_collected": len(result.evidence),
                "elapsed_seconds": result.run_stats.elapsed_seconds})
            log_event(run_id, latest)
        except Exception as exc:
            failed = latest.model_copy(update={"current_stage": "failed"})
            log_event(run_id, failed, exc)
            self.store.fail(run_id, safe_error(exc, latest.current_stage))
        finally:
            retrieval_context.reset(context_token)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        # Finish the active bounded run; cancel queued work on a graceful stop.
        self._executor.shutdown(wait=True, cancel_futures=True)
