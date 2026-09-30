"""Small, secret-safe diagnostics and failure classification for literature I/O."""

from contextvars import ContextVar
import json
import logging
import re
import traceback

import httpx


# Bound by the HTTP worker; direct CLI use still gets query diagnostics.
retrieval_context: ContextVar[tuple] = ContextVar("retrieval_context", default=(None, lambda s: s))


def is_transient(error: Exception) -> bool:
    """Only transport availability failures, rate limits and server errors."""
    return isinstance(error, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)) or (
        isinstance(error, httpx.HTTPStatusError)
        and (error.response.status_code == 429 or 500 <= error.response.status_code <= 599)
    )


def error_details(error: Exception) -> dict:
    details = {"error_type": type(error).__name__, "traceback": [
        dict(file=f.filename, line=f.lineno, function=f.name)
        for f in traceback.extract_tb(error.__traceback__)
    ]}
    if isinstance(error, httpx.HTTPError):
        # Never log request headers, bodies, URL query strings or provider bodies.
        message = str(error)
        try:
            request = error.request
        except RuntimeError:
            request = None
        if request is not None:
            for name in ("api_key", "key", "token"):
                for value in request.url.params.get_list(name):
                    if value:
                        message = message.replace(value, "[REDACTED]")
        message = re.sub(r"https?://[^\s'\"]+", "[request URL omitted]", message)
        details["error_message"] = message
        if isinstance(error, httpx.HTTPStatusError):
            details["http_status"] = error.response.status_code
            retry_after = error.response.headers.get("Retry-After")
            if retry_after is not None:
                # Keep only a numeric delay, never arbitrary header text.
                try:
                    seconds = int(retry_after)
                except ValueError:
                    details["retry_after"] = "date_or_unrecognized"
                else:
                    details["retry_after_seconds"] = seconds
    # Generic exception messages can contain prompts. Keep their stack/type only.
    causes, cause = [], error.__cause__
    while cause is not None and len(causes) < 8:
        causes.append({"error_type": type(cause).__name__, "errno": getattr(cause, "errno", None)})
        cause = cause.__cause__
    if causes:
        details["causes"] = causes
    return details


def log_retrieval(event: str, *, error: Exception | None = None, **fields) -> None:
    run_id, redact = retrieval_context.get()
    payload = dict(run_id=run_id, event=event, **fields)
    if error is not None:
        payload.update(error_details(error))
    logging.getLogger("researchpilot.runs").log(
        logging.WARNING if error is not None else logging.INFO,
        redact(json.dumps(payload)),
    )
