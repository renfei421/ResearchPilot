"""Minimal synchronous HTTP search for OpenAlex works."""

from contextlib import nullcontext
from time import sleep

import httpx

from researchpilot.openalex import openalex_work_to_paper
from researchpilot.paper import Paper
from researchpilot.retrieval_diagnostics import is_transient, log_retrieval


_BASE_URL = "https://api.openalex.org"
_SELECT_FIELDS = (
    "id,title,doi,publication_year,cited_by_count,authorships,"
    "abstract_inverted_index,open_access"
)


class OpenAlexClient:
    """Search OpenAlex and return normalized papers.

    By default, each search creates and closes its own HTTP client. An injected
    http_client remains owned by the caller and can use httpx.MockTransport.
    Retries are opt-in (0 by default); the production agent uses at most 2.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        http_client: httpx.Client | None = None,
        max_retries: int = 0,
    ) -> None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 2:
            raise ValueError("max_retries must be an integer between 0 and 2.")
        self._api_key = api_key
        self._http_client = http_client
        self._max_retries = max_retries

    def search_works(
        self,
        query: str,
        per_page: int = 5,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[Paper]:
        """Fetch one results page, using a 10-second timeout for each I/O phase.

        Optional year bounds use inclusive publication-date filters without
        changing the text query. No filter is sent when both bounds are None.
        Invalid inputs or response envelopes raise ValueError. HTTP, transport,
        JSON decoding, and paper normalization errors propagate to the caller.
        Opt-in retries cover transient I/O, 429 and 5xx only, with 0.5/1s delays.
        Retry-After is respected when numeric and <=5s; otherwise fail promptly.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if (
            isinstance(per_page, bool)
            or not isinstance(per_page, int)
            or not 1 <= per_page <= 100
        ):
            raise ValueError("per_page must be an integer between 1 and 100.")

        for name, year in (("year_from", year_from), ("year_to", year_to)):
            if year is not None and (
                isinstance(year, bool) or not isinstance(year, int)
            ):
                raise ValueError(
                    f"{name} must be an integer or None; bool is not allowed."
                )
        if year_from is not None and year_to is not None and year_from > year_to:
            raise ValueError("year_from must be less than or equal to year_to.")

        params: dict[str, str | int] = {
            "search": query.strip(),
            "per_page": per_page,
            "select": _SELECT_FIELDS,
        }
        filters = []
        if year_from is not None:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to is not None:
            filters.append(f"to_publication_date:{year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)

        if self._api_key is not None:
            params["api_key"] = self._api_key

        client_context = (
            nullcontext(self._http_client)
            if self._http_client is not None
            else httpx.Client()
        )
        with client_context as client:
            for attempt in range(self._max_retries + 1):
                try:
                    response = client.get(f"{_BASE_URL}/works", params=params, timeout=10.0)
                    response.raise_for_status()
                    break
                except httpx.HTTPError as exc:
                    if attempt >= self._max_retries or not is_transient(exc):
                        raise
                    delay = 0.5 * 2 ** attempt
                    if isinstance(exc, httpx.HTTPStatusError):
                        retry_after = exc.response.headers.get("Retry-After")
                        if retry_after is not None:
                            try:
                                delay = max(delay, float(retry_after))
                            except ValueError:
                                # Do not retry early when a date/unknown delay was requested.
                                raise exc
                            if not 0 <= delay <= 5:
                                raise
                    log_retrieval("openalex_retry", query=query.strip(), year_from=year_from,
                                  year_to=year_to, attempt=attempt + 1, delay_seconds=delay, error=exc)
                    sleep(delay)
            payload = response.json()

        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise ValueError("OpenAlex response must contain a 'results' list.")

        return [openalex_work_to_paper(work) for work in payload["results"]]
