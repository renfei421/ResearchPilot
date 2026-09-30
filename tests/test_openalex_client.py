"""Offline HTTP client tests using small OpenAlex responses and MockTransport."""

import unittest
from unittest.mock import patch

import httpx

from researchpilot.openalex_client import OpenAlexClient
from researchpilot.paper import Paper


class OpenAlexClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requests: list[httpx.Request] = []
        self.work = {
            "id": "https://openalex.org/W123",
            "title": "Matrix completion",
            "doi": "https://doi.org/10.1000/EXAMPLE",
            "publication_year": 2024,
            "cited_by_count": 12,
            "authorships": [{"author": {"display_name": "Ada Example"}}],
            "abstract_inverted_index": {"matrices.": [2], "We": [0], "recover": [1]},
            "open_access": {"oa_url": "https://example.org/paper.pdf"},
        }

    def make_client(
        self,
        payload: object,
        *,
        status_code: int = 200,
        api_key: str | None = None,
    ) -> OpenAlexClient:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if payload is None:
                return httpx.Response(status_code, content=b"null")
            return httpx.Response(status_code, json=payload)

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(http_client.close)
        return OpenAlexClient(api_key=api_key, http_client=http_client)

    def test_successful_search_converts_every_work_to_paper(self) -> None:
        client = self.make_client(
            {
                "meta": {"count": 2},
                "results": [
                    self.work,
                    {"id": "https://openalex.org/W456", "title": "Second paper"},
                ],
            }
        )

        papers = client.search_works(query="matrix completion", per_page=5)

        self.assertIsInstance(papers, list)
        self.assertEqual(len(papers), 2)
        self.assertTrue(all(isinstance(paper, Paper) for paper in papers))
        self.assertEqual(
            papers[0].model_dump(),
            {
                "paper_id": "doi:10.1000/example",
                "title": "Matrix completion",
                "abstract": "We recover matrices.",
                "authors": ["Ada Example"],
                "publication_year": 2024,
                "doi": "10.1000/example",
                "citation_count": 12,
                "source": "openalex",
                "source_id": "W123",
                "open_access_url": "https://example.org/paper.pdf",
                "relevance_score": None,
            },
        )
        self.assertEqual(papers[1].paper_id, "openalex:W456")
        self.assertEqual(papers[1].title, "Second paper")
        self.assertEqual(len(self.requests), 1)

    def test_sends_endpoint_query_per_page_and_exact_select_fields(self) -> None:
        client = self.make_client({"results": []})

        client.search_works(query="  matrix completion & graph/谱  ", per_page=7)

        request = self.requests[0]
        self.assertEqual(request.method, "GET")
        self.assertEqual(request.url.scheme, "https")
        self.assertEqual(request.url.host, "api.openalex.org")
        self.assertEqual(request.url.path, "/works")
        self.assertEqual(
            dict(request.url.params),
            {
                "search": "matrix completion & graph/谱",
                "per_page": "7",
                "select": (
                    "id,title,doi,publication_year,cited_by_count,authorships,"
                    "abstract_inverted_index,open_access"
                ),
            },
        )

    def test_omits_filter_when_both_year_bounds_are_none(self) -> None:
        client = self.make_client({"results": []})

        client.search_works("matrix completion", year_from=None, year_to=None)

        self.assertNotIn("filter", self.requests[0].url.params)
        self.assertEqual(self.requests[0].url.params["search"], "matrix completion")

    def test_filters_by_year_from_only(self) -> None:
        client = self.make_client({"results": []})

        client.search_works("matrix completion", year_from=2015)

        self.assertEqual(
            self.requests[0].url.params["filter"],
            "from_publication_date:2015-01-01",
        )
        self.assertEqual(self.requests[0].url.params["search"], "matrix completion")

    def test_filters_by_year_to_only(self) -> None:
        client = self.make_client({"results": []})

        client.search_works("matrix completion", year_to=2026)

        self.assertEqual(
            self.requests[0].url.params["filter"],
            "to_publication_date:2026-12-31",
        )
        self.assertEqual(self.requests[0].url.params["search"], "matrix completion")

    def test_filters_by_both_year_bounds(self) -> None:
        client = self.make_client({"results": []})

        client.search_works("matrix completion", year_from=2015, year_to=2026)

        self.assertEqual(
            self.requests[0].url.params["filter"],
            "from_publication_date:2015-01-01,to_publication_date:2026-12-31",
        )
        self.assertEqual(self.requests[0].url.params["search"], "matrix completion")

    def test_accepts_same_start_and_end_year(self) -> None:
        client = self.make_client({"results": []})

        client.search_works("matrix completion", year_from=2024, year_to=2024)

        self.assertEqual(
            self.requests[0].url.params["filter"],
            "from_publication_date:2024-01-01,to_publication_date:2024-12-31",
        )
        self.assertEqual(self.requests[0].url.params["search"], "matrix completion")

    def test_rejects_reversed_year_bounds_before_request(self) -> None:
        client = self.make_client({"results": []})

        with self.assertRaisesRegex(
            ValueError, "year_from.*less than or equal.*year_to"
        ):
            client.search_works("matrix completion", year_from=2026, year_to=2015)

        self.assertEqual(self.requests, [])

    def test_rejects_non_integer_year_from_before_request(self) -> None:
        client = self.make_client({"results": []})

        for year in ("2015", 2015.0, [], {}):
            with self.subTest(year=year):
                with self.assertRaisesRegex(ValueError, "year_from.*integer or None"):
                    client.search_works("matrix completion", year_from=year)
                self.assertEqual(self.requests, [])

    def test_rejects_non_integer_year_to_before_request(self) -> None:
        client = self.make_client({"results": []})

        for year in ("2026", 2026.0, [], {}):
            with self.subTest(year=year):
                with self.assertRaisesRegex(ValueError, "year_to.*integer or None"):
                    client.search_works("matrix completion", year_to=year)
                self.assertEqual(self.requests, [])

    def test_rejects_bool_for_either_year_bound_before_request(self) -> None:
        client = self.make_client({"results": []})

        for name in ("year_from", "year_to"):
            for value in (True, False):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(
                        ValueError, f"{name}.*bool is not allowed"
                    ):
                        client.search_works("matrix completion", **{name: value})
                    self.assertEqual(self.requests, [])

    def test_does_not_impose_historical_or_future_year_limits(self) -> None:
        client = self.make_client({"results": []})

        client.search_works("matrix completion", year_from=0, year_to=12000)

        self.assertEqual(
            self.requests[0].url.params["filter"],
            "from_publication_date:0-01-01,to_publication_date:12000-12-31",
        )

    def test_includes_api_key_when_supplied(self) -> None:
        for api_key in ("test-key", ""):
            with self.subTest(api_key=api_key):
                client = self.make_client({"results": []}, api_key=api_key)
                client.search_works("matrix completion")
                self.assertEqual(self.requests[-1].url.params["api_key"], api_key)

    def test_omits_api_key_when_absent(self) -> None:
        client = self.make_client({"results": []})

        client.search_works("matrix completion")

        self.assertNotIn("api_key", self.requests[0].url.params)

    def test_rejects_blank_or_non_string_query_before_request(self) -> None:
        client = self.make_client({"results": []})

        for query in ("", " \t\n ", None, 123):
            with self.subTest(query=query):
                with self.assertRaisesRegex(ValueError, "query.*non-empty string"):
                    client.search_works(query)

        self.assertEqual(self.requests, [])

    def test_rejects_per_page_below_one_before_request(self) -> None:
        client = self.make_client({"results": []})

        with self.assertRaisesRegex(ValueError, "per_page.*between 1 and 100"):
            client.search_works("matrix completion", per_page=0)

        self.assertEqual(self.requests, [])

    def test_rejects_per_page_above_one_hundred_before_request(self) -> None:
        client = self.make_client({"results": []})

        with self.assertRaisesRegex(ValueError, "per_page.*between 1 and 100"):
            client.search_works("matrix completion", per_page=101)

        self.assertEqual(self.requests, [])

    def test_rejects_non_integer_per_page_before_request(self) -> None:
        client = self.make_client({"results": []})

        for per_page in (None, "5", 5.5, True, False):
            with self.subTest(per_page=per_page):
                with self.assertRaisesRegex(ValueError, "per_page.*integer"):
                    client.search_works("matrix completion", per_page=per_page)

        self.assertEqual(self.requests, [])

    def test_accepts_inclusive_per_page_boundaries(self) -> None:
        client = self.make_client({"results": []})

        for per_page in (1, 100):
            with self.subTest(per_page=per_page):
                client.search_works("matrix completion", per_page=per_page)
                self.assertEqual(
                    self.requests[-1].url.params["per_page"], str(per_page)
                )

    def test_empty_results_returns_empty_list(self) -> None:
        client = self.make_client({"meta": {"count": 0}, "results": []})

        self.assertEqual(client.search_works("matrix completion"), [])

    def test_malformed_envelopes_raise_clear_error(self) -> None:
        for payload in (
            {"meta": {}},
            {"results": None},
            {"results": {}},
            {"results": "invalid"},
            {"results": 0},
            None,
            [],
            "invalid",
        ):
            with self.subTest(payload=payload):
                client = self.make_client(payload)
                with self.assertRaisesRegex(ValueError, "'results' list"):
                    client.search_works("matrix completion")

    def test_http_errors_propagate_without_retry(self) -> None:
        for status_code in (401, 429, 500):
            with self.subTest(status_code=status_code):
                self.requests.clear()
                client = self.make_client({"error": "failed"}, status_code=status_code)
                with self.assertRaises(httpx.HTTPStatusError) as caught:
                    client.search_works("matrix completion")
                self.assertEqual(caught.exception.response.status_code, status_code)
                self.assertEqual(len(self.requests), 1)

    def test_uses_finite_timeout(self) -> None:
        client = self.make_client({"results": []})

        client.search_works("matrix completion")

        self.assertEqual(
            self.requests[0].extensions["timeout"],
            {"connect": 10.0, "read": 10.0, "write": 10.0, "pool": 10.0},
        )

    def test_transport_error_propagates_without_retry(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            raise httpx.ReadTimeout("Read timed out", request=request)

        with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
            client = OpenAlexClient(http_client=http_client)
            with self.assertRaises(httpx.ReadTimeout):
                client.search_works("matrix completion")

        self.assertEqual(len(self.requests), 1)

    def test_normalization_errors_propagate(self) -> None:
        client = self.make_client({"results": [{**self.work, "title": None}]})

        with self.assertRaisesRegex(ValueError, "title must be a non-empty string"):
            client.search_works("matrix completion")

    def test_injected_http_client_remains_open(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"results": []})
        )
        with httpx.Client(transport=transport) as http_client:
            client = OpenAlexClient(http_client=http_client)
            client.search_works("matrix completion")
            self.assertFalse(http_client.is_closed)

    def test_default_http_client_is_closed_after_search(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"results": []})
        )
        http_client = httpx.Client(transport=transport)
        self.addCleanup(http_client.close)

        with patch(
            "researchpilot.openalex_client.httpx.Client", return_value=http_client
        ):
            papers = OpenAlexClient().search_works("matrix completion")

        self.assertEqual(papers, [])
        self.assertTrue(http_client.is_closed)
