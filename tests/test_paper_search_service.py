"""Candidate-model and aggregation tests using an in-memory search client."""

import unittest

from pydantic import ValidationError

from researchpilot.paper import Paper
from researchpilot.paper_candidate import PaperCandidate, QueryHit, SearchQuery
from researchpilot.paper_search_service import PaperSearchService


def make_paper(source_id: str, title: str = "Example paper") -> Paper:
    return Paper(
        paper_id=f"openalex:{source_id}",
        title=title,
        abstract=None,
        authors=[],
        publication_year=None,
        doi=None,
        citation_count=0,
        source="openalex",
        source_id=source_id,
        open_access_url=None,
    )


class FakeOpenAlexClient:
    def __init__(self, responses: dict[str, list[Paper] | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def search_works(
        self,
        query: str,
        per_page: int = 5,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[Paper]:
        self.calls.append(
            {
                "query": query,
                "per_page": per_page,
                "year_from": year_from,
                "year_to": year_to,
            }
        )
        response = self.responses[query]
        if isinstance(response, Exception):
            raise response
        return response


class CandidateModelTests(unittest.TestCase):
    def test_query_trims_edges_and_preserves_search_text(self) -> None:
        query = SearchQuery(query_id=" q1 ", text='  Matrix  completion & "谱"  ')

        self.assertEqual(query.query_id, "q1")
        self.assertEqual(query.text, 'Matrix  completion & "谱"')

    def test_blank_or_non_string_query_id_is_rejected(self) -> None:
        for value in ("", " \t\n ", None, 123, True, b"q1"):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    SearchQuery(query_id=value, text="matrix completion")

    def test_blank_or_non_string_query_text_is_rejected(self) -> None:
        for value in ("", " \t\n ", None, 123, True, b"matrix completion"):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    SearchQuery(query_id="q1", text=value)

    def test_hit_rank_must_be_positive(self) -> None:
        for rank in (-1, 0):
            with self.subTest(rank=rank):
                with self.assertRaises(ValidationError):
                    QueryHit(query_id="q1", query_text="matrix completion", rank=rank)

        hit = QueryHit(query_id="q1", query_text="matrix completion", rank=1)
        self.assertEqual(hit.rank, 1)


class PaperSearchServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = make_paper("W1")
        self.b = make_paper("W2")
        self.c = make_paper("W3")
        self.d = make_paper("W4")
        self.q1 = SearchQuery(query_id="q1", text="matrix completion")
        self.q2 = SearchQuery(query_id="q2", text="spectral expansion")
        self.q3 = SearchQuery(query_id="q3", text="deterministic sampling")

    def test_single_query_candidates_have_one_based_ranks(self) -> None:
        client = FakeOpenAlexClient({self.q1.text: [self.a, self.b]})

        candidates = PaperSearchService(client).search_candidates([self.q1])

        self.assertEqual(
            candidates,
            [
                PaperCandidate(
                    paper=self.a,
                    hits=[QueryHit(query_id="q1", query_text=self.q1.text, rank=1)],
                ),
                PaperCandidate(
                    paper=self.b,
                    hits=[QueryHit(query_id="q1", query_text=self.q1.text, rank=2)],
                ),
            ],
        )
        self.assertEqual(
            client.calls,
            [
                {
                    "query": self.q1.text,
                    "per_page": 10,
                    "year_from": None,
                    "year_to": None,
                }
            ],
        )

    def test_non_overlapping_queries_preserve_query_and_result_order(self) -> None:
        client = FakeOpenAlexClient(
            {self.q1.text: [self.a, self.b], self.q2.text: [self.c]}
        )

        candidates = PaperSearchService(client).search_candidates([self.q1, self.q2])

        self.assertEqual([item.paper for item in candidates], [self.a, self.b, self.c])
        self.assertEqual([item.hits[0].rank for item in candidates], [1, 2, 1])
        self.assertEqual(
            [call["query"] for call in client.calls], [self.q1.text, self.q2.text]
        )

    def test_shared_paper_merges_and_retains_every_query_hit(self) -> None:
        client = FakeOpenAlexClient(
            {
                self.q1.text: [self.a],
                self.q2.text: [self.b, self.a],
                self.q3.text: [self.c, self.b, self.a],
            }
        )

        candidates = PaperSearchService(client).search_candidates(
            [self.q1, self.q2, self.q3]
        )

        self.assertEqual(len(candidates), 3)
        self.assertEqual(
            candidates[0].hits,
            [
                QueryHit(query_id="q1", query_text=self.q1.text, rank=1),
                QueryHit(query_id="q2", query_text=self.q2.text, rank=2),
                QueryHit(query_id="q3", query_text=self.q3.text, rank=3),
            ],
        )

    def test_duplicates_within_query_keep_best_rank_without_renumbering(self) -> None:
        duplicate = make_paper("W1", title="Later metadata")
        client = FakeOpenAlexClient({self.q1.text: [self.a, duplicate, self.b, self.a]})

        candidates = PaperSearchService(client).search_candidates([self.q1])

        self.assertEqual(len(candidates), 2)
        self.assertIs(candidates[0].paper, self.a)
        self.assertEqual(
            candidates[0].hits,
            [QueryHit(query_id="q1", query_text=self.q1.text, rank=1)],
        )
        self.assertEqual(candidates[1].hits[0].rank, 3)

    def test_retains_first_seen_paper_object_across_queries(self) -> None:
        later = make_paper("W1", title="Updated title")
        later.citation_count = 100
        client = FakeOpenAlexClient({self.q1.text: [self.a], self.q2.text: [later]})

        candidates = PaperSearchService(client).search_candidates([self.q1, self.q2])

        self.assertEqual(len(candidates), 1)
        self.assertIs(candidates[0].paper, self.a)
        self.assertEqual(candidates[0].paper.title, "Example paper")
        self.assertEqual(candidates[0].paper.citation_count, 0)
        self.assertEqual(later.title, "Updated title")

    def test_first_seen_order_is_stable_across_repeated_calls(self) -> None:
        client = FakeOpenAlexClient(
            {
                self.q1.text: [self.b, self.a, self.b],
                self.q2.text: [self.c, self.a, self.d, self.c],
                self.q3.text: [self.d, self.b],
            }
        )
        service = PaperSearchService(client)

        first = service.search_candidates([self.q1, self.q2, self.q3])
        second = service.search_candidates([self.q1, self.q2, self.q3])

        self.assertEqual(
            [item.paper for item in first], [self.b, self.a, self.c, self.d]
        )
        self.assertEqual(first, second)
        self.assertIsNot(first[0], second[0])
        self.assertEqual(client.responses[self.q1.text], [self.b, self.a, self.b])

    def test_identical_titles_with_different_paper_ids_stay_separate(self) -> None:
        client = FakeOpenAlexClient({self.q1.text: [self.a, self.b]})

        candidates = PaperSearchService(client).search_candidates([self.q1])

        self.assertEqual(
            [item.paper.paper_id for item in candidates], ["openalex:W1", "openalex:W2"]
        )

    def test_identical_text_with_distinct_query_ids_retains_both_hits(self) -> None:
        other_query = SearchQuery(query_id="other", text=self.q1.text)
        client = FakeOpenAlexClient({self.q1.text: [self.a]})

        candidates = PaperSearchService(client).search_candidates(
            [self.q1, other_query]
        )

        self.assertEqual([hit.query_id for hit in candidates[0].hits], ["q1", "other"])
        self.assertEqual(len(client.calls), 2)

    def test_provenance_preserves_the_text_sent_to_the_client(self) -> None:
        query = SearchQuery(query_id="q1", text='  Matrix  completion & "谱"  ')
        client = FakeOpenAlexClient({query.text: [self.a]})

        candidates = PaperSearchService(client).search_candidates([query])

        self.assertEqual(client.calls[0]["query"], 'Matrix  completion & "谱"')
        self.assertEqual(candidates[0].hits[0].query_text, client.calls[0]["query"])

    def test_empty_queries_are_rejected_before_search(self) -> None:
        client = FakeOpenAlexClient({})

        with self.assertRaisesRegex(ValueError, "queries must not be empty"):
            PaperSearchService(client).search_candidates([])

        self.assertEqual(client.calls, [])

    def test_duplicate_query_ids_are_rejected_before_any_search(self) -> None:
        duplicate = SearchQuery(query_id=" q1 ", text="another query")
        client = FakeOpenAlexClient({})

        with self.assertRaisesRegex(ValueError, "query_id values must be unique"):
            PaperSearchService(client).search_candidates([self.q1, self.q2, duplicate])

        self.assertEqual(client.calls, [])

    def test_invalid_per_query_is_rejected_before_search(self) -> None:
        client = FakeOpenAlexClient({})
        service = PaperSearchService(client)

        for per_query in (0, -1, 101, "10", 10.0, None, True, False):
            with self.subTest(per_query=per_query):
                with self.assertRaisesRegex(
                    ValueError, "per_query.*integer between 1 and 100"
                ):
                    service.search_candidates([self.q1], per_query=per_query)
                self.assertEqual(client.calls, [])

    def test_per_query_accepts_inclusive_boundaries(self) -> None:
        client = FakeOpenAlexClient({self.q1.text: []})
        service = PaperSearchService(client)

        for per_query in (1, 100):
            with self.subTest(per_query=per_query):
                service.search_candidates([self.q1], per_query=per_query)
                self.assertEqual(client.calls[-1]["per_page"], per_query)

    def test_year_bounds_and_page_size_are_passed_to_every_search(self) -> None:
        bounds = ((None, None), (2015, None), (None, 2026), (2015, 2026))
        for year_from, year_to in bounds:
            with self.subTest(year_from=year_from, year_to=year_to):
                client = FakeOpenAlexClient({self.q1.text: [], self.q2.text: []})
                PaperSearchService(client).search_candidates(
                    [self.q1, self.q2],
                    per_query=7,
                    year_from=year_from,
                    year_to=year_to,
                )
                self.assertEqual(
                    client.calls,
                    [
                        {
                            "query": query.text,
                            "per_page": 7,
                            "year_from": year_from,
                            "year_to": year_to,
                        }
                        for query in (self.q1, self.q2)
                    ],
                )

    def test_empty_search_results_return_empty_pool(self) -> None:
        client = FakeOpenAlexClient({self.q1.text: [], self.q2.text: []})

        self.assertEqual(
            PaperSearchService(client).search_candidates([self.q1, self.q2]), []
        )
        self.assertEqual(len(client.calls), 2)

    def test_underlying_exception_propagates_unchanged(self) -> None:
        failure = RuntimeError("search failed")
        client = FakeOpenAlexClient({self.q1.text: failure})

        with self.assertRaises(RuntimeError) as caught:
            PaperSearchService(client).search_candidates([self.q1])

        self.assertIs(caught.exception, failure)
        self.assertEqual(len(client.calls), 1)

    def test_later_failure_returns_no_partial_results_and_stops_searching(self) -> None:
        failure = RuntimeError("second search failed")
        client = FakeOpenAlexClient(
            {self.q1.text: [self.a], self.q2.text: failure, self.q3.text: [self.c]}
        )
        service = PaperSearchService(client)
        not_returned = object()
        result = not_returned

        with self.assertRaises(RuntimeError) as caught:
            result = service.search_candidates([self.q1, self.q2, self.q3])

        self.assertIs(caught.exception, failure)
        self.assertIs(result, not_returned)
        self.assertEqual(
            [call["query"] for call in client.calls], [self.q1.text, self.q2.text]
        )
        # A later request must not inherit any candidates from the failed request.
        fresh = service.search_candidates([self.q3])
        self.assertEqual([item.paper for item in fresh], [self.c])
