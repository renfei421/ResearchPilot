"""Synthetic, offline hybrid retrieval and cache contracts; no benchmark text."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import Mock

from phase5_helpers import OfflineTest, passage
from researchpilot.embedding_cache import EmbeddingCache, EmbeddingStats, normalized_embedding_text
from researchpilot.embeddings import EmbeddingBatch, EmbeddingIdentity, OpenAIEmbeddingClient
from researchpilot.evidence import RetrievalView
from researchpilot.hybrid_retrieval import DensePassageRetriever, HybridPassageRetriever
from researchpilot.passage_retrieval import LexicalRetriever


QUERY = "restricted curvature guarantee"
HIDDEN = "The Hessian of the objective is uniformly positive on admissible tangent-space perturbations."
EXACT = "Assumption A1 requires bounded coherence of the factors."


class FakeEmbeddingClient:
    identity = EmbeddingIdentity("fake", "synthetic", "v1", 3)

    def __init__(self, vectors=None):
        self.vectors = vectors or {}
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return EmbeddingBatch([self.vectors.get(t, [0.0, 1.0, 0.0]) for t in texts], len(texts))


def fake_hybrid(root, vectors=None, batch_size=64):
    client = FakeEmbeddingClient(vectors)
    cache = EmbeddingCache(root, client, batch_size=batch_size)
    return HybridPassageRetriever(DensePassageRetriever(cache)), client


class CacheTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.root = self.temporary_directory()
        self.client = FakeEmbeddingClient()
        self.cache = EmbeddingCache(self.root, self.client, batch_size=2)

    def embed(self, texts):
        stats = EmbeddingStats()
        result = self.cache.embed(texts, stats, passages=True)
        return result, stats

    def test_miss_generates_and_records_usage(self):
        _, stats = self.embed(["sample"])
        self.assertEqual((stats.cache_hits, stats.cache_misses, stats.passage_embeddings_generated), (0, 1, 1))
        self.assertEqual(stats.embedding_input_tokens, 1)

    def test_hit_reused_by_fresh_instance(self):
        first, _ = self.embed(["sample"])
        self.cache = EmbeddingCache(self.root, self.client)
        result, stats = self.embed(["sample"])
        self.assertEqual(result, first)
        self.assertEqual((stats.cache_hits, stats.cache_misses, stats.embedding_batches), (1, 0, 0))
        self.assertEqual(len(self.client.calls), 1)

    def test_duplicate_normalized_text_embedded_once_and_mapped_back(self):
        result, stats = self.embed(["cafe\u0301  data", "café\ndata"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], result[1])
        self.assertEqual(stats.cache_misses, 1)

    def test_normalization_preserves_math_case_and_symbols(self):
        self.assertEqual(normalized_embedding_text(" A1  σ₂ \nRSC "), "A1 σ₂ RSC")
        self.assertNotEqual(self.cache.key("A1"), self.cache.key("a1"))

    def test_identity_changes_invalidate(self):
        self.embed(["sample"])
        for changes in ({"model": "new"}, {"version": "v2"}, {"backend": "other"}):
            with self.subTest(changes=changes):
                self.client.identity = replace(FakeEmbeddingClient.identity, **changes)
                _, stats = self.embed(["sample"])
                self.assertEqual(stats.cache_misses, 1)

    def test_dimension_change_invalidates(self):
        self.embed(["sample"])
        self.client.identity = replace(self.client.identity, dimensions=2)
        self.client.vectors["sample"] = [1.0, 0.0]
        result, stats = self.embed(["sample"])
        self.assertEqual((len(result[0]), stats.cache_misses), (2, 1))

    def test_corrupt_json_rebuilt_atomically(self):
        self.embed(["sample"])
        self.cache.path_for("sample").write_text("broken", encoding="utf-8")
        _, stats = self.embed(["sample"])
        self.assertEqual((stats.corrupt_entries, stats.cache_misses), (1, 1))
        self.assertEqual(self.embed(["sample"])[1].cache_hits, 1)
        self.assertEqual(list(self.root.rglob("*.tmp")), [])

    def test_modified_vector_checksum_rebuilt(self):
        self.embed(["sample"])
        path = self.cache.path_for("sample")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["vector"][0] = 0.5
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.embed(["sample"])[1].corrupt_entries, 1)

    def test_cache_does_not_store_text_or_secrets(self):
        self.embed(["private passage text"])
        data = json.loads(self.cache.path_for("private passage text").read_text(encoding="utf-8"))
        self.assertEqual(set(data), {"key", "identity", "vector", "checksum"})
        self.assertNotIn("private passage text", json.dumps(data))

    def test_batching_and_original_order(self):
        texts = [f"text{i}" for i in range(5)]
        _, stats = self.embed(texts)
        self.assertEqual(self.client.calls, [texts[:2], texts[2:4], texts[4:]])
        self.assertEqual(stats.embedding_batches, 3)

    def test_batch_byte_cap(self):
        self.cache = EmbeddingCache(self.root, self.client)
        self.embed([str(i) + "x" * 7990 for i in range(32)])
        self.assertEqual(len(self.client.calls), 2)
        self.assertTrue(all(sum(len(t.encode()) for t in c) <= 131072 for c in self.client.calls))

    def test_bad_vectors_rejected_not_cached(self):
        for vector in ([0, 0, 0], [1, 2], [float("nan"), 1, 0], [float("inf"), 1, 0], [True, 1, 0]):
            with self.subTest(vector=vector):
                self.client.vectors["bad"] = vector
                with self.assertRaises(ValueError):
                    self.embed(["bad"])
                self.assertFalse(self.cache.path_for("bad").exists())

    def test_incomplete_batch_rejected(self):
        self.client.embed_texts = Mock(return_value=EmbeddingBatch([]))
        with self.assertRaisesRegex(ValueError, "count"):
            self.embed(["text"])

    def test_oversize_not_truncated_or_sent(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            self.embed(["x" * 8192])
        self.assertEqual(self.client.calls, [])

    def test_invalid_batch_size(self):
        for size in (0, 65, True, 1.5):
            with self.assertRaises(ValueError):
                EmbeddingCache(self.root, self.client, batch_size=size)


class AdapterTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.sdk = Mock()
        self.sdk.with_options.return_value = self.sdk
        self.sdk.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=1, embedding=[0.5] * 1536),
                  SimpleNamespace(index=0, embedding=[1.0] * 1536)],
            usage=SimpleNamespace(total_tokens=7))
        self.adapter = OpenAIEmbeddingClient(client=self.sdk)

    def test_batched_payload_only_text_and_embedding_configuration(self):
        result = self.adapter.embed_texts(["passage", "query"])
        self.sdk.with_options.assert_called_once_with(max_retries=0, timeout=60.0)
        self.sdk.embeddings.create.assert_called_once_with(model="text-embedding-3-small",
            input=["passage", "query"], dimensions=1536, encoding_format="float")
        self.assertEqual(result.vectors[0], [1.0] * 1536)
        self.assertEqual(result.input_tokens, 7)

    def test_empty_has_no_request(self):
        self.assertEqual(self.adapter.embed_texts([]).vectors, [])
        self.sdk.embeddings.create.assert_not_called()

    def test_duplicate_response_index_rejected(self):
        self.sdk.embeddings.create.return_value.data[1].index = 1
        with self.assertRaisesRegex(ValueError, "index"):
            self.adapter.embed_texts(["one", "two"])

    def test_provider_failure_propagates_without_retry(self):
        self.sdk.embeddings.create.side_effect = RuntimeError("failure")
        with self.assertRaises(RuntimeError):
            self.adapter.embed_texts(["one"])
        self.assertEqual(self.sdk.embeddings.create.call_count, 1)


class HybridTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.root = self.temporary_directory()
        self.corpus = [passage("exact", text=EXACT), passage("semantic", number=7, text=HIDDEN),
                       passage("noise", text="Unrelated crops and seasonal irrigation schedules.")]
        self.views = [RetrievalView(view_id="gap", text=QUERY), RetrievalView(view_id="identifier", text="Assumption A1")]
        self.hybrid, self.client = fake_hybrid(self.root, {QUERY: [1, 0, 0], HIDDEN: [1, 0, 0],
            "Assumption A1": [1, 0, 0], EXACT: [0, 0, 1]})

    def search(self, **kw):
        return self.hybrid.search_views(self.views, self.corpus, **({"top_k": 2, "per_view_top_k": 1} | kw))

    def test_lexical_behavior_preserved(self):
        expected = LexicalRetriever().retrieve_views(self.views, self.corpus, 2, 1)
        self.assertEqual(HybridPassageRetriever().search_views(self.views, self.corpus, 2, 1).passages, expected)

    def test_generic_semantic_match_and_exact_identifier_both_surface(self):
        self.assertEqual(LexicalRetriever().retrieve(QUERY, self.corpus, 1), [])
        batch = self.search()
        self.assertEqual({p.passage.passage_id for p in batch.passages}, {"exact", "semantic"})
        self.assertEqual(batch.diagnostics.lexical_only_count, 1)
        self.assertEqual(batch.diagnostics.dense_only_count, 1)
        self.assertEqual(batch.diagnostics.overlap_count, 0)

    def test_cosine_handles_non_unit_vectors(self):
        self.client.vectors[QUERY] = [2, 0, 0]
        self.client.vectors[HIDDEN] = [8, 0, 0]
        hits = self.hybrid.dense.retrieve_views(self.views[:1], self.corpus, 1, EmbeddingStats())
        self.assertEqual(hits[0][0].passage.passage_id, "semantic")
        self.assertAlmostEqual(hits[0][0].similarity, 1.0)

    def test_overlap_unique_and_hand_calculated_rrf(self):
        batch = self.search(per_view_top_k=3)
        self.assertEqual(len({r.passage.passage_id for r in batch.passages}), len(batch.passages))
        self.assertEqual(batch.diagnostics.overlap_count, 1)
        # Across views: semantic has ranks 1,2; exact has ranks 3,1.
        by_id = {r.passage.passage_id: r for r in batch.passages}
        self.assertAlmostEqual(by_id["semantic"].fusion_score, 1 / 61 + 1 / 62)

    def test_deterministic_order_and_rrf_scores(self):
        first, second = self.search(), self.search()
        self.assertEqual(first.passages, second.passages)
        self.assertEqual(first.diagnostics.views, second.diagnostics.views)

    def test_equal_scores_use_first_encounter_order(self):
        views = [RetrievalView(view_id="id", text="Assumption A1")]
        batch = self.hybrid.search_views(views, self.corpus, 2, 1)
        self.assertEqual([r.passage.passage_id for r in batch.passages], ["exact", "semantic"])

    def test_original_passage_provenance_and_inputs_unchanged(self):
        before = deepcopy(self.corpus)
        result = self.search()
        originals = {p.passage_id: p for p in self.corpus}
        for item in result.passages:
            self.assertIs(item.passage, originals[item.passage.passage_id])
        self.assertEqual(self.corpus, before)

    def test_final_cap_and_view_caps(self):
        self.assertEqual(len(self.search(top_k=1).passages), 1)
        for views in ([RetrievalView(view_id=str(i), text=QUERY) for i in range(9)], self.views * 2):
            with self.assertRaises(ValueError):
                self.hybrid.search_views(views, self.corpus)

    def test_invalid_limits_precede_embedding(self):
        for kwargs in ({"top_k": 0}, {"per_view_top_k": 21}, {"top_k": True}):
            with self.assertRaises(ValueError):
                self.search(**kwargs)
        self.assertEqual(self.client.calls, [])

    def test_duplicate_passage_identity_rejected(self):
        with self.assertRaises(ValueError):
            self.hybrid.search_views(self.views, self.corpus * 2)
        self.assertEqual(self.client.calls, [])

    def test_concept_embedding_uses_full_origin_text(self):
        self.hybrid.search_views([RetrievalView(view_id="concept", text="curvature", origin_text=QUERY)], self.corpus)
        self.assertIn([QUERY], self.client.calls)

    def test_corpus_embedded_once_for_all_views(self):
        batch = self.search()
        self.assertEqual(batch.diagnostics.passage_embeddings_generated, 3)
        self.assertEqual(batch.diagnostics.query_embeddings_generated, 2)
        self.assertEqual(batch.diagnostics.embedding_batches, 2)

    def test_repeated_search_hits_disk_cache(self):
        self.search()
        second = self.search()
        self.assertEqual(second.diagnostics.cache_hits, 5)
        self.assertEqual(second.diagnostics.embedding_batches, 0)

    def test_dense_failure_exact_lexical_fallback_with_safe_warning(self):
        self.client.embed_texts = Mock(side_effect=RuntimeError("secret should not appear"))
        batch = self.search()
        self.assertEqual(batch.passages, LexicalRetriever().retrieve_views(self.views, self.corpus, 2, 1))
        self.assertTrue(batch.diagnostics.dense_fallback_used)
        self.assertEqual(batch.diagnostics.mode, "lexical")
        self.assertEqual(batch.diagnostics.dense_candidates, 0)
        self.assertNotIn("secret", str(batch.warnings))

    def test_empty_corpus_no_embedding(self):
        self.assertEqual(self.hybrid.search_views(self.views, []).passages, [])
        self.assertEqual(self.client.calls, [])

    def test_dense_similarity_is_diagnostic_not_evidence_confidence(self):
        batch = self.search()
        self.assertTrue(any(r.dense_similarity is not None for v in batch.diagnostics.views for r in v.ranks))
        for item in batch.passages:
            self.assertNotIn("confidence", item.model_dump())
            self.assertNotIn("dense_similarity", item.passage.model_dump())
