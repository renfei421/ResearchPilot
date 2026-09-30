"""Bounded BM25 + local cosine retrieval, fused by ranks rather than raw scores."""

from dataclasses import asdict, dataclass, field
import logging
from math import fsum, hypot
from time import monotonic

from pydantic import Field

from researchpilot.embedding_cache import EmbeddingCache, EmbeddingStats
from researchpilot.evidence import (
    EvidenceModel, EvidencePassage, PassageViewHit, RetrievalView, RetrievedPassage, passage_index,
)
from researchpilot.passage_retrieval import LexicalRetriever


class PassageRanks(EvidenceModel):
    passage_id: str
    lexical_rank: int | None = None
    dense_rank: int | None = None
    fused_rank: int
    lexical_score: float = 0.0
    dense_similarity: float | None = None


class HybridViewTrace(EvidenceModel):
    view_id: str
    ranks: list[PassageRanks] = Field(default_factory=list)


class RetrievalDiagnostics(EvidenceModel):
    mode: str = "lexical"
    embedding_model: str | None = None
    lexical_candidates: int = 0
    dense_candidates: int = 0
    fused_candidates: int = 0
    lexical_only_count: int = 0
    dense_only_count: int = 0
    overlap_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    corrupt_entries: int = 0
    passage_embeddings_generated: int = 0
    query_embeddings_generated: int = 0
    embedding_batches: int = 0
    embedding_input_tokens: int = 0
    embedding_elapsed_seconds: float = 0.0
    hybrid_elapsed_seconds: float = 0.0
    dense_fallback_used: bool = False
    views: list[HybridViewTrace] = Field(default_factory=list)
    fused_passage_ids: list[str] = Field(default_factory=list)


@dataclass
class RetrievalBatch:
    passages: list[RetrievedPassage]
    diagnostics: RetrievalDiagnostics
    warnings: list[str] = field(default_factory=list)


@dataclass
class DenseHit:
    passage: EvidencePassage
    similarity: float


class DensePassageRetriever:
    def __init__(self, cache: EmbeddingCache) -> None:
        self.cache = cache

    def retrieve_views(self, views: list[RetrievalView], passages: list[EvidencePassage],
                       top_k: int, stats: EmbeddingStats) -> list[list[DenseHit]]:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("Dense top_k must be a positive integer.")
        passage_index(passages)
        _validate_views(views)
        if not passages or not views:
            return [[] for _ in views]
        vectors = self.cache.embed([p.text for p in passages], stats, passages=True)
        # Rare-token lexical views retain their full concept in origin_text.
        queries = self.cache.embed([v.origin_text or v.text for v in views], stats, passages=False)
        unit = []
        for vector in vectors:
            norm = hypot(*vector)
            unit.append([x / norm for x in vector])
        results = []
        for query in queries:
            norm = hypot(*query)
            q = [x / norm for x in query]
            hits = [DenseHit(p, max(-1.0, min(1.0, fsum(a * b for a, b in zip(q, v)))))
                    for p, v in zip(passages, unit)]
            results.append(sorted(hits, key=lambda h: h.similarity, reverse=True)[:top_k])
        return results


def _validate_views(views: list[RetrievalView]) -> None:
    if len(views) > 8 or len({v.view_id for v in views}) != len(views):
        raise ValueError("At most eight unique retrieval views are allowed.")


class HybridPassageRetriever:
    """RRF k=60 within each view, then RRF k=60 across views.

    Each channel contributes at most per_view_top_k (1..20). Retain the whole
    bounded union (at most 2*k) within a view so single-channel hits are eligible.
    One vote per passage per view in the second stage; final top_k is unchanged.
    Ties use first encounter (view order, lexical before dense, corpus order).
    A missing dense dependency is explicit lexical mode for offline injection.
    """

    def __init__(self, dense: DensePassageRetriever | None = None) -> None:
        self.dense = dense
        self.lexical = LexicalRetriever()

    def search_views(self, views: list[RetrievalView], passages: list[EvidencePassage],
                     top_k: int = 20, per_view_top_k: int = 8) -> RetrievalBatch:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if type(per_view_top_k) is not int or not 1 <= per_view_top_k <= 20:
            raise ValueError("per_view_top_k must be between 1 and 20.")
        _validate_views(views)
        passage_index(passages)
        started = monotonic()
        diag, stats, warnings = RetrievalDiagnostics(), EmbeddingStats(), []
        lexical = [self.lexical.retrieve(v.text, passages, per_view_top_k) for v in views]
        dense = [[] for _ in views]
        if self.dense is not None and views and passages:
            diag.embedding_model = self.dense.cache.client.identity.model
            try:
                dense = self.dense.retrieve_views(views, passages, per_view_top_k, stats)
                diag.mode = "hybrid"
            except Exception as exc:
                # No exception body: HTTP/provider errors can contain input or secrets.
                diag.dense_fallback_used = True
                warning = f"Dense passage retrieval unavailable ({type(exc).__name__}); using lexical retrieval."
                warnings.append(warning)
                logging.getLogger(__name__).warning(warning)
        pool, lexical_ids, dense_ids = {}, set(), set()
        for view, lex, sem in zip(views, lexical, dense):
            scores, records = {}, {}
            for rank, item in enumerate(lex, 1):
                key = item.passage.passage_id
                lexical_ids.add(key)
                scores[key] = 1.0 / (60 + rank)
                records[key] = PassageRanks(passage_id=key, lexical_rank=rank, fused_rank=1,
                                           lexical_score=item.lexical_score)
                pool.setdefault(key, RetrievedPassage(passage=item.passage, lexical_score=item.lexical_score))
            for rank, item in enumerate(sem, 1):
                key = item.passage.passage_id
                dense_ids.add(key)
                scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
                record = records.setdefault(key, PassageRanks(passage_id=key, fused_rank=1))
                record.dense_rank, record.dense_similarity = rank, item.similarity
                pool.setdefault(key, RetrievedPassage(passage=item.passage, lexical_score=0.0))
            ordered = sorted(scores, key=scores.get, reverse=True)
            for rank, key in enumerate(ordered, 1):
                records[key].fused_rank = rank
                pool[key].fusion_score += 1.0 / (60 + rank)
                pool[key].view_hits.append(PassageViewHit(view_id=view.view_id, rank=rank))
            diag.views.append(HybridViewTrace(view_id=view.view_id, ranks=[records[key] for key in ordered]))
        ranked = sorted(pool.values(), key=lambda p: p.fusion_score, reverse=True)[:top_k]
        # Lexical mode/failure must preserve the original multi-view algorithm exactly.
        if diag.mode == "lexical":
            ranked = self.lexical.retrieve_views(views, passages, top_k, per_view_top_k)
        diag.lexical_candidates, diag.dense_candidates = len(lexical_ids), len(dense_ids)
        diag.lexical_only_count = len(lexical_ids - dense_ids)
        diag.dense_only_count = len(dense_ids - lexical_ids)
        diag.overlap_count = len(lexical_ids & dense_ids)
        diag.fused_candidates = len(ranked)
        diag.fused_passage_ids = [p.passage.passage_id for p in ranked]
        for key, value in asdict(stats).items():
            setattr(diag, key, value)
        diag.hybrid_elapsed_seconds = monotonic() - started
        return RetrievalBatch(ranked, diag, warnings)
