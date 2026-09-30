"""Deterministic page-bound chunks and small, dependency-free BM25 retrieval."""

from collections import Counter
from hashlib import sha256
import json
from math import log
import re
import unicodedata

from researchpilot.evidence import (DocumentPage, EvidencePassage, RetrievedPassage,
                                   RetrievalView, PassageViewHit, passage_index)


_STOP_WORDS = frozenset("a an the and or of to in on for by with as is are was were be been being how what which does do did can may under from this that these those it its their our we they into through about between using use used affect improve".split())


class PageChunker:
    def __init__(self, max_chars: int = 1800, overlap_chars: int = 320) -> None:
        if (type(max_chars) is not int or type(overlap_chars) is not int
                or max_chars < 200 or not 0 <= overlap_chars < max_chars // 2):
            raise ValueError("max_chars must be >= 200; overlap_chars must be < half the chunk size.")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, pages: list[DocumentPage]) -> list[EvidencePassage]:
        passages = []
        for page in pages:
            text = " ".join(page.text.split())
            # Skip page numbers and tiny headers, never merge across pages.
            if len(text) < 30:
                continue
            start = 0
            while start < len(text):
                end = min(start + self.max_chars, len(text))
                if end < len(text):
                    boundary = text.rfind(" ", start + self.max_chars // 2, end)
                    if boundary != -1:
                        end = boundary
                    # Keep a short tail with its predecessor (bounded by 1.2x).
                    if len(text) - end < self.max_chars // 5:
                        end = len(text)
                body = text[start:end].strip()
                identity = [page.paper_id, page.source_type, page.page_number, start, body]
                passage_id = "passage:" + sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:24]
                passages.append(EvidencePassage(
                    **page.model_dump(exclude={"text"}), text=body, passage_id=passage_id,
                ))
                if end == len(text):
                    break
                next_start = end - self.overlap_chars
                while next_start < end and text[next_start] != " ":
                    next_start += 1
                start = max(start + 1, next_start)
        passage_index(passages)
        return passages


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold().replace("\u00ad", "")
    # Normalize notation, not meaning: retain digits and mathematical labels.
    text = re.sub(r"\b([a-z])\.\s*(\d+)\b", r"\1\2", text)
    text = re.sub(r"\bnon[\-\u2010-\u2015\u2212\s]+(?=[a-z])", "non", text)
    words = re.findall(r"[^\W_]+", text)
    return [w for w in words if w not in _STOP_WORDS and (len(w) > 1 or w.isdigit())]


def retrieval_views(question: str, concepts: list[str], queries: list[str],
                    gaps: list[tuple[str, str]] = (),
                    passages: list[EvidencePassage] = ()) -> list[RetrievalView]:
    """At most six views, entirely derived from existing question/plan/gap text.

    Three narrow concept views use each concept's least frequent indexed token.
    Corpus document frequency chooses specificity, not a domain word list or an
    LLM rewrite. Original concepts remain in the academic-terminology view.
    Limiting overlapping views prevents broad paraphrases drowning a rare facet
    in RRF. Ties retain plan order; diagnostics retain the source phrase.
    """
    views = [RetrievalView(view_id="question", text=question)]
    frequencies = Counter(token for p in passages for token in set(_tokens(p.text)))
    focuses = []
    for concept in concepts:
        terms = [t for t in dict.fromkeys(_tokens(concept)) if frequencies[t]]
        term = min(terms, key=frequencies.get) if terms else concept
        focuses.append((frequencies[term] if terms else len(passages) + 1, term, concept))
    for i, (_, term, concept) in enumerate(sorted(focuses, key=lambda x: x[0])[:3], 1):
        views.append(RetrievalView(view_id=f"concept:{i}", text=term, origin_text=concept))
    if gaps:
        views.append(RetrievalView(view_id="gaps", text=" ".join(f"{d} {f}" for d, f in gaps)))
    if queries or concepts:
        views.append(RetrievalView(view_id="academic", text=" ".join([*queries, *concepts])))
    unique = {}
    for view in views:
        key = tuple(sorted(set(_tokens(view.text))))
        if key:
            unique.setdefault(key, view)
    return list(unique.values())


class LexicalRetriever:
    """BM25 with k1=1.5, b=0.75; ties retain passage order; zero matches omitted."""

    def retrieve(self, query: str, passages: list[EvidencePassage], top_k: int = 20) -> list[RetrievedPassage]:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        passage_index(passages)
        terms = sorted(set(_tokens(query)))
        documents = [Counter(_tokens(p.text)) for p in passages]
        if not documents or not terms:
            return []
        average = sum(sum(d.values()) for d in documents) / len(documents)
        if average == 0:
            return []
        frequencies = {term: sum(term in d for d in documents) for term in terms}
        scored = []
        for passage, counts in zip(passages, documents):
            length = sum(counts.values())
            score = 0.0
            for term in terms:
                frequency = counts[term]
                if frequency:
                    idf = log(1 + (len(documents) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                    score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * length / average))
            if score > 0:
                scored.append(RetrievedPassage(passage=passage, lexical_score=score))
        return sorted(scored, key=lambda item: item.lexical_score, reverse=True)[:top_k]

    def retrieve_views(self, views: list[RetrievalView], passages: list[EvidencePassage],
                       top_k: int = 20, per_view_top_k: int = 8) -> list[RetrievedPassage]:
        """Unweighted RRF (k=60), stable first encounter ties, unique passage IDs.

        Scores/ranks are internal diagnostics, never source evidence. Each view
        runs its own BM25 search, so narrow facets can recover otherwise lost text.
        """
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if type(per_view_top_k) is not int or not 1 <= per_view_top_k <= 20:
            raise ValueError("per_view_top_k must be between 1 and 20.")
        if len(views) > 8 or len({v.view_id for v in views}) != len(views):
            raise ValueError("At most eight unique retrieval views are allowed.")
        passage_index(passages)
        pool = {}
        for view in views:
            for rank, item in enumerate(self.retrieve(view.text, passages, per_view_top_k), 1):
                key = item.passage.passage_id
                if key not in pool:
                    pool[key] = item.model_copy(update={"fusion_score": 0.0, "view_hits": []})
                result = pool[key]
                result.fusion_score += 1.0 / (60 + rank)
                result.view_hits.append(PassageViewHit(view_id=view.view_id, rank=rank))
        return sorted(pool.values(), key=lambda item: item.fusion_score, reverse=True)[:top_k]
