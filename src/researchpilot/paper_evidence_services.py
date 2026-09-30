"""Shared paper selection/acquisition used by both product modes."""

import httpx
from openai import OpenAIError
from researchpilot.document_acquisition import abstract_document
from researchpilot.evidence import EvidencePassage
from researchpilot.passage_retrieval import PageChunker
from researchpilot.paper import Paper
from researchpilot.paper_candidate import PaperCandidate
from researchpilot.ranked_paper_group import RankedPaperGroup
from researchpilot.research_models import RunStats, SelectedPaper

def _representative(group: RankedPaperGroup) -> Paper:
    def quality(candidate: PaperCandidate) -> tuple[bool, bool, int]:
        paper = candidate.paper
        return (bool(paper.abstract and paper.abstract.strip()), bool(paper.open_access_url),
                sum((bool(paper.doi), bool(paper.authors), paper.publication_year is not None)))
    # max retains the first encountered member for equal quality.
    return max(group.group.members, key=quality).paper


def _select_papers(self, question: str, ranked: list[RankedPaperGroup], count: int,
                   stats: RunStats, warnings: list[str]) -> list[SelectedPaper]:
    self._progress(f"Semantic paper assessment: {len(ranked)} groups")
    try:
        assessed = self.semantic_reranker.rerank(question, ranked)
    except (OpenAIError, httpx.HTTPError, RuntimeError) as exc:
        warnings.append(f"Semantic reranking unavailable ({type(exc).__name__}); using original RRF order.")
        stats.paper_ranking = "rrf_fallback"
        ordered = ranked
    else:
        originals = {g.group.group_id: g for g in ranked}
        ids = [item.ranked_group.group.group_id for item in assessed]
        if len(ids) != len(set(ids)) or set(ids) != set(originals):
            raise ValueError("Semantic reranker must return exactly the retrieved groups.")
        # Use actual retrieval objects even for injected implementations.
        ordered = [originals[key] for key in ids]
        stats.paper_ranking = "semantic"
    return [SelectedPaper(citation_label=f"P{i}", group_id=g.group.group_id,
                          paper=_representative(g))
            for i, g in enumerate(ordered[:count], 1)]

def _acquire_passages(self, papers: list[SelectedPaper], stats: RunStats,
                      warnings: list[str]) -> list[EvidencePassage]:
    pages = []
    for selected in papers:
        paper = selected.paper
        self._progress(f"Acquiring {selected.citation_label}: {paper.title}")
        try:
            document = self.document_fetcher.acquire(paper)
        except Exception as exc:
            document = abstract_document(paper, [f"{paper.paper_id}: document fetcher failed ({type(exc).__name__})."])
        warnings.extend(document.warnings)
        selected.cache_hit = document.cache_hit
        for page in document.pages:
            if page.paper_id != paper.paper_id or page.title != paper.title:
                raise ValueError("Document pages must match the retrieved source paper.")
        kinds = {p.source_type for p in document.pages}
        if len(kinds) > 1:
            raise ValueError("Document acquisition must return PDF pages or an abstract, not both.")
        if kinds:
            selected.source_type = next(iter(kinds))
            if selected.source_type == "pdf":
                stats.full_text_papers += 1
            else:
                stats.abstract_only_papers += 1
        pages.extend(document.pages)
    passages = PageChunker().chunk(pages)
    stats.passages_created = len(passages)
    return passages

