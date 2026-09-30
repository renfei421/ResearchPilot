"""Public research facade and reusable paper services for the bounded graph."""

from collections.abc import Callable
from time import monotonic
from typing import Protocol
from uuid import uuid4

import httpx
from langgraph.checkpoint.base import BaseCheckpointSaver
from openai import OpenAI, OpenAIError

from researchpilot.paper_evidence_services import _select_papers, _acquire_passages, _representative
from researchpilot.config import Settings
from researchpilot.progress import ResearchProgress

from researchpilot.document_acquisition import DocumentAcquirer, abstract_document
from researchpilot.document_rescue import DocumentRescuer
from researchpilot.evidence_assembly import EvidenceAssembler
from researchpilot.pdf_page_renderer import PDFPageRenderer
from researchpilot.openai_page_evidence_client import OpenAIPageEvidenceClient
from researchpilot.evidence import (
    AcquiredDocument, EvidencePassage,
)
from researchpilot.openai_evidence_client import (
    ClaimSynthesizer, ClaimVerifier, EvidenceSelector, OpenAIClaimSynthesizer,
    OpenAIClaimVerifier, OpenAIEvidenceSelector,
)
from researchpilot.openai_query_planner import OpenAIQueryPlanner
from researchpilot.openai_relevance_client import OpenAIRelevanceClient
from researchpilot.openai_research_iteration import OpenAIEvidenceAssessor, OpenAIFollowUpPlanner
from researchpilot.openalex_client import OpenAlexClient
from researchpilot.paper import Paper
from researchpilot.paper_candidate import PaperCandidate, SearchQuery
from researchpilot.paper_search_service import PaperSearchService
from researchpilot.passage_retrieval import PageChunker
from researchpilot.embeddings import OpenAIEmbeddingClient
from researchpilot.embedding_cache import EmbeddingCache
from researchpilot.hybrid_retrieval import DensePassageRetriever, HybridPassageRetriever
from researchpilot.query_planner import QueryPlanner
from researchpilot.ranked_paper_group import RankedPaperGroup
from researchpilot.relevance import SemanticallyRankedGroup
from researchpilot.research_models import ResearchRequest, ResearchResult, RunStats, SelectedPaper
from researchpilot.research_graph import build_research_graph
from researchpilot.research_iteration import EvidenceAssessor, FollowUpPlanner
from researchpilot.semantic_reranker import SemanticReranker


class CandidateSearch(Protocol):
    def search_candidates(self, queries: list[SearchQuery], per_query: int = 10,
                          year_from: int | None = None, year_to: int | None = None) -> list[PaperCandidate]: ...


class PaperReranker(Protocol):
    def rerank(self, question: str, ranked_groups: list[RankedPaperGroup]) -> list[SemanticallyRankedGroup]: ...


class DocumentFetcher(Protocol):
    def acquire(self, paper: Paper) -> AcquiredDocument: ...


class ResearchAgent:
    """Small DI surface; construction does not call APIs.

    Operational semantic failures fall back to RRF with a warning. Iteration
    assessment/planning failures stop follow-up search safely. Initial search,
    synthesis and verification errors propagate without an unverified answer.
    Transient individual search failures allow explicitly warned partial retrieval;
    all-query outages and malformed query/response failures still propagate.
    """

    def __init__(self, *, planner: QueryPlanner | None = None,
                 paper_search: CandidateSearch | None = None,
                 semantic_reranker: PaperReranker | None = None,
                 document_fetcher: DocumentFetcher | None = None,
                 evidence_selector: EvidenceSelector | None = None,
                 synthesizer: ClaimSynthesizer | None = None,
                 verifier: ClaimVerifier | None = None,
                 evidence_assessor: EvidenceAssessor | None = None,
                 follow_up_planner: FollowUpPlanner | None = None,
                 document_rescuer: DocumentRescuer | None = None,
                 passage_retriever: HybridPassageRetriever | None = None,
                 evidence_assembler: EvidenceAssembler | None = None,
                 checkpointer: BaseCheckpointSaver | None = None,
                 model: str = "gpt-5.6-terra",
                 progress: Callable[[str], None] | None = None,
                 on_progress: Callable[[ResearchProgress], None] | None = None,
                 settings: Settings | None = None,
                 openai_client: OpenAI | None = None) -> None:
        settings = settings or Settings.from_env()
        self.planner = planner if planner is not None else OpenAIQueryPlanner(model=model, client=openai_client)
        self.paper_search = paper_search if paper_search is not None else PaperSearchService(
            OpenAlexClient(api_key=settings.openalex_api_key.get_secret_value() if settings.openalex_api_key else None,
                           max_retries=2))
        self.semantic_reranker = semantic_reranker if semantic_reranker is not None else SemanticReranker(
            OpenAIRelevanceClient(model=model, client=openai_client))
        self.document_fetcher = document_fetcher if document_fetcher is not None else DocumentAcquirer(settings.cache_dir / "documents")
        self.evidence_selector = evidence_selector if evidence_selector is not None else OpenAIEvidenceSelector(model=model, client=openai_client)
        self.synthesizer = synthesizer if synthesizer is not None else OpenAIClaimSynthesizer(model=model, client=openai_client)
        self.verifier = verifier if verifier is not None else OpenAIClaimVerifier(model=model, client=openai_client)
        self.evidence_assessor = evidence_assessor if evidence_assessor is not None else OpenAIEvidenceAssessor(model=model, client=openai_client)
        self.follow_up_planner = follow_up_planner if follow_up_planner is not None else OpenAIFollowUpPlanner(model=model, client=openai_client)
        self.passage_retriever = passage_retriever if passage_retriever is not None else HybridPassageRetriever(
            DensePassageRetriever(EmbeddingCache(settings.cache_dir / "embeddings",
                                                OpenAIEmbeddingClient(client=openai_client))))
        self.document_rescuer = document_rescuer if document_rescuer is not None else DocumentRescuer(
            passage_retriever=self.passage_retriever,
            pdf_provider=self.document_fetcher.cached_pdf if isinstance(self.document_fetcher, DocumentAcquirer) else None,
            renderer=PDFPageRenderer(settings.cache_dir / "page_images"),
            page_client=OpenAIPageEvidenceClient(model=model, client=openai_client))
        self.evidence_assembler = evidence_assembler if evidence_assembler is not None else EvidenceAssembler()
        self._checkpointer = checkpointer
        self._progress = progress or (lambda message: None)
        self._on_progress = on_progress or (lambda event: None)

    def _select_papers(self, question, ranked, count, stats, warnings):
        return _select_papers(self, question, ranked, count, stats, warnings)

    def _acquire_passages(self, papers, stats, warnings):
        return _acquire_passages(self, papers, stats, warnings)

    def run(self, request: ResearchRequest) -> ResearchResult:
        """Run the bounded graph; each invocation has a fresh checkpoint identity."""
        graph = build_research_graph(self, checkpointer=self._checkpointer)
        state = graph.invoke(
            {"request": request, "started_at": monotonic()},
            config={"recursion_limit": 8 * request.max_search_rounds + 8 + 5 * request.rescue_limits.max_local_rescue_passes + 5 * request.assembly_limits.max_bundles_per_run,
                    "configurable": {"thread_id": uuid4().hex}},
        )
        return state["final_result"]
