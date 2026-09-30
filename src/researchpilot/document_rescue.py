"""Bounded gap-directed search inside already acquired documents; no downloads."""

from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import re
from typing import TYPE_CHECKING

from pydantic import Field

from researchpilot.evidence import EvidenceModel, EvidencePassage, RetrievalView, Text, VerifiedClaim
from researchpilot.math_evidence import VisualEvidenceRecord, visual_passage
from researchpilot.openai_page_evidence_client import EXTRACTION_VERSION, PageEvidenceClient, visual_evidence_id
from researchpilot.paper import Paper
from researchpilot.passage_retrieval import LexicalRetriever, _tokens, retrieval_views
from researchpilot.pdf_page_renderer import PDFPageRenderer
from researchpilot.document_acquisition import MAX_PDF_BYTES
from researchpilot.hybrid_retrieval import HybridPassageRetriever, RetrievalDiagnostics
from researchpilot.research_iteration import EvidenceGap

if TYPE_CHECKING:
    from researchpilot.research_models import SelectedPaper


class RescueLimits(EvidenceModel):
    max_local_rescue_passes: int = Field(default=2, strict=True, ge=0, le=3)
    max_rescue_gaps: int = Field(default=2, strict=True, ge=1, le=3)
    max_local_rescue_papers: int = Field(default=3, strict=True, ge=1, le=6)
    max_rescue_views_per_gap: int = Field(default=4, strict=True, ge=1, le=6)
    per_view_rescue_top_k: int = Field(default=12, strict=True, ge=1, le=20)
    max_rescue_passages_per_gap: int = Field(default=12, strict=True, ge=1, le=24)
    max_rescue_passages_per_run: int = Field(default=24, strict=True, ge=1, le=48)
    max_structural_candidates: int = Field(default=6, strict=True, ge=1, le=12)
    max_visual_pages_per_gap: int = Field(default=1, strict=True, ge=0, le=4)
    max_visual_pages_per_run: int = Field(default=6, strict=True, ge=0, le=6)


class FormulaExtractionAssessment(EvidenceModel):
    risky: bool
    reasons: list[str]


class VisualPageTrace(EvidenceModel):
    paper_id: Text
    page_number: int
    gap_id: Text
    risk_reasons: list[str]
    rendered: bool = False
    vision_called: bool = False
    render_cache_hit: bool = False
    evidence_id: str | None = None
    pdf_sha256: str | None = None
    outcome: str = "pending"


class RescueTrace(EvidenceModel):
    round: int
    corpus_fingerprint: str
    local_rescue_triggered: bool = True
    gaps_targeted: list[str] = Field(default_factory=list)
    papers_searched: list[str] = Field(default_factory=list)
    passages_scanned: int = 0
    structural_candidates: list[str] = Field(default_factory=list)
    retrieval_views: list[RetrievalView] = Field(default_factory=list)
    offered_evidence_ids: list[str] = Field(default_factory=list)
    added_evidence_ids: list[str] = Field(default_factory=list)
    gap_evidence_ids: dict[str, list[str]] = Field(default_factory=dict)
    visual_pages: list[VisualPageTrace] = Field(default_factory=list)
    gaps_resolved: list[str] = Field(default_factory=list)
    outcome: str = "no_new_evidence"
    passage_retrieval: list[RetrievalDiagnostics] = Field(default_factory=list)


# Structure is metadata, not relevance. Names without numbering (e.g. Proof)
# are useful too; punctuation and appendix labels are retained for diagnostics.
_STRUCTURE = re.compile(r"\b(?:Theorem|Proposition|Lemma|Corollary|Assumption|Condition|Definition|Remark|Proof|Equation|Eq\.)"
                        r"(?:\s*\(?[A-Z]?\d+(?:\.\d+)*\)?|\s*\(?[A-Z]\.\d+\)?)?", re.I)
_FORMAL = re.compile(r"\b(?:theorems?|propositions?|lemmas?|assumptions?|conditions?|formal|guarantees?|spectral|curvature|proofs?|equations?|mathematical|bounds?)\b", re.I)
_GENERIC = frozenset("need needs evidence direct primary statement statements theorem proposition lemma assumption assumptions condition conditions formal guarantee guarantees proof paper papers result results provide provided lack lacking missing precise specific study studies support supported answer question finding findings gap comparison empirical".split())


def structural_markers(text: str) -> list[str]:
    return [m.group(0).strip() for m in _STRUCTURE.finditer(text)]


def needs_formal_evidence(gap: EvidenceGap) -> bool:
    return bool(_FORMAL.search(gap.description + " " + gap.search_focus))


def formula_risk(text: str) -> FormulaExtractionAssessment:
    """Signals only: never repair, normalize or reconstruct source mathematics."""
    reasons = []
    math = bool(re.search(r"[=≤≥≠√∑∏∥^_]|[α-ωΑ-Ω]|\b(?:sqrt|sigma|lambda)\b", text))
    if re.search(r"√\s*(?!\()\S|\bsqrt\s+(?!\()\S|[⎷⎸⎹]", text):
        reasons.append("ambiguous radical scope")
    if re.search(r"[\^_]\s*(?:$|[=≤≥+\n])|[α-ωΑ-Ω]\s*\n\s*\d|\b[a-zA-Z]\s*\n\s*\d", text):
        reasons.append("lost or detached superscript/subscript")
    if re.search(r"[≤≥<>=]\s*(?:\n\s*){2,}|[≤≥]\s*$|\n\s*[≤≥]\s*\n", text):
        reasons.append("fragmented inequality")
    if math and any(text.count(a) != text.count(b) for a, b in (("(", ")"), ("[", "]"), ("{", "}"))):
        reasons.append("broken mathematical parentheses")
    if math and (re.search(r"(?m)^\s*\d{1,3}\s*$", text) or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]", text)):
        reasons.append("detached number or extraction-order corruption")
    # Flattened degree expressions can have lost radicals entirely. Flag without
    # proposing what the missing mathematical structure might be.
    if re.search(r"\b2\s+d\d\s*[-−]\s*1\s*\+\s*2\s+d\d\s*[-−]\s*1", text):
        reasons.append("ambiguous flattened formula structure")
    # A prose reference to a table plus scattered measurements/citations is not
    # evidence of damaged columns. Require an actual flattened numeric row.
    if (re.search(r"\btable\s+\d", text, re.I)
            and re.search(r"(?<![\w.])(?:\d+(?:\.\d+)?%?\s+){3}\d+(?:\.\d+)?%?(?![\w.])", text)):
        reasons.append("table row/column layout dependency")
    return FormulaExtractionAssessment(risky=bool(reasons), reasons=reasons)


def corpus_fingerprint(passages: list[EvidencePassage]) -> str:
    # Normal passage IDs bind their content. Also bind text for custom fetchers.
    return sha256("\n".join(f"{p.passage_id}:{p.text}" for p in passages if p.source_type == "pdf").encode()).hexdigest()


def gap_views(question: str, gap: EvidenceGap, concepts: list[str], claims: list[VerifiedClaim],
              corpus: list[EvidencePassage], limit: int) -> list[RetrievalView]:
    related = " ".join(c.claim.text for c in claims if c.claim.claim_id in gap.related_claim_ids)
    texts = [RetrievalView(view_id=f"{gap.gap_id}:focus", text=gap.search_focus),
             RetrievalView(view_id=f"{gap.gap_id}:context", text=" ".join([gap.description, question, related]))]
    texts += [v.model_copy(update={"view_id": f"{gap.gap_id}:{v.view_id}"})
              for v in retrieval_views(question, concepts, [], passages=corpus) if v.view_id.startswith("concept:")]
    unique = {}
    for view in texts:
        key = tuple(sorted(set(_tokens(view.text))))
        if key:
            unique.setdefault(key, view)
    return list(unique.values())[:limit]


def neighbors(target: EvidencePassage, corpus: list[EvidencePassage]) -> list[EvidencePassage]:
    """Separate original chunks, immediately adjacent in that document's order."""
    pages = [p for p in corpus if p.paper_id == target.paper_id and p.source_type == "pdf"]
    index = next(i for i, p in enumerate(pages) if p.passage_id == target.passage_id)
    return [p for p in pages[max(0, index - 1):index + 2]
            if abs(p.page_number - target.page_number) <= 1]


@dataclass
class RescuePlan:
    gap: EvidenceGap
    corpus: list[EvidencePassage]
    views: list[RetrievalView]
    targets: list[EvidencePassage]
    offered: list[EvidencePassage]
    selected_ids: set[str] = field(default_factory=set)
    structural_targets: list[EvidencePassage] = field(default_factory=list)


@dataclass
class RescueOutcome:
    passages: list[EvidencePassage]
    trace: RescueTrace
    visual_evidence: dict[str, VisualEvidenceRecord] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class DocumentRescuer:
    def __init__(self, *, pdf_provider: Callable[[Paper], Path | None] | None = None,
                 renderer: PDFPageRenderer | None = None, page_client: PageEvidenceClient | None = None,
                 passage_retriever: HybridPassageRetriever | None = None) -> None:
        self.pdf_provider, self.renderer, self.page_client = pdf_provider, renderer, page_client
        self.passage_retriever = passage_retriever

    def plan(self, question: str, gaps: list[EvidenceGap], papers: list["SelectedPaper"],
             corpus: list[EvidencePassage], selected: dict[str, EvidencePassage], concepts: list[str],
             claims: list[VerifiedClaim], limits: RescueLimits) -> list[RescuePlan]:
        plans = []
        chosen_papers: set[str] = set()
        for gap in gaps:
            if gap.relevance_to_question != "core":
                continue
            terms = set(_tokens(gap.description + " " + gap.search_focus)) - _GENERIC
            context_terms = set(_tokens(question + " " + " ".join(concepts))) - _GENERIC
            related_ids = {e for c in claims if c.claim.claim_id in gap.related_claim_ids for e in c.claim.evidence_ids}
            scored = []
            for paper in papers:
                document = [p for p in corpus if p.paper_id == paper.paper.paper_id and p.source_type == "pdf"]
                if not document:
                    continue
                metadata = set(_tokens(paper.paper.title + " " + (paper.paper.abstract or "")))
                document_terms = set(t for p in document for t in _tokens(p.text))
                focus_matches = terms & (metadata | document_terms)
                if not focus_matches or not (context_terms & (metadata | document_terms)):
                    continue
                score = len(terms & metadata) * 2 + len(focus_matches) + sum(p.passage_id in related_ids for p in document)
                scored.append((score, document))
            scoped = []
            for _, document in sorted(scored, key=lambda item: item[0], reverse=True):
                key = document[0].paper_id
                if key not in chosen_papers and len(chosen_papers) >= limits.max_local_rescue_papers:
                    continue
                chosen_papers.add(key)
                scoped.extend(document)
            if not scoped:
                continue
            views = gap_views(question, gap, concepts, claims, scoped, limits.max_rescue_views_per_gap)
            ranked = LexicalRetriever().retrieve_views(views, scoped, limits.max_rescue_passages_per_gap,
                                                      per_view_top_k=limits.per_view_rescue_top_k)
            structural = []
            if needs_formal_evidence(gap):
                matching = [p for p in scoped if structural_markers(p.text) and terms & set(_tokens(p.text))]
                structural = [r.passage for r in LexicalRetriever().retrieve_views(views, matching,
                    limits.max_structural_candidates, per_view_top_k=limits.per_view_rescue_top_k)]
            targets = list({p.passage_id: p for p in [*structural, *(r.passage for r in ranked)]}.values())
            offered = {}
            for target in targets:
                # Target first so a tight cap cannot replace it with a neighbor.
                items = [target, *neighbors(target, scoped)] if structural_markers(target.text) else [target]
                for item in items:
                    if item.passage_id not in selected and len(offered) < limits.max_rescue_passages_per_gap:
                        offered.setdefault(item.passage_id, item)
            visual_possible = (self.page_client is not None and self.renderer is not None and self.pdf_provider is not None
                               and limits.max_visual_pages_per_gap and limits.max_visual_pages_per_run
                               and any(self._visual_target(p, gap) for p in targets))
            # Routing may plan several times: never embed here. A scoped document
            # with unseen chunks remains eligible even when lexical search missed.
            dense_possible = (self.passage_retriever is not None and self.passage_retriever.dense is not None
                              and any(p.passage_id not in selected for p in scoped))
            if offered or visual_possible or dense_possible:
                plans.append(RescuePlan(gap, scoped, views, targets, list(offered.values()), set(selected), structural))
            if len(plans) >= limits.max_rescue_gaps:
                break
        return plans

    @staticmethod
    def _visual_target(passage: EvidencePassage, gap: EvidenceGap) -> bool:
        risk = formula_risk(passage.text)
        if not ((set(_tokens(gap.description + " " + gap.search_focus)) - _GENERIC) & set(_tokens(passage.text))):
            return False
        # Prose gaps must not inherit vision from unrelated math on the page.
        formal = needs_formal_evidence(gap)
        table = "table row/column layout dependency" in risk.reasons and bool(re.search(
            r"\b(?:table|quantitative|accuracy|latency|percentage|measurement)\b", gap.description + " " + gap.search_focus, re.I))
        math_risk = any(reason != "table row/column layout dependency" for reason in risk.reasons)
        return (math_risk and formal) or table

    def rescue(self, question: str, plans: list[RescuePlan], papers: list["SelectedPaper"],
               limits: RescueLimits, history: list[RescueTrace], round_number: int, fingerprint: str,
               *, visual_only: bool = False) -> RescueOutcome:
        trace = RescueTrace(round=round_number, corpus_fingerprint=fingerprint)
        outcome = RescueOutcome([], trace)
        used_pages = {(p.paper_id, p.page_number) for h in history for p in h.visual_pages}
        used_pdf_pages = {(p.pdf_sha256, p.page_number) for h in history for p in h.visual_pages if p.pdf_sha256}
        remaining_pages = limits.max_visual_pages_per_run - len(used_pages)
        remaining_passages = max(0, limits.max_rescue_passages_per_run - sum(len(h.offered_evidence_ids) for h in history))
        offered = {}
        by_paper = {p.paper.paper_id: p.paper for p in papers}
        scanned = set()
        for plan in plans:
            if self.passage_retriever is not None and not visual_only:
                batch = self.passage_retriever.search_views(plan.views, plan.corpus,
                    limits.max_rescue_passages_per_gap, limits.per_view_rescue_top_k)
                trace.passage_retrieval.append(batch.diagnostics)
                outcome.warnings.extend(batch.warnings)
                targets = list({p.passage_id: p for p in [*plan.structural_targets,
                    *(r.passage for r in batch.passages)]}.values())
                selected = {}
                for target in targets:
                    items = [target, *neighbors(target, plan.corpus)] if structural_markers(target.text) else [target]
                    for item in items:
                        if item.passage_id not in plan.selected_ids and len(selected) < limits.max_rescue_passages_per_gap:
                            selected.setdefault(item.passage_id, item)
                # New plan, same original passage objects; routing inputs stay intact.
                plan = RescuePlan(plan.gap, plan.corpus, plan.views, targets, list(selected.values()),
                                  plan.selected_ids, plan.structural_targets)
            gap = plan.gap
            trace.gaps_targeted.append(gap.gap_id)
            trace.gap_evidence_ids[gap.gap_id] = []
            trace.retrieval_views.extend(plan.views)
            for passage in plan.corpus:
                scanned.add(passage.passage_id)
                if passage.paper_id not in trace.papers_searched:
                    trace.papers_searched.append(passage.paper_id)
            trace.structural_candidates.extend(p.passage_id for p in plan.targets if structural_markers(p.text)
                and p.passage_id not in trace.structural_candidates)
            # Reserve at most one slot for each possible visual page; text still
            # stays available in the immutable parser corpus/audit IDs.
            page_limit = min(remaining_pages, limits.max_visual_pages_per_gap, limits.max_rescue_passages_per_gap)
            for target in plan.targets:
                key = (target.paper_id, target.page_number)
                if (page_limit <= 0 or len(offered) >= remaining_passages or key in used_pages
                        or not self._visual_target(target, gap) or not self.pdf_provider
                        or not self.renderer or not self.page_client):
                    continue
                try:
                    pdf = self.pdf_provider(by_paper[target.paper_id])
                except Exception:
                    pdf = None
                if pdf is None:
                    continue
                # The same downloaded PDF can appear under more than one record.
                # Its page must not be sent twice, even under different paper IDs.
                try:
                    if pdf.stat().st_size > MAX_PDF_BYTES:
                        raise ValueError("PDF exceeds the rendering byte limit.")
                    pdf_hash = sha256(pdf.read_bytes()).hexdigest()
                except (OSError, ValueError) as exc:
                    outcome.warnings.append(f"Local PDF unavailable ({type(exc).__name__}); no visual evidence invented.")
                    continue
                if (pdf_hash, target.page_number) in used_pdf_pages:
                    continue
                used_pdf_pages.add((pdf_hash, target.page_number))
                used_pages.add(key)  # Failures consume the budget too; no preferred-output retries.
                page_limit -= 1
                remaining_pages -= 1
                event = VisualPageTrace(paper_id=target.paper_id, page_number=target.page_number,
                    gap_id=gap.gap_id, risk_reasons=formula_risk(target.text).reasons, pdf_sha256=pdf_hash)
                trace.visual_pages.append(event)
                try:
                    page = self.renderer.render(pdf, paper_id=target.paper_id, page_number=target.page_number)
                    if page.pdf_sha256 != pdf_hash:
                        raise ValueError("Cached PDF changed during page rescue.")
                    event.rendered, event.render_cache_hit = True, page.cache_hit
                    evidence_id = visual_evidence_id(page, question, gap, self.page_client.model)
                    event.vision_called = True
                    item = self.page_client.extract(question, gap, target.title, page, evidence_id)
                    if (item.evidence_id, item.paper_id, item.page_number) != (evidence_id, *key):
                        raise ValueError("Visual evidence identity mismatch.")
                    record = VisualEvidenceRecord(evidence=item, pdf_sha256=page.pdf_sha256,
                        image_sha256=page.image_sha256, render_version=page.render_version,
                        extraction_version=EXTRACTION_VERSION, model=self.page_client.model, gap_id=gap.gap_id,
                        parser_passage_ids=[p.passage_id for p in plan.corpus if (p.paper_id, p.page_number) == key],
                        parser_passages=[p for p in neighbors(target, plan.corpus) if p.page_number == target.page_number])
                    outcome.visual_evidence[evidence_id] = record
                    event.evidence_id = evidence_id
                    event.outcome = "ambiguous" if item.ambiguity else "transcribed"
                    if item.relevant_to_gap:
                        offered[evidence_id] = visual_passage(record, target)
                        trace.gap_evidence_ids[gap.gap_id].append(evidence_id)
                    else:
                        event.outcome = "irrelevant"
                except Exception as exc:
                    event.outcome = f"failed:{type(exc).__name__}"
                    outcome.warnings.append(f"Local page rescue failed ({type(exc).__name__}); no visual evidence invented.")
            for passage in plan.offered:
                if len(trace.gap_evidence_ids[gap.gap_id]) >= limits.max_rescue_passages_per_gap:
                    break
                if passage.passage_id not in offered and len(offered) >= remaining_passages:
                    break
                offered.setdefault(passage.passage_id, passage)
                trace.gap_evidence_ids[gap.gap_id].append(passage.passage_id)
        trace.passages_scanned = len(scanned)
        outcome.passages = list(offered.values())
        trace.offered_evidence_ids = list(offered)
        return outcome
