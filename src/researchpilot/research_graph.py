"""Bounded LangGraph orchestration over ordinary Phase 5 Python services."""

from time import monotonic
from typing import TYPE_CHECKING, TypedDict

import httpx
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from openai import OpenAIError

from researchpilot.answer_renderer import render_answer
from researchpilot.evidence_bundle import EvidenceBundle, AssemblyLimits
from researchpilot.bundle_runtime import assemble_selected, assembly_targets
from researchpilot.claim_stability import (DowngradeAudit, assert_evidence_union, merge_targeted_draft,
    recheck_supported, audit_downgrade)
from researchpilot.bundle_support import guard_bundle_support, reconcile_bundle_gaps
from researchpilot.document_rescue import RescueTrace, RescueLimits, corpus_fingerprint
from researchpilot.math_evidence import VisualEvidenceRecord, guard_ambiguous_visual_support
from researchpilot.evidence import (
    AnswerDraft, EvidencePassage, EvidenceSelection, RetrievedPassage, VerifiedClaim,
    passage_index, require_known_ids, validate_draft, validate_verification,
)
from researchpilot.paper_candidate import PaperCandidate, SearchQuery
from researchpilot.paper_version_resolver import PaperVersionResolver
from researchpilot.passage_retrieval import retrieval_views
from researchpilot.hybrid_retrieval import RetrievalDiagnostics
from researchpilot.query_plan import SearchPlan
from researchpilot.progress import state_progress
from researchpilot.query_planner import PlannerRequest
from researchpilot.research_iteration import (
    EvidenceAssessment, EvidenceGapRecord, FollowUpSearchPlan, active_gaps, searchable_gaps,
    new_follow_up_queries, update_gap_ledger, validate_assessment,
)
from researchpilot.research_models import (
    ResearchRequest, ResearchResult, RoundTrace, RunStats, SelectedPaper, TerminationReason,
)
from researchpilot.rrf_ranker import RRFRanker
from researchpilot.retrieval_diagnostics import is_transient, log_retrieval
from researchpilot.search_plan_adapter import search_plan_to_queries
from researchpilot.project_reuse import seed_project

if TYPE_CHECKING:
    from researchpilot.research_agent import ResearchAgent


class ResearchState(TypedDict, total=False):
    request: ResearchRequest
    started_at: float
    search_plan: SearchPlan
    search_round: int
    executed_queries: list[SearchQuery]
    current_queries: list[SearchQuery]
    follow_up_plans: list[FollowUpSearchPlan]
    accumulated_candidates: dict[str, PaperCandidate]
    selected_papers: list[SelectedPaper]
    pending_paper_ids: list[str]
    # Page text is discarded after chunking. Maps preserve first-seen order and
    # share passage objects in memory rather than duplicating documents per round.
    passages: dict[str, EvidencePassage]
    retrieved_passages: dict[str, RetrievedPassage]
    selected_evidence: dict[str, EvidencePassage]
    selection: EvidenceSelection | None
    draft: AnswerDraft
    claims: list[VerifiedClaim]  # Includes verifications; no second claims copy.
    evidence_assessment: EvidenceAssessment | None
    gap_ledger: list[EvidenceGapRecord]
    warnings: list[str]
    run_stats: RunStats
    round_trace: list[RoundTrace]
    termination_reason: TerminationReason | None
    final_result: ResearchResult
    rescue_trace: list[RescueTrace]
    visual_evidence: dict[str, VisualEvidenceRecord]
    evidence_bundles: list[EvidenceBundle]
    assembly_cache: dict[str, str]
    assembly_context_counts: dict[str, int]
    assembly_visual_trace: list[RescueTrace]
    verified_evidence: dict[str, EvidencePassage]
    downgrade_audit: list[DowngradeAudit]
    assembly_attempts: list[str]
    assembly_added: bool
    atom_bundle_support: dict[str, dict[str, list[str]]]


def merge_candidates(previous: dict[str, PaperCandidate], incoming: list[PaperCandidate]) -> dict[str, PaperCandidate]:
    """Retain first-seen metadata and merge best hits without mutating any input."""
    merged = dict(previous)
    for candidate in incoming:
        key = candidate.paper.paper_id
        prior = merged.get(key)
        if prior is None:
            merged[key] = candidate.model_copy(deep=True)
        else:
            hits = {h.query_id: h for h in prior.hits}
            for hit in candidate.hits:
                if hit.query_id not in hits or hit.rank < hits[hit.query_id].rank:
                    hits[hit.query_id] = hit
            merged[key] = PaperCandidate(paper=prior.paper, hits=list(hits.values()))
    return merged


def _trace_update(state: ResearchState, **updates) -> list[RoundTrace]:
    return [*state["round_trace"][:-1], state["round_trace"][-1].model_copy(update=updates)]


class _ResearchNodes:
    def __init__(self, services: "ResearchAgent") -> None:
        self.services = services

    def _progress(self, state: ResearchState, message: str) -> None:
        self.services._progress(f"[Round {state['search_round']}] {message}")

    def initial_plan(self, state: ResearchState) -> dict:
        request = state["request"]
        context = getattr(self.services, "project_context", None)
        if context is not None and context.project_id != request.project_id:
            raise ValueError("Project context must belong to the current request.")
        papers, passages, visuals, gaps = seed_project(context, max_papers=request.max_papers)
        self.services._progress("[Round 1] Planning/searching")
        plan = self.services.planner.plan(PlannerRequest(
            research_question=request.question, year_from=request.year_from,
            year_to=request.year_to, max_queries=request.max_queries,
            project_context=context.planner_payload() if context else None,
        ))
        if plan.research_question != request.question or len(plan.queries) > request.max_queries:
            raise ValueError("Planner returned an inconsistent research question/query count.")
        return dict(
            started_at=state.get("started_at", monotonic()), search_plan=plan, search_round=1,
            current_queries=search_plan_to_queries(plan), executed_queries=[], follow_up_plans=[],
            accumulated_candidates={p.paper.paper_id: PaperCandidate(paper=p.paper, hits=[]) for p in papers},
            selected_papers=papers, pending_paper_ids=[], passages=passages,
            retrieved_passages={}, selected_evidence={}, selection=None,
            draft=AnswerDraft(claims=[], limitations=[]), claims=[], evidence_assessment=None, gap_ledger=gaps,
            warnings=[], run_stats=RunStats(), round_trace=[], termination_reason=None,
            rescue_trace=[], visual_evidence=visuals, evidence_bundles=[], assembly_cache={},
            assembly_context_counts={}, assembly_visual_trace=[], verified_evidence={}, downgrade_audit=[],
            assembly_attempts=[], assembly_added=False, atom_bundle_support={},
        )

    def search_and_rank(self, state: ResearchState) -> dict:
        request, queries = state["request"], state["current_queries"]
        number = state["search_round"]
        if number > request.max_search_rounds:
            raise ValueError("Search round exceeds the hard budget.")
        stats = state["run_stats"].model_copy(deep=True)
        warnings = list(state["warnings"])
        previous = state["accumulated_candidates"]
        trace = RoundTrace(round=number, queries=len(queries), query_ids=[q.query_id for q in queries],
                           evidence_passages=len(state["selected_evidence"]),
                           remaining_gaps=len(active_gaps(state["gap_ledger"])),
                           gap_ids_carried=[g.gap_id for g in active_gaps(state["gap_ledger"])])
        self._progress(state, f"Searching {len(queries)} queries")
        updates = dict(executed_queries=[*state["executed_queries"], *queries],
                       run_stats=stats, warnings=warnings, round_trace=[*state["round_trace"], trace])
        stats.search_rounds = number
        stats.planned_queries += len(queries)
        log_retrieval("literature_queries_planned", round=number,
                      queries=[q.model_dump() for q in queries],
                      year_from=request.year_from, year_to=request.year_to)
        try:
            query_pool: dict[str, PaperCandidate] = {}
            successes, last_failure = 0, None
            for query in queries:
                context = dict(round=number, query_id=query.query_id, query=query.text,
                               year_from=request.year_from, year_to=request.year_to)
                log_retrieval("literature_query_started", **context)
                try:
                    candidates = self.services.paper_search.search_candidates(
                        queries=[query], per_query=10, year_from=request.year_from, year_to=request.year_to,
                    )
                except Exception as exc:
                    log_retrieval("literature_query_failed", error=exc, **context)
                    if not is_transient(exc):
                        raise
                    last_failure = exc
                    warnings.append(f"[Round {number}] Literature query {query.query_id} failed temporarily "
                                    f"({type(exc).__name__}); results cover successful queries only.")
                else:
                    successes += 1  # Empty results are still a successful request.
                    query_pool = merge_candidates(query_pool, candidates)
                    log_retrieval("literature_query_completed", candidates=len(candidates), **context)
            if not successes and last_failure is not None:
                raise last_failure
            incoming = list(query_pool.values())
        except (OpenAIError, httpx.HTTPError, RuntimeError) as exc:
            if number == 1:
                raise
            warnings.append(f"Follow-up search failed ({type(exc).__name__}); retaining previously verified evidence.")
            return updates | {"termination_reason": "follow_up_search_failed"}
        merged = merge_candidates(previous, incoming)
        groups = PaperVersionResolver().group_versions(list(merged.values()))
        old_ids = set(previous)
        incoming_ids = {c.paper.paper_id for c in incoming}
        # Group IDs include membership and can change. Check member identities,
        # so an additional version of a seen work is not a new research work.
        new_groups = [g for g in groups if not any(m.paper.paper_id in old_ids for m in g.members)]
        new_work_ids = {m.paper.paper_id for g in new_groups for m in g.members}
        trace.candidate_papers = len(incoming)
        trace.new_candidate_papers = len(incoming_ids - old_ids)
        trace.already_seen_papers = len(incoming_ids - new_work_ids)
        stats.raw_candidates = len(merged)
        stats.retrieved_query_hits = sum(len(c.hits) for c in merged.values())
        stats.version_groups = len(groups)
        membership = {m.paper.paper_id: g.group_id for g in groups for m in g.members}
        papers = [p.model_copy(update={"group_id": membership[p.paper.paper_id]}) for p in state["selected_papers"]]
        pending_prior = [p.paper.paper_id for p in papers] if number == 1 else []
        updates.update(accumulated_candidates=merged, selected_papers=papers, pending_paper_ids=pending_prior)
        stats.selected_papers = len(papers)
        if number > 1 and not new_groups:
            # New gap/query views can find evidence inside already acquired PDFs.
            return updates
        if not new_groups:
            if papers:
                return updates
            warnings.append("No relevant literature found for the current search constraints "
                            "in successful queries; no evidence-grounded answer is available.")
            return updates | {"termination_reason": "no_literature_found"}
        ranked = RRFRanker().rank_groups(new_groups)
        self.services._on_progress(state_progress("reranking", state))
        remaining = request.max_papers - len(papers) if number == 1 else request.max_papers
        selected = self.services._select_papers(request.question, ranked, remaining, stats, warnings) if remaining else []
        for offset, paper in enumerate(selected, len(papers) + 1):
            paper.citation_label = f"P{offset}"
        trace.new_papers = len(selected)
        trace.ranking_path = stats.paper_ranking
        stats.selected_papers = len(papers) + len(selected)
        self._progress(state, f"{len(selected)} new papers selected")
        return updates | {"selected_papers": [*papers, *selected],
                          "pending_paper_ids": [*pending_prior, *[p.paper.paper_id for p in selected]]}

    def acquire_documents(self, state: ResearchState) -> dict:
        stats = state["run_stats"].model_copy(deep=True)
        warnings = list(state["warnings"])
        papers = [p.model_copy(deep=True) for p in state["selected_papers"]]
        pending = set(state["pending_paper_ids"])
        chunks = self.services._acquire_passages([p for p in papers if p.paper.paper_id in pending], stats, warnings)
        passages = dict(state["passages"])
        for passage in chunks:
            if passage.passage_id in passages and passage != passages[passage.passage_id]:
                raise ValueError("Conflicting passage content for an existing passage ID.")
            passages.setdefault(passage.passage_id, passage)
        new_count = len(passages) - len(state["passages"])
        stats.passages_created = len(passages)
        return dict(selected_papers=papers, passages=passages, warnings=warnings, run_stats=stats,
                    round_trace=_trace_update(state, new_passages=new_count), termination_reason=None)

    def retrieve_evidence(self, state: ResearchState) -> dict:
        request = state["request"]
        gaps = searchable_gaps(state["gap_ledger"])
        views = retrieval_views(request.question, state["search_plan"].concepts,
                                [q.text for q in state["current_queries"]],
                                [(g.description, g.search_focus) for g in gaps],
                                list(state["passages"].values()))
        batch = self.services.passage_retriever.search_views(views, list(state["passages"].values()), request.top_passages)
        ranked = batch.passages
        retrieved = dict(state["retrieved_passages"])
        for item in ranked:
            retrieved[item.passage.passage_id] = item
        stats = state["run_stats"].model_copy(deep=True)
        stats.passages_retrieved = len(retrieved)
        stats.record_retrieval(batch.diagnostics)
        warnings = [*state["warnings"], *batch.warnings]
        evidence = dict(state["selected_evidence"])
        selection = state["selection"]
        if ranked:
            offered = [item.passage for item in ranked]
            self.services._on_progress(state_progress("selecting", state))
            selection = self.services.evidence_selector.select(request.question, offered)
            available = passage_index(offered)
            require_known_ids(selection.selected_passage_ids, available, "selected passage IDs")
            for key in selection.selected_passage_ids:
                evidence.setdefault(key, available[key])
            if selection.insufficient_evidence:
                warnings.append(f"[Round {state['search_round']}] Evidence selector reports insufficient coverage; any answer is partial.")
        else:
            warnings.append("No usable passages matched the question and planned concepts.")
        new_count = len(evidence) - len(state["selected_evidence"])
        stats.evidence_selected = len(evidence)
        if not evidence:
            warnings.append("No supporting passages selected; synthesis and verification skipped.")
        return dict(retrieved_passages=retrieved, selected_evidence=evidence, selection=selection,
                    run_stats=stats, warnings=warnings,
                    round_trace=_trace_update(state, new_evidence=new_count, evidence_passages=len(evidence),
                                             retrieval_views=views, passage_retrieval=batch.diagnostics),
                    termination_reason=("no_new_papers" if not state["round_trace"][-1].new_papers
                                        else "no_new_evidence")
                    if state["search_round"] > 1 and not new_count else None)

    def synthesize_claims(self, state: ResearchState) -> dict:
        if not state["selected_evidence"]:
            return {}
        assert_evidence_union(state.get("verified_evidence", {}), state["selected_evidence"])
        self._progress(state, f"Synthesizing from {len(state['selected_evidence'])} accumulated evidence passages")
        evidence = list(state["selected_evidence"].values())
        previous = state["claims"]
        additions = [p for key, p in state["selected_evidence"].items() if key not in state.get("verified_evidence", {})]
        if previous and callable(getattr(type(self.services.synthesizer), "synthesize_targets", None)):
            draft = self.services.synthesizer.synthesize_targets(state["request"].question, evidence,
                state["evidence_bundles"], [c.claim for c in previous if c.verification.status != "supported"],
                [c.claim for c in previous if c.verification.status == "supported"], active_gaps(state["gap_ledger"]))
        elif state["evidence_bundles"] and callable(getattr(type(self.services.synthesizer), "synthesize_with_bundles", None)):
            draft = self.services.synthesizer.synthesize_with_bundles(
                state["request"].question, evidence, state["evidence_bundles"])
        else:
            draft = self.services.synthesizer.synthesize(state["request"].question, evidence)
        if previous:
            draft = merge_targeted_draft(previous, draft, additions, state["selected_evidence"], state["evidence_bundles"])
        validate_draft(draft, evidence)
        stats = state["run_stats"].model_copy(deep=True)
        stats.synthesis_rounds += 1
        stats.claims_drafted = len(draft.claims)
        return dict(draft=draft, run_stats=stats,
                    round_trace=_trace_update(state, claims_drafted=len(draft.claims), synthesis_performed=True,
                        evidence_passages=len(evidence), new_evidence=state["round_trace"][-1].new_evidence +
                        max(0, len(evidence) - state["round_trace"][-1].evidence_passages)))

    def verify_claims(self, state: ResearchState) -> dict:
        stats = state["run_stats"].model_copy(deep=True)
        for status in ("supported", "partially_supported", "unsupported", "conflicting"):
            setattr(stats, status, 0)
        claims = []
        audits = list(state.get("downgrade_audit", []))
        previous = {c.claim.claim_id: c for c in state["claims"]}
        additions = [p for key,p in state["selected_evidence"].items() if key not in state.get("verified_evidence", {})]
        assert_evidence_union(state.get("verified_evidence", {}), state["selected_evidence"])
        for claim in state["draft"].claims:
            old = previous.get(claim.claim_id)
            if old and old.verification.status == "supported":
                if claim.text != old.claim.text:
                    raise ValueError("Supported claim text must remain stable during rescue")
                if not recheck_supported(old, additions, state["selected_evidence"], active_gaps(state["gap_ledger"])):
                    if claim.evidence_ids != old.claim.evidence_ids:
                        raise ValueError("evidence_lost: stable claim evidence IDs changed")
                    claims.append(old.model_copy(deep=True))
                    stats.supported += 1
                    continue
                papers = {state["selected_evidence"][key].paper_id for key in old.claim.evidence_ids}
                claim = claim.model_copy(update={"evidence_ids": list(dict.fromkeys(
                    [*old.claim.evidence_ids, *[p.passage_id for p in additions if p.paper_id in papers]]))})
            cited = [state["selected_evidence"][key] for key in claim.evidence_ids]
            if state["evidence_bundles"] and callable(getattr(type(self.services.verifier), "verify_with_bundles", None)):
                verification = self.services.verifier.verify_with_bundles(claim, cited, state["evidence_bundles"])
            else:
                verification = self.services.verifier.verify(claim, cited)
            # Availability of a newly assembled chain cannot invalidate an
            # independently verified, unchanged finding. The atomic verifier
            # above still receives the evidence union and can invalidate it.
            if not (old and old.verification.status == "supported" and old.claim.text == claim.text):
                verification = guard_bundle_support(verification, state["evidence_bundles"], cited)
            verification = guard_ambiguous_visual_support(verification, state["visual_evidence"])
            validate_verification(claim, verification)
            current = VerifiedClaim(claim=claim, verification=verification)
            if old:
                audit = audit_downgrade(old, current)
                if audit:
                    audits.append(audit)
            claims.append(current)
            setattr(stats, verification.status, getattr(stats, verification.status) + 1)
            stats.verification_calls += 1
        mapping = {c.claim.claim_id: {a.assertion_id: list(dict.fromkeys(
            key for b in state["evidence_bundles"] for key in a.evidence_ids
            if key in {m.evidence_id for m in b.members})) for a in c.verification.assertions} for c in claims}
        return dict(claims=claims, run_stats=stats, downgrade_audit=audits, atom_bundle_support=mapping,
                    verified_evidence=dict(state["selected_evidence"]),
                    round_trace=_trace_update(state, supported_claims=stats.supported,
                                             partial_claims=stats.partially_supported,
                                             unsupported_claims=stats.unsupported,
                                             conflicting_claims=stats.conflicting))

    def assess_evidence(self, state: ResearchState) -> dict:
        warnings = list(state["warnings"])
        try:
            assessment = self.services.evidence_assessor.assess(
                state["request"].question, state["claims"], list(state["selected_evidence"].values()),
                prior_gaps=active_gaps(state["gap_ledger"]),
            )
            validate_assessment(assessment, state["claims"], list(state["selected_evidence"].values()),
                                active_gaps(state["gap_ledger"]))
            # Ledger identity/supersession checks are part of assessment validation.
            # Reject the whole reassessment if they fail after evidence rescue.
            ledger = update_gap_ledger(state["gap_ledger"], assessment, state["search_round"])
        except (OpenAIError, httpx.HTTPError, RuntimeError, ValueError) as exc:
            after_rescue = (state["rescue_trace"] and state["rescue_trace"][-1].round == state["search_round"]
                            and state["rescue_trace"][-1].outcome == "evidence_added")
            if after_rescue or state.get("assembly_added", False):
                # Reject the invalid assessment in full. Never infer resolution,
                # discard the last valid ledger, or discard verified findings.
                warnings.append(f"Rescue reassessment failed ({type(exc).__name__}); previous gaps remain open.")
                ledger = [g.model_copy(deep=True) for g in state["gap_ledger"]]
                for gap in ledger:
                    if state["rescue_trace"] and gap.gap_id in state["rescue_trace"][-1].gaps_targeted:
                        gap.rescue_result = "reassessment_failed"
                return dict(evidence_assessment=None, gap_ledger=ledger, warnings=warnings,
                    termination_reason="max_search_rounds" if state["search_round"] >= state["request"].max_search_rounds else None)
            if isinstance(exc, ValueError):
                raise  # Preserve initial/ordinary assessment contract failures.
            warnings.append(f"Evidence assessor failed ({type(exc).__name__}); no further search will be attempted.")
            return dict(evidence_assessment=None, warnings=warnings, termination_reason="evidence_assessor_failed")
        ledger = reconcile_bundle_gaps(ledger, state["evidence_bundles"], state["search_round"], state["claims"])
        remaining = active_gaps(ledger)
        self._progress(state, "Evidence sufficient" if assessment.sufficient else f"{len(remaining)} evidence gaps remain")
        if assessment.sufficient and not remaining:
            reason = "sufficient_evidence"
        elif state["search_round"] >= state["request"].max_search_rounds:
            reason = "max_search_rounds"
        elif not searchable_gaps(ledger):
            reason = "no_meaningful_gaps"
        else:
            reason = None
        previous_ids = {g.gap_id for g in state["gap_ledger"]}
        history = [t.model_copy(deep=True) for t in state["rescue_trace"]]
        for gap in ledger:
            if gap.rescue_attempted:
                gap.rescue_result = gap.status if gap.status in ("resolved", "partially_resolved") else gap.rescue_result
                for trace in history:
                    if (gap.gap_id in trace.gaps_targeted and gap.status == "resolved"
                            and set(gap.evidence_ids) & set(trace.added_evidence_ids)
                            and gap.gap_id not in trace.gaps_resolved):
                        trace.gaps_resolved.append(gap.gap_id)
        return dict(evidence_assessment=assessment, gap_ledger=ledger, termination_reason=reason,
                    rescue_trace=history,
                    round_trace=_trace_update(state, remaining_gaps=len(remaining),
                        gap_ids_new=list(dict.fromkeys([*state["round_trace"][-1].gap_ids_new,
                            *[g.gap_id for g in ledger if g.gap_id not in previous_ids]])),
                        **{f"gap_ids_{status}": list(dict.fromkeys([*getattr(state["round_trace"][-1], f"gap_ids_{status}"),
                            *[u.gap_id for u in assessment.gap_updates if u.status == status]]))
                           for status in ("resolved", "partially_resolved", "superseded")}))

    def _rescue_plans(self, state: ResearchState):
        limits, history = state["request"].rescue_limits, state["rescue_trace"]
        corpus = list(state["passages"].values())
        fingerprint = corpus_fingerprint(corpus)
        # A stronger per-corpus guard also prevents reworded/new gap IDs looping.
        if (len(history) >= limits.max_local_rescue_passes
                or any(t.corpus_fingerprint == fingerprint for t in history)
                or sum(len(t.offered_evidence_ids) for t in history) >= limits.max_rescue_passages_per_run):
            return []
        return self.services.document_rescuer.plan(
            state["request"].question, searchable_gaps(state["gap_ledger"]), state["selected_papers"],
            corpus, state["selected_evidence"], state["search_plan"].concepts, state["claims"], limits)

    def rescue_local_evidence(self, state: ResearchState) -> dict:
        plans = self._rescue_plans(state)
        if not plans:
            return {"termination_reason": "max_search_rounds" if state["search_round"] >= state["request"].max_search_rounds else None}
        self._progress(state, f"Rescuing evidence for {len(plans)} gaps inside acquired documents")
        outcome = self.services.document_rescuer.rescue(state["request"].question, plans, state["selected_papers"],
            state["request"].rescue_limits, [*state["rescue_trace"], *state["assembly_visual_trace"]], state["search_round"],
            corpus_fingerprint(list(state["passages"].values())))
        evidence, retrieved = dict(state["selected_evidence"]), dict(state["retrieved_passages"])
        warnings = [*state["warnings"], *outcome.warnings]
        selection = state["selection"]
        if outcome.passages:
            try:
                selection = self.services.evidence_selector.select(state["request"].question, outcome.passages)
                offered = passage_index(outcome.passages)
                require_known_ids(selection.selected_passage_ids, offered, "rescued passage IDs")
                for key in selection.selected_passage_ids:
                    evidence.setdefault(key, offered[key])
                for passage in outcome.passages:
                    retrieved.setdefault(passage.passage_id, RetrievedPassage(passage=passage, lexical_score=0.0))
            except (OpenAIError, httpx.HTTPError, RuntimeError, ValueError) as exc:
                warnings.append(f"Rescue selection failed ({type(exc).__name__}); proceeding with existing evidence.")
                selection = state["selection"]
        trace = outcome.trace
        trace.added_evidence_ids = [key for key in evidence if key not in state["selected_evidence"]]
        trace.outcome = "evidence_added" if trace.added_evidence_ids else "no_new_evidence"
        ledger = [g.model_copy(deep=True) for g in state["gap_ledger"]]
        for gap in ledger:
            if gap.gap_id in trace.gaps_targeted:
                gap.rescue_attempted, gap.rescue_round = True, state["search_round"]
                gap.rescued_evidence_ids = list(dict.fromkeys([*gap.rescued_evidence_ids,
                    *[key for key in trace.gap_evidence_ids[gap.gap_id] if key in trace.added_evidence_ids]]))
                gap.rescue_result = trace.outcome
        stats = state["run_stats"].model_copy(deep=True)
        stats.local_rescue_passes += 1
        for diagnostics in trace.passage_retrieval:
            stats.record_retrieval(diagnostics)
        stats.visual_pages_rendered += sum(p.rendered for p in trace.visual_pages)
        stats.visual_evidence_calls += sum(p.vision_called for p in trace.visual_pages)
        stats.evidence_selected, stats.passages_retrieved = len(evidence), len(retrieved)
        return dict(selected_evidence=evidence, retrieved_passages=retrieved, selection=selection,
            gap_ledger=ledger, rescue_trace=[*state["rescue_trace"], trace], warnings=warnings, run_stats=stats,
            visual_evidence=state["visual_evidence"] | outcome.visual_evidence,
            round_trace=_trace_update(state, local_rescue_triggered=True, evidence_passages=len(evidence),
                new_evidence=state["round_trace"][-1].new_evidence + len(trace.added_evidence_ids)),
            termination_reason=None if trace.added_evidence_ids or state["search_round"] < state["request"].max_search_rounds
                else "max_search_rounds")

    def assemble_evidence(self, state: ResearchState) -> dict:
        update = assemble_selected(state, self.services)
        added = set(update.get("selected_evidence", {})) - set(state["selected_evidence"])
        token = corpus_fingerprint(list(state["passages"].values())) + ":" + str(len(state["selected_evidence"]))
        return update | dict(assembly_added=bool(added), assembly_attempts=[*state["assembly_attempts"], token])

    def _can_assemble(self, state):
        token = corpus_fingerprint(list(state["passages"].values())) + ":" + str(len(state["selected_evidence"]))
        return (len(state["evidence_bundles"]) < state["request"].assembly_limits.max_bundles_per_run
                and token not in state["assembly_attempts"] and bool(assembly_targets(state)))

    def after_assembly(self, state):
        if state["assembly_added"]:
            return "synthesize"
        if self._rescue_plans(state):
            return "rescue"
        return "stop" if state["termination_reason"] else "follow_up"

    def after_assessment(self, state: ResearchState) -> str:
        if state["termination_reason"] != "evidence_assessor_failed" and self._can_assemble(state):
            return "assemble"
        if state["termination_reason"] not in ("sufficient_evidence", "evidence_assessor_failed") and self._rescue_plans(state):
            return "rescue"
        return "stop" if state["termination_reason"] else "follow_up"

    def after_retrieval(self, state: ResearchState) -> str:
        if state["termination_reason"] and self._rescue_plans(state):
            return "rescue"
        return "stop" if state["termination_reason"] else "synthesize"

    def after_rescue(self, state: ResearchState) -> str:
        if state["rescue_trace"] and state["rescue_trace"][-1].outcome == "evidence_added":
            return "synthesize"
        return "stop" if state["termination_reason"] else "follow_up"

    def plan_follow_up(self, state: ResearchState) -> dict:
        warnings = list(state["warnings"])
        # Routing already checks this; keep an independent hard guard at the
        # only back-edge, so no planner output can bypass the budget.
        if state["search_round"] >= state["request"].max_search_rounds:
            return {"termination_reason": "max_search_rounds"}
        gaps = searchable_gaps(state["gap_ledger"])
        if not gaps:
            return {"termination_reason": "no_meaningful_gaps"}
        try:
            plan = self.services.follow_up_planner.plan(
                question=state["request"].question, gaps=gaps,
                executed_queries=state["executed_queries"], concepts=state["search_plan"].concepts,
                max_queries=state["request"].max_follow_up_queries,
            )
            queries = new_follow_up_queries(plan, gaps, state["executed_queries"],
                                           state["request"].max_follow_up_queries)
        except (OpenAIError, httpx.HTTPError, RuntimeError, ValueError) as exc:
            warnings.append(f"Follow-up planner failed ({type(exc).__name__}); retaining current verified evidence.")
            return dict(warnings=warnings, termination_reason="follow_up_planner_failed")
        updates = {"follow_up_plans": [*state["follow_up_plans"], plan]}
        if not plan.queries:
            return updates | {"termination_reason": "no_follow_up_queries"}
        if not queries:
            return updates | {"termination_reason": "duplicate_queries"}
        if len(queries) < len(plan.queries):
            warnings.append("Duplicate follow-up query texts were filtered before execution.")
        self.services._progress(f"[Round {state['search_round'] + 1}] {len(queries)} follow-up queries")
        return updates | dict(current_queries=queries, search_round=state["search_round"] + 1,
                              warnings=warnings, termination_reason=None)

    def finalize(self, state: ResearchState) -> dict:
        self.services._progress("Finalizing verified answer")
        stats = state["run_stats"].model_copy(deep=True)
        stats.elapsed_seconds = round(monotonic() - state["started_at"], 3)
        warnings = list(state["warnings"])
        reason = state["termination_reason"]
        if reason is None:
            raise RuntimeError("Graph reached finalize without an explicit termination reason.")
        assessment = state["evidence_assessment"]
        if reason != "sufficient_evidence":
            gaps = len(active_gaps(state["gap_ledger"])) if assessment else "unassessed"
            warnings.append(f"Research stopped: {reason}; remaining evidence gaps: {gaps}. Answer may be incomplete.")
        if state["draft"].limitations:
            warnings.append("Synthesis reported coverage limitations; see draft_limitations in the audit JSON.")
        if stats.unsupported:
            warnings.append(f"Removed {stats.unsupported} unsupported claim(s) from the final answer.")
        if stats.supported == 0:
            warnings.append("No fully supported claims; evidence is insufficient for a firm conclusion.")
        elif stats.supported * 2 < stats.claims_drafted:
            warnings.append("Most drafted claims were not fully supported; interpret this as a partial answer.")
        incomplete = (reason != "sufficient_evidence" or bool(state["draft"].limitations)
                      or any(c.verification.status != "supported" for c in state["claims"])
                      or bool(state["selection"] and state["selection"].insufficient_evidence))
        evidence = list(state["selected_evidence"].values())
        result = ResearchResult(
            question=state["request"].question,
            answer=render_answer(state["claims"], evidence, state["selected_papers"], incomplete=incomplete,
                                 gaps=active_gaps(state["gap_ledger"])),
            claims=state["claims"], evidence=evidence, papers=state["selected_papers"],
            search_plan=state["search_plan"], warnings=warnings, run_stats=stats,
            evidence_selection=state["selection"], retrieved_passages=list(state["retrieved_passages"].values()),
            draft_limitations=state["draft"].limitations, evidence_assessment=assessment,
            round_trace=state["round_trace"], executed_queries=state["executed_queries"],
            follow_up_plans=state["follow_up_plans"], termination_reason=reason, gap_ledger=state["gap_ledger"],
            rescue_trace=state["rescue_trace"], visual_evidence=list(state["visual_evidence"].values()),
            evidence_bundles=state["evidence_bundles"], assembly_visual_trace=state["assembly_visual_trace"],
            downgrade_audit=state["downgrade_audit"], atom_bundle_support=state["atom_bundle_support"],
        )
        return {"final_result": result, "warnings": warnings, "run_stats": stats}


def build_research_graph(services: "ResearchAgent", *, checkpointer: BaseCheckpointSaver | None = None):
    """Compile ordinary Python services into a bounded, synchronous graph.

    No checkpoint storage by default. An optional InMemorySaver works locally;
    persistent checkpointers can later be supplied through the same seam.
    Nodes return replacements without mutating prior checkpoint state.
    """
    nodes = _ResearchNodes(services)
    graph = StateGraph(ResearchState)
    stages = dict(initial_plan="planning", search_and_rank="searching", acquire_documents="acquiring",
                  retrieve_evidence="retrieving", synthesize_claims="synthesizing", verify_claims="verifying",
                  assess_evidence="assessing", assemble_evidence="rescuing", rescue_local_evidence="rescuing", plan_follow_up="follow_up", finalize="finalizing")

    def observed(node, stage):
        def execute(state: ResearchState) -> dict:
            services._on_progress(state_progress(stage, state))
            update = node(state)
            services._on_progress(state_progress(stage, state | update))
            return update
        return execute

    for name, stage in stages.items():
        graph.add_node(name, observed(getattr(nodes, name), stage))
    graph.add_edge(START, "initial_plan")
    graph.add_edge("initial_plan", "search_and_rank")
    for source, target in (("search_and_rank", "acquire_documents"),
                           ("acquire_documents", "retrieve_evidence"),
                           ("plan_follow_up", "search_and_rank")):
        graph.add_conditional_edges(source, lambda state: "stop" if state["termination_reason"] else "continue",
                                    {"stop": "finalize", "continue": target})
    graph.add_conditional_edges("retrieve_evidence", nodes.after_retrieval,
        {"rescue": "rescue_local_evidence", "stop": "finalize", "synthesize": "synthesize_claims"})
    graph.add_conditional_edges("assess_evidence", nodes.after_assessment,
        {"assemble": "assemble_evidence", "rescue": "rescue_local_evidence", "stop": "finalize", "follow_up": "plan_follow_up"})
    graph.add_conditional_edges("assemble_evidence", nodes.after_assembly,
        {"synthesize": "synthesize_claims", "rescue": "rescue_local_evidence", "stop": "finalize", "follow_up": "plan_follow_up"})
    graph.add_conditional_edges("rescue_local_evidence", nodes.after_rescue,
        {"synthesize": "synthesize_claims", "stop": "finalize", "follow_up": "plan_follow_up"})
    graph.add_edge("synthesize_claims", "verify_claims")
    graph.add_edge("verify_claims", "assess_evidence")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def create_memory_checkpointer() -> InMemorySaver:
    """Opt-in local checkpoints with an explicit deserialization allowlist.

    No pickle fallback, files, or external persistence service. The caller owns
    the saver lifetime; the normal CLI retains only the compact result trace.
    """
    models = (ResearchRequest, SearchPlan, SearchQuery, FollowUpSearchPlan, PaperCandidate,
              SelectedPaper, EvidencePassage, RetrievedPassage, EvidenceSelection,
              AnswerDraft, VerifiedClaim, EvidenceAssessment, EvidenceGapRecord, RunStats, RoundTrace, ResearchResult,
              RescueLimits, RescueTrace, VisualEvidenceRecord, RetrievalDiagnostics, EvidenceBundle, AssemblyLimits, DowngradeAudit)
    return InMemorySaver(serde=JsonPlusSerializer(
        allowed_msgpack_modules=[(model.__module__, model.__name__) for model in models],
    ))
