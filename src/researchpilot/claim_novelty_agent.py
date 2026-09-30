"""Bounded sibling service; shares retrieval/evidence components, not answer semantics."""

from collections import Counter
from time import monotonic
from types import SimpleNamespace

import httpx
from openai import OpenAIError

from researchpilot.bundle_runtime import assemble_selected
from researchpilot.claim_check import (
    ClaimCheckRequest, ClaimCheckResult, ClaimCheckStats, ClaimDecomposition,
    ClaimSearchTrace, query_key,
)
from researchpilot.claim_recovery import (
    RecoveryDiagnostic, RecoveryPlan, recovery_query_key, MAX_RECOVERY_ROUNDS,
    MAX_QUERIES_PER_ROUND, MAX_QUERIES_PER_CLAIM, MAX_PAPERS_PER_CLAIM,
    MAX_ADDITIONAL_DOCUMENTS, MAX_ADDITIONAL_CANDIDATES,
)
from researchpilot.openai_claim_recovery import OpenAIClaimRecoveryPlanner
from researchpilot.paper_candidate import PaperCandidate
from researchpilot.claim_literature_map import build_maps, build_boundary, closest_papers
from researchpilot.document_rescue import RescueLimits, corpus_fingerprint
from researchpilot.evidence import RetrievalView
from researchpilot.evidence_bundle import AssemblyLimits
from researchpilot.openai_claim_check import (
    OpenAIClaimDecomposer, OpenAIClaimSearchPlanner, OpenAIClaimRelationAnalyzer, validate_batch,
)
from researchpilot.paper_candidate import SearchQuery
from researchpilot.paper_evidence_services import _select_papers, _acquire_passages
from researchpilot.paper_version_resolver import PaperVersionResolver
from researchpilot.progress import ResearchProgress
from researchpilot.research_agent import ResearchAgent
from researchpilot.research_graph import merge_candidates
from researchpilot.research_iteration import EvidenceGapRecord
from researchpilot.retrieval_diagnostics import is_transient
from researchpilot.rrf_ranker import RRFRanker
from researchpilot.project_reuse import seed_project

# Fixed internal budgets, deliberately not another large public configuration.
MAX_CANDIDATES = 100
MAX_SEMANTIC_GROUPS_PER_ROUND = 24
MAX_DOCUMENTS = 16
MAX_QUERIES = 40


class ClaimNoveltyAgent:
    def __init__(self, *, services=None, decomposer=None, search_planner=None,
                 relation_analyzer=None, recovery_planner=None, settings=None, openai_client=None,
                 model="gpt-5.6-terra", on_progress=None):
        # ResearchAgent also serves as the existing DI registry. Its run/graph,
        # answer synthesis, verifier and research prompts are never called here.
        self.services = services if services is not None else ResearchAgent(
            settings=settings, openai_client=openai_client, model=model)
        self.decomposer = decomposer or OpenAIClaimDecomposer(model=model, client=openai_client)
        self.search_planner = search_planner or OpenAIClaimSearchPlanner(model=model, client=openai_client)
        self.relation_analyzer = relation_analyzer or OpenAIClaimRelationAnalyzer(model=model, client=openai_client)
        self.recovery_planner = recovery_planner or OpenAIClaimRecoveryPlanner(model=model, client=openai_client)
        self.on_progress = on_progress or (lambda event: None)

    def run(self, request: ClaimCheckRequest) -> ClaimCheckResult:
        started = monotonic()
        stats, warnings = ClaimCheckStats(), []
        context = getattr(self, "project_context", None)
        if context is not None and context.project_id != request.project_id:
            raise ValueError("Project context must belong to the current request.")
        sources, corpus, visuals, project_gaps = seed_project(context, max_papers=min(4, request.max_papers_per_claim))
        relations, traces = {}, []
        candidates, memory, mappings = {}, {}, {}
        selected = {}
        if hasattr(self.search_planner, "project_context_payload"):
            self.search_planner.project_context_payload = context.planner_payload() if context else None
        assessed_inputs = {}
        limits = SimpleNamespace(question=request.claim, rescue_limits=RescueLimits(), assembly_limits=AssemblyLimits())
        state = dict(request=limits, selected_evidence=selected, passages=corpus, claims=[], gap_ledger=[],
            evidence_bundles=[], assembly_cache={}, assembly_context_counts={}, assembly_visual_trace=[],
            visual_evidence=visuals, rescue_trace=[], warnings=warnings, run_stats=stats,
            selected_papers=sources, retrieved_passages={})

        def progress(stage, round_number):
            self.on_progress(ResearchProgress(current_stage=stage, current_round=round_number,
                selected_papers=len(sources), evidence_collected=len(selected),
                elapsed_seconds=round(monotonic()-started, 3), warnings=list(warnings)))

        progress("planning", 1)
        decomposition = ClaimDecomposition.model_validate(self.decomposer.decompose(request))
        if decomposition.main_claim != request.claim:
            raise ValueError("Decomposition changed the original claim.")
        stats.assertions = len(decomposition.assertions)
        core_ids = [a.claim_id for a in decomposition.assertions if a.importance == "core"]
        targets, gaps, termination = core_ids, [g.description for g in project_gaps], "max_search_rounds"
        for cid in core_ids:
            mappings[cid] = [s.paper.paper_id for s in sources]
        maps = build_maps(decomposition, [], [])

        recovery_errors = []

        def analyze(target_ids=None):
            progress("verifying", stats.search_rounds)
            for source in sources:
                pid = source.paper.paper_id
                assertions = [a for a in decomposition.assertions if pid in mappings.get(a.claim_id, [])
                              and (target_ids is None or a.claim_id in target_ids)]
                offered = [p for p in selected.values() if p.paper_id == pid]
                bundles = [b for b in state["evidence_bundles"] if b.paper_id == pid]
                if not assertions or not offered:
                    continue
                fingerprint = (tuple(p.passage_id for p in offered),
                               tuple((b.bundle_id, b.complete) for b in bundles))
                if target_ids is not None:
                    assertions = [a for a in assertions if assessed_inputs.get((pid, a.claim_id)) != fingerprint]
                elif all(assessed_inputs.get((pid, a.claim_id)) == fingerprint for a in assertions):
                    continue
                if not assertions:
                    continue
                try:
                    batch = self.relation_analyzer.analyze(assertions, pid, offered, bundles)
                    if target_ids is not None:
                        # Revalidate injected/constructed objects before accepting a recovery update.
                        from researchpilot.claim_check import RelationBatch
                        batch = RelationBatch.model_validate(batch.model_dump())
                    validate_batch(batch, assertions, pid, offered)
                except (OpenAIError, httpx.HTTPError, RuntimeError, ValueError) as exc:
                    if target_ids is None:
                        raise
                    recovery_errors.append(type(exc).__name__)
                    warnings.append(f"Supplementary relation assessment unavailable ({type(exc).__name__}); earlier relations retained.")
                    continue
                stats.relation_calls += 1
                for relation in batch.relations:
                    relations[(relation.claim_id, pid)] = relation
                for assertion in assertions:
                    assessed_inputs[(pid, assertion.claim_id)] = fingerprint

        def evaluate_sources(fresh, round_number, target_ids=None):
            nonlocal selected, warnings, stats, maps
            progress("acquiring", round_number)
            for passage in _acquire_passages(self.services, fresh, stats, warnings):
                corpus.setdefault(passage.passage_id, passage)
            stats.selected_papers = len(sources)
            progress("retrieving", round_number)
            # Retrieve per assertion per acquired paper; union original immutable
            # passages before one paper-level relation call covering its assertions.
            for source in sources:
                pid = source.paper.paper_id
                document = [p for p in corpus.values() if p.paper_id == pid]
                if not document:
                    continue
                for assertion in decomposition.assertions:
                    if pid not in mappings.get(assertion.claim_id, []) or (target_ids is not None and assertion.claim_id not in target_ids):
                        continue
                    batch = self.services.passage_retriever.search_views(
                        [RetrievalView(view_id=assertion.claim_id, text=assertion.text)], document,
                        top_k=4, per_view_top_k=6)
                    stats.record_retrieval(batch.diagnostics)
                    warnings.extend(batch.warnings)
                    for hit in batch.passages:
                        selected.setdefault(hit.passage.passage_id, hit.passage)
            state.update(selected_evidence=selected, passages=corpus, warnings=warnings, run_stats=stats,
                         selected_papers=sources, search_round=round_number)
            state["gap_ledger"] = [EvidenceGapRecord(gap_id="idea:"+a.claim_id, description=a.text,
                search_focus=a.text, related_claim_ids=[a.claim_id], severity="useful",
                first_seen_round=round_number, last_updated_round=round_number)
                for a in decomposition.assertions if a.importance == "core"
                and (target_ids is None or a.claim_id in target_ids)]
            progress("selecting", round_number)
            state.update(assemble_selected(state, self.services))
            selected, warnings, stats = state["selected_evidence"], state["warnings"], state["run_stats"]
            analyze(target_ids)
            maps = build_maps(decomposition, list(relations.values()), traces)
            weak = {m.claim_id for m in maps if m.coverage_status != "strong"
                    and (target_ids is None or m.claim_id in target_ids)}
            if weak and len(state["rescue_trace"]) < limits.rescue_limits.max_local_rescue_passes:
                progress("rescuing", round_number)
                rescue_gaps = [g for g in state["gap_ledger"] if not g.gap_id.startswith("assembly:")
                               and any(cid in weak for cid in g.related_claim_ids)]
                plans_local = self.services.document_rescuer.plan(request.claim, rescue_gaps, sources,
                    list(corpus.values()), selected, [c for a in decomposition.assertions for c in a.concepts], [], limits.rescue_limits)
                if plans_local:
                    outcome = self.services.document_rescuer.rescue(request.claim, plans_local, sources,
                        limits.rescue_limits, [*state["rescue_trace"], *state["assembly_visual_trace"]],
                        round_number, corpus_fingerprint(list(corpus.values())))
                    new_ids = []
                    for passage in outcome.passages:
                        if passage.passage_id not in selected:
                            selected[passage.passage_id] = passage
                            new_ids.append(passage.passage_id)
                    outcome.trace.added_evidence_ids = new_ids
                    outcome.trace.outcome = "evidence_added" if new_ids else "no_new_evidence"
                    state["rescue_trace"].append(outcome.trace)
                    state["visual_evidence"].update(outcome.visual_evidence)
                    warnings.extend(outcome.warnings)
                    for diagnostic in outcome.trace.passage_retrieval:
                        stats.record_retrieval(diagnostic)
                    stats.local_rescue_passes += 1
                    stats.visual_evidence_calls += sum(p.vision_called for p in outcome.trace.visual_pages)
                    stats.visual_pages_rendered += sum(p.rendered for p in outcome.trace.visual_pages)
                    if new_ids:
                        analyze(target_ids)

        for round_number in range(1, request.max_search_rounds+1):
            stats.search_rounds = round_number
            progress("planning" if round_number == 1 else "follow_up", round_number)
            plans = self.search_planner.plan(request, decomposition, targets,
                list(dict.fromkeys([q.query for q in context.previous_queries] + [record[0].text for record in memory.values()]))
                if context else [record[0].text for record in memory.values()], gaps)
            plans.validate_targets(targets)
            stats.planned_queries += sum(len(plan.queries) for plan in plans.plans)
            before_candidates = set(candidates)
            old_evidence_pairs = {(r.claim_id, r.paper_id, key) for r in relations.values()
                                  if r.relation != "ADJACENT" for key in r.evidence_ids}
            new_queries = 0
            progress("searching", round_number)
            for plan in plans.plans:
                for query in plan.queries:
                    key = query_key(query.text)
                    reused = key in memory
                    if not reused:
                        if len(memory) >= MAX_QUERIES:
                            raise ValueError("Claim-search query budget exceeded.")
                        search = SearchQuery(query_id=f"q{len(memory)+1}", text=query.text)
                        error = None
                        try:
                            found = self.services.paper_search.search_candidates([search], per_query=10,
                                year_from=request.year_from, year_to=request.year_to)
                        except httpx.HTTPError as exc:
                            if not is_transient(exc):
                                raise
                            found, error = [], f"Literature query failed ({type(exc).__name__}); coverage is incomplete."
                            warnings.append(error)
                        memory[key] = (search, found, error)
                        new_queries += 1
                    search, found, error = memory[key]
                    # Hard cap the union, while recording the entire returned IDs
                    # in the trace so bounded selection is not mistaken for absence.
                    accepted = [c for c in found if c.paper.paper_id in candidates]
                    remaining = MAX_CANDIDATES-len(candidates)
                    accepted += [c for c in found if c.paper.paper_id not in candidates][:remaining]
                    candidates = merge_candidates(candidates, accepted)
                    if len(accepted) < len(found):
                        warning = "Global candidate cap reached; not all returned papers could be evaluated."
                        if warning not in warnings:
                            warnings.append(warning)
                    traces.append(ClaimSearchTrace(round=round_number, claim_id=plan.claim_id,
                        query_id=search.query_id, query=search.text, role=query.role, reused=reused,
                        status="failed" if error else "completed", candidate_ids=[c.paper.paper_id for c in found], limitation=error))
            if memory and all(record[2] for record in memory.values()):
                raise RuntimeError("All prior-work queries failed; no literature map can be reported.")
            if round_number > 1 and not new_queries:
                termination = "duplicate_queries"
                break
            if round_number > 1 and set(candidates) == before_candidates:
                termination = "no_new_papers"
                break
            stats.raw_candidates = len(candidates)
            stats.retrieved_query_hits = sum(len(c.hits) for c in candidates.values())
            if not candidates and not sources:
                termination = "no_literature_found"
                break

            grouped = PaperVersionResolver().group_versions(list(candidates.values()))
            ranked = RRFRanker().rank_groups(grouped)
            stats.version_groups = len(grouped)
            # Interleave claim-specific RRF lists before bounded semantic assessment,
            # so one prolific assertion cannot monopolize the candidate budget.
            by_claim = {}
            for cid in core_ids:
                qids = {t.query_id for t in traces if t.claim_id == cid}
                by_claim[cid] = [r for r in ranked if any(h.query_id in qids for h in r.fused_hits)]
            pool = []
            for i in range(max((len(v) for v in by_claim.values()), default=0)):
                for values in by_claim.values():
                    if i < len(values) and values[i] not in pool:
                        pool.append(values[i])
            pool = pool[:MAX_SEMANTIC_GROUPS_PER_ROUND]
            progress("reranking", round_number)
            known = {s.paper.paper_id for s in sources}
            pool = [g for g in pool if not any(m.paper.paper_id in known for m in g.group.members)]
            ordered = _select_papers(self.services, request.claim, pool, len(pool), stats, warnings) if pool else []
            ordered_ids = [s.group_id for s in ordered]
            group_papers = {s.group_id: s for s in ordered}
            lists = {cid: [gid for gid in ordered_ids if gid in {g.group.group_id for g in groups}]
                     for cid, groups in by_claim.items()}
            fresh = list(sources) if round_number == 1 else []
            source_ids = {s.paper.paper_id for s in sources}
            # Reserve four document slots for targeted follow-up on multi-claim runs.
            round_cap = 12 if round_number == 1 and request.max_search_rounds > 1 else MAX_DOCUMENTS
            per_claim_cap = max(1, request.max_papers_per_claim-2) if round_number == 1 and request.max_search_rounds > 1 else request.max_papers_per_claim
            for i in range(max((len(ids) for ids in lists.values()), default=0)):
                for cid, ids in lists.items():
                    if i >= len(ids):
                        continue
                    item = group_papers[ids[i]]
                    pid = item.paper.paper_id
                    assigned = mappings.setdefault(cid, [])
                    if pid not in assigned and len(assigned) < per_claim_cap:
                        if pid not in source_ids:
                            if len(sources) >= round_cap:
                                continue
                            item.citation_label = f"P{len(sources)+1}"
                            sources.append(item)
                            source_ids.add(pid)
                            fresh.append(item)
                        assigned.append(pid)
            # Supporting assertions share the acquired corpus; no extra search/download.
            for assertion in decomposition.assertions:
                if assertion.importance == "supporting":
                    mappings[assertion.claim_id] = [s.paper.paper_id for s in sources][:request.max_papers_per_claim]
            if not fresh and round_number > 1:
                termination = "no_new_papers"
                break
            evaluate_sources(fresh, round_number)
            progress("assessing", round_number)
            maps = build_maps(decomposition, list(relations.values()), traces)
            targets = [m.claim_id for m in maps if m.claim_id in core_ids and m.coverage_status != "strong"]
            if not targets:
                termination = "sufficient_coverage"
                break
            new_evidence_pairs = {(r.claim_id, r.paper_id, key) for r in relations.values()
                                 if r.relation != "ADJACENT" for key in r.evidence_ids}
            if round_number > 1 and not new_evidence_pairs-old_evidence_pairs:
                termination = "no_new_relation_evidence"
                break
            gaps = [m.claim_id+": "+"; ".join(m.search_limitations) for m in maps if m.claim_id in targets]
            gaps += [r.claim_id+": "+"; ".join(r.differences + r.scope_notes) for r in relations.values()
                     if r.claim_id in targets and r.relation != "ADJACENT"]

        # Supplementary recall is separate from the initial search budget. It is
        # entered once, only for weak core assertions, before building a boundary.
        recovery = []
        maps = build_maps(decomposition, list(relations.values()), traces)
        initial_source_count, initial_candidate_count = len(sources), len(candidates)
        prior_queries = [q.query for q in context.previous_queries] if context else []
        previous_texts = [*prior_queries, *[record[0].text for record in memory.values()]]
        seen_queries = {recovery_query_key(text) for text in previous_texts}
        if context and hasattr(self.services.paper_search, "previous_query_texts"):
            # Local duplicate checks need no extra Project contents in model input.
            seen_queries.update(recovery_query_key(text) for text in self.services.paper_search.previous_query_texts())

        recovery_targets = [a for a in decomposition.assertions if a.importance == "core"
                            and next(m.coverage_status for m in maps if m.claim_id == a.claim_id) == "weak"]
        for target_index, assertion in enumerate(recovery_targets):
            current = next(m for m in maps if m.claim_id == assertion.claim_id)
            cid = assertion.claim_id
            remaining_targets = len(recovery_targets) - target_index
            document_share = (initial_source_count + MAX_ADDITIONAL_DOCUMENTS - len(sources) + remaining_targets - 1) // remaining_targets
            candidate_share = (initial_candidate_count + MAX_ADDITIONAL_CANDIDATES - len(candidates) + remaining_targets - 1) // remaining_targets
            claim_candidate_start = len(candidates)
            diagnostic = RecoveryDiagnostic(claim_id=cid,
                trigger_reason="Initial core-claim coverage is weak: " + "; ".join(current.search_limitations))
            recovery.append(diagnostic)
            baseline_pairs = {(r.paper_id, key) for r in relations.values() if r.claim_id == cid
                              and r.relation != "ADJACENT" for key in r.evidence_ids}
            baseline_relations = {r.paper_id: r.model_dump() for r in relations.values() if r.claim_id == cid}
            for recovery_round in range(1, MAX_RECOVERY_ROUNDS+1):
                slots = min(MAX_PAPERS_PER_CLAIM, request.max_papers_per_claim, document_share) - len(diagnostic.new_selected_papers)
                slots = min(slots, initial_source_count + MAX_ADDITIONAL_DOCUMENTS - len(sources))
                if slots <= 0 or len(candidates) - claim_candidate_start >= candidate_share:
                    diagnostic.stop_reason = "budget_exhausted"
                    break
                diagnostic.rounds_used = recovery_round
                round_number = request.max_search_rounds + recovery_round
                progress("follow_up", round_number)
                missing = list(current.search_limitations) + [
                    "; ".join(r.differences + r.scope_notes) for r in relations.values() if r.claim_id == cid]
                try:
                    plan = self.recovery_planner.plan(request, assertion,
                        [a for a in decomposition.assertions if a.claim_id in assertion.depends_on],
                        previous_texts, missing, context.planner_payload() if context else None)
                    plan = RecoveryPlan.model_validate(plan.model_dump())
                    if plan.claim_id != cid:
                        raise ValueError("Recovery plan changed the target claim ID.")
                except (OpenAIError, httpx.HTTPError, RuntimeError, ValueError) as exc:
                    diagnostic.incomplete, diagnostic.stop_reason = True, "recovery_unavailable"
                    warnings.append(f"{cid}: supplementary query planning unavailable ({type(exc).__name__}).")
                    break
                diagnostic.generated_queries.extend(plan.queries)
                pending = []
                for query in plan.queries[:MAX_QUERIES_PER_ROUND]:
                    key = recovery_query_key(query.text)
                    if key in seen_queries or not key:
                        diagnostic.skipped_duplicate_queries.append(query.text)
                        continue
                    if len(diagnostic.recovery_queries) + len(pending) >= MAX_QUERIES_PER_CLAIM:
                        break
                    seen_queries.add(key)
                    previous_texts.append(query.text)
                    pending.append(query)
                if not pending:
                    diagnostic.stop_reason = "duplicate_queries"
                    break
                stats.planned_queries += len(pending)
                before_candidates = set(candidates)
                successful = 0
                progress("searching", round_number)
                for query in pending:
                    diagnostic.recovery_queries.append(query.text)
                    search = SearchQuery(query_id=f"recovery_{cid}_{len(diagnostic.recovery_queries)}", text=query.text)
                    error = None
                    try:
                        found = self.services.paper_search.search_candidates([search], per_query=10,
                            year_from=request.year_from, year_to=request.year_to)
                        successful += 1
                        diagnostic.successful_queries.append(query.text)
                    except (httpx.HTTPError, OpenAIError) as exc:
                        found = []
                        error = f"Supplementary query failed ({type(exc).__name__}); recovery is incomplete."
                        diagnostic.incomplete = True
                        warnings.append(error)
                    remaining = candidate_share - (len(candidates) - claim_candidate_start)
                    accepted = [c for c in found if c.paper.paper_id in candidates]
                    accepted += [c for c in found if c.paper.paper_id not in candidates][:remaining]
                    candidates = merge_candidates(candidates, accepted)
                    traces.append(ClaimSearchTrace(round=round_number, claim_id=cid,
                        query_id=search.query_id, query=search.text, role="recovery_"+query.role,
                        reused=False, status="failed" if error else "completed",
                        candidate_ids=[c.paper.paper_id for c in found], limitation=error))
                new_ids = set(candidates) - before_candidates
                diagnostic.new_candidate_papers.extend(pid for pid in candidates if pid in new_ids)
                if not successful:
                    diagnostic.stop_reason = "recovery_unavailable"
                    break
                # Seeded Project sources also participate in version resolution;
                # existing selected representatives always remain in the run.
                union = {s.paper.paper_id: PaperCandidate(paper=s.paper, hits=[]) for s in sources}
                union = merge_candidates(union, list(candidates.values()))
                grouped = PaperVersionResolver().group_versions(list(union.values()))
                known = {s.paper.paper_id for s in sources}
                fresh_groups = [g for g in grouped if any(m.paper.paper_id in new_ids for m in g.members)
                                and not any(m.paper.paper_id in known for m in g.members)]
                stats.raw_candidates = len(candidates)
                stats.retrieved_query_hits = sum(len(c.hits) for c in candidates.values())
                stats.version_groups = len(grouped)
                if not fresh_groups:
                    diagnostic.stop_reason = "no_new_papers"
                    break
                progress("reranking", round_number)
                fresh = _select_papers(self.services, assertion.text,
                    RRFRanker().rank_groups(fresh_groups)[:MAX_SEMANTIC_GROUPS_PER_ROUND], slots, stats, warnings)
                for item in fresh:
                    item.citation_label = f"P{len(sources)+1}"
                    sources.append(item)
                    diagnostic.new_selected_papers.append(item.paper.paper_id)
                mappings[cid] = list(dict.fromkeys([*mappings.get(cid, []), *[s.paper.paper_id for s in fresh]]))
                before_pairs = {(r.paper_id, key) for r in relations.values() if r.claim_id == cid
                                and r.relation != "ADJACENT" for key in r.evidence_ids}
                errors_before = len(recovery_errors)
                try:
                    evaluate_sources(fresh, round_number, {cid})
                except (httpx.HTTPError, OpenAIError) as exc:
                    diagnostic.incomplete = True
                    warnings.append(f"{cid}: supplementary evidence retrieval unavailable ({type(exc).__name__}).")
                diagnostic.incomplete |= len(recovery_errors) > errors_before
                maps = build_maps(decomposition, list(relations.values()), traces)
                current = next(m for m in maps if m.claim_id == cid)
                after_pairs = {(r.paper_id, key) for r in relations.values() if r.claim_id == cid
                               and r.relation != "ADJACENT" for key in r.evidence_ids}
                diagnostic.new_evidence_count = len({key for _, key in after_pairs - baseline_pairs})
                diagnostic.new_relations_count = sum(r.claim_id == cid and r.relation != "ADJACENT"
                    and baseline_relations.get(r.paper_id) != r.model_dump() for r in relations.values())
                if current.coverage_status != "weak":
                    diagnostic.stop_reason = "coverage_sufficient"
                    break
                if not after_pairs - before_pairs:
                    diagnostic.stop_reason = "no_new_substantive_evidence"
                    break
            maps = build_maps(decomposition, list(relations.values()), traces)
            diagnostic.coverage_after = next(m.coverage_status for m in maps if m.claim_id == cid)
            warnings.append(diagnostic.summary())

        progress("finalizing", stats.search_rounds)
        rows = list(relations.values())
        maps = build_maps(decomposition, rows, traces)
        stats.papers_evaluated = len({r.paper_id for r in rows})
        stats.relation_counts = dict(Counter(r.relation for r in rows))
        stats.queries_per_claim = {cid: len({t.query_id for t in traces if t.claim_id == cid}) for cid in core_ids}
        stats.evidence_selected = len(selected)
        stats.passages_created = len(corpus)
        stats.elapsed_seconds = round(monotonic()-started, 3)
        limitations = ["OpenAlex only; bounded queries, candidate selection and accessible PDFs may miss prior work.",
            "At most 100 candidate records, 24 semantic assessments per round, 16 documents per run and 8 papers per assertion.",
            "Relations are model assessments of cited passages, not exhaustive novelty or correctness proofs.",
            "Abstract-only findings cannot establish details absent from the abstract."]
        if recovery:
            limitations[1] = ("Initial search: at most 100 candidates, 16 documents and 8 papers per assertion. "
                "Core recovery: at most 2 rounds, 3 queries per round, 6 queries and 8 new papers per claim; "
                "shared additional caps of 100 candidates and 8 documents. Semantic assessment: at most 24 groups per round.")
        if any(not b.complete for b in state["evidence_bundles"]):
            limitations.append("Some formal evidence bundles remain incomplete; missing assumptions or formulas limit theoretical comparisons.")
        return ClaimCheckResult(request=request, decomposition=decomposition, claim_maps=maps, source_recovery=recovery,
            closest_prior_work=closest_papers(rows, list(selected.values())),
            novelty_boundary=build_boundary(decomposition, rows, maps), relations=rows,
            search_trace=traces, sources=sources, evidence=list(selected.values()),
            evidence_bundles=state["evidence_bundles"], visual_evidence=list(state["visual_evidence"].values()),
            rescue_trace=[*state["rescue_trace"], *state["assembly_visual_trace"]],
            warnings=list(dict.fromkeys(warnings)), limitations=limitations, run_stats=stats,
            termination_reason=termination)
