"""Per-run assembly integration, sharing the existing target-page vision budget."""

from time import monotonic

from researchpilot.claim_stability import assert_evidence_union
from researchpilot.passage_retrieval import _tokens

from researchpilot.bundle_support import bundle_gap_id, reconcile_bundle_gaps
from researchpilot.document_rescue import RescuePlan, corpus_fingerprint, formula_risk
from researchpilot.evidence_assembly import (
    AssemblyDraft, assembly_cache_key, finish_bundle, should_assemble,
)
from researchpilot.math_evidence import visual_passage
from researchpilot.evidence import RetrievedPassage
from researchpilot.research_iteration import EvidenceGap, active_gaps


def assembly_targets(state):
    """Only unresolved claims/core gaps can spend assembly quota."""
    if not state["request"].assembly_limits.enabled:
        return []
    evidence = state["selected_evidence"]
    corpus = list(state["passages"].values())
    jobs = []
    targets = [("claim:"+c.claim.claim_id, c.claim.text, [c.claim.claim_id], c.claim.evidence_ids)
               for c in state["claims"] if c.verification.status != "supported"]
    for gap in active_gaps(state["gap_ledger"]):
        if gap.relevance_to_question != "core" or gap.gap_id.startswith("assembly:"):
            continue
        ids = [key for c in state["claims"] if c.claim.claim_id in gap.related_claim_ids for key in c.claim.evidence_ids]
        targets.append((gap.gap_id, gap.description+" "+gap.search_focus, gap.related_claim_ids,
                        ids or list(evidence)))
    for context, target, claim_ids, ids in targets:
        anchors = [evidence[key] for key in dict.fromkeys(ids) if key in evidence and should_assemble(target, evidence[key])]
        anchors.sort(key=lambda p: -len(set(_tokens(target)) & set(_tokens(p.text))))
        for anchor in anchors:
            # New documents can change what is resolvable; cache is corpus-aware.
            key = assembly_cache_key(anchor, corpus, target, state["request"].assembly_limits)
            if key not in state["assembly_cache"]:
                jobs.append((context, target, claim_ids, anchor, key))
    return jobs


def assemble_selected(state: dict, services) -> dict:
    limits = state["request"].assembly_limits
    if not limits.enabled:
        return {}
    start = monotonic()
    corpus = list(state["passages"].values())
    evidence = dict(state["selected_evidence"])
    bundles = list(state["evidence_bundles"])
    cache = dict(state["assembly_cache"])
    contexts = dict(state["assembly_context_counts"])
    visual_history = list(state["assembly_visual_trace"])
    visuals = dict(state["visual_evidence"])
    warnings = list(state["warnings"])
    stats = state["run_stats"].model_copy(deep=True)
    vision_elapsed = 0.0
    for context, target, claim_ids, anchor, key in assembly_targets(state):
        if len(bundles) >= limits.max_bundles_per_run:
            break
        # Quota is per actual unresolved target and acquired corpus, not the
        # entire question forever. The global run cap remains independent.
        scope = context+":"+corpus_fingerprint(corpus)
        if contexts.get(scope, 0) >= limits.max_bundles_per_gap:
            continue
        if sum(b.paper_id == anchor.paper_id for b in bundles) >= limits.max_bundles_per_paper:
            continue
        rescuer = services.document_rescuer
        reserve = min(limits.max_visual_pages_per_bundle, max(0, limits.max_bundle_members-1)) if (
            rescuer.pdf_provider and rescuer.renderer and rescuer.page_client) else 0
        text_limits = limits.model_copy(update={"max_bundle_members": limits.max_bundle_members-reserve})
        draft = services.evidence_assembler.assemble(target, anchor, corpus, text_limits)
        draft.target_claim_ids = list(claim_ids)
        preliminary = finish_bundle(draft, max_members=limits.max_bundle_members)
        prior = next((b for b in bundles if b.anchor_evidence_id == anchor.passage_id
            and [m.passage_id for m in b.members if m.passage_id] == [p.passage_id for p in draft.passages]
            and b.required_components == preliminary.required_components
            and b.resolved_references == preliminary.resolved_references
            and b.dependencies == preliminary.dependencies), None)
        if prior is not None:
            shared = prior.model_copy(deep=True)
            shared.target_claim_ids = list(dict.fromkeys([*prior.target_claim_ids, *claim_ids]))
            shared.target_descriptions = list(dict.fromkeys([*prior.target_descriptions, target]))
            bundles[bundles.index(prior)] = shared
            cache[key] = prior.bundle_id
            continue
        gap = EvidenceGap(gap_id=bundle_gap_id(preliminary), description=target,
            search_focus=target, related_claim_ids=claim_ids, severity="critical")
        targets = [p for p in draft.passages if p.passage_id in draft.required_text
                   and formula_risk(draft.required_text[p.passage_id]).risky]
        # Connecting proof pages take priority over generic assumptions. Clean
        # prose is never offered just because it belongs to this bundle.
        proof_ids = {e.target_id for e in draft.dependencies if e.dependency_type in ("proof", "continuation")}
        targets.sort(key=lambda p: p.passage_id not in proof_ids)
        if reserve and targets:
            t0 = monotonic()
            visual_limits = state["request"].rescue_limits.model_copy(update={"max_visual_pages_per_gap": reserve})
            outcome = rescuer.rescue(state["request"].question,
                [RescuePlan(gap, [p for p in corpus if p.paper_id == anchor.paper_id], [], targets, [])],
                state["selected_papers"], visual_limits,
                [*state["rescue_trace"], *visual_history], state["search_round"], corpus_fingerprint(corpus),
                visual_only=True)
            vision_elapsed += monotonic() - t0
            visuals.update(outcome.visual_evidence)
            warnings.extend(outcome.warnings)
            if outcome.trace.visual_pages:
                outcome.trace.local_rescue_triggered = False
                visual_history.append(outcome.trace)
                stats.visual_pages_rendered += sum(p.rendered for p in outcome.trace.visual_pages)
                stats.visual_evidence_calls += sum(p.vision_called for p in outcome.trace.visual_pages)
                stats.assembly_model_calls += sum(p.vision_called for p in outcome.trace.visual_pages)
        visual_members = []
        for record in visuals.values():
            source = next((p for p in draft.passages if (p.paper_id, p.page_number) ==
                (record.evidence.paper_id, record.evidence.page_number)), None)
            if source is not None and record.evidence.relevant_to_gap:
                visual_members.append(visual_passage(record, source))
        bundle = finish_bundle(draft, visuals, visual_members, limits.max_bundle_members)
        bundle.target_descriptions = [target]
        admitted = {m.evidence_id for m in bundle.members}
        for p in [*draft.passages, *visual_members]:
            if p.passage_id in admitted:
                evidence.setdefault(p.passage_id, p)
        # Two targets can resolve to the same original members. Keep one bundle
        # identity while retaining the union of explicit target requirements.
        prior = next((b for b in bundles if b.bundle_id == bundle.bundle_id), None)
        if prior is None:
            bundles.append(bundle)
        else:
            merged = prior.model_copy(deep=True)
            merged.target_claim_ids = list(dict.fromkeys([*prior.target_claim_ids, *claim_ids]))
            merged.required_components.update(bundle.required_components)
            merged.missing_components = list(dict.fromkeys([*prior.missing_components, *bundle.missing_components]))
            merged.complete = not merged.missing_components
            bundles[bundles.index(prior)] = merged
        cache[key] = bundle.bundle_id
        contexts[scope] = contexts.get(scope, 0) + 1
        services._progress(f"[Round {state['search_round']}] Assembled {len(bundle.members)} evidence members; "
                           f"{'complete' if bundle.complete else 'incomplete'}: {anchor.title}")
    assert_evidence_union(state["selected_evidence"], evidence)
    stats.bundles_created = len(bundles)
    stats.bundle_members = sum(len(b.members) for b in bundles)
    stats.assembly_local_seconds += max(0.0, monotonic() - start - vision_elapsed)
    stats.evidence_selected = len(evidence)
    retrieved = dict(state.get("retrieved_passages", {}))
    for passage in evidence.values():
        retrieved.setdefault(passage.passage_id, RetrievedPassage(passage=passage, lexical_score=0.0))
    stats.passages_retrieved = len(retrieved)
    return dict(evidence_bundles=bundles, assembly_cache=cache, assembly_context_counts=contexts,
        assembly_visual_trace=visual_history, visual_evidence=visuals, selected_evidence=evidence,
        retrieved_passages=retrieved, warnings=warnings, run_stats=stats,
        gap_ledger=reconcile_bundle_gaps(state["gap_ledger"], bundles, state["search_round"], state["claims"]))
