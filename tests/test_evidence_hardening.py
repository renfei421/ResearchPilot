"""Synthetic, offline regressions for evidence retrieval, atoms and gap memory."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

from phase5_helpers import OfflineTest, QUESTION, page, paper, passage
from researchpilot.answer_renderer import render_answer
from researchpilot.evidence import (
    AcquiredDocument, AnswerClaim, AnswerDraft, AtomicAssertion, ClaimVerification,
    EvidenceSelection, RetrievalView, VerifiedClaim, aggregate_assertions,
    validate_verification,
)
from researchpilot.openai_evidence_client import (
    OpenAIClaimVerifier, OpenAIEvidenceSelector, _SupportCheck, _VerificationAudit, _ClaimVerdict,
)
from researchpilot.openai_research_iteration import OpenAIEvidenceAssessor
from researchpilot.passage_retrieval import LexicalRetriever, PageChunker, _tokens, retrieval_views
from researchpilot.product_views import markdown_export, result_html, uncertainties
from researchpilot.research_iteration import (
    EvidenceAssessment, EvidenceGap, EvidenceGapRecord, GapUpdate, active_gaps,
    searchable_gaps, update_gap_ledger, validate_assessment, gap_uncertainty,
)
from researchpilot.research_models import ResearchRequest, SelectedPaper
from test_research_graph import loop_fixture, missing_b


def atom(key="a1", text="Latency fell in the measured experiment.", status="supported", **changes):
    return AtomicAssertion(**(dict(assertion_id=key, text=text, kind="factual", status=status,
        evidence_ids=["e1"], reason="Measured experiment only.", scope_supported=True,
        inference_supported=True, quantities_supported=True) | changes))


def audit(atoms):
    check = _SupportCheck(supported=True, reason="Applicable checks reflected in atoms.")
    aliased = [a.model_dump() | {"assertion_id": f"A{i}",
        "evidence_ids": ["E1" if key == "e1" else "E99" for key in a.evidence_ids]}
        for i, a in enumerate(atoms, 1)]
    return _VerificationAudit(assertions=aliased, scope=check, inference=check, quantities=check,
        verification=_ClaimVerdict(status="supported", reason="Per-atom assessment."))


def audit_claim():
    return AnswerClaim(claim_id="c1", text="Synthetic claim.", evidence_ids=["e1"])


def record(atoms):
    return VerifiedClaim(claim=AnswerClaim(claim_id="c1", text=" ".join(a.text for a in atoms), evidence_ids=["e1"]),
                         verification=audit(atoms).checked_verification(audit_claim()))


def gap_record(key="g1", **changes):
    return EvidenceGapRecord(**(dict(gap_id=key, description="Missing contention evidence.",
        search_focus="contention isolation", severity="critical", related_claim_ids=[],
        first_seen_round=1, last_updated_round=1) | changes))


def gap_update(status="resolved", key="g1", **changes):
    return GapUpdate(**(dict(gap_id=key, status=status, reason="New evidence addresses this setting.",
        evidence_ids=["e1"] if status in ("resolved", "partially_resolved") else [],
        related_claim_ids=[], superseded_by=None) | changes))


class MultiViewTests(OfflineTest):
    def test_original_question_is_first_view(self):
        self.assertEqual(retrieval_views(QUESTION, [], [])[0], RetrievalView(view_id="question", text=QUESTION))

    def test_independent_views_fuse_ranks_not_bm25_scores(self):
        ps = [passage("a", text="apples"), passage("b", text="bananas")]
        result = LexicalRetriever().retrieve_views([RetrievalView(view_id="a", text="apples"),
                                                  RetrievalView(view_id="b", text="bananas")], ps)
        self.assertEqual([r.passage.passage_id for r in result], ["a", "b"])
        self.assertAlmostEqual(result[0].fusion_score, 1 / 61)
        self.assertEqual(result[1].view_hits[0].view_id, "b")

    def test_duplicate_passage_is_once_with_both_view_hits(self):
        views = [RetrievalView(view_id="a", text="storage"), RetrievalView(view_id="b", text="latency")]
        result = LexicalRetriever().retrieve_views(views, [passage()])
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].view_hits), 2)
        self.assertAlmostEqual(result[0].fusion_score, 2 / 61)

    def test_deterministic_without_mutating_inputs(self):
        views = retrieval_views(QUESTION, ["capacity"], [])
        ps = [passage("b"), passage("a")]
        before = deepcopy((views, ps))
        a = LexicalRetriever().retrieve_views(views, ps)
        self.assertEqual(a, LexicalRetriever().retrieve_views(views, ps))
        self.assertEqual((views, ps), before)

    def test_concept_recovers_passage_missing_from_original_view(self):
        ps = [passage("a", text="storage latency"), passage("b", text="contention isolation guarantees")]
        self.assertEqual([r.passage.passage_id for r in LexicalRetriever().retrieve("storage latency", ps)], ["a"])
        result = LexicalRetriever().retrieve_views(retrieval_views("storage latency", ["contention isolation"], []), ps)
        self.assertEqual({r.passage.passage_id for r in result}, {"a", "b"})

    def test_gap_recovers_passage_missing_from_original_view(self):
        ps = [passage("a", text="storage latency"), passage("b", text="contention isolation guarantees")]
        views = retrieval_views("storage latency", [], [], [("Missing contention evidence", "isolation guarantees")])
        self.assertIn("b", [r.passage.passage_id for r in LexicalRetriever().retrieve_views(views, ps)])

    def test_hyphen_dash_and_non_prefix_variants(self):
        for a, b in (("low-rank", "low rank"), ("chain-of-thought", "chain of thought"),
                     ("non-uniform", "nonuniform"), ("non uniform", "nonuniform"),
                     ("second–singular–value", "second singular value"), ("rank-2", "rank 2")):
            with self.subTest(a=a):
                self.assertEqual(_tokens(a), _tokens(b))

    def test_unicode_normalization(self):
        self.assertEqual(_tokens("ＡＳＳＵＭＰＴＩＯＮ Ａ１ café"), _tokens("assumption A1 cafe\u0301"))

    def test_mathematical_labels_and_digits_are_not_lost(self):
        for query in ("A1", "Assumption A1", "A.1"):
            self.assertTrue(LexicalRetriever().retrieve(query, [passage(text="Assumption A.1 provides the rank-2 bound.")]))
        self.assertIn("2", _tokens("rank-2"))
        self.assertNotEqual(_tokens("A1"), _tokens("A2"))

    def test_structural_terms_remain_searchable_without_bonus(self):
        for term in ("theorem", "proposition", "lemma", "assumption", "corollary", "proof"):
            self.assertEqual(_tokens(term), [term])
        ps = [passage("a", text="Proposition curvature"), passage("b", text="Observed latency")]
        self.assertEqual([r.passage.passage_id for r in LexicalRetriever().retrieve("latency", ps)], ["b"])

    def test_concept_focus_uses_corpus_frequency_and_retains_origin(self):
        ps = [passage("a", text="general guarantees"), passage("b", text="curvature guarantees")]
        views = retrieval_views("guarantees", ["curvature guarantees"], [], passages=ps)
        focus = next(v for v in views if v.view_id.startswith("concept:"))
        self.assertEqual(focus.text, "curvature")
        self.assertEqual(focus.origin_text, "curvature guarantees")
        self.assertTrue(all(t in _tokens(focus.origin_text) for t in _tokens(focus.text)))

    def test_overlap_retains_boundary_statement_on_its_physical_page(self):
        statement = "Proposition 3. Under Assumption A1 and local initialization the curvature is positive."
        ps = PageChunker(400, 100).chunk([page(number=7, text="background " * 31 + statement + " details" * 60),
                                        page(number=8, text="An unrelated page with different results.")])
        self.assertTrue(any(statement in p.text and p.page_number == 7 for p in ps))
        self.assertFalse(any("unrelated" in p.text and "curvature" in p.text for p in ps))
        self.assertTrue(all(len(p.text) <= 480 for p in ps))

    def test_bounds_on_views_candidates_and_final_passages(self):
        views = retrieval_views(QUESTION, [f"concept{i}" for i in range(30)], [f"query{i}" for i in range(20)],
                                [(f"gap{i}", f"focus{i}") for i in range(8)])
        self.assertLessEqual(len(views), 8)
        ps = [passage(str(i)) for i in range(80)]
        result = LexicalRetriever().retrieve_views(views, ps, top_k=3, per_view_top_k=2)
        self.assertLessEqual(len(result), 3)
        self.assertTrue(all(h.rank <= 2 for r in result for h in r.view_hits))
        with self.assertRaises(ValueError):
            LexicalRetriever().retrieve_views(views * 2, ps)

    def test_no_retrieval_diagnostics_enter_selector_payload(self):
        sdk = Mock(); sdk.with_options.return_value = sdk
        sdk.responses.parse.return_value = SimpleNamespace(status="completed", output=[], output_parsed=
            EvidenceSelection(selected_passage_ids=["e1"], coverage_notes="Evidence only", insufficient_evidence=False))
        retrieved = LexicalRetriever().retrieve_views(retrieval_views(QUESTION, [], []), [passage()])
        OpenAIEvidenceSelector(client=sdk).select(QUESTION, [r.passage for r in retrieved])
        payload = json.loads(sdk.responses.parse.call_args.kwargs["input"])
        self.assertEqual(payload["passages"], [passage().model_dump()])
        for field in ("lexical_score", "fusion_score", "view_hits", "rank"):
            self.assertNotIn(field, payload["passages"][0])


class AtomicVerificationTests(OfflineTest):
    def test_single_supported_assertion(self):
        self.assertEqual(audit([atom()]).checked_verification(audit_claim()).status, "supported")

    def test_compound_requires_all_atoms_supported(self):
        self.assertEqual(aggregate_assertions([atom(), atom("a2")]), "supported")
        self.assertEqual(aggregate_assertions([atom(), atom("a2", status="unsupported")]), "partially_supported")

    def test_atom_conflict_cannot_be_hidden_by_negative_global_verdict(self):
        result = audit([atom(status="conflicting")])
        result.verification.status = "unsupported"
        self.assertEqual(result.checked_verification(audit_claim()).status, "conflicting")

    def test_inconsistent_partial_global_verdict_cannot_promote_atoms(self):
        result = audit([atom()])
        result.verification.status = "partially_supported"
        checked = result.checked_verification(audit_claim())
        self.assertEqual(checked.status, "partially_supported")
        self.assertFalse(any(a.status == "supported" for a in checked.assertions))

    def test_no_majority_vote_or_support_without_evidence(self):
        self.assertEqual(aggregate_assertions([atom(str(i)) for i in range(5)] + [atom("bad", status="unsupported")]), "partially_supported")
        self.assertEqual(aggregate_assertions([atom(evidence_ids=[])]), "unsupported")

    def test_causal_inference_requires_its_own_support(self):
        a = atom("cause", "Therefore caching caused the reduction.", kind="causal", inference_supported=False)
        self.assertEqual(a.status, "partially_supported")
        self.assertEqual(aggregate_assertions([atom(), a]), "partially_supported")

    def test_scope_expansion_one_family_or_task_is_not_universal(self):
        for text in ("Every model family improves.", "The method improves all tasks."):
            a = atom(text=text, kind="scope", scope_supported=False)
            self.assertNotEqual(a.status, "supported")

    def test_one_experiment_cannot_establish_universal_behavior(self):
        bad = atom("universal", "Latency always falls in every deployment.", kind="scope", scope_supported=False)
        self.assertEqual(record([atom(), bad]).verification.status, "partially_supported")

    def test_theorem_missing_assumptions_is_partial(self):
        a = atom(text="The curvature is positive unconditionally.", kind="theoretical", scope_supported=False)
        self.assertEqual(aggregate_assertions([a]), "partially_supported")

    def test_unsupported_quantity_does_not_survive(self):
        a = atom("number", "Latency fell by 70 percent.", kind="quantitative", quantities_supported=False)
        self.assertEqual(record([atom(), a]).verification.status, "partially_supported")
        self.assertNotIn("70 percent", self.render(record([atom(), a])))

    def test_exact_quantity_with_baseline_preserved(self):
        a = atom(text="In this trial, latency fell from 12 ms to 9 ms against the uncached baseline.", kind="quantitative")
        self.assertEqual(record([a]).verification.status, "supported")
        self.assertIn("12 ms to 9 ms", self.render(record([a])))

    def test_historical_comparison_cannot_imply_current_superiority(self):
        past = atom(text="Earlier solvers were slower in the reported comparison.", kind="historical")
        broad = atom("b", "The baseline remains stronger than current solvers.", kind="comparative", scope_supported=False)
        result = record([past, broad])
        self.assertEqual(result.verification.status, "partially_supported")
        rendered = self.render(result)
        self.assertIn(past.text, rendered)
        self.assertNotIn(broad.text, rendered)

    def test_conflict_is_not_rendered_as_established_fact(self):
        conflicted = atom("c", "System A dominates every alternative.", status="conflicting")
        text = self.render(record([atom(), conflicted]))
        self.assertIn("Unresolved disagreement", text)
        self.assertNotIn(conflicted.text, text)
        self.assertIn(atom().text, text)

    def test_no_unsupported_atom_survives_and_audit_not_mutated(self):
        r = record([atom(), atom("b", "An invented explanation.", status="unsupported")])
        before = deepcopy(r)
        text = self.render(r)
        self.assertIn("Supported portion only", text)
        self.assertNotIn("invented explanation", text)
        self.assertEqual(before, r)

    def test_atom_cannot_cite_an_uncited_passage(self):
        with self.assertRaises(ValueError):
            audit([atom(evidence_ids=["unknown"])]).checked_verification(audit_claim())

    def test_real_adapter_atomization_is_one_structured_call(self):
        sdk = Mock(); sdk.with_options.return_value = sdk
        sdk.responses.parse.return_value = SimpleNamespace(status="completed", output=[], output_parsed=audit([atom()]))
        result = OpenAIClaimVerifier(client=sdk).verify(record([atom()]).claim, [passage()])
        self.assertEqual(len(result.assertions), 1)
        self.assertEqual(sdk.responses.parse.call_count, 1)
        instructions = sdk.responses.parse.call_args.kwargs["instructions"]
        for term in ("ALL material", "sufficient is not necessary", "Historical comparisons", "denominators", "chain-of-thought"):
            self.assertIn(term, instructions)

    def test_verifier_explicitly_checks_quantity_identity_and_normalization(self):
        sdk = Mock(); sdk.with_options.return_value = sdk
        sdk.responses.parse.return_value = SimpleNamespace(status="completed", output=[], output_parsed=audit([atom()]))
        OpenAIClaimVerifier(client=sdk).verify(record([atom()]).claim, [passage()])
        instructions = sdk.responses.parse.call_args.kwargs["instructions"]
        for term in ("raw or rescaled weights", "normalized", "normalization factors",
                     "dimension/sample-size dependence", "definition or scale",
                     "quantities_supported false"):
            self.assertIn(term, instructions)

    def test_rescaled_weight_bound_cannot_render_as_normalized_probability_bound(self):
        evidence = passage(text="Weights satisfy w_i >= c and sum_i w_i = N; probabilities are p_i = w_i/N.")
        narrow = atom(text="The rescaled weights have lower bound c.", kind="quantitative")
        broad = atom("bad", "The normalized probabilities have dimension-independent lower bound c.",
                     kind="quantitative", quantities_supported=False,
                     reason="The source bounds w_i, not p_i; normalization by N cannot be omitted.")
        sdk = Mock(); sdk.with_options.return_value = sdk
        sdk.responses.parse.return_value = SimpleNamespace(status="completed", output=[], output_parsed=audit([narrow, broad]))
        claim = AnswerClaim(claim_id="c1", text=narrow.text + " " + broad.text, evidence_ids=["e1"])
        result = OpenAIClaimVerifier(client=sdk).verify(claim, [evidence])
        self.assertEqual(result.status, "partially_supported")
        rendered = render_answer([VerifiedClaim(claim=claim, verification=result)], [evidence],
            [SelectedPaper(citation_label="P1", group_id="g", paper=paper())])
        self.assertIn(narrow.text, rendered)
        self.assertNotIn(broad.text, rendered)

    def render(self, r):
        return render_answer([r], [passage()], [SelectedPaper(citation_label="P1", group_id="g", paper=paper())])


class GapLedgerTests(OfflineTest):
    def assessment(self, updates=(), gaps=()):
        return EvidenceAssessment(sufficient=False, gaps=list(gaps), gap_updates=list(updates), rationale="Coverage assessment.")

    def test_omitted_gap_carries_forward_without_implicit_resolution(self):
        before = [gap_record()]
        after = update_gap_ledger(before, self.assessment(), 2)
        self.assertEqual(after, before)
        self.assertIsNot(after[0], before[0])

    def test_resolved_gap_retains_identity_and_history(self):
        after = update_gap_ledger([gap_record()], self.assessment([gap_update()]), 2)
        self.assertEqual((after[0].status, after[0].first_seen_round, after[0].last_updated_round), ("resolved", 1, 2))
        self.assertEqual(active_gaps(after), [])

    def test_partial_resolution_remains_visible_and_searchable(self):
        after = update_gap_ledger([gap_record()], self.assessment([gap_update("partially_resolved")]), 2)
        self.assertEqual(searchable_gaps(after)[0].gap_id, "g1")
        self.assertEqual(active_gaps(after)[0].status, "partially_resolved")

    def test_partial_gap_displays_current_remaining_scope(self):
        g = gap_record(status="partially_resolved", resolution_reason="Latency is covered; throughput remains unmeasured.")
        self.assertIn(g.resolution_reason, gap_uncertainty(g))
        self.assertNotIn(g.description, gap_uncertainty(g))
        self.assertEqual(searchable_gaps([g])[0].description, g.resolution_reason)
        self.assertEqual(searchable_gaps([g])[0].gap_id, g.gap_id)

    def test_uncertainty_appears_once_in_each_product_presentation(self):
        f = loop_fixture()
        result = f.agent.run(ResearchRequest(question=QUESTION, max_queries=3, max_search_rounds=1))
        self.assertEqual(markdown_export(result).count("## Remaining uncertainty"), 1)
        self.assertEqual(result_html(result).count(result.gap_ledger[0].description), 1)

    def test_new_gap_and_exact_duplicate_stable_id(self):
        old = gap_record()
        data = old.model_dump(include=set(EvidenceGap.model_fields))
        duplicate = EvidenceGap(**(data | {"gap_id": "another-model-id"}))
        new = EvidenceGap(**(data | {"gap_id": "g2", "description": "Missing failure rates."}))
        after = update_gap_ledger([old], self.assessment(gaps=[duplicate, new]), 2)
        self.assertEqual([g.gap_id for g in after], ["g1", "g2"])
        self.assertEqual(after[1].first_seen_round, 2)

    def test_resolved_gaps_not_sent_to_planner(self):
        records = [gap_record(), gap_record("resolved", status="resolved"), gap_record("partial", status="partially_resolved")]
        self.assertEqual([g.gap_id for g in searchable_gaps(records)], ["g1", "partial"])

    def test_unknown_resolution_or_missing_prior_decision_rejected(self):
        with self.assertRaises(ValueError):
            update_gap_ledger([gap_record()], self.assessment([gap_update(key="unknown")]), 2)
        with self.assertRaisesRegex(ValueError, "Every prior"):
            validate_assessment(self.assessment(), [record([atom()])], [passage()], [gap_record()])

    def test_gap_identity_cannot_be_reused_for_new_information(self):
        new = EvidenceGap.model_validate(gap_record(description="Different missing information.").model_dump(include=set(EvidenceGap.model_fields)))
        with self.assertRaisesRegex(ValueError, "reuse"):
            update_gap_ledger([gap_record()], self.assessment(gaps=[new]), 2)

    def test_superseding_requires_active_replacement_with_same_importance(self):
        g = EvidenceGap.model_validate(gap_record("g2").model_dump(include=set(EvidenceGap.model_fields)))
        g.description = "A narrower formulation of the same missing evidence."
        result = update_gap_ledger([gap_record()], self.assessment(
            [gap_update("superseded", superseded_by="g2")], [g]), 2)
        self.assertEqual([x.gap_id for x in active_gaps(result)], ["g2"])
        for replacement in ("unknown", "g1"):
            with self.assertRaises(ValueError):
                update_gap_ledger([gap_record()], self.assessment([gap_update("superseded", superseded_by=replacement)]), 2)
        g.relevance_to_question = "peripheral"
        with self.assertRaisesRegex(ValueError, "importance"):
            update_gap_ledger([gap_record()], self.assessment([gap_update("superseded", superseded_by="g2")], [g]), 2)

    def test_resolution_requires_known_actual_evidence(self):
        for ids in ([], ["invented"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_assessment(self.assessment([gap_update(evidence_ids=ids)]), [], [passage()], [gap_record()])

    def test_rejected_claim_atoms_cannot_resolve_a_gap(self):
        checked = audit([atom()])
        checked.verification.status = "unsupported"
        rejected = VerifiedClaim(
            claim=AnswerClaim(claim_id="c1", text="Rejected overall.", evidence_ids=["e1"]),
            verification=checked.checked_verification(audit_claim()),
        )
        with self.assertRaisesRegex(ValueError, "supported finding"):
            validate_assessment(self.assessment([gap_update()]), [rejected], [passage()], [gap_record()])

    def test_assessor_receives_prior_ids_and_original_question(self):
        sdk = Mock(); sdk.with_options.return_value = sdk
        expected = self.assessment([gap_update("unresolved")])
        sdk.responses.parse.return_value = SimpleNamespace(status="completed", output=[], output_parsed=expected)
        OpenAIEvidenceAssessor(client=sdk).assess(QUESTION, [], [passage()], prior_gaps=[gap_record()])
        payload = json.loads(sdk.responses.parse.call_args.kwargs["input"])
        self.assertEqual(payload["prior_gaps"][0]["gap_id"], "g1")
        self.assertEqual(payload["research_question"], QUESTION)

    def test_peripheral_gap_does_not_force_another_round(self):
        f = loop_fixture()
        g = EvidenceGap.model_validate(gap_record(relevance_to_question="peripheral").model_dump(include=set(EvidenceGap.model_fields)))
        f.assessor.assess.side_effect = None
        f.assessor.assess.return_value = self.assessment(gaps=[g])
        result = f.agent.run(ResearchRequest(question=QUESTION, max_queries=3))
        self.assertEqual(result.run_stats.search_rounds, 1)
        self.assertEqual(result.termination_reason, "no_meaningful_gaps")
        f.follow_up.plan.assert_not_called()
        self.assertEqual(len(result.gap_ledger), 1)

    def test_max_round_and_final_uncertainty_keep_critical_gap(self):
        f = loop_fixture()
        result = f.agent.run(ResearchRequest(question=QUESTION, max_queries=3, max_search_rounds=1))
        self.assertEqual(result.termination_reason, "max_search_rounds")
        self.assertEqual(result.gap_ledger[0].status, "unresolved")
        self.assertIn("Remaining uncertainty", result.answer)
        self.assertTrue(any("critical / unresolved" in text for text in uncertainties(result)))

    def test_two_round_graph_carries_resolves_and_plans_by_stable_id(self):
        f = loop_fixture()
        result = f.agent.run(ResearchRequest(question=QUESTION, max_queries=3))
        self.assertEqual(result.gap_ledger[0].status, "resolved")
        self.assertEqual(result.round_trace[1].gap_ids_carried, ["contention"])
        self.assertEqual(result.round_trace[1].gap_ids_resolved, ["contention"])
        self.assertEqual(f.follow_up.plan.call_args.kwargs["gaps"][0].gap_id, "contention")


class EvidenceIntegrationTests(OfflineTest):
    def test_local_hidden_result_rescued_before_second_external_round(self):
        f = loop_fixture()
        f.follow_records = [f.a]  # HTTP found no new papers: local evidence must still run.
        hidden = "Lemma 4. Under isolation assumptions, contention is bounded by the partition capacity."
        f.fetcher.acquire.side_effect = lambda p: AcquiredDocument(pages=[
            page(p, 3, "Shared caches reduce storage latency for repeated reads."), page(p, 9, hidden)])
        f.planner.plan.return_value = f.planner.plan.return_value.model_copy(update={"concepts": ["storage latency"]})
        # Initial academic expressions deliberately omit the hidden terminology.
        for i, q in enumerate(f.planner.plan.return_value.queries):
            q.text = f"storage latency repeated reads measurement{i}"
        f.raw.search_works.side_effect = lambda **kwargs: [f.a]
        f.synthesizer.synthesize.side_effect = lambda q, evidence: AnswerDraft(claims=[
            AnswerClaim(claim_id=f"c{i}", text=p.text, evidence_ids=[p.passage_id])
            for i, p in enumerate(evidence)], limitations=[])
        f.verifier.verify.side_effect = lambda c, ps: ClaimVerification(claim_id=c.claim_id, status="supported",
            evidence_ids=c.evidence_ids, reason="Explicit result.", assertions=[atom(text=c.text, evidence_ids=c.evidence_ids)])
        def assess(question, claims, evidence, prior_gaps=()):
            if not any(p.page_number == 9 for p in evidence):
                return missing_b(prior_gaps)
            return EvidenceAssessment(sufficient=True, gaps=[], rationale="Both results present.", gap_updates=[
                gap_update(key=g.gap_id, evidence_ids=[p.passage_id for p in evidence if p.page_number == 9])
                for g in prior_gaps])
        f.assessor.assess.side_effect = assess
        result = f.agent.run(ResearchRequest(question=QUESTION, max_queries=3))
        self.assertEqual([r.new_evidence for r in result.round_trace], [2])
        self.assertEqual(result.run_stats.local_rescue_passes, 1)
        f.follow_up.plan.assert_not_called()
        self.assertEqual(f.fetcher.acquire.call_count, 1)
        self.assertEqual(result.gap_ledger[0].status, "resolved")
        self.assertIn(hidden, result.answer)
        self.assertIn("[P1, p. 9]", result.answer)
        self.assertIn("[P1, p. 3]", result.answer)
        self.assertTrue(all(c.verification.assertions for c in result.claims))

    def test_synthetic_graph_curvature_statement_missed_by_broad_query(self):
        # Mechanism regression only: no DOI, real-paper text, cache or benchmark.
        ps = [passage(f"intro{i}", text="Fixed observation graphs and second singular values permit matrix recovery. " * 4)
              for i in range(22)]
        ps.append(passage("local", number=13, text=
            "Proposition 1. Local curvature. Under Assumption A.1 and incoherent initialization, "
            "the loss has positive curvature in the stated neighborhood. Lemma 4 controls the mixing error."))
        question = "How do fixed observation graphs and second singular values affect matrix recovery?"
        self.assertNotIn("local", [r.passage.passage_id for r in LexicalRetriever().retrieve(question, ps, 20)])
        views = retrieval_views(question, ["local curvature", "Assumption A1"], [], passages=ps)
        result = LexicalRetriever().retrieve_views(views, ps, 20)
        recovered = next(r.passage for r in result if r.passage.passage_id == "local")
        self.assertEqual(recovered.page_number, 13)
        self.assertIn("Assumption A.1", recovered.text)
