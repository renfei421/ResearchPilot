from unittest.mock import Mock

from phase5_helpers import OfflineTest, page, paper
from test_research_graph import loop_fixture
from test_openai_page_evidence import gap, math_item
from test_pdf_page_renderer import renderable_pdf
from researchpilot.document_rescue import DocumentRescuer, RescueLimits
from researchpilot.evidence import AcquiredDocument, AnswerClaim, AnswerDraft, AtomicAssertion, ClaimVerification, EvidenceSelection
from researchpilot.pdf_page_renderer import PDFPageRenderer
from researchpilot.research_graph import build_research_graph, create_memory_checkpointer
from researchpilot.research_iteration import EvidenceAssessment, GapUpdate, FollowUpSearchPlan, FollowUpQuery
from researchpilot.research_models import ResearchRequest as _ResearchRequest, ResearchResult
from researchpilot.evidence_bundle import AssemblyLimits


def ResearchRequest(**kwargs):
    """Isolate existing rescue/hybrid contracts from the new assembly stage."""
    kwargs.setdefault("assembly_limits", AssemblyLimits(enabled=False))
    return _ResearchRequest(**kwargs)



QUESTION = "How do spectral observations control local curvature?"
INTRO = "Spectral observations control local curvature in fixed observation settings."
HIDDEN = "Proposition 4.2. Under incoherence, restricted convexity holds locally with a bound ≤ √d - 1."
QUALIFIED = "Under incoherence, the proposition establishes positive curvature only in the local neighborhood."


class RescueGraphTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.f = f = loop_fixture()
        f.a = paper(title="Spectral observations and local curvature", abstract=INTRO)
        f.b = paper("W2", title="Additional curvature conditions", abstract=INTRO)
        f.raw.search_works.side_effect = lambda **kwargs: [f.a]
        f.planner.plan.return_value = f.planner.plan.return_value.model_copy(update={
            "research_question": QUESTION, "concepts": ["spectral observations"]})
        f.fetcher.acquire.side_effect = lambda p: AcquiredDocument(pages=[page(p, 1, INTRO), page(p, 8, HIDDEN)])
        self.root = self.temporary_directory()
        self.pdf = self.root / "source.pdf"
        self.pdf.write_bytes(renderable_pdf([INTRO] * 7 + ["Proposition 4.2: Under A, local curvature is positive."]))
        self.visual = Mock(model="fake-vision")
        self.visual.extract.side_effect = lambda q, g, title, pg, eid: math_item(
            evidence_id=eid, paper_id=pg.paper_id, page_number=pg.page_number,
            statement_text=QUALIFIED, formula_text=None, assumptions=["Incoherence", "Local neighborhood"],
            ambiguity="Coefficient of the bound is unreadable; no exact coefficient established.")
        f.agent.document_rescuer = DocumentRescuer(pdf_provider=lambda p: self.pdf,
            renderer=PDFPageRenderer(self.root / "images"), page_client=self.visual)
        f.synthesizer.synthesize.side_effect = self.synthesize
        f.verifier.verify.side_effect = self.verify
        f.assessor.assess.side_effect = self.assess
        f.follow_up.plan.return_value = FollowUpSearchPlan(queries=[FollowUpQuery(query_id="extra",
            text="local curvature proposition assumptions", gap_id="curvature", rationale="Missing local conditions.")])
        self.request = ResearchRequest(question=QUESTION, top_passages=1, max_queries=3)

    def synthesize(self, question, evidence):
        visual = next((p for p in evidence if p.passage_id.startswith("visual:")), None)
        source = visual or evidence[0]
        return AnswerDraft(claims=[AnswerClaim(claim_id="c", text=QUALIFIED if visual else INTRO,
            evidence_ids=[source.passage_id])], limitations=[])

    def verify(self, claim, evidence):
        # The ambiguity concerns the numerical coefficient, not the visible
        # scope statement. No exact formula is claimed or repaired by this fake.
        status = "supported" if any(p.passage_id.startswith("visual:") for p in evidence) else "partially_supported"
        return ClaimVerification(claim_id=claim.claim_id, status=status, evidence_ids=claim.evidence_ids,
            reason="Only the visible, qualified scope is established.", assertions=[AtomicAssertion(
                assertion_id="scope", text=claim.text, kind="scope", status=status, evidence_ids=claim.evidence_ids,
                reason="Visible local assumptions; no uncertain formula claimed.",
                scope_supported=True, inference_supported=True, quantities_supported=True)])

    def assess(self, question, claims, evidence, prior_gaps=()):
        found = [p.passage_id for p in evidence if p.passage_id.startswith("visual:")]
        return EvidenceAssessment(sufficient=bool(found), gaps=[] if prior_gaps or found else [gap(
            description="Need the proposition's local restricted-convexity assumptions; no exact constant required.",
            search_focus="restricted convexity incoherence local proposition")], rationale="Assess visible local assumptions.",
            gap_updates=[GapUpdate(gap_id=g.gap_id, status="resolved" if found else "unresolved",
                reason="Local assumptions established." if found else "Still need local assumptions.",
                evidence_ids=found, related_claim_ids=["c"], superseded_by=None) for g in prior_gaps])

    def run_agent(self, **changes):
        return self.f.agent.run(ResearchRequest(**(self.request.model_dump() | changes)))

    def test_generic_math_end_to_end_with_ambiguity_and_real_page_citation(self):
        result = self.run_agent()
        self.assertEqual(result.termination_reason, "sufficient_evidence")
        self.assertEqual(result.run_stats.search_rounds, 1)
        self.assertEqual(result.run_stats.synthesis_rounds, 2)
        self.assertEqual(result.run_stats.visual_evidence_calls, 1)
        self.assertIn(QUALIFIED, result.answer)
        self.assertIn("[P1, p. 8]", result.answer)
        self.assertNotIn("√d", result.answer)
        self.assertIsNotNone(result.visual_evidence[0].evidence.ambiguity)
        self.assertEqual(result.gap_ledger[0].status, "resolved")
        self.f.follow_up.plan.assert_not_called()

    def test_rescue_occurs_before_follow_up_and_reenters_synthesis_verification(self):
        events = list(build_research_graph(self.f.agent).stream({"request": self.request},
            config={"recursion_limit": 40}, stream_mode="updates"))
        names = [next(iter(e)) for e in events]
        position = names.index("rescue_local_evidence")
        self.assertEqual(names[position - 1:position + 4], ["assess_evidence", "rescue_local_evidence",
            "synthesize_claims", "verify_claims", "assess_evidence"])
        self.assertNotIn("plan_follow_up", names)

    def test_rescue_never_redownloads_paper(self):
        self.run_agent()
        self.f.fetcher.acquire.assert_called_once()
        self.assertEqual(self.f.raw.search_works.call_count, 3)

    def test_visual_evidence_is_reverified(self):
        self.run_agent()
        cited = self.f.verifier.verify.call_args_list[-1].args[1]
        self.assertTrue(any(p.passage_id.startswith("visual:") for p in cited))
        self.assertTrue(any(not p.passage_id.startswith("visual:") for p in cited))
        self.assertTrue(any("AMBIGUITY" in p.text for p in cited))

    def test_round_and_gap_trace_rescue_outcome(self):
        result = self.run_agent()
        trace, ledger = result.rescue_trace[0], result.gap_ledger[0]
        self.assertTrue(result.round_trace[0].local_rescue_triggered)
        self.assertEqual(trace.gaps_targeted, ["curvature"])
        self.assertEqual(trace.gaps_resolved, ["curvature"])
        self.assertTrue(ledger.rescue_attempted)
        self.assertEqual(ledger.rescue_round, 1)
        self.assertEqual(ledger.rescue_result, "resolved")
        self.assertTrue(ledger.rescued_evidence_ids)
        self.assertEqual(trace.visual_pages[0].page_number, 8)

    def test_no_gap_does_not_trigger(self):
        original = self.f.verifier.verify.side_effect
        def supported(c, ps):
            v=original(c,ps);v.status="supported";v.assertions[0].status="supported";return v
        self.f.verifier.verify.side_effect=supported
        self.f.assessor.assess.side_effect = lambda *args, **kw: EvidenceAssessment(sufficient=True, gaps=[], rationale="Already sufficient.")
        result = self.run_agent()
        self.assertEqual(result.rescue_trace, [])
        self.visual.extract.assert_not_called()

    def test_unsuccessful_rescue_allows_external_follow_up(self):
        original = self.f.selector.select.side_effect
        def select(question, passages):
            if any(p.page_number == 8 for p in passages):
                return EvidenceSelection(selected_passage_ids=[], coverage_notes="Not useful.", insufficient_evidence=True)
            return original(question, passages)
        self.f.selector.select.side_effect = select
        result = self.run_agent()
        self.f.follow_up.plan.assert_called_once()
        self.assertEqual(result.run_stats.search_rounds, 2)
        self.assertEqual(result.rescue_trace[0].outcome, "no_new_evidence")

    def test_vision_failure_preserves_gap_and_continues(self):
        self.visual.extract.side_effect = RuntimeError("vision unavailable")
        result = self.run_agent()
        self.f.follow_up.plan.assert_called_once()
        self.assertEqual(result.run_stats.local_rescue_passes, 1)
        self.assertEqual(self.visual.extract.call_count, 1)
        self.assertTrue(any("page rescue failed" in w for w in result.warnings))

    def test_same_corpus_not_rescued_infinitely(self):
        self.f.assessor.assess.side_effect = lambda q, c, e, prior_gaps=(): self.assess(q, c, [], prior_gaps)
        result = self.run_agent(max_search_rounds=3)
        self.assertEqual(result.run_stats.local_rescue_passes, 1)
        self.assertEqual(self.visual.extract.call_count, 1)

    def test_new_document_can_enable_later_rescue(self):
        other_pdf = self.root / "other.pdf"
        other_pdf.write_bytes(renderable_pdf([INTRO] * 7 + ["A different document: Proposition 4.2 under additional conditions."]))
        self.f.agent.document_rescuer.pdf_provider = lambda p: self.pdf if p.paper_id == self.f.a.paper_id else other_pdf
        initial_queries = {q.text for q in self.f.planner.plan.return_value.queries}
        self.f.raw.search_works.side_effect = lambda query, **kw: [self.f.a] if query in initial_queries else [self.f.a, self.f.b]
        self.f.assessor.assess.side_effect = lambda q, c, e, prior_gaps=(): self.assess(q, c, [], prior_gaps)
        # Always keep just the introductory normal passage; rescue has a new
        # document even when normal retrieval contributes no new evidence.
        self.f.selector.select.side_effect = lambda q, ps: EvidenceSelection(
            selected_passage_ids=[ps[0].passage_id] if len(ps) == 1 and ps[0].page_number == 1 else [],
            coverage_notes="Conservative selection.", insufficient_evidence=True)
        result = self.run_agent()
        self.assertEqual(result.run_stats.local_rescue_passes, 2)
        self.assertNotEqual(result.rescue_trace[0].corpus_fingerprint, result.rescue_trace[1].corpus_fingerprint)
        self.assertEqual(self.visual.extract.call_count, 2)

    def test_same_pdf_page_under_different_paper_ids_is_not_sent_twice(self):
        initial_queries = {q.text for q in self.f.planner.plan.return_value.queries}
        self.f.raw.search_works.side_effect = lambda query, **kw: [self.f.a] if query in initial_queries else [self.f.a, self.f.b]
        self.f.assessor.assess.side_effect = lambda q, c, e, prior_gaps=(): self.assess(q, c, [], prior_gaps)
        result = self.run_agent()
        self.assertEqual(result.run_stats.visual_evidence_calls, 1)

    def test_rescue_allowed_at_external_round_limit(self):
        result = self.run_agent(max_search_rounds=1)
        self.assertEqual(result.termination_reason, "sufficient_evidence")
        self.assertEqual(result.run_stats.local_rescue_passes, 1)

    def test_zero_rescue_budget(self):
        result = self.run_agent(rescue_limits=RescueLimits(max_local_rescue_passes=0))
        self.assertEqual(result.run_stats.local_rescue_passes, 0)
        self.f.follow_up.plan.assert_called_once()

    def test_local_checkpoint_and_result_roundtrip(self):
        self.f.agent._checkpointer = create_memory_checkpointer()
        result = self.run_agent()
        self.assertEqual(ResearchResult.model_validate_json(result.model_dump_json()), result)

    def test_exact_formula_atom_cannot_bypass_visual_ambiguity(self):
        original = self.f.verifier.verify.side_effect
        def verify(claim, evidence):
            result = original(claim, evidence)
            if any(p.passage_id.startswith("visual:") for p in evidence):
                result.assertions[0].kind = "quantitative"
                result.assertions[0].text = "Curvature >= 2."
                result.assertions[0].evidence_ids=[p.passage_id for p in evidence if p.passage_id.startswith("visual:")]
            return result
        self.f.verifier.verify.side_effect = verify
        self.f.assessor.assess.side_effect = lambda q, c, e, prior_gaps=(): self.assess(q, c, [], prior_gaps)
        result = self.run_agent(max_search_rounds=1)
        self.assertEqual(result.claims[0].verification.status, "partially_supported")
        self.assertNotIn("Curvature >= 2.", result.answer)

    def test_no_images_in_saved_result(self):
        result = self.run_agent()
        output = result.model_dump_json()
        self.assertNotIn("base64", output)
        self.assertNotIn("data:image", output)
        self.assertNotIn(str(self.pdf), output)

    def test_invalid_post_rescue_gap_resolution_keeps_ledger_and_follows_up(self):
        def assess(q, claims, evidence, prior_gaps=()):
            if prior_gaps:
                raise ValueError("A resolved gap must cite evidence used by a supported finding.")
            return self.assess(q, claims, [], prior_gaps)
        self.f.assessor.assess.side_effect = assess
        result = self.run_agent()
        self.f.follow_up.plan.assert_called_once()
        self.assertTrue(result.claims)
        self.assertEqual(result.gap_ledger[0].status, "unresolved")
        self.assertEqual(result.gap_ledger[0].rescue_result, "reassessment_failed")
        self.assertTrue(any("Rescue reassessment failed" in w for w in result.warnings))

    def test_invalid_post_rescue_assessment_at_budget_keeps_verified_partial_answer(self):
        self.f.assessor.assess.side_effect = lambda q, c, e, prior_gaps=(): (
            EvidenceAssessment(sufficient=True, gaps=[], rationale="Incorrectly sufficient.", gap_updates=[])
            if prior_gaps else self.assess(q, c, [], prior_gaps))
        result = self.run_agent(max_search_rounds=1)
        self.assertEqual(result.termination_reason, "max_search_rounds")
        self.assertEqual(result.gap_ledger[0].status, "unresolved")
        self.assertIn(QUALIFIED, result.answer)
        self.assertIsNone(result.evidence_assessment)
        self.f.follow_up.plan.assert_not_called()

    def test_post_rescue_gap_identity_collision_preserves_verified_result_and_ledger(self):
        def assess(q, claims, evidence, prior_gaps=()):
            if not prior_gaps:
                return self.assess(q, claims, [], prior_gaps)
            valid = self.assess(q, claims, evidence, prior_gaps)
            # Reference and resolution checks pass, but this attempted new gap
            # illegally reuses the old identity for different information.
            return EvidenceAssessment(sufficient=False, rationale="A further gap remains.",
                gaps=[gap(gap_id=prior_gaps[0].gap_id,
                    description="Different missing theorem conditions.", search_focus="different theorem")],
                gap_updates=valid.gap_updates)
        self.f.assessor.assess.side_effect = assess
        result = self.run_agent(max_search_rounds=1)
        self.assertEqual(result.termination_reason, "max_search_rounds")
        self.assertEqual(result.claims[0].verification.status, "supported")
        self.assertTrue(any(p.passage_id.startswith("visual:") for p in result.evidence))
        self.assertEqual(len(result.gap_ledger), 1)
        self.assertEqual(result.gap_ledger[0].description,
            "Need the proposition's local restricted-convexity assumptions; no exact constant required.")
        self.assertEqual(result.gap_ledger[0].status, "unresolved")
        self.assertEqual(result.gap_ledger[0].rescue_result, "reassessment_failed")
        self.assertIn(QUALIFIED, result.answer)
        self.assertIsNone(result.evidence_assessment)
        self.assertTrue(any("Rescue reassessment failed" in w for w in result.warnings))
        self.f.follow_up.plan.assert_not_called()

    def test_post_rescue_invalid_supersession_preserves_original_gap_and_continues(self):
        def assess(q, claims, evidence, prior_gaps=()):
            if not prior_gaps:
                return self.assess(q, claims, [], prior_gaps)
            # This has valid references but illegally replaces a core gap with
            # a peripheral one, the exact ledger guard hit by the live run.
            return EvidenceAssessment(sufficient=False, rationale="Replacement proposed.",
                gaps=[gap(gap_id="replacement", description="Peripheral question.",
                    relevance_to_question="peripheral")],
                gap_updates=[GapUpdate(gap_id=g.gap_id, status="superseded",
                    reason="Replace this gap.", evidence_ids=[], related_claim_ids=[],
                    superseded_by="replacement") for g in prior_gaps])
        self.f.assessor.assess.side_effect = assess
        result = self.run_agent()
        self.f.follow_up.plan.assert_called_once()
        self.assertEqual(result.run_stats.search_rounds, 2)
        self.assertEqual(result.claims[0].verification.status, "supported")
        self.assertEqual([g.gap_id for g in result.gap_ledger], ["curvature"])
        self.assertEqual(result.gap_ledger[0].relevance_to_question, "core")
        self.assertEqual(result.gap_ledger[0].status, "unresolved")
        self.assertEqual(result.gap_ledger[0].rescue_result, "reassessment_failed")
        self.assertTrue(any("Rescue reassessment failed" in w for w in result.warnings))
        self.assertIsNone(result.evidence_assessment)

    def test_initial_invalid_assessment_still_rejected(self):
        self.f.assessor.assess.side_effect = ValueError("invalid initial output")
        with self.assertRaisesRegex(ValueError, "initial"):
            self.run_agent()
