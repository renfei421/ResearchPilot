"""Production graph and adapter seams with local synthetic evidence only."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from phase5_helpers import OfflineTest, page, paper
from test_evidence_assembly import QUESTION, ASSUMPTION, SPECTRAL, THEOREM, chain, assemble, verdict
from test_research_graph import loop_fixture
from test_openai_page_evidence import math_item, visual_record
from test_pdf_page_renderer import renderable_pdf
from researchpilot.evidence import (AcquiredDocument, AnswerClaim, AnswerDraft, EvidenceSelection,
    ClaimVerification, AtomicAssertion, VerifiedClaim, passage_index)
from researchpilot.evidence_assembly import finish_bundle, EvidenceAssembler
from researchpilot.evidence_bundle import AssemblyLimits
from researchpilot.bundle_runtime import assemble_selected
from researchpilot.document_rescue import DocumentRescuer, RescueLimits
from researchpilot.pdf_page_renderer import PDFPageRenderer
from researchpilot.research_models import ResearchRequest, ResearchResult, RunStats, SelectedPaper
from researchpilot.research_graph import _ResearchNodes, create_memory_checkpointer
from researchpilot.research_iteration import EvidenceAssessment, EvidenceGapRecord, GapUpdate, FollowUpSearchPlan
from researchpilot.openai_evidence_client import OpenAIClaimSynthesizer, OpenAIClaimVerifier, _VERIFICATION
from researchpilot.bundle_support import bundle_gap_id


class BundleGraphTests(OfflineTest):
    def fixture(self, include_theorem=True):
        f=loop_fixture();f.agent.document_rescuer=DocumentRescuer()
        f.a=paper(title="A synthetic graph curvature theorem")
        f.raw.search_works.side_effect=lambda **kw:[f.a]
        f.planner.plan.return_value=f.planner.plan.return_value.model_copy(update={"research_question":QUESTION})
        texts=[(5,ASSUMPTION),(6,SPECTRAL)]+([(7,THEOREM)] if include_theorem else [])
        f.fetcher.acquire.side_effect=lambda p: AcquiredDocument(pages=[page(p,n,t) for n,t in texts])
        f.selector.select.side_effect=lambda q,ps: EvidenceSelection(
            selected_passage_ids=[p.passage_id for p in ps if p.page_number==6],
            coverage_notes="Only the proposition anchor retrieved.",insufficient_evidence=False)
        f.synthesizer.synthesize.side_effect=lambda q,ps: AnswerDraft(claims=[AnswerClaim(claim_id="c",
            text="Under Assumption A1 and Proposition 2, restricted curvature holds locally.",
            evidence_ids=[p.passage_id for p in ps])],limitations=[])
        def verify(c,ps):
            v=verdict([p.passage_id for p in ps]);v.claim_id=c.claim_id
            if not any(p.page_number==7 for p in ps):
                v.status="partially_supported";v.assertions[0].status="partially_supported"
                v.assertions[0].inference_supported=False
            return v
        f.verifier.verify.side_effect=verify
        f.assessor.assess.side_effect=lambda q,cs,ps,prior_gaps=(): EvidenceAssessment(
            sufficient=all(c.verification.status=="supported" for c in cs),gaps=[],rationale="Check explicit connecting result.",
            gap_updates=[GapUpdate(gap_id=g.gap_id,status="resolved" if any(c.verification.status=="supported" for c in cs) else "unresolved",
                reason="Scoped chain verified." if include_theorem else "Connecting theorem is absent.",
                evidence_ids=[p.passage_id for p in ps],related_claim_ids=["c"],superseded_by=None) for g in prior_gaps])
        f.follow_up.plan.return_value=FollowUpSearchPlan(queries=[])
        return f

    def run_fixture(self,f):
        return f.agent.run(ResearchRequest(question=QUESTION,max_queries=3,rescue_limits=RescueLimits(max_local_rescue_passes=0)))

    def test_page6_anchor_recovers_pages5_and7_without_external_search(self):
        f=self.fixture();r=self.run_fixture(f)
        self.assertEqual({m.page_number for m in r.evidence_bundles[0].members},{5,6,7})
        self.assertTrue(r.evidence_bundles[0].complete)
        self.assertEqual(r.claims[0].verification.status,"supported")
        self.assertEqual(r.termination_reason,"sufficient_evidence")
        self.assertEqual(r.run_stats.search_rounds,1)
        f.follow_up.plan.assert_not_called()
        for n in (5,6,7): self.assertIn(f"[P1, p. {n}]",r.answer)
        self.assertEqual(ResearchResult.model_validate_json(r.model_dump_json()),r)

    def test_missing_page7_preserves_gap_and_partial_status(self):
        r=self.run_fixture(self.fixture(False))
        self.assertFalse(r.evidence_bundles[0].complete)
        self.assertEqual(r.claims[0].verification.status,"partially_supported")
        self.assertTrue(any(g.status=="unresolved" and "curvature" in g.description for g in r.gap_ledger))
        self.assertIn("Incomplete mathematical",r.answer)

    def test_existing_chain_gap_resolved_only_after_joint_verification(self):
        f=self.fixture(); original=_ResearchNodes.initial_plan
        def seed(node,state):
            data=original(node,state)
            data['gap_ledger']=[EvidenceGapRecord(gap_id=bundle_gap_id(assemble()),
                description="Need the curvature connecting theorem.",search_focus=QUESTION,
                related_claim_ids=[],severity="critical",first_seen_round=1,last_updated_round=1)]
            return data
        with patch.object(_ResearchNodes,"initial_plan",seed):r=self.run_fixture(f)
        self.assertEqual(r.gap_ledger[0].status,"resolved")
        f.follow_up.plan.assert_not_called()

    def test_checkpoint_roundtrip_includes_bundle_models(self):
        f=self.fixture();f.agent._checkpointer=create_memory_checkpointer()
        self.assertTrue(self.run_fixture(f).evidence_bundles[0].complete)


class BundleRuntimeTests(OfflineTest):
    def fixture(self, risky=False):
        c=chain()
        if risky:c[1]=c[1].model_copy(update={"text":SPECTRAL+" x = (a + b"})
        s=dict(request=ResearchRequest(question=QUESTION),passages=passage_index(c),selected_evidence={"p":c[1]},
            evidence_bundles=[],assembly_cache={},assembly_context_counts={},assembly_visual_trace=[],
            rescue_trace=[],visual_evidence={},warnings=[],run_stats=RunStats(),claims=[],gap_ledger=[],search_round=1,
            selected_papers=[SelectedPaper(paper=paper(),citation_label="P1",group_id="g",source_type="pdf")])
        v=verdict(["p"]);v.status="partially_supported";v.assertions[0].status="partially_supported"
        s['claims']=[VerifiedClaim(claim=AnswerClaim(claim_id="c",text=QUESTION,evidence_ids=["p"]),verification=v)]
        services=SimpleNamespace(document_rescuer=DocumentRescuer(),evidence_assembler=EvidenceAssembler(),_progress=lambda s:None)
        return s,services

    def test_cache_avoids_reassembly(self):
        s,f=self.fixture();f.evidence_assembler=Mock(wraps=EvidenceAssembler())
        s.update(assemble_selected(s,f));s.update(assemble_selected(s,f))
        self.assertEqual(f.evidence_assembler.assemble.call_count,1)
        self.assertEqual(s['run_stats'].bundles_created,1)

    def test_context_and_run_caps(self):
        s,f=self.fixture();s['request'].assembly_limits.max_bundles_per_run=1
        s.update(assemble_selected(s,f));self.assertEqual(len(s['evidence_bundles']),1)

    def test_formula_risk_reuses_existing_page_vision(self):
        s,f=self.fixture(True);root=self.temporary_directory();pdf=root/'paper.pdf'
        pdf.write_bytes(renderable_pdf(tuple("PAGE "+str(i) for i in range(1,8))))
        vision=Mock();vision.model="fake"
        vision.extract.side_effect=lambda q,g,t,p,e: math_item(evidence_id=e,paper_id=p.paper_id,page_number=p.page_number,
            ambiguity="Formula continues outside this page.")
        f.document_rescuer=DocumentRescuer(pdf_provider=lambda p:pdf,renderer=PDFPageRenderer(root/'images'),page_client=vision)
        update=assemble_selected(s,f);s.update(update)
        self.assertEqual(vision.extract.call_count,1)
        self.assertEqual(s['run_stats'].assembly_model_calls,1)
        b=s['evidence_bundles'][0];self.assertFalse(b.complete)
        self.assertTrue(any(m.math_evidence_id and m.page_number==6 for m in b.members))
        self.assertTrue(any('ambiguous visual' in m for m in b.missing_components))
        assemble_selected(s,f);self.assertEqual(vision.extract.call_count,1)

    def test_exhausted_shared_visual_budget_does_not_call_again(self):
        s,f=self.fixture(True)
        from researchpilot.document_rescue import RescueTrace,VisualPageTrace
        s['request'].rescue_limits.max_visual_pages_per_run=2
        s['rescue_trace']=[RescueTrace(round=1,corpus_fingerprint="x",visual_pages=[
            VisualPageTrace(paper_id='other',page_number=n,gap_id='g',risk_reasons=['formula'],vision_called=True) for n in (1,2)])]
        f.document_rescuer=DocumentRescuer(pdf_provider=Mock(),renderer=Mock(),page_client=Mock())
        assemble_selected(s,f);f.document_rescuer.page_client.extract.assert_not_called()

    def test_prose_zero_bundles_zero_model_calls(self):
        s,f=self.fixture();s['request'].question="What improves long document reasoning?";s['claims']=[]
        out=assemble_selected(s,f)
        self.assertEqual(out['evidence_bundles'],[])
        self.assertEqual(out['run_stats'].assembly_model_calls,0)


class BundleAdapterTests(OfflineTest):
    def test_synthesis_preserves_member_text_and_ids(self):
        client=OpenAIClaimSynthesizer();client._parse=Mock(return_value=AnswerDraft(claims=[],limitations=[]))
        client.synthesize_with_bundles(QUESTION,chain(),[assemble()])
        payload=client._parse.call_args.args[1]
        self.assertEqual(payload['evidence'],[p.model_dump() for p in chain()])
        self.assertEqual(payload['evidence_bundles'][0]['members'][0]['evidence_id'],'p')

    def test_verifier_only_sees_cited_members_and_retains_strict_prompt(self):
        client=OpenAIClaimVerifier();client._parse=Mock(return_value=SimpleNamespace(checked_verification=lambda claim:verdict(['p'])))
        c=AnswerClaim(claim_id='c',text='Spectral bound holds under A1.',evidence_ids=['p'])
        client.verify_with_bundles(c,chain(),[assemble()])
        instructions,payload,_=client._parse.call_args.args
        self.assertTrue(instructions.startswith(_VERIFICATION))
        self.assertEqual([p['passage_id'] for p in payload['evidence']],['E1'])
        self.assertEqual([m['evidence_id'] for m in payload['evidence_bundles'][0]['members']],['E1'])
        self.assertFalse(payload['evidence_bundles'][0]['complete'])
        self.assertTrue(payload['evidence_bundles'][0]['uncited_members_exist'])

    def test_synthesizer_rejects_forged_page_provenance(self):
        b=assemble();b.members[0].page_number=99
        client=OpenAIClaimSynthesizer();client._parse=Mock()
        with self.assertRaises(ValueError):client.synthesize_with_bundles(QUESTION,chain(),[b])
        client._parse.assert_not_called()
