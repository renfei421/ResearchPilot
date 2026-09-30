"""Phase 10.1 invariants, dependency closure, and generic regression classes."""

from types import SimpleNamespace
from unittest.mock import Mock

from phase5_helpers import OfflineTest, passage
from test_evidence_assembly import chain, verdict, QUESTION, ASSUMPTION, SPECTRAL, THEOREM
import test_bundle_integration as fixtures
from test_openai_page_evidence import math_item
from test_pdf_page_renderer import renderable_pdf
from researchpilot.claim_stability import assert_evidence_union, audit_downgrade, merge_targeted_draft, recheck_supported
from researchpilot.evidence import AnswerClaim, AnswerDraft, VerifiedClaim, passage_index
from researchpilot.evidence_assembly import EvidenceAssembler, finish_bundle, references
from researchpilot.evidence_bundle import AssemblyLimits, validate_bundle, bundle_identity
from researchpilot.statement_index import build_statement_index
from researchpilot.bundle_support import guard_bundle_support, reconcile_bundle_gaps
from researchpilot.bundle_runtime import assemble_selected, assembly_targets
from researchpilot.document_rescue import DocumentRescuer, RescueLimits
from researchpilot.pdf_page_renderer import PDFPageRenderer
from researchpilot.research_models import RoundTrace
from researchpilot.research_iteration import EvidenceGapRecord
from researchpilot.research_graph import _ResearchNodes
from researchpilot.openai_evidence_client import OpenAIClaimSynthesizer


def record(key="c", status="supported", text="The spectral bound holds.", ids=("p",)):
    v=verdict(ids,text);v.claim_id=key;v.status=status;v.assertions[0].status=status
    return VerifiedClaim(claim=AnswerClaim(claim_id=key,text=text,evidence_ids=list(ids)),verification=v)


def bundle(corpus=None, target=QUESTION, **kwargs):
    c=corpus or chain();limits=AssemblyLimits(**kwargs)
    return finish_bundle(EvidenceAssembler().assemble(target,c[1],c,limits),max_members=limits.max_bundle_members)


class StabilityTests(OfflineTest):
    def test_addition_keeps_previous_evidence(self):
        old=passage_index(chain()[:2]);new=passage_index(chain());assert_evidence_union(old,new)

    def test_evidence_lost_raises(self):
        with self.assertRaisesRegex(ValueError,"evidence_lost"):
            assert_evidence_union(passage_index(chain()),passage_index(chain()[:2]))

    def test_changed_passage_raises(self):
        old=passage_index(chain());new=dict(old);new['p']=new['p'].model_copy(update={'text':'Changed'})
        with self.assertRaisesRegex(ValueError,"evidence_lost"):assert_evidence_union(old,new)

    def test_supported_claim_text_stable(self):
        old=record();new=AnswerDraft(claims=[old.claim.model_copy(update={'text':'Drift'})],limitations=[])
        self.assertEqual(merge_targeted_draft([old],new,chain(),passage_index(chain())).claims,[old.claim])

    def test_stable_evidence_ids(self):
        old=record();new=AnswerDraft(claims=[],limitations=[])
        self.assertEqual(merge_targeted_draft([old],new,chain(),passage_index(chain())).claims[0].evidence_ids,['p'])

    def test_partial_revision_uses_union(self):
        old=record(status='partially_supported');new=AnswerDraft(claims=[old.claim.model_copy(update={'text':'Scoped','evidence_ids':['t']})],limitations=[])
        result=merge_targeted_draft([old],new,[chain()[0]],passage_index(chain()))
        self.assertEqual(result.claims[0].evidence_ids,['p','t','a'])

    def test_unrelated_bundle_does_not_downgrade(self):
        b=bundle(chain()[:2]);v=verdict(['x'],'A cache stores blocks.')
        self.assertEqual(guard_bundle_support(v,[b], [passage('x',text='Cache stores blocks.')]),v)

    def test_conflicting_downgrade_audited(self):
        old=record();new=record(status='conflicting',ids=('p','t'))
        self.assertEqual(audit_downgrade(old,new).reason,'conflicting_new_evidence')

    def test_invalidated_support_audited(self):
        self.assertEqual(audit_downgrade(record(),record(status='partially_supported')).reason,'prior_support_invalidated')

    def test_claim_change_reason(self):
        self.assertEqual(audit_downgrade(record(),record(status='unsupported',text='Different')).reason,'claim_text_changed')

    def test_cited_evidence_loss_raises(self):
        with self.assertRaisesRegex(ValueError,'evidence_lost'):
            audit_downgrade(record(),record(ids=('t',)))

    def test_potential_conflict_can_recheck_unchanged_claim(self):
        old=record(text='The spectral bound holds for this graph.')
        self.assertTrue(recheck_supported(old,[passage('new',text='The spectral bound does not hold for this graph.')],passage_index(chain()),[]))

    def test_unrelated_additions_do_not_recheck(self):
        self.assertFalse(recheck_supported(record(),[passage('new',text='Storage latency is reduced.')],passage_index(chain()),[]))

    def test_adapter_targets_only_unresolved_claims(self):
        client=OpenAIClaimSynthesizer();client._parse=Mock(return_value=AnswerDraft(claims=[],limitations=[]))
        client.synthesize_targets(QUESTION,chain(),[],[record('partial').claim],[record('stable').claim],[])
        instructions,payload,_=client._parse.call_args.args
        self.assertEqual(payload['target_claims'][0]['claim_id'],'partial')
        self.assertEqual(payload['stable_claims'][0]['claim_id'],'stable')
        self.assertIn('Do not return or rewrite stable_claims',instructions)

    def test_verifier_reuses_independent_supported_claim(self):
        s,f=fixtures.BundleRuntimeTests.fixture(self);old=record();s.update(claims=[old],draft=AnswerDraft(claims=[old.claim],limitations=[]),
            verified_evidence={'p':chain()[1]},round_trace=[RoundTrace(round=1,queries=1,query_ids=['q'])])
        f.verifier=Mock();out=_ResearchNodes(f).verify_claims(s)
        f.verifier.verify.assert_not_called();self.assertEqual(out['claims'],[old])

    def test_actual_recheck_records_conflict(self):
        s,f=fixtures.BundleRuntimeTests.fixture(self);old=record(text='The spectral bound holds for this graph.')
        extra=passage('new',text='The spectral bound does not hold for this graph.')
        s['selected_evidence']['new']=extra
        s.update(claims=[old],draft=AnswerDraft(claims=[old.claim],limitations=[]),verified_evidence={'p':chain()[1]},
            round_trace=[RoundTrace(round=1,queries=1,query_ids=['q'])])
        f.verifier=Mock();f.verifier.verify.side_effect=lambda c,ps: record(status='conflicting',text=c.text,ids=c.evidence_ids).verification
        out=_ResearchNodes(f).verify_claims(s)
        self.assertEqual(out['downgrade_audit'][0].reason,'conflicting_new_evidence')
        self.assertEqual(f.verifier.verify.call_args.args[0].evidence_ids,['p','new'])

    def test_bundle_risk_cannot_override_unchanged_verified_quantity(self):
        s,f=fixtures.BundleRuntimeTests.fixture(self,True)
        old=record(text='The spectral bound holds with constant two.')
        old.verification.assertions[0].kind='quantitative'
        b=bundle(list(s['passages'].values()))
        extra=passage('new',text='The spectral bound with constant two does not hold without the stated assumptions.')
        s['selected_evidence']['new']=extra
        s.update(claims=[old],draft=AnswerDraft(claims=[old.claim],limitations=[]),
            verified_evidence={'p':s['selected_evidence']['p']},evidence_bundles=[b],
            round_trace=[RoundTrace(round=1,queries=1,query_ids=['q'])])
        f.verifier=Mock()
        def verify(c,ps):
            v=verdict(c.evidence_ids,c.text);v.assertions[0].kind='quantitative';v.assertions[0].evidence_ids=['p'];return v
        f.verifier.verify.side_effect=verify
        out=_ResearchNodes(f).verify_claims(s)
        self.assertEqual(out['claims'][0].verification.status,'supported')
        self.assertEqual(out['downgrade_audit'],[])
        f.verifier.verify.assert_called_once()

    def test_unrelated_negative_sentence_not_a_conflict_trigger(self):
        old=record(text='The spectral bound holds with constant two.')
        extra=passage('new',text='The spectral bound holds with constant two. The implementation does not use caching.')
        self.assertFalse(recheck_supported(old,[extra],passage_index(chain()),[]))


class DependencyTests(OfflineTest):
    def check_heading(self,heading,key):
        index=build_statement_index([passage('x',number=17,text=heading+' The graph bound holds.')])
        self.assertEqual(index[key][0].page,17);self.assertEqual(index[key][0].evidence_id,'x')

    def test_assumption_a1(self):self.check_heading('Assumption A1.','assumption:a1')
    def test_condition_35(self):self.check_heading('Condition 3.5.','condition:3_5')
    def test_lemma_a1(self):self.check_heading('Lemma A.1.','lemma:a_1')
    def test_proposition_42(self):self.check_heading('Proposition 4.2.','proposition:4_2')
    def test_definition(self):self.check_heading('Definition 2.','definition:2')
    def test_corollary(self):self.check_heading('Corollary 3.','corollary:3')
    def test_bibliography_not_local(self):self.assertEqual(references('By [12] and [37, Theorem 4].'),[])
    def test_using_equation(self):self.assertEqual(references('Using (12), the bound holds.'),['equation:12'])

    def test_outside_radius_assumption(self):
        c=chain();c[0]=c[0].model_copy(update={'page_number':1})
        self.assertIn('a',bundle(c,max_page_radius=0).resolved_references['assumption:a1'])

    def test_nested_lemma(self):
        c=chain();c[2]=c[2].model_copy(update={'text':THEOREM+' By Lemma A.1, the bound holds.'})
        c += [passage('l',number=30,text='Lemma A.1. Under Condition 3.5, the spectral bound holds.'),
              passage('cond',number=40,text='Condition 3.5. The initialization is local.')]
        b=bundle(c);self.assertIn('l',[m.evidence_id for m in b.members]);self.assertIn('cond',[m.evidence_id for m in b.members])

    def test_depth_cap(self):
        c=chain();c[0]=c[0].model_copy(update={'text':ASSUMPTION+' Under Condition 3.5.'})
        c += [passage('cond',number=40,text='Condition 3.5. The initialization is local.')]
        b=bundle(c,max_reference_depth=1);self.assertFalse(b.complete)
        self.assertTrue(any('depth budget' in m for m in b.missing_components))

    def test_member_cap(self):self.assertEqual(len(bundle(max_bundle_members=1).members),1)
    def test_page_cap(self):self.assertLessEqual(len({m.page_number for m in bundle(max_dependency_pages=2).members}),2)
    def test_deterministic_traversal(self):self.assertEqual(bundle(),bundle())

    def test_unrelated_nearby_statement_excluded(self):
        c=chain()+[passage('irrelevant',number=7,text='Theorem 99. Under Assumption X7, storage latency is bounded.')]
        b=bundle(c);self.assertNotIn('irrelevant',[m.evidence_id for m in b.members])
        self.assertFalse(any('x7' in m for m in b.missing_components))

    def test_directed_references(self):
        b=bundle();self.assertTrue(any(e.referenced_by=='p' and e.target_id=='a' and e.dependency_type=='reference' for e in b.dependencies))

    def test_component_specific_spectral_only(self):
        b=bundle(chain()[:2],target='The spectral bound holds under Assumption A1.')
        self.assertTrue(b.complete);self.assertEqual(list(b.required_components),['spectral_bound'])

    def test_curvature_requires_connecting_result(self):
        b=bundle(chain()[:2]);self.assertFalse(b.complete);self.assertEqual(b.required_components['curvature_connection'],[])

    def test_partial_chain_keeps_spectral_atom(self):
        c=chain()[:2];b=bundle(c)
        self.assertEqual(guard_bundle_support(verdict(['p'],'The spectral bound holds.'),[b],c).status,'supported')
        self.assertEqual(guard_bundle_support(verdict(['p']),[b],c).status,'partially_supported')

    def test_complete_chain_supports_scope(self):
        self.assertTrue(bundle().complete);self.assertEqual(guard_bundle_support(verdict(),[bundle()],chain()).status,'supported')

    def test_only_material_component_in_gap(self):
        b=bundle(chain()[:2]);g=reconcile_bundle_gaps([], [b],1)
        self.assertEqual(len(g),1);self.assertIn('curvature',g[0].description)
        self.assertNotIn('recovery',g[0].description)

    def test_peripheral_missing_no_gap(self):
        c=chain()+[passage('noise',text='Theorem 99. Under Condition 99, cache latency is bounded.')]
        self.assertEqual(reconcile_bundle_gaps([], [bundle(c)],1),[])

    def test_scanning_spans_do_not_change_originals(self):
        c=chain();before=[p.model_dump() for p in c];bundle(c)
        self.assertEqual(before,[p.model_dump() for p in c])

    def test_discussion_is_not_a_statement_declaration(self):
        c=[passage('l',text='Lemma 12 yields a useful bound. Lemma 4 upper bounds the norm.')]
        self.assertEqual(build_statement_index(c),{})

    def test_bold_repeated_graph_labels_keep_original_text(self):
        p=passage('g',text='• GGG 2: σ2(GGG) ≤ c. • GGG 3: The graph is biregular.')
        index=build_statement_index([p])
        self.assertEqual(set(index),{'assumption:g2','assumption:g3'})
        self.assertIn('GGG 2',index['assumption:g2'][0].local_text)
        self.assertEqual(references('Under assumptions GGG 2 and GGG 3.'),['assumption:g2','assumption:g3'])

    def test_historical_bundle_identity_still_valid(self):
        b=bundle();b.assembly_version='structured-evidence-v1'
        b.bundle_id=bundle_identity(b.paper_id,b.anchor_evidence_id,[m.evidence_id for m in b.members],
                                    b.purpose,b.assembly_version)
        validate_bundle(b,passage_index(chain()))

    def test_different_target_requirements_have_distinct_identity(self):
        ids=['a','p','t']
        self.assertNotEqual(bundle_identity('paper','p',ids,'spectral bound'),bundle_identity('paper','p',ids,'curvature'))

    def test_proof_continuation_reaches_conclusion_before_next_proof(self):
        c=chain()+[passage('proof',number=20,text='Proof of Theorem 4. Under Assumption A1, begin the proof.'),
            passage('middle',number=21,text='The argument continues by Proposition 2.'),
            passage('end',number=22,text='Combining the bounds yields local curvature. Proof of Theorem 99. Unrelated result.')]
        b=bundle(c)
        self.assertTrue({'proof','middle','end'} <= {m.evidence_id for m in b.members})
        self.assertFalse(any('99' in m for m in b.missing_components))


class VisionStabilizationTests(OfflineTest):
    def runtime(self,count=1):
        s,f=fixtures.BundleRuntimeTests.fixture(self,True)
        c=chain();c[1]=s['passages']['p']
        for p in c[:count]:
            s['passages'][p.passage_id]=p.model_copy(update={'text':p.text+' x = (a + b'})
        s['selected_evidence']['p']=s['passages']['p']
        root=self.temporary_directory();pdf=root/'paper.pdf';pdf.write_bytes(renderable_pdf(['math']*8))
        client=Mock(model='fake');client.extract.side_effect=lambda q,g,t,p,e: math_item(
            evidence_id=e,paper_id=p.paper_id,page_number=p.page_number,ambiguity='Unreadable coefficient.')
        f.document_rescuer=DocumentRescuer(pdf_provider=lambda p:pdf,renderer=PDFPageRenderer(root/'images'),page_client=client)
        return s,f,client

    def test_required_formula_triggers(self):
        s,f,v=self.runtime();assemble_selected(s,f);self.assertGreater(v.extract.call_count,0)

    def test_clean_member_no_vision(self):
        s,f=fixtures.BundleRuntimeTests.fixture(self);f.document_rescuer=DocumentRescuer(pdf_provider=Mock(),renderer=Mock(),page_client=Mock())
        assemble_selected(s,f);f.document_rescuer.page_client.extract.assert_not_called()

    def test_page_reused_across_assemblies(self):
        s,f,v=self.runtime(3);s.update(assemble_selected(s,f));count=v.extract.call_count
        s['assembly_cache']={};s['assembly_context_counts']={};assemble_selected(s,f)
        self.assertEqual(v.extract.call_count,count)

    def test_bundle_visual_budget(self):
        s,f,v=self.runtime(3);s['request'].assembly_limits.max_visual_pages_per_bundle=1
        assemble_selected(s,f);self.assertEqual(v.extract.call_count,1)

    def test_run_visual_budget(self):
        s,f,v=self.runtime(3);s['request'].rescue_limits.max_visual_pages_per_run=1
        assemble_selected(s,f);self.assertEqual(v.extract.call_count,1)

    def test_ambiguity_incomplete(self):
        s,f,v=self.runtime();out=assemble_selected(s,f)
        self.assertFalse(out['evidence_bundles'][0].complete)
        self.assertTrue(any('ambiguous visual' in m for m in out['evidence_bundles'][0].missing_components))

    def test_page_provenance(self):
        s,f,v=self.runtime();out=assemble_selected(s,f);b=out['evidence_bundles'][0]
        validate_bundle(b,out['selected_evidence'])
        for r in out['visual_evidence'].values():
            self.assertTrue(r.pdf_sha256);self.assertTrue(r.parser_passage_ids)


class AssemblyQuotaTests(OfflineTest):
    def test_rephrased_gap_same_actual_dependency_set_reused(self):
        s,f=fixtures.BundleRuntimeTests.fixture(self)
        s['gap_ledger']=[EvidenceGapRecord(gap_id='gap2',description=QUESTION+' Please check.',
            search_focus=QUESTION,related_claim_ids=['c'],severity='critical',first_seen_round=1,last_updated_round=1)]
        out=assemble_selected(s,f)
        self.assertEqual(len(out['evidence_bundles']),1)
        self.assertEqual(len(out['assembly_cache']),2)
        self.assertEqual(len(out['evidence_bundles'][0].target_descriptions),2)

    def test_single_paper_cannot_consume_run_bundle_quota(self):
        s,f=fixtures.BundleRuntimeTests.fixture(self)
        s['selected_evidence']=passage_index(chain())
        s['claims']=[record('c'+str(i),status='partially_supported',text=QUESTION,ids=(p.passage_id,)) for i,p in enumerate(chain())]
        out=assemble_selected(s,f)
        self.assertLessEqual(len(out['evidence_bundles']),2)
        self.assertLess(len(out['evidence_bundles']),s['request'].assembly_limits.max_bundles_per_run)

    def test_later_paper_keeps_assembly_capacity(self):
        s,f=fixtures.BundleRuntimeTests.fixture(self)
        s['selected_evidence']=passage_index(chain())
        s['claims']=[record('c'+str(i),status='partially_supported',text=QUESTION,ids=(p.passage_id,)) for i,p in enumerate(chain())]
        s.update(assemble_selected(s,f))
        later=[p.model_copy(update={'paper_id':'later-paper','title':'Later acquired paper','passage_id':p.passage_id+'2'}) for p in chain()]
        s['passages'].update(passage_index(later));s['selected_evidence']['p2']=later[1]
        s['claims'].append(record('later',status='partially_supported',text=QUESTION,ids=('p2',)))
        out=assemble_selected(s,f)
        self.assertTrue(any(b.paper_id=='later-paper' for b in out['evidence_bundles']))


class GenericRegressionClasses(OfflineTest):
    def test_independent_supported_claim_survives_incomplete_bundle(self):
        f=fixtures.BundleGraphTests.fixture(self,False);original=f.synthesizer.synthesize.side_effect
        def synth(q,ps):
            draft=original(q,ps)
            draft.claims.insert(0,AnswerClaim(claim_id='stable',text='The spectral bound is conditional.',evidence_ids=[ps[0].passage_id]))
            return draft
        f.synthesizer.synthesize.side_effect=synth;verify=f.verifier.verify.side_effect
        def check(c,ps):
            if c.claim_id=='stable':
                v=verdict(c.evidence_ids,c.text);v.claim_id='stable';return v
            return verify(c,ps)
        f.verifier.verify.side_effect=check
        result=fixtures.BundleGraphTests.run_fixture(self,f)
        self.assertEqual(result.claims[0].claim.text,'The spectral bound is conditional.')
        self.assertEqual(result.claims[0].verification.status,'supported')
        self.assertEqual(sum(c.args[0].claim_id=='stable' for c in f.verifier.verify.call_args_list),1)

    def synthetic_chain(self,complete=True):
        c=[passage('a',number=20,text=ASSUMPTION),passage('p',number=21,text=SPECTRAL),
           passage('l',number=22,text='Lemma 3. Under Proposition 2, the spectral condition controls the local objective.')]
        if complete:c.append(passage('t',number=23,text='Theorem 4. Under Assumption A1 and Lemma 3, restricted curvature holds locally.'))
        return c

    def test_four_page_chain_beyond_cheap_window(self):
        c=self.synthetic_chain();b=bundle(c,max_page_radius=0)
        self.assertEqual({m.page_number for m in b.members},{20,21,22,23});self.assertTrue(b.complete)
        self.assertEqual(guard_bundle_support(verdict(['a','p','l','t']),[b],c).status,'supported')

    def test_missing_connection_keeps_spectral_and_gap(self):
        c=self.synthetic_chain(False);b=bundle(c,max_page_radius=0)
        self.assertFalse(b.complete)
        self.assertEqual(guard_bundle_support(verdict(['p'],'The spectral bound holds.'),[b],c).status,'supported')
        self.assertIn('curvature',reconcile_bundle_gaps([], [b],1)[0].description)
