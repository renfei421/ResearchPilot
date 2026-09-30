"""Synthetic same-paper math chains; no literature fixtures or network access."""

from unittest.mock import Mock

from phase5_helpers import OfflineTest, paper, passage
from researchpilot.evidence_assembly import (
    EvidenceAssembler, finish_bundle, should_assemble, references, declarations,
    assembly_cache_key, reliable_section,
)
from researchpilot.evidence_bundle import AssemblyLimits, EvidenceBundleMember, validate_bundle
from researchpilot.bundle_support import guard_bundle_support, reconcile_bundle_gaps, bundle_gap_id
from researchpilot.evidence import AnswerClaim, AtomicAssertion, ClaimVerification, VerifiedClaim
from researchpilot.answer_renderer import render_answer
from researchpilot.research_models import SelectedPaper
from researchpilot.research_iteration import gap_uncertainty

QUESTION = "What theoretical spectral evidence establishes restricted curvature?"
ASSUMPTION = "Assumption A1. The observation graph is biregular and the matrix is incoherent."
SPECTRAL = "Proposition 2. Under Assumption A1, the second singular value satisfies a spectral bound."
THEOREM = "Theorem 4. Under Assumption A1 and Proposition 2, restricted curvature holds in the stated local neighborhood."


def chain():
    return [passage("a", number=5, text=ASSUMPTION), passage("p", number=6, text=SPECTRAL),
            passage("t", number=7, text=THEOREM)]


def assemble(corpus=None, anchor=None, **limits):
    corpus = chain() if corpus is None else corpus
    anchor = corpus[1] if anchor is None else anchor
    lim = AssemblyLimits(**limits)
    draft = EvidenceAssembler().assemble(QUESTION, anchor, corpus, lim)
    return finish_bundle(draft, max_members=lim.max_bundle_members)


def verdict(ids=("a", "p", "t"), text="Under Assumption A1 and Proposition 2, restricted curvature holds locally."):
    atom = AtomicAssertion(assertion_id="atom", text=text, kind="theoretical", status="supported",
        evidence_ids=list(ids), reason="Synthetic cited chain.", scope_supported=True,
        inference_supported=True, quantities_supported=True)
    return ClaimVerification(claim_id="c", status="supported", evidence_ids=list(ids),
                             reason="Synthetic strict verifier result.", assertions=[atom])


class AssemblyTests(OfflineTest):
    def test_empirical_prose_does_not_trigger(self):
        self.assertFalse(should_assemble("What improves long document accuracy?", passage(text="Table 1 improves accuracy by 3 percent.")))

    def test_theorem_anchor_triggers(self):
        self.assertTrue(should_assemble(QUESTION, chain()[2]))

    def test_abstract_is_not_cross_page_evidence(self):
        self.assertFalse(should_assemble(QUESTION, passage(number=None, text=THEOREM)))

    def test_previous_page_assumption(self):
        self.assertIn("a", [m.evidence_id for m in assemble().members])

    def test_next_page_conclusion(self):
        self.assertIn("t", [m.evidence_id for m in assemble().members])

    def test_assumption_reference_resolved(self):
        self.assertEqual(assemble().resolved_references["assumption:a1"], ["a"])

    def test_condition_reference_resolved_beyond_radius(self):
        p = passage("x", number=8, text="Theorem 4. Under Condition 3.5, curvature holds.")
        c = passage("y", number=1, text="Condition 3.5. The graph is regular.")
        self.assertEqual(assemble([p, c], p).resolved_references["condition:3_5"], ["y"])

    def test_plural_assumptions_resolved(self):
        p = passage("x", text="Theorem 4. Under Assumptions A1 and A2, curvature holds.")
        c = [p, chain()[0], passage("a2", text="Assumption A2. The initialization is local.")]
        self.assertEqual(assemble(c,p).resolved_references["assumption:a2"], ["a2"])

    def test_explicit_reference_types(self):
        self.assertEqual(references("by Eq. (12), Lemma A.1 and Theorem 4"),
                         ["equation:12", "lemma:a_1", "theorem:4"])

    def test_reference_mention_not_declaration(self):
        self.assertEqual(declarations("We use Theorem 4 and Assumption A1 in the proof."), {})

    def test_cited_work_identifiers_not_resolved_inside_current_document(self):
        self.assertEqual(references("By [17, Lemma 7.1] and Lemma 2 in [19], use Assumption A1."),["assumption:a1"])

    def test_statement_heading_with_source_attribution_still_resolves(self):
        text="Lemma 9. [17, Lemma 4] Let the graph be regular. The bound holds."
        self.assertEqual(declarations(text),{"lemma:9":"proposition"})
        self.assertEqual(references(text),["lemma:9"])

    def test_other_paper_assumption_excluded(self):
        c = chain(); c[0] = passage("wrong", paper("other"), 5, ASSUMPTION)
        b = assemble(c)
        self.assertFalse(b.complete)
        self.assertNotIn("wrong", [m.evidence_id for m in b.members])

    def test_page_radius_restricts_non_reference_expansion(self):
        b = assemble(max_page_radius=0)
        self.assertIn("t", [m.evidence_id for m in b.members])  # Explicit claim-component lookup bypasses radius.
        self.assertIn("a", [m.evidence_id for m in b.members])  # Explicit lookup exception.

    def test_unrelated_neighbor_excluded(self):
        c = chain() + [passage("noise", number=7, text="Proposition 8. The cache latency is bounded by storage bandwidth.")]
        self.assertNotIn("noise", [m.evidence_id for m in assemble(c).members])

    def test_pages_and_roles_preserved(self):
        self.assertEqual({m.evidence_id: (m.page_number,m.role) for m in assemble().members},
                         {"a":(5,"assumption"),"p":(6,"proposition"),"t":(7,"theorem")})

    def test_deterministic_id(self):
        self.assertEqual(assemble().bundle_id, assemble().bundle_id)

    def test_duplicate_member_removed(self):
        c = chain(); b = assemble(c + c)
        self.assertEqual(len(b.members), 3)

    def test_member_cap(self):
        b = assemble(max_bundle_members=1)
        self.assertEqual(len(b.members),1)
        self.assertFalse(b.complete)

    def test_missing_conclusion_reported(self):
        b = assemble(chain()[:2]); self.assertFalse(b.complete)
        self.assertIn("explicit connecting curvature theorem/result",b.missing_components)

    def test_complete_chain(self):
        self.assertTrue(assemble().complete)

    def test_no_source_mutation(self):
        c = chain(); before = [p.model_dump() for p in c]; assemble(c)
        self.assertEqual(before,[p.model_dump() for p in c])

    def test_formula_continuity_incomplete(self):
        c = chain(); c[1] = c[1].model_copy(update={"text":SPECTRAL + " x = (a + b"})
        self.assertTrue(any("formula" in m for m in assemble(c).missing_components))

    def test_valid_provenance(self):
        validate_bundle(assemble(), {p.passage_id:p for p in chain()})

    def test_no_synthetic_page(self):
        b=assemble(); b.members[0].page_number=99
        with self.assertRaises(ValueError): validate_bundle(b, {p.passage_id:p for p in chain()})

    def test_member_identity_checked(self):
        with self.assertRaises(ValueError): EvidenceBundleMember(evidence_id="x",passage_id="y",page_number=1,role="theorem")

    def test_limits_hard_caps(self):
        for field,value in (("max_page_radius",3),("max_bundle_members",21),("max_reference_depth",5)):
            with self.subTest(field=field), self.assertRaises(ValueError): AssemblyLimits(**{field:value})

    def test_reference_depth_stays_incomplete(self):
        c = chain(); c[0]=c[0].model_copy(update={"text":ASSUMPTION+" Under Condition 3.5."})
        c.append(passage("condition",number=20,text="Condition 3.5. The graph is bounded."))
        b=assemble(c, max_page_radius=0,max_reference_depth=1)
        self.assertFalse(b.complete)

    def test_section_requires_reliable_heading(self):
        self.assertIsNone(reliable_section("Section 2 discusses assumptions in prose."))
        self.assertEqual(reliable_section("2.1 Graph assumptions\nContent."),"2.1 Graph assumptions")

    def test_cache_key_changes_when_document_changes(self):
        c=chain(); key=assembly_cache_key(c[1],c,QUESTION,AssemblyLimits())
        c[0]=c[0].model_copy(update={"text":"Assumption A1. Changed conditions."})
        self.assertNotEqual(key,assembly_cache_key(c[1],c,QUESTION,AssemblyLimits()))


class BundleSupportTests(OfflineTest):
    def test_uncited_pages_formula_risk_does_not_downgrade_clear_quantity(self):
        b=assemble(); b.complete=False;b.missing_components=["formula extraction/continuity on page 5"]
        v=verdict(["p"],"The stated constant is two.");v.assertions[0].kind="quantitative"
        self.assertEqual(guard_bundle_support(v,[b],chain()).status,"supported")

    def test_missing_proof_lemma_does_not_invalidate_explicit_theorem_statement(self):
        b=assemble(); b.complete=False;b.missing_components=["statement lemma:9"]
        v=verdict(["t"])
        self.assertEqual(guard_bundle_support(v,[b],chain()).status,"supported")

    def test_cited_page_formula_risk_still_limits_quantity(self):
        b=assemble(); b.complete=False;b.missing_components=["formula extraction/continuity on page 6"]
        v=verdict(["p"],"The stated constant is two.");v.assertions[0].kind="quantitative"
        self.assertEqual(guard_bundle_support(v,[b],chain()).status,"partially_supported")

    def test_uncited_assumption_reference_does_not_contaminate_independent_atom(self):
        b=assemble(); b.complete=False;b.missing_components=["statement assumption:b2"]
        self.assertEqual(guard_bundle_support(verdict(["t"]),[b],chain()).status,"supported")

    def test_assumption_and_theorem_scoped_support_preserved(self):
        self.assertEqual(guard_bundle_support(verdict(),[assemble()]).status,"supported")

    def test_assumption_alone_does_not_establish_curvature(self):
        c=chain()[:1]; b=assemble(c,c[0])
        self.assertNotEqual(guard_bundle_support(verdict(["a"]),[b]).status,"supported")

    def test_spectral_bound_alone_does_not_establish_curvature(self):
        c=chain()[:2]; self.assertNotEqual(guard_bundle_support(verdict(["p"]),[assemble(c)]).status,"supported")

    def test_missing_assumption_downgrades_theorem(self):
        c=chain()[1:]; b=assemble(c,c[1])
        self.assertEqual(guard_bundle_support(verdict(["p","t"]),[b]).status,"partially_supported")

    def test_all_conditions_supported(self):
        v=verdict(); self.assertEqual(guard_bundle_support(v,[assemble()]),v)

    def test_narrow_assumption_remains_supported(self):
        v=verdict(["a"],"The graph is assumed biregular.")
        self.assertEqual(guard_bundle_support(v,[assemble(chain()[:2])]).status,"supported")

    def test_unrelated_neighbor_cannot_supply_connecting_theorem(self):
        c=chain()[:2]+[passage("x",number=7,text="Theorem 8. Under unrelated cache allocations, bandwidth is bounded.")]
        b=assemble(c); self.assertFalse(b.complete)
        self.assertNotEqual(guard_bundle_support(verdict(["p"]),[b]).status,"supported")

    def test_guard_never_promotes_unsupported_verdict(self):
        v=verdict();v.status="unsupported"
        self.assertEqual(guard_bundle_support(v,[assemble()]).status,"unsupported")

    def test_multi_page_citations_resolve_separately(self):
        v=verdict();claim=AnswerClaim(claim_id="c",text=v.assertions[0].text,evidence_ids=v.evidence_ids)
        text=render_answer([VerifiedClaim(claim=claim,verification=v)],chain(),
            [SelectedPaper(citation_label="P1",group_id="g",paper=paper(),source_type="pdf")])
        for n in (5,6,7): self.assertIn(f"[P1, p. {n}]",text)
        self.assertNotIn("pp.",text)

    def test_incomplete_bundle_creates_gap(self):
        self.assertEqual(len(reconcile_bundle_gaps([],[assemble(chain()[:2])],1)),1)

    def test_missing_connection_remains_gap(self):
        g=reconcile_bundle_gaps([],[assemble(chain()[:2])],1)[0]
        self.assertIn("connecting curvature",gap_uncertainty(g))

    def test_repeated_missing_components_do_not_duplicate_gap(self):
        b=assemble(chain()[:2]);g=reconcile_bundle_gaps([],[b],1)
        self.assertEqual(len(reconcile_bundle_gaps(g,[b,b],2)),1)

    def test_unrelated_supported_evidence_cannot_close_gap(self):
        b=assemble(chain()[:2]);g=reconcile_bundle_gaps([],[b],1);g[0].status="resolved"
        self.assertEqual(reconcile_bundle_gaps(g,[b],2)[0].status,"unresolved")

    def test_completed_bundle_requires_explicit_verified_resolution(self):
        g=reconcile_bundle_gaps([],[assemble(chain()[:2])],1)
        self.assertEqual(reconcile_bundle_gaps(g,[assemble()],2)[0].status,"unresolved")

    def test_completed_supported_chain_preserves_correct_resolution(self):
        g=reconcile_bundle_gaps([],[assemble(chain()[:2])],1);g[0].status="resolved"
        v=verdict();c=VerifiedClaim(claim=AnswerClaim(claim_id="c",text=v.assertions[0].text,evidence_ids=v.evidence_ids),verification=v)
        self.assertEqual(reconcile_bundle_gaps(g,[assemble()],2,[c])[0].status,"resolved")

    def test_complete_bundle_unrelated_atom_does_not_resolve(self):
        g=reconcile_bundle_gaps([],[assemble(chain()[:2])],1);g[0].status="resolved"
        self.assertEqual(reconcile_bundle_gaps(g,[assemble()],2)[0].status,"unresolved")
