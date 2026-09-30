from unittest.mock import Mock

from phase5_helpers import OfflineTest, paper, passage
from test_openai_page_evidence import gap, math_item
from test_pdf_page_renderer import renderable_pdf
from researchpilot.document_rescue import (
    DocumentRescuer, RescueLimits, corpus_fingerprint, formula_risk, gap_views,
    neighbors, structural_markers, needs_formal_evidence,
)
from researchpilot.passage_retrieval import LexicalRetriever
from researchpilot.pdf_page_renderer import PDFPageRenderer
from researchpilot.research_models import SelectedPaper


QUESTION = "How do spectral observations control local curvature?"


class StructureAndRiskTests(OfflineTest):
    def test_proposition_decimal_number(self):
        self.assertIn("Proposition 3.2", structural_markers("See Proposition 3.2 for curvature."))

    def test_assumption_appendix_number(self):
        self.assertIn("Assumption A1", structural_markers("Assumption A1 requires coherence."))

    def test_lemma_dotted_appendix_number(self):
        self.assertIn("Lemma A.2", structural_markers("Lemma A.2 is used here."))

    def test_all_structure_types(self):
        for text in ("Theorem 3.2", "Corollary 4", "Condition 3.5", "Definition 2", "Remark 3", "Proof",
                     "Eq. (12)", "Equation (4.3)"):
            with self.subTest(text=text):
                self.assertIn(text, structural_markers(text))

    def test_plural_formal_requirements_recognized(self):
        for term in ("assumptions", "conditions", "guarantees", "theorems", "equations"):
            with self.subTest(term=term):
                self.assertTrue(needs_formal_evidence(gap(description=f"Need {term} for recovery.", search_focus="recovery")))

    def test_clean_prose_no_risk(self):
        self.assertFalse(formula_risk("We measure latency at different request loads.").risky)

    def test_simple_equation_no_risk(self):
        self.assertFalse(formula_risk("Theorem 1. Under A, x = 2.").risky)

    def test_ambiguous_radical_detected(self):
        self.assertIn("ambiguous radical scope", formula_risk("σ2 ≤ √d1 - 1 + √d2 - 1").reasons)

    def test_lost_subscript_detected(self):
        self.assertTrue(formula_risk("sigma_\n = 2").risky)

    def test_lost_superscript_detected(self):
        self.assertTrue(formula_risk("x^\n + y = 1").risky)

    def test_fragmented_inequality(self):
        self.assertIn("fragmented inequality", formula_risk("x\n≤\n2").reasons)

    def test_no_heuristic_formula_repair(self):
        text = "2 d1 - 1 + 2 d2 - 1"
        result = formula_risk(text)
        self.assertTrue(result.risky)
        self.assertEqual(set(result.model_dump()), {"risky", "reasons"})
        self.assertEqual(text, "2 d1 - 1 + 2 d2 - 1")

    def test_table_layout_detected(self):
        self.assertTrue(formula_risk("Table 2. latency 1 2 3 4 5 6 7 8").risky)

    def test_prose_table_reference_and_scattered_numbers_are_not_layout_risk(self):
        text = ("In Section 4.2 we test 6 models [15, 21, 43], with 60% recall and "
                "20% noise. Figure 2 shows the trend; Table 2 provides details.")
        self.assertFalse(formula_risk(text).risky)

    def test_broken_parentheses_detected(self):
        self.assertTrue(formula_risk("x = (a + b").risky)

    def test_neighbor_context_preserves_separate_pages(self):
        corpus = [passage("prior", number=3), passage("target", number=4), passage("next", number=4)]
        result = neighbors(corpus[1], corpus)
        self.assertEqual([p.page_number for p in result], [3, 4, 4])
        self.assertEqual([p.passage_id for p in result], ["prior", "target", "next"])

    def test_no_distant_or_other_document_neighbors(self):
        corpus = [passage("far", number=1), passage("target", number=7), passage("other", paper("other"), 7)]
        self.assertEqual([p.passage_id for p in neighbors(corpus[1], corpus)], ["target"])


class LocalRescueTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.paper = paper(title="Spectral observations and curvature", abstract="Local curvature for spectral observations.")
        self.papers = [SelectedPaper(citation_label="P1", group_id="g", paper=self.paper, source_type="pdf")]
        self.corpus = [passage(f"intro{i}", self.paper, 1, "Spectral observations control local curvature. " * 5) for i in range(24)]
        self.hidden = passage("hidden", self.paper, 8, "Proposition 4.2. Under incoherence, restricted convexity holds in a local neighborhood.")
        self.corpus.append(self.hidden)
        self.rescuer = DocumentRescuer()
        self.limits = RescueLimits()

    def plans(self, gaps=None, concepts=None, selected=None, **limits):
        return self.rescuer.plan(QUESTION, [gap(search_focus="restricted convexity")] if gaps is None else gaps,
            self.papers, self.corpus, selected or {}, concepts or [], [], RescueLimits(**limits))

    def test_no_gap_no_rescue(self):
        self.assertEqual(self.plans(gaps=[]), [])

    def test_peripheral_gap_no_rescue(self):
        self.assertEqual(self.plans(gaps=[gap(relevance_to_question="peripheral")]), [])

    def test_meaningful_gap_triggers(self):
        self.assertEqual(len(self.plans()), 1)

    def test_all_document_passages_scanned(self):
        self.assertEqual(len(self.plans()[0].corpus), 25)

    def test_proposition_outside_top_k_recovered(self):
        self.assertNotIn("hidden", [r.passage.passage_id for r in LexicalRetriever().retrieve(QUESTION, self.corpus, 20)])
        self.assertIn("hidden", [p.passage_id for p in self.plans()[0].offered])

    def test_search_focus_recovers_hidden_statement(self):
        self.assertEqual(self.plans()[0].targets[0].passage_id, "hidden")

    def test_concept_view_recovers_rare_notation(self):
        views = gap_views(QUESTION, gap(), ["restricted convexity"], [], self.corpus, 4)
        self.assertTrue(any(v.origin_text == "restricted convexity" for v in views))
        retrieved = LexicalRetriever().retrieve_views(views, self.corpus, 3)
        self.assertIn("hidden", [p.passage.passage_id for p in retrieved])

    def test_formal_gap_prioritizes_structure(self):
        self.assertEqual(self.plans()[0].targets[0], self.hidden)

    def test_structure_does_not_dominate_empirical_gap(self):
        self.corpus = [passage("data", self.paper, text="Measured latency decreases under cache contention."),
                       passage("theorem", self.paper, text="Theorem 2. Spectral observations have curvature.")]
        results = self.plans(gaps=[gap(description="Need measured latency under contention.", search_focus="measured latency")])
        self.assertEqual(results[0].targets[0].passage_id, "data")

    def test_selected_duplicates_removed(self):
        plans = self.plans(selected={"hidden": self.hidden})
        self.assertNotIn("hidden", [p.passage_id for p in plans[0].offered])

    def test_candidate_and_view_caps(self):
        result = self.plans(max_rescue_passages_per_gap=2, max_rescue_views_per_gap=2)[0]
        self.assertLessEqual(len(result.offered), 2)
        self.assertLessEqual(len(result.views), 2)

    def test_abstract_not_deeply_searched(self):
        self.corpus = [passage("abstract", self.paper, None, "Spectral curvature theorem.")]
        self.assertEqual(self.plans(), [])

    def test_unrelated_documents_not_searched(self):
        self.corpus = [passage("data", paper("biology", title="Genomes", abstract="Gene sequencing"), text="Gene sequencing.")]
        self.assertEqual(self.plans(), [])

    def test_already_selected_all_clean_evidence_no_attempt(self):
        self.assertEqual(self.plans(selected={p.passage_id: p for p in self.corpus}), [])

    def test_text_only_rescue_never_needs_pdf_provider(self):
        provider = Mock(side_effect=AssertionError("no downloads or rendering"))
        self.rescuer = DocumentRescuer(pdf_provider=provider)
        result = self.rescuer.rescue(QUESTION, self.plans(), self.papers, self.limits, [], 1, "corpus")
        self.assertTrue(result.passages)
        provider.assert_not_called()

    def test_corpus_fingerprint_changes_with_new_document(self):
        first = corpus_fingerprint(self.corpus)
        self.assertNotEqual(first, corpus_fingerprint([*self.corpus, passage("new")]))

    def test_visual_not_used_for_empirical_prose_gap(self):
        risky = passage(text="Theorem 2. latency ≤ √x + 1")
        self.assertFalse(self.rescuer._visual_target(risky, gap(description="Latency user experience", search_focus="latency")))

    def test_table_risk_does_not_use_generic_proof_request_to_trigger_vision(self):
        target = passage(text="Table 2. filtering 1 2 3 4 5 6 7 8")
        request = gap(description="Need proof that filtering prevents distraction.", search_focus="filtering")
        self.assertFalse(self.rescuer._visual_target(target, request))

    def test_table_risk_can_trigger_for_explicit_quantitative_gap(self):
        target = passage(text="Table 2. latency 1 2 3 4 5 6 7 8")
        request = gap(description="Need latency measurements from the table.", search_focus="latency")
        self.assertTrue(self.rescuer._visual_target(target, request))


class RescueVisualBudgetTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.root = self.temporary_directory()
        self.source = self.root / "source.pdf"
        self.source.write_bytes(renderable_pdf())
        self.paper = paper(title="Spectral curvature", abstract="Spectral curvature conditions.")
        self.papers = [SelectedPaper(citation_label="P1", group_id="g", paper=self.paper, source_type="pdf")]
        self.corpus = [passage(f"e{i}", self.paper, i, "Proposition 4.2. Spectral curvature ≤ √d - 1") for i in (1, 2, 3)]
        self.client = Mock(model="fake")
        self.client.extract.side_effect = lambda q, g, t, pg, eid: math_item(evidence_id=eid, paper_id=pg.paper_id, page_number=pg.page_number)
        self.rescuer = DocumentRescuer(pdf_provider=lambda p: self.source, renderer=PDFPageRenderer(self.root / "images"), page_client=self.client)

    def run_rescue(self, limits=None, history=None):
        limits = limits or RescueLimits()
        plans = self.rescuer.plan(QUESTION, [gap()], self.papers, self.corpus, {}, [], [], limits)
        return self.rescuer.rescue(QUESTION, plans, self.papers, limits, history or [], 1, "fingerprint")

    def test_per_gap_page_limit(self):
        result = self.run_rescue()
        self.assertEqual(len(result.trace.visual_pages), 1)
        self.assertEqual(self.client.extract.call_count, 1)

    def test_run_page_limit_across_rescues(self):
        limits = RescueLimits(max_visual_pages_per_gap=2, max_visual_pages_per_run=2)
        first = self.run_rescue(limits)
        second = self.run_rescue(limits, [first.trace])
        self.assertEqual(self.client.extract.call_count, 2)
        self.assertEqual(second.trace.visual_pages, [])

    def test_same_page_not_retried_after_failure(self):
        self.client.extract.side_effect = RuntimeError("vision failed")
        first = self.run_rescue()
        second = self.run_rescue(history=[first.trace])
        self.assertEqual([t.page_number for t in first.trace.visual_pages + second.trace.visual_pages], [1, 2])
        self.assertTrue(first.warnings)

    def test_zero_vision_budget(self):
        result = self.run_rescue(RescueLimits(max_visual_pages_per_run=0))
        self.client.extract.assert_not_called()
        self.assertEqual(result.trace.visual_pages, [])

    def test_clean_equation_no_visual_call(self):
        self.corpus = [passage("e", self.paper, text="Proposition 1. Spectral curvature x = 2 under A.")]
        self.run_rescue()
        self.client.extract.assert_not_called()

    def test_visual_and_parser_provenance_retained(self):
        result = self.run_rescue()
        visual = next(iter(result.visual_evidence.values()))
        self.assertEqual(visual.parser_passage_ids, ["e1"])
        self.assertIn("e1", [p.passage_id for p in result.passages])
        self.assertEqual(visual.evidence.page_number, 1)

    def test_total_candidate_cap_includes_visual_evidence(self):
        result = self.run_rescue(RescueLimits(max_rescue_passages_per_run=2, max_rescue_passages_per_gap=2))
        self.assertEqual(len(result.passages), 2)

    def test_trace_records_risk_and_render_counts(self):
        trace = self.run_rescue().trace
        self.assertEqual(trace.passages_scanned, 3)
        self.assertEqual(trace.papers_searched, [self.paper.paper_id])
        self.assertTrue(trace.visual_pages[0].risk_reasons)
        self.assertTrue(trace.visual_pages[0].rendered)
