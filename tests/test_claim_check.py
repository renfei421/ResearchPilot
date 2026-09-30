import json
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import ValidationError

from claim_check_helpers import *
from phase5_helpers import OfflineTest
from researchpilot.claim_check_views import claim_html, claim_markdown
from researchpilot.claim_literature_map import build_maps, build_boundary, closest_papers
from researchpilot.evidence import EvidencePassage
from researchpilot.openai_claim_check import OpenAIClaimDecomposer, OpenAIClaimSearchPlanner, OpenAIClaimRelationAnalyzer


class ClaimContractsTests(OfflineTest):
    def test_single_assertion(self):
        self.assertEqual(len(ClaimDecomposition(main_claim="A", assertions=[assertion()]).assertions), 1)

    def test_compound_claim_dependencies_preserved(self):
        self.assertEqual(decomposition().assertions[-1].depends_on, ["C1", "C2"])

    def test_theoretical_type(self):
        self.assertEqual(assertion().claim_type, "theoretical")

    def test_methodological_type(self):
        data = assertion().model_dump() | {"claim_type": "methodological"}
        self.assertEqual(ClaimAssertion(**data).claim_type, "methodological")

    def test_blank_claim_rejected(self):
        for value in ["", " \n ", None, 2]:
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ClaimCheckRequest(claim=value)

    def test_duplicate_claim_ids_rejected(self):
        with self.assertRaises(ValidationError):
            ClaimDecomposition(main_claim="x", assertions=[assertion(), assertion()])

    def test_core_and_supporting(self):
        d = decomposition().model_dump()
        d["assertions"][0]["importance"] = "supporting"
        self.assertEqual(ClaimDecomposition(**d).assertions[0].importance, "supporting")

    def test_core_required(self):
        with self.assertRaises(ValidationError):
            ClaimDecomposition(main_claim="x", assertions=[assertion().model_copy(update={"importance":"supporting"})])

    def test_unknown_dependency(self):
        with self.assertRaises(ValidationError):
            ClaimDecomposition(main_claim="x", assertions=[assertion().model_copy(update={"depends_on":["bad"]})])

    def test_cycle_rejected(self):
        with self.assertRaises(ValidationError):
            ClaimDecomposition(main_claim="x", assertions=[assertion().model_copy(update={"depends_on":["C1"]})])

    def test_invalid_years(self):
        for data in [{"year_from": True}, {"year_to": "2025"}, {"year_from":2026,"year_to":2020}, {"year_from":0}]:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                ClaimCheckRequest(claim="x", **data)

    def test_budget_bounds(self):
        for data in [{"max_papers_per_claim":9}, {"max_papers_per_claim":False}, {"max_search_rounds":3}, {"max_search_rounds":0}]:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                ClaimCheckRequest(claim="x", **data)

    def test_plan_needs_direct(self):
        with self.assertRaises(ValidationError):
            ClaimSearchPlan(claim_id="C1", queries=[ClaimQuery(text="a",role="bridge"),ClaimQuery(text="b",role="terminology")])

    def test_complementary_query_roles(self):
        p = fixture().planner.plan(ClaimCheckRequest(claim=CLAIM), decomposition(), ["C1"], [], [])
        self.assertEqual([q.role for q in p.plans[0].queries], ["direct","terminology","bridge"])

    def test_plan_covers_all_requested_core_claims(self):
        p = fixture().planner.plan(ClaimCheckRequest(claim=CLAIM), decomposition(), ["C1"], [], [])
        with self.assertRaises(ValueError):
            p.validate_targets(["C1", "C2"])

    def test_duplicate_query_normalization(self):
        self.assertEqual(query_key("  Graph \n WEIGHTS "), query_key("graph weights"))
        with self.assertRaises(ValidationError):
            ClaimSearchPlan(claim_id="C1", queries=[ClaimQuery(text=" Graph weights ",role="direct"),ClaimQuery(text="graph  WEIGHTS",role="bridge")])

    def test_too_many_assertions(self):
        with self.assertRaises(ValidationError):
            ClaimDecomposition(main_claim="x", assertions=[assertion(str(i)) for i in range(7)])


class ClaimRelationTests(OfflineTest):
    def test_all_seven_relations(self):
        for label in ("DIRECT_OVERLAP","PARTIAL_OVERLAP","SUPPORTING","BRIDGING","CONTRADICTING","METHOD_SIMILAR","ADJACENT"):
            with self.subTest(label=label):
                self.assertEqual(relation(label=label).relation, label)

    def test_random_and_fixed_not_direct(self):
        comparisons = relation().comparisons
        comparisons[0] = comparisons[0].model_copy(update={"claim_scope":"fixed sampling","paper_scope":"random sampling","alignment":"different"})
        with self.assertRaises(ValidationError):
            relation(label="DIRECT_OVERLAP", comparisons=comparisons)

    def test_empirical_not_theorem(self):
        comparisons = relation().comparisons
        comparisons[5] = comparisons[5].model_copy(update={"claim_scope":"theoretical guarantee","paper_scope":"empirical gains","alignment":"different"})
        with self.assertRaises(ValidationError):
            relation(label="DIRECT_OVERLAP", comparisons=comparisons)

    def test_method_similar_different_conclusion(self):
        comparisons = relation().comparisons
        comparisons[5] = comparisons[5].model_copy(update={"alignment":"different"})
        self.assertEqual(relation(label="METHOD_SIMILAR", comparisons=comparisons).relation, "METHOD_SIMILAR")

    def test_different_assumptions_not_contradiction(self):
        comparisons = relation().comparisons
        comparisons[1] = comparisons[1].model_copy(update={"alignment":"different"})
        with self.assertRaises(ValidationError):
            relation(label="CONTRADICTING", comparisons=comparisons)

    def test_contradiction_requires_explicit_incompatibility(self):
        with self.assertRaises(ValidationError):
            relation(label="CONTRADICTING", explicit_incompatibility=False)

    def test_contradiction_requires_high_confidence(self):
        with self.assertRaises(ValidationError):
            relation(label="CONTRADICTING", confidence="low")

    def test_nonadjacent_requires_evidence(self):
        with self.assertRaises(ValidationError):
            relation(ids=[])

    def test_bibliography_cannot_support_relation(self):
        with self.assertRaises(ValidationError):
            relation(evidence_quality="bibliographic_only")

    def test_unknown_evidence_rejected(self):
        with self.assertRaises(ValueError):
            validate_relation(relation(), [assertion()], [])

    def test_wrong_paper_evidence_rejected(self):
        p = EvidencePassage(passage_id="eA",paper_id="B",title="B",text="substantive",source_type="abstract",page_number=None)
        with self.assertRaises(ValueError):
            validate_relation(relation(), [assertion()], [p])

    def test_overlap_needs_similarities_and_differences(self):
        with self.assertRaises(ValidationError):
            relation(label="PARTIAL_OVERLAP", differences=[])

    def test_no_close_match_is_not_paper_relation(self):
        with self.assertRaises(ValidationError):
            relation(label="NO_CLOSE_MATCH_FOUND")

    def test_generated_novelty_certification_rejected(self):
        for text in ["Your idea is novel.", "No one has done this.", "This is the first work to improve curvature."]:
            with self.subTest(text=text), self.assertRaises(ValidationError):
                relation(relation_summary=text)


class ClaimAgentTests(OfflineTest):
    def run_fixture(self, **kwargs):
        f = fixture(**kwargs)
        return f, f.agent.run(ClaimCheckRequest(claim=CLAIM, year_from=2020, year_to=2026))

    def test_generic_end_to_end(self):
        f, result = self.run_fixture()
        c3 = [r for r in result.relations if r.claim_id == "C3"]
        self.assertEqual({r.paper_id:r.relation for r in c3}, {"A":"BRIDGING", "B":"SUPPORTING", "C":"PARTIAL_OVERLAP"})
        self.assertFalse(any(r.relation == "DIRECT_OVERLAP" for r in c3))
        self.assertTrue(result.novelty_boundary.established_components)
        self.assertTrue(result.novelty_boundary.potentially_underexplored_connections)
        available = {p.passage_id for p in result.evidence}
        self.assertTrue(all(set(r.evidence_ids) <= available for r in c3))
        self.assertEqual(result.novelty_boundary.caveat, CAVEAT)

    def test_shared_queries_execute_once(self):
        f, result = self.run_fixture()
        self.assertEqual(len(f.client.calls), 3)
        self.assertEqual(sum(t.reused for t in result.search_trace), 6)

    def test_years_passed_through(self):
        f, _ = self.run_fixture()
        self.assertTrue(all(c["year_from"]==2020 and c["year_to"]==2026 for c in f.client.calls))

    def test_acquire_each_paper_once(self):
        f, _ = self.run_fixture()
        self.assertEqual(f.fetcher.calls, ["A", "B", "C"])

    def test_all_core_plans(self):
        f, result = self.run_fixture()
        self.assertEqual(set(f.planner.calls[0][0]), {"C1","C2","C3"})
        self.assertEqual(set(result.run_stats.queries_per_claim), {"C1","C2","C3"})

    def test_no_research_answer_path(self):
        with patch.object(ResearchAgent, "run", side_effect=AssertionError("Wrong product path")):
            _, result = self.run_fixture()
        self.assertIsInstance(result, ClaimCheckResult)

    def test_empty_search_is_weak_not_novel(self):
        _, result = self.run_fixture(blank=True)
        self.assertTrue(all(m.coverage_status=="weak" and not m.no_close_match_found for m in result.claim_maps))

    def test_partial_overlap_is_a_close_match_with_sufficient_coverage(self):
        _, result = self.run_fixture()
        self.assertFalse(next(m for m in result.claim_maps if m.claim_id=="C3").no_close_match_found)
        self.assertIn("not proof", result.novelty_boundary.caveat)

    def test_boundaries_retain_dependency_ids(self):
        _, result = self.run_fixture()
        self.assertEqual(result.novelty_boundary.potentially_underexplored_connections[0].claim_ids, ["C1","C2","C3"])

    def test_partial_boundary_preserves_differences(self):
        _, result = self.run_fixture()
        self.assertIn("not established", result.novelty_boundary.partially_established_components[0].text)

    def test_contradictions_remain_visible(self):
        d = ClaimDecomposition(main_claim="x", assertions=[assertion()])
        rows = [relation(label="CONTRADICTING")]
        boundary = build_boundary(d, rows, build_maps(d, rows, []))
        self.assertEqual(len(boundary.contradictions), 1)

    def test_no_partial_result_on_relation_failure(self):
        f = fixture()
        f.analyzer.analyze = Mock(side_effect=RuntimeError("failed"))
        with self.assertRaisesRegex(RuntimeError,"failed"):
            f.agent.run(ClaimCheckRequest(claim=CLAIM))

    def test_query_memory_across_rounds(self):
        f = fixture()
        original=f.analyzer.analyze
        def weak(*args):
            result=original(*args)
            return RelationBatch(relations=[relation(r.claim_id,r.paper_id,"ADJACENT") for r in result.relations])
        f.analyzer.analyze=weak
        result=f.agent.run(ClaimCheckRequest(claim=CLAIM))
        self.assertEqual(result.termination_reason,"duplicate_queries")
        self.assertEqual(len(f.client.calls),3)
        self.assertEqual(len(f.planner.calls[1][1]),3)

    def test_no_new_papers_terminates(self):
        f=fixture(follow_up=True)
        f.analyzer.analyze=lambda assertions,pid,evidence,bundles: RelationBatch(
            relations=[relation(a.claim_id,pid,"ADJACENT") for a in assertions])
        result=f.agent.run(ClaimCheckRequest(claim=CLAIM))
        self.assertEqual(result.termination_reason,"no_new_papers")
        self.assertEqual(len(f.client.calls),6)
        self.assertEqual(len(f.fetcher.calls),3)

    def test_per_claim_document_budget(self):
        f=fixture()
        result=f.agent.run(ClaimCheckRequest(claim=CLAIM,max_papers_per_claim=1))
        for cid in ["C1","C2","C3"]:
            self.assertLessEqual(sum(r.claim_id==cid for r in result.relations),1)

    def test_transient_partial_search_prevents_absence_conclusion(self):
        import httpx
        f=fixture(); original=f.client.search_works
        def search(**kw):
            if kw["query"]=="weights curvature":
                raise httpx.ReadTimeout("temporary")
            return original(**kw)
        f.client.search_works=search
        result=f.agent.run(ClaimCheckRequest(claim=CLAIM))
        self.assertTrue(result.warnings)
        self.assertFalse(any(m.no_close_match_found for m in result.claim_maps))

    def test_all_searches_failed_propagates(self):
        import httpx
        f=fixture(); f.client.search_works=Mock(side_effect=httpx.ReadTimeout("temporary"))
        with self.assertRaisesRegex(RuntimeError,"All prior-work"):
            f.agent.run(ClaimCheckRequest(claim=CLAIM))

    def test_shared_evidence_assembly_integration(self):
        f=fixture(); original=f.fetcher.acquire
        def acquire(p):
            doc=original(p)
            if p.paper_id=="A":
                doc.pages[0].text="Proposition 1. Under Assumption 2, graph weight bounds imply restricted curvature. Assumption 2. The graph is regular."
            return doc
        f.fetcher.acquire=acquire
        result=f.agent.run(ClaimCheckRequest(claim=CLAIM))
        self.assertTrue(result.evidence_bundles)
        available={p.passage_id:p for p in result.evidence}
        for b in result.evidence_bundles:
            self.assertTrue(all(available[m.evidence_id].paper_id==b.paper_id for m in b.members))

    def test_known_combination_requires_direct_combined_proposition(self):
        d=decomposition()
        rows=[relation("C3", label="DIRECT_OVERLAP")]
        boundary=build_boundary(d,rows,build_maps(d,rows,[]))
        self.assertEqual(len(boundary.known_combinations),1)
        self.assertFalse(boundary.potentially_underexplored_connections)

    def test_result_roundtrip(self):
        _, result = self.run_fixture()
        self.assertEqual(result, ClaimCheckResult.model_validate_json(result.model_dump_json()))

    def test_unknown_boundary_id_rejected(self):
        _, result = self.run_fixture()
        data=result.model_dump()
        data["novelty_boundary"]["established_components"][0]["evidence_ids"]=["invented"]
        with self.assertRaises(ValueError):
            ClaimCheckResult(**data)

    def test_absence_conclusion_cannot_outlive_search_provenance(self):
        _, result=self.run_fixture()
        data=result.model_dump(); data["search_trace"]=[]
        with self.assertRaisesRegex(ValueError,"derived from actual search"):
            ClaimCheckResult(**data)

    def test_renamed_duplicate_queries_do_not_inflate_coverage(self):
        _, result=self.run_fixture()
        data=result.model_dump()
        for trace in data["search_trace"]:
            trace["query"]=" SAME   query "
        with self.assertRaisesRegex(ValueError,"derived from actual search"):
            ClaimCheckResult(**data)

    def test_markdown_has_all_sections_and_citations(self):
        _, result = self.run_fixture()
        md=claim_markdown(result)
        for text in ["Your Research Claim","Claim Decomposition","Closest Prior Work","Same / useful", "Different / not established", "Potential Novelty Boundary", "Sources", CAVEAT, "[P1, p. 1]"]:
            self.assertIn(text,md)

    def test_html_escapes_user_text(self):
        f=fixture()
        result=f.agent.run(ClaimCheckRequest(claim="<script>alert(1)</script>"))
        html=claim_html(result)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;",html)

    def test_rendered_citations_link_real_passages(self):
        _, result=self.run_fixture()
        html=claim_html(result)
        import re
        self.assertTrue(set(re.findall('href="#([^\"]+)"',html)) <= set(re.findall('id="([^\"]+)"',html)))
        self.assertIn(TEXTS["A"],html)

    def test_production_has_no_evaluation_file_reads(self):
        for path in Path("src/researchpilot").glob("*claim*.py"):
            text=path.read_text(encoding="utf-8")
            for marker in ["eval/datasets", "eval/runs", "gold_labels", "human_gold"]:
                self.assertNotIn(marker,text)
