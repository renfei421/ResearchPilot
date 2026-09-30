"""Offline checks of the Phase 11.1 acceptance invariants."""

from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from claim_check_helpers import CLAIM, ClaimCheckRequest, ClaimCheckResult, RelationBatch, fixture, relation
from phase5_helpers import OfflineTest
from researchpilot.claim_check import ClaimPaperRelation
from researchpilot.claim_literature_map import build_maps
from researchpilot.openai_claim_check import OpenAIClaimRelationAnalyzer, _RelationJudgments


class ClaimAcceptanceTests(OfflineTest):
    def result(self):
        return fixture().agent.run(ClaimCheckRequest(claim=CLAIM))

    def test_partial_match_prevents_absence_conclusion_and_has_precise_boundary(self):
        result = self.result()
        m = next(m for m in result.claim_maps if m.claim_id == "C3")
        self.assertEqual(m.coverage_status, "strong")
        self.assertFalse(m.no_close_match_found)
        missing = result.novelty_boundary.missing_evidence[0].text
        self.assertIn("partial overlap was found", missing)
        self.assertNotIn("coverage remains insufficient", missing)

    def test_stored_partial_match_cannot_claim_no_close_match(self):
        data = self.result().model_dump()
        data["claim_maps"][2]["no_close_match_found"] = True
        with self.assertRaisesRegex(ValueError, "without direct or partial overlap"):
            ClaimCheckResult.model_validate(data)

    def test_sufficient_search_with_only_support_still_allows_no_close_match(self):
        result = self.result()
        rows = [relation("C3", "A", "SUPPORTING"), relation("C3", "B", "BRIDGING")]
        m = build_maps(result.decomposition, rows, result.search_trace)[2]
        self.assertEqual(m.coverage_status, "moderate")
        self.assertTrue(m.no_close_match_found)

    def test_duplicate_dimension_alias_rejected(self):
        row = relation(pid="P1", ids=["E1"]).model_dump()
        row["comparisons"][0]["evidence_ids"] = ["E1", "E1"]
        with self.assertRaisesRegex(ValueError, "dimension evidence IDs"):
            ClaimPaperRelation.model_validate(row)

    def test_duplicate_relation_alias_rejected(self):
        with self.assertRaisesRegex(ValueError, "relation evidence IDs"):
            relation(pid="P1", ids=["E1", "E1"])

    def test_adapter_rejects_duplicates_even_from_constructed_objects(self):
        result = self.result()
        p = next(p for p in result.evidence if p.paper_id == "A")
        row = relation(pid="P1", ids=["E1"])
        row.comparisons[0].evidence_ids.append("E1")
        sdk = MagicMock()
        sdk.with_options.return_value = sdk
        sdk.responses.parse.return_value = SimpleNamespace(status="completed", output=[],
            output_parsed=_RelationJudgments.model_validate({"relations":[
                row.model_dump(exclude={"paper_id", "evidence_ids"})]}))
        with self.assertRaisesRegex(ValueError, "evidence handles"):
            OpenAIClaimRelationAnalyzer(client=sdk).analyze([result.decomposition.assertions[0]], "A", [p], [])

    def test_relation_evidence_is_derived_from_all_dimension_citations(self):
        from researchpilot.evidence import EvidencePassage
        p = EvidencePassage(passage_id="passage:one", paper_id="A", title="A",
            text="A scoped component.", source_type="abstract", page_number=None)
        q = p.model_copy(update={"passage_id":"passage:two", "text":"Another scoped component."})
        row = relation(pid="P1", label="PARTIAL_OVERLAP", ids=["E1"]).model_dump(exclude={"paper_id","evidence_ids"})
        row["comparisons"][1]["evidence_ids"] = ["E2", "E1"]
        sdk = MagicMock()
        sdk.with_options.return_value = sdk
        sdk.responses.parse.return_value = SimpleNamespace(status="completed",output=[],
            output_parsed=_RelationJudgments.model_validate({"relations":[row]}))
        result = OpenAIClaimRelationAnalyzer(client=sdk).analyze([self.result().decomposition.assertions[0]],"A",[p,q],[])
        self.assertEqual(result.relations[0].evidence_ids,[p.passage_id,q.passage_id])
        self.assertEqual(result.relations[0].paper_id,"A")
        self.assertEqual(result.relations[0].relation,"PARTIAL_OVERLAP")
        schema = sdk.responses.parse.call_args.kwargs["text_format"].model_json_schema()
        fields = schema["$defs"]["_RelationJudgment"]["properties"]
        self.assertNotIn("evidence_ids",fields)
        self.assertNotIn("paper_id",fields)
        self.assertEqual(sdk.responses.parse.call_count,1)

    def test_derived_bookkeeping_does_not_relax_semantic_scope(self):
        row = relation(label="DIRECT_OVERLAP").model_dump(exclude={"paper_id","evidence_ids"})
        row["comparisons"][0].update(alignment="unknown",evidence_ids=[])
        sdk=MagicMock()
        sdk.with_options.return_value=sdk
        sdk.responses.parse.return_value=SimpleNamespace(status="completed",output=[],
            output_parsed=_RelationJudgments.model_validate({"relations":[row]}))
        # Use a genuine same-paper input and valid aliases, so only the scope rule fails.
        result=self.result()
        p=next(p for p in result.evidence if p.paper_id=="A")
        for d in row["comparisons"][1:]: d["evidence_ids"]=["E1"]
        sdk.responses.parse.return_value.output_parsed=_RelationJudgments.model_validate({"relations":[row]})
        with self.assertRaisesRegex(ValueError,"Direct overlap requires all applicable dimensions"):
            OpenAIClaimRelationAnalyzer(client=sdk).analyze([result.decomposition.assertions[0]],"A",[p],[])

    def test_rescue_ignores_unbound_assembly_gaps_and_checks_all_claim_references(self):
        from researchpilot.research_models import EvidenceGapRecord
        f=fixture()
        # Keep all assertions below strong coverage so local rescue is exercised.
        def analyze(assertions,pid,evidence,bundles):
            return RelationBatch(relations=[relation(a.claim_id,pid,"SUPPORTING",[evidence[0].passage_id])
                for a in assertions])
        f.analyzer.analyze=analyze
        def assemble(state,services):
            def gap(key,ids):
                return EvidenceGapRecord(gap_id=key,description="Missing condition",search_focus="condition",
                    related_claim_ids=ids,severity="useful",first_seen_round=1,last_updated_round=1)
            return {"gap_ledger":[gap("assembly:unbound",[]),gap("unbound",[]),
                gap("assembly:bound",["C1"]),gap("shared",["unrelated","C3"])]}
        with patch("researchpilot.claim_novelty_agent.assemble_selected",side_effect=assemble), \
             patch.object(f.services.document_rescuer,"plan",return_value=[]) as planner:
            result=f.agent.run(ClaimCheckRequest(claim=CLAIM,max_search_rounds=1))
        self.assertEqual([g.gap_id for g in planner.call_args.args[1]],["shared"])
        self.assertTrue(result.relations)
