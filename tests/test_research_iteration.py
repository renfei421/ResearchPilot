"""Offline contracts and actual SDK parsing for iterative research stages."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import OpenAI
from pydantic import ValidationError

from phase5_helpers import OfflineTest, QUESTION, passage
from researchpilot.evidence import AnswerClaim, ClaimVerification, VerifiedClaim
from researchpilot.openai_research_iteration import OpenAIEvidenceAssessor, OpenAIFollowUpPlanner
from researchpilot.paper_candidate import SearchQuery
from researchpilot.research_iteration import (
    EvidenceAssessment, EvidenceGap, FollowUpQuery, FollowUpSearchPlan,
    new_follow_up_queries, query_key, validate_assessment,
)
from researchpilot.research_models import ResearchRequest


def gap():
    return EvidenceGap(gap_id="g1", description="Evidence on cache contention latency is missing.",
                       related_claim_ids=[], severity="critical", search_focus="cache contention latency")


def follow(query_id="f1", text='"cache contention" AND latency', gap_id="g1"):
    return FollowUpQuery(query_id=query_id, text=text, gap_id=gap_id,
                         rationale="Find latency evidence for the missing contention setting.")


def verified():
    return VerifiedClaim(claim=AnswerClaim(claim_id="c1", text="Repeated reads benefit.", evidence_ids=["e1"]),
                         verification=ClaimVerification(claim_id="c1", status="supported", evidence_ids=["e1"], reason="Explicit result."))


class IterationContractTests(OfflineTest):
    def test_request_defaults_and_all_budgets(self):
        r = ResearchRequest(question=QUESTION)
        self.assertEqual((r.max_search_rounds, r.max_follow_up_queries), (2, 3))
        for field in ("max_search_rounds", "max_follow_up_queries"):
            for value in (0, 4, True, "2", 1.5):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    ResearchRequest(question=QUESTION, **{field: value})
            for value in (1, 2, 3):
                self.assertEqual(getattr(ResearchRequest(question=QUESTION, **{field: value}), field), value)

    def test_query_normalization_only_changes_comparison_key(self):
        text = '  "CACHE  contention"   AND latency  '
        self.assertEqual(query_key(text), '"cache contention" and latency')
        planned = FollowUpSearchPlan(queries=[follow(text=text)])
        self.assertEqual(new_follow_up_queries(planned, [gap()], [], 3)[0].text, text.strip())

    def test_duplicate_texts_filtered_against_history_and_current_plan(self):
        executed = [SearchQuery(query_id="old", text='"CACHE CONTENTION" and latency')]
        plan = FollowUpSearchPlan(queries=[follow(), follow("f2", "new query"), follow("f3", "NEW   query")])
        result = new_follow_up_queries(plan, [gap()], executed, 3)
        self.assertEqual([q.query_id for q in result], ["f2"])

    def test_all_duplicates_returns_no_queries(self):
        self.assertEqual(new_follow_up_queries(FollowUpSearchPlan(queries=[follow()]), [gap()],
                         [SearchQuery(query_id="old", text=follow().text)], 3), [])

    def test_unknown_gap_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown gap"):
            new_follow_up_queries(FollowUpSearchPlan(queries=[follow(gap_id="invented")]), [gap()], [], 3)

    def test_new_text_cannot_reuse_old_query_id(self):
        with self.assertRaisesRegex(ValueError, "unused query_id"):
            new_follow_up_queries(FollowUpSearchPlan(queries=[follow()]), [gap()],
                                  [SearchQuery(query_id="f1", text="old query")], 3)

    def test_follow_up_budget_enforced(self):
        with self.assertRaises(ValueError):
            new_follow_up_queries(FollowUpSearchPlan(queries=[follow(), follow("f2", "different")]), [gap()], [], 1)
        with self.assertRaises(ValueError):
            new_follow_up_queries(FollowUpSearchPlan(queries=[]), [], [], True)

    def test_empty_plan_is_valid(self):
        self.assertEqual(new_follow_up_queries(FollowUpSearchPlan(queries=[]), [gap()], [], 3), [])

    def test_query_guards_reuse_original_planner_validation(self):
        for text in ("", " ", "text\n", "https://example.org", "filter=year:2020"):
            with self.subTest(text=text), self.assertRaises(ValidationError):
                follow(text=text)

    def test_duplicate_query_ids_and_too_many_queries_rejected(self):
        for queries in ([follow(), follow()], [follow(str(i), str(i)) for i in range(4)]):
            with self.assertRaises(ValidationError):
                FollowUpSearchPlan(queries=queries)

    def test_gap_fields_and_severity_validated(self):
        for update in (dict(description=" "), dict(search_focus=" "), dict(severity="optional"), dict(related_claim_ids=["c", "c"])):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                EvidenceGap.model_validate(gap().model_dump() | update)

    def test_assessment_cannot_claim_sufficiency_with_gaps(self):
        with self.assertRaises(ValidationError):
            EvidenceAssessment(sufficient=True, gaps=[gap()], rationale="x")

    def test_duplicate_gap_ids_rejected(self):
        with self.assertRaises(ValidationError):
            EvidenceAssessment(sufficient=False, gaps=[gap(), gap()], rationale="x")

    def test_unknown_related_claim_id_rejected(self):
        g = gap().model_copy(update={"related_claim_ids": ["invented"]})
        with self.assertRaisesRegex(ValueError, "Unknown related"):
            validate_assessment(EvidenceAssessment(sufficient=False, gaps=[g], rationale="x"), [verified()], [passage()])

    def test_sufficiency_without_verified_evidence_rejected(self):
        with self.assertRaises(ValueError):
            validate_assessment(EvidenceAssessment(sufficient=True, gaps=[], rationale="x"), [], [])

    def test_sufficiency_cannot_override_uncertain_claim_statuses(self):
        for status in ("unsupported", "partially_supported", "conflicting"):
            record = verified()
            record.verification.status = status
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "cannot override"):
                validate_assessment(EvidenceAssessment(sufficient=True, gaps=[], rationale="x"), [record], [passage()])


class IterationAdapterTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.sdk = Mock()
        self.sdk.with_options.return_value = self.sdk
        self.assessment = EvidenceAssessment(sufficient=False, gaps=[gap()], rationale="Missing contention result.")
        self.follow_up = FollowUpSearchPlan(queries=[follow()])

    def response(self, value):
        self.sdk.responses.parse.return_value = SimpleNamespace(status="completed", output=[], output_parsed=value)

    def test_assessor_receives_actual_evidence_claims_and_statuses_only(self):
        self.response(self.assessment)
        result = OpenAIEvidenceAssessor(client=self.sdk).assess(QUESTION, [verified()], [passage()])
        call = self.sdk.responses.parse.call_args.kwargs
        self.assertEqual(json.loads(call["input"]), {"research_question": QUESTION,
                         "verified_claims": [verified().model_dump()], "evidence": [passage().model_dump()], "prior_gaps": []})
        self.assertIs(call["text_format"], EvidenceAssessment)
        self.assertEqual(call["model"], "gpt-5.6-terra")
        self.assertFalse(call["store"])
        self.assertEqual(result, self.assessment)
        self.sdk.with_options.assert_called_once_with(max_retries=0, timeout=60.0)

    def test_follow_up_payload_targets_real_gaps_and_history(self):
        self.response(self.follow_up)
        executed = [SearchQuery(query_id="q1", text="shared caches")]
        result = OpenAIFollowUpPlanner(model="custom", client=self.sdk).plan(QUESTION, [gap()], executed, ["latency"], 1)
        call = self.sdk.responses.parse.call_args.kwargs
        payload = json.loads(call["input"])
        self.assertEqual(payload, {"research_question": QUESTION, "evidence_gaps": [gap().model_dump()],
                                  "executed_queries": [q.model_dump() for q in executed], "concepts": ["latency"], "max_queries": 1})
        self.assertEqual(call["model"], "custom")
        self.assertEqual(result.queries[0].gap_id, gap().gap_id)
        self.assertIs(call["text_format"], FollowUpSearchPlan)
        self.assertFalse(call["store"])

    def test_adapter_rejects_unknown_claim_or_gap_ids(self):
        self.response(EvidenceAssessment(sufficient=False, gaps=[gap().model_copy(update={"related_claim_ids": ["unknown"]})], rationale="x"))
        with self.assertRaises(ValueError):
            OpenAIEvidenceAssessor(client=self.sdk).assess(QUESTION, [verified()], [passage()])
        self.response(FollowUpSearchPlan(queries=[follow(gap_id="unknown")]))
        with self.assertRaises(ValueError):
            OpenAIFollowUpPlanner(client=self.sdk).plan(QUESTION, [gap()], [], [], 3)

    def test_refusal_incomplete_wrong_schema_and_api_error(self):
        for value, status in ((None, "completed"), (self.assessment, "incomplete"), (self.follow_up, "completed")):
            self.sdk.responses.parse.return_value = SimpleNamespace(status=status, output=[], output_parsed=value)
            with self.assertRaises(RuntimeError):
                OpenAIEvidenceAssessor(client=self.sdk).assess(QUESTION, [], [])
        self.sdk.responses.parse.side_effect = RuntimeError("provider failure")
        with self.assertRaisesRegex(RuntimeError, "provider failure"):
            OpenAIFollowUpPlanner(client=self.sdk).plan(QUESTION, [gap()], [], [], 3)

    def test_real_sdk_structured_parsing_is_offline(self):
        responses = [self.assessment, self.follow_up]
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, json={"id": "resp_test", "object": "response", "created_at": 0,
                "status": "completed", "model": "gpt-5.6-terra", "output": [{"id": "msg_test", "type": "message",
                "status": "completed", "role": "assistant", "content": [{"type": "output_text",
                "text": responses[len(requests) - 1].model_dump_json(), "annotations": []}]}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            with OpenAI(api_key="offline", http_client=http) as sdk:
                self.assertEqual(OpenAIEvidenceAssessor(client=sdk).assess(QUESTION, [verified()], [passage()]), self.assessment)
                self.assertEqual(OpenAIFollowUpPlanner(client=sdk).plan(QUESTION, [gap()], [], [], 3), self.follow_up)
        for body in requests:
            self.assertTrue(body["text"]["format"]["strict"])
            self.assertFalse(body["store"])
            self.assertNotIn("tools", body)
