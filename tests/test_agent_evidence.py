import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
from openai import OpenAI
from pydantic import ValidationError

from phase5_helpers import OfflineTest, QUESTION, passage
from researchpilot.evidence import (
    AnswerClaim, AnswerDraft, AtomicAssertion, ClaimVerification, DocumentPage, EvidenceSelection,
    VerifiedClaim, validate_draft, validate_verification,
)
from researchpilot.openai_evidence_client import (
    OpenAIClaimSynthesizer, OpenAIClaimVerifier, OpenAIEvidenceSelector,
    _SupportCheck, _VerificationAudit, _ClaimVerdict,
)
from researchpilot.research_models import ResearchRequest


class ContractTests(OfflineTest):
    def test_request_defaults_and_trimmed_question(self):
        request = ResearchRequest(question="  question  ")
        self.assertEqual(request.question, "question")
        self.assertEqual((request.max_papers, request.max_queries, request.top_passages), (6, 5, 20))

    def test_invalid_requests(self):
        for data in [dict(question=" "), dict(question=None), dict(question=12),
                     dict(max_papers=0), dict(max_papers=True), dict(max_papers=21),
                     dict(max_queries=2), dict(max_queries=6), dict(top_passages=0),
                     dict(top_passages=51), dict(top_passages=2.5), dict(year_from=True),
                     dict(year_to="2025"), dict(year_from=2026, year_to=2020)]:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                ResearchRequest(**(dict(question=QUESTION) | data))

    def test_equal_year_bounds_and_none(self):
        self.assertEqual(ResearchRequest(question=QUESTION, year_from=2020, year_to=2020).year_to, 2020)
        self.assertIsNone(ResearchRequest(question=QUESTION).year_to)

    def test_pdf_and_abstract_page_invariant(self):
        for source, number in [("pdf", None), ("pdf", 0), ("abstract", 1), ("pdf", True)]:
            with self.subTest(source=source, number=number), self.assertRaises(ValidationError):
                DocumentPage(paper_id="p", title="t", source_type=source, page_number=number, text="evidence")

    def test_selection_schema_and_unique_ids(self):
        good = dict(selected_passage_ids=[], coverage_notes="No evidence", insufficient_evidence=True)
        self.assertTrue(EvidenceSelection(**good).insufficient_evidence)
        for data in [dict(selected_passage_ids=["e1", "e1"]), dict(coverage_notes=" "), dict(insufficient_evidence="true")]:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                EvidenceSelection(**(good | data))

    def test_claim_requires_nonempty_text_id_and_evidence(self):
        for change in [dict(claim_id=" "), dict(text=" "), dict(evidence_ids=[]), dict(evidence_ids=["e", "e"])]:
            with self.subTest(change=change), self.assertRaises(ValidationError):
                AnswerClaim(**(dict(claim_id="c", text="claim", evidence_ids=["e"]) | change))

    def test_duplicate_claim_ids_rejected(self):
        claim = AnswerClaim(claim_id="c", text="claim", evidence_ids=["e"])
        with self.assertRaises(ValidationError):
            AnswerDraft(claims=[claim, claim], limitations=[])

    def test_unknown_synthesis_evidence_is_rejected(self):
        draft = AnswerDraft(claims=[AnswerClaim(claim_id="c", text="claim", evidence_ids=["invented"])], limitations=[])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate_draft(draft, [passage()])

    def test_verification_rejects_wrong_claim_or_uncited_evidence(self):
        claim = AnswerClaim(claim_id="c", text="claim", evidence_ids=["e1"])
        for cid, ids in [("other", ["e1"]), ("c", ["e2"]), ("c", ["e1", "e2"])]:
            with self.subTest(cid=cid, ids=ids), self.assertRaises(ValueError):
                validate_verification(claim, ClaimVerification(claim_id=cid, evidence_ids=ids,
                                       status="supported", reason="reason"))

    def test_audit_record_rejects_mismatched_claim(self):
        with self.assertRaises(ValidationError):
            VerifiedClaim(claim=AnswerClaim(claim_id="c", text="claim", evidence_ids=["e1"]),
                          verification=ClaimVerification(claim_id="other", evidence_ids=["e1"],
                                                         status="supported", reason="reason"))


class AdapterTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.sdk = MagicMock()
        self.sdk.with_options.return_value = self.sdk
        self.selection = EvidenceSelection(selected_passage_ids=["e1"], coverage_notes="Direct evidence", insufficient_evidence=False)
        self.claim = AnswerClaim(claim_id="c1", text="Caches reduce latency for repeated reads.", evidence_ids=["e1"])
        self.draft = AnswerDraft(claims=[self.claim], limitations=[])
        self.verification = ClaimVerification(claim_id="c1", status="supported", evidence_ids=["e1"], reason="Stated in the evidence.")
        self.verification.assertions = [AtomicAssertion(assertion_id="A1", text=self.claim.text,
            kind="factual", status="supported", evidence_ids=["e1"], reason="Stated in evidence.",
            scope_supported=True, inference_supported=True, quantities_supported=True)]
        self.response(self.selection)

    def response(self, value, status="completed", output=None):
        if isinstance(value, ClaimVerification):
            value = self.audit(value)
        self.sdk.responses.parse.return_value = SimpleNamespace(status=status, output_parsed=value, output=output or [])

    def audit(self, verification, **checks):
        defaults = {name: _SupportCheck(supported=True, reason="Supported or not applicable.")
                    for name in ("scope", "inference", "quantities")}
        atoms = verification.assertions or self.verification.assertions
        atoms = [a.model_dump() | {"evidence_ids": ["E1" if key == "e1" else "E99" for key in a.evidence_ids]}
                 for a in atoms]
        return _VerificationAudit(verification=_ClaimVerdict(status=verification.status, reason=verification.reason),
                                  assertions=atoms, **(defaults | checks))

    def test_selection_structured_output_payload_and_no_scores(self):
        result = OpenAIEvidenceSelector(client=self.sdk).select(QUESTION, [passage()])
        self.assertEqual(result, self.selection)
        call = self.sdk.responses.parse.call_args.kwargs
        self.assertIs(call["text_format"], EvidenceSelection)
        self.assertFalse(call["store"])
        self.assertEqual(call["model"], "gpt-5.6-terra")
        self.assertEqual(set(call), {"model", "store", "instructions", "input", "text_format"})
        payload = json.loads(call["input"])
        self.assertEqual(set(payload), {"research_question", "passages"})
        self.assertEqual(payload["research_question"], QUESTION)
        self.assertEqual(payload["passages"], [passage().model_dump()])
        self.sdk.with_options.assert_called_once_with(max_retries=0, timeout=60.0)
        self.sdk.close.assert_not_called()

    def test_synthesis_structured_schema_and_custom_model(self):
        self.response(self.draft)
        result = OpenAIClaimSynthesizer(model="configured-model", client=self.sdk).synthesize(QUESTION, [passage()])
        self.assertEqual(result, self.draft)
        self.assertIs(self.sdk.responses.parse.call_args.kwargs["text_format"], AnswerDraft)
        self.assertEqual(self.sdk.responses.parse.call_args.kwargs["model"], "configured-model")

    def test_verifier_only_receives_cited_passages(self):
        self.response(self.verification)
        result = OpenAIClaimVerifier(client=self.sdk).verify(self.claim, [passage(), passage("extra")])
        payload = json.loads(self.sdk.responses.parse.call_args.kwargs["input"])
        self.assertEqual([p["passage_id"] for p in payload["evidence"]], ["E1"])
        self.assertEqual(set(payload), {"claim", "evidence"})
        self.assertEqual(result.status, "supported")

    def test_observation_does_not_override_failed_explanatory_check(self):
        self.claim = AnswerClaim(claim_id="c1", evidence_ids=["e1"],
                                 text="Latency fell in the trial, so caching caused the improvement.")
        self.response(self.audit(self.verification, inference=_SupportCheck(
            supported=False, reason="The observation does not identify caching as the cause.")))
        result = OpenAIClaimVerifier(client=self.sdk).verify(
            self.claim, [passage(text="Latency fell in one trial; the cause needs further study.")])
        self.assertEqual(result.status, "partially_supported")
        self.assertIn("does not identify", result.reason)
        self.assertEqual(result.evidence_ids, ["e1"])

    def test_historical_comparison_does_not_override_failed_scope_check(self):
        self.claim = AnswerClaim(claim_id="c1", evidence_ids=["e1"],
                                 text="Current systems remain slower than the baseline.")
        self.response(self.audit(self.verification, scope=_SupportCheck(
            supported=False, reason="The passage describes earlier systems, not the current method.")))
        result = OpenAIClaimVerifier(client=self.sdk).verify(
            self.claim, [passage(text="Earlier systems were slower. Our method addresses this limitation.")])
        self.assertEqual(result.status, "partially_supported")
        self.assertIn("earlier systems", result.reason)

    def test_ambiguous_denominator_prevents_fully_supported_formula(self):
        self.claim = AnswerClaim(claim_id="c1", evidence_ids=["e1"],
                                 text="The stated bound is C times sqrt(n).")
        self.response(self.audit(self.verification, quantities=_SupportCheck(
            supported=False, reason="The extracted formula includes an unresolved denominator 3.")))
        result = OpenAIClaimVerifier(client=self.sdk).verify(
            self.claim, [passage(text="The stated bound: C sqrt(n) 3 .")])
        self.assertEqual(result.status, "partially_supported")
        self.assertIn("denominator 3", result.reason)

    def test_failed_checks_never_promote_unsupported_or_conflicting_verdict(self):
        for status in ("unsupported", "conflicting", "partially_supported"):
            with self.subTest(status=status):
                original = self.verification.model_copy(update={"status": status})
                audit = self.audit(original, scope=_SupportCheck(supported=False, reason="Scope missing."))
                before = audit.model_dump()
                self.response(audit)
                result = OpenAIClaimVerifier(client=self.sdk).verify(self.claim, [passage()])
                self.assertEqual(result.status, status)
                self.assertEqual(audit.model_dump(), before)

    def test_checks_are_required_in_provider_schema_and_not_in_public_result(self):
        with self.assertRaises(ValidationError):
            _VerificationAudit(verification=self.verification)
        self.response(self.verification)
        result = OpenAIClaimVerifier(client=self.sdk).verify(self.claim, [passage()])
        call = self.sdk.responses.parse.call_args.kwargs
        self.assertIs(call["text_format"], _VerificationAudit)
        self.assertEqual(type(result), ClaimVerification)
        self.assertEqual(result.model_dump(), self.verification.model_dump())
        for term in ("Historical background", "causal", "denominator", "ALL its assertions"):
            self.assertIn(term, call["instructions"])

    def test_downgraded_claim_cannot_be_declared_sufficient(self):
        from researchpilot.research_iteration import EvidenceAssessment, validate_assessment
        self.response(self.audit(self.verification, inference=_SupportCheck(
            supported=False, reason="An explanation is not established.")))
        result = OpenAIClaimVerifier(client=self.sdk).verify(self.claim, [passage()])
        record = VerifiedClaim(claim=self.claim, verification=result)
        with self.assertRaises(ValueError):
            validate_assessment(EvidenceAssessment(sufficient=True, gaps=[], rationale="Enough."),
                                [record], [passage()])

    def test_unknown_selected_id_rejected(self):
        self.response(EvidenceSelection(selected_passage_ids=["invented"], coverage_notes="x", insufficient_evidence=False))
        with self.assertRaises(ValueError):
            OpenAIEvidenceSelector(client=self.sdk).select(QUESTION, [passage()])

    def test_unknown_synthesized_id_rejected(self):
        self.response(AnswerDraft(claims=[AnswerClaim(claim_id="c", text="x", evidence_ids=["invented"])], limitations=[]))
        with self.assertRaises(ValueError):
            OpenAIClaimSynthesizer(client=self.sdk).synthesize(QUESTION, [passage()])

    def test_verifier_unknown_alias_rejected_even_if_passage_available(self):
        audit = self.audit(self.verification)
        audit.assertions[0].evidence_ids = ["E99"]
        self.response(audit)
        with self.assertRaisesRegex(ValueError, "Unknown evidence aliases"):
            OpenAIClaimVerifier(client=self.sdk).verify(self.claim, [passage(), passage("extra")])

    def test_invalid_claim_input_prevents_verifier_call(self):
        with self.assertRaises(ValueError):
            OpenAIClaimVerifier(client=self.sdk).verify(self.claim, [])
        self.sdk.responses.parse.assert_not_called()

    def test_refusal_incomplete_missing_or_wrong_output_rejected(self):
        refusal = [SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal")])]
        for value, status, output in [(None, "completed", []), (self.selection, "incomplete", []),
                                       (self.selection, "completed", refusal), (self.draft, "completed", [])]:
            self.response(value, status, output)
            with self.subTest(status=status, value=type(value).__name__), self.assertRaises(RuntimeError):
                OpenAIEvidenceSelector(client=self.sdk).select(QUESTION, [passage()])

    def test_provider_failure_propagates(self):
        self.sdk.responses.parse.side_effect = RuntimeError("offline failure")
        with self.assertRaisesRegex(RuntimeError, "offline failure"):
            OpenAIEvidenceSelector(client=self.sdk).select(QUESTION, [passage()])

    def test_real_sdk_parses_all_three_schemas_over_offline_transport(self):
        seen = []
        responses = [self.selection, self.draft, self.audit(self.verification)]
        def handler(request):
            body = json.loads(request.content)
            seen.append(body)
            parsed = responses[len(seen) - 1]
            return httpx.Response(200, json={
                "id": "resp_offline", "object": "response", "created_at": 0,
                "status": "completed", "model": "gpt-5.6-terra", "output": [{
                    "id": "msg_offline", "type": "message", "status": "completed", "role": "assistant",
                    "content": [{"type": "output_text", "text": parsed.model_dump_json(), "annotations": []}],
                }],
            })
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            with OpenAI(api_key="offline-only", http_client=http, max_retries=0) as sdk:
                self.assertEqual(OpenAIEvidenceSelector(client=sdk).select(QUESTION, [passage()]), self.selection)
                self.assertEqual(OpenAIClaimSynthesizer(client=sdk).synthesize(QUESTION, [passage()]), self.draft)
                self.assertEqual(OpenAIClaimVerifier(client=sdk).verify(self.claim, [passage()]), self.verification)
        self.assertEqual(len(seen), 3)
        for body in seen:
            self.assertFalse(body["store"])
            self.assertTrue(body["text"]["format"]["strict"])
            self.assertEqual(body["text"]["format"]["type"], "json_schema")
            self.assertNotIn("tools", body)
