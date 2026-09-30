import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from pydantic import ValidationError

from phase5_helpers import OfflineTest, passage
from researchpilot.evidence import AtomicAssertion, ClaimVerification
from researchpilot.math_evidence import MathEvidence, VisualEvidenceRecord, guard_ambiguous_visual_support, visual_passage
from researchpilot.openai_page_evidence_client import OpenAIPageEvidenceClient, visual_evidence_id
from researchpilot.pdf_page_renderer import RenderedPage
from researchpilot.research_iteration import EvidenceGap


def gap(**changes):
    return EvidenceGap(**(dict(gap_id="curvature", description="Need curvature proposition assumptions.",
        related_claim_ids=[], severity="critical", search_focus="spectral curvature") | changes))


def math_item(**changes):
    return MathEvidence(**(dict(evidence_id="visual:test", paper_id="openalex:W1", page_number=1,
        statement_type="proposition", statement_text="Under A, a local curvature bound holds.",
        formula_text="c > 0", assumptions=["A"], conclusion="Local curvature is positive.",
        relevant_to_gap=True, ambiguity=None) | changes))


def visual_record(item=None):
    return VisualEvidenceRecord(evidence=item or math_item(), pdf_sha256="pdfhash", image_sha256="imagehash",
        render_version="render-v1", extraction_version="extract-v1", model="fake", gap_id="curvature",
        parser_passage_ids=["e1"])


class PageEvidenceClientTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.sdk = MagicMock()
        self.sdk.with_options.return_value = self.sdk
        self.page = RenderedPage("openalex:W1", 1, b"ONE PAGE ONLY", "pdf", "image", "render", 100, 120)
        self.client = OpenAIPageEvidenceClient(client=self.sdk)
        self.output(math_item())

    def output(self, item):
        self.sdk.responses.parse.return_value = SimpleNamespace(status="completed", output=[], output_parsed=item)

    def extract(self):
        return self.client.extract("Research question", gap(), "Title", self.page, "visual:test")

    def payload(self):
        self.extract()
        return self.sdk.responses.parse.call_args.kwargs

    def test_only_target_page_sent(self):
        call = self.payload()
        content = call["input"][0]["content"]
        self.assertEqual(len(call["input"]), 1)
        self.assertEqual(len(content), 2)
        self.assertEqual(base64.b64decode(content[1]["image_url"].split(",")[1]), self.page.png)
        self.assertEqual(content[1]["detail"], "high")

    def test_question_and_gap_supplied(self):
        data = json.loads(self.payload()["input"][0]["content"][0]["text"])
        self.assertEqual(data["research_question"], "Research question")
        self.assertEqual(data["evidence_gap"], dict(description=gap().description, search_focus=gap().search_focus))

    def test_no_unrelated_context(self):
        data = json.loads(self.payload()["input"][0]["content"][0]["text"])
        self.assertEqual(set(data), {"research_question", "evidence_gap", "title", "paper_id", "page_number", "evidence_id"})

    def test_structured_schema_and_storage_disabled(self):
        call = self.payload()
        self.assertIs(call["text_format"], MathEvidence)
        self.assertFalse(call["store"])
        self.assertEqual(call["max_output_tokens"], 4000)
        self.assertEqual(call["model"], "gpt-5.6-terra")
        self.sdk.with_options.assert_called_once_with(max_retries=0, timeout=60.0)

    def test_unknown_page_rejected(self):
        self.output(math_item(page_number=2))
        with self.assertRaises(ValueError):
            self.extract()

    def test_unknown_paper_rejected(self):
        self.output(math_item(paper_id="wrong"))
        with self.assertRaises(ValueError):
            self.extract()

    def test_hallucinated_id_rejected(self):
        self.output(math_item(evidence_id="visual:invented"))
        with self.assertRaises(ValueError):
            self.extract()

    def test_visual_ambiguity_preserved(self):
        self.output(math_item(ambiguity="Radical endpoint is unreadable."))
        result = self.extract()
        self.assertEqual(result.ambiguity, "Radical endpoint is unreadable.")
        self.assertIn(result.ambiguity, visual_passage(visual_record(result), passage()).text)

    def test_schema_rejects_unknown_fields_blank_statement_and_invalid_page(self):
        for update in ({"unexpected": "x"}, {"statement_text": " "}, {"page_number": True}):
            with self.subTest(update=update), self.assertRaises(ValidationError):
                math_item(**update)

    def test_failure_propagates_without_retry(self):
        self.sdk.responses.parse.side_effect = RuntimeError("failed")
        with self.assertRaises(RuntimeError):
            self.extract()
        self.assertEqual(self.sdk.responses.parse.call_count, 1)

    def test_incomplete_output_is_not_evidence(self):
        self.sdk.responses.parse.return_value.status = "incomplete"
        with self.assertRaises(RuntimeError):
            self.extract()

    def test_visual_identity_deterministic_and_config_bound(self):
        first = visual_evidence_id(self.page, "q", gap(), "m")
        self.assertEqual(first, visual_evidence_id(self.page, "q", gap(), "m"))
        self.assertTrue(first.startswith("visual:"))
        self.assertNotEqual(first, visual_evidence_id(self.page, "q", gap(), "other"))
        self.assertNotEqual(first, visual_evidence_id(self.page, "different", gap(), "m"))


class AmbiguousEvidenceTests(OfflineTest):
    def verification(self, kind="theoretical", status="supported", ids=None):
        atom = AtomicAssertion(assertion_id="a", text="Under A, curvature exceeds a constant.", kind=kind,
            status=status, evidence_ids=ids or ["visual:test"], reason="Visible statement.",
            scope_supported=True, inference_supported=True, quantities_supported=True)
        return ClaimVerification(claim_id="c", status=status, evidence_ids=atom.evidence_ids,
                                 reason="Source assessment.", assertions=[atom])

    def guard(self, verification, ambiguity="Formula scope unclear"):
        return guard_ambiguous_visual_support(verification, {"visual:test": visual_record(math_item(ambiguity=ambiguity))})

    def test_ambiguous_formula_cannot_be_fully_supported(self):
        self.assertEqual(self.guard(self.verification()).status, "partially_supported")

    def test_ambiguous_quantitative_atom_cannot_be_supported(self):
        atom = self.guard(self.verification(kind="quantitative")).assertions[0]
        self.assertFalse(atom.quantities_supported)

    def test_clearly_visible_scope_atom_remains_supported(self):
        self.assertEqual(self.guard(self.verification(kind="scope")).status, "supported")

    def test_no_mutation(self):
        original = self.verification()
        before = original.model_dump()
        self.guard(original)
        self.assertEqual(original.model_dump(), before)

    def test_clear_transcription_verifier_is_authoritative(self):
        original = self.verification()
        self.assertEqual(self.guard(original, ambiguity=None), original)

    def test_unsupported_verdict_never_promoted(self):
        self.assertEqual(self.guard(self.verification(status="unsupported")).status, "unsupported")

    def test_legacy_non_atomic_ambiguous_verdict_guarded(self):
        original = self.verification().model_copy(update={"assertions": []})
        self.assertEqual(self.guard(original).status, "partially_supported")

    def test_original_parser_text_not_replaced(self):
        source = passage(text="flattened parser formula")
        visual = visual_passage(visual_record(), source)
        self.assertEqual(source.text, "flattened parser formula")
        self.assertNotEqual(source.passage_id, visual.passage_id)
        self.assertEqual((source.paper_id, source.page_number), (visual.paper_id, visual.page_number))

    def test_irrelevant_visual_not_admitted(self):
        with self.assertRaises(ValueError):
            visual_passage(visual_record(math_item(relevant_to_gap=False)), passage())
