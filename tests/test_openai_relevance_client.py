"""Offline adapter tests: SDK parsing calls and HTTP dispatch are mocked."""

import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from openai import OpenAI
from pydantic import ValidationError

from researchpilot.openai_relevance_client import OpenAIRelevanceClient
from researchpilot.relevance import RelevanceAssessment


class OpenAIRelevanceClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assessment = RelevanceAssessment(
            category="direct", score=0.9, reason="Studies fixed sampling patterns."
        )
        self.response = SimpleNamespace(
            status="completed", output_parsed=self.assessment
        )
        self.sdk = MagicMock()
        self.sdk.with_options.return_value = self.sdk
        self.sdk.responses.parse.return_value = self.response
        self.client = OpenAIRelevanceClient(client=self.sdk)

    def test_default_model_and_pydantic_structured_output_are_used(self) -> None:
        result = self.client.assess("Question", "Title", "Abstract")

        self.assertIs(result, self.assessment)
        self.sdk.responses.parse.assert_called_once()
        params = self.sdk.responses.parse.call_args.kwargs
        self.assertEqual(params["model"], "gpt-5.6-terra")
        self.assertIs(params["text_format"], RelevanceAssessment)
        self.assertIs(params["store"], False)
        self.assertEqual(
            set(params), {"model", "store", "instructions", "input", "text_format"}
        )

    def test_model_is_configurable(self) -> None:
        client = OpenAIRelevanceClient(model="configured-model", client=self.sdk)

        client.assess("Question", "Title", None)

        self.assertEqual(
            self.sdk.responses.parse.call_args.kwargs["model"], "configured-model"
        )

    def test_only_question_title_and_abstract_are_sent_unchanged(self) -> None:
        question = "  How does spectral expansion help?\n"
        title = '  A title with "quotes" and α\n'
        abstract = "  First line.\nSecond line.\t"

        self.client.assess(question, title, abstract)

        payload = json.loads(self.sdk.responses.parse.call_args.kwargs["input"])
        self.assertEqual(
            payload,
            {"research_question": question, "title": title, "abstract": abstract},
        )

    def test_none_abstract_is_sent_as_null_with_conservative_instruction(self) -> None:
        self.client.assess("Question", "Title", None)

        params = self.sdk.responses.parse.call_args.kwargs
        self.assertIsNone(json.loads(params["input"])["abstract"])
        self.assertIn("judge conservatively", params["instructions"])
        self.assertIn("only the title", params["instructions"])
        self.assertIn("explicitly state the uncertainty", params["instructions"])

    def test_instructions_define_the_rubric_and_limit_evidence(self) -> None:
        self.client.assess("Question", "Title", "Abstract")

        instructions = " ".join(
            self.sdk.responses.parse.call_args.kwargs["instructions"].split()
        )
        for phrase in (
            "DIRECT / CORE (direct)",
            "SUPPORTING (supporting)",
            "OFF_TARGET (off_target)",
            "category describes the paper's ROLE in answering the question",
            "core literature needed to answer the research question",
            "target problem setting, an essential component of that setting",
            "specific relationship or mechanism being investigated",
            "does NOT need to cover every concept in a multi-part research question",
            "fixed-pattern matrix completion can be DIRECT even if spectral "
            "expansion is not discussed",
            "paper's main problem is not the target problem itself",
            "theory or methodology directly useful",
            "spectral-gap paper on bipartite biregular graphs may be SUPPORTING "
            "when its results are applicable to deterministic matrix completion",
            "not materially useful",
            "overall usefulness for answering the question, not keyword or concept "
            "coverage",
            "core paper can receive a high score while addressing only one "
            "essential component of the question",
            "Use only the supplied title and abstract",
            "Do not assume facts",
            "Keep the reason concise and grounded in the supplied evidence",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, instructions)

    def test_injected_client_has_retries_disabled_and_finite_timeout(self) -> None:
        self.client.assess("Question", "Title", None)

        self.sdk.with_options.assert_called_once_with(max_retries=0, timeout=60.0)
        self.sdk.close.assert_not_called()
        self.sdk.__exit__.assert_not_called()

    def test_missing_structured_output_or_refusal_raises(self) -> None:
        self.response.output_parsed = None

        with self.assertRaisesRegex(RuntimeError, "refusal or missing output"):
            self.client.assess("Question", "Title", None)

        self.sdk.responses.parse.assert_called_once()

    def test_incomplete_or_failed_response_is_not_used(self) -> None:
        for status in ("incomplete", "failed"):
            with self.subTest(status=status):
                self.response.status = status
                with self.assertRaisesRegex(RuntimeError, "not completed"):
                    self.client.assess("Question", "Title", None)

    def test_sdk_exception_propagates_without_retry(self) -> None:
        failure = RuntimeError("SDK request failed")
        self.sdk.responses.parse.side_effect = failure

        with self.assertRaises(RuntimeError) as caught:
            self.client.assess("Question", "Title", "Abstract")

        self.assertIs(caught.exception, failure)
        self.sdk.responses.parse.assert_called_once()

    def test_sdk_validation_exception_propagates_without_fallback(self) -> None:
        try:
            RelevanceAssessment(category="direct", score=1.1, reason="Reason")
        except ValidationError as error:
            failure = error
        self.sdk.responses.parse.side_effect = failure

        with self.assertRaises(ValidationError) as caught:
            self.client.assess("Question", "Title", None)

        self.assertIs(caught.exception, failure)
        self.sdk.responses.parse.assert_called_once()

    def test_real_sdk_builds_strict_json_schema_with_http_dispatch_mocked(self) -> None:
        # Exercise the installed SDK's schema conversion, stopping before I/O.
        with OpenAI(api_key="offline-test-key") as sdk:
            with patch.object(OpenAI, "post", return_value=self.response) as post:
                result = OpenAIRelevanceClient(client=sdk).assess(
                    "Question", "Title", None
                )

            self.assertIs(result, self.assessment)
            self.assertFalse(sdk.is_closed())
            self.assertEqual(sdk.max_retries, 2)  # Caller configuration is unchanged.
            post.assert_called_once()
            self.assertEqual(post.call_args.args[0], "/responses")
            body = post.call_args.kwargs["body"]
            output_format = body["text"]["format"]
            self.assertEqual(output_format["type"], "json_schema")
            self.assertIs(output_format["strict"], True)
            schema = output_format["schema"]
            self.assertEqual(set(schema["required"]), {"category", "score", "reason"})
            self.assertFalse(schema["additionalProperties"])
            properties = schema["properties"]
            self.assertEqual(set(properties), {"category", "score", "reason"})
            self.assertEqual(
                properties["category"]["enum"], ["direct", "supporting", "off_target"]
            )
            self.assertEqual(properties["score"]["minimum"], 0.0)
            self.assertEqual(properties["score"]["maximum"], 1.0)
            self.assertEqual(properties["reason"]["minLength"], 1)

    def test_default_sdk_uses_environment_and_closes_with_http_dispatch_mocked(self) -> None:
        created = []

        def record_client(sdk):
            created.append(sdk)
            return self.response

        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-environment-key"}):
            with patch.object(OpenAI, "post", autospec=True) as post:
                post.side_effect = lambda sdk, *args, **kwargs: record_client(sdk)
                client = OpenAIRelevanceClient()
                post.assert_not_called()

                result = client.assess("Question", "Title", None)

        self.assertIs(result, self.assessment)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].api_key, "offline-environment-key")
        self.assertEqual(created[0].max_retries, 0)
        self.assertEqual(created[0].timeout, 60.0)
        self.assertTrue(created[0].is_closed())


if __name__ == "__main__":
    unittest.main()
