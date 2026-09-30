"""Offline evidence-contract regressions; synthetic, not a recovered live payload."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from phase5_helpers import OfflineTest, paper, passage
from phase12_helpers import project, research_fixture, QUESTION
from researchpilot.answer_renderer import render_answer, rendered_evidence_ids, rendered_findings
from researchpilot.evidence import (
    AnswerClaim, ClaimVerification, VerifiedClaim, supported_findings, validate_verification,
)
from researchpilot.openai_evidence_client import OpenAIClaimVerifier, _VerificationAudit
from researchpilot.research_models import ResearchRequest, SelectedPaper
from researchpilot.run_store import RunStore


IDS = [f"passage:long-opaque-source-identifier:{i}:0123456789abcdef" for i in range(1, 4)]


def atom(number=1, ids=("E1",), status="supported", **changes):
    return dict(assertion_id=f"A{number}", text=f"Finding {number} holds in the measured setting.",
        kind="factual", status=status, evidence_ids=list(ids), reason="The cited passage establishes this scope.",
        scope_supported=True, inference_supported=True, quantities_supported=True) | changes


def audit(atoms, status="supported"):
    check = dict(supported=True, reason="Supported or not applicable.")
    return _VerificationAudit(assertions=atoms, scope=check, inference=check, quantities=check,
        verification=dict(status=status, reason="Per-atom evidence assessment."))


class VerificationContractTests(OfflineTest):
    def setUp(self):
        super().setUp()
        self.claim = AnswerClaim(claim_id="claim:opaque:original", text="The measured findings hold.", evidence_ids=IDS)
        self.evidence = [passage(key, number=i, text=f"Finding {i} holds in the measured setting.")
                         for i, key in enumerate(IDS, 1)]
        self.papers = [SelectedPaper(citation_label="P1", group_id="g", paper=paper())]
        self.sdk = MagicMock()
        self.sdk.with_options.return_value = self.sdk
        self.client = OpenAIClaimVerifier(client=self.sdk)

    def verify(self, atoms, *, status="supported"):
        self.sdk.responses.parse.return_value = SimpleNamespace(status="completed", output=[],
            output_parsed=audit(atoms, status))
        return self.client.verify(self.claim, self.evidence)

    def record(self, atoms, **kw):
        return VerifiedClaim(claim=self.claim, verification=self.verify(atoms, **kw))

    def render(self, record):
        return render_answer([record], self.evidence, self.papers)

    def test_exact_set_remains_valid(self):
        result = self.verify([atom(ids=("E1", "E2", "E3"))])
        self.assertEqual(result.evidence_ids, IDS)
        self.assertEqual(result.status, "supported")

    def test_supporting_subset_is_valid_without_redundant_union_copy(self):
        record = self.record([atom(ids=("E1", "E3"))])
        self.assertEqual(record.verification.evidence_ids, [IDS[0], IDS[2]])
        self.assertEqual(rendered_evidence_ids(record), [IDS[0], IDS[2]])
        self.assertNotIn("[P1, p. 2]", self.render(record))
        self.assertIn("[P1, p. 1]", self.render(record))
        self.assertIn("[P1, p. 3]", self.render(record))

    def test_unknown_alias_hard_failure(self):
        with self.assertRaisesRegex(ValueError, "Unknown evidence aliases: E4"):
            self.verify([atom(ids=("E1", "E4"))])

    def test_uncited_evidence_cannot_be_used_even_when_available(self):
        self.evidence.append(passage("another-paper-and-run", record=paper("W2")))
        with self.assertRaisesRegex(ValueError, "Unknown evidence aliases"):
            self.verify([atom(ids=("E4",))])

    def test_unknown_alias_in_rejected_atom_also_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown evidence aliases"):
            self.verify([atom(ids=("E99",), status="unsupported")])

    def test_long_id_is_not_accepted_as_output_alias(self):
        with self.assertRaises(ValueError):
            self.verify([atom(ids=(IDS[0],))])

    def test_aliases_are_exact_no_whitespace_case_or_fuzzy_matching(self):
        for key in ("e1", "E01", " E1", "E1 ", "E0", "E1\n"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify([atom(ids=(key,))])

    def test_duplicate_aliases_deduplicated_and_ordered_by_source(self):
        result = self.verify([atom(ids=("E3", "E1", "E3", "E1"))])
        self.assertEqual(result.assertions[0].evidence_ids, [IDS[0], IDS[2]])
        self.assertEqual(result.evidence_ids, [IDS[0], IDS[2]])

    def test_union_derived_from_atoms_in_original_citation_order(self):
        result = self.verify([atom(ids=("E3",)), atom(2, ("E2", "E1"))])
        self.assertEqual(result.evidence_ids, IDS)

    def test_unsupported_atom_evidence_not_in_union_or_answer(self):
        record = self.record([atom(ids=("E1",)), atom(2, ("E2",), "unsupported"), atom(3, ("E3",))])
        self.assertEqual(record.verification.status, "partially_supported")
        self.assertEqual(record.verification.evidence_ids, [IDS[0], IDS[2]])
        self.assertEqual(record.verification.assertions[1].evidence_ids, [IDS[1]])  # Negative audit retained.
        self.assertNotIn("Finding 2", self.render(record))
        self.assertNotIn("[P1, p. 2]", self.render(record))

    def test_partial_atom_is_audited_but_only_supported_subset_rendered(self):
        record = self.record([atom(), atom(2, ("E2",), "partially_supported")])
        self.assertEqual(record.verification.evidence_ids, IDS[:2])
        self.assertEqual(rendered_evidence_ids(record), IDS[:1])
        self.assertNotIn("Finding 2", self.render(record))
        self.assertNotIn(self.claim.text, self.render(record))

    def test_conflict_citations_survive_without_asserting_conflict_as_fact(self):
        record = self.record([atom(), atom(2, ("E2", "E3"), "conflicting")])
        self.assertEqual(record.verification.status, "conflicting")
        self.assertEqual(rendered_evidence_ids(record), IDS)
        findings = rendered_findings(record)
        self.assertEqual(findings[1][1], IDS[1:])
        self.assertIn("Unresolved disagreement", self.render(record))
        self.assertNotIn("Finding 2", self.render(record))
        self.assertEqual(supported_findings(record)[0][1], IDS[:1])

    def test_no_support_produces_empty_verified_and_rendered_evidence(self):
        record = self.record([atom(ids=(), status="unsupported")])
        self.assertEqual(record.verification.evidence_ids, [])
        self.assertEqual(rendered_findings(record), [])

    def test_overall_unsupported_blocks_individually_positive_atoms(self):
        record = self.record([atom()], status="unsupported")
        self.assertEqual(supported_findings(record), [])
        self.assertEqual(rendered_evidence_ids(record), [])

    def test_partial_without_supported_atoms_has_no_factual_citations(self):
        record = self.record([atom(status="partially_supported")])
        self.assertEqual(rendered_evidence_ids(record), [])
        self.assertNotIn("[P1, p. 1]", self.render(record))

    def test_input_and_synthesis_audit_not_mutated_and_ids_stored_separately(self):
        before = deepcopy((self.claim, self.evidence))
        record = self.record([atom(ids=("E3", "E1"))])
        restored = VerifiedClaim.model_validate_json(record.model_dump_json())
        self.assertEqual(restored.claim.evidence_ids, IDS)
        self.assertEqual(restored.verification.evidence_ids, [IDS[0], IDS[2]])
        self.assertEqual(rendered_evidence_ids(restored), [IDS[0], IDS[2]])
        self.render(record)
        self.assertEqual((self.claim, self.evidence), before)

    def test_payload_has_only_request_local_identifiers_and_cited_passages(self):
        self.evidence.append(passage("uncited-secret-id", text="Unrelated secret text"))
        self.verify([atom()])
        call = self.sdk.responses.parse.call_args.kwargs
        payload = json.loads(call["input"])
        for key in IDS + [self.claim.claim_id, paper().paper_id, "uncited-secret-id", "Unrelated secret text"]:
            self.assertNotIn(key, call["input"])
        self.assertEqual(payload["claim"], dict(text=self.claim.text, evidence_ids=["E1", "E2", "E3"]))
        self.assertEqual([p["passage_id"] for p in payload["evidence"]], ["E1", "E2", "E3"])
        self.assertEqual([p["text"] for p in payload["evidence"]], [p.text for p in self.evidence[:3]])
        self.assertEqual([p["paper_id"] for p in payload["evidence"]], ["P1"] * 3)
        self.assertFalse(call["store"])
        self.assertEqual(self.sdk.responses.parse.call_count, 1)

    def test_alias_mapping_is_request_local_and_deterministic(self):
        first, _ = self.client._payload(self.claim, list(reversed(self.evidence)))
        second, aliases = self.client._payload(self.claim, self.evidence)
        self.assertEqual(first, second)
        self.assertEqual(aliases, dict(zip(IDS, ["E1", "E2", "E3"])))
        self.claim.evidence_ids = [IDS[2]]
        record = self.record([atom()])
        self.assertEqual(record.verification.evidence_ids, [IDS[2]])

    def test_output_schema_has_no_redundant_claim_identity_or_evidence_union(self):
        schema = _VerificationAudit.model_json_schema()
        self.assertEqual(set(schema["$defs"]["_ClaimVerdict"]["properties"]), {"status", "reason"})
        self.assertEqual(set(schema["properties"]), {"assertions", "scope", "inference", "quantities", "verification"})

    def test_malformed_or_duplicate_assertion_ids_fail(self):
        for key in ("", "a1", "A0", "A01", "A1 ", "other", "A1\n"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify([atom(assertion_id=key)])
        with self.assertRaisesRegex(ValueError, "assertion IDs must be unique"):
            self.verify([atom(), atom()])

    def test_unknown_domain_id_hard_failure(self):
        result = ClaimVerification(claim_id=self.claim.claim_id, status="supported",
            evidence_ids=[IDS[0], "foreign-run"], reason="Forged reference.")
        with self.assertRaisesRegex(ValueError, "Unknown verification evidence IDs"):
            validate_verification(self.claim, result)

    def test_final_reference_missing_from_verified_set_fails_at_render_boundary(self):
        record = self.record([atom(ids=("E1", "E3"))])
        record.verification.evidence_ids = [IDS[0]]
        with self.assertRaisesRegex(ValueError, "verified assertion evidence IDs"):
            self.render(record)

    def test_unknown_atom_id_fails_even_if_claim_level_is_valid(self):
        record = self.record([atom()])
        record.verification.assertions[0].evidence_ids = ["invented"]
        with self.assertRaisesRegex(ValueError, "Unknown assertion evidence IDs"):
            self.render(record)

    def test_mutated_supported_status_cannot_publish_unsupported_atom(self):
        record = self.record([atom(), atom(2, (), "unsupported")])
        record.verification.status = "supported"
        with self.assertRaisesRegex(ValueError, "all material assertions"):
            self.render(record)

    def test_empty_supported_verification_is_invalid(self):
        result = ClaimVerification(claim_id=self.claim.claim_id, status="supported", evidence_ids=[], reason="Empty.")
        with self.assertRaisesRegex(ValueError, "requires verified evidence"):
            validate_verification(self.claim, result)

    def test_historical_atomless_verification_remains_readable(self):
        result = ClaimVerification(claim_id=self.claim.claim_id, status="supported", evidence_ids=IDS, reason="Legacy.")
        record = VerifiedClaim(claim=self.claim, verification=result)
        self.assertEqual(rendered_evidence_ids(VerifiedClaim.model_validate_json(record.model_dump_json())), IDS)

    def test_historical_full_union_does_not_reintroduce_unused_atomic_citations(self):
        record = self.record([atom()])
        record.verification.evidence_ids = IDS.copy()
        restored = VerifiedClaim.model_validate_json(record.model_dump_json())
        self.assertEqual(rendered_evidence_ids(restored), IDS[:1])
        self.assertEqual(supported_findings(restored)[0][1], IDS[:1])

    def test_unused_synthesis_evidence_cannot_resolve_a_gap(self):
        from researchpilot.research_iteration import EvidenceAssessment, EvidenceGapRecord, GapUpdate, validate_assessment
        record = self.record([atom()])
        gap = EvidenceGapRecord(gap_id="g1", description="Missing finding.", search_focus="finding",
            severity="critical", related_claim_ids=[], first_seen_round=1, last_updated_round=1)
        assessment = EvidenceAssessment(sufficient=False, gaps=[], rationale="Still incomplete.", gap_updates=[
            GapUpdate(gap_id="g1", status="resolved", reason="Cites unused synthesis evidence.",
                      evidence_ids=[IDS[1]], related_claim_ids=[], superseded_by=None)])
        with self.assertRaisesRegex(ValueError, "supported finding"):
            validate_assessment(assessment, [record], self.evidence, [gap])

    def test_bundle_context_uses_aliases_without_exposing_uncited_members(self):
        from researchpilot.evidence_bundle import EvidenceBundle, EvidenceBundleMember, bundle_identity
        original_claim = self.claim.model_dump()
        bundle = EvidenceBundle(bundle_id=bundle_identity(paper().paper_id, IDS[0], IDS),
            paper_id=paper().paper_id, title=paper().title, anchor_evidence_id=IDS[0],
            members=[EvidenceBundleMember(evidence_id=key, passage_id=key, page_number=i, role="condition")
                     for i, key in enumerate(IDS, 1)], purpose="The conditional result.", complete=True,
            missing_components=[])
        claim = self.claim.model_copy(update={"evidence_ids": [IDS[0], IDS[2]]})
        self.sdk.responses.parse.return_value = SimpleNamespace(status="completed", output=[],
            output_parsed=audit([atom(ids=("E1", "E2"))]))
        result = self.client.verify_with_bundles(claim, self.evidence, [bundle])
        self.assertEqual(result.evidence_ids, [IDS[0], IDS[2]])
        payload = self.sdk.responses.parse.call_args.kwargs["input"]
        for key in IDS + [bundle.bundle_id, paper().paper_id, claim.claim_id]:
            self.assertNotIn(key, payload)
        context = json.loads(payload)["evidence_bundles"][0]
        self.assertEqual(context["bundle_id"], "B1")
        self.assertEqual([m["evidence_id"] for m in context["members"]], ["E1", "E2"])
        self.assertEqual([m["passage_id"] for m in context["members"]], ["E1", "E2"])
        self.assertFalse(context["complete"])
        self.assertTrue(context["uncited_members_exist"])
        self.assertEqual(self.claim.model_dump(), original_claim)

    def test_paper_identity_bound_by_code_for_multiple_cited_sources(self):
        self.evidence[1] = passage(IDS[1], record=paper("W2"), number=2)
        self.verify([atom(ids=("E2",))])
        payload = json.loads(self.sdk.responses.parse.call_args.kwargs["input"])
        self.assertEqual([p["paper_id"] for p in payload["evidence"]], ["P1", "P2", "P1"])
        self.assertEqual(payload["evidence"][1]["title"], paper("W2").title)

    def prepare_project_result(self, record):
        store = RunStore(self.temporary_directory() / "runs.sqlite3")
        p = project(store)
        request = ResearchRequest(question=QUESTION, project_id=p.project_id, max_search_rounds=1)
        run = store.create(request)
        store.start(run.run_id)
        result = research_fixture().agent.run(request)  # Fake providers, sockets prohibited.
        result.claims = [record]
        result.evidence = self.evidence
        result.papers = self.papers
        result.answer = self.render(record)
        result.evidence_bundles = []
        result.visual_evidence = []
        return store, p, run, result

    def test_project_ingests_only_surviving_supported_evidence(self):
        record = self.record([atom(), atom(2, ("E2",), "unsupported"), atom(3, ("E3",))])
        store, p, run, result = self.prepare_project_result(record)
        store.complete(run.run_id, result)
        rows = store.projects.list_records(p.project_id, "evidence")
        self.assertEqual({r.passage.passage_id for r in rows}, {IDS[0], IDS[2]})
        self.assertEqual(len(store.projects.list_records(p.project_id, "findings")), 2)
        saved = store.get(run.run_id).result.claims[0]
        self.assertEqual(saved.claim.evidence_ids, IDS)
        self.assertEqual(rendered_evidence_ids(saved), [IDS[0], IDS[2]])

    def test_project_legacy_full_union_does_not_trust_unused_evidence(self):
        record = self.record([atom()])
        record.verification.evidence_ids = IDS.copy()
        store, p, run, result = self.prepare_project_result(record)
        store.complete(run.run_id, result)
        self.assertEqual([r.passage.passage_id for r in store.projects.list_records(p.project_id, "evidence")], IDS[:1])

    def test_project_overall_rejection_does_not_ingest_positive_internal_atom(self):
        store, p, run, result = self.prepare_project_result(self.record([atom()], status="unsupported"))
        store.complete(run.run_id, result)
        self.assertEqual(store.projects.list_records(p.project_id, "evidence"), [])
        self.assertEqual(store.projects.list_records(p.project_id, "findings"), [])

    def test_project_conflict_and_partial_atoms_are_not_positive_attestations(self):
        record = self.record([atom(), atom(2, ("E2",), "partially_supported"), atom(3, ("E3",), "conflicting")])
        store, p, run, result = self.prepare_project_result(record)
        store.complete(run.run_id, result)
        self.assertEqual([r.passage.passage_id for r in store.projects.list_records(p.project_id, "evidence")], IDS[:1])

    def test_invalid_evidence_cannot_reach_project_and_completion_rolls_back(self):
        store, p, run, result = self.prepare_project_result(self.record([atom()]))
        result.claims[0].verification.assertions[0].evidence_ids = ["foreign-run-evidence"]
        with self.assertRaises(ValueError):
            store.complete(run.run_id, result)
        self.assertEqual(store.projects.list_records(p.project_id, "evidence"), [])
        self.assertEqual(store.projects.list_records(p.project_id, "papers"), [])
        self.assertEqual(store.get(run.run_id).status, "running")
