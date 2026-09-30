"""Bounded structural assembly over immutable parser passages; no network I/O."""

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
import unicodedata

from researchpilot.document_rescue import formula_risk
from researchpilot.evidence import EvidencePassage
from researchpilot.evidence_bundle import (
    ASSEMBLY_VERSION, AssemblyLimits, EvidenceBundle, EvidenceBundleMember, DependencyEdge, bundle_identity,
)
from researchpilot.math_evidence import VisualEvidenceRecord
from researchpilot.passage_retrieval import _tokens

_ID = r"(?:[A-Z](?:\.)?\d+(?:\.\d+)*|\d+(?:\.\d+)*)"
_NAMED = re.compile(r"\b(Assumptions?|Conditions?|Theorems?|Propositions?|Lemmas?|"
                    r"Definitions?|Corollar(?:y|ies)|Equations?|Eq\.)\s*\(?(" + _ID + r")\)?", re.I)
_MORE = re.compile(r"\s*(?:,\s*(?:and\s+)?|and\s+)\(?(" + _ID + r")\)?", re.I)
_KINDS = {"eq.": "equation", "corollaries": "corollary"}
_ROLE = {"lemma": "proposition", "corollary": "consequence", "equation": "formula"}
_FORMAL_QUESTION = re.compile(r"\b(?:theor\w*|spectral|curvature|guarantees?|"
    r"incoheren\w*|singular value|deterministic|proof|mathematical|formula|equation)\b", re.I)
_CONDITIONS = re.compile(r"\b(?:under|assum(?:e|ing|ptions?)|suppose|provided|whenever|if|let)\b", re.I)
_RESULT = re.compile(r"\b(?:curvature|recover\w*|converg\w*|guarantee\w*|bound\w*|satisf\w*|holds?|then)\b|[≤≥=]")


def normalized_structure(kind: str, number: str) -> str:
    kind = kind.casefold()
    kind = _KINDS.get(kind, kind.rstrip("s"))
    return kind + ":" + number.casefold().replace(".", "_")


def references(text: str) -> list[str]:
    """Only identifiers actually written in the source, including plural lists."""
    text = unicodedata.normalize("NFKC", text)
    original = text
    # Identifiers inside bibliography citations belong to the cited work, not
    # automatically to this PDF. Keep the source text untouched; mask only the
    # temporary identifier-scanning view.
    text = re.sub(r"\[[^\]]*\]", lambda m: " " * len(m[0]), text)
    found = []
    for match in _NAMED.finditer(text):
        if re.match(r"\s+(?:in|of|from)\s+\[", original[match.end():], re.I):
            continue
        found.append(normalized_structure(*match.groups()))
        end = match.end()
        while more := _MORE.match(text, end):
            found.append(normalized_structure(match[1], more[1]))
            end = more.end()
    for m in re.finditer(r"\b(?:using|by|from|in|inequality)\s*\((\d+(?:\.\d+)*)\)", text, re.I):
        found.append(normalized_structure("equation", m[1]))
    for m in re.finditer(r"\b(?:under|assumptions?|by|and)\s+([AG])\1{0,2}\s*(\d+)\b", text, re.I):
        found.append(normalized_structure("assumption", m[1]+m[2]))
    for m in re.finditer(r"\b([AG])\1{1,2}\s*(\d+)\b", text):
        found.append(normalized_structure("assumption", m[1]+m[2]))
    return list(dict.fromkeys(found))


def declarations(text: str) -> dict[str, str]:
    """Conservative numbered statement headings, not mentions or bibliography.

    PageChunker flattens whitespace, so sentence boundaries also delimit headings.
    A reference such as 'by Theorem 4' never declares Theorem 4.
    """
    text = unicodedata.normalize("NFKC", text)
    found = {}
    for m in _NAMED.finditer(text):
        before, after = text[:m.start()].rstrip(), text[m.end():].lstrip()
        if before.rfind("[") > before.rfind("]"):
            continue
        boundary = not before or before[-1] in ".:;\n" or before.endswith(")")
        after = re.sub(r"^[.:]\s*", "", after)
        after = re.sub(r"^(?:\[[^\]]*\]\s*)+", "", after)
        # A statement must contain substantive content after its heading.
        body = bool(re.match(r"(?:[.:]\s*)?(?:\([^)]{1,100}\)[.:]?\s*)?[A-Z∀∃]", after))
        if boundary and body and not re.match(r"(?:and|is used|is given|of\s|in\s)", after, re.I):
            key = normalized_structure(*m.groups())
            found[key] = _ROLE.get(key.split(":")[0], key.split(":")[0])
    # Labelled graph/incoherence assumptions often use (A1), (G2), etc.
    for m in re.finditer(r"(?:^|[.;:]\s*)\(?([AG]\d+)\)\s*[:.]?\s*(?=[A-Za-z∀∃])", text):
        found[normalized_structure("assumption", m[1])] = "assumption"
    # Numbered displayed equations: require mathematics immediately before label.
    for m in re.finditer(r"[=≤≥][^.;]{1,160}\((\d+(?:\.\d+)*)\)(?:\s|[.,;]|$)", text):
        found[normalized_structure("equation", m[1])] = "formula"
    return found


def reliable_section(text: str) -> str | None:
    # Never infer a heading from flattened running prose.
    match = re.search(r"(?:^|\n)(\d+(?:\.\d+)*|Appendix [A-Z])\s+([A-Z][A-Za-z -]{3,65})\n", text)
    return match[0].strip() if match else None


def passage_role(p: EvidencePassage) -> str:
    roles = list(declarations(p.text).values())
    for role in ("theorem", "proposition", "consequence", "assumption", "condition", "definition", "formula"):
        if role in roles:
            return role
    if re.search(r"\bProof(?: of|\.)", p.text):
        return "proof"
    if re.search(r"\bRemark\s+\d", p.text):
        return "remark"
    if re.match(r"Scope[.:]\s", p.text):
        return "scope"
    if re.match(r"Definition[.:]\s", p.text):
        return "definition"
    return "supporting_context"


def purpose_for(question: str) -> str:
    if re.search(r"\bcurvature\b", question, re.I):
        return "curvature guarantee with assumptions and connecting result"
    if re.search(r"\brecover\w*\b", question, re.I):
        return "recovery guarantee with assumptions and conclusion"
    if re.search(r"\b(?:compar\w*|differ\w*)\b", question, re.I) and re.search(r"\bformula\b", question, re.I):
        return "formula comparison with definitions and scope"
    return "formal result with conditions and conclusion"


def should_assemble(question: str, anchor: EvidencePassage) -> bool:
    return (anchor.source_type == "pdf" and not anchor.passage_id.startswith("visual:")
            and bool(_FORMAL_QUESTION.search(question))
            and bool(declarations(anchor.text) or (
                references(anchor.text) and _RESULT.search(anchor.text))))


@dataclass
class AssemblyDraft:
    anchor: EvidencePassage
    passages: list[EvidencePassage]
    purpose: str
    unresolved: list[str]
    required_components: dict[str, list[str]] = field(default_factory=dict)
    dependencies: list[DependencyEdge] = field(default_factory=list)
    resolved_references: dict[str, list[str]] = field(default_factory=dict)
    target_claim_ids: list[str] = field(default_factory=list)
    required_text: dict[str, str] = field(default_factory=dict)


def assembly_cache_key(anchor: EvidencePassage, corpus: list[EvidencePassage],
                       question: str, limits: AssemblyLimits) -> str:
    body = [ASSEMBLY_VERSION, anchor.passage_id, question, limits.model_dump(),
            [(p.passage_id, sha256(p.text.encode()).hexdigest()) for p in corpus if p.paper_id == anchor.paper_id]]
    return sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class EvidenceAssembler:
    def assemble(self, question: str, anchor: EvidencePassage, corpus: list[EvidencePassage],
                 limits: AssemblyLimits) -> AssemblyDraft:
        from collections import deque
        from researchpilot.statement_index import build_statement_index, required_components, matches_component
        document = list({p.passage_id: p for p in corpus if p.paper_id == anchor.paper_id
                         and p.source_type == "pdf" and p.title == anchor.title}.values())
        document = document[:limits.max_document_passages]
        by_id = {p.passage_id: p for p in document}
        by_id.setdefault(anchor.passage_id, anchor)
        index = build_statement_index(document)
        entries = [s for values in index.values() for s in values]
        selected = {anchor.passage_id: anchor}
        components = {name: [] for name in required_components(question)}
        edges, unresolved, resolved, texts = [], [], {}, {}
        topic = set(_tokens(question))
        roots = [s for s in entries if s.evidence_id == anchor.passage_id and not s.proof_of]
        roots.sort(key=lambda s: -len(topic & set(_tokens(s.local_text))))
        queue = deque((s, 0) for s in roots[:1])
        visited, looked_up = set(), set()

        def admit(entry, parent, relation):
            p = by_id[entry.evidence_id]
            if p.passage_id not in selected:
                if len(selected) >= limits.max_bundle_members:
                    unresolved.append("statement " + entry.statement_id + " (member budget)")
                    return False
                if p.page_number not in {v.page_number for v in selected.values()} and len(
                        {v.page_number for v in selected.values()}) >= limits.max_dependency_pages:
                    unresolved.append("statement " + entry.statement_id + " (dependency page budget)")
                    return False
                selected[p.passage_id] = p
            if parent != entry.evidence_id:
                edge = DependencyEdge(referenced_by=parent, target_id=entry.evidence_id,
                    statement_id=entry.statement_id, dependency_type=relation)
                if edge not in edges:
                    edges.append(edge)
            return True

        # Claim-specific connecting results may lie beyond the cheap radius.
        # A component search only seeds one matching statement; its explicit
        # dependencies are traversed below before any incidental context.
        for name in components:
            candidates = [s for s in entries if matches_component(name, s)]
            candidates.sort(key=lambda s: (s.evidence_id != anchor.passage_id, s.evidence_id not in selected,
                not bool(set(references(s.local_text)) & {r.statement_id for r in roots}),
                name == "graph_assumption" and s.statement_type not in ("assumption", "condition"),
                -len(topic & set(_tokens(s.local_text))), abs(s.page-anchor.page_number), s.page, s.evidence_id))
            if candidates:
                entry = candidates[0]
                if admit(entry, anchor.passage_id, "component"):
                    queue.append((entry, 0))

        while queue:
            entry, depth = queue.popleft()
            token = (entry.statement_id, entry.evidence_id)
            if token in visited:
                continue
            visited.add(token)
            texts[entry.evidence_id] = texts.get(entry.evidence_id, "") + " " + entry.local_text
            for name in components:
                if matches_component(name, entry) and entry.evidence_id not in components[name]:
                    components[name].append(entry.evidence_id)
            deps = [key for key in references(entry.local_text) if key != entry.statement_id]
            for key in deps:
                if key not in looked_up and len(looked_up) >= limits.max_reference_lookups:
                    unresolved.append("statement " + key + " (lookup budget)")
                    continue
                looked_up.add(key)
                choices = index.get(key, [])
                if not choices:
                    unresolved.append("statement " + key)
                    continue
                choices = sorted(choices, key=lambda s: (abs(s.page-entry.page), s.page, s.evidence_id))
                target = choices[0]
                if len({s.local_text for s in choices if s.page == target.page}) > 1:
                    unresolved.append("ambiguous statement " + key)
                    continue
                if (target.statement_id, target.evidence_id) not in visited and depth >= limits.max_reference_depth:
                    unresolved.append("statement " + key + " (dependency depth budget)")
                    continue
                if admit(target, entry.evidence_id, "reference"):
                    resolved.setdefault(key, [])
                    if target.evidence_id not in resolved[key]:
                        resolved[key].append(target.evidence_id)
                    queue.append((target, depth+1))
            # Proof blocks are followed only for a required connecting result.
            # Stop at the next statement/proof heading; at most four physical
            # pages, still subject to the same member/page/dependency budgets.
            if any(entry.evidence_id in components[n] for n in components if n.endswith("connection")):
                for proof in index.get("proof:"+entry.statement_id, [])[:1]:
                    if depth >= limits.max_reference_depth or not admit(proof, entry.evidence_id, "proof"):
                        continue
                    queue.append((proof, depth+1))
                    following = sorted((p for p in document if proof.page < p.page_number <= proof.page+3),
                                       key=lambda p: (p.page_number, document.index(p)))
                    for p in following:
                        from researchpilot.statement_index import LocalStatement
                        # Truncate only the scanning view at a new local heading.
                        next_heads = [s for s in entries if s.evidence_id == p.passage_id and s.statement_type != "equation"]
                        first = min(next_heads, key=lambda s: s.offset) if next_heads else None
                        local = unicodedata.normalize("NFKC", p.text)[:first.offset] if first else p.text
                        if local.strip():
                            continuation = LocalStatement(proof.statement_id+":continuation:"+p.passage_id,
                                "proof", p.page_number, p.passage_id, local)
                            if admit(continuation, proof.evidence_id, "continuation"):
                                queue.append((continuation, depth+1))
                        if first:
                            break
        return AssemblyDraft(anchor, list(selected.values()), question, list(dict.fromkeys(unresolved)),
            components, edges, resolved, required_text=texts)


def finish_bundle(draft: AssemblyDraft, visuals: dict[str, VisualEvidenceRecord] = None,
                  visual_passages: list[EvidencePassage] = (), max_members: int = 8) -> EvidenceBundle:
    visuals = visuals or {}
    passages = list({p.passage_id: p for p in [*draft.passages, *visual_passages]}.values())[:max_members]
    admitted = {p.passage_id for p in passages}
    components = {key: [v for v in values if v in admitted] for key, values in draft.required_components.items()}
    missing = list(draft.unresolved)
    labels = {"curvature_connection": "explicit connecting curvature theorem/result",
              "recovery_connection": "explicit connecting recovery theorem/result"}
    for key, values in components.items():
        if not values:
            missing.append(labels.get(key, key))
    if not components:
        missing.append("claim-specific components unavailable")
    members = []
    for p in passages:
        visual = visuals.get(p.passage_id)
        if visual:
            role = _ROLE.get(visual.evidence.statement_type, visual.evidence.statement_type)
            if role not in ("theorem", "proposition", "assumption", "condition", "formula"):
                role = "supporting_context"
            if visual.evidence.ambiguity:
                missing.append(f"ambiguous visual formula on page {p.page_number}")
        else:
            role = passage_role(p)
            clear = any(v.evidence.paper_id == p.paper_id and v.evidence.page_number == p.page_number
                        and v.evidence.relevant_to_gap and not v.evidence.ambiguity
                        and v.evidence.evidence_id in {s.passage_id for s in passages} for v in visuals.values())
            if formula_risk(draft.required_text.get(p.passage_id, p.text)).risky and not clear:
                missing.append(f"formula extraction/continuity on page {p.page_number}")
        members.append(EvidenceBundleMember(evidence_id=p.passage_id,
            passage_id=None if visual else p.passage_id, math_evidence_id=p.passage_id if visual else None,
            page_number=p.page_number, role=role))
    missing = list(dict.fromkeys(missing))
    return EvidenceBundle(bundle_id=bundle_identity(draft.anchor.paper_id, draft.anchor.passage_id,
        [m.evidence_id for m in members], draft.purpose), paper_id=draft.anchor.paper_id, title=draft.anchor.title,
        anchor_evidence_id=draft.anchor.passage_id, members=members, purpose=draft.purpose,
        complete=not missing, missing_components=missing,
        resolved_references={key: [v for v in values if v in admitted] for key, values in draft.resolved_references.items()
                             if any(v in admitted for v in values)},
        target_claim_ids=draft.target_claim_ids, required_components=components,
        dependencies=[e for e in draft.dependencies if e.referenced_by in admitted and e.target_id in admitted])
