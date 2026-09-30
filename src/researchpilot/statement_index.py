"""Lightweight local statement spans; source passages remain untouched."""

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class LocalStatement:
    statement_id: str
    statement_type: str
    page: int
    evidence_id: str
    local_text: str
    proof_of: str | None = None
    offset: int = 0


def build_statement_index(passages):
    from researchpilot.evidence_assembly import _NAMED, declarations, normalized_structure
    result = {}
    for p in passages:
        text = unicodedata.normalize("NFKC", p.text)
        declared = declarations(text)
        heads = [(m.start(), normalized_structure(*m.groups())) for m in _NAMED.finditer(text)
                 if normalized_structure(*m.groups()) in declared
                 and (not text[:m.start()].rstrip() or text[:m.start()].rstrip()[-1] in ".:;\n)")
                 and not re.search(r"\b(?:by|under|using|from|of)\s*$", text[:m.start()], re.I)
                 and text[:m.start()].rfind("[") <= text[:m.start()].rfind("]")]
        # PDF bold labels can repeat glyphs (AAA 1, GGG 2). Canonicalize only
        # identifiers, never mathematical content or the original passage text.
        for m in re.finditer(r"(?:^|[.;:•]\s*|\n)\(?([AG])\1{0,2}\s*(\d+)\)?\s*[.:]\s*", text):
            heads.append((m.start(), normalized_structure("assumption", m[1]+m[2])))
        for m in re.finditer(r"\bProof of (Theorem|Proposition|Lemma|Corollary)\s*(\d+(?:\.\d+)*)", text, re.I):
            heads.append((m.start(), "proof:"+normalized_structure(*m.groups())))
        # Equations with reliable display labels have no heading token.
        for key in declared:
            if key.startswith("equation:") and not any(k == key for _, k in heads):
                heads.append((0, key))
        heads.sort()
        for i, (start, key) in enumerate(heads):
            end = heads[i+1][0] if i+1 < len(heads) else len(text)
            if end <= start:
                continue
            entry = LocalStatement(key, key.split(":")[0], p.page_number, p.passage_id,
                text[start:end], key[6:] if key.startswith("proof:") else None, start)
            result.setdefault(key, []).append(entry)
    return result


def required_components(target):
    target = target.casefold()
    names = []
    if re.search(r"graph|biregular|observation pattern", target):
        names.append("graph_assumption")
    if re.search(r"spectral|singular value", target):
        names.append("spectral_bound")
    if "curvature" in target:
        names.append("curvature_connection")
    elif re.search(r"recover\w*", target):
        names.append("recovery_connection")
    if not names:
        names.append("result_conclusion")
    return names


def matches_component(name, statement):
    text = statement.local_text.casefold()
    kind = statement.statement_type
    result = kind in ("theorem", "proposition", "lemma", "corollary")
    if name == "graph_assumption":
        return bool(re.search(r"graph|biregular|regular|sampling", text)) and (
            kind in ("assumption", "condition") or bool(re.search(r"\bunder|\bassum|\bsuppos", text)))
    if name == "spectral_bound":
        return (result or kind == "assumption") and bool(re.search(r"spectral|singular value|operator norm|σ2", text))
    if name == "curvature_connection":
        return result and "curvature" in text
    if name == "recovery_connection":
        return result and bool(re.search(r"recover\w*", text))
    return result
