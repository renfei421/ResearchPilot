"""Pure, relevance-independent work alignment above frozen retrieval pools."""

from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from typing import Any, Literal
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, field_validator


Source = Literal["human", "llm"]


class AlignmentMember(BaseModel):
    """One original snapshot record; its identity and rank are never rewritten."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source: Source
    original_rank: int = Field(ge=1)
    group_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    year: int | None
    version_count: int = Field(ge=1)

    @field_validator("group_id", "paper_id", "title")
    @classmethod
    def require_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Identity fields and title must not be blank.")
        return value


class IdentityDecision(BaseModel):
    """An explicit human identity decision, independent of relevance labels."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    left_source: Source
    left_group_id: str = Field(min_length=1)
    right_source: Source
    right_group_id: str = Field(min_length=1)
    decision: Literal["same_work", "different_work"]
    note: str | None = None


class IdentityReviews(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    benchmark_id: str = Field(min_length=1)
    version: Literal["v1"]
    decisions: list[IdentityDecision]


def normalize_alignment_title(title: str) -> str:
    """Normalize exact title collisions for review, never automatic merging."""
    normalized = unicodedata.normalize("NFKC", title).casefold()
    separated = "".join(
        " " if unicodedata.category(character).startswith(("P", "Z")) else character
        for character in normalized
    )
    return " ".join(separated.split())


def snapshot_members(snapshot: dict[str, Any], source: Source) -> list[AlignmentMember]:
    """Select only identity metadata from a raw snapshot, without changing it."""
    if source not in ("human", "llm"):
        raise ValueError("source must be 'human' or 'llm'.")
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("candidates"), list):
        raise ValueError("Snapshot must contain a candidates list.")
    members = []
    fields = ("group_id", "paper_id", "title", "year", "version_count")
    for record in snapshot["candidates"]:
        if not isinstance(record, dict) or any(field not in record for field in (*fields, "rrf_rank")):
            raise ValueError("Candidate record is missing required identity metadata or rrf_rank.")
        members.append(AlignmentMember(
            source=source, original_rank=record["rrf_rank"],
            **{field: record[field] for field in fields},
        ))
    return members


class _Components:
    """Union-find; root choice has no effect on the exported canonical identity."""

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _member_key(member: AlignmentMember) -> tuple:
    return (member.source != "human", member.original_rank, member.group_id, member.paper_id)


def _union_id(members: list[AlignmentMember]) -> str:
    identities = sorted({(member.source, member.group_id, member.paper_id) for member in members})
    encoded = json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"union:{sha256(encoded).hexdigest()}"


def canonical_source_ranking(works: list[dict[str, Any]], source: Source) -> list[str]:
    """Keep each work's best original source rank, then sort by that rank."""
    if source not in ("human", "llm"):
        raise ValueError("source must be 'human' or 'llm'.")
    best_ranks = {}
    for work in works:
        ranks = [member["original_rank"] for member in work["members"] if member["source"] == source]
        if ranks:
            best = min(ranks)
            union_id = work["union_id"]
            best_ranks[union_id] = min(best_ranks.get(union_id, best), best)
    return sorted(best_ranks, key=lambda union_id: (best_ranks[union_id], union_id))


def align_candidates(
    benchmark_id: str,
    members: list[AlignmentMember],
    reviews: IdentityReviews | None = None,
) -> dict[str, Any]:
    """Align exact identities and explicit reviews, then expose remaining title pairs.

    different_work is a constraint on entire final components. It cannot undo
    automatic evidence or transitive same_work edges; contradictions fail.
    Input objects and their original ranks remain unchanged.
    """
    if reviews is not None and reviews.benchmark_id != benchmark_id:
        raise ValueError("Review benchmark_id does not match the requested benchmark.")
    ordered = sorted(members, key=_member_key)
    ranks = [(member.source, member.original_rank) for member in ordered]
    if len(set(ranks)) != len(ranks):
        raise ValueError("Original ranks must be unique within each source snapshot.")
    components = _Components(len(ordered))
    groups, papers, references = {}, {}, {}
    for index, member in enumerate(ordered):
        for key, seen in ((member.group_id, groups), (member.paper_id, papers)):
            if key in seen:
                components.union(index, seen[key])
            else:
                seen[key] = index
        references[(member.source, member.group_id)] = index

    # Resolve all references before applying reviews. Multiple records with the
    # same source/group_id are already in one exact-match component.
    decisions = reviews.decisions if reviews is not None else []
    resolved = []
    pair_decisions = {}
    for decision in decisions:
        left = (decision.left_source, decision.left_group_id)
        right = (decision.right_source, decision.right_group_id)
        for reference in (left, right):
            if reference not in references:
                raise ValueError(f"Identity review refers to unknown candidate group: {reference!r}.")
        pair = tuple(sorted((left, right)))
        if pair in pair_decisions and pair_decisions[pair] != decision.decision:
            raise ValueError(f"Conflicting manual decisions for candidate pair {pair!r}.")
        pair_decisions[pair] = decision.decision
        resolved.append((references[left], references[right], decision.decision))
    for left, right, decision in resolved:
        if decision == "same_work":
            components.union(left, right)

    different_pairs = set()
    for left, right, decision in resolved:
        if decision == "different_work":
            pair = tuple(sorted((components.find(left), components.find(right))))
            if pair[0] == pair[1]:
                raise ValueError("Conflicting identity decisions: different_work contradicts exact/same_work connections.")
            different_pairs.add(pair)

    by_component = defaultdict(list)
    by_title = defaultdict(dict)
    for index, member in enumerate(ordered):
        root = components.find(index)
        by_component[root].append(member)
        title = normalize_alignment_title(member.title)
        # Empty normalization is not title evidence. Keep one inspectable
        # representative per component and title to avoid redundant pair output.
        if title:
            by_title[title].setdefault(root, member)

    review_candidates = []
    for title, title_components in sorted(by_title.items()):
        for left, right in combinations(sorted(title_components), 2):
            if (left, right) not in different_pairs:
                review_candidates.append({
                    "benchmark_id": benchmark_id,
                    "normalized_title": title,
                    "candidate_a": title_components[left].model_dump(),
                    "candidate_b": title_components[right].model_dump(),
                })

    works = []
    for component in sorted(by_component.values(), key=lambda value: _member_key(value[0])):
        human_rank = min((member.original_rank for member in component if member.source == "human"), default=None)
        llm_rank = min((member.original_rank for member in component if member.source == "llm"), default=None)
        works.append({
            "union_id": _union_id(component),
            "members": [member.model_dump() for member in component],
            "present_in_human": human_rank is not None,
            "present_in_llm": llm_rank is not None,
            "human_best_rank": human_rank,
            "llm_best_rank": llm_rank,
        })
    human_ranking = canonical_source_ranking(works, "human")
    llm_ranking = canonical_source_ranking(works, "llm")
    human_ids, llm_ids = set(human_ranking), set(llm_ranking)
    complete = not review_candidates
    return {
        "benchmark_id": benchmark_id,
        "version": "v1",
        "identity_review_complete": complete,
        "counts_status": "final" if complete else "provisional",
        "works": works,
        "human_union_ranking": human_ranking,
        "llm_union_ranking": llm_ranking,
        "review_candidates": review_candidates,
        "review_decisions": [decision.model_dump() for decision in decisions],
        "summary": {
            "raw_human_groups": sum(member.source == "human" for member in ordered),
            "raw_llm_groups": sum(member.source == "llm" for member in ordered),
            "canonical_human_works": len(human_ids),
            "canonical_llm_works": len(llm_ids),
            "intersection": len(human_ids & llm_ids),
            "human_only": len(human_ids - llm_ids),
            "llm_only": len(llm_ids - human_ids),
            "unresolved_title_pairs": len(review_candidates),
        },
    }
