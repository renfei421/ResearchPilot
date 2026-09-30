"""Pure migration and annotation preparation using frozen canonical identities.

No identity reconstruction, relevance inference, retrieval, or metric calculation
belongs here. Only explicit Human labels can enter the resulting union gold.
"""

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from eval.identity_alignment import AlignmentMember


# Accepted, reviewed canonical counts: inherited / new / union.
EXPECTED_COUNTS = {
    "matrix_completion_v1": (22, 18, 40),
    "rag_hallucination_v1": (17, 26, 43),
    "cot_reasoning_faithfulness_v1": (22, 35, 57),
}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _label(value: Any) -> int:
    if type(value) is not int or value not in (0, 1, 2):
        raise ValueError("relevance label must be exactly the integer 0, 1, or 2.")
    return value


class EvidenceRecord(BaseModel):
    """An explicit whitelist of bibliographic fields safe for Human review."""

    model_config = ConfigDict(strict=True, extra="forbid")

    group_id: str
    paper_id: str
    title: str
    year: int | None
    abstract: str | None

    @field_validator("group_id", "paper_id", "title")
    @classmethod
    def require_text(cls, value: str) -> str:
        return _text(value, "bibliographic identifier/title")


class AnnotationTask(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    union_id: str
    title: str
    year: int | None
    abstract: str | None
    identifiers: list[str]
    aliases: list[EvidenceRecord] = Field(min_length=1)
    label: int | None
    note: str

    @field_validator("label", mode="before")
    @classmethod
    def require_label(cls, value: Any) -> int | None:
        return None if value is None else _label(value)


class AnnotationDocument(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    schema_version: int
    benchmark_id: str
    version: str
    research_question: str
    relevance_scale: dict[str, str]
    expected_task_count: int = Field(ge=0)
    tasks: list[AnnotationTask]


def validate_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Check the supplied partition; never discover or merge identities."""
    _text(manifest.get("benchmark_id"), "manifest benchmark_id")
    if (
        manifest.get("version") != "v1"
        or manifest.get("identity_review_complete") is not True
        or manifest.get("counts_status") != "final"
        or manifest.get("summary", {}).get("unresolved_title_pairs") != 0
        or manifest.get("review_candidates") != []
    ):
        raise ValueError("A final union manifest with completed identity reviews is required.")
    works = manifest.get("works")
    if not isinstance(works, list) or not works:
        raise ValueError("manifest works must be a non-empty list.")
    by_id = {}
    member_unions = {}
    for work in works:
        union_id = _text(work.get("union_id"), "union_id")
        if union_id in by_id:
            raise ValueError(f"Duplicate union_id: {union_id}.")
        if not isinstance(work.get("members"), list) or not work["members"]:
            raise ValueError(f"Union work {union_id} must have members.")
        sources = set()
        for raw_member in work["members"]:
            member = AlignmentMember.model_validate(raw_member)
            sources.add(member.source)
            key = (member.source, member.group_id)
            if key in member_unions and member_unions[key] != union_id:
                raise ValueError(f"Ambiguous canonical mapping for {key}.")
            member_unions[key] = union_id
        for source in ("human", "llm"):
            if work.get(f"present_in_{source}") is not (source in sources):
                raise ValueError(f"Incorrect present_in_{source} flag for {union_id}.")
        by_id[union_id] = work

    inherited = sum(work["present_in_human"] for work in works)
    new = sum(not work["present_in_human"] and work["present_in_llm"] for work in works)
    shared = sum(work["present_in_human"] and work["present_in_llm"] for work in works)
    counts = {
        "canonical_human_works": inherited,
        "canonical_llm_works": new + shared,
        "intersection": shared,
        "human_only": inherited - shared,
        "llm_only": new,
    }
    if any(manifest["summary"].get(key) != value for key, value in counts.items()):
        raise ValueError("Manifest summary does not match its canonical works.")
    actual = (inherited, new, len(works))
    if inherited + new != len(works):
        raise ValueError("Inherited and new works must partition the canonical union.")
    expected = EXPECTED_COUNTS.get(manifest["benchmark_id"])
    if expected is not None and actual != expected:
        raise ValueError(f"Canonical inherited/new/union counts {actual} must equal {expected}.")
    return by_id


def _gold_metadata(manifest: dict[str, Any], gold: dict[str, Any]) -> None:
    if gold.get("benchmark_id") != manifest["benchmark_id"]:
        raise ValueError("Human gold benchmark_id does not match the union manifest.")
    if gold.get("schema_version") != 1:
        raise ValueError("Expected Human gold schema_version 1.")
    _text(gold.get("research_question"), "research_question")
    scale = gold.get("relevance_scale")
    if not isinstance(scale, dict) or set(scale) != {"0", "1", "2"}:
        raise ValueError("Human gold must provide its frozen 0/1/2 relevance_scale.")
    for description in scale.values():
        _text(description, "relevance_scale description")


def migrate_human_gold(
    manifest: dict[str, Any], gold: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Map every Human group judgment to its manifest union_id, losslessly."""
    works = validate_manifest(manifest)
    _gold_metadata(manifest, gold)
    human_map = {
        member["group_id"]: union_id
        for union_id, work in works.items()
        for member in work["members"] if member["source"] == "human"
    }
    judgments = gold.get("judgments")
    if not isinstance(judgments, dict):
        raise ValueError("Human gold judgments must be a mapping keyed by group_id.")
    migrated = {}
    for group_id, judgment in judgments.items():
        if group_id not in human_map:
            raise ValueError(f"Unknown Human gold group_id; cannot map {group_id}.")
        if not isinstance(judgment, dict) or judgment.get("group_id") != group_id:
            raise ValueError(f"Human judgment key/group_id mismatch: {group_id}.")
        label = _label(judgment.get("relevance"))
        _text(judgment.get("title"), "Human judgment title")
        _text(judgment.get("reason"), "Human judgment reason")
        paper_ids = judgment.get("known_paper_ids")
        if not isinstance(paper_ids, list):
            raise ValueError("Human known_paper_ids must be a list.")
        for paper_id in paper_ids:
            _text(paper_id, "known_paper_id")
        union_id = human_map[group_id]
        if union_id not in migrated:
            migrated[union_id] = {
                "title": judgment["title"],
                "union_id": union_id,
                "known_paper_ids": [],
                "relevance": label,
                "reason": judgment["reason"],
                "provenance": "inherited_human_gold",
                "inherited_judgments": [],
            }
        record = migrated[union_id]
        if record["relevance"] != label:
            raise ValueError(f"Conflicting inherited Human labels for {union_id}.")
        record["inherited_judgments"].append(deepcopy(judgment))
        record["known_paper_ids"] = sorted(set(record["known_paper_ids"] + paper_ids))
        record["reason"] = "\n".join(dict.fromkeys(
            item["reason"] for item in record["inherited_judgments"]
        ))
    missing = set(human_map) - set(judgments)
    if missing:
        raise ValueError(f"Missing Human gold judgments for groups: {sorted(missing)}.")
    expected_ids = {uid for uid, work in works.items() if work["present_in_human"]}
    if set(migrated) != expected_ids:
        raise ValueError("Every canonical Human work must receive one inherited judgment.")
    return {uid: migrated[uid] for uid in sorted(migrated)}


def _snapshot_evidence(
    manifest: dict[str, Any], gold: dict[str, Any], snapshots: dict[str, dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    records = {}
    for source in ("human", "llm"):
        snapshot = snapshots[source]
        # The original Matrix Human snapshot predates benchmark_id.
        legacy_human = source == "human" and manifest["benchmark_id"] == "matrix_completion_v1" and "benchmark_id" not in snapshot
        if not legacy_human and snapshot.get("benchmark_id") != manifest["benchmark_id"]:
            raise ValueError("Candidate snapshot benchmark_id mismatch.")
        if snapshot.get("question") != gold["research_question"]:
            raise ValueError("Candidate snapshot question does not match Human gold.")
        if not isinstance(snapshot.get("candidates"), list):
            raise ValueError("Snapshot candidates must be a list.")
        for candidate in snapshot["candidates"]:
            # Deliberately copy only bibliographic evidence, not hits or scores.
            record = EvidenceRecord.model_validate({key: candidate.get(key) for key in (
                "group_id", "paper_id", "title", "year", "abstract",
            )}).model_dump()
            if record["abstract"] is not None and not record["abstract"].strip():
                record["abstract"] = None
            key = (source, record["group_id"])
            if key in records:
                raise ValueError(f"Duplicate snapshot group: {key}.")
            records[key] = record
    required = set()
    for work in manifest["works"]:
        for member in work["members"]:
            key = (member["source"], member["group_id"])
            required.add(key)
            if key not in records:
                raise ValueError(f"Missing frozen candidate evidence for {key}.")
            if any(records[key][field] != member[field] for field in (
                "group_id", "paper_id", "title", "year",
            )):
                raise ValueError(f"Frozen candidate evidence disagrees with manifest: {key}.")
    if set(records) != required:
        raise ValueError("Frozen candidate groups must exactly match manifest members.")
    return records


def prepare_annotation_tasks(
    manifest: dict[str, Any], gold: dict[str, Any], snapshots: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Prepare null-label tasks for LLM-only canonical works, in union_id order."""
    migrate_human_gold(manifest, gold)
    records = _snapshot_evidence(manifest, gold, snapshots)
    tasks = []
    for work in sorted(manifest["works"], key=lambda item: item["union_id"]):
        if work["present_in_human"]:
            continue

        def display_key(member: dict[str, Any]) -> tuple:
            record = records[(member["source"], member["group_id"])]
            # Snapshots contain only year and stable IDs as additional metadata.
            richness = sum(record[key] is not None for key in ("year", "paper_id", "group_id"))
            return (
                -bool(record["abstract"]), -richness, member["original_rank"],
                member["group_id"], member["paper_id"],
            )

        display = records[("llm", min(work["members"], key=display_key)["group_id"])]
        # Preserve each bibliographic member and all its available abstract text.
        aliases = [deepcopy(records[key]) for key in sorted({
            (member["source"], member["group_id"]) for member in work["members"]
        })]
        tasks.append({
            "union_id": work["union_id"],
            "title": display["title"], "year": display["year"],
            "abstract": display["abstract"],
            "identifiers": sorted({alias[key] for alias in aliases for key in ("group_id", "paper_id")}),
            "aliases": aliases,
            "label": None,
            "note": "",
        })
    return AnnotationDocument.model_validate({
        "schema_version": gold["schema_version"],
        "benchmark_id": manifest["benchmark_id"],
        "version": "v1",
        "research_question": gold["research_question"],
        "relevance_scale": deepcopy(gold["relevance_scale"]),
        "expected_task_count": len(tasks),
        "tasks": tasks,
    }).model_dump()


def validate_annotations(
    annotations: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    """Only label/note may differ from the frozen-evidence task template."""
    document = AnnotationDocument.model_validate(annotations).model_dump()
    if {k: v for k, v in document.items() if k != "tasks"} != {
        k: v for k, v in expected.items() if k != "tasks"
    }:
        raise ValueError("Annotation metadata must match the frozen task template.")
    tasks = {task["union_id"]: task for task in document["tasks"]}
    if len(tasks) != len(document["tasks"]):
        raise ValueError("Duplicate annotation union_id.")
    if set(tasks) != {task["union_id"] for task in expected["tasks"]}:
        raise ValueError("Annotation union_ids must exactly match required LLM-only works.")
    ordered = []
    for template in expected["tasks"]:
        task = tasks[template["union_id"]]
        evidence = {k: v for k, v in task.items() if k not in ("label", "note")}
        if evidence != {k: v for k, v in template.items() if k not in ("label", "note")}:
            raise ValueError(f"Annotation evidence changed for {task['union_id']}.")
        ordered.append(task)
    document["tasks"] = ordered
    return document


def build_union_gold(
    manifest: dict[str, Any], gold: dict[str, Any],
    snapshots: dict[str, dict[str, Any]], annotations: dict[str, Any],
) -> dict[str, Any]:
    """Combine inherited and completed manual labels; reject incomplete gold."""
    inherited = migrate_human_gold(manifest, gold)
    expected = prepare_annotation_tasks(manifest, gold, snapshots)
    document = validate_annotations(annotations, expected)
    missing = [task["union_id"] for task in document["tasks"] if task["label"] is None]
    if missing:
        raise ValueError(f"Incomplete union gold: {len(missing)} required tasks remain unlabeled.")
    judgments = deepcopy(inherited)
    for task in document["tasks"]:
        judgments[task["union_id"]] = {
            "title": task["title"],
            "union_id": task["union_id"],
            "known_paper_ids": sorted({alias["paper_id"] for alias in task["aliases"]}),
            "relevance": task["label"],
            "reason": task["note"],
            "provenance": "manual_union_expansion",
        }
    if set(judgments) != {work["union_id"] for work in manifest["works"]}:
        raise ValueError("Final judgments must cover each known union_id exactly once.")
    return {
        "schema_version": gold["schema_version"],
        "benchmark_id": manifest["benchmark_id"],
        "research_question": gold["research_question"],
        "relevance_scale": deepcopy(gold["relevance_scale"]),
        "provenance": {
            "evaluation_unit": "canonical_union_work",
            "identity_manifest": f"eval/alignment/{manifest['benchmark_id']}_union_v1.json",
            "human_gold": f"eval/datasets/{manifest['benchmark_id']}.json",
            "annotations": f"eval/annotations/{manifest['benchmark_id']}_new_gold_tasks_v1.json",
            "inherited_human_gold_provenance": deepcopy(gold.get("provenance", {})),
            "inherited_count": len(inherited),
            "manual_count": len(document["tasks"]),
            "total_count": len(judgments),
        },
        "judgments": {uid: judgments[uid] for uid in sorted(judgments)},
    }
