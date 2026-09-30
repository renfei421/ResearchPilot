# Canonical work alignment V1

This evaluation layer reads the two frozen candidate snapshots and an optional
identity-review file. It never reads gold labels, semantic results, benchmark
queries or plans, and never calls retrieval or LLM providers.

## Identity rules

Exact `group_id` or exact representative `paper_id` creates an automatic edge.
Connected components include all transitive edges, within and across sources.
Only an explicit `same_work` review can add a manual edge. Titles never create
merge edges: NFKC, casefold, punctuation/separator-to-space normalization and
whitespace collapse only detect review candidates. Empty normalized titles are
not evidence of identity. No fuzzy matching is performed.

`union_id` is `union:` plus SHA-256 of the sorted unique
`(source, group_id, paper_id)` identities in the final component, serialized as
compact UTF-8 JSON. It does not use input order, ranks or runtime timestamps.
Changing component membership can change this evaluation identity.

Every original candidate remains a member, with its unchanged original rank.
Members are displayed in Human-then-LLM source order and original-rank order.
Each source's canonical ranking contains a union ID once at its lowest original
rank. The original snapshot RRF rankings remain untouched.

## Manual decisions

Create `eval/alignment/<benchmark_id>_identity_reviews_v1.json` when reviewing.
The script only reads this file; it never creates or overwrites it. Fictional
example:

```json
{
  "benchmark_id": "example_v1",
  "version": "v1",
  "decisions": [
    {
      "left_source": "human",
      "left_group_id": "human-group-a",
      "right_source": "llm",
      "right_group_id": "llm-group-b",
      "decision": "same_work",
      "note": "Record the evidence for the identity decision here."
    }
  ]
}
```

Sources are `human` or `llm`; decisions are `same_work` or `different_work`.
`note` is optional. References must identify existing source/group pairs.
Malformed reviews, unknown references, opposite decisions for one pair, and
`different_work` constraints contradicted by any exact/manual transitive path
are rejected before writing outputs. Identical repeated decisions are harmless.

A `different_work` decision applies to the two whole final components. It
suppresses their unresolved review pairs, including equivalent member references.
It cannot split an automatic component. Such a conflict must be resolved in the
review artifact; the script never overrides frozen identity evidence.

## Outputs and review completeness

Run `python -m scripts.build_union_alignment --benchmark <benchmark_id>`.
It creates or updates three generated files under `eval/alignment/`:

- `<benchmark_id>_union_v1.json`: canonical works, original member metadata,
  source presence/best ranks, deduplicated rankings, applied decisions, unresolved
  review candidates and summary counts.
- `<benchmark_id>_review_candidates_v1.json`: unresolved component pairs, each
  with a normalized title and inspectable original records from both sides.
- `<benchmark_id>_alignment_summary_v1.json`: identity/coverage counts only.

There is one unresolved pair per normalized title and distinct final component
pair; redundant member-level pairs are collapsed. Human/Human, LLM/LLM and
Human/LLM collisions all follow the same rules. Review a pair by copying its
source/group references into the manual decision file, then rerun the script.

Any unresolved pair makes `identity_review_complete=false` and
`counts_status="provisional"`. In particular, Human-only and LLM-only counts
must not be treated as final. Completeness means all collisions detected by
these conservative rules are resolved; it is not proof that no other duplicate
works exist. No relevance metrics or relevance judgments are produced.

The legacy Human snapshot without a top-level benchmark ID is accepted via its
conventional filename and a question matching the identified LLM snapshot.
