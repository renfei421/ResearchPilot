# Idea Check core-claim source recovery

This document describes the bounded source-recovery mechanism in v0.3.1.

## Trigger and ordering

The normal Idea Check search, evidence acquisition and relation analysis run first.
Immediately before finalization, `ClaimNoveltyAgent` uses the existing `build_maps()`
coverage calculation. Only assertions with `importance == "core"` and
`coverage_status == "weak"` enter supplementary source recovery.

There is no second coverage metric, changed decomposition, or new relation taxonomy.
Moderate/strong core assertions and supporting assertions do not initiate recovery.
The existing Novelty Boundary is built after recovery, with the unchanged caveat.

## Bounds

| Limit | Value |
| --- | ---: |
| Recovery rounds per weak core assertion | 2 |
| New queries per round | 3 |
| New queries per assertion | 6 |
| New selected papers per assertion | At most 8; honors a smaller request limit |
| Additional acquired documents across all recovery | 8 |
| Additional canonical candidate records across all recovery | 100 |
| Semantic assessments in one recovery round | 24 |

Remaining document/candidate budgets are shared among remaining weak assertions,
so the first target cannot spend all supplementary slots. The existing initial
search limits remain unchanged. The maximum combined run has 24 selected documents
(16 initial plus 8 supplementary) and 200 candidate records. No budget is recursive.
`run_stats.search_rounds` describes the initial search; `source_recovery[*].rounds_used`
records additional rounds. Recovery query traces have `recovery_` role prefixes.

Stop at moderate/strong coverage, budget exhaustion, no new canonical/version group,
no new substantive cited evidence, or all generated queries being duplicates.
Unavailable planning/all failed queries also terminate recovery with an explicit
incomplete diagnostic. Weak coverage may legitimately remain weak.

## Query planning and deduplication

`OpenAIClaimRecoveryPlanner` uses the existing configured model and structured
Responses adapter (`store=False`, finite timeout, no added retries). It requests
only claim-directed direct, terminology, bridge, or useful method-neighbor queries.
It does not receive or output novelty judgments. The existing decomposition and
relation analyzer prompts are untouched.

The adapter follows the existing Pydantic-based
[Structured Outputs interface](https://developers.openai.com/api/docs/guides/structured-outputs).

Recovery query comparison normalizes Unicode, case, punctuation and whitespace;
it preserves word order, Boolean operator words, and mathematical symbols.
Different technical queries sharing keywords remain distinct. Deduplication checks
initial queries, all earlier recovery attempts (including failed attempts), and
Project query history. Full local Project query history is checked locally without
adding it to model context. The planner receives the existing bounded Project
context and the current run's query history.

Queries use the same paper-search dependency and `ProjectSearchSession`, so existing
query audit, exact cache, filters, page size and provenance mechanisms apply.

## Evidence, relationships and persistence

Recovered records pass the existing canonical candidate merge and version resolver,
including already selected Project sources. Existing selected representatives remain;
new versions of an already selected group do not create another selected source.
Initial literature is retained when recovery produces additional records.

Initial and supplementary paths share acquisition, PDF/abstract handling, hybrid
passage retrieval, evidence assembly, gated local/visual rescue and relation validation.
Recovery does not bypass full evidence collection. Existing cache layers and global
assembly/vision budgets remain shared; no additional cache or vision policy exists.

Only the target core assertion is reassessed during recovery. A previous claim-paper
assessment with identical offered passages/bundles is reused; additional evidence
triggers a new assessment of that assertion. Unrelated assertions are not reassessed.
Invalid supplementary relation output is discarded before updating stored relations.
Earlier valid results survive supplementary provider/relationship failures.

Project evidence is offered as evidence for the current assertion and rechecked;
historical support does not become current support. The existing Project ingestion
boundary persists canonical selected papers, validated non-ADJACENT relation evidence,
scoped findings, unresolved gaps and query records. The Project schema is unchanged.

## Diagnostics and display

`ClaimCheckResult.source_recovery` defaults to an empty list for older runs. Each
record contains claim ID, trigger, before/after coverage, generated/executed/successful
queries, skipped duplicates, rounds, candidate/selected IDs, new substantive evidence
and relation counts, incomplete flag and stop reason. New relation counts include
substantive updates to existing claim-paper pairs, not only new papers.

Evidence counts mean newly cited substantive passage IDs, rather than all retrieved
chunks. Diagnostics contain no hidden instructions or reasoning traces. They persist
inside the existing result JSON without a database migration. A concise summary in
the existing warnings/limitations display states whether coverage remains weak or
recovery was incomplete; the UI layout and novelty wording remain unchanged.

## Offline validation

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -q
```

After the final code change: **1243 tests passed in 41.853 seconds** (1210 existing
tests plus 33 focused recovery tests). Tests cover eligibility, budgets, duplicate
queries/papers/versions, normal evidence processing, scope-limited reassessment,
all stop conditions, supplementary failures, Project reuse/ingestion, persisted
diagnostics, unchanged Research Mode and the mandatory caveat.

The [matrix-completion case study](case-studies/matrix-completion-idea-check.md)
records a completed run in which no core assertion met the recovery trigger.
It demonstrates the no-op boundary; the triggered branch is covered by offline tests.
