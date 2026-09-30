# Offline Human union-gold expansion

Step 4.8A reuses the frozen Human judgments and prepares **79 unlabeled**
canonical works for manual review. It does not calculate evaluation metrics.

| Benchmark | Inherited Human | New manual tasks | Final union |
| --- | ---: | ---: | ---: |
| matrix_completion_v1 | 22 | 18 | 40 |
| rag_hallucination_v1 | 17 | 26 | 43 |
| cot_reasoning_faithfulness_v1 | 22 | 35 | 57 |
| Total | 61 | 79 | 140 |

## Artifacts and evidence

`<benchmark>_new_gold_tasks_v1.json` contains `schema_version`, `benchmark_id`,
`version`, `research_question`, the original `relevance_scale`,
`expected_task_count`, and `tasks`. Each task has `union_id`, `title`, `year`,
`abstract`, `identifiers`, `aliases`, `label`, and `note`.

All initial labels are `null`; notes are empty. Tasks are ordered by `union_id`,
independently of retrieval order. Aliases retain each frozen member's `group_id`,
`paper_id`, title, year, and abstract. The display record prefers a nonempty
abstract, then more populated bibliographic fields (year and stable IDs), then
the lowest original rank as a final tie breaker. All distinct available
abstracts remain in the aliases and can be read in the CLI.

Only bibliographic evidence is copied from candidate snapshots. Rank, score,
query, planner, and semantic assessment fields are excluded. Identity comes
exclusively from the reviewed union manifest; no matching is rerun.

## Manual labeling

Run from the repository root using the existing virtual environment:

```powershell
& '.\.venv\Scripts\python.exe' -B -m scripts.label_union_gold --benchmark matrix_completion_v1
```

Use `0`, `1`, or `2` according to the displayed frozen relevance scale, `s` to
skip, and `q` to quit. A note is optional. Every judgment is saved atomically
before the optional note prompt; a submitted note is then saved as well.
EOF or Ctrl+C preserves completed judgments. The same command resumes only
unlabeled tasks; there is no automatic relabeling or edit mode. Run one labeling
session per benchmark; a stale session refuses to overwrite a changed file.

Read the outstanding evidence without editing:

```powershell
& '.\.venv\Scripts\python.exe' -B -m scripts.label_union_gold --benchmark matrix_completion_v1 --list-unlabeled
```

The rubric terms are copied verbatim: `2 = Direct/Core`,
`1 = Bridge/Supporting`, `0 = Off-target for this benchmark`. Core literature
can address an essential component of a composite question. Supporting work
provides directly useful theory, methods, evidence, or analysis. Use only the
frozen title and abstract; missing evidence does not justify inferring details.

## Preparation and finalization

The three task files are already prepared. Preparation is offline and refuses
to overwrite an existing file, including one with partial labeling progress:

```powershell
& '.\.venv\Scripts\python.exe' -B -m scripts.prepare_union_gold_tasks --benchmark matrix_completion_v1
```

After **every** new task for a benchmark has a manual label:

```powershell
& '.\.venv\Scripts\python.exe' -B -m scripts.build_union_gold --benchmark matrix_completion_v1
```

This writes `eval/datasets/<benchmark>_union_gold_v1.json`. Until then it raises
an error and writes no final gold. It also refuses to overwrite existing final
gold. Substitute either of the other benchmark IDs in these commands.

Migration maps each existing Human `group_id` through the manifest to a
`union_id`, preserving its `relevance`, `reason`, `known_paper_ids`, and original
judgment record. Same-label records in one union retain all their provenance;
conflicting labels or missing/unknown Human mappings fail explicitly. Shared
and Human-only works inherit their labels and never become new annotation tasks.

Final gold retains `schema_version`, `benchmark_id`, `research_question`,
`relevance_scale`, `provenance`, and the `judgments` mapping. Judgment keys and
the record's `union_id` identify canonical works. `relevance` and `reason` retain
their existing meaning; each record has provenance `inherited_human_gold` or
`manual_union_expansion`. An optional empty manual note remains an empty reason.
No baseline, metrics, or new rankings are computed. Original gold, snapshots,
manifests, and all other frozen experiment artifacts remain unchanged.
