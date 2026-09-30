# Hybrid passage retrieval

BM25 and dense retrieval search the existing page-aware `EvidencePassage`
corpus. Paper search, paper RRF, semantic paper ranking, evidence selection,
verification, citations and the target-page vision policy are unchanged.

## Embeddings and local storage

Production uses **OpenAI `text-embedding-3-small`, 1536 dimensions**, through
the already installed official SDK. No additional dependency, vector database,
server, tokenizer, cross-encoder or second embedding backend is needed.
The adapter is synchronous, lazy, has a finite 60-second I/O timeout and no
retries. `EmbeddingClient` is injectable; automated tests use synthetic vectors.

Only the necessary passage text and bounded retrieval-view text are embedded.
Titles, identifiers, PDFs, credentials, run histories and evaluation artifacts
are not included in embedding inputs. Search, cosine calculations, fusion and
citation mapping happen locally.

`<cache_dir>/embeddings/` is under the existing Git-ignored
`.researchpilot_cache/` by default. SHA-256 keys bind the cache format,
backend/model/adapter version/dimensions, and NFC + whitespace-normalized text.
Normalization preserves case and mathematical symbols. Cache files contain only
identity metadata, the key, vector and checksum. Reads validate identity,
checksum, dimension, finite values and a nonzero norm. Corrupt entries are
rebuilt; writes use temporary files and atomic replacement on the same volume.
Changing identity creates different keys. The backend exposes a model alias;
an intentional upstream model refresh requires an adapter-version bump.

Unchanged text reuses vectors across documents and runs. Within one lookup,
duplicate normalized texts are embedded once and mapped back to every original
passage. Requests have at most 64 texts and 131072 UTF-8 bytes total. Each text
has at most 8191 UTF-8 bytes (a conservative byte-token upper bound); oversized
inputs fail explicitly instead of truncating evidence. Passage embeddings are
batched once per corpus lookup; all view embeddings are batched together.

## Bounded rank fusion

For each existing retrieval view:

1. BM25 returns up to `per_view_top_k` passages (default 8, allowed 1–20).
2. Cosine similarity returns up to the same number, stably in corpus order on
   ties. The full concept `origin_text`, when present, is used for its embedding;
   BM25 retains the original rare-token view.
3. Form the union by `passage_id` and calculate
   `view_score(p) = sum(1 / (60 + channel_rank(p)))` over channels containing `p`.
4. Retain the entire bounded union (at most twice the per-view cap), sorted by
   this score. Missing channels contribute zero, not an invented rank.
5. Across these view rankings calculate
   `final_score(p) = sum(1 / (60 + fused_view_rank(p)))`.
6. Return the first `top_passages` (default 20, existing request limit 1–50).

Each passage gets at most one vote per channel/view, then one vote per view in
the second fusion. Ties preserve first encounter: view order, lexical channel
before dense channel, and each channel's stable order. Raw BM25 and cosine
scores are never averaged. Single-channel candidates remain eligible; the
fixed final cap means eligibility does not guarantee inclusion.

No new views are generated. Normal retrieval uses the existing at-most-six
views; the retrieval API enforces at most eight unique views. Rescue keeps its
existing four views per gap by default, 12 candidates per channel/view,
12 offered passages per gap, 24 per run, three scoped papers, six structural
priorities and two rescue passes. Existing stricter request overrides apply.

## Integration and failure

Normal graph retrieval passes only the fused original passages to the selector.
Document-rescue **planning stays local and free of embedding calls**, even when
the graph checks routing more than once. Scoped documents with unseen passages
can schedule a rescue despite a lexical miss. Actual rescue uses hybrid search,
then retains the existing structural priority, adjacent-page context and offer
limits. The formula/layout vision predicate and page budgets remain unchanged.

Any dense/cache/embedding failure records a safe exception-type warning,
`dense_fallback_used=true`, and returns the original multi-view lexical result.
No zero-vector fallback, synthetic page or silently successful hybrid result is
created. A deliberate `HybridPassageRetriever()` injection without a dense
dependency provides explicit lexical mode for offline legacy fixtures.

## Diagnostics

Normal round traces and rescue traces store per-view lexical/dense/fused ranks,
diagnostic raw scores, final passage IDs and mode. Unique candidate counts and
lexical-only/dense-only/overlap counts describe the pre-final-cap channel union;
`fused_candidates` describes the final capped result. These ranks and scores
are not source evidence or confidence and are not passed to the selector or
verifier.

Trace and run statistics record cache hits/misses, corruption, generated
passage/query vectors, attempted batches, input tokens, embedding/cache time,
hybrid elapsed time and fallback. Hits/misses count unique normalized texts per
lookup; ratio is `hits / (hits + misses)` (zero lookups: not applicable).
Embedding time includes disk cache access; hybrid time includes embedding time.
Input tokens reflect returned usage, so an unsuccessful request without usage
may still have provider-side cost. Estimate embedding cost from recorded token
usage and the provider's applicable pricing; this excludes all other API usage.

References: [embedding guide](https://developers.openai.com/api/docs/guides/embeddings),
[model and pricing](https://developers.openai.com/api/docs/models/text-embedding-3-small).
