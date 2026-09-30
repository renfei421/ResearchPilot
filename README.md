# ResearchPilot

**Evidence-Grounded Academic Research Agent · v0.3.1**

ResearchPilot is a local-first academic research agent that searches scholarly
literature, retrieves page-level evidence, verifies claims, identifies evidence
gaps, analyzes prior-work overlap, and maintains persistent research projects.

- **Traceable evidence:** page-aware passages, atomic claim verification, and
  citations rendered from the final support assessment.
- **Evidence recovery:** hybrid BM25 + dense retrieval, bounded local rescue,
  gated PDF-page vision, and cross-page mathematical `EvidenceBundle` assembly.
- **Two workflows:** cited answers with gap-directed follow-up research, and
  Idea Check for conservative prior-work analysis.
- **Persistent workspaces:** reuse papers and evidence across runs, with fresh
  assessment against the current claim.
- **Tested contracts:** **1,243 automated tests passing at the v0.3.1 portfolio
  baseline**, using offline fixtures and injected providers.

[Architecture & diagrams](docs/architecture.md) ·
[Quick Start](#quick-start) · [Tests](#running-tests) ·
[Release readiness](docs/public_release_readiness.md)

## Why ResearchPilot?

An academic answer needs more than plausible prose. Readers need to inspect
which source supports each claim, whether a theorem's assumptions still apply,
and what the search has left unresolved. ResearchPilot keeps retrieval,
verification, and prior-work analysis separate so those decisions remain
inspectable. Model assessments can still be wrong; provenance makes them
reviewable rather than conclusive.

## Product Modes

### Research Mode

Ask a research question, optionally constrain years and search budgets, and
receive a scoped answer with inspectable citations and remaining gaps. The
pipeline plans queries, searches OpenAlex, resolves paper versions, fuses search
ranks, and semantically reranks papers. It acquires accessible PDFs or explicitly
marked abstracts, retrieves passages, synthesizes claims, and verifies their
atomic assertions. Meaningful gaps can trigger local recovery or bounded
follow-up searches.

PDF citations use **one-based physical PDF pages**, not printed journal page
labels. Abstract-only evidence is labeled as such and has no PDF page number.

### Idea Check Mode

**Check My Idea** decomposes a proposed research claim, searches for prior work,
and evaluates claim-to-paper relationships using retrieved evidence. It returns
a claim literature map, closest prior work, missing evidence, and a potential
novelty boundary. Weakly covered core assertions can trigger bounded additional
source recovery without relaxing the evidence requirements.

| Relation | Meaning |
| --- | --- |
| `DIRECT_OVERLAP` | Evidence establishes the same core contribution under compatible scope. |
| `PARTIAL_OVERLAP` | Some claim components overlap, with material differences retained. |
| `SUPPORTING` | Evidence supports a component or premise. |
| `BRIDGING` | Evidence connects useful components without establishing the full claim. |
| `CONTRADICTING` | Evidence conflicts with the claim under comparable conditions. |
| `METHOD_SIMILAR` | A related method addresses a different problem or setting. |
| `ADJACENT` | Topically related, without a substantive established relationship. |

> **Failure to find a close match is not proof that no prior work exists.**
> Idea Check does not certify novelty or prove the proposed claim.

### Persistent Projects

```text
Project
├── Questions
├── Claims
├── Papers
├── Verified Evidence
├── Findings
├── Open Gaps
├── Notes
└── Run History (Research and Idea Check)
```

Create a project, add questions or notes, and use **Continue Research** to work
from an existing question, claim, or gap. `ProjectContextBuilder` selects bounded
relevant context; papers and evidence can be reused without inheriting their
earlier support status. Research evidence is re-verified against new claims,
and Idea Check evidence is reassessed for the current relationship. Projects
and completed runs persist in local SQLite and export as JSON or Markdown.

## How It Works

**Research:** question → query plan → OpenAlex → canonical candidates and version
groups → RRF → semantic paper ranking → PDF/abstract acquisition → hybrid passage
retrieval → claim synthesis → atomic verification → evidence-gap assessment.
Optional recovery and follow-up branches feed new evidence back into synthesis
and verification before a final cited answer is rendered.

**Idea Check:** claim → decomposition → claim-directed search → evidence retrieval
and relation assessment → coverage review → optional weak-core source recovery
→ prior-work map and potential novelty boundary → validated Project ingestion.

### Architecture

The [architecture guide](docs/architecture.md) contains separate **Research** and
**Idea Check + Project reuse** Mermaid diagrams, module responsibilities, and
evidence trust boundaries. Assembly, rescue, and vision are conditional branches.

| Layer | Implementation |
| --- | --- |
| Local product | FastAPI, Uvicorn, plain HTML/CSS/JavaScript, bounded background worker |
| Research orchestration | LangGraph over injectable Python services |
| Idea Check | Separate bounded `ClaimNoveltyAgent` service |
| Scholarly retrieval | OpenAlex, canonical paper IDs, conservative version grouping, RRF, semantic reranking |
| Evidence | Page-aware PDF extraction, BM25 + local dense similarity, gated vision, cross-page bundles |
| Models | Official OpenAI SDK: Responses API with structured outputs; embeddings API |
| Persistence | SQLite run/project records; local document, embedding, and rendered-page caches |

## Evidence Reliability

**Retrieved evidence is not a verified claim.**

- Evidence IDs are exact. Unknown references are rejected; the verifier can
  select a subset of the supplied claim evidence, but cannot invent sources.
- Atomic verification checks support, scope, inference, and quantities.
  Factual output and its citations derive from surviving supported assertions.
  Unsupported wording is omitted; partial support and conflicts are qualified.
- The original synthesized claim and its cited evidence remain in the audit
  record alongside the final verification.
- Project ingestion uses final verified Research findings. Idea Check relations
  retain a separate `relation_assessed` status; they do not turn a proposed claim
  into a verified fact.

## Case Studies

Three concise accounts of existing completed runs show how evidence and scope
shape the output. No new research calls were made for these case studies.

| Case | What it demonstrates |
| --- | --- |
| [RAG research](docs/case-studies/rag-research.md) | A cited, explicitly partial answer that leaves unsupported comparisons open. |
| [RAG Idea Check](docs/case-studies/rag-idea-check.md) | A medical vision-language result remains partial overlap, with its domain and comparator preserved. |
| [Matrix completion Idea Check](docs/case-studies/matrix-completion-idea-check.md) | Fixed support, weight design, spectral quantities, curvature, and recovery guarantees are compared separately. |

See the [case-study index](docs/case-studies/README.md) for provenance, reading
notes, and two reviewed product screenshots. These historical examples are not an exhaustive benchmark or novelty certification.

## Quick Start

Prerequisites: **Python 3.11+**, **uv**, and an `OPENAI_API_KEY` with access to the
configured models for live work. The repository pins Python 3.11 in
`.python-version` and locks dependencies in `uv.lock`. Live work also needs
network access to OpenAlex and document hosts. Tests do not need provider keys.

Replace `<repository-url>` with the URL of the public repository when it is created.

```powershell
git clone <repository-url> researchpilot
cd researchpilot
uv sync --locked
Copy-Item .env.example .env
```

Edit `.env` locally and replace `your_openai_api_key` with your key. Leave
`OPENALEX_API_KEY` blank or supply your own. Never commit the populated file.
On macOS/Linux, use `cp .env.example .env` for the copy step.

```powershell
uv run --env-file .env uvicorn researchpilot.api:app --host 127.0.0.1 --port 8000
```

Open [the local app](http://127.0.0.1:8000/). Select Research or Check My Idea,
optionally choose a Project, and submit the question or claim. Progress appears
while the job runs; completed results can be reopened and exported.

**The application itself does not auto-load `.env`.** The command above asks uv
to load it into the server environment. If your shell already supplies the keys,
omit `--env-file .env`. Restart the server after changing configuration.

## Configuration

| Variable | Use / default |
| --- | --- |
| `OPENAI_API_KEY` | Required for live Research and Idea Check; not needed to open the app or run offline tests. |
| `OPENALEX_API_KEY` | Optional in the current client; omitted from requests when unset. Provider access limits still apply. |
| `RESEARCHPILOT_CACHE_DIR` | Local cache directory; default `.researchpilot_cache`. |
| `RESEARCHPILOT_RUN_DB` | SQLite file; default `<cache directory>/runs.sqlite3`. |

The current default Responses model is `gpt-5.6-terra`; embeddings use
`text-embedding-3-small`. There is no model environment setting in `Settings`.
Python constructors and the Research CLI's `--model` option configure the
Responses model. Keep custom cache/database paths outside version control.

## Running the App

Use **one server process on loopback**. The local UI has no login or multi-tenant
isolation. Do not expose it publicly or start multiple workers against the same
run database. Avoid `--reload` during research: restarts can interrupt work.
Completed records survive restarts; in-flight jobs do not resume.

- [Health](http://127.0.0.1:8000/health): local status and key-presence flag;
  this does not validate credentials or contact a provider.
- [API docs](http://127.0.0.1:8000/docs): `POST /research`, `POST /claim-check`,
  run polling/export, and `/projects` resources.
- Research CLI: `uv run python -m scripts.run_research_agent --help`.
  Add `--env-file .env` immediately after `uv run` when loading keys for live CLI work.

## Running Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Platform-neutral equivalent:

```sh
uv run --locked python -m unittest discover -s tests -q
```

**1,243 tests pass** at this portfolio baseline. They use handcrafted evidence,
fake clients, HTTP mock transports, temporary databases, and frozen evaluation
fixtures. No live provider calls are required. The count measures automated
coverage, not scientific correctness or complete literature recall.

## Project Structure

```text
src/researchpilot/   Domain models, retrieval, evidence, agents, API, persistence
  web/              Local HTML/CSS/JavaScript interface
tests/              Offline unit, integration, and product-contract tests
scripts/            Research CLI and explicit evaluation/data-building tools
eval/               Frozen benchmarks, judgments, runs, and evaluation utilities
docs/               Architecture and implementation documentation
pyproject.toml      Python package and dependencies
uv.lock             Locked dependency resolution
.env.example        Public configuration template
```

The original [project specification](PROJECT_SPEC.md) records the initial design;
the [architecture guide](docs/architecture.md) describes the current implementation.
Evaluation scripts are separate tools: some intentionally make paid model or
scholarly API calls. They are not part of the offline test command.

## Current Limitations

- Literature recall is bounded by OpenAlex, query budgets, and source access.
  Results are not an exhaustive survey.
- Full-text availability varies. Sources may yield only abstracts or cannot be
  acquired; source metadata and PDF text extraction can also be imperfect.
- Difficult mathematical PDFs may need gated page-level vision. An assembled
  theorem/assumption chain can remain incomplete or be misinterpreted.
- Complex runs can take several minutes and require many paid model calls.
  Provider failures may reduce coverage or stop a run.
- Model-based support and overlap assessments need human review. Idea Check
  cannot certify novelty, and a “supported” label is not a mathematical proof.
- The product targets local single-user research, without public-service
  authentication, distributed workers, or durable job resumption.

## Security and Privacy

**Local-first does not mean fully offline.** The app, SQLite state, and caches
run locally. Selected research text and bounded Project context go to OpenAI for
model assessment; passage/query text goes to its embeddings API; selected PDF
page images can go to its vision-capable model. Search queries go to OpenAlex,
and full-text acquisition contacts remote document hosts.

Treat databases, cached documents, logs, and exports as private research material.
Logs can include query text and run identifiers even with credential redaction.
Review exports before sharing. Do not commit keys, `.env`, caches, or unpublished
manuscripts. The [release readiness notes](docs/public_release_readiness.md) describe publication checks and remaining owner decisions.

## Roadmap

The portfolio baseline includes both modes, persistent Projects, hybrid evidence
retrieval, atomic verification, and bounded core-claim source recovery. Three
case studies and two reviewed product screenshots document existing results.
Further evaluation remains future work; GitHub publication is a separate owner
action.

## License

A license has not yet been selected. This repository currently has no `LICENSE`
file; public-release licensing remains an owner decision.
