# Public release readiness — v0.3.1

This is a local portfolio snapshot of ResearchPilot. Publication remains an
owner action. No GitHub repository, remote, push, or hosted release is created
by this preparation.

## Verification

| Check | Result |
| --- | --- |
| Version | Package and root lock entry: **0.3.1**; dependency versions unchanged. |
| Clean installation | Independent environment installed with `uv sync --locked --offline`, using available local dependency caches. |
| Automated tests | **1,243 passed**, 38.478 seconds; all offline. |
| Build | Source distribution and wheel built successfully with `uv build --offline`. Both contain the four required web assets. |
| Package / CLI | Version and public-source import verified; Research CLI `--help` succeeds. |
| Local application | Isolated loopback service on port 8011: home, health, API docs, OpenAPI schema, static assets, and both saved-result views returned HTTP 200. Temporary service stopped after verification. |
| Product behavior | Source, tests, scripts, and frozen evaluation files retained byte-for-byte. No prompts, algorithms, or product workflows changed. |
| History | Fresh `main` repository with one initial public commit; no inherited development commits or remote. |
| Privacy | Reviewed public whitelist, credential/private-path scan, and staged-file review; no actual credentials or private runtime artifacts included. |
| Documentation | Relative Markdown links and local anchors checked; no private-machine result links. |
| Screenshots | Two real, reviewed UI excerpts of existing completed RAG results; see the [case-study index](case-studies/README.md#product-screenshots). No new research calls. |
| License | **LICENSE NOT YET SELECTED**. No license file has been added. |
| Tag | No local release tag created while the licensing decision is pending. |

Run the same offline test suite after installation:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -q
```

The measured duration is from one local run. Tests use injected providers,
handcrafted inputs, temporary state, and frozen fixtures; they do not establish
scientific correctness or complete literature recall. Historical case studies
are evidence-backed examples rather than an all-features benchmark.

## Public contents and exclusions

The [file manifest](public_file_manifest.json) is the explicit release file list.
It includes product source, tests, CLI/evaluation utilities, selected frozen
evaluation data needed for reproduction, package configuration, architecture
documentation, and three curated case studies.

Excluded: prior Git history, populated environment files, virtual environments,
runtime databases and journals, caches, downloaded PDFs, rendered PDF pages,
embedding/API caches, exports, logs, temporary files, manuscripts, personal
research, internal acceptance reports, and validation diffs. UI screenshots
are selected presentation assets, not PDF page images or raw run exports.
The ignore rules protect common generated state while leaving test fixtures
and reproducible evaluation artifacts visible.

## Owner steps before publication

1. Select an appropriate license, add its file, and review redistribution terms
   for any included third-party bibliographic/abstract material.
2. Review the public files, screenshots, package author attribution, and initial
   Git commit identity. No personal author email is stored in package metadata;
   Git uses the owner's existing configured identity.
3. Create the GitHub repository manually and replace the README clone-URL
   placeholder. Add a remote and push only when ready.
4. After the remaining decisions and documentation edits, create an annotated
   `v0.3.1` tag and GitHub release if desired.

Provider credentials are required only for live work. No live OpenAI/OpenAlex
research, new Agent features, or new development phase was part of this release
preparation.
