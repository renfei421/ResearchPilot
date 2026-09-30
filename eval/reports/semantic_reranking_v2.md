# Semantic Reranking Evaluation Report

Report: `semantic_reranking_v2` · Generated (UTC): 2026-09-17T04:26:18.155738+00:00

## Experimental Setup

This offline report covers 3 frozen benchmarks. Handcrafted multi-query retrieval and version-aware grouping produced the fixed candidate pools; group-level RRF supplies the baseline order. The saved semantic reranker assessed only the research question, title and abstract, without gold labels or judgment reasons. Complete human gold judgments are joined by group_id only for offline evaluation.

Recorded model(s): `gpt-5.6-terra`. Semantic ranking uses score alone, retaining RRF order on exact ties. This report reads completed runs and makes no API requests.

All metrics are recomputed and checked against saved results (relative tolerance 1e-9; absolute tolerance 1e-12). Precision counts gold grades >= 1; strict precision counts grade 2; both divide by 10. nDCG uses gain 2**relevance - 1, logarithmic rank discount, and all gold judgments for the ideal ranking. Deltas are semantic minus RRF.

## Overall Results

| Benchmark | Candidates | RRF P@10 | Semantic P@10 | RRF Strict P@10 | Semantic Strict P@10 | RRF nDCG@10 | Semantic nDCG@10 | Δ nDCG |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matrix_completion_v1 | 22 | 0.6000000 | 1.0000000 | 0.5000000 | 0.7000000 | 0.6781498 | 1.0000000 | 0.3218502 |
| rag_hallucination_v1 | 17 | 1.0000000 | 1.0000000 | 0.3000000 | 0.4000000 | 0.8104157 | 0.9588342 | 0.1484186 |
| cot_reasoning_faithfulness_v1 | 22 | 0.8000000 | 1.0000000 | 0.3000000 | 0.6000000 | 0.6398671 | 0.9669081 | 0.3270410 |

## Aggregate Results

Unweighted macro averages across 3 benchmarks; each benchmark contributes equally regardless of pool size.

| Metric | Mean RRF | Mean semantic | Mean delta |
| --- | --- | --- | --- |
| Precision@10 | 0.8000000 | 1.0000000 | 0.2000000 |
| Strict Precision@10 | 0.3666667 | 0.5666667 | 0.2000000 |
| nDCG@10 | 0.7094775 | 0.9752475 | 0.2657699 |

## Benchmark: matrix_completion_v1

**Question:** How does spectral expansion affect deterministic matrix completion under fixed observation patterns?

Candidates: 22; gold: 7 direct, 5 supporting, 10 off_target. Model: `gpt-5.6-terra`; semantic run timestamp: 2026-09-16T11:11:06.262222+00:00.

### Metric Comparison

| Metric | RRF | Semantic | Delta |
| --- | --- | --- | --- |
| Precision@10 | 0.6000000 | 1.0000000 | 0.4000000 |
| Strict Precision@10 | 0.5000000 | 0.7000000 | 0.2000000 |
| nDCG@10 | 0.6781498 | 1.0000000 | 0.3218502 |

### Semantic Top 10

| Semantic rank | Group ID | Title | RRF rank | Score | Model category | Gold relevance | Gold label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | version:dec105bceb966bf1aa8f6cd51a7d24e6c163e5d91b465af831dbc2f01d4e6d84 | Matrix Completion with Deterministic Sampling: Theories and Methods | 10 | 0.9500 | direct | 2 | direct |
| 2 | version:f29052f8eb35c02608aae9b52dda225bfcd26fd4251a36b06d926ad88dffe7da | Spectral Gap-Based Seismic Survey Design | 13 | 0.9500 | direct | 2 | direct |
| 3 | version:fb9cd1e79b4020882e1225a96ef81ab3fbae5a6c39cb6844328dabbd8b559644 | A Characterization of Deterministic Sampling Patterns for Low-Rank Matrix Completion | 1 | 0.9000 | direct | 2 | direct |
| 4 | version:0510fb81d5019e24410f9236266b95194b46e2987f16d2ed215fa52f5154c2e0 | De-biasing low-rank projection for matrix completion | 4 | 0.9000 | direct | 2 | direct |
| 5 | version:380ff13e0cdc1f5a5130e3ea67fb6f4173ee368a9a4595c2d8d3814e469fce24 | Matrix Completion With Deterministic Pattern: A Geometric Perspective | 7 | 0.9000 | direct | 2 | direct |
| 6 | version:6486d681af3f09c380f1939ae9d5598a641897854f94562ab38cd24d05ab0e32 | On Deterministic Sampling Patterns for Robust Low-Rank Matrix Completion | 3 | 0.8200 | direct | 2 | direct |
| 7 | version:356d9b588772daac1414fad417b775c5521ffbbd756daa09a98e93562cf6ae89 | A New Theory for Matrix Completion | 12 | 0.8200 | direct | 2 | direct |
| 8 | version:cad2b6d607cc39f41f44cf1acb5b92a031f84e5b34656c89554398ddf5ae4257 | A simulation-free seismic survey design by maximizing the spectral gap | 22 | 0.8200 | direct | 1 | supporting |
| 9 | version:38ed31c056bf1fc70ccfb98c98e97763737dc5cf0950682f39575abaa37a9306 | Spectral gap in random bipartite biregular graphs and applications | 6 | 0.7800 | supporting | 1 | supporting |
| 10 | version:6f259d593832712fb17c9941aafdae060b5c062eff0ff90f3ce28badd05a7c6d | A characterization of sampling patterns for low-rank multi-view data completion problem | 18 | 0.7800 | direct | 1 | supporting |

### Rank Movements

Rank change = original RRF rank - semantic rank. Positive is upward; negative is downward. Equal changes retain original RRF order.

#### Top 5 upward movements

| Title | Gold relevance | RRF rank | Semantic rank | Change | Score |
| --- | --- | --- | --- | --- | --- |
| A simulation-free seismic survey design by maximizing the spectral gap | 1 | 22 | 8 | +14 | 0.8200 |
| Spectral Gap-Based Seismic Survey Design | 2 | 13 | 2 | +11 | 0.9500 |
| Corrections to “A Characterization of Deterministic Sampling Patterns for Low-Rank Matrix Completion” | 1 | 21 | 11 | +10 | 0.7000 |
| Matrix Completion with Deterministic Sampling: Theories and Methods | 2 | 10 | 1 | +9 | 0.9500 |
| A characterization of sampling patterns for low-rank multi-view data completion problem | 1 | 18 | 10 | +8 | 0.7800 |

#### Top 5 downward movements

| Title | Gold relevance | RRF rank | Semantic rank | Change | Score |
| --- | --- | --- | --- | --- | --- |
| A survey of matrix completion methods for recommendation systems | 0 | 5 | 19 | -14 | 0.1200 |
| Implicit Regularization in Nonconvex Statistical Estimation: Gradient Descent Converges Linearly for Phase Retrieval, Matrix Completion, and Blind Deconvolution | 0 | 2 | 15 | -13 | 0.2000 |
| Spectroscopic stimulated Raman scattering imaging of highly dynamic specimens through matrix completion | 0 | 8 | 20 | -12 | 0.1200 |
| Cartesian MR fingerprinting in the eye at 7T using compressed sensing and matrix completion‐based reconstructions | 0 | 14 | 22 | -8 | 0.0800 |
| Matrix Completion and Extrapolation via Kernel Regression | 0 | 16 | 21 | -5 | 0.1200 |

### Category Calibration

Category accuracy over all candidates: 0.7727273. Rows are gold labels; columns are model categories. This is a diagnostic comparison and does not affect ranking.

| Gold / model | direct | supporting | off_target |
| --- | --- | --- | --- |
| direct | 7 | 0 | 0 |
| supporting | 3 | 2 | 0 |
| off_target | 0 | 2 | 8 |

**Artifact compatibility:** Candidate snapshot predates benchmark_id; identity validated by the standard path, matching question and complete group_id sets.

## Benchmark: rag_hallucination_v1

**Question:** How does retrieval-augmented generation reduce hallucination and improve factuality in large language models?

Candidates: 17; gold: 4 direct, 12 supporting, 1 off_target. Model: `gpt-5.6-terra`; semantic run timestamp: 2026-09-16T13:45:22.378870+00:00.

### Metric Comparison

| Metric | RRF | Semantic | Delta |
| --- | --- | --- | --- |
| Precision@10 | 1.0000000 | 1.0000000 | 0.0000000 |
| Strict Precision@10 | 0.3000000 | 0.4000000 | 0.1000000 |
| nDCG@10 | 0.8104157 | 0.9588342 | 0.1484186 |

### Semantic Top 10

| Semantic rank | Group ID | Title | RRF rank | Score | Model category | Gold relevance | Gold label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | version:dad91b6a012ec4c70b8c5162153987b08808c3719a75aba246b0f3838528c413 | Retrieval-Augmented Generation for Large Language Models: A Survey | 1 | 0.9800 | direct | 2 | direct |
| 2 | version:9dc16cf734022405d7deac7415627d9338799889e4702ee8da9e36de88d48cf7 | Benchmarking Large Language Models in Retrieval-Augmented Generation | 6 | 0.9300 | direct | 2 | direct |
| 3 | version:8e3ccef640c165b95e5abae732b290ab2fa11bea83328aaa15b508e96ae00cbc | Retrieval-Augmented Generation for AI-Generated Content: A Survey | 13 | 0.9300 | direct | 1 | supporting |
| 4 | version:530f9fcd99cbb05e7123fde8ac6b4fb565b1fadf7614b73643d6f759b35b4413 | Graph Retrieval-Augmented Generation: A Survey | 10 | 0.9200 | direct | 2 | direct |
| 5 | version:55b8fea1a274330da0186b53fb630885f5b43687f531008e089414a626c5a690 | Integrating Retrieval-Augmented Generation with Large Language Models in Nephrology: Advancing Practical Applications | 4 | 0.9000 | direct | 1 | supporting |
| 6 | version:3b183b2565682c9e0a852f62d6f8661369e28531a17a4cc762c10cc4483b972a | LightRAG: Simple and Fast Retrieval-Augmented Generation | 7 | 0.9000 | direct | 1 | supporting |
| 7 | version:157358bbf0b8d63559f35a5d80b621cb0826a1a8ae80b31fcad798e6065b6e9f | Retrieval-augmented generation for educational application: A systematic survey | 9 | 0.9000 | direct | 1 | supporting |
| 8 | version:5fbbc5b4185fff73f68233c73d34144a6a7b1bca46c0e0247ffc6045722948c7 | Retrieval augmented generation for large language models in healthcare: A systematic review | 12 | 0.9000 | direct | 1 | supporting |
| 9 | version:581a31ee29ae92d2bfa971839069ce1f9b79d77361ff5f7ee06d46299349bfd9 | Retrieval-Augmented Generation (RAG) in Healthcare: A Comprehensive Review | 16 | 0.9000 | direct | 2 | direct |
| 10 | version:4fbc6af4e0bd5627101a52f16405d33b4b15043f4efaac31ca9a5240f4977fb7 | Benchmarking Retrieval-Augmented Generation for Medicine | 3 | 0.8800 | direct | 1 | supporting |

### Rank Movements

Rank change = original RRF rank - semantic rank. Positive is upward; negative is downward. Equal changes retain original RRF order.

#### Top 5 upward movements

| Title | Gold relevance | RRF rank | Semantic rank | Change | Score |
| --- | --- | --- | --- | --- | --- |
| Retrieval-Augmented Generation for AI-Generated Content: A Survey | 1 | 13 | 3 | +10 | 0.9300 |
| Retrieval-Augmented Generation (RAG) in Healthcare: A Comprehensive Review | 2 | 16 | 9 | +7 | 0.9000 |
| Graph Retrieval-Augmented Generation: A Survey | 2 | 10 | 4 | +6 | 0.9200 |
| CareerX: A Retrieval-Augmented Generation Framework for Personalized AI-Driven Career Guidance | 1 | 17 | 12 | +5 | 0.8000 |
| Benchmarking Large Language Models in Retrieval-Augmented Generation | 2 | 6 | 2 | +4 | 0.9300 |

#### Top 5 downward movements

| Title | Gold relevance | RRF rank | Semantic rank | Change | Score |
| --- | --- | --- | --- | --- | --- |
| Active Retrieval Augmented Generation | 1 | 2 | 13 | -11 | 0.7200 |
| RAGAs: Automated Evaluation of Retrieval Augmented Generation | 1 | 5 | 16 | -11 | 0.6000 |
| Benchmarking Retrieval-Augmented Generation for Medicine | 1 | 3 | 10 | -7 | 0.8800 |
| Retrieval-Augmented Generation (RAG) | 0 | 11 | 17 | -6 | 0.1800 |
| Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering | 1 | 8 | 11 | -3 | 0.8800 |

### Category Calibration

Category accuracy over all candidates: 0.4117647. Rows are gold labels; columns are model categories. This is a diagnostic comparison and does not affect ranking.

| Gold / model | direct | supporting | off_target |
| --- | --- | --- | --- |
| direct | 4 | 0 | 0 |
| supporting | 10 | 2 | 0 |
| off_target | 0 | 0 | 1 |

## Benchmark: cot_reasoning_faithfulness_v1

**Question:** How does chain-of-thought prompting affect reasoning accuracy and faithfulness in large language models?

Candidates: 22; gold: 6 direct, 11 supporting, 5 off_target. Model: `gpt-5.6-terra`; semantic run timestamp: 2026-09-17T04:22:44.106015+00:00.

### Metric Comparison

| Metric | RRF | Semantic | Delta |
| --- | --- | --- | --- |
| Precision@10 | 0.8000000 | 1.0000000 | 0.2000000 |
| Strict Precision@10 | 0.3000000 | 0.6000000 | 0.3000000 |
| nDCG@10 | 0.6398671 | 0.9669081 | 0.3270410 |

### Semantic Top 10

| Semantic rank | Group ID | Title | RRF rank | Score | Model category | Gold relevance | Gold label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | version:e9104717ecfc98f2059ef436df547e8a46594334eda802a3e7aca3fe6a8c9787 | Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting | 17 | 0.9900 | direct | 2 | direct |
| 2 | version:d03caa07de55f8058e95b7c349f29e340d42935e5c07ebf295205c96262f8f6a | Measuring Faithfulness in Chain-of-Thought Reasoning | 11 | 0.9800 | direct | 2 | direct |
| 3 | version:d13364bb8b96515019ee3ad86b9d07dba9720e9a85da21e758e81999acc1dd25 | SCOTT: Self-Consistent Chain-of-Thought Distillation | 14 | 0.9500 | direct | 1 | supporting |
| 4 | version:a6f8cb54e40040f477e0c85b8191b979e22ce74bdd8cb2a2e6f96d6c4091fa42 | Faithful Logical Reasoning via Symbolic Chain-of-Thought | 9 | 0.9400 | direct | 2 | direct |
| 5 | version:2830797d09de086393f52b6252c18eba4cf2b896b5f9174d9782b821f2225f42 | Large Language Models are Zero-Shot Reasoners | 18 | 0.9000 | direct | 2 | direct |
| 6 | version:c00e8ba42890369d7f38211de65512e1c1c7833d347b76f6ea34b5b696a69ea0 | Automatic Chain of Thought Prompting in Large Language Models | 1 | 0.8400 | direct | 2 | direct |
| 7 | version:56d9d87e4b754bd411bf976805e42bf5d5f4b6e50cc59858e6eb6d621ee1b23c | Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models | 4 | 0.8400 | direct | 1 | supporting |
| 8 | version:5bb599df2e747b966e596e1604002f6b891fd237384aa00332d349642c513b6e | Self-Consistency Improves Chain of Thought Reasoning in Language Models | 7 | 0.8400 | direct | 2 | direct |
| 9 | version:993e12e4013d782d9415f5a747dedb772fa95b995f3ae941c2ce38d10e3230a1 | Faithful Chain-of-Thought Reasoning | 2 | 0.8300 | direct | 1 | supporting |
| 10 | version:a4e5a68c9f6143dad856a15d062d20e20b896fd3834de4098139e3ecfcdfd4df | Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions | 3 | 0.8200 | direct | 1 | supporting |

### Rank Movements

Rank change = original RRF rank - semantic rank. Positive is upward; negative is downward. Equal changes retain original RRF order.

#### Top 5 upward movements

| Title | Gold relevance | RRF rank | Semantic rank | Change | Score |
| --- | --- | --- | --- | --- | --- |
| Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting | 2 | 17 | 1 | +16 | 0.9900 |
| Large Language Models are Zero-Shot Reasoners | 2 | 18 | 5 | +13 | 0.9000 |
| SCOTT: Self-Consistent Chain-of-Thought Distillation | 1 | 14 | 3 | +11 | 0.9500 |
| Measuring Faithfulness in Chain-of-Thought Reasoning | 2 | 11 | 2 | +9 | 0.9800 |
| Tree of Thoughts: Deliberate Problem Solving with Large Language Models | 1 | 19 | 11 | +8 | 0.8200 |

#### Top 5 downward movements

| Title | Gold relevance | RRF rank | Semantic rank | Change | Score |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning | 1 | 8 | 18 | -10 | 0.6800 |
| Applying large language models and chain-of-thought for automatic scoring | 0 | 10 | 19 | -9 | 0.6600 |
| Reasoning Implicit Sentiment with Chain-of-Thought Prompting | 0 | 5 | 13 | -8 | 0.7800 |
| Faithful Chain-of-Thought Reasoning | 1 | 2 | 9 | -7 | 0.8300 |
| Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions | 1 | 3 | 10 | -7 | 0.8200 |

### Category Calibration

Category accuracy over all candidates: 0.3181818. Rows are gold labels; columns are model categories. This is a diagnostic comparison and does not affect ranking.

| Gold / model | direct | supporting | off_target |
| --- | --- | --- | --- |
| direct | 6 | 0 | 0 |
| supporting | 10 | 1 | 0 |
| off_target | 2 | 3 | 0 |

## Observations

- `matrix_completion_v1`: nDCG@10 increased (delta +0.3218502); Precision@10 changed by +0.4000000, and Strict Precision@10 by +0.2000000.

- `rag_hallucination_v1`: nDCG@10 increased (delta +0.1484186); Precision@10 changed by +0.0000000, and Strict Precision@10 by +0.1000000.

- `cot_reasoning_faithfulness_v1`: nDCG@10 increased (delta +0.3270410); Precision@10 changed by +0.2000000, and Strict Precision@10 by +0.3000000.

## Limitations

- Only 3 frozen benchmarks are included; candidate pools are small (22, 17, 22 groups).
- Human relevance judgments are manually constructed.
- One included benchmark, matrix_completion_v1, was involved in prompt-rubric refinement and is not an independent held-out test of that rubric.
- Semantic scores come from one recorded model/configuration; no model comparison is available here.
- No repeated-run variance analysis has been performed.
- Evidence is limited to the included frozen benchmarks. These descriptive results do not establish statistical significance or universal generalization.
