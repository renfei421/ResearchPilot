# Query Planner V1: frozen canonical retrieval evaluation

## 1. Experiment definition

Human handcrafted queries are compared with frozen LLM Query Planner V1 using the existing canonical RRF orders. No retrieval, API calls, candidate generation, or semantic reranking is performed.

Recall uses all retrieved canonical works and the full union denominator. Precision@10 divides by 10. nDCG@10 uses gain `2**relevance - 1`, discount `log2(rank + 1)`, and an ideal order drawn from the full union. Relevance >= 1 is relevant; relevance == 2 is Direct/Core.

IDCG now uses the expanded union, so these nDCG values are not directly comparable with earlier baselines whose ideal pools contained only Human-retrieved works.

## 2. Final relevance-set construction

The final union relevance set combines 61 frozen inherited Human judgments and 79 project-owner-adopted judgments. Human judgments take precedence; adopted labels fill only LLM-only works. Historical model-assisted provenance and confidence are retained for audit, with no weighting, exclusion, or further review requirement. Manual task files and semantic assessment files are not evaluation inputs.

| Benchmark | Inherited | Adopted | Final | Label 0 | Label 1 | Label 2 |
| --- | --- | --- | --- | --- | --- | --- |
| matrix_completion_v1 | 22 | 18 | 40 | 19 | 9 | 12 |
| rag_hallucination_v1 | 17 | 26 | 43 | 7 | 24 | 12 |
| cot_reasoning_faithfulness_v1 | 22 | 35 | 57 | 26 | 23 | 8 |

Total: 140 benchmark-work judgments; labels 0/1/2 = 52/56/32.

## 3. Canonical pool statistics

| Benchmark | Union | Human | LLM | Intersection | Human-only | LLM-only |
| --- | --- | --- | --- | --- | --- | --- |
| matrix_completion_v1 | 40 | 22 | 30 | 12 | 10 | 18 |
| rag_hallucination_v1 | 43 | 17 | 38 | 12 | 5 | 26 |
| cot_reasoning_faithfulness_v1 | 57 | 22 | 42 | 7 | 15 | 35 |

## 4. Per-benchmark metrics

### matrix_completion_v1

How does spectral expansion affect deterministic matrix completion under fixed observation patterns?

Shared recall denominators: relevant = 21; Direct/Core = 12.

| Metric | Human | LLM | Delta (LLM - Human) |
| --- | --- | --- | --- |
| Relevant Recall | 0.5714285714285714 | 0.8095238095238095 | 0.23809523809523814 |
| Direct Recall | 0.5833333333333334 | 0.8333333333333334 | 0.25 |
| Precision@10 | 0.6 | 0.9 | 0.30000000000000004 |
| Strict Precision@10 | 0.5 | 0.5 | 0.0 |
| nDCG@10 | 0.5880434804412078 | 0.7010817729721175 | 0.11303829253090969 |

### rag_hallucination_v1

How does retrieval-augmented generation reduce hallucination and improve factuality in large language models?

Shared recall denominators: relevant = 36; Direct/Core = 12.

| Metric | Human | LLM | Delta (LLM - Human) |
| --- | --- | --- | --- |
| Relevant Recall | 0.4444444444444444 | 0.8888888888888888 | 0.4444444444444444 |
| Direct Recall | 0.3333333333333333 | 0.9166666666666666 | 0.5833333333333333 |
| Precision@10 | 1.0 | 1.0 | 0.0 |
| Strict Precision@10 | 0.3 | 0.5 | 0.2 |
| nDCG@10 | 0.5747405489838048 | 0.6885976968000369 | 0.11385714781623213 |

### cot_reasoning_faithfulness_v1

How does chain-of-thought prompting affect reasoning accuracy and faithfulness in large language models?

Shared recall denominators: relevant = 31; Direct/Core = 8.

| Metric | Human | LLM | Delta (LLM - Human) |
| --- | --- | --- | --- |
| Relevant Recall | 0.5483870967741935 | 0.6451612903225806 | 0.09677419354838712 |
| Direct Recall | 0.75 | 0.625 | -0.125 |
| Precision@10 | 0.8 | 0.6 | -0.20000000000000007 |
| Strict Precision@10 | 0.3 | 0.4 | 0.10000000000000003 |
| nDCG@10 | 0.5731797942603793 | 0.6298879429856533 | 0.056708148725273966 |

## 5. Macro metrics

Equal-weight arithmetic mean across the three benchmarks; no weighting by candidate-pool size.

| Metric | Human | LLM | Delta (LLM - Human) |
| --- | --- | --- | --- |
| Relevant Recall | 0.5214200375490697 | 0.7811913295784264 | 0.25977129202935667 |
| Direct Recall | 0.5555555555555556 | 0.7916666666666666 | 0.23611111111111105 |
| Precision@10 | 0.8000000000000002 | 0.8333333333333334 | 0.033333333333333215 |
| Strict Precision@10 | 0.3666666666666667 | 0.4666666666666666 | 0.09999999999999992 |
| nDCG@10 | 0.5786546078951306 | 0.6731891375859359 | 0.09453452969080534 |

## 6. Unique relevance composition

| Benchmark | Source-only | Total | Off-target | Supporting | Direct/Core | Relevant |
| --- | --- | --- | --- | --- | --- | --- |
| matrix_completion_v1 | human_only | 10 | 6 | 2 | 2 | 4 |
| matrix_completion_v1 | llm_only | 18 | 9 | 4 | 5 | 9 |
| rag_hallucination_v1 | human_only | 5 | 1 | 3 | 1 | 4 |
| rag_hallucination_v1 | llm_only | 26 | 6 | 12 | 8 | 20 |
| cot_reasoning_faithfulness_v1 | human_only | 15 | 4 | 8 | 3 | 11 |
| cot_reasoning_faithfulness_v1 | llm_only | 35 | 21 | 12 | 2 | 14 |
| Total benchmark-work pairs | human_only | 30 | 11 | 13 | 6 | 19 |
| Total benchmark-work pairs | llm_only | 79 | 36 | 28 | 15 | 43 |

## 7. Shared-work rank diagnostics

Ranks are one-based positions after canonical deduplication. Positive delta (`Human - LLM`) means LLM placed the shared work higher. These are diagnostics, not primary quality metrics. Original best source ranks are retained in JSON.

| Benchmark | Shared | LLM higher | Human higher | Tied |
| --- | --- | --- | --- | --- |
| matrix_completion_v1 | 12 | 4 | 7 | 1 |
| rag_hallucination_v1 | 12 | 3 | 6 | 3 |
| cot_reasoning_faithfulness_v1 | 7 | 3 | 3 | 1 |

### Shared works: matrix_completion_v1

| union_id | Title | Relevance | Human rank | LLM rank | Delta |
| --- | --- | --- | --- | --- | --- |
| union:5b5e3936a729135b882463b7d9076fd420b827764c807af1837e5e4436d80e2a | A Characterization of Deterministic Sampling Patterns for Low-Rank Matrix Completion | 2 | 1 | 1 | 0 |
| union:8c208e41b2f7692be27d2b865ec81b764c7588593776e60050ff12c9cbd90ce7 | Implicit Regularization in Nonconvex Statistical Estimation: Gradient Descent Converges Linearly for Phase Retrieval, Matrix Completion, and Blind Deconvolution | 0 | 2 | 11 | -9 |
| union:9605632ea0321cae4bd5d59f31c66db4b8c1674f2011ea8ab9a09b8493ef19da | On Deterministic Sampling Patterns for Robust Low-Rank Matrix Completion | 2 | 3 | 4 | -1 |
| union:8acd5654dad7b3d70043f9057a711574dab2f5cb4c8c0135c26c95ed649d727d | De-biasing low-rank projection for matrix completion | 2 | 4 | 19 | -15 |
| union:68a8fd801cffce5fa46db30cbee335bd5e3225963cad070d777eae41533fb57e | Spectral gap in random bipartite biregular graphs and applications | 1 | 6 | 2 | 4 |
| union:38e35de0a38fa7cad8de50775cdd2e8bfa0ee6895f4ae8a6a6b143a9a52f8131 | Size biased couplings and the spectral gap for random regular graphs | 0 | 9 | 16 | -7 |
| union:e12f93b69cbc4325423a27b89e776e1a89b549a663a1e4c2585524dfca583c3b | Matrix Completion with Deterministic Sampling: Theories and Methods | 2 | 10 | 12 | -2 |
| union:2603d17873118180508539c71341059b98496dbfeb574c7e9b72d0a10a97b9a8 | Entrywise eigenvector analysis of random matrices with low expected rank | 0 | 11 | 21 | -10 |
| union:f38fd1fd59b1645ebb9488dc75e0ff42a144de7c43988dddbd395e291d27ff60 | Spectral Gap-Based Seismic Survey Design | 2 | 13 | 3 | 10 |
| union:5d93de44ef3ddb61824e4e79d34876684ddea489fee8521b8571e05f01e4c2e0 | 1-Bit Matrix Completion under Exact Low-Rank Constraint | 0 | 15 | 26 | -11 |
| union:0d8caa4ab9f1db38af460d427ccb777c5293327859ed69d2e8ed7fa11f816300 | Matrix Completion from $O(n)$ Samples in Linear Time | 1 | 17 | 9 | 8 |
| union:02d6416cacc6d69210d51d5f6ecddc38077f7a57f4564c3740bfad9d64000c50 | Corrections to “A Characterization of Deterministic Sampling Patterns for Low-Rank Matrix Completion” | 1 | 21 | 6 | 15 |

### Shared works: rag_hallucination_v1

| union_id | Title | Relevance | Human rank | LLM rank | Delta |
| --- | --- | --- | --- | --- | --- |
| union:a4ddb3a6e9858f507eb9e977e7ccc4dfbdaaeb8371c0c7b2846b3ec90635bb4b | Retrieval-Augmented Generation for Large Language Models: A Survey | 2 | 1 | 1 | 0 |
| union:0f068f25a4a6f68544edd1a9cd83f1ff9fcaf9455d9ca3ce2470945147ba83a2 | Active Retrieval Augmented Generation | 1 | 2 | 2 | 0 |
| union:995e2944980324fe0b1d9a04b09927594ade2ce461260685fbe8abcda17e8656 | Benchmarking Retrieval-Augmented Generation for Medicine | 1 | 3 | 26 | -23 |
| union:1a71fa5a65388c849277677dc12e09c3e0778fefad2789a222d7af2148448056 | Integrating Retrieval-Augmented Generation with Large Language Models in Nephrology: Advancing Practical Applications | 1 | 4 | 30 | -26 |
| union:10237208d498c736e85239aa7bf1b872ab4ef571678c77b4e5fd1add7ba37656 | RAGAs: Automated Evaluation of Retrieval Augmented Generation | 1 | 5 | 5 | 0 |
| union:074db086ba7eec9b7e52f6fd804fde2b1b67da0674adf4f4242dd379e0114d9d | Benchmarking Large Language Models in Retrieval-Augmented Generation | 2 | 6 | 4 | 2 |
| union:1d75e64d4955cac034356ba7e1dd8ede7007967fc44dc800fc38c836968d6b35 | LightRAG: Simple and Fast Retrieval-Augmented Generation | 1 | 7 | 16 | -9 |
| union:334aa4bfbd445cd5e87b8967c2c30daba6a5213853559aae9c46d4f8cbaf96a6 | Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering | 1 | 8 | 17 | -9 |
| union:67c18afddba53182dee206eede36d91b8a128fbf08e4693d1103a2ad0b08725b | Retrieval-augmented generation for educational application: A systematic survey | 1 | 9 | 35 | -26 |
| union:e63a3cc4f8ede09d97009385734b2a7fe56e0149bb3d18c698397b4720e19e7a | Retrieval augmented generation for large language models in healthcare: A systematic review | 1 | 12 | 3 | 9 |
| union:ac6faaeb766e720e5090dae3af25160b8e5a6f7aff7ce3bb4d8f42b7649fdb16 | Retrieval-Augmented Generation for AI-Generated Content: A Survey | 1 | 13 | 29 | -16 |
| union:a2d3897bc4e6f9b940fe85509775abb17e349f5151724105aeb114c716b6b401 | Retrieval-Augmented Generation (RAG) in Healthcare: A Comprehensive Review | 2 | 16 | 7 | 9 |

### Shared works: cot_reasoning_faithfulness_v1

| union_id | Title | Relevance | Human rank | LLM rank | Delta |
| --- | --- | --- | --- | --- | --- |
| union:122d9d1b5f8486c091fd5f4bc5cca18fe3ab054f6aa79a197eb26871eabafab5 | Automatic Chain of Thought Prompting in Large Language Models | 2 | 1 | 1 | 0 |
| union:36842c292cc2146163dd9b623ec1ddf92b4168d62c246bca03e4fe4d96c19de8 | Faithful Chain-of-Thought Reasoning | 1 | 2 | 37 | -35 |
| union:05b3b2cebe5ea98dc7029424079b6b63722d0fc67a065e4f7cb695f89a34d9b8 | Reasoning Implicit Sentiment with Chain-of-Thought Prompting | 0 | 5 | 9 | -4 |
| union:60ea6e18d95245a7f9d3181dffa7e4d358a146fc84bc168758e594234cb7cba2 | DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning | 1 | 8 | 17 | -9 |
| union:4febb348917aff092437bf928ca720c441900ceecd770f8448047817a8840ff1 | Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting | 2 | 17 | 7 | 10 |
| union:a62750f462ab6bf0e14ecd109fa89a7e2cb213520fcd5c2163f87c76f9c907b1 | Large Language Models are Zero-Shot Reasoners | 2 | 18 | 5 | 13 |
| union:62c128f6b00f4eb66cc7fad4f52fc21a2b3caff28c0c423610ec874832afc397 | Tree of Thoughts: Deliberate Problem Solving with Large Language Models | 1 | 19 | 10 | 9 |

## 8. Main empirical observations

- Relevant Recall: LLM is higher on 3 benchmarks, lower on 0, and equal on 0.
- Direct Recall: LLM is higher on 2 benchmarks, lower on 1, and equal on 0.
- Precision@10: LLM is higher on 1 benchmarks, lower on 1, and equal on 1.
- Strict Precision@10: LLM is higher on 2 benchmarks, lower on 0, and equal on 1.
- nDCG@10: LLM is higher on 3 benchmarks, lower on 0, and equal on 0.

- human_only: 19 relevant discoveries, including 6 Direct/Core works; 11 off-target works (summed benchmark-work pairs).
- llm_only: 43 relevant discoveries, including 15 Direct/Core works; 36 off-target works (summed benchmark-work pairs).

## 9. Limitations

- Three fixed research questions do not establish general superiority or statistical significance.
- Recall is relative to the pooled retrieved union, not all relevant literature. Unretrieved works are not judged.
- Source pools differ in size; this is the frozen configuration comparison, not a cost- or query-budget-normalized result.
- Adopted labels retain their model-assisted history and title/abstract evidence limitations; project acceptance does not imply independently collected human annotations.
- The experiment evaluates retrieval from frozen plans and existing RRF only. It does not evaluate semantic reranking or downstream research-answer quality.

Reproduce from the repository root: `python -B -m scripts.run_query_planner_eval`. JSON retains raw floats, complete canonical rankings, source-file hashes, and rank diagnostics. Existing identical derived artifacts are left unchanged; `--overwrite` permits replacing only these derived outputs.
