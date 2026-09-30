# ResearchPilot

**An Evidence-Grounded Agentic RAG System for Academic Research and Claim Verification**

Version: `0.1.0`
Stage: MVP

---

## 1. Problem Definition

传统大语言模型可以回答研究问题，但存在三个核心问题：

1. 可能依赖模型内部知识，而不是最新文献；
2. 生成的结论可能没有可靠证据支持；
3. 即使使用普通 RAG，也通常只是“一次检索 + 一次生成”，无法发现证据不足并主动继续研究。

ResearchPilot 的目标不是简单地“总结论文”，而是模拟一个基础研究助理的工作流：

$$
\text{Question}
\rightarrow
\text{Plan}
\rightarrow
\text{Search}
\rightarrow
\text{Retrieve}
\rightarrow
\text{Verify}
\rightarrow
\text{Search Again if Needed}
\rightarrow
\text{Synthesize}
$$

核心原则：

> **No important claim without evidence.**

---

## 2. Target User

MVP 的目标用户是：

* 硕士生
* 博士生
* Research Assistant
* 需要快速进入某个研究领域的研究人员
* 需要进行 literature review 的技术人员

第一版重点支持：

**Computer Science / Machine Learning / Mathematics**

暂时不追求覆盖所有学科。

---

## 3. Core User Scenario

用户提出：

> How does spectral expansion affect deterministic matrix completion under fixed observation patterns?

ResearchPilot 不直接回答。

首先生成研究计划：

```text
Main Question
│
├── Q1. What deterministic matrix completion methods use graph structures?
│
├── Q2. How is spectral expansion defined in these works?
│
├── Q3. How do σ₂ or σ₂/σ₁ enter recovery guarantees?
│
├── Q4. What assumptions are required?
│
└── Q5. Is there an explicit relationship with RSC/RSS?
```

然后针对这些问题搜索论文并收集证据。

---

## 4. MVP Input

第一版输入只保留必要变量。

```json
{
  "question": "How does spectral expansion affect deterministic matrix completion?",
  "year_from": 2015,
  "year_to": 2026,
  "max_papers": 15
}
```

其中只有：

```text
question
```

是必填参数。

其他参数都有默认值。

默认：

```text
year_from = null
year_to = current_year
max_papers = 15
```

---

## 5. MVP Output

ResearchPilot 最终输出五类信息。

### A. Research Plan

```json
{
  "main_question": "...",
  "sub_questions": [
    "...",
    "...",
    "..."
  ],
  "search_queries": [
    "...",
    "...",
    "..."
  ]
}
```

---

### B. Paper Shortlist

每篇论文至少保存：

```json
{
  "paper_id": "...",
  "title": "...",
  "authors": [],
  "year": 2024,
  "abstract": "...",
  "doi": "...",
  "arxiv_id": "...",
  "citation_count": 120,
  "source": "OpenAlex",
  "relevance_score": 0.91
}
```

---

### C. Evidence

每条 evidence 必须具有来源。

```json
{
  "evidence_id": "E001",
  "paper_id": "...",
  "section": "Theorem 4.1",
  "page": 12,
  "text": "...",
  "retrieval_score": 0.87
}
```

---

### D. Verified Claims

Agent 不能只生成自然语言 conclusion。

每个核心结论首先转化成 Claim。

例如：

```json
{
  "claim_id": "C001",
  "claim": "Spectral expansion controls deviation from uniform sampling.",
  "status": "supported",
  "confidence": 0.91,
  "evidence_ids": [
    "E001",
    "E017"
  ]
}
```

`status` 第一版限制为：

```text
supported

partially_supported

unsupported

conflicting
```

---

### E. Final Report

最终报告：

```text
Research Question

Executive Summary

Literature Landscape

Key Findings

Evidence

Comparison of Methods

Conflicting / Uncertain Findings

Research Gaps

Suggested Next Steps

References
```

所有重要结论必须能够追溯到：

```text
claim
→ evidence
→ paper
→ page/section
```

---

## 6. MVP Agent Behaviour

第一版只有一个 Agent：

```text
Research Orchestrator
```

它可以调用 Tools，但不拆成多个 Agent。

Agent 负责决定：

```text
What should I search?

Which papers are relevant?

What evidence do I need?

Is current evidence sufficient?

Should I search again?

Can this claim be included in the report?
```

Agent 不直接负责底层 retrieval 算法。

---

## 7. MVP Tools

最终 MVP 预计提供以下工具：

```text
search_papers()

get_paper_metadata()

get_citations()

download_paper()

retrieve_evidence()

verify_claim()

search_missing_evidence()
```

第一阶段不需要一次性实现全部工具。

---

## 8. Retrieval Architecture

系统采用两级 Retrieval。

### Level 1 — Paper Retrieval

目标：

$$
\text{Research Question}
\rightarrow
\text{Relevant Papers}
$$

数据来源：

```text
OpenAlex
arXiv
Semantic Scholar
```

例如：

```text
300 candidates
      ↓
deduplication
      ↓
100
      ↓
metadata ranking
      ↓
15 selected papers
```

---

### Level 2 — Evidence Retrieval

针对选中的论文：

```text
PDF
 ↓
Parser
 ↓
Structure-aware chunks
 ↓
BM25
 +
Dense Retrieval
 ↓
RRF
 ↓
Cross Encoder
 ↓
Top Evidence
```

因此我们明确区分：

$$
\boxed{\text{Paper Retrieval}}
$$

和

$$
\boxed{\text{Passage / Evidence Retrieval}}
$$

---

## 9. Agent Loop

MVP 的核心逻辑：

```text
START
  │
  ▼
Understand Question
  │
  ▼
Generate Research Plan
  │
  ▼
Search Papers
  │
  ▼
Select Papers
  │
  ▼
Retrieve Evidence
  │
  ▼
Generate Candidate Claims
  │
  ▼
Verify Claims
  │
  ▼
Evidence sufficient?
  │
  ├──────── YES ────────┐
  │                     │
  NO                    ▼
  │                Generate Report
  ▼                     │
Generate New Query       ▼
  │                Check Citations
  │                     │
  └────── Search ◄──────┘
                        │
                        ▼
                       END
```

---

## 10. ResearchState

所有流程共享一个统一状态对象。

概念设计：

```python
ResearchState = {
    "research_question": str,

    "sub_questions": list[str],

    "search_queries": list[str],

    "candidate_papers": list,

    "selected_papers": list,

    "retrieved_evidence": list,

    "claims": list,

    "unsupported_claims": list,

    "iteration": int,

    "final_report": str
}
```

第一版最大 research iteration：

```text
3
```

防止 Agent 无限搜索。

---

## 11. MVP Non-Goals

以下功能明确不属于 Version 0.1：

```text
Multi-Agent

用户登录

长期用户 Memory

论文推荐系统

自动写完整论文

自动生成 LaTeX manuscript

Google Scholar scraping

Web UI

手机 App

复杂权限管理

云部署

Fine-tuning
```

这些功能以后可以增加，但现在不允许影响 MVP。

---

## 12. Technology Boundary

第一版预计采用：

```text
Language
Python

API Framework
FastAPI

Data Validation
Pydantic

Agent
OpenAI Agents SDK / equivalent tool-calling architecture

Paper Metadata
OpenAlex
arXiv
Semantic Scholar

PDF Parsing
PyMuPDF / suitable scientific PDF parser

Sparse Retrieval
BM25

Dense Retrieval
Embedding Model + FAISS

Fusion
Reciprocal Rank Fusion

Reranking
Cross Encoder

Metadata Database
SQLite

Vector Store
FAISS
```

第一版不使用 Kubernetes、Redis、Kafka 或复杂微服务。

---

## 13. Project Directory Target

最终项目计划采用：

```text
researchpilot/
│
├── app/
│   ├── agents/
│   │
│   ├── tools/
│   │
│   ├── retrieval/
│   │
│   ├── verification/
│   │
│   ├── models/
│   │
│   ├── services/
│   │
│   └── api/
│
├── data/
│   ├── papers/
│   ├── parsed/
│   └── indexes/
│
├── eval/
│   ├── datasets/
│   ├── retrieval/
│   └── agent/
│
├── tests/
│
├── scripts/
│
├── .env.example
├── pyproject.toml
├── README.md
└── PROJECT_SPEC.md
```

目前只是目标结构，不在 Step 0 创建所有文件。

---

## 14. MVP Success Criteria

只有达到以下标准，我们才认为 MVP 完成。

| Module             | Minimum Requirement            |
| ------------------ | ------------------------------ |
| Question Planning  | 能把一个研究问题拆成合理子问题                |
| Paper Search       | 能从真实学术 API 获取候选论文              |
| Deduplication      | 同一论文不会因不同来源重复出现                |
| Paper Ranking      | 能选出与问题相关的 Top Papers           |
| PDF Pipeline       | 能下载并解析可公开访问的论文                 |
| Chunking           | 保留 paper/section/page metadata |
| Retrieval          | 能根据问题检索相关 evidence             |
| Claim Verification | 能识别至少 supported / unsupported  |
| Agent Loop         | evidence 不足时能够再次搜索             |
| Citation           | 重要 claim 能追溯到具体论文证据            |
| Evaluation         | 至少有一组 quantitative benchmark   |

---

## 15. First End-to-End Test

整个项目以后始终保留一个固定测试问题：

```text
How does spectral expansion affect deterministic matrix
completion under fixed observation patterns?
```

预期系统能够发现：

```text
matrix completion

deterministic sampling

graph / bipartite graph

spectral expansion

singular values

incoherence

restricted curvature / recovery guarantee
```

等相关概念。

但系统不能因为这些是预期关键词，就硬编码搜索结果。

所有结果必须来自真实检索。

---

## 16. Core Engineering Principle

ResearchPilot 的价值不是：

> “LLM knows research papers.”

而是：

> “The system knows how to obtain, retrieve, verify and trace evidence.”

因此整个系统的核心对象不是 `answer`。

而是：

$$
\boxed{
Question
\rightarrow
Claim
\rightarrow
Evidence
\rightarrow
Source
}
$$

任何无法建立这条链的重要结论，都必须标记为 uncertain 或 unsupported，而不能直接写进最终结论。
