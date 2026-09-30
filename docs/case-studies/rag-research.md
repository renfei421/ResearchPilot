# RAG research: a useful answer with explicit gaps

## Research Question

> How does retrieval-augmented generation reduce hallucination in large language
> models, and under what conditions can poor retrieval or irrelevant context make
> factuality worse?

This case summarizes a completed historical Research Mode run. It reached its
two-round search budget and returned an explicitly partial answer. The saved
record contained seven claim assessments: three supported and four partially
supported. Those are support judgments within the retrieved evidence, not seven
independent scientific discoveries.

## Why This Is Difficult

The question combines a mechanism, an empirical benefit, and conditions under
which that benefit reverses. A survey saying that retrieval can improve accuracy
does not by itself establish the size of the improvement against the same model
without retrieval. Likewise, evidence that noisy context can mislead a model is
not automatically evidence that every irrelevant document makes it worse.

There is also a distinction between an answer being faithful to retrieved text
and that text being factually correct. A system can reproduce retrieved
misinformation faithfully. The investigation therefore needed evidence for
specific mechanisms and comparisons, with their scope left visible.

## What ResearchPilot Did

ResearchPilot planned and executed eight queries across two search rounds,
aggregated paper candidates, grouped publication versions, and ranked the groups
before acquiring sources. Of twelve selected papers, seven yielded PDF text and
five were represented by marked abstracts.

It retrieved and selected page-aware passages, synthesized claims, checked their
atomic assertions, and reassessed the remaining evidence gaps. Two local rescue
passes revisited available documents. No PDF-page vision call was needed in this
run. The retained findings were rendered with source/page references, while
unsupported portions were omitted or qualified.

This was a bounded evidence review: reaching the search limit did not cause the
system to declare the entire question answered.

## Selected Findings

| Finding retained or qualified in the saved answer | Evidence location |
| --- | --- |
| Retrieved external knowledge was described as a way to improve accuracy and reliability, but a controlled improvement over a no-retrieval baseline remained unestablished in this evidence pool. | *Benchmarking Large Language Models in Retrieval-Augmented Generation*, physical pp. 1, 3. |
| Irrelevant, distracting, insufficient, or false retrieved information can contribute to incorrect outputs. The supported part was retained without asserting a universal performance comparison. | *Survey on Factuality in Large Language Models: Knowledge, Retrieval and Domain-Specificity*, physical p. 28. |
| Similar-looking information can be misleading: the benchmark described documents about the 2021 Nobel Prize distracting a question about the 2022 prize. | *Benchmarking Large Language Models in Retrieval-Augmented Generation*, physical p. 2. |
| Integrating information across multiple documents remained a challenge in the evaluated benchmark setting. | The same benchmark, physical p. 2. |

These locations make citations such as `[P1, p. 2]` inspectable. The public
references are the [benchmark paper](https://doi.org/10.1609/aaai.v38i16.29728),
[factuality survey](https://doi.org/10.48550/arxiv.2310.07521), and
[RAG survey](https://doi.org/10.48550/arxiv.2312.10997). The latter also contributed
to the unresolved comparison between harmful noise and reports of benefits from
some irrelevant context. The run did not reconcile those outcomes into a general
rule.

## What ResearchPilot Did Not Claim

The final answer explicitly retained gaps about:

- a controlled, same-model estimate of hallucination reduction relative to
  generation without retrieval;
- the conditions distinguishing harmful irrelevant context from context that
  improves an evaluated outcome;
- whether relevance and faithfulness jointly form a necessary condition for
  factual answers;
- the exact no-answer-bearing-context scope of a reported rejection result.

It did not fill those gaps with a universal percentage, a claim that all noise
is harmful, or a statement that RAG eliminates hallucination. In particular,
benchmark-specific numbers were not promoted to universal factuality estimates.

## What This Demonstrates

ResearchPilot preferred an explicitly partial answer to a more complete-looking
answer whose comparisons were unsupported. A reader can distinguish a supported
mechanism from an unresolved causal or quantitative claim, then decide what
additional evidence would actually be useful.

The value here is the connection between findings, citations, verification, and
the remaining work. The presence of a fluent answer was not used as a substitute
for evidence coverage.

## Limitations

This is one historical result, not evidence of exhaustive recall or guaranteed
verification accuracy. Accessible surveys supplied much of the evidence, several
papers had only abstracts, and the two-round budget left material gaps. Current
ResearchPilot also has hybrid passage retrieval and cross-page assembly; this
case does not retroactively attribute those later capabilities to the historical
run. A fresh search can select different papers and reach different assessments.

[All case studies](README.md) · [Architecture](../architecture.md)
