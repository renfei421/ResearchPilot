# RAG Idea Check: preserving the scope of a counterexample

## Claim

> Poor retrieval quality can cause retrieval-augmented generation to become less
> factual than generation without retrieval.

This completed Idea Check run examined a comparative claim. The important
baseline was **generation without retrieval**, not merely a better retriever or
a different way of selecting context. The saved outcome was five
`PARTIAL_OVERLAP` relationships and three `ADJACENT` relationships, with no
`DIRECT_OVERLAP` relationship.

## Why This Is Difficult

Several nearby findings could be mistaken for the whole claim: irrelevant
documents can confuse models; retrieval can fail to include the answer; or a
method can outperform another RAG system. None alone establishes a comparison
with generation without retrieval.

Even a paper that observes the requested comparison may establish it only for
a particular domain, model, and failure mechanism. The task is therefore to
recognize relevant evidence without silently broadening its scope. Finding a
counterexample in medical vision-language generation should not turn into a
claim about every text-only RAG application.

## What ResearchPilot Did

The decomposition retained one core comparative assertion and made its concepts
explicit: retrieval quality, RAG, factuality, and generation without retrieval.
It did not replace the user's baseline with an easier one.

Across two rounds, six queries produced 42 candidate version groups. Eight
papers were selected and evaluated: five with PDF text and three with abstracts.
The pipeline used hybrid passage retrieval and two local rescue passes, with no
page-vision calls in this run. Claim-to-paper assessments compared setting,
assumptions, mechanism, method, quantity, conclusion, and scope against the
offered evidence.

The final coverage was moderate and the normal search budget was exhausted.
These are diagnostics about the bounded search, not a probability that all
relevant prior work was found.

## Selected Findings

The clearest scoped comparison came from
[*RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models*](https://doi.org/10.18653/v1/2024.emnlp-main.62).
Its cited physical PDF pages 1–2 describe cases where a medical vision-language
model originally answers correctly on its own, but incorporating retrieved
contexts leads to an incorrect answer through over-reliance on that context.
The same passages identify limited context coverage and excessive, irrelevant,
or inaccurate references as problems.

ResearchPilot recorded **`PARTIAL_OVERLAP`**, retaining the matching mechanism
and comparative outcome while preserving three differences:

1. The evidence concerned medical vision-language models and medical question
   answering/report generation, rather than RAG systems generally.
2. The described harm involved particular retrieved-context conditions and model
   over-reliance; it did not establish every interpretation of “poor quality.”
3. The paper studied specific calibration and preference-tuning methods. That
   setting was not interchangeable with an unspecified general RAG system.

The cited page 4 provides another useful boundary. Its over-reliance ratios count
errors attributed to over-reliance **among incorrect retrieval-augmented
responses**. ResearchPilot did not treat those ratios as the percentage-point
factuality loss relative to a no-retrieval baseline. The denominator and the
comparison answer different questions.

Other partial matches included
[*Benchmarking Large Language Models in Retrieval-Augmented Generation*](https://doi.org/10.1609/aaai.v38i16.29728)
and the [RAG survey](https://doi.org/10.48550/arxiv.2312.10997). Their supplied
passages supported retrieval-related failure mechanisms while leaving parts of
the requested no-retrieval comparison unestablished.

## What ResearchPilot Did Not Claim

It did not classify a narrow-domain result as universal direct overlap. It did
not say RAG is generally less factual than non-RAG, that additional context
always hurts, or that the proposed claim had no precedent. It also did not turn
the paper's mitigation results into a new evaluation of ResearchPilot itself.

**Related evidence is not full claim equivalence. Failure to find a direct match
is not proof of novelty.** Partial overlap remains useful prior work even when
the full scope does not align.

## What This Demonstrates

Idea Check makes a comparison inspectable rather than reducing it to a title
match. Here, a relevant paper stayed prominent while its medical scope, causal
mechanism, and quantitative interpretation remained attached to the result.
That helps a researcher refine a proposed claim and identify which comparison
still needs evidence.

## Limitations

This summary reflects one saved run and its retrieved passages. The categories
are model-assisted assessments subject to review, not a theorem about semantic
equivalence. Three selected sources lacked full text, the search was bounded,
and no new experiment or medical validation was performed for this case study.
The medical result is discussed as research evidence, not clinical guidance.

[All case studies](README.md) · [Architecture](../architecture.md)
