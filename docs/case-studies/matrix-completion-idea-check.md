# Matrix completion: testing each link in a proposed mechanism

## Claim

> Under a fixed observation pattern, optimizing edge weights to reduce the
> spectral ratio rho = sigma_2 / sigma_1 improves restricted curvature and can
> sharpen guarantees for nonconvex matrix completion.

This case uses a completed Idea Check with a clarifying scope: the proposed
method reweights an existing observation support rather than changing that
support. The intended chain runs through a weighted sampling-deviation
certificate, with incoherence and row-norm assumptions retained. It does not
assert that a lower spectral ratio universally improves empirical recovery.

## Why This Is Difficult

The vocabulary overlaps with several substantial literatures, but the links
between their results matter as much as the terms they share.

| Distinction | Why it matters to the comparison |
| --- | --- |
| Fixed observation support vs. random sampling | A guarantee over a sampling distribution does not automatically apply to a particular fixed pattern. |
| Edge-weight optimization vs. weighted loss or estimator | Using weights is not evidence that those weights are optimized to change the observation graph's spectrum. |
| `sigma_2 / sigma_1` vs. generic spectral quantities | A graph parameter or eigenvalue bound cannot be substituted without checking its definition and normalization. |
| Sampling deviation vs. restricted curvature | An operator-deviation estimate needs an explicit argument and assumptions before it becomes a curvature bound. |
| Restricted curvature vs. nonconvex recovery guarantees | Curvature is an ingredient; an algorithmic guarantee needs the relevant objective, region, and other hypotheses. |

A paper addressing one link could be important prior work without establishing
the complete proposed mechanism.

## What ResearchPilot Did

The decomposition separated three core assertions: fixed-support weight design;
the proposed spectral-ratio-to-deviation connection; and the
deviation-to-curvature-to-nonconvex-guarantee chain. A fourth, supporting assertion
retained the limitation about empirical recovery.

The normal workflow ran two search rounds, selected sixteen papers, retrieved
evidence, and assessed claim-paper relationships. Reused project sources were
offered for current-claim assessment; earlier support was not copied into the
new judgment. The run used local evidence recovery, cross-page context, and two
targeted vision calls for mathematical evidence.

The final 27 relationships were **9 `PARTIAL_OVERLAP`, 2 `BRIDGING`, 2
`METHOD_SIMILAR`, and 14 `ADJACENT`**. There were **0 `DIRECT_OVERLAP`**
relationships. A paper can have different relationships to different assertions;
these are relationship counts, not counts of unique papers.

## Selected Findings

**Weighting is not automatically weight design.**
[*Restricted strong convexity and weighted matrix completion: Optimal bounds
with noise*](https://doi.org/10.48550/arxiv.1009.2118) was `METHOD_SIMILAR` for
the weight-design assertion. The assessed passages treated row/column sampling
weights and a weighted estimator, rather than optimizing fixed-support edge
weights to reduce `rho`. For other assertions it supplied a `BRIDGING` curvature
ingredient and `PARTIAL_OVERLAP` with the curvature-to-error-bound connection.
The comparison preserved its sampling model and convex estimator.

**A deterministic pattern does not establish the proposed spectral mechanism.**
[*Matrix Completion from General Deterministic Sampling Patterns*](https://doi.org/10.48550/arxiv.2306.02283)
provided partial overlap with the fixed-support setting (cited physical pages
1–2 and 7–8). Its cited page 13 contained a tangent-space sampling-deviation
lemma. The assessment did not identify that lemma with a weighted certificate
controlled by `sigma_2 / sigma_1`: the needed weights and connection were not
established in the supplied evidence.

**Related nonconvex analysis retained its sampling assumptions.**
[*Convergence Analysis for Rectangular Matrix Completion Using Burer-Monteiro
Factorization and Gradient Descent*](https://doi.org/10.48550/arxiv.1605.07051)
partially overlapped with the third assertion. Its offered evidence linked
sampling control, incoherence-related row-norm conditions, and local curvature
for a factorized method. Random/Bernoulli sampling and an unweighted analysis
remained explicit differences from fixed-support weight design.

**A curvature theorem is not the missing derivation.**
[*Global Optimality in Low-Rank Matrix Optimization*](https://doi.org/10.1109/tsp.2018.2835403)
connected restricted strong convexity/smoothness to properties of a nonconvex
factored objective. ResearchPilot retained it as partial overlap, without saying
that the paper derived those properties from the proposed weighted certificate.

## What ResearchPilot Did Not Claim

Zero direct matches did not become a novelty certificate. The final boundary
retained missing evidence for the complete core propositions and no established
full combination. Moderate coverage of the first two core assertions and strong
coverage of the third described the available comparisons; **strong coverage did
not mean that the third assertion had been proved**.

It also did not substitute a spectral statistic for restricted curvature, turn
a convex guarantee into a nonconvex guarantee, or infer that empirical recovery
must improve whenever `rho` decreases.

## What This Demonstrates

The result is a map of usable components and missing implications. One paper can
be methodologically related, another can supply a bridge, and another can overlap
only under different assumptions. These distinctions give a researcher concrete
places to inspect instead of an unsupported claim of equivalence or originality.

## Limitations

The search was bounded and the mathematical assessments still require expert
review. PDF extraction and vision can misread notation; an evidence chain can
remain incomplete. This case is not a proof or an independent correctness audit
of the proposed result.

The current implementation includes bounded supplementary source recovery for
**weak core** assertions. It did **not trigger in this run**: all core assertions
were moderate or strong, while the weak assertion was supporting. The observed
sixteen papers and two vision calls belong to the normal workflow, not recovery
increments. The recovery branch has offline tests; this run does not demonstrate
its live execution.

**Failure to find a close match is not proof that no prior work exists.**

[All case studies](README.md) · [Architecture](../architecture.md)
