# Native31 step12→13: chronological replay completed

The [foreground guardian](root_execution.json) closed with exit 0 in 2.76 s
(uncontrolled wall time). The [actual replay](actual/result.json) passed all
200 declared numerical controls. All frozen source and metadata bindings
were unchanged. No Hamiltonian, CI solver, physical gradient, Dirac solve,
or optimizer was called.

The consumer decoded 94 declared fields from 14 original/derived arrays,
7,544,472 bytes in total, and performed five small polar Gram eigensolves.
Starting with empty history at point0, the original f043 source reproduced
all 13 recorded direction slopes and the literal accepted13 polar endpoint
with **zero coordinate error**. It transported 65 old secant pairs, admitted
12 new pairs, and visited 150 pairs in the two-loop recursions. The history
contained ten pairs before the direction that produced accepted13.

| Diagnostic at point12 | Observed value |
| --- | ---: |
| Original preconditioned steepest direction norm | 0.008436936470758398 |
| Actual L-BFGS direction norm | 0.5423932804566725 |
| Norm ratio | 64.28794175902057 |
| Steepest directional slope | −0.0009986367642709692 |
| Actual directional slope | −0.00949997208089642 |
| Direction cosine with steepest | 0.1000788480935497 |
| Two new5f columns' share of squared direction norm | 0.7347834916082527 |

The complete signed 31-orbital slope partition and 10×10 secant cross matrix
are retained in the result. The two new5f signed slope contributions are
−0.0076381277091318546 and −0.004207856009816158; their sum is not the total
slope because other orbitals contribute with both signs. No absolute-value
replacement or orbital pruning was used.

This excludes missing optimizer history or a different replayed endpoint
as the explanation of the step13 growth. It **does not** certify a Hessian
condition number, prove that the step was an overshoot, verify the physical
gradient at accepted13, or establish stationarity. A fixed, independently
qualified two-scale physical derivative check at accepted13 remains the next
discriminator. The original 16-step run and all failed stationarity/experimental
gates remain unchanged; there is no automatic optimization continuation.

The archived source-only handoff is retained unchanged. This result and its
root guardian are subsequent execution evidence, not rewritten preparation.
