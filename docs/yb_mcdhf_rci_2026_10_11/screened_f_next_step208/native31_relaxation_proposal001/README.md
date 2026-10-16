# Native31 source qualification before orbital relaxation

The frozen f043 calculation has passed its single combined 4f–5f direction at
both predetermined step sizes: two ordinary reported-energy gates and four
independent centered diagnostic gates. Its closure records 730 live controls
and 81 metadata controls. It remains nonstationary. These measurements do not
qualify the changed current source or the whole orbital space.

The next experiment must first qualify the current source at the literal
recovered31 point, with its original seven nuclear projectors, 3689 coordinates,
inverse blocks, complete 1595-CSF space, original five tracked UIDs and equal
weights. [The qualification contract](source_epoch_qualification_contract.json)
keeps a strict whole-H bitwise comparison, all-root residual/Gram and complete
signed subspace comparisons, and whole raw-g/pg gates of 1e−9/1e−10. It does
not require bitwise CI coefficients from a changed eigensolver.

Current `ci.py` removes a scalar diagonal origin before diagonalization.
Current `matrix.py` also selects the semiseparable pair-potential route at
ns=123. The latter can change matrix rounding independently of the solver;
the strict whole-H gate may reject it. The parent has reported two unresolved
current regression failures. They remain prerequisites, not evidence that
this proposal has passed. A failed matrix comparison requires an explicitly
reviewed successor, without weakening this contract during execution.

Five fresh current-source H/CI points are proposed, rather than borrowing the
old finite-difference result. H, gradient and signed-subspace agreement alone
does not determine the rounding of reported roots and their averaged finite
slope. The baseline also produces the public retained gradient and two
complete normalized, unfiltered signed-weight gradients. The four fixed
trials preserve the old direction and both h values. Ordinary reported EOL
and mixed radial/CI-response/Lagrange-centered diagnostics remain separate.
No quotient replaces the production energy and no energy callback is added.

Only after that independently closed qualification may the separately sealed
[bounded relaxation](bounded_relaxation_contract.json) make up to 16 public
optimizer steps. It uses default history_size=10, empty resumed history, the
same metric and stopping gates, and checks full normalized signed gradients
at every baseline/accepted point. This costs at most 17 public gradients plus
34 extra gradients, rather than hiding the full-density control for speed.
It archives every lower/upper change, four transitions and both multiplets.
NIST is used only for reporting. No terminal state is chosen by its gap.

The old exact acceptance helper has a 1046-dimension guard. The private
[1595 successor](exact_reported_guard1595.py) changes only that dimension
limit and the cumulative examined-pair resource limit. Its exact arithmetic
and inequality are unchanged, as shown in the [source review](guard_dimension_review.json).
Complete signed columns must have exact zero support outside their physical
symmetry block; no coefficient may be dropped. Their worst-case support is
271,965 examined pairs per point, or 4,623,405 over 17 points, within the new
5,000,000 cap. There are no budget resets. This proves only a finite stored-H
acceptance inequality, not a global radial error or stationary solution.

This directory contains source and JSON preparation only. No executor is
implemented or authorized, no numerical module was imported, and no NPZ was
read or rehashed. Root must first freeze the complete current source and
resolve its regression prerequisites. Existing preparation, failures,
scientific outputs and shared source were preserved.
