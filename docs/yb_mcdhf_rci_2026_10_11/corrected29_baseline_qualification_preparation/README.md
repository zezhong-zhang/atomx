# Corrected-source native Yb⁺ baseline and constrained derivative qualification

This is a prospective consumer, not an executed result. It uses the private
`f0435dd6…` source with only the two nonnegative-rank inner Gauss orders changed.
The closed [scalar and fixed-CI discriminator](../rank_aware_cell_discriminator_preparation/actual/result.json)
passed its declared anchors. Its largest fixed-CI transition correction was
about `7.68e-12 cm⁻¹`; that measurement does not provide a global radial error
bound or a new complete-CI spectrum.

The declared run is exactly one native complete baseline plus four fixed finite
points: at most **5 H builds, 5 CI solves, 3 public weighted gradients, no Dirac
solve and no optimizer**. The baseline uses the production public gradient, then
the full normalized signed angular table and its difference from the retained
table. All exact nonzero terms remain. The four CI-only points use the original
saved unit directions (PCG64 seeds 20261011 and 20261012) and `h=1e-6`. No direction
or step scan is allowed.

All points retain the original 69 electrons, Fermi A175 nucleus, 29 radial keys,
1046 complete CSFs in five target sectors, original five physical UIDs, equal
weights, seven nuclear no-pair projectors, authoritative 3451 coordinates and
inverse metric. The old source's H, roots, signed eigenvectors and gradients are
archived comparisons; corrected numerical values have no old-source equality
requirement. Geometry, point, basis, metric and state identities retain exact
checks. Original lower-state energies are selected by archived label and UID,
then their original archived column, never by a fresh column or insertion order.

`run.py` reuses public CI, tracking, angular contractions, gradients, polar
retraction and the complete mixed-CI finite diagnostic. `constraint_bookkeeping.py`
only performs exact stored-array algebra: a fixed baseline
`Λ=(CᵀgA+gAᵀC)/4`, complete signed same-κ Gram changes and their quadratic and
rounded-reconstruction identities. Here `gA` contains the full normalized public
HF partial and the complete known raw-h/ket-norm ambient correction. It keeps
finite CI phase, norm, residual and response terms in the public finite diagnostic.
Each centered central derivative is checked against its corresponding ideal
constrained action with the unchanged `max(1e-8,1e-5|action|)` threshold.

These are Q-Lagrange diagnostic gates. The production objective remains ordinary
reported EOL; no Rayleigh replacement, energy callback, uniform error bound or
convergence claim is introduced. The original reported and uncentered derivative
failures remain failures. Direct stored-H quotient differences and total-energy
ULP resolution are separate forensic records. Original stationarity thresholds
`1e-7` and `1e-8 Ha` are reported on the new gradient; derivative qualification
cannot substitute for stationarity or experimental agreement.

Resources are CPU31, all numerical threads one, 3.95 GiB address space, 300 s
inner and 315 s outer wall caps. Up to 70 nuclear h/S calls are explicit (7 initial,
35 internal CI, 28 finite-radial), with 25 complete-sector CI eigensolves and
28 polar calls. Capturing the actual public tracker output avoids redundant
resolves: 13 resolves, 65 SVDs and 14 overlap calls including initialization.
Timing is uncontrolled. A new empty NumBa cache must belong to this directory.
Every phase persists attempted/completed counts and gates. Failure stops before
additional physical work and preserves complete available arrays.

[Preparation verification](preparation_verification.json) contains 16 source,
JSON, AST, UID permutation and refusal controls; it imported no numerical module
and executed no science. Ruff passed. An initial preparation-only refusal check
encountered a missing proof file before the authority test; the
[record](preparation_attempt001.json) preserves this and the correction.
A separate exact root seal is still required. No execution is authorized by this
preparation or by the fixed-CI discriminator.
