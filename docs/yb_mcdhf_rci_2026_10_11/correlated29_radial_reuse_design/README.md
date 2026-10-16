# Complete29 two-body radial reuse: source-only proposal

The smallest useful candidate is to route the **CI B-spline pair-potential
producer** through the existing exact semiseparable direct kernel at the
existing `SEMISEPARABLE_DIRECT_NS=96` boundary. It requires two imports and a
branch in `matrix._bspline_pair_potential`; [the proposed diff](proposed_matrix.patch)
is preparation only. Shared source and tests have not been edited, and no new
H, CI, gradient, radial integral, eigensolve or optimizer has been executed.

This is a numerical kernel reuse candidate, not a remedy certified for the
current complete29 derivative failure. The original reported derivative and
the later on-manifold Rayleigh rotation derivative failures remain unchanged;
actual optimization remains on hold until its physical gradient control passes.

## What is already reused

`matrix._radial_evaluator` already has bounded exact-key caches for radial
values (8192), pair densities (512), and B-spline source-pair potentials (1024).
It retains the complete original source/target role order. `CIRadialSnapshot`
already freezes orbitals/workspace/nucleus and exposes those same caches across
declared CI builds. It does not currently expose its pair potentials as a
gradient consumer API.

`CSFOrbitalFunctional.gradient_from_weights` already groups each signed
two-body term into both derivative source pairs and reuses each distinct
potential action on a target spinor. Its one-body actions also reuse identical
targets. Adding another pair dictionary or claiming this reuse as new would
repeat existing work. `ConvolCache` already shares exact Coulomb banks and
factors across equal-content geometries.

The remaining producer mismatch is concrete: CI `_bspline_pair_potential`
always contracts four dense transposed cell banks through `_convol_case1_t`;
the current gradient and scalar Slater path already use
`_convol_case1_semiseparable_t` at `ns>=96`. The candidate uses that existing
prefix/suffix and exact-overlap-strip representation. It adds no physical
cutoff, low-rank approximation, coefficient screen, new quadrature, alternative
backend or GPU execution. Every source pair, including signed off-diagonal
densities, remains present.

The gradient materializes two `ns×ns` potentials and streams their actions;
changing that representation or sharing a potential across H and gradient is
deferred. Those paths currently use different reduction orders, and a typed,
same-point radial owner would need a larger reviewed contract. In particular,
the raw one-body CI operator and lower-triangle-completed functional operator
must not be equated or altered during this performance increment.

## Evidence and limits

The old [finite He profile](../performance/gradient_probe001/profile.txt),
12 CSFs and `ns=120`, measured approximately 0.020 s CI/H construction and
0.014 s gradient; dense CI pair potentials consumed 0.017 s, versus 0.010 s
for gradient pair potentials. It recorded 31 source pairs. Its host load guard
failed, and it supplies no controlled speedup. These are historical He
observations, not a complete29 profile.

The original complete29 [actual counter ledger](../correlated29_mcdhf_probe_preparation/actual/progress.json)
records 4 H / 4 CI / 1 full gradient, 28 native one-body h/S calls, and no
optimizer. The subsequent rotation ledger records 3 H / 3 CI / 0 gradients.
Neither separates radial, angular, CI eigensolver and tracking times. Thus
**the dominant complete29 stage and its prospective speedup are unknown**.
The metadata-only [source review](review.json) counts source pairs and actions
from already saved signed density keys; these are work-structure counts, not
new radial evaluations or actual CI cache miss counts.

Both the stored unfiltered density (12403 terms / 6311 negative) and the
currently consumed density (10738 terms / 5488 negative) name **593 gradient
source pairs**, implying 2372 component convolutions per complete gradient.
The existing action reuse implies 8724 P/Q actions for the consumed density.
These metadata counts suggest a materially larger kernel workload than the
old 31-pair He model, but they do not measure time. The existing public density
filter at 1e−15 is recorded unchanged; the candidate adds no further screen
and must not use dropped terms as its performance gain.

The separately certified [retained-H chain](../retained_hamiltonian_performance/RESULTS.md)
saved 19.52% of median checked-chain time on a different complete531/24-orbital
point by removing a repeated H build. It does not establish a kernel speedup
or complete29 timing. Retained-H consumption remains available for future
symmetrically checked experiments; it must not be counted as this candidate's
gain.

## Reusable independent anchors

| Existing anchor | What it establishes | Remaining boundary |
|:---|:---|:---|
| `test_convol.py::test_overlap_strip_matches_independent_full_bank` | Each PP/QQ/PQ/QP overlap-strip cell against the separately assembled complete bank, ranks 0/1/6; existing rtol 2e−12 / atol 1e−14 | Small RKB geometry, not actual complete29 H |
| `test_convol.py::test_factored_direct_matches_full_cell_bank` | Four component contractions, including unequal P/Q orders, signed random density elements; existing rtol 2e−12 / atol 1e−12 | Reduction-order equality is numerical, not bitwise |
| `test_cell_coulomb_features.py::test_signed_quartets_match_full_native_cell_integral` | Four independent signed orbitals, explicitly isolated PP/QQ/PQ/QP contributions | Separate representation; do not substitute it into native MCDHF |
| `test_cell_coulomb_features.py::test_origin_cell_monomial_double_integral` | Analytic same-cell Coulomb double integral | PP analytic anchor; it alone does not validate all four components |
| `test_slater_kernel_crosscheck.py` | Hydrogenic 5/8, 3/8 and independently integrated R² scalar references | Q=0; must not be presented as a QQ/PQ/QP anchor |
| Existing Decimal80 finite-cell oracle | Full signed four-component contractions of captured binary64 banks, with every off-diagonal term | Prior assets have ns=35; no repeat of its 13-point physics panel is proposed |
| `test_mcdhf_gradient_reuse.py` | Complete/partial/signed transition action identities and independent positive-space tangent derivatives at ns35/120 | Does not clear the measured complete29 derivative failure |

The anchor sources and their exact bytes are bound in [contract.json](contract.json).
This source-only audit has not run them and makes no fresh test-passed claim.

## Minimal follow-up, contingent on root review

First measure one ordinary retained-H energy-trial → same-result signed
gradient chain at the **exact saved complete29 recovered point**, with complete
existing CSFs, original five target UIDs/equal weights, Fermi A175 DC operator,
all29 radial inputs, unchanged 3451-coordinate metric/projectors/inverse and
unchanged full eigenspace. Forward-only profiling must count complete H/CI,
radial key misses, pair cache misses/evictions, dense/factored calls, density
calls, one-body builds, gradient sources/actions, tracking/overlap/SVD and
archive work. It may share an already authorized physics-control call instead
of adding a duplicate profile call. If radial construction is not material,
stop this candidate rather than implementing an unmeasured cache layer.

After a source-edit seal, the numerical qualification proposal is one ordinary
and one candidate checked chain in isolated processes: **2 H / 2 CI / 2 full
gradients total**, no optimizer and no new projector construction, with all
source/geometry/model setup costs included separately. Each chain uses one
public genuine H producer; saved arbitrary matrices are comparison evidence,
not accepted owners. The proposal caps 28 h/S calls, 2 tracking constructors,
4 resolve calls / 20 tracking SVDs / 6 overlap calls, 10 complete CI sector
eigensolves, and zero polar/eigvalsh/projector eigensolves, CPU1, 3.95GiB,
300 s. A finalized driver must bind the exact saved point, count aliases and
setup, and obtain a separate root execution seal; none exists yet.

The original full eigenpair Euclidean-column residual 1e−8, CI/radial Gram
1e−10, original projected-g 1e−9 and pg 1e−10 equality gates remain. Complete
one-body raw/L bytes, basis order, keys, target mapping, radial coordinates,
projector/inverse bytes and no-pair gates are unchanged. Ordinary new-source
replay must reproduce its old-source baseline bitwise. For the candidate,
semiseparable reassociation is **not** claimed bitwise: complete-H/root/span
comparison and four-component numerical gates must be declared separately in
the reviewed new-epoch qualification contract, before execution. Failure at
any original derivative/physical gate remains a failure; performance parity
cannot reclassify it.

Only after correctness and a complete29 radial profile support the candidate
should a controlled alternating five-pair warm chain experiment be prepared.
Two cold observations and two explicit warmups would be accounted separately;
both sides must include equal owner binding, full eigenpair checks, gradient,
tracking and checked archive costs. No speed ratio is proposed or certified
here. No cutoff or small signed term may be dropped to obtain a result.

## Source and documentation disposition

Current source is the immutable 356-file map
`500ac4c3c372e0eb51e6d9910ef3830bf8eb323b211b52ba3ec08ccbf7b1b794`,
whose only change from the prior physical-source epoch is accepted-line-search
metadata. [review.json](review.json) checks the current files and the existing
complete archive against that map using stdlib hashing and AST inspection,
with zero Atomx/NumPy imports and zero array loading. It preserves all original
failures and validates the proposed diff without applying it.

Only this preparation directory is new. Shared scientific source/tests,
historical evidence, progress cards, website and registries remain unchanged.
The old action-reuse README's next-kernel candidate remains applicable; current
architecture guides correctly describe existing caches and need no capability
update. [documentation_impact.json](documentation_impact.json) records this
disposition. No new general insight is promoted before qualification/timing.
