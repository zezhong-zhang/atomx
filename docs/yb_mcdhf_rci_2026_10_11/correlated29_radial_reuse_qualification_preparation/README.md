# Complete29 private semiseparable-H qualification preparation

Prepared, not executed. The private 356-file candidate changes only
`matrix.py` imports and `_bspline_pair_potential`: at the existing `ns>=96`
boundary use the existing exact semiseparable convolution, retaining all
PP/QQ/PQ/QP near and distant cells. Shared source, the 500ac archive, physical
operators, RDM cutoff, states, weights and metrics remain unchanged.

The launcher first calibrates five workspaces: ns95/96, ordinary/mixed RKB,
and an origin single-cell analytic signed quartet; ranks0/1/6 and all four
components are checked against independent full-cell banks and existing
analytic formulas/tolerances. Five small NumPy `leggauss` companion eigvalsh
calls are quadrature setup and explicitly counted separately from zero new
Dirac spectra. There are 180 direct contractions, 150 scalar band contractions,
20 density bands, 15 actual candidate helper routes and five basis setups.
No full physics builder is invoked by these anchors.

Then two separate fresh processes import the ordinary 500ac source and private
candidate, respectively. Each performs exactly one genuine public retained-H
energy trial, complete eigenpair/Gram/owner checks, same-trial full gradient,
original-coordinate/preconditioner replay and checked compressed archive.
Together: 2H/2CI/2g, 28nativehS, 4tracking resolves, 20SVD, 6overlaps, 10CI block
eigh, 0newDirac spectra/polar retractions/optimizer. All1046 CSFs in five
complete target sectors, all29 orbitals and the original3451 metric are used.
Raw h, lower-triangle-completed L(h), S and source identities are distinct and
strictly matched. The original raw/tangent g1e−9 and pg1e−10 gates are retained.
The ordinary H/roots must reproduce saved bytes. Candidate representation
parity uses prospectively reviewed Hentry1e−9Ha, allroots1e−8Ha and signed
cluster-projector1e−8 gates with a1e−7Ha cluster gap. These are numerical
algorithm comparisons, not experimental accuracy or stationarity certificates.

Full CI support, signed RDM support/coefficients, complete source keys and
physical five-state UIDs/equal weights are checked. The only sector reduction
in the saved cluster comparison removes entries independently shown to be
exact zeros; full eigenpair checks still cover every entry. No retained-owner
restore/forgery, arbitrary-H substitution, small-term screening or hidden
physical repeat is permitted.

Every worker has a separate NumPy/Numba process/cache. Forward-only counters
and cProfile record actual bank/factor getters, kernel, H/CI/g and tracking
costs; stage timing includes setup, owner/tracking, eigenpair checks, gradient,
compressed NPZ/fsync/readback. Whole subprocess time also includes numerical
imports and immutable input/source admission. These are uncontrolled cold
stage observations, with no ratio or speedup certification. Root reported a
4090 GPU as available hardware; this runner performs zero GPU work. GPU,
parallel and full-optimizer performance are unmeasured.

The original reported/unconstrained-Rayleigh derivative failures and original
stationarity/indicator failures remain archived. Both saved conditional
fixed-Lambda centering controls are bound, with their explicit scope and
unresolved normalization/error-bound limitations. The separate native public
normalized full-gradient control is not fabricated as a prerequisite pass.
Equivalence is useful even when the scientific solution remains unqualified.
Any production integration or optimizer change needs a successor review.

Root must seal exact driver/contract/preparation and source maps, then reserve
CPU31 after the native full-gradient control has closed. The driver refuses
without that seal. One sequential launch only; first failure/time/call/memory
limit preserves logs/partials and stops, without retry or model/gate changes.
Total inner cap300s, outer315s, CPU31, one BLAS/Numba thread,3.95GiB:

```sh
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 timeout --signal=TERM --kill-after=5s 315s /data/code/atomx/.venv/bin/python docs/yb_mcdhf_rci_2026_10_11/correlated29_radial_reuse_qualification_preparation/run.py --execution docs/yb_mcdhf_rci_2026_10_11/correlated29_radial_reuse_qualification_preparation/execution_contract.json
```

The absolute venv interpreter must not be resolved through its symlink.
`preparation_verification.json` is stdlib source/input/AST admission only;
Ruff and syntax checks have no numerical qualification meaning. Numerical
results belong in `actual/` only after root execution, never in preparation.
