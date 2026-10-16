# Native mixed-CI finite energy diagnostic candidate

Status: implemented source and executable finite qualification prepared; no H,
CI, radial gradient or optimizer has been executed in this increment. Root
source-edit authorization002 binds the frozen preceding proposal. Numerical
qualification requires a separate root execution contract and actual tests.

The new opt-in `CSFOrbitalFunctional.mixed_ci_energy_difference` consumes two
complete bound native DC CI results, the same float64 B-spline model, explicit
radial increments and frozen isolated physical targets. It validates complete
column Euclidean eigenpair residuals (1e−8), CI and radial Gram (1e−10), ordered
signed angular-table identity, operator coverage and objective consistency
before physical radial work. Existing fixed-density/constrained interfaces,
CI defaults, original signed-gradient machinery and optimizer are unchanged.
Concurrent eigenpair mutation is unsupported; final owner checks cover physical
inputs, and the entry residual/Gram controls cover the supplied eigenpairs.

The pure numerical owner separates reported eigenvalue and retained-vector
Rayleigh domains, lifts before centering and records the whole-column phase:
`x0=σc0`, `d=c1−x0`, `r0bar=σr0`. The reported response contains all four terms
`[dᵀ(H0−λ0I)d + 2dᵀr0bar + x0ᵀr0bar − c1ᵀr1]/n1`.
The Rayleigh response contains the corresponding three q0-centered terms.
Both consume the full signed trial density divided by its finite n1. Lost
precision in reported eigenvalues is not reconstructed.

The physical owner uses the original native bank/banded P/Q Coulomb loop,
extracted without changing its operation order. Ordered one-body elements
retain the public ket norm. Complete quadratic and pair-product changes
separate the declared `reference+explicit` domain from the actual rounded
trial; the full reconstruction contribution is reported as energy, alongside
its waveform defect. These are diagnostic scalars without a general rigorous
error interval or an optimizer decision adapter.

The sealed finite qualification uses the original nine Helium configurations,
ns35/ks9/Z2 point nucleus and the same complete J blocks. Two prebound PCG64
raw coordinate assets generate original nuclear positive-space selectors and
12 polar-retracted signed endpoints. Complete P/x/eta/waves inputs are written
and hashed before the first H/CI/gradient. Frozen354 ordinary comparators run
in a fresh process. Decimal80 independently interprets every captured binary64
one-body/metric/cell-bank and signed angular-table asset, with complete CSF
projection, permutation and symmetrization. All four P/Q components and
negative off-diagonal terms remain. Decimal arithmetic operates under precision
80 at every public entry, including creation of explicit endpoint domains.

CPU1, 3.95 GiB and 300 seconds bound the entire panel. Its maximum native work
is 18 full H builds, 13 complete CI solves and three full signed gradients,
with no optimizer and no hidden warmup. Candidate setup/consumption has at
most 48 one-body assemblies, three nuclear Dirac solves, 36 polar retractions,
41 physical overlaps, 80 tracking SVDs and 64 total SciPy eigensolves; the old
source process separately allows six one-body assemblies, two overlaps and
two SVDs. The SciPy limit covers three Dirac and all finite CI block solves,
including any actual native retry/refinement rather than hiding it. No
additional nuclear projector or physical model is constructed by Decimal.
The complete Decimal matrix interpretation has 25 arithmetic matrix domains
(baseline and 12 ideal/actual pairs), counted separately from native H builds.
Forward-only counters are written before and after work. Any failed cap,
identity or numerical gate is preserved; the model and gates are not reduced.

Prepared controls cover finite nonunit norms/nonzero residuals, phases,
zero/tiny/signed/reversed/common-offset identities, independent complete
radial and finite response contractions, native complete eigenpair/Gram
controls, old354 bitwise H and gradient parity, and zero-call unsupported
scope exits. Original Yb stationarity/indicator gates remain unchanged. This
candidate does not assert Yb convergence, an improved experimental gap,
Armijo adoption or performance/GPU gains.

Files: [contract](qualification_contract.json), [driver](run.py),
[test panel](../../../tests/test_mixed_ci_finite_increment.py),
[Decimal oracle](decimal_oracle.py), [frozen comparator](frozen_comparator.py),
[raw inputs](raw_directions_identity.json), [candidate identity](candidate_source_identity.json),
[readonly preparation proof](preparation_verification.json),
[original frozen proposal](../mixed_ci_finite_increment_proposal/README.md).

Run only after the root seal, with single-CPU affinity and all numerical thread
environment variables set to one. `run.py` refuses absent/mismatched root
execution authorization. It captures partial counters, exception/timeout and
pytest exit status as actual evidence. Preparation proof and Ruff checks are
metadata/static validation, not numerical qualification.
