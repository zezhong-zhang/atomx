# Frequency-dependent transverse exchange

## Operator and frequency prescription

The explicit frozen-orbital CI path now supports the real Coulomb-gauge
one-photon transverse interaction. In atomic units, with `k = alpha * omega`,

```
T(k) = -alpha1.alpha2 cos(k R)/R
       + (alpha1.grad1)(alpha2.grad2) [cos(k R)-1]/(k² R).
```

Here `alpha1` and `alpha2` are Dirac matrices, while `alpha` in the wave
number is the fine-structure constant. The zero-frequency limit is full
static Breit, including its retardation term.

For a two-electron matrix element `<ab|T|cd>`, the implementation averages
`T(abs(epsilon_a-epsilon_c))` and `T(abs(epsilon_b-epsilon_d))`. It averages
the **operators**, not the two frequencies. This is the prescription in
the supplied FAC `faclib/radial.c::BreitRK/BreitSK`; `BreitWW` contracts
the corresponding radial integrals. It preserves Hermiticity and electron
exchange symmetry with a fixed, real set of orbital energies.

These frequencies describe virtual photon exchange. They are not the
many-electron emitted photon frequency in a radiative transition.

## Explicit CI API

```python
from atomx.methods.ci import ci_solve

ci = ci_solve(
    csfs, orbitals, orbital_basis, workspace, Z, ns,
    include_breit=True,
    breit_orbital_energies=energies_by_n_kappa,  # Hartree
)
```

The energy mapping must include every `(n, kappa)` in `orbitals`, and all
energies must be finite. Without the mapping, `include_breit=True` retains
static Breit. Providing energies while disabling Breit raises an error.
The same keyword is available on `physics.csf.matrix.build_ci_matrix`.

For an `AtomicResult` whose orbital energies use subshell labels, construct
the mapping explicitly:

```python
energies_by_n_kappa = {
    (s.n, s.kappa): result.orbital_energies[s.label]
    for s in result.subshells
}
```

The transverse matrix **replaces** static Breit. To report the incremental
shift, compare CI eigenvalues with and without the mapping, keeping the
spinors, nuclear potential, CSFs and all other settings fixed. Match states
by symmetry and mixing when their ordering changes. The result records
the frequency prescription and the numerical energies supplied to it.

## Public calculation controls

The public `Corrections` object and `CorrectionStack` now accept
`breit_frequency="static"` (default) or `"orbital_differences"`:

```python
from atomx import Atom, Corrections

atom = Atom(10, engine="shooting", corrections=Corrections(
    breit=True, qed=True, breit_frequency="orbital_differences"))
atom.solve()
```

For ordinary DHF, the finite-frequency expectation uses the converged
orbital energies and the same jj ensemble as the SCF. Both radial engines
use the same operator. No MCDHF calculation is introduced by this option.
The expectation replaces the static correction under the single `breit`
ledger entry; applying another Breit correction afterward is rejected.

CSF-based Hamiltonian DF and MCDHF require an explicit complete energy map:

```python
settings = Corrections(
    breit=True, qed=False, breit_frequency="orbital_differences",
    breit_orbital_energies=energies_by_n_kappa,  # every active orbital, Hartree
)
atom = Atom(Z, config=configurations, corrections=settings)
atom.solve()
```

Choose these energies consistently with the orbital model and record how
they were obtained. The API does not infer physical energies from arbitrary
correlation-orbital labels or optimization multipliers. Missing/nonfinite
maps are rejected before the CSF solve. The final CI Hamiltonian contains
the finite-frequency operator; no static scalar is added. The reported
Breit shift compares corrected and Dirac-Coulomb CI eigenvalues on the same
final spinors. Per-block orbital calculations retain their existing
limitation: a global applied shift is not reported.

The `DiracFock` and `MCDHF` method objects expose the same frequency and
energy-map fields. An explicit correction stack is authoritative; conflicting
public inputs raise. Static presets keep their existing physics. Provenance
records the frequency prescription, numerical energy map, application stage
and whether energies came from DHF or explicit input.

There is no finite-frequency orbital relaxation or absorptive interaction
in this implementation. An orbital-difference prescription is an effective
one-photon model; it is not exact energy-dependent bound-state QED.

## Numerical and physical boundaries

- `physics/csf/transverse.py` owns the transverse operator and frequency
  assignment; it reuses existing Gaunt and divergence angular contractions.
- `physics/csf/retardation.py` supplies transition-current divergences from
  radial spinors and derivatives. No local-potential continuity identity
  substitutes orbital energies for derivatives; nonlocal HF is supported.
- `numerics/helmholtz.py` owns the scalar real Helmholtz Green functions,
  regular scaled Bessel evaluation, and ordered radial triangle integration.
- `methods/ci.py` assembles and diagonalizes the requested Hamiltonian.

The Racah-normalized Green-function multipole is
`-(2L+1) k j_L(k r_<) y_L(k r_>)`. Its scaled factors tend to one. Subtracted
power series evaluate `(cos(k R)-1)/k²` without catastrophic cancellation
near zero. They evaluate the full function; they do not switch the physics
to a truncated static operator.

Each radial contraction has linear work/storage in the cell count for fixed
Gauss order. Kernel geometry is reused by rank and frequency, potentials
by transition density, and matrix elements by their complete radial key.
All caches belong to one evaluator and therefore one frozen orbital set.
B-splines use analytic spinor derivatives. Shooting uses cubic interpolants
and eight-point Gauss integration in each native interval.

## Independent evidence and limits

The live FAC comparison uses point-Coulomb spinors at Z=1,10,26,54,82,
infinite mass, and SE/VP disabled. Six closed-subshell pair exchange energies
and six off-diagonal matrix elements are compared in both static and finite
frequency modes: 120 values. The off-diagonal cases include unequal vertex
frequencies and reversed bra/ket labels.

FAC uses its unmodified `libfac.a`; an isolated diagnostic frontend exposes
its integrals. Each frequency mode runs in a separate process because FAC's
Breit cache does not include the mode. No reference source or library was
changed. Inputs, outputs, source/executable hashes and the diagnostic
frontend are retained with the benchmark artifacts.

The initial panel agreed within 24 ppm. Its point-count-only FAC refinement
did not refine the inner grid spacing and was inconclusive. The follow-up
below separates orbital errors, operator errors and reference approximations.
The point-Coulomb Pb 1s–2p3/2 closed-subshell pair has a finite-minus-static
exchange shift of about -0.5596573 Ha. This is a pair contribution on
hydrogenic spinors, not a neutral-Pb level shift.

Tests also compare multipoles with direct angular projections, radial
contractions with split adaptive quadrature, and the zero-frequency CI
matrix with the existing independently checked static implementation.
Tracked FAC Ne/Pb fixtures make those external checks reproducible without
requiring the FAC executable in the test environment.

Artifacts: [full matrix-element panel](../benchmarks/runs/2026-09-29-transverse-ci-live/),
[grid refinement](../benchmarks/runs/2026-09-29-transverse-refinement/),
[kernel reuse panel](../benchmarks/runs/2026-09-29-transverse-cached-live/).
Broad correlated-state, finite-nucleus and continuum validation remains open.

### Reference refinement and common-spinor diagnosis

The follow-up exports FAC's numerical spinors and orbital energies at full
double precision through a read-only diagnostic frontend. Atomx evaluates
its own current-divergence transverse operator on those same spinors.
The two FAC frequency modes must export identical orbitals; the runner
checks every array before comparing matrix elements.

Three reference controls matter:

1. Increasing `SetRadialGrid`'s point count alone does not refine its
   logarithmic ratio near the nucleus or its asymptotic spacing. The new
   `--fac-grid-density=s` sets `ratio=1.1**(1/s)` and `asymp=50*s`.
   Increase the allocation proportionally to retain radial coverage.
2. FAC's default small-argument Breit threshold is 0.01. Below it, some
   radial factors retain only their leading terms. Lowering it to 0.001
   removes the 1.3-ppm discrepancy in one Ne off-diagonal element on common
   spinors. Atomx already evaluates the full regular series there.
3. The inner orbital energy controls are separate from SCF convergence.
   `--fac-energy-tolerance=1e-12` sets FAC's `orbital:enerelerr`,
   `orbital:enerelerr1` and `orbital:eneabserr` explicitly. This removes the
   nonmonotonic hydrogen discrepancy left by grid refinement alone.

At density factor 2 with a 0.001 Breit threshold, the full common-spinor
panel (Z=1,10,26,54,82, static and finite frequency, 120 values) agrees within
5e-8 relative, or 0.05 ppm. This isolates operator/integration agreement from
the quality of the input orbitals. It does not certify that those FAC
orbitals themselves are exact. Shared spinors also fix their phases, so the
common-spinor tests check signed off-diagonal elements, including the signs
needed for CI interference, rather than only their magnitudes.

With independent orbitals, density factor 8 and tightened reference orbital
energies give agreement within 0.3 ppm in the final panel. The approximate
factor-four improvement per density doubling is consistent with the
second-order near-origin small-component recurrence in the supplied FAC
`orbital.c::DiracSmall`. This identifies a plausible leading numerical
source; it is not a proof that every residual comes from that recurrence.

Tracked tests now impose 0.1-ppm agreement on common Ne/Pb spinors and
1-ppm agreement on independently solved Ne/Pb spinors, replacing the earlier
30-ppm external gate. The compressed common-spinor fixture includes its
source hash and reference settings. No Atomx operator was fitted to FAC.

Evidence: [common spinors](../benchmarks/runs/2026-09-29-transverse-common-panel/),
[independent precision panel](../benchmarks/runs/2026-09-29-transverse-precision-panel/).
The numerical grid and orbital controls are shared by the transition and
recoil benchmark runners in `benchmarks/fac_runner.py`.

### Public-path integration checks

The [public API audit](../benchmarks/runs/2026-09-30-transverse-public/)
checks both engines, a statistical open-shell ensemble, explicit
multi-configuration final CI, the method objects, input contradictions and
duplicate correction rejection. Seven dedicated integration tests pass.
The CI tests deliberately use small spaces and limited orbital sweeps to
check the final Hamiltonian; they do not certify variational completeness.

In the neutral-Ne point-nucleus public run (120 B-splines / 2,000 shooting
points, SCF tolerance 1e-9), all four static/finite runs converge. The
finite-minus-static corrections are -3.74858e-6 Ha (shooting) and
-3.74854e-6 Ha (B-splines). Absolute finite-frequency expectations differ
by 4.1e-9 Ha. A 1e-11 SCF request did not converge on the B-spline route
within 100 iterations; that failed target is retained separately. These
energy-stopping tests do not replace independently controlled orbital-shape
convergence, nor the external FAC operator comparison above.
The subsequent [inner-eigensolver repair](DF_SOLVER_CONTROLS.md#inner-eigenpair-accuracy-and-tight-b-spline-convergence)
resolves that B-spline failure: the repeated public Ne run now converges in
both static and finite-frequency modes at 1e-11. Its numerical controls still
do not imply identical orbital-error bounds across the two engines.

An internal cross-engine check on neutral Ne uses 120 B-splines versus
2,000 shooting points, independently converged orbital shapes, and both
point and Fermi nuclear profiles. The finite-frequency Breit expectations
agree within 2.7e-12 Ha; their finite-minus-static shift is about
-3.74854e-6 Ha. This verifies the common operator on two numerical engines,
including one finite-nucleus case; it is not independent external
finite-nucleus validation. Inputs and results are in `neon-engines.py/json`
under the kernel reuse artifact.

## Performance measurement

Profiling the 2,000-point Pb panel attributed about 58% of operator time to
repeated scaled Bessel evaluation. Caching kernel geometry by `(rank,
frequency)` reduced finite-frequency operator time by 1.49–1.66 times
across the five charges, with all 120 matrix values unchanged. For Pb,
the mean of the two repeated evaluations changed from 57.2 ms to 36.9 ms.
These timings include evaluator construction and twelve matrix-element
contractions, exclude orbital solving/imports, and use one BLAS/OMP thread.
They are an operator optimization measurement, not a whole-SCF speedup.
FAC's retained process timings also include orbital solving and output,
so they must not be divided by these operator timings as a speed ratio.
