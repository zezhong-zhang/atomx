# Dirac–Fock controls and solver routing

Repository availability: paths marked “local evidence” are outside the tracked
checkout. They retain the original evidence location and are not GitHub links.

## Argument and mechanism table

This describes the current working implementation. An SCF convergence flag
establishes stationarity in the chosen discretization; it does not establish
agreement with experiment or completeness of correlation/QED.

| Argument | Default / example | Calculation mechanism | Cost and boundary |
|---|---|---|---|
| `Z`, `charge` | Required Z, neutral by default | Nuclear charge and electron count | Physical state selection |
| `config` | Ground configuration; one string | One nonrelativistic configuration | No generated excited configurations |
| `config=[...]` or `Correlation(expansion="s"/"sd")` | Explicit request only | Expanded CSF state space, MCDHF orbital optimization | Potentially large configuration and radial spaces |
| `Correlation(averaging="hamiltonian")` | Default for single-configuration HF | Diagonalize within one configuration; use full coherent `c†Hc` orbital equations; Eq. 10 occupations | Dedicated direct DF solver; no continuum basis construction |
| `averaging="statistical"` | Explicit for HF; default for local exchange | Fixed degeneracy-weighted ensemble of integer jj distributions | Cheaper angular model; physically different state objective |
| `averaging="equal"` | Explicit | Equal weights for integer jj distributions | Fixed ensemble |
| `averaging="fractional"` | Explicit | Products of mean occupations | Approximate pair statistics; distinct from Hamiltonian averaging |
| `averaging="ci"` | Explicit configuration lists/expansions | Configuration interaction in the requested space | Rejected for a single configuration string; use `hamiltonian` there |
| `exchange="hf"` | Default | Nonlocal Coulomb exchange | Same physical contraction across radial engines |
| `exchange="gu"` / `"slater"` | Explicit | Local exchange models | Faster alternative model; incompatible with Hamiltonian averaging |
| `engine="shooting"` | Single-configuration default | Coupled inhomogeneous Dirac boundary-value solves and occupied rotations | Direct DF uses radial samples; explicit MCDHF uses a finite electron space |
| `engine="bspline"` | Explicit; implicit default for expanded configurations | Galerkin radial equations; direct DF solves constrained source equations | Same DF electron branch as shooting; explicit MCDHF retains its no-pair variational optimizer |
| `nuclear` | `"fermi"` | Finite nuclear charge potential; `"point"` is explicit | Physical input; use the same nucleus when comparing engines |
| `corrections` | Breit + QED enabled | DF orbitals optimized for Dirac–Coulomb; static Breit added to state Hamiltonian, QED evaluated afterward | These switches do not imply self-consistent Breit/QED orbital optimization |
| `Corrections(stack=...)` | Explicit | Requested recoil/QED/Breit stack | Included operators and stages recorded in results; recoil is not implied by `qed=True` |
| `Basis(r_max=...)` | Automatic radial box | Boundary truncation | Refine independently of SCF tolerance |
| `Basis(sampling=Sampling(n_grid=...))` | Direct HF shooting: 1000 points | Shooting grid refinement | No continuum cutoff for direct DF |
| `Sampling(radial_refinement=RadialRefinement(...))` | Explicit, off by default | Repeat shooting calculations until grid and automatic-box checks pass | Additional solves; records limits separately from SCF convergence |
| `Basis(ns=..., ks=..., grid_hi=..., ...)` | Automatic DBSR-style knots, order 9 | B-spline discretization | Engine-specific; does not select a physics approximation |
| `Basis(shooting_space=...)` | Explicit MCDHF only | Finite electron-space cutoff and grid | Rejected for direct DF; increasing only the radial grid cannot repair a low continuum cutoff |
| `Correlation(radial_optimization=...)` | `"mcdhf"` for expanded configurations | Variational or explicit frozen-orbital protocol | The historical option name is not evidence that a DF run invokes MCDHF |
| `solve(tol=..., max_iter=...)` | `1e-5`, 40 | SCF acceptance and iteration cap | Direct DF checks energy, undamped orbital updates, occupied stationarity and state-density consistency; initializer and DF sweeps are separate stages |
| `environment` | Isolated atom | Optional plasma screening potential | Physical input; separate from discretization |

## Implementation boundaries

- `methods/hamiltonian_df.py`: single-configuration method policy and result type.
- `methods/csf/direct_shooting.py`, `direct_bspline.py`: couple state amplitudes
  to the radial equations. Sharing CSF algebra does not invoke MCDHF.
- `physics/csf/`: coherent densities, Coulomb fields, Breit and matrix elements.
- `angular/csf_hamiltonian.py`: exact angular contractions, with common-core
  contractions reused without freezing core orbitals.
- `numerics/shooting/`, `numerics/bspline/`: radial equations and linear algebra.
- `numerics/orbital_rotations.py`: representation-independent occupied rotations.
- `numerics/shooting/adaptive.py`: radial convergence checks; the caller
  supplies complete state solves and numerical measurements.

Coherent quartic orbital energies can contain odd rotation harmonics. Their
nine-point Fourier reconstruction retains harmonics 1–4; diagonal ensembles
can retain the cheaper five-point reconstruction of harmonics 2 and 4.

## Design assessment

The DF/MCDHF dispatch and coherent operator contracts are separated. Both direct
DF engines follow the same physical stationary equations. Numerical rotations
are shared, and fixed-ensemble averaging remains an explicit physical choice.

Some API and implementation debt remains: `Correlation` still bundles averaging,
configuration expansion and a numerical preconditioner; `radial_optimization`
retains the legacy `"mcdhf"` spelling; the shared state/correction driver is large.
These do not change a direct DF call into MCDHF, but they are reasons not to
claim that every interface is fully decoupled or performance is fully optimized.

## Adaptive shooting grid and box

```python
from atomx import Atom, Basis, Sampling, RadialRefinement

atom = Atom(98, engine="shooting", exchange="hf",
    basis=Basis(sampling=Sampling(
        n_grid=1000,
        radial_refinement=RadialRefinement(
            energy_atol=1e-6, orbital_atol=1e-5,
            tail_norm=1e-10, max_points=8000, max_radius=1200.0))))
atom.solve()
print(atom.total_energy)
print(atom.result.radial_convergence)
```

Hamiltonian averaging and Breit/QED retain their usual defaults. This still
uses direct DF, with corrections evaluated again on each new orbital set.

The controller checks normalized probability in the outer 20% of the box
for every orbital, including slow tails induced by nonlocal exchange. Core
orbitals are not truncated using their own binding energies. An unresolved
tail or a bound-state box failure grows the automatic box. Calculations with
1.5 times as many points check absolute changes in total energy, every orbital
energy and all available state excitation energies. A further box enlargement,
with additional points, checks the boundary. SGRID and SLAG6 remain intact.

SCF tolerance is tightened to at most `RadialRefinement.scf_tol` (default
1e-11). The audit records every attempt. Point, radius and solve limits
produce `converged=False`; unrelated integration failures are raised. An
explicit `Basis(r_max=...)` or nondefault sampling box stays fixed: grid
convergence can pass, but `box_converged` and `radial_space_converged` remain
false. Without this policy, sampling continues to request one fixed grid.

These are measured refinement differences, not certified error bounds. The
inner endpoint remains fixed; convergence of energies does not establish
convergence of every transition or contact observable. Expanded MCDHF spaces
need continuum and configuration-space checks; this policy rejects that
route. Refinement adds runtime and is not enabled by default.

Measured He, Ne and Cf runs passed the default criteria: 1000 and 1500 points
at 75 bohr, followed by 2250 points at 112.5 bohr. Cf's last comparison changed
total energy by 1.16e-8 Ha, the largest orbital energy by 2.93e-8 Ha, and the
largest excitation energy by 1.89e-10 Ha. Its three calculations took 34.27 s
together, including first Cf angular setup. This is a verification cost,
distinct from an ordinary repeated solve.

## Local initializer cost (2026-09-29)

The approximate Gu starter now stops at local-field SCF tolerance 1e-5
instead of 1e-9. It supplies orbitals to the nonlocal DF equations; their
energy, orbital-update and occupied-stationarity criteria are unchanged.
Initializer iterations, tolerance and convergence are recorded separately.

In a same-process comparison, default Cf with Breit/QED required 31 starter
iterations instead of 103. Starter time fell from 1.70 s to 0.70 s; full warm
solve time fell from 5.07 s to 4.06 s (about 20%). Both used seven coherent
DF sweeps; total energies differed by 1.6e-10 Ha. Ne took 0.20 s and 0.17 s.
These timings are individual measurements, not a general speed guarantee.

The final current-code run gave warm medians of 0.175 s (Ne), 0.251 s (C),
and 3.964 s (Cf), using two repeats after each first solve. Cf's first solve
after Ne and C took 10.667 s. Source hashes accompany that run.

A separate Dirac–Coulomb comparison at SCF tolerance 1e-11 covered H, C, Ca,
Xe and Cf. Both starter choices converged throughout. Maximum changes were
2.2e-11 Ha in total energy and 9.5e-12 Ha in orbital energies.

Raw probes and scripts are in
`shooting-refinement` (local evidence: `../benchmarks/runs/2026-09-29-shooting-refinement/`).
`benchmarks/bench_df_solver.py --refine --z 98 --repeats 1 --out <new-directory>`
reproduces the adaptive measurement. Set OpenBLAS and OMP to one thread for
comparisons; JIT and first-use angular setup affect first solves.

Verification: 45 shooting/refinement tests and 107 API, persistence, sampling
and architecture tests passed. Tests cover box recovery, fixed boxes, exhausted
limits, SCF failure, analytic hydrogen energy, coherent C states and saved
refinement metadata. Logs are in the same measurement directory. This is a
focused suite, not a claim that every repository test passes.

## Propagation buffers and fixed integration geometry (2026-09-30)

DIR0 coefficient arrays now belong to one radial sweep and are reused by
adaptive DIR propagation, DOUTW/DINW and the scaled box Green solver. Each
call owns its buffers; concurrent solves share no scratch arrays. The
recurrence, adaptive step rules, normalization and acceptance tolerances
are unchanged. Only the active coefficient prefix is consumed, and a zero
middle-region solution resets its turning-point coefficient explicitly.

The Breit configuration-average evaluation owns an immutable copy of its
radial mesh. This permits reuse of bounded Coulomb integration geometry
without changing or freezing a caller's mutable workspace. Densities and
physical matrix elements are recomputed for each evaluation.

Four alternating fresh processes (after/before/after/before), with two warm
solves per element in each process, give the following medians. The baseline
is commit `0a1685e`; the settings are default shooting HF, Hamiltonian
averaging, Breit/QED, one OpenBLAS/OMP thread. First calls are excluded.

| Complete solve | Before | After | Time reduction |
|---|---:|---:|---:|
| Ne | 0.1723 s | 0.1658 s | 3.8% |
| Cf | 4.3690 s | 3.8369 s | 12.2% |

Every measured total energy, orbital energy and iteration count is identical.
These are scoped measurements: other diagnostic processes were active during
part of the panel. The preceding six-process propagation-only panel gives a
smaller full-Cf benefit, about 2%; geometry reuse accounts for most of the
combined improvement.

A live Salvat RADIAL DBOUND panel covers 27 bound states, through n=3, on
matched point/screened local fields at Z=2,26,98. Both codes receive the same
1000-point potential/output grid, initial energy, tolerance 1e-11 and speed
of light. Potential preparation and startup are excluded from kernel timings.
Maximum energy difference is **2.27e-13 Ha**; maximum P/Q error relative to the
reference peak is **1.46e-15**. Native before/after wave hashes are identical.
Median native kernel time falls from **1.167 to 1.104 ms**; the sum of the
per-case medians falls by **9.3%**. RADIAL's median CPU time is **1.150 ms**.
Native wall time and reference CPU time are different measurements, so this
does not establish a general Atomx/RADIAL throughput ratio.

The live 36-case DFREE panel at radius 200 Bohr and 3000 source points retains
identical errors: **1.22e-8** relative to asymptotic amplitude and **1.22e-8 rad**
modulo pi. Its summed per-case time changes by only about 0.4%; special-function
matching dominates some cases. A short neutral box failed the existing
matching check and is retained as a failed attempt, rather than counted as
agreement. Automatic continuum box selection remains unfinished.

Focused suites pass 155, 123 and 113 tests (overlapping selections), including
poisoned scratch-buffer reuse, heavy bound states, sourced orbitals, adaptive
grids and Breit operators. The earlier grouped-change gate passed 681 tests;
the current full suite collects 3817 tests. These are not a full-suite pass.

Use `benchmarks/bench_radial_bound_kernels.py --probe <DBOUND-probe>
--radial-source <matched-radial.f> --out <new-directory>` for the frozen-field
comparison, and `benchmarks/bench_df_solver.py` for the complete DF measurement.
The DBOUND probe source is `benchmarks/reference_probes/radial_bound.f90`.
Raw runs, profiles, source hashes and the audited summary are retained under
`benchmarks/runs/2026-09-30-shooting-performance/`.

Remaining costs include the local initializer, constrained Green-response
assembly, nonlocal exchange contractions and occupied rotations. RADIAL's
local-field solve does not include these nonlocal Hamiltonian DF operations.

## Earlier optimization measurements (2026-09-29)

Two changes reduce repeated work without relaxing the physical operator or
the SCF acceptance criteria:

- Gaunt magnetic sums use the exact rule `m_a-m_c=q+mu` and combine
  identical radial keys before determinant-table assembly. All allowed
  multipoles and static retardation remain included; no cutoff is added.
- Occupied rotations retain fixed one-body energies, density pairs and
  multipole potentials for the orbitals that do not change. Each orbital
  pair gets a fresh callback. State or SCF updates cannot reuse those caches.
  `physics.hartree_fock` and `physics.csf.radial_fock` supply the coefficients;
  `numerics.shooting.energy` performs radial contractions, and
  `numerics.orbital_rotations` selects the stationary angle.

The current-working-tree benchmark uses one OpenBLAS/OMP thread, shooting HF,
Hamiltonian averaging, default Breit/QED and default convergence controls.
Times include `Atom.solve()` only, with no profiler. These are local
measurements, not hardware-independent performance guarantees.

| Cf (Z=98) | Before these optimizations | After |
|---|---:|---:|
| First Cf solve after Ne and C | 22.11 s | 15.17 s |
| Repeated solve | 7.22 s (one repeat) | 5.84 s (median of two repeats) |
| Coherent DF sweeps | 7 | 7 |
| Total energy (Ha) | -32979.043209568204 | -32979.043209568226 |

The energy difference is 2.2e-11 Ha from summation order. Both runs converged.
Ne and C repeated solves remain roughly 0.24 s and 0.30 s, respectively;
these cases show no material speedup from this change. The first measured
solve may reuse disk JIT caches and is not described as a cold compilation.

Raw measurements and source hashes are in
`benchmarks/runs/2026-09-29-df-optimization` (local evidence: `../benchmarks/runs/2026-09-29-df-optimization/`).
Reproduce the current implementation with a new output directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
  python benchmarks/bench_df_solver.py --out /tmp/atomx-df-timing --repeats 3
```

Validation includes the unrestricted magnetic sum through g channels,
independent DBSR static Breit totals, cached versus full coherent/ensemble
energies during rotations, and independent radial-engine checks. The broader
property run still has a hyperfine snapshot discrepancy (610.60427 versus
610.46 ± 0.05 ppm); restoring the pre-optimization functions reproduces it.
The rank-3 direct-versus-Poisson Slater discrepancy was subsequently traced
to product quadrature across the Coulomb cusp. The direct reference now
integrates the two triangles in each coincident cell independently; ranks
0–6 agree near 1e-14 relative, with analytic/high-precision checks on both
implementations. The test tolerance was tightened. See
[verification status](VERIFICATION.md) for current external comparisons.
The API checks also exposed a comparison
formatter treating nested provenance as energy; it now formats numeric
energy entries and omits private metadata. Its regression test passes.

The new profile still identifies angular-table assembly as the largest
first-run cost. Repeated Cf solves spend about 1.9 profiled seconds in the
local seed and 1.3 seconds in inhomogeneous orbital solves; occupied rotations
now account for about 1.0 second. These remain targets for further work.

## Historical performance audit (2026-09-29)

Compared the isolated September 24 commit `b456fab` with the working direct DF
implementation. Runs used the same Python, one OpenBLAS/OMP thread, `Atom.solve()`
defaults and Breit/QED enabled. Timings below include Python profiling overhead;
repeated runs reuse JIT and angular caches. The physical implementations differ,
so this is a performance audit, not a claim of identical calculated energies.

| Case | September 24 | Current direct DF | Observation |
|---|---:|---:|---|
| Ne, repeated | 0.234 s | 0.524 s | Current coupled stationarity machinery still costs more |
| C, repeated | 1.665 s | 0.619 s | Direct coherent DF is about 2.7 times faster |
| Cf, first run after light atoms | Stopped at 120 s | 54.985 s, converged | Old case was still constructing angular tables; no old convergence claim |
| Cf, repeated | Not measured | 14.112 s, converged | Seven coherent DF sweeps after initialization |

Commit `9703c74` (September 29) introduced finite-electron-space orbital
optimization on the shooting CSF route, also reached by the old single-config
`averaging="ci"` default. Earlier code used population-averaged radial updates
and omitted coherent CSF terms from those equations. The direct DF route
restores the efficient radial mechanism and retains the coherent terms.

A final unprofiled run of the exact Cf example took 36.12 seconds, converged
in seven coherent sweeps after 11 initializer sweeps, and returned
`HamiltonianDFResult` with total energy -32979.043209568204 Ha. The
orbital residual was 2.25e-4 against a 3.16e-4 stopping threshold.
Breit and QED remained enabled, with their stages recorded.

The current first profiled Cf run spent about 35 seconds constructing static Breit
angular tables. On repetition, direct radial DF accounted for about 12.7 of
14.1 seconds; occupied rotations accounted for about 4 seconds. These are
measured remaining costs, not an assertion that optimization is finished.

## Accuracy checks

Carbon direct shooting at 2000 radial points and tight SCF tolerance agrees
with the independent B-spline source equations within 2e-9 hartree. Tests also
check Eq. 10 occupations, coherent Coulomb energy derivatives, normalized
Dirac source equations including negative-energy response, and odd-harmonic
occupied stationarity. A matched self-consistent electron branch is essential:
a fixed nuclear positive-energy variational space defines a different constraint.

Cf shooting Dirac–Coulomb energies were -33059.93118330816 Ha on 1000 points
and -33059.931183808054 Ha on 2000 points with tighter SCF tolerance. The
approximately 5e-7 Ha change includes both grid and stopping-tolerance effects.
This is a limited refinement check, not an experimental-accuracy guarantee.

The independent B-spline Cf calculation also converged (seven coherent sweeps):
-33059.93118402019 Ha at `tol=1e-7`, within 7.2e-7 Ha of the 1000-point
shooting result. It took 126 seconds with profiling, for Dirac–Coulomb only.
Heavy B-spline DF remains substantially slower than direct shooting.

B-spline improvements include selected-orbital gradients, exact reuse of
unchanged density-pair potentials during occupied rotations, and the existing
factorized direct Coulomb kernel starting at 96 splines. On a 113-spline Cf
physics grid, individual scalar Slater contractions were 13–30 times faster
for ranks 0, 2, and 6, agreeing within 2e-12 relative. This is a kernel timing,
not an end-to-end speedup claim; exchange kernel thresholds are unchanged.

## Independent orbital accuracy for properties

A variational energy has second-order errors in an orbital perturbation;
momentum integrals and many other properties generally have first-order
errors. An energy threshold alone therefore cannot specify property accuracy.
`Atom.solve`, `Sampling`, native shooting HF, and B-spline DF accept
`orbital_tolerance` separately:

```python
from atomx import Atom, Basis, Sampling

atom = Atom(10, engine="shooting", corrections="none",
    basis=Basis(sampling=Sampling(n_grid=2000, orbital_tolerance=1e-9)))
atom.solve(tol=1e-9, max_iter=100)
print(atom.result.solver_diagnostics)
```

This bounds the undamped radial L2 update before mixing, using the native
quadrature. It also controls the state-update threshold and the stagnation
trigger; energy and occupied-rotation checks remain active. Omitting it
preserves the existing `sqrt(min(tol, 1e-7))` orbital threshold. Its resolved
value is recorded in `solver_diagnostics`, and the requested override is in
the calculation specification. An explicit `Atom.solve(orbital_tolerance=...)`
overrides the sampling value and is forwarded to every adaptive grid solve.

On B-splines the norm uses the full Dirac overlap matrix, including both P
and Q. The additional gate checks both the undamped proposal and the complete
sweep after orthonormalization and rotations; existing coefficient/energy
criteria remain active. Local-exchange B-spline calculations evaluate the
unmixed-potential proposal as well as the mixed-potential update. This adds
an eigensolve per kappa only when the independent threshold is requested.
Inner selected eigenpairs are refined to a target no looser than one tenth
of the requested orbital-update threshold. That algebraic error estimate
does not certify the radial representation.

Hamiltonian DF supports the override on both engines and separately reports
the final coherent-density change and occupied-rotation defect (in Hartree).
Diagnostics retain their individual units; a variational-gradient residual
is not substituted for a dimensionless orbital change. MCDHF, analytic and
nonrelativistic method dispatch, local-exchange shooting and noniterative
TFDA/GSZ reject unsupported requests explicitly. Full MCDHF continues to use
its existing gradient and energy convergence controls. The L2 update itself
is a stopping test, not a certified distance to the exact orbital.

### Public property checks and balanced direct DF (2026-09-30)

The public control panel (local evidence: `../benchmarks/runs/2026-09-30-orbital-controls/`)
holds the Ne SCF tolerance at 1e-9 and compares 2000-point shooting against
120 B-splines. Corrections use the same converged orbitals; QED is off.
For the point nucleus:

| Shooting orbital-update request | Absolute engine difference in total DC energy (Ha) | Finite-frequency Breit (Ha) | Breit SMS coefficient (a.u.) |
|---|---:|---:|---:|
| Default derived threshold | 1.11e-9 | 4.08e-9 | 1.24e-4 |
| 1e-7 | 1.19e-12 | 6.35e-11 | 4.33e-7 |
| 1e-9 | 1.28e-12 | 2.67e-12 | 5.34e-9 |

Fermi-nucleus cases show the same trend. Refining to 4000 shooting points
and 160 B-splines leaves energy differences below 1.8e-13 Ha and SMS
coefficient differences below 2.4e-9 a.u. These are internal discretization
comparisons of the specified operators. In the Fermi cases the SMS operator
still has point-Coulomb recoil form; this is not a finite-size recoil result
or an experimental precision claim.

This panel also exposed a separate root-selection bug: specifying
`Basis(ns=100)` for Hamiltonian-averaged carbon previously used equal P/Q
B-spline orders, allowing the direct source equations to follow a spurious
positive-branch root. The calculation falsely converged near -49.27168949 Ha.
Direct Hamiltonian DF now requires the balanced P/Q representation on both
automatic and explicit-size grids. The 80/120/160-spline results agree with
2000-point shooting near -37.70513100357 Ha, within 3.5e-12 Ha in this panel.
Requested basis sizes remain in effect; basis orders are recorded. The
result iteration count now reports the final orbital optimizer's sweeps
instead of the B-spline initializer's outer iterations.

### Ne engine agreement and finite-nucleus grid repair

The fixed-size B-spline grid previously used a first knot of `0.05/Z` for
Fermi nuclei. At Ne this first interval was much wider than the nucleus.
Increasing `ns` refined the outer grid while leaving this interval intact,
creating false convergence of the finite-size shift. A fixed-size Fermi
basis now uses the same nuclear surface scale (`fermi_grid_step`) as the
automatic physics grid. Explicitly supplied `r_min` is still honored.
A larger radial dynamic range can require a larger fixed `ns`; the change
does not certify an arbitrary basis size at every element.

The Ne radial audit (local evidence: `../benchmarks/runs/2026-09-29-neon-radial-audit/`) retains
before/after values, raw DBSR output, solver controls and timing scripts.
With 120 B-splines and 2000 shooting points, corrections disabled:

| Nuclear model | Energy difference, Ha | Relative Breit-SMS difference |
|---|---:|---:|
| Point | 1.22e-12 | 3.63e-10 |
| Fermi | 1.19e-12 | 2.31e-10 |

The B-spline solves used `tol=1e-9`; shooting used `tol=1e-10` and
`orbital_tolerance=1e-9`. All these refined runs converged. These are separate
engine controls meeting matched observable accuracy, not an assertion that
the historical scalar `tol` has identical semantics in both solvers.
The earlier request for 1e-11 B-spline orbital changes exposed an inner
eigensolver error floor. The residual-controlled refinement described below
now resolves this case without changing the SCF stopping threshold.

For the Fermi case, median full construction-plus-SCF times over three
repeated runs were 1.10 s (B-spline) and 0.51 s (shooting), one thread,
excluding imports and the subsequent SMS evaluation. First-run times are
also retained; they are not guaranteed to represent cold compilation.

A 2026-09-29 DBSR_HF Fermi calculation using the previously documented matched-C
build differs in total energy by 4.57e-8 Ha. It did not meet its requested
1e-9 orbital tolerance (reported 7.89e-8 after 100 iterations), despite a small
energy change. The [follow-up reference audit](VERIFICATION.md#dbsr-orbital-tails-and-angular-precision-2026-09-30)
resolves the convergence failure by making the reference tail cutoff consistent
with its orbital tolerance. It also identifies a single-precision angular
prefactor in DBSR that accounts for the remaining energy difference on common
spinors. The earlier focused regression run passed 136 tests.

## Inner eigenpair accuracy and tight B-spline convergence

An accurate total energy does not ensure an accurate eigenvector. The
generalized Dirac spectrum includes a negative-energy branch and large
positive box energies; a backward-stable full-spectrum diagonalization can
leave a low bound orbital inaccurate enough to stall a tight SCF request.
Refining only high principal roots therefore missed the Ne case.

`DiracResult.get_orbital(index, tolerance=...)` now estimates the error of
each requested root in the actual constrained eigenproblem. With residual
`r = H x - epsilon S x` and S-orthonormal eigenvectors `v_j`, the first-order
unwanted-mode coefficient is `v_j.T r / (epsilon_j - epsilon)`. Both electron
and negative-energy modes enter this estimate. The parallel residual gives
the energy error estimate. Residual products use extended accumulation;
these are algebraic error estimates, not radial-basis error bounds.

If the estimate exceeds the requested threshold, the existing bordered
Newton solve refines that root, preserving metric normalization,
occupied-orbital constraints and root identity. Unrequested continuum
vectors are not refined. A successful loose check returns the original
embedded vector without recomputing its projection; this avoids introducing
different summation roundoff into an otherwise unchanged SCF sequence.
The historical high-root refinement remains available without an explicit
accuracy request.

All B-spline SCF routes now use the common requested-root interface. The
inner threshold is `0.001*tol`, ten times below the existing outer relative
energy-change threshold `0.01*tol`. Neither outer stopping condition was
relaxed. The resolved inner threshold is recorded in `CalculationSpec`.
The alternate per-orbital HF mode also rejects a missing bound root instead
of substituting the last physical/continuum eigenvector.

The retained panel (local evidence: `../benchmarks/runs/2026-09-30-bspline-convergence/`)
compares identical 120-spline calculations with the previous low-root
extraction and the new accuracy check. Corrections are disabled. At
`tol=1e-11`:

| System | Previous iterations / outcome | Refined iterations / outcome | Previous / refined time, s |
|---|---|---|---|
| Ne, point nucleus | 100 / not converged | 27 / converged | 4.59 / 1.50 |
| Ne, Fermi nucleus | 100 / not converged | 27 / converged | 4.59 / 1.54 |
| Fe16+, Fermi nucleus | 100 / not converged | 12 / converged | 4.62 / 0.72 |
| Xe44+, Fermi nucleus | 10 / converged | 9 / converged | 0.51 / 0.55 |

Total energies differ by at most 1.2e-13 Ha. At `tol=1e-5`, the iteration
counts are unchanged and the four totals are bit-identical; the new path
performs zero Newton refinements except for one early Ne/Fermi root. The
residual checks add about 4–5% in the loose Fe/Xe samples. The Ne loose
timings include first-case setup differences and do not establish a speedup.
These are single sequential samples with one BLAS/OMP thread, not statistical
performance estimates. A run that failed its accuracy target is not an
equal-accuracy speed reference.

The earlier public finite-frequency Ne check was repeated at `tol=1e-11`:
both radial engines and both static/finite-frequency modes now converge.
This repairs the B-spline failure retained in the public-API artifact. It
does not make the engines' historical scalar `tol` semantics identical;
the public independent orbital control described above addresses direct DF.
Other physical methods and the old API energy snapshots remain separate audits.

## Analytic exponential moments in shooting source integration

The six-node Green-source interpolation now uses analytic exponential
moments instead of repeated subdivided Gauss quadrature. This changes the
integration of the interpolating polynomial, retaining the same source
interpolation, evanescent rates, log-amplitude cancellation, stencil and
coarse-grid rejection. Radial refinement remains necessary for the actual
source. This does not change the Hamiltonian or the SCF accuracy controls.

For a cell with rate `s >= 0`, the moments are
`M_m = (-1)^m integral_0^1 u^m exp(-s*u) du`, for `m=0,...,5`.
Above `s=8`, integration by parts gives a stable upward recurrence. Below
eight, the highest moment is computed with a positive convergent series,
followed by downward recurrence using positive additions. This avoids
subtracting nearly equal quantities near zero. The six moments and 180
fixed polynomial coefficients replace the large Gauss basis table. Scratch
belongs to each preparation call; no mutable global workspace is introduced.

The [retained measurements](../benchmarks/runs/2026-09-30-shooting-moments/)
use baseline `eb7bdae`. Four fresh processes run in before/after/after/before
order, with four solves per element per process; the first solve is excluded.
All runs retain default Breit + QED and Hamiltonian averaging.

| Element | Previous warm median, s | Current warm median, s | Change |
|---|---:|---:|---:|
| H | 0.00599 | 0.00620 | +3.5% |
| C | 0.23504 | 0.21767 | -7.4% |
| Ne | 0.16545 | 0.15857 | -4.2% |
| Cf | 3.87274 | 3.77270 | -2.6% |

These are workload measurements, with six warm samples per implementation
per element. H does not use the optimized nonlocal source integration;
its small timing change is not evidence of a changed bound solver. Iteration
counts and convergence are unchanged. Total energies change by at most
`4.4e-11` Ha and occupied orbital energies by `1.2e-11` Ha in this panel.

On a 2000-node kernel control, source-weight preparation is 1.6–4.1 times
faster at slopes 0–20 and 21.7 times faster at slope 100. A complete local
Dirac response, including preparation and application, improves from
0.821 to 0.592 ms. These isolated measurements use one process per
implementation, 30 kernel samples and 20 response samples after warmup.
They do not predict equal improvements in complete atomic calculations.

Before/after checks cover native Ne orbitals, E1/M1/E2 transitions in both
gauges, response, screened continuum waves and phase shifts, plus CE, EII,
PI and RR using shooting continua. The latter coefficient APIs explicitly
use B-spline bound states: plain `ShootingResult` bound states are currently
unsupported by those entry points. No replacement calculation is hidden.
Their sampled cross sections, continuum waves and phases are identical.
AI/DR, other properties and correlated methods are included in the full
repository audit; its failures are recorded separately from focused numerical
checks. Existing reference values and acceptance bands are unchanged.

The remaining warm Cf profile is led by the local initializer, sourced-orbital
iteration, Fock actions, derivative evaluation and occupied rotations.
Analytic moments reduce Green-map preparation but do not eliminate the
nonlocal exchange or variational constraints that dominate a complete solve.

The full audit after the moments change, before the Schrödinger follow-up,
reports **3746 passed, 53 failed, 2 setup errors, 11 skipped, 21 expected
failures**. All 55 failures/errors reproduce on baseline `eb7bdae`. They
include old energy/RR snapshots, a removed determinism script, configuration
API assumptions and independent physics/property checks; they are not all
classified as harmless snapshots. Existing gates and references are retained.

### Schrödinger matching and failed-root recovery

The audit exposed six Breit–Pauli failures caused by the nonrelativistic
shooting root solver. A trial energy below the effective-potential minimum
has no classical turning point. Salvat SBOUND matches near the origin and
uses the derivative mismatch to recover the energy. Atomx incorrectly
selected the outer edge and reported an insufficient box. The fallback now
matches RADIAL. Twelve live p/d controls at varied initial energies change
from seven failed roots to twelve successes, with bit-identical reference
energies and waves. This is a correctness repair, not a new speed claim.

Failed roots and iteration exhaustion return no orbital payload. For s waves,
the returned Schrödinger Q=dP/dr at the origin is reconstructed from the
regular series; the zero used to start outward propagation is not the
physical derivative. Independent hydrogenic P/dP/dr and normalization controls
cover Z=1,2,10, multiple angular momenta and starting energies. The compiled
local-SCF caller is updated with the callee so its cache is rebuilt.

The final focused numerical suite passes **161 tests**, including all six
repaired Breit–Pauli failures. Final H/C/Ne/Cf Dirac calculations retain
exactly the moments-panel energies and iteration counts. The other full-audit
baseline failures remain separate work; the complete repository is not
certified by the focused numerical gate.

Two additional safety gates now exercise the current Atom API directly,
replacing calls to the retired determinism CLI. Three fresh full-panel
processes return identical energies and wave hashes; a separate fresh
compilation passes Numba bounds checking. The panel covers shooting,
B-splines, explicit MCDHF and Cf. Test and nightly workflows retain their
full-panel repeat counts through the new tests. Together with the numerical
suite, **163 checks pass**, repairing eight failures from the initial audit.
