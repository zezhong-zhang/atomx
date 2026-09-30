# Verification status and comparison contracts

Updated 2026-09-30. The objective remains agreement in physics, numerical
accuracy and performance across structure, spectra, properties and collision
calculations. The current live panel below covers only a subset. Neither a
converged SCF flag nor a small total-energy difference validates every output.

## Current live structure evidence

[`benchmarks/bench_reference_structure.py`](../benchmarks/bench_reference_structure.py)
preserves each executable command, input, stdout/stderr, raw reference files,
source hashes, convergence status and timings. A row marked `measured` means
that it was measured; it is not an accuracy pass. Missing orbitals are reported
as missing, never as zero error. Printed rounding bounds describe serialization
precision and are not experimental uncertainties.

### DBSR_HF: matched point-nucleus Dirac–Coulomb calculations

The reference is the supplied `~/code/dbsr_hf/DBSR_HF` source compiled in an
isolated directory. The physical comparison uses the actual integer jj ensemble
in DBSR's `.conf`, infinite nuclear mass, point nuclei and corrections disabled.
For the current panel, `c_au`, `one_over_alpha` and `alpha` were set to Atomx's
CODATA 2022 values in an isolated reference build. A diagnosed single-precision
square root in the reference's `Cjkj` angular prefactor was also changed to
double precision, as detailed below. A read-only diagnostic prints final Fock
expectations. The original source and earlier binaries are preserved. This is
explicitly a comparison to that corrected reference build; other reference
constants and algorithms are unchanged.

| Neutral atom | Shooting − corrected DBSR total energy (Ha) | Atomx repeated solve (s) | DBSR process (s) |
|---|---:|---:|---:|
| He | +2.0e-14 | 0.066 | 1.225 |
| C | +2.3e-13 | 0.573 | 4.378 |
| Ne | +1.3e-12 | 0.508 | 8.836 |
| Xe | −2.4e-9 | 3.180 | 64.055 |

These timings use one BLAS/OMP thread. Atomx timing includes construction and
solve, excluding imports; DBSR includes process startup and file output. One
reference process and two warm Atomx repeats were measured per case (the table
uses their median). Atomx uses its installed BLAS; the isolated reference uses
the recorded LAPACK/BLAS build. This is a small panel, not an algorithmic or
general performance ratio. All eight SCF calculations met their requested
stopping criteria, whose definitions are recorded separately.

Raw evidence:
[`2026-09-30-dbsr-orbital-accuracy`](../benchmarks/runs/2026-09-30-dbsr-orbital-accuracy/).
All 26 reported orbital mean radii across these four atoms agree within DBSR's
three-decimal print rounding. That coarse output does not demonstrate high
precision for individual radii. Orbital energies need the separate convention
check below.

### Why the initial DBSR comparisons disagreed

1. **Different states.** DBSR's default carbon `.conf` has one electron in
   each 2p_j subshell. It is not Atomx's nonrelativistic statistical ensemble.
   Matching the actual configurations removed a 0.0049-Ha discrepancy.
2. **Nuclear input was ignored.** The supplied reference reads `nuclear=point`
   from the `.knot` file; the same text on the command line does not select it.
3. **Unresolved point-nucleus cusp.** Xe's default initial knot spacing
   `hi=0.25` gives a 0.135-Ha discrepancy. Refining only `hi` to 6.1e-5
   reduces it to 2.2e-6 Ha with the original reference constants.
4. **Different constants.** DBSR uses `c=137.03599976`; Atomx uses
   `c=137.03599917759013`. Matching these reduces the same-grid Xe discrepancy
   to 2.2e-7 Ha. The later angular-prefactor audit below identifies the leading
   persistent discrepancy that remained after radial refinement.
5. **Different reported quantities.** DBSR retires converged orbitals before
   the full SCF ends. Its printed `e(nl)` may be an eigenvalue from an earlier
   field. Final Fock expectations are recorded separately by an isolated
   diagnostic build. Neither quantity is an excitation energy or an ionization
energy obtained from two total-energy calculations. Diagonal Fock values
also require care with occupied-space rotations.

### DBSR orbital tails and angular precision (2026-09-30)

The earlier Ne/Fermi run requested `orb_tol=1e-9` while retaining
`end_tol=1e-7`. It repeatedly truncated coefficients larger than its stopping
threshold and failed to converge after 100 sweeps. Changing only the tail
cutoff to 1e-12 meets the requested orbital residual (9.63e-10), taking 13 s
on the retained build. The benchmark runner now records both controls and
rejects a tail cutoff at or above the requested orbital tolerance. Atomx's
independent L2 control is recorded separately; DBSR uses a relative maximum
P-coefficient update, so their threshold values are not equivalent error bounds.

A remaining Ne/Fermi total-energy difference of 4.56976e-8 Ha persists when
Atomx evaluates the exact same DBSR B-spline coefficients on identical knots.
One-electron energies agree within 6e-13 Ha; the discrepancy is in exchange.
`dbsr_lib_zcom.f90::Cjkj` evaluates
`sqrt(real((j1+1)*(j2+1)))` in default single precision. Its rounded prefactor
predicts a 4.56977e-8-Ha shift; the unexplained remainder is 1.4e-13 Ha.
Changing only this `real` conversion to `kind=8` in an isolated build confirms
agreement on the frozen spinors within 2e-13 Ha. This identifies a reference
precision limitation; Atomx's physical operator was not changed to imitate it.

The new He/C/Ne/Xe point panel uses that explicitly corrected reference, with
`hi=1.52587890625e-5`, `he=0.125`, `hmax=1`, orbital tolerance 1e-9 and tail
cutoff 1e-12. B-splines on the same grid settings differ from corrected DBSR
by at most 2.1e-11 Ha; the corresponding shooting differences are in the table.
Refining Xe's initial B-spline spacing by four and shooting to 4000 points
reduces the difference between those engines from 2.4e-9 to 2.0e-10 Ha,
consistent with the point-nucleus cusp being the remaining radial limitation.
The largest difference between their final orbital Fock expectations is
2.4e-9 Ha for the shooting panel. These are DF discretization/operator checks,
not correlated excitation energies or NIST accuracy claims. Saved Ne spinors,
one-/two-electron energies and final Fock expectations provide independent
regression fixtures in `test_dbsr_common_spinors.py`.

In the earlier 2026-09-29 final-Fock diagnostic run, the maximum orbital differences
at the then-default Atomx controls were 7.8e-9 Ha (He), 5.4e-6 Ha (C),
5.1e-6 Ha (Ne), and 8.5e-6 Ha (Xe). Tightening Xe to 4000 points and SCF
tolerance 1e-12 reduces its maximum difference to 3.5e-7 Ha. The diagnostic
does not alter reference orbitals or total energies; outputs and its patch
are in [`final-fock`](../benchmarks/runs/2026-09-29-dbsr-final-fock/).

Atomx Xe point-nucleus energies at 2,000 and 4,000 grid points differ by
1.2e-10 Ha; reducing the inner grid radius from 1e-7 to 1e-8 bohr at 4,000
points changes the energy by about 3e-12 Ha. These refinement observations
apply to this calculation, not to all atoms or operators.

Refinement traces are in
[`cusp-refinement`](../benchmarks/runs/2026-09-29-dbsr-cusp-refinement/),
[`cusp-tight`](../benchmarks/runs/2026-09-29-dbsr-cusp-tight/) and
[`grid-check`](../benchmarks/runs/2026-09-29-dbsr-grid-check/).
The earlier full-grid run timed out after 180 seconds; its partial trace is
retained and is not used as a converged reference.

### RADIAL/DHFS: local Slater exchange

The point-nucleus He/C/Ne/Xe panel uses the reference input's occupations,
exchange parameter, grid size and radial box. Both energies are HF expectations
on local-exchange orbitals. The differences are −4.2e-8, +9.9e-8, +2.7e-6
and −2.3e-5 Ha, respectively, all within the reference's printed rounding bounds.
This establishes agreement at that output precision. RADIAL's constant values
are not exactly Atomx's CODATA 2022 values.

Finite Fermi comparisons require explicit charge-profile matching: RADIAL uses
`c=1.07*A^(1/3)` fm and `a=0.546` fm, whereas Atomx's DBSR-style profile uses
another radius/diffuseness prescription. Equal atomic weights alone do not
select the same nuclear potential. The runner rejects that unsupported claim.

### FAC: model comparison

The live He-like 1s² panel at Z=2,6,10,54 disables Breit, SE, VP **and both mass
shifts** (`SetMS(0,0)`). FAC's default mass-shift settings are otherwise active.
The total-energy differences are −3.52e-4, −3.69e-4, −4.45e-4 and −1.20e-2 Ha.
FAC's optimized central potential and Atomx's nonlocal DF are different orbital
models. These discrepancies are not treated as numerical errors or passes.
The current FAC output also does not certify the radial convergence residual.

RADIAL/FAC raw data:
[`2026-09-29-structure-matched`](../benchmarks/runs/2026-09-29-structure-matched/).
Its original DBSR rows have been annotated with the later grid/constant findings.
The first `2026-09-29-structure` run also had unmatched carbon occupations and
FAC mass shifts; it is retained as an audited failed comparison setup.

## Executable numerical and design checks

### Live radiative-transition audit (2026-09-29)

`benchmarks/bench_reference_transitions.py` retains FAC commands, inputs,
outputs, executable/source hashes and timings for one-electron, point-nucleus,
infinite-mass ions at Z=1,10,26,54,82. Breit, SE, VP and mass shifts are off.
States are matched by n, l, J and parity. The completed panel contains 115
rows: E1–E4 in FAC modes 0 and 1, M2–M4 in both modes, and M1 in mode 0
(FAC forces its relativistic M1 operator even when mode 1 is requested).

The audit exposed and repaired three convention defects:

- FAC's resolved `PrintTRTable` columns are **A**, not gA, and a signed
  reduced amplitude, not a second gf. The parser, writer and benchmark
  callers now use these meanings. Old rate comparisons that divided FAC's
  printed rate by `2J_upper+1` must be regenerated.
- Higher-multipole gf values now satisfy
  `gf = (2J_upper+1) A_au / (2 alpha^3 omega^2)`. The former conversion
  omitted the frequency factors above dipole order. Forbidden-line gf is
  a generalized FAC convention; NIST usually reports A and S instead.
- In the shared E/M rate normalization, the magnetic angular factor is
  `(kappa_i+kappa_f)/(L+1)`. The old denominator `2L+1` gave low-Z M2/M3/M4
  rate ratios of approximately 0.360/0.327/0.309. Magnetic transitions and
  B-spline recombination now retain the finite-frequency spherical-Bessel
  radial operator. The wrong M1 factor had accidentally compensated omitted
  retardation in the inhibited 2s–1s line, masking the normalization error.

After these fixes, E1–E4 agree with FAC's **matching long-wavelength mode**
within 1.2e-5 relative across this panel. Finite-frequency M2–M4 agree within
2.9e-4. For M1, differences at Z=10,26,54,82 are about 0.26%, 0.04%, 0.008%
and 0.003%, respectively. These observations do not establish converged
reference error bounds.

The hydrogen inhibited M1 reference is unresolved: changing FAC's radial
settings gives 1.64e-7, 2.37e-6 and 6.53e-6 s^-1. Atomx's value stays near
2.4959235e-6 s^-1 from 1000 through 8000 points (about 1.1e-7 relative
spread), consistent with the independent leading-order Drake benchmark.
The initial FAC hydrogen discrepancy is **not** an agreement pass.

Electric finite-frequency operators are now implemented in both gauges.
The follow-up [live panel](../benchmarks/runs/2026-09-29-electric-finite-frequency-live/)
compares FAC mode 0 with the full Atomx operator: E1–E2 rate differences
are below 9e-6 relative and E1–E4 below 2.7e-4 over Z=1,10,26,54,82.
The maximum electric gauge **rate** discrepancy is 4.2e-12.
At Z=82 the finite/static rate ratio spans 0.4434–1.0642 in these channels;
the former ~126% maximum discrepancy was largely an operator mismatch.

The shared kernels now serve orbital/CSF transitions, determinant CI length
transitions, direct photoionization, RR and orbital DR radiative widths.
CI kernels use signed state frequencies; a static determinant-trace sum
rule is offered only with the explicit long-wavelength operator. See
[operator definitions and limits](TRANSITION_OPERATORS.md). Correlated-state
and continuum external validation remains incomplete; this one-electron
panel does not validate those models or omitted higher-order QED.

The finite-electric combined regression run passed **256 tests**, with two
existing He-like M1 xfails and one neon SMS basis-convergence failure.
The SMS failure was reproduced with the original `HEAD` static radial
operator: K_SMS(ns=50) = -117.83870453639743 versus
K_SMS(ns=70) = -117.83870438580823, outside its 1e-9 relative gate.
It is unresolved and is not caused by adding photon retardation. Logs and
the isolated baseline script are retained with the follow-up live panel.

Evidence: [initial convention panel](../benchmarks/runs/2026-09-29-transition-conventions-live/),
[magnetic correction panel](../benchmarks/runs/2026-09-29-magnetic-retardation-live/).
Tests cover Einstein absorption/emission consistency, live FAC data columns,
M2–M4 radial operators, B-spline hydrogenic M1, and recombination dispatch.
The focused physics run passed 104 tests with two existing He-like M1 xfails;
the final FAC convention/IO run passed 33 tests, including exported rate and
amplitude round trips. Some convention tests occur in both runs.
those xfails remain missing validation. Timings are retained, but FAC includes
23 transition rows and output while the Atomx orbital timer does not: they
are not an equal-work speed ratio.

The physical A/f/S distinction follows
[NIST's spectral-line conventions](https://physics.nist.gov/Pubs/AtSpec/node17.html).
The exact FAC column and amplitude contracts were checked against the supplied
`faclib/dbase.c::PrintTRTable`, `OscillatorStrength`, and
`faclib/radial.c::MultipoleRadialFRGrid` sources.

| Contract | Evidence | Limit |
|---|---|---|
| Full coherent Hamiltonian DF gradients and direct solver routing | `tests/test_direct_hamiltonian_df.py` | Selected cases; no complete-table accuracy claim |
| Independent shooting and B-spline orbital stationarity | Direct-DF tests, `test_shooting_open_shell_constraints.py`, `test_shooting_core_hole_stationarity.py` | Must match state, nuclear model and electron-branch constraint |
| Gaunt magnetic contractions and full static Breit | `test_breit_operator.py`, `test_static_breit_retardation.py` | Static default; not all QED orders |
| Explicit finite-frequency transverse CI | `test_finite_frequency_breit.py`; live FAC 120-value Z=1,10,26,54,82 panels | Within 0.3 ppm with independent point-Coulomb spinors and 0.05 ppm on shared FAC spinors; correlation-orbital prescription and full MCDHF integration remain open |
| Radial Coulomb cusp and multipole quadrature | `test_slater_integrator.py`, `test_ci_analytic_slater_condon.py` | Direct triangles and Poisson compared to analytic/high-precision integrals |
| External state labels, rounding and missing records | `test_external_structure_contracts.py` | Parser correctness is not solver correctness |
| Numerical/physical code does not import methods | `test_architecture_boundaries.py` | Executing import checks cover the selected drivers |
| API request fidelity and explicit MCDHF routing | `test_architecture_scientific_fidelity.py`, direct-DF tests | Does not certify every property consumer |

Finite-frequency transverse exchange now uses a separate physics operator
and reusable numerical Helmholtz kernels. The explicit CI API replaces
static Breit when a complete orbital-energy mapping is supplied; automatic
SCF/MCDHF defaults remain static. Direct angular and radial quadrature,
zero-frequency recovery, signed Hermiticity, live FAC matrix elements and
provenance are checked. See [transverse interaction](TRANSVERSE_INTERACTION.md)
for the exact scope and retained raw comparisons. The final focused
integration run passed 123 tests after kernel-geometry caching, using a
fresh isolated Numba cache. An earlier cache-reuse run failed while Numba
serialized a dead dispatcher weak reference; it is retained in the artifact.
Neutral Ne point/Fermi shooting and B-spline finite-frequency expectations
agree within 2.7e-12 Ha. This internal engine comparison does not establish
the same precision against an external code.

The [public orbital-control panel](../benchmarks/runs/2026-09-30-orbital-controls/)
now reproduces that comparison through `Atom.solve`/`Sampling`, with an
independent L2 update threshold. At fixed Ne energy tolerance 1e-9, tightening
shooting's orbital threshold to 1e-9 reduces its difference from B-splines
in the Breit SMS coefficient from 1.24e-4 to 5.34e-9 a.u. Point and Fermi
cases were checked, followed by radial refinement. This supports numerical
consistency of those observables, not accuracy against experiment.

The same work repaired a falsely converged Hamiltonian-DF carbon solution
on explicit-size unbalanced B-spline bases. Direct DF now constructs the
balanced representation for those grids too; 80/120/160 splines agree with
shooting within 3.5e-12 Ha in the retained panel. See
[solver controls](DF_SOLVER_CONTROLS.md#public-property-checks-and-balanced-direct-df-2026-09-30)
for the controls, unsupported routes and remaining limits.

The public API now exposes an opt-in orbital-difference frequency prescription
through `Corrections`, `CorrectionStack` and both relativistic method objects.
Ordinary DHF uses converged orbital energies; Hamiltonian DF/MCDHF require
an explicit map. Seven dedicated integration tests pass, including both
engines and rejection of double application. A broader run passed 183 tests
and failed 12: one new test used a wrong enum import (fixed; the seven-test
rerun passes), and 11 failures remain in the older API snapshot/control file.
The Ne snapshot difference matches a retained pre-change run; these old
expected numbers were not replaced. The full suite is still not green.
The [public Ne audit](../benchmarks/runs/2026-09-30-transverse-public/)
converges on both engines at an SCF tolerance of 1e-9; B-splines missed a
1e-11 target within 100 iterations. Both outcomes are retained.
That tighter B-spline failure is superseded by the
[requested-root eigenpair refinement](DF_SOLVER_CONTROLS.md#inner-eigenpair-accuracy-and-tight-b-spline-convergence):
Ne point/Fermi and Fe16+ now meet 1e-11 without relaxing the SCF criterion.
The corrected public Ne static/finite run converges on both engines.
The focused eigensolver, SCF sentinel, hydrogenic, CI and transverse-operator
regression run passes 136 tests. This includes metric/constraint preservation,
negative-energy contamination, the unchanged loose-request path and the
tight Ne stopping target; it does not resolve the older API snapshots.

The subsequent [FAC numerical-control audit](TRANSVERSE_INTERACTION.md#reference-refinement-and-common-spinor-diagnosis)
separates inner-grid spacing, inner orbital energy stopping criteria, and
the reference's small-argument Breit approximation. These controls explain
the earlier 24-ppm transverse discrepancies without changing Atomx's
physical operator. The Ne/Pb regression gates now require 1 ppm with
independent spinors and 0.1 ppm on shared FAC spinors. The live common-spinor
panel reaches 0.05 ppm; this certifies operator agreement on those orbitals,
not accuracy of all correlated energy levels.

The shared reference controls also improve Ne/Pb recoil radial comparisons
to 2.9 ppm. The subsequent [radiative operator audit](TRANSITION_OPERATORS.md#exact-line-frequencies-and-shared-spinors)
identifies the earlier 263-ppm Pb discrepancy as FAC's grouped photon-frequency
approximation. One resolved upper level per transition-table call removes
that discrepancy. The direct 100-value E1–E4/M1–M4 panel at five charges
agrees within 3.44e-8 relative on common spinors (1.76e-9 excluding H).
Independent spinors remain a separate accuracy question: the weak H/Ne M1
rates converge approximately quadratically with FAC grid density, while
Atomx shooting and B-spline refinement give substantially smaller changes.
At density factor 16, FAC still differs by about 802.5 ppm for H and
7.92 ppm for Ne relative to the refined B-spline result. The common-spinor
tests do not certify these independently solved FAC rates. Independent Ne
E4 rates also retain a 2.9-ppm discrepancy in this panel.
The original failures remain in retained data. Timings include different
work and are not used as controlled performance comparisons.
The strengthened integration suite passed 129 tests; the final signed
common-spinor/FAC-runner subset passed 49 tests (overlapping that suite).
The separate radiative integration run passed 129 tests: photon common-spinor
fixtures, independent electric multipoles, velocity gauge, transition
conventions, engine uniformity and CI line strengths. Its H/Ne/Pb fixtures
include raw spinors and hashes; electric FAC regression tolerance tightens
from 300 ppm to 10 ppm. This is not a full repository-suite result.

The Slater direct reference now resolves the r=s cusp by splitting coincident
cells into triangles. Ne ranks 0–6 agree with the default Poisson integrator
at about 1e-14 relative. Independent origin-cell monomial, hydrogenic and
high-precision exponential-density checks cover the individual implementations.
The direct/Poisson regression that previously failed has been repaired without
relaxing its tolerance; the comparison gate is now tighter.

The follow-up [recoil operator audit](RECOIL_OPERATORS.md) corrected the SMS
property's energy-dipole substitution and occupation overcounting, and an
unequal-j tensor-phase error in relativistic SMS2. The original isotope failure
above records the former implementation; it is superseded by direct-momentum
basis refinement. The focused isotope/recoil/correction run passes 124 tests.
The live FAC radial panel has 80 directed comparisons at five charges, agreeing
within 41 ppm. The subsequent [Ne radial audit](DF_SOLVER_CONTROLS.md#independent-orbital-accuracy-for-properties)
resolved the finite-nucleus energy gap by correcting the first knot of fixed
Fermi B-spline bases. Independent shooting orbital accuracy control brings
refined point/Fermi SMS agreement below 4e-10 relative and energy differences
to 1.3e-12 Ha. The focused regression run passed 136 tests. The native solver
control still needs integration into general public calculation controls;
the historical scalar tol retains different meanings across solvers.
A fresh DBSR Fermi comparison differs by 4.57e-8 Ha, but the reference missed
its 1e-9 orbital target (7.89e-8 residual); that gate remains incomplete.

The hyperfine numerical snapshot at 610.46 ± 0.05 ppm still fails at 610.60427
ppm and was reproduced with the pre-optimization functions. Its recoil scaling
identity passes. The full repository suite has not been established green.
A passing snapshot means reproducibility, not agreement with an external source.

## Frozen CI levels and E1 transitions

The 2026-09-30 live Ne-like Ar/Fe/Xe/Pb panel covers all 64 excited levels
of `2p5 3s` and `2p5 3d`, plus 40 E1 rates (five lines per ion, each with
long-wavelength and finite-frequency photons). Both codes use point nuclei,
infinite mass, Dirac–Coulomb energies and separate CI spaces for the two
excited configurations. Each code optimizes its own local ground potential;
the remaining differences include orbital-model differences.

An audit found that the historical FAC deck's `ConfigEnergy(0/1)` computed
configuration relaxation shifts and added them to diagonal CI matrix elements.
Atomx's frozen CI has no corresponding shifts. Disabling them with
`SetConfigEnergyMode(-1)` gives the following comparisons, with all other
controls held fixed in the `shift-control` / `frozen` runs:

| Ne-like ion (Z) | Largest level difference with FAC shifts (eV) | Without shifts (eV) | Largest finite-frequency A difference without shifts |
|---|---:|---:|---:|
| Ar (18) | 2.757 | 0.399 | 3.72% |
| Fe (26) | 2.673 | 0.409 | 1.62% |
| Xe (54) | 3.960 | 0.419 | 1.06% |
| Pb (82) | 7.494 | 0.515 | 0.23% |

The old photon fixture was FAC `MODE=1`, while Atomx's default had become
finite wavelength. The reference rate normalization itself was correct:
FAC E1 `S = amplitude²/2`, and its printed rate is A, not gA. Current tests
compare each photon mode explicitly and use the directly printed rates.
Finite wavelength changes a weak Xe line by about 15% in both codes.
This radiative photon effect is distinct from frequency-dependent transverse
exchange in the structure Hamiltonian; Breit is disabled in this panel.

The runner retains raw levels, transition tables, commands, process outputs,
source hashes and timings:
[`bench_reference_ci_transitions.py`](../benchmarks/bench_reference_ci_transitions.py).
Reference data are in `tests/data/fac_ne_like_ci_reference.json`;
the artifact directory is
`benchmarks/runs/2026-09-30-fe17-transition-audit/`.
The focused run passes 22 tests, including complete level/line inventories
across all four ions. The Fe rate gate tightened from 8% to 3%, and its
excitation-energy gate from 6 eV to 0.6 eV after correcting the contract.
The older fixture is retained and labeled for historical UTA arithmetic.

Matching compares eigenvalue multisets within each configuration/J/parity
block by energy rank, without nearest-energy fitting. This does not certify
state identity through avoided crossings. Common-spinor correlated transition
matrix elements, larger active spaces, MCDHF optimization, experimental
accuracy and correlated magnetic transitions remain to be validated.

## CI Hamiltonians on common integrals and spinors

The follow-up live FAC audit uses the same radial inputs to separate angular
assembly from orbital optimization. It includes a complete two-electron
configuration space through n=3 (98 levels), carbon `1s2 {2s2 2p2, 2s 2p3,
2p4}` (20 levels), and the joint seven-configuration Ne-like n=2/3 space
for Fe and Pb (37 levels each). This includes interconfiguration mixing,
equivalent electrons and both parities.

| Case | Levels | Max eigenvalue difference with FAC integrals (Ha) | With independently integrated FAC spinors and matched c (Ha) |
|---|---:|---:|---:|
| He-like Ne | 98 | 1.6e-13 | 6.3e-11 |
| Neutral C | 20 | 1.1e-13 | 3.0e-11 |
| Ne-like Fe | 37 | 3.2e-12 | 1.4e-9 |
| Ne-like Pb | 37 | 3.9e-11 | 7.5e-9 |

The diagnostic frontend exports one-body and Slater integrals by calling the
unmodified FAC library. Its binary CI matrices provide independent reference
eigenvalues. Atomx then evaluates those integrals with its own angular kernel.
The second check recomputes all 2003 radial integrals through the production
shooting-grid kernels and the public CI builder. The supplied FAC and Atomx
fine-structure constants differ; an explicitly reported analytic change of
the fixed-spinor Dirac expectation aligns c. Unadjusted results are retained.
Individual one-body errors reach 5.4e-8 Ha even when eigenvalues agree more
closely; weak off-diagonal couplings are also checked, not hidden by mixing.

Two numerical issues were isolated:

- FAC `GetYk` caches single-precision Coulomb potentials and fitted tails.
  Its existing `LimitArray(0,0)` control disables that compression while
  retaining double-precision Slater caching. The cache comparison uses
  byte-identical exported spinors. With it disabled, individual Coulomb
  integrals agree within 4.4e-11 Ha in the refined panel.
- The comparison adapter's polynomial interpolation through an appended
  origin distorted the nonanalytic point-Coulomb cusp. Factoring out
  `r**sqrt(kappa**2-(Z*alpha)**2)` before interpolating the regular part removes
  that error. Simply increasing the source grid had made it worse. Production
  SCF orbitals were not changed to imitate the reference.

Raw outputs, failure diagnostics, refinement panels and build sources are in
`benchmarks/runs/2026-09-30-ci-common-integrals/`.
The reusable runner is `benchmarks/bench_reference_ci_operator.py`;
external matrices/integrals and unmodified spinors are retained in
`tests/data/fac_ci_operator.json` and `fac_ci_spinors.npz`.
The focused external/analytic/grid-CI run passes 30 tests. These are spectrum
comparisons within symmetry blocks, not signed matrix-element comparisons
between different CSF phase conventions. They support the exact CSF builder. The separate frozen-core reduction is
checked below. Larger variational correlation spaces, correlated transition
tensors and MCDHF convergence remain outside this evidence.

## Native frozen-core Hamiltonians and virtual orbitals

The frozen-core line pipeline now uses the same radial Coulomb evaluator as
full CI. Bound photons use native SLAG6 weights or B-spline coefficients;
changing the continuum grid cannot change a bound level or photon integral.
Closed-core direct/exchange contraction lives in `physics/csf/frozen_core.py`,
independent of the radial engine. The analytic magnetic-substate sum is tested
for 25 combinations of s, p and d relativistic shells, including off-diagonal
radial one-body elements. Partially filled or overlapping core/active spaces
are rejected.

The audit found three defects:

- The line pipeline interpolated bound spinors onto a continuum grid and
  used second-order Coulomb/photon quadrature. On common FAC spinors, switching
  its core and active interactions to the native operator reduces the refined
  Ne-like Fe/Pb excitation differences to about 1.6e-9 eV. All 37 levels per
  ion and their J inventory are checked. FAC bare Dirac elements hold the
  physical constant fixed; this comparison isolates Coulomb and core folding.
- The local-potential one-body reconstruction assumed `V_nuc=-Z/r`, including
  when the actual nucleus was finite. Shooting results now retain and serialize
  the sampled nuclear/environment field. At Z=82, point/Fermi cases with an
  external field agree with the independent differential Dirac matrix within
  8e-8 Ha. Old checkpoints remain readable, but cannot provide this bound-CI
  reconstruction when the actual field was not saved.
- The B-spline virtual Fock builder omitted the mixed exchange blocks and
  the opposite-component sources in each direct block. The repaired J/K
  matrices contract to the complete scalar Slater integral. Native occupied
  and virtual orbitals now satisfy their shared metric: the maximum Gram
  defect on Ne-like Z=10,26,82 is 1.1e-10. Frozen-core/full-CI excitation
  energies agree within 2.2e-11 Ha on that finite-nucleus panel.

The four-ion FAC panel was rerun with the same explicit frozen-CI contract:

| Z | Previous maximum level difference (eV) | Native bound integrals (eV) | Maximum finite-photon A difference |
|---|---:|---:|---:|
| 18 | 0.399 | 0.374 | 3.77% |
| 26 | 0.409 | 0.367 | 1.61% |
| 54 | 0.419 | 0.314 | 1.22% |
| 82 | 0.515 | 0.328 | 0.242% |

These independently optimized orbital models still disagree. Rate agreement
does not uniformly improve when the numerical error is removed; the corrected
common-spinor tests, rather than closeness alone, establish the repair.
Full run times remain about 0.79–0.87 s for Atomx and 2.05–2.14 s for FAC,
with different grids and stopping criteria. Bound-line work is about 45–52 ms;
these are not equal-accuracy algorithm speed ratios.

The shooting CI evaluator now reuses pair densities and their SLAG6 Coulomb
potentials. It preserves the original discrete source/target ordering.
`benchmarks/bench_ci_pair_potentials.py` checks every requested integral against
scalar recomputation before reporting time; differences are below 1.5e-13 Ha.
It measures a fresh radial build per repetition, separately from SCF and CI
angular assembly. The four batches contain 30–1400 integrals and 11–56 unique
source potentials. See the retained timing samples for the kernel speedups.

Evidence and reproduction: `benchmarks/runs/2026-09-30-frozen-core/`.
The final focused run passes **164 tests**, with three full-MCDHF transition
checks excluded. It includes independent FAC fixtures, analytic CI, core
contractions, native finite nuclei, persistence, B-spline J/K and MBPT checks.
It is not a full-suite result or an experimental-accuracy certification.

The DR driver now folds the appropriate inactive core separately for each
hole manifold. The subsequent DR audit below replaces the historical
spectator-energy photon approximation and checks the continuum separately.

## DR final-state CI and continuum repair (2026-09-30)

`ResonanceManifold.radiative_channels` diagonalizes the filled-core plus
spectator final space, including radial mixing within each kappa. Photon
energies are initial-minus-final CI energies from the same Hamiltonian.
Each line uses the finite-frequency Babushkin operator and coherent
determinant amplitudes; `dr_levels` retains the resolved channels. A direct
construction with only dressed orbital energies is rejected for photons.
The static operator remains an explicit diagnostic. The shared determinant
one-body action retains small amplitudes and complex interference.

The matched FAC panel includes **237 initial levels and 393 nonzero E1 lines
per ion and photon mode**, for Ne-like Z=26 and 82. The four hole/parity CI
spaces are mixed separately; all five n=3 final levels are included. Point
nuclei, infinite mass, corrections off, and no configuration relaxation
shifts are used. Every FAC transition call contains one upper and one lower
state, so it uses the individual photon frequency.

On identical FAC spinors and bare one-body integrals, native Atomx Coulomb CI
and photons agree with all **1,572 printed FAC rates within 5.1e-7 relative**.
FAC's default small-coefficient cuts initially caused weak Pb widths to
disagree by up to 2.08%. Disabling `SetMixCut` and `SetAngZCut` leaves the
spinors and level energies unchanged and removes that discrepancy. These
are operator checks in selected spaces, not experimental accuracy claims.
For Pb, individual finite/static total radiative-width ratios span
0.462–1.144; the ratio of summed widths is 0.97635.

The complete native Fe calculation exposed two continuum defects:

- VINT provides a dedicated final interval for constant `r*V`. The continuum
  caller clipped its interval index one entry too early and extrapolated the
  preceding cubic into the outer box. This created a growing artificial
  potential, incorrect wavefunctions, and excessive adaptive stepping.
  Selecting the actual Coulomb-tail interval repairs both physics and speed.
- Auger integrals used a persistent cache without continuum energy in its key.
  They now belong to one energy/potential evaluation and cannot leak between
  resonance calculations. A manufactured-wave test failed before the repair
  (a doubled wave left the rate unchanged), and now gives the required factor
  of four and independence from call order.

A screened-Z=26 continuum agrees in radial shape with independent DOP853
integration at relative L2 errors 1.7–2.5e-10 for kappa=-1,+1,-3. This check
fits an overall amplitude and therefore does not certify energy normalization;
separate asymptotic and stored FAC wave tests cover that aspect. One profiled
Fe Auger evaluation fell from **29.890 s to 0.0163 s**, including first-use
cache loading. This is a specific defective-field case, not a general solver
speedup claim. No tolerance was relaxed.

The regenerated native Fe panel uses independently optimized local ground
potentials, the same configuration inventory and the corrected continuum:

| Quantity | Atomx versus FAC |
|---|---:|
| Maximum absolute resonance energy difference | 0.3983 eV |
| Sum of Auger widths, ratio | 0.98469 |
| Sum of finite-frequency stabilization widths, ratio | 0.98990 |
| DR rate ratio at 1 / 5 / 10 / 50 MK | 0.98701 / 0.98522 / 0.98503 / 0.98488 |

Individual widths do not share this aggregate agreement. For example, FAC
level 84 has an Auger-width ratio of 0.326; level 42 has 1.863. Independent
orbital/continuum and mixing differences remain to be isolated. Combining
each code's widths through the same DR kernel compares the structure/width
inputs; it does not independently validate the thermal prefactor.

The focused continuum/photon suite and refreshed end-to-end suite pass
**34 tests** in two invocations (30 + 4). This is not a full-suite result.

The native Atomx setup plus all four groups took about 3.92 s, versus 4.49 s
for the FAC process including its exports. Different grids, implementation
stages and first-use overhead prevent an equal-accuracy speed ranking.
The refreshed end-to-end regression uses an independent matched FAC fixture
and unity-centered 2% aggregate gates, replacing historical bias-centered
bands. This benchmark exercises native manifold construction and `dr_levels`.
The following public-API audit removes its separate B-spline construction.
Higher capture shells and excited residual-ion Auger channels are not covered.

Raw decks, outputs, timing profiles and reproduction notes are retained in
`benchmarks/runs/2026-09-30-dr-radiative/`. Tests retain independent FAC
photon data and `fac_fe_dr33_matched.json`; the common-orbital panel reuses
`fac_ci_spinors.npz` without duplicating the large arrays.

## Public DR integration and heavy-ion sampling (2026-09-30)

The public `dr_calculate` path now delegates to `manifold_from_configs`.
Duplicated B-spline interpolation, virtual selection and core contraction were
removed. It reuses the solved native local shooting potential or B-spline Fock
basis without another SCF solve or an engine change. Nonlocal shooting HF
resonance virtuals are not implemented; that input now raises an explicit
error rather than failing later on absent B-spline attributes. This is a
remaining capability gap, not an approximation selected automatically.

The driver bounds the largest resonance energy using the Hermitian CI matrix
in the M=1/2 determinant block (which includes every half-integer J). The
Gershgorin upper bound sizes the continuum interpolation grid; bound-state
integrals stay on the native representation. Fe's initial 1-keV sampling
budget is sufficient. Pb requires 5.43–6.38 keV and 10,944–11,683 points,
depending on the hole/parity space. Each calculated outgoing-electron energy
is checked against its selected sampling budget in the heavy-ion regression.
This adapts the interpolation grid, not the outer matching radius.

Resonance IDs include the hole/parity space, J and level index. The previous
truncated dominant-configuration labels could collide and overwrite the
per-resonance thermal breakdown. The new breakdown sums to the total, and
the ground-channel Auger width is retained explicitly on each state.

Fresh external FAC runs and the actual public Atomx API give:

| Ne-like ion | Levels | Max energy difference | DR rate ratio range | Native SCF / DR time | FAC process time |
|---|---:|---:|---:|---:|---:|
| Fe, Z=26 | 237 | 0.3983 eV | 0.98488–0.98701 | 2.128 / 1.683 s | 4.312 s |
| Pb, Z=82 | 237 | 0.2963 eV | 0.99672–0.99725 | 0.752 / 2.170 s | 4.474 s |

Fe temperatures are 1/5/10/50 MK; Pb uses those temperatures multiplied by
(82/26)^2. These are finite-frequency E1 rates, the closed-shell ground Auger
channel and the same four separately mixed dr33 spaces. Point nuclei,
infinite mass and corrections off isolate the comparison. The FAC runs
disable configuration shifts and coefficient pruning. Individual widths
remain more sensitive than these summed rates. The timings include unequal
grids, different output work and first-use overhead, so they do not rank
the solvers at equal numerical accuracy.

The public Fe result equals the independently exercised manifold result
within 2e-12 relative, with unchanged occupied orbitals and unique state IDs.
The current benchmark CSV uses the matched independent FAC fixture and
unity-centered 2% gates. Historical fixtures remain unchanged. Pb has a
separate independent FAC fixture and 1% aggregate gate. Raw outputs,
source hashes, controls and replay script are in
`benchmarks/runs/2026-09-30-dr-public/`.
The final focused public-DR, hook, reference and frozen-core suite passes
**66 tests**. This is not a full-suite or full-periodic-table certification.

## Dirac–Coulomb continuum normalization (2026-09-30)

Continuum shooting now matches both spinor components to regular and irregular
Dirac–Coulomb functions. Their exact determinant determines the scattering
phase and delta-energy normalization. The former WKB amplitude estimate and
nonrelativistic Coulomb phase are removed. The numerical reference lives in
`numerics/shooting/dirac_coulomb.py`; the existing Coulomb/Slater integral module
retains its separate purpose. A coupled inverse-radius expansion is the fast
path; when it cannot meet tolerance, a private 40-digit mpmath context evaluates
the Coulomb-function representation. No global precision or physical model is
changed by that fallback. The integrator is Salvat's adaptive power-series
method, not Numerov.

Live RADIAL `DCOULF` evidence:

- 160 reference pairs cover charges 0/1/17/73, energies
  0.001/0.1/10/150/7500 Ha, kappa -1/+1/-3/+5, and radii 5/75 bohr.
  Maximum complex-spinor relative difference is 5.11e-12 against the
  unmodified supplied RADIAL source and its original speed of light.
- A second isolated RADIAL build changes only its `SL` constant to Atomx's
  CODATA-2022 value. Its reference-pair difference is at most 2.98e-12.
  Without matching constants, high-energy integrated-wave differences can
  reach 4.71e-7 of asymptotic amplitude; these are not discretization errors.
- 96 actual shooting solves cover charges 0/1/17/73, energies
  0.001/0.1/10/7500 Ha, kappa -1/+1/-3, and boxes 5/75 bohr. At 420 accepted
  external sample points, maximum wave error is 6.61e-10 of asymptotic
  amplitude. No amplitude fitting is used; only the arbitrary real sign is
  aligned. Maximum pure-Coulomb phase error modulo pi is 6.51e-10 radians.
- DCOULF fails or reports an error above 1e-8 at 60 inner sample points, mostly
  low energy. Raw output and rejected points are retained. They are missing
  external evidence and are not counted as successful comparisons.
- After JIT warmup, the 96 solves on 4,000-point grids have median 0.89 ms and
  maximum 15.0 ms in this panel. The fallback costs more than the asymptotic
  series. These timings are not a RADIAL/Atomx speed ranking: DCOULF evaluates
  reference values while the Atomx measurement integrates full waves.

Using the same exported FAC bound spinors and continuum field isolates the
continuum and Auger contraction from independent orbital optimization:

| Ne-like ion | Sum Auger rate / FAC, former WKB | Exact Coulomb matching | Largest individual relative discrepancy, rates >1e8/s |
|---|---:|---:|---:|
| Fe, Z=26 | 0.99998595 | 0.99999483 | 0.0761% |
| Pb, Z=82 | 0.99994907 | 0.99999363 | 0.00505% |

Each ion includes all 237 dr33 levels. The common grid has 16,000 points;
continuum shooting uses its existing default. Refinement of the five previously
worst widths uses 8,000/16,000/32,000 common points and
6,000/12,000/24,000 shooting points. At the finest pair, maximum discrepancies
among those selected widths are 0.0125% (Fe) and 0.00339% (Pb). Residual errors,
FAC output precision, cancellation-sensitive amplitudes and independent
interpolation remain; the new normalization does not make every width exact.

The public DR driver was also rerun against fresh FAC output with independently
optimized local orbitals: rate ratios are 0.984886–0.987014 for Fe and
0.996756–0.997289 for Pb at the previously specified four temperatures. Thus
removing the continuum normalization bias does not remove the larger
orbital/mixing contribution to the public comparison. Point nuclei, infinite
mass, corrections off, resolved finite-frequency E1 stabilization and the
closed-shell ground Auger channel define this comparison.

The affected continuum, Auger, DR and photoionization suite passes **314 tests**;
the stricter phase tests separately pass **19 tests**. This is not a complete
scattering/property audit. One public rerun failed while serializing an old
Numba cache; fresh-cache and subsequent warm-cache runs succeeded with bitwise
identical rates. Warm-cache DR times are 1.707 s (Fe) and 2.170 s (Pb); these
are comparable to the earlier 1.683/2.170 s baseline, not a measured speedup.
Raw failures are retained.
Reproducible probe source/build controls, raw output, all numerical cases and
logs are in `benchmarks/runs/2026-09-30-dirac-coulomb/`; the shared-FAC baseline
and refinement are in `benchmarks/runs/2026-09-30-dr-continuum-common/`.
`benchmarks/bench_reference_dirac_coulomb.py` reproduces the live RADIAL panel.

## Screened continuum and dense radial sampling (2026-09-30)

The full RADIAL `DFREE` routine now provides an independent check of screened
potentials, not just analytic Coulomb reference functions. The matched model is
`r*V = -q - (Z-q)*exp(-r/a)`, with (Z,q,a) = (10,0,1), (26,17,0.3),
(82,73,0.1); energies 0.01/10/150 Ha and kappa -1/+1/-3/+5. Only RADIAL's
speed-of-light constant is changed. Both codes receive the same potential
table and requested output radii; normalization differs by the explicitly
applied delta-energy factor and an arbitrary real sign.

The scalar and batch continuum APIs now evaluate `interp_grid` with dense
power-series output. `numerics/shooting/dirac_sampling.py` reuses `DIR0`
coefficients for every requested point within an interval; unconverged long
intervals use the existing adaptive integrator. The regular origin powers
are retained. When even the first interval requires subdivision, a smaller
regular seed is propagated outward and its scale matched to the stored
endpoint. Backward propagation towards the singular origin is avoided.
An independent spherical-Bessel stress test exercises this path in a constant
deep potential with many oscillations in the first interval.

This removes a separate error source: linear interpolation of accurate
continuum waves in Auger, excitation and ionization matrix elements. It does
not remove interpolation of the supplied potential or observable quadrature
error. Batch output now returns the actual requested grid alongside its wave
arrays. Out-of-domain queries raise rather than clamp the wave to an endpoint.
A failed batch wave raises with its energy, kappa and original cause rather
than returning `None` without its diagnostic.

Live matched DFREE results, 1,000 potential/propagation points and 2,000 output
points, three warmed repetitions per native case:

| Screened model | Largest wave error / asymptotic amplitude | Median Atomx full-solve wall time | Median RADIAL VINT+DFREE CPU time |
|---|---:|---:|---:|
| Z=10, q=0 | 1.16e-8 | 0.230 ms | 2.944 ms |
| Z=26, q=17 | 1.11e-10 | 0.288 ms | 3.285 ms |
| Z=82, q=73 | 1.52e-10 | 0.298 ms | 3.532 ms |

There are 36 cases. Maximum phase difference modulo pi is 1.16e-8 radians.
The Fortran timing excludes process startup and text output; native timings
include sampled-potential checking, propagation, normalization and dense
sampling. The table describes this task and these controls, not overall code
speed. Initial calls and all repetitions are retained in the artifact.

With common FAC orbitals/potential, the 237-level Fe Auger sum changes from
0.99999483 to 0.99999954 times FAC; its worst individual rate above 1e8/s
changes from 0.0761% to 0.0116% discrepancy. Pb's sum is 0.99999450 and its
worst individual discrepancy is 0.00467%. The five-width refinement panel
compares 500/1,000/2,000/4,000/12,000 shooting points at a fixed 16,000-point
observable grid. Relative changes from 2,000 to 12,000 are at most 2.54e-8
(Fe) and 3.67e-9 (Pb). The paired median 12,000/2,000 runtime ratios are
2.01 and 1.89. This supports a smaller propagation grid for these cases;
it does not establish a universal point count. Global defaults are unchanged.

The box study compares radii 2/5/10/20/50 bohr against 100 bohr, checking both
inner-wave normalization and phase. Neutral Z=10 still has a phase difference
3.64e-7 radians at 20 bohr and 0.00476 radians at 10 bohr; the charged Z=26
model reaches about 1.90e-10 at 10 bohr, and Z=82 about 1.04e-10 at 5 bohr.
Small boxes can also make Coulomb reference evaluation slower by requiring its
high-precision fallback. A future automatic radius choice must account for
these distinct effects and the radial extent required by the observable.
This panel does not implement or certify automatic continuum box selection.

The public Fe/Pb DR rerun retains all 237 levels and gives thermal ratios
0.984887–0.987014 / 0.996757–0.997290 against fresh FAC. It confirms that the
larger independent-orbital/mixing differences remain. The broad affected
suite passes **442 tests**, with **3 slow tests deselected** and **1 existing
expected failure** (stored high-energy EII/FAC partial-wave discrepancy).
Stored EII panel gates are not a fresh external EII validation. Initial origin
sampling failures were fixed, not hidden by loosening gates. The separate
slow EII mesh/box refinement regression also passes.

Reproduction: `benchmarks/bench_reference_screened_continuum.py`; raw inputs,
DFREE probes, build commands, box study, FAC comparisons and logs are in
`benchmarks/runs/2026-09-30-screened-continuum/` and the preceding common-FAC
artifact. That panel also identified an incorrect nuclear boundary convention
in the former distorted-wave potential builder. The subsequent correction
and separate EII comparison are described below.

### Shared collision fields and native bound sampling (2026-09-30)

The former collision potential interpolated the total nuclear-plus-Hartree
field and clamped it below the first quadrature point. For point-nucleus He+
its `rV` approached zero at the origin and reached -3.22327 at twice the
first Gauss radius, instead of -2. DW excitation, EII and the affected Auger
paths now use the shared mean Hartree factory. It preserves the solved
nucleus separately and scales the complete electronic density by `(N-1)/N`.
The point-nucleus one-electron field is exactly Coulomb; tests also check
finite nuclei, both supported radial representations and charge accounting.
Explicit EII potential models now affect the actual calculation, with matching
provenance. This does not make mean Hartree equivalent to FAC's general
optimized field.

A fresh FAC panel removes that ambiguity by comparing hydrogen-like ions:
both fields have `rV=-Z` to floating-point precision. It uses point nuclei,
corrections off, and incident energies 1.5/2/5 times the native binding energy.
Native shooting uses 3,000 source points, 25 energy-sharing nodes and cutoff
10; FAC uses DW mode with incident cutoff 36 and ejected cutoff 4. Partial
wave spaces and integration rules therefore still differ.

| Ion | Atomx/FAC at 1.5 I | At 2 I | At 5 I |
|---|---:|---:|---:|
| He+ | 1.01404048 | 1.01373270 | 1.00792334 |
| Fe25+ | 1.01300027 | 1.01238363 | 1.01067941 |
| Pb81+ | 1.00591107 | 1.00493640 | 1.01010679 |

These values at this stage use direct evaluation of the native bound B-spline
expansion. The former Gauss-sample interpolation clamped the EII orbital at
the origin; an audit found relative L2 transfer errors 3.43e-4, 4.23e-4 and
1.04e-3 for these ions. DW excitation now uses the same numerical evaluator,
with zero extension beyond each orbital's own box. Independent Bernstein
polynomial tests check both kinetic-balance orders, endpoints and rejection
of extrapolation. The EII rate changes from native bound sampling are under
8.9e-5 relatively, so this correction alone does not explain the FAC gap.

Before the bound-sampling correction, separate He+/Pb81+ refinements found:

- Doubling the shooting source grid from 3,000 to 6,000 changes cross sections
  by at most 2.86e-10 relatively.
- Increasing energy-sharing nodes from 25 to 97 changes them by up to 0.221%.
  Fixed endpoints and the gap between the logarithmic and linear grids remain;
  this is not proof of convergence of the full energy integral.
- FAC discrepancies persist after those refinements. Bound-orbital accuracy,
  radial integration, energy endpoints, partial-wave tails and reference
  refinement remain separate checks.

Timings are retained per case: the initial native EII calls took 6.8–15.5 s
and the FAC processes 21.9–24.1 s. These timings have different scopes and
partial-wave settings, and include concurrent work; they are not a solver
speed ranking. The stored helium-like EII snapshot was refreshed only in its
native/derived columns. External FAC records and scientific gates are
unchanged. Current native/FAC RMS on that stored panel is 0.607%; it is
distinct from the fresh hydrogen-like panel above.

Final checks pass **108 tests** across the sampling, field, stored FAC,
ionization, excitation and selected DR suites, with **3 slow tests deselected**
and **1 existing expected failure** for the stored high-energy EII/FAC
partial-wave discrepancy. No external agreement gate was loosened.

Reproduction and raw evidence:
`benchmarks/bench_reference_coulomb_eii.py`,
`benchmarks/bench_coulomb_eii_convergence.py`, and
`benchmarks/runs/2026-09-30-mean-hartree/`. The latter retains FAC decks,
raw tables, executable/source hashes, old snapshots, convergence results
and verification logs.

### Full EII energy interval and partial-wave audit (2026-09-30)

The preceding EII table used the old truncated energy-sharing rule and an
automatic geometric tail. Both contracts have now changed. The default
integrates the full half-sharing interval with `E1=I*sinh(x)^2` Gauss-Legendre
quadrature and returns the explicit partial-wave sum. A capped geometric
tail is available only by `partial_wave_tail="geometric"`. The result
separates computed and estimated contributions; neither a finite result nor
the estimator's decay flag certifies partial-wave convergence.

Independent linear/squared Gauss maps confirm that the earlier endpoint
cutoffs omitted a contribution. At twice threshold, full integration raises
the tested He+/Pb81+ rates by about 0.6%. Analytic tests cover constants,
half-integer moments (including an integrable inverse-square-root endpoint)
and a logarithmic integral over widely separated interval/scale ratios.
With the initial tail prescription held fixed, 16-to-32-node changes over
1.5/2/5 times threshold are at most 9.10e-5 (He+), 2.27e-6 (Fe25+) and
2.52e-7 (Pb81+) relatively. This is panel-specific quadrature evidence.
Single-energy and multi-energy calls may use different radial meshes; their
quadrature differences must not be mistaken for energy-sharing errors.

The high-energy Pb81+ case at 20 times threshold exposed a separate defect:
the former EII tail jumped to zero when the last partial waves began rising.
Doubling sharing nodes from 16 to 32 changed the explicit sum by 1.12e-7
relatively but changed the reported total by **4.40%**, entirely through the
tail. The shared continuous capped estimate reduces that numerical variation
to 1.12e-7, but supplies about 40% of its own completed total and remains
outside its decay regime at every node. It is consequently opt-in, not a
default correction. The default computed value for this cutoff-10 case is
about 1.0289476e-24 cm2; a larger partial-wave space is still required.

FAC source inspection identified another unmatched control: `IONMAXK=6`
means **multipole rank 3**, independently of `SetCIPWOptions`. Raising that
rank while keeping the ejected cutoff 4 changes FAC rates by 1.65–1.91%
(He+) and 0.83–1.45% (Pb81+) over the panel. Raising the ejected cutoff to 8
and rank ceiling to 9 gives this comparison with Atomx's explicit cutoff-16
sum (FAC incident cutoff remains 36):

| Ion | Atomx/FAC at 1.5 I | At 2 I | At 5 I |
|---|---:|---:|---:|
| He+ | 1.00172756 | 0.99848490 | 0.97581636 |
| Pb81+ | 1.00226136 | 0.99905389 | 0.97979080 |

The remaining high-energy differences are not resolved. Increasing Atomx's
explicit cutoff from 16 to 24 at 5 I changes He+ from 4.78901395e-18 to
4.85334248e-18 cm2 and Pb81+ from 2.47808690e-24 to 2.49790029e-24 cm2.
The corresponding refined FAC values are 4.9077e-18 and 2.5292e-24 cm2.
Further cutoff, radial-grid and reference integration refinement is needed.
The calculations still have different partial-wave spaces.

The stored helium-like regression panel now uses an explicit cutoff 16,
with **0.991% RMS** against its unchanged historical FAC records. Cutoff 10
without the tail gave 4.42% RMS and is retained as a failed convergence stage;
no scientific gate was widened. The unrelated excitation implementation
snapshot was refreshed after the earlier native bound-sampling correction,
which changed its three high-energy values by about 0.11%; it is explicitly
an own-output snapshot, not external evidence.

Numerical ownership is `numerics.quadrature.sinh_squared_quadrature` and
`numerics.extrapolation.geometric_tail_estimate`. Collision drivers choose
the physical interval, binding-energy scale and whether to apply a tail.
Full-precision continuum energy keys avoid rounding positive soft-ejection
energies to zero, and repeated incident energies reuse solved waves.

Final verification passes **94 tests**, with **one slow test deselected**
and **two existing expected failures** for the unresolved high-energy
EII/FAC and excitation/Born partial-wave comparisons. This includes the
unchanged external agreement gates and the refreshed implementation snapshots.

Raw FAC outputs, input decks, per-node quadratures, failed intermediate
results, source hashes and final tests are retained under
`benchmarks/runs/2026-09-30-eii-quadrature/`. Runners are
`bench_eii_energy_sharing.py`, `bench_fac_eii_partial_waves.py`, and
`bench_coulomb_eii_convergence.py`. Earlier artifact directories remain
historical evidence and do not describe the new default tail contract.

### Reference integration refinement and Slater loop layout (2026-09-30)

Increasing the native explicit cutoff to 32 at 5 I gives 4.860742671e-18 cm2
for He+ and 2.499086642e-24 cm2 for Pb81+. These retain the full energy interval,
16 sharing nodes and no tail completion. Their remaining differences from
the preceding FAC panel required a reference integration audit.

FAC's supplied source uses 12 directly evaluated energy-sharing nodes,
interpolated onto 128 integration nodes. Isolated diagnostic executables
replace only `NINT0=12` by 24 or 48; the reference source tree and library
remain unchanged. A rebuilt 12-node control reproduces every printed cross
section of the original executable. For He+, with the same rank-9/ejected-8
and incident-36 settings, the cross sections are:

| FAC direct sharing nodes | 1.5 I (cm2) | 2 I (cm2) | 5 I (cm2) |
|---|---:|---:|---:|
| 12, rebuilt control | 3.9744e-18 | 5.1545e-18 | 4.9077e-18 |
| 24 | 3.9713e-18 | 5.1406e-18 | 4.8737e-18 |
| 48 | 3.9711e-18 | 5.1398e-18 | 4.8708e-18 |

Thus much of the earlier ~1% difference came from the reference integration.
The native cutoff-32 / refined FAC ratio at 5 I is **0.99793518**. The
24-to-48 reference change is still 0.0595% there. Independently, evaluating
every FAC incident partial wave through 36 instead of interpolating its
sparse grid changes the original 12-node result by up to 0.0530%. These
controls were varied separately; their effects were not combined by hand.
The angular spaces and soft-ejection endpoints still differ, and neither
calculation is certified fully converged. The 24/48 sharing-node reference
study covers He+ only.

Profiling the native cutoff-32 runs identified the shared batched Slater
kernel as the main radial cost. It formerly interleaved consumer updates at
each radius, repeatedly writing output scalars and reading strided density
rows. The new layout computes one Y row into the existing suffix scratch
array, then performs each consumer dot product with contiguous reads and a
local scalar sum. Each accumulation retains its original radial order;
there is no extra full matrix, `fastmath`, pruning, or quadrature change.

A captured Pb81+ batch has 15,613 source rows, 31,177 consumer slots and
80,684 radial points. All output bits match the former implementation at
1/4/8 threads. Alternating seven-repeat median speedups are **1.33/1.32/1.19x**
respectively. A separate full EII comparison at cutoff 16, eight sharing
nodes and four Numba threads gives **1.18x** warm median speedup, with
identical cross sections in every paired run. Concurrent reference work and
thread-count effects are recorded; these are specific-case measurements.

Independent dense Coulomb Green matrices check nonuniform-grid weights,
endpoints, support cuts, multiple consumers and missing rows. Focused
kernel/angular and collision/FAC suites pass **75 distinct tests**, with
one existing expected failure for the high-energy EII partial-wave gap.
The current support selector retained nearly the whole captured radial grid;
a better justified support bound remains an optimization opportunity.
An additional read-only Racah-recoupling audit matched 3,552 existing angular
coefficients within 2.34e-15 absolute. Production angular code was unchanged
at that stage; the implementation and high-angular-momentum audit follow below.

All diagnostic sources, exact build commands, executable hashes, input
decks, raw tables, the captured radial batch, old kernel, profiles and
timing/test logs are retained in
`benchmarks/runs/2026-09-30-eii-refinement/`. The replay driver is
`benchmarks/bench_fac_eii_integration.py`.

## Coupled Coulomb algebra and Wigner stability (2026-09-30)

CE and EII now share `angular.coefficients.coupled_coulomb_direct` and
`coupled_coulomb_exchange`. A single Racah 6j contraction replaces each
magnetic-sublevel sum. The exchange coefficient includes the coupled-state
permutation phase; the antisymmetrized amplitude remains `D R_d - E R_e`.
The reduced-C convention, radial integrals, channel cutoffs and quadrature are
unchanged. Auger and CI photon code import the shared `cg`/`ckm` directly;
the four former private DW angular helpers/aliases were removed.

Testing higher angular momenta uncovered a separate numerical defect: long
alternating floating Racah sums lost digits. In 420 independently generated
3j and 250 6j comparisons with exact SymPy values, the old maximum absolute
errors were 8.45e-7 and 7.52e-11. They are now **7.13e-15 and 1.59e-15**.
Long or strongly cancelling sums use exact integer binomial/multinomial
arithmetic before a scaled square root. A separate test retains a nonzero
representable symbol whose square underflows. Short, well conditioned sums
remain compiled, and the former log-factorial routine now actually reads its
precomputed table instead of recomputing logarithms. Compiled callers reach
the cached integer fallback through Numba object mode.

The final 121,204-call 1s-ionization angular workload at orbital cutoff 24 is
**19.7x faster** by alternating three-repeat medians. Its maximum absolute
coefficient difference from the former implementation is 3.52e-12. Independent
magnetic sums at multiple M projections, monopole identities, Pauli
cancellation and exact high-angular-momentum values check the new reduction.

| Pb81+ EII timing, 5 times threshold | Before | After |
|---|---:|---:|
| Full calculation, angular caches cleared | 22.719 s | 20.953 s |
| Full calculation, angular caches retained | 20.935 s | 21.194 s |

These are three-repeat medians with unchanged radial kernels, cutoff 24,
eight sharing nodes, 3000 shooting source points and four Numba threads.
Every cross section is exactly 2.497901127143586e-24 cm² in these runs.
The complete calculation improves 1.084x when coefficients must be rebuilt;
there is **no demonstrated warm full-calculation speedup**. Radial work still
dominates. The full timings precede the final square-root scaling guard;
`wigner_timed.py` preserves that exact source. The final angular timing and
operator tests include the guard, with source hashes recorded separately.
Final-code full EII reruns at cutoff 32 and 16 sharing nodes reproduce the
preceding He+/Pb81+ results within 5e-16 relative (He+ exactly). The residual
He+/refined-FAC difference at five times threshold remains 0.2065%; this
algebraic optimization does not remove the remaining convergence gap.

A fresh external FAC panel on identical point-Coulomb spinors at Z=10 and 82
checks **48 signed static/finite-frequency transverse values**, including
off-diagonal elements. Maximum relative error is **4.98e-8**. This is an
operator comparison on shared spinors, not many-electron spectral accuracy
or finite-frequency orbital relaxation. The diagnostic frontend containing
`DumpSpinor` must be used for `--common-spinors`; the runner now selects that
frontend by default for this flag. The initial incompatible-frontend failure
is retained separately and contributes no comparison values.

The angular/collision/CI suite passes **218 tests**, with two existing expected
failures. The final angular/Breit/photon/DBSR suite passes **116 tests**;
there are **302 distinct passing tests** across the two suites. FAC stored
reference gates are unchanged. Fresh external data, exact-symbol cases,
historical source snapshots, timing scripts, source hashes and test logs are
in `benchmarks/runs/2026-09-30-coulomb-recoupling/`. The broader remaining
scope below is not certified by these checks.

## Stable radial Coulomb kernels and observable meshes (2026-09-30)

Refining only the collision integration mesh, while keeping the continuum
propagation mesh fixed, exposes a separate second-order quadrature error.
The arbitrary-grid Y/Slater kernels also had a dynamic-range failure:
rank 64 on a 1000-point mesh from 1e-7 to 80 a0 produced 254 nonfinite
potential samples, although the underlying Coulomb Green matrix is finite.

The new kernels use block-local anchor radii. With `x=r/a`, their factors
are `x**k` and `x**(-k-1)/a`; the carried moments are rescaled by bounded
anchor ratios between blocks. The trapezoidal Green action is unchanged.
This removes absolute-radius power overflow without truncating multipoles.
Independent dense Green matrices verify ranks 0 through 256, symmetry under
exchange of the two densities, batch/scalar agreement, and writable-grid
cache safety. The two endpoint weights now come from
`RadialMetric.trapezoid`; `np.gradient(r)` had doubled them. The unbounded
relative-density tail heuristic was removed, and exact-zero support retains
the following endpoint's half interval.

The first stable, point-by-point recurrence was slower; retained benchmark
artifacts document that rejected candidate. Block rescaling restores the
simple inner accumulation. Explicit nonnegative indices avoid per-load
negative-index adjustments in the compiled block loops. On the captured
15,613-source, 80,684-point Pb81+ batch:

| Measurement | Before | After |
|---|---:|---:|
| One thread, median of five alternating repeats | 3.924 s | 3.108 s |
| Four threads, median of five alternating repeats | 1.134 s | 0.917 s |
| Cached geometry | 80.684 MB | 32.274 MB |

This is **1.26/1.24x faster**, with **60% less cached geometry**. The maximum
absolute output difference divided by the maximum output magnitude is
4.94e-16. In a full Pb81+ EII calculation at 5I, cutoff 16, eight sharing
nodes, 3000 shooting points and four threads, the three warm paired medians
are **7.625 versus 6.523 seconds (1.17x)**. The largest relative cross-section
difference is 1.12e-16. Both variants use the same corrected endpoint weights
and full spinor support; this isolates the kernel change.

Public CE and EII functions now expose `radial_refinement`, independently
of shooting source points and energy-sharing nodes. The default is one;
integer refinement subdivides each original integration interval, and EII
records the actual mesh and rule. The public runner
`benchmarks/bench_eii_radial_quadrature.py` finds:

| Fixed physics/partial-wave setting | Change from refinement 1 to 4 |
|---|---:|
| He+, cutoff 8, 2I / 5I | -0.0180% / -0.0229% |
| Pb81+, cutoff 8, 2I / 5I | -0.0195% / -0.0227% |
| He+, cutoff 32, 5I, 16 sharing nodes | -0.0189% |

Successive differences have ratios 4.0004–4.0008, consistent with the
second-order trapezoid. Richardson estimates are diagnostic only and are
not substituted for calculated cross sections. For the last He+ case,
refinement 4 gives **4.859823619144986e-18 cm²**, versus the previously
regenerated 48-sharing-point FAC value **4.8708e-18 cm²**. The difference
increases from 0.2065% to **0.2254%**: radial quadrature partly masked it.
Neither the different partial-wave spaces nor FAC's own sharing/endpoints
are fully converged. These measurements do not establish full EII accuracy.

The focused collision, Slater, continuum-control, Born, frozen-core and
determinant-Auger suite passes **134 tests**, with two existing expected
failures. No external reference values or acceptance thresholds were changed.
Source snapshots, prototype failures, final timings, refinement panels,
FAC-reference hashes and test logs are retained in
`benchmarks/runs/2026-09-30-slater-stability/`.

## EII exterior domain and live FAC sharing refinement

The native ionization integration radius can now exceed the bound-orbital
box without modifying the bound expansion. This permits an independent check
of the continuum-continuum Coulomb source outside the localized consumer.
Zero extension describes the supplied finite-box bound model; adequacy of
that bound box must still be checked separately. Dense Green-matrix tests
verify that compact consumer support does not truncate the exterior source.

For He+ at 5I, l max 32, 16 sharing nodes and radial refinement 4, increasing
the collision radius from 50.965 to 400 Bohr changes the cross section by
**+0.00920%**. The 200-to-400 change is **-0.0003016%**; doubling source points
from 3000 to 6000 at 400 Bohr changes it by **-2.473e-10 relative**. The
radius dependence is oscillatory, so these differences are not rigorous
infinite-domain error bounds. At the original radius, l max 32 to 48 changes
the result by **+0.02206%**. The subsequent l max 64 run exposed a dense-output
underflow hang; the repair and successful calculation are recorded below.

A freshly built isolated FAC diagnostic doubles direct energy-sharing samples
from 48 to 96. All three cross sections agree at FAC's printed precision;
at 5I this is **4.8708e-18 cm²**. Its final interpolation grid, small-sharing
endpoint and partial-wave controls were unchanged. The native l max 32,
400-Bohr result is **4.860270667049888e-18 cm²**, still **0.2162%** below FAC.
The native l max 48, original-radius result is **0.2033%** below FAC. Joint
convergence and matched native/FAC partial-wave spaces remain unverified.
Both panels use Coulomb collision amplitudes, without transverse exchange.

The final focused suite passes **78 tests**, with one existing expected
failure; no external values or tolerances changed. Concurrent runs support
no relative speed claim. Sources, hashes, raw FAC data, numerical controls
and logs are in `benchmarks/runs/2026-09-30-eii-convergence/`.

## High-partial-wave dense-output stability and angular memory

A He+ low-energy continuum wave at kappa=-65 underflowed to zero on inner
source nodes. Dense output then required a relative series residual of
`0 < tolerance*0`, causing indefinite step halving. An isolated reproduction
and stack dump identified this mechanism; the hanging jobs were interrupted.
Dense sampling now resolves the local homogeneous Dirac solution at unit
amplitude before restoring its scale. A larger right endpoint is used when
the left endpoint cannot represent the requested relative error budget.
Exactly zero middle-interval initial data return the zero solution, while
the regular origin boundary remains distinct. Nonadvancing adaptive steps
raise rather than loop indefinitely. No weak partial wave is discarded.

The formerly hanging dense sample completes in milliseconds. The first
successful full l max 64 He+ run completes in **113.035 seconds**, producing
**4.860931751470222e-18 cm²** at 5I. The l max 48-to-64 increase is
**0.0007439%**, leaving a **0.2026%** gap below the FAC 96-sharing-point
result. The l max 32 result is unchanged. The remaining gap is not attributed
to transverse exchange: both benchmark collision operators are Coulomb-only.

Bounding the shared coupled-Coulomb LRU caches to 131072 entries each reduces
peak process RSS from **376268 to 296776 KiB (21.1%)** for a 926692-request
l max 48 angular panel. Values are bitwise identical and direct-coefficient
cache hits are unchanged. Enumeration times, 1.554 and 1.552 seconds, show no
speed improvement beyond noise. These measurements isolate angular work;
they are not full-calculation memory or speed ratios.

The final focused suite passes **440 tests**, with one existing expected
failure. A live 36-case screened RADIAL DFREE panel retains maximum wave and
phase errors of **1.15e-8** (wave error measured relative to asymptotic
amplitude). Tiny high-kappa Coulomb tests additionally check relative wave
accuracy with no absolute tolerance. Sources, failure traces, final tests,
live inputs/output and per-run hashes are in
`benchmarks/runs/2026-09-30-continuum-barrier/`. FAC radial-grid refinement is
being checked independently of energy sharing and partial waves.

## Bounded EII wave storage and FAC radial-grid uncertainty

EII now consumes each outgoing energy pair at its sharing node and releases
unused continuum waves. Only the incident energy and current pair are kept;
repeated incident energies reuse the completed shell cross section and
diagnostics. Exact energy keys, continuum equations, quadrature and radial
contraction order are retained. No wave amplitude cutoff is introduced.

Six alternating fresh-process runs at l max 24, radius 200 Bohr, refinement 4
and 16 sharing nodes give median peak RSS **1117656 -> 537540 KiB (-51.9%)**.
Median times are **7.8320 -> 8.1534 seconds (+4.1%)**, so this is a memory
improvement with a measured runtime cost. All cross sections are bitwise
identical. Increasing sharing order to 32 keeps RSS at **541452 KiB** and
changes the cross section by approximately **-6.59e-6 relative** in this
partial-wave space. A separate combined l max 64, 200-Bohr calculation gives
**4.861409421971861e-18 cm²** at 5I. One FAC reference process was active during
the paired timings; they do not establish general throughput performance.

The live FAC 3000-point radial refinement changes its 5I cross section from
**4.8708e-18 to 4.8575e-18 cm² (-0.2731%)**, larger than the previous native/FAC
gap. The 48/96 sharing-node agreement alone did not establish an accurate
reference. The combined native result is now **0.0805% above** this finer FAC
value. The earlier roughly 0.203% discrepancy must not be interpreted as a
verified Atomx error. The completed 6000-point FAC run gives
**4.8576e-18 cm²**, a further **+0.00206%** at FAC's printed precision.
No joint convergence is claimed.

The reference changes points, ratio and asymptotic density together, leaving
the other collision controls fixed. Its outer endpoint changes slightly
(1568.39 -> 1580.65 Bohr); both fields retain rV=-2 to roundoff. This refines
the complete FAC radial calculation, rather than isolating one integrator.
Native/FAC partial-wave spaces and their remaining integration controls still
need matched convergence checks. Both collision operators are Coulomb-only.

The focused suite passes **51 tests**, with one existing expected failure;
external values and acceptance thresholds are unchanged. Source snapshots,
paired measurements, profiles, tests and reference input/output hashes are
retained in `benchmarks/runs/2026-09-30-eii-streaming/` and the preceding
continuum-barrier panel.

## Continuum worker reuse and EII buffer reuse

Repeated continuum batches can now share a numerical worker context. It owns
thread lifetime and numerical controls, checks the field on every solve and
retains no result arrays. EII supplies the physical requests and reuses its
outgoing-wave buffers. Missing rows are explicitly zeroed in diagnostic mode;
normal failure handling remains strict.

An isolated screened-field panel of 24 four-wave batches improves from median
**12.242 to 10.221 ms (1.198x)**, with bitwise-identical output. In six full-EII
alternating runs, median times are **8.0439 and 8.1114 seconds**, with overlapping
ranges; no full-EII speed gain is established. Peak RSS falls from **536032 to
488328 KiB (-8.9%)** and every cross section is bitwise identical. An external
FAC calculation ran concurrently, so timings are scoped workload observations.

Worker instrumentation records 17 pools becoming one and main-thread submit
time falling from 0.330 to 0.00966 seconds for 1617 requests. Submission overlaps
execution; its reduction is not an equal end-to-end saving. Both entry points
retain the same worker-count policy and radial numerical implementation.

The final focused suite passes **130 tests**, with one existing expected
failure. New checks cover mutable-field refresh, worker cleanup, closed-context
use and stale-buffer prevention after a missing sharing node. Evidence and
source snapshots are in `benchmarks/runs/2026-09-30-eii-workers/`.

## Scope still required for the full objective

| Area | Required comparison contract and evidence | Current status |
|---|---|---|
| Structure across the table and ions | Same charge, configurations, jj/CSF objective, occupations, nuclear profile, constants, correction stages; orbital and basis refinement | Small live panel only |
| Correlation / MCDHF / CI | Same active space, frozen core, state weights, no-pair projector, orbital optimization; convergence as space grows | Live frozen CI and 192-level common-integral/common-spinor CI panels; native frozen-core reduction verified; growing MCDHF spaces pending |
| Breit, QED and recoil | Individual operator shifts, off-diagonal matrix elements and stage of application; isotope mass and charge distribution | Selected tests; broad matched external panel pending |
| Spectra | Matched J/parity/configuration identities; energy differences, wavelengths, uncertainties and completeness | Full live DBSR/RADIAL/FAC/NIST spectrum audit pending |
| Transition properties | E1/M1/E2/M2 and supported higher multipoles; common gauge, normalization, line strength, f and A definitions | Live finite-frequency one-electron E1–E4/M1–M4 panel; broad correlated-state validation incomplete |
| Radial properties | Densities, normalization, moments, orbital conventions and common mesh/metric | Mean radii at coarse reference print precision measured here |
| Continuum | Boundary conditions, phase shifts, energy/flux normalization, exchange and asymptotic charges | Live RADIAL Coulomb reference and integrated-wave normalization panels; common-FAC Auger field comparison; 36 live screened DFREE cases and dense sampling verified; nonlocal continuum exchange and broader channels pending |
| Collision/recombination processes | CE, ionization, PI/RR, AI/DR, R-matrix; thresholds, channel/state IDs, grids, partial waves and detailed balance | Live matched Fe/Pb dr33 public-API and common-orbital photon panels; other processes and broader channels pending |
| Other properties | Hyperfine, g/Zeeman, polarizability/Stark, isotope shifts, PNC/EDM where implemented | Current operator/model audit and external evidence incomplete |
| Plasma and compound calculations | CRM/cascades, rates, average atoms and supported compound channels; consistent inputs and conservation | Not certified by the structure panel or synthetic fixtures |
| Performance | First and repeated runs, threads, compilation, memory, iterations and equal accuracy targets for each mechanism | DF/structure and CI radial-kernel timings; other mechanisms pending |

Not all reference codes implement every mechanism. A missing reference feature
must be recorded explicitly and covered by an appropriate independent physical
identity or another source; it must not be counted as an agreement pass.

## Historical evidence

The previous verification inventory is preserved in
[`archive/2026-09/VERIFICATION_BEFORE_LIVE_AUDIT_2026-09-29.md`](archive/2026-09/VERIFICATION_BEFORE_LIVE_AUDIT_2026-09-29.md).
It records prior fixtures, claims and limitations. Those entries are not fresh
verification of the current worktree. Existing `external_fac` markers can refer
to stored data; only a retained executable run establishes live regeneration.
See [DF solver controls](DF_SOLVER_CONTROLS.md) for the current optimization
measurements and [migration notes](MIGRATION.md) for removed import aliases.

## Shooting source integration and complete test audit (2026-09-30)

Analytic exponential moments replace per-cell Gauss quadrature for the same
six-node Green-source interpolation. Independent high-precision controls
cover all boundary stencils, weak/stiff rates, signed sources, log shifts,
linearity, reciprocity and energy response. The measured warm DF improvements
are 4.2% for Ne and 2.6% for Cf; iterations are unchanged and total energies
change by at most 4.4e-11 Ha. These are selected workloads, not full-table
performance or experimental-accuracy claims.

Live Salvat panels retain 27 DBOUND and 36 screened DFREE cases. The largest
energy difference is 2.3e-13 Ha and continuum phase difference 1.2e-8 radians.
Native Ne E1/M1/E2 transitions agree with the baseline within 1e-11 relative.
CE/EII/PI/RR regression cross sections and screened continuum waves are
identical. These reaction tests explicitly use B-spline bound states and
shooting continua; their coefficient APIs do not yet accept plain native
shooting bound states. They retain limited partial-wave spaces and do not
certify scattering convergence or nonlocal continuum exchange.

The full repository run after the moments optimization completes with
3746 passes, 53 failures, 2 setup errors, 11 skips and 21 expected failures.
Every failure/error reproduces on archived baseline `eb7bdae`. These include
old snapshots, missing retired-script consumers, configuration API assumptions
and physics/property checks; references and tolerances were not changed to
make the audit pass.

The subsequent Schrödinger repair fixes six of those failures: a below-minimum
trial energy now uses RADIAL's inner matching fallback instead of a false box
failure. Failed/unconverged roots return no orbitals, and physical s-wave
origin derivatives replace the outward seed placeholder. All twelve live
varied-start p/d cases now exactly match RADIAL, including seven previously
failed native roots. Analytic P/dP/dr and normalization controls verify s waves
and other angular momenta independently. The final numerical suite passes
161 tests; final H/C/Ne/Cf Dirac energies, orbital energies and iteration counts
are identical to the moments panel. A green full-suite result is not claimed.

The two retired-script reproducibility/bounds tests have also been migrated
to direct Atom API subprocesses and pass on the full panel, including Cf and
explicit MCDHF. CI and nightly workflows now use these tests, retaining their
existing full-panel repeat counts. The final numerical and safety gates total
163 passes and repair eight failures identified by the full audit.

Inputs, outputs, source hashes, baseline failure reproductions and profiles
are retained in
[`2026-09-30-shooting-moments`](../benchmarks/runs/2026-09-30-shooting-moments/README.md).
The [controls documentation](DF_SOLVER_CONTROLS.md#analytic-exponential-moments-in-shooting-source-integration)
separates performance measurements from remaining physics and API gaps.

## Compiled shooting radial operators (30 September 2026)

The derivative stencils, symmetric Dirac energy contraction and prepared
full-box Green action now execute through compiled numerical kernels.
Arithmetic order, SLAG6 integration, physical corrections and root stopping
criteria are retained. The measured warm C/Ne/Xe/Cf solve reductions are
8–11%, with identical total/orbital energies and iteration counts. These
measurements exclude imports, constructors and each process's first solve.

The final focused verification passes 271 numerical/physics/architecture
tests plus two full-panel reproducibility/bounds tests. New independent
analytic controls check boundary stencils, Dirac radial moments and refinement.
All 26 saved observable arrays, continuum phases and seven photon-transition
records are bit-identical to the baseline. Collision controls use explicitly
selected B-spline bound states and shooting continua; they do not establish
native bound-state collision API support or channel-space convergence.

See [the radial operator review](../benchmarks/runs/2026-09-30-shooting-radial-kernels/README.md)
for matched timings, source/output hashes, profiling evidence and requirements
for safe sparse contraction and field reuse. This is focused verification;
the existing full-suite issues reported above were not re-audited.
