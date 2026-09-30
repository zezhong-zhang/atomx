# Shooting analytic-moment audit

Baseline: `eb7bdae`, archived under
`/tmp/atomx-shooting-moments-baseline`. Both implementations use the same
physics, interpolation, radial mesh and convergence controls. Imported
source hashes and thread settings are in each result JSON.

## Numerical change

Green-source prefix weights integrate the existing degree-five interpolation
with analytic exponential moments. A positive series/downward recurrence at
small rates avoids cancellation; an upward recurrence is used above eight.
The boundary stencils, supplied evanescent rates and coarse-cell guard remain.
Tests compare moments and individual weights against independent 60-digit
quadrature, including zero, the recurrence crossover and rate 100. They also
check signed sources, log shifts, linearity, reciprocity and energy response.

## Runs

- `df-before-0`, `df-after-0`, `df-after-1`, `df-before-1`: ordered fresh-process
  H/C/Ne/Cf solves; default Breit + QED, Hamiltonian averaging, four repeats.
  Exclude repeat zero from timing. All iterations and convergence agree;
  energies differ only at the recorded roundoff scale.
- `observables-before-supported`, `observables-after`: native response and
  Ne orbitals/transitions; CE/EII/PI/RR with **explicit B-spline bound states**
  and shooting continuum waves, plus twelve screened phase/wave controls.
  These small partial-wave spaces are regression checks, not converged
  scattering predictions. Plain native shooting bound states are not accepted
  by the current coefficient-facing CE/EII/PI/RR APIs.
- `bound-radial`: 27 live matched Salvat DBOUND cases, including Z=98.
- `continuum-radial`: 36 live screened DFREE cases; 200-Bohr radius, 3000
  source nodes and 2000 dense output radii. Speed of light and input samples
  match. These local-field comparisons do not validate full nonlocal HF.
- `profile-after`: cold/warm Cf profiles; concurrent testing, so do not use
  these wall times for performance claims.
- `full-tests-final.log` and `full-tests.xml`: full repository audit, including
  supported scattering, correlation and property paths, **before the subsequent
  Schrödinger repair**. It finishes with 3746 passes, 53 failures, 2 setup
  errors, 11 skips and 21 expected failures. All 55 failures/errors reproduce
  on the baseline; none is changed to a pass or skip by adjusting a gate.
- `baseline-*.xml/log`: unchanged-source reproductions of all full-audit
  failures, using the archived repository and its source/import paths.
- `sbound-diagnostic`: twelve live Salvat SBOUND comparisons with identical
  Coulomb potentials and varied starting energies. Seven native root failures
  become successful, with bit-identical reference energies, P and dP/dr.
  The small reference frontend is derived from the DBOUND frontend by selecting
  SBOUND; its source and compiled executable are hashed. Exact s-wave
  derivative behavior at the origin is checked by separate analytic controls.
- `current-numerics.xml/log`: **161 passing tests after all repairs**. This
  includes the six Breit–Pauli failures from the original full audit, now
  passing, and 26 new Schrödinger controls. The remaining baseline issues
  have not been repaired; no green full-suite claim is made.
- `current-determinism-full.xml/log`: **two passing safety gates** after
  migrating from the removed `check_determinism.py` CLI to direct Atom API
  subprocesses. Three fresh full-panel processes agree exactly; a separate
  freshly compiled panel passes Numba bounds checking. Cases cover open-shell
  shooting, closed-shell B-splines, explicit Be MCDHF and heavy-Z Cf shooting.
  CI/nightly jobs call the same tests, preserving their full-panel repeat
  counts. These repair two additional failures from the initial full audit.
- `df-after-sbound`: final H/C/Ne/Cf checks retain exactly the moments-panel
  energies, orbital energies and iteration counts. These first-call timings
  include compilation/testing activity and are excluded from speed claims.

The initial `observables-before` and `observables-before-final` runs used
inapplicable result accessors and failed before reaction measurements. They
are retained for provenance and excluded. `source-tests.log` retained the
initial new-oracle failure: degree-five boundary dot products amplified
roundoff. The final oracle accounts explicitly for that conditioning and
also checks each individual weight independently. No existing gate was
relaxed. The first full audit was interrupted to correct this new test;
`full-tests-final.log` is the authoritative run.

The first new derivative oracle additionally caught a real s-wave origin
issue: Q(0)=0 was the outward seed, although Q=dP/dr is finite there for an
s state. The repaired result reconstructs the derivative from the regular
origin series and normalized first sample. Root-iteration exhaustion and
failed derivative reconstruction now return an error without orbital arrays.
`final-numerics-tests.log` and `sbound-tests-final.log` retain intermediate
new-oracle/harness failures; `current-numerics.log` is the final numerical gate.

Use `benchmarks/bench_df_solver.py`, `benchmarks/bench_shooting_observables.py`,
`benchmarks/bench_radial_bound_kernels.py` and
`benchmarks/bench_reference_screened_continuum.py` to replay. Set
`PYTHONPATH=src:.`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and a writable
Numba cache. Before runs select the archived source tree through PYTHONPATH.
Reference probes additionally require the retained Fortran runtime library.

Run `summarize.py` after all audits finish. It checks identities and numerical
agreement, writes `summary.json`, and hashes the local artifacts into
`manifest.json`. Raw workspaces are ignored by git; summaries, manifest,
runner and this explanation are tracked.
