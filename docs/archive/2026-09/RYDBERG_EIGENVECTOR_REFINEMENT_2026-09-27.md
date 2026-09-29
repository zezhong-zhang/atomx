# Diffuse-state eigensolver accuracy diagnosis

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The corrected first-20 ledger still has convergence failures for high-n B I
and Mg I states. Increasing iteration counts and changing initial fields did
not resolve B I 38d: a Slater-initialized DHF run reached 300 iterations with
orbital-change residual 2.14e-4 despite stable total energy. This result was
not accepted into the ledger.

`scripts/probe_rydberg_eigensolver.py` captures the actual 905 by 905 B I 38d
Dirac matrix after 30 SCF sweeps. Comparisons use precisely the same matrix,
overlap, boundary conditions, and requested root:

| LAPACK driver | Eigenvector residual norm | Time, seconds |
| --- | ---: | ---: |
| gvd | 5.24e-8 | 0.083 |
| gv | 3.92e-10 | 0.275 |
| gvx | 3.92e-10 | 0.280 |

The neighboring eigenvalue gap is only 1.76e-5 Ha. The gv and gvd vectors
differ by 1.66e-5 relative to the largest coefficient. Earlier driver timing
and agreement on compact Ne matrices therefore do not establish accuracy
for these diffuse states.

A bordered Newton correction solves the eigen-equation together with its
S-normalization constraint. On the saved matrix, one correction reduced
the eigenvector residual to 5.51e-15; a second reached 2.33e-15. The Rayleigh
quotient is accumulated in extended precision. This changes neither the
Hamiltonian nor the radial basis.

The experimental `--refine` probe applies two such corrections to the
requested d roots during each B I 38d sweep. It converged in 10 iterations
with orbital-change residual 2.486e-8 and corrected total energy
-24.241875437123 Ha. Static DBSR Breit, model QED, and operator recoil remain
enabled. The production solver is unchanged pending neighboring-state and
grid checks, and this diagnostic energy has not been written to the ledger.

The probe is deliberately limited to unconstrained boron d channels. A
production implementation must preserve the occupied-orbital constraint
subspace for magnesium p states, verify root identity and residual reduction,
and avoid refining every unoccupied eigenpair.

## Independent checks

- B I 39d also converged, residual 2.478e-8, corrected energy
  -24.241857885908104 Ha.
- Refining B I 38d from hmax=19 to 15 Bohr gave residual 2.486e-8 and
  energy -24.241875389050513 Ha. The total-energy change is 4.81e-8 Ha
  (0.0106 cm^-1). This measures remaining grid dependence; it is not an
  experimental-accuracy claim.

Both checks used the experimental refinement. The outstanding next step is
to implement targeted refinement within the solver's constrained eigenproblem
and validate Mg I, rather than loosening convergence thresholds or accepting
the failed production rows.

## Integrated implementation and current validation

`DiracResult.get_orbital` now refines requested bound-root indices >= 10,
once per eigenproblem. The correction operates in the same projected basis
as the original eigensolve, preserving occupied-orbital constraints. Root
overlap and neighboring-energy-gap guards reject root changes; a correction
is retained only when its eigen-equation residual improves. Unrequested
eigenpairs are not refined. The policy is recorded in DHF energy components.

Twenty focused refinement, hydrogenic, and radial-phase tests passed; eleven
selected core-hole and rotation regression tests also passed. The integrated
B I 39d solve converged in 14 sweeps at tolerance 1e-7, residual 4.06e-8.
Both 38d and 39d production jobs were subsequently recomputed successfully.
Their previous failures are preserved in `rydberg_eigenvector_repair_history`.

Mg I 39p still shows large SCF oscillations with refined eigenvectors. A
Slater seed reaches an ordered state, but subsequent orbital DIIS can precede
large energy jumps. A separate `--no-diis` diagnostic is running to distinguish
extrapolation instability from eigenproblem conditioning. This is not yet a
validated magnesium repair; the full-table release remains held.

The no-DIIS Mg I 39p diagnostic subsequently converged in 13 DHF sweeps,
residual 2.860e-7, total energy -199.6342680357217 Ha, with ordered orbitals.
This used the same refined eigenproblem and Slater initializer as the
oscillating run. It isolates orbital extrapolation as an additional failure
source. Inspection found that DIIS errors are differences between consecutive
stored output vectors, although the actual input may have been extrapolated
and rotated in between. The next repair should use the actual sweep residual
and validate acceleration against the converged no-DIIS reference; simply
loosening the stopping tolerance would not address this defect.

## DIIS sweep-residual repair

DIIS now stores each actual input before the orbital sweep and supplies
`F(x)-x` to extrapolation. The first history entry also has its measured
residual, replacing the former artificial zero. The correction does not
change the Hamiltonian or stopping tolerance.

Mg I 39p converged in 19 sweeps with corrected DIIS, compared with 34 for
the previous residual definition using the same Slater seed and 13 without
DIIS. A hydrogenic start with the earlier DIIS policy converged in 68 sweeps.
All four corrected-eigenvector calculations agree within 2.46e-11 Ha.
The corrected DIIS total is -199.63426803569962 Ha, residual 7.69e-7.

A regression test captures actual sweep inputs and outputs and verifies
that every supplied DIIS history error matches them. It passed, as did ten
selected eigenvector/core-hole/rotation tests and three generator seed/retry
tests. Diffuse neutral Mg retries now use a converged Slater initializer,
followed by nonlocal DHF with the full corrected physics and a 1e-6 damping
release threshold. The Slater energy is discarded.

Failed 39p, 40p, 42p, and 52p rows were preserved in
`mg_refined_diis_repair_history` and queued to fresh workers. Live jobs were
not reclaimed. Grid validation and the remaining Mg production jobs must
finish before releasing the full-table queue.

The Mg I 39p hmax=15 Bohr check converged in 13 sweeps and changed the
total energy by 8.17e-8 Ha (0.0179 cm^-1) from hmax=19.5. This is a radial
grid diagnostic, not an uncertainty estimate for agreement with experiment.

After those checks, five remaining pre-fix Mg workers were deliberately
terminated to apply the demonstrated numerical repairs: 38p, 55p, 57p, 58p,
and 59p. Each PID and process start time was checked, its running ledger row
saved in `pre_refinement_running_history`, and process termination verified
before requeuing. This restart was prompted by a verified solver defect, not
by elapsed runtime. Three fresh workers now consume the Mg retry queue.
