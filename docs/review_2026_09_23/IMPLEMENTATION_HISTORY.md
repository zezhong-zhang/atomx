# Remediation tracking

Objective: fix all findings in the September 23 review, with first-principles
physics checks and measured numerical performance. This file tracks work in
progress; an unchecked item is not resolved.

- [x] 1. Constrained orthonormal orbitals, consistent CI, honest convergence.
- [x] 2. Exact one-body transition operators in the Hamiltonian CSF convention.
- [x] 3. Correct flux-normalized R-matrix boundary matching and reference waves.
- [x] 4. Reciprocal collision strengths and correct statistical weights.
- [x] 5. Correct Atom-to-CRM state identities, emission directions, degeneracies.
- [x] 6. Stable spontaneous Milne RR and explicitly field-dependent stimulated RR.
- [x] 7. Forward all radial box/grid controls through MCDHF.
- [x] 8. Consistent initial-state averaging in shooting PI/RR.
- [x] 9. Physical occupations and density without solver occupation floors.
- [x] 10. Reuse continuum eigensystems with safe invalidation and bounded lifetime.
- [x] 11. CI dimension preflight, M-sector enumeration, bounded caches and sparse connectivity.
- [x] 12. Block-aware requested eigensolvers and low-state/matrix-free execution.
- [x] Model/validation follow-up: disclose radial optimization and Gaunt/QED approximations;
      residual-core continuum consistency; discretization convergence; correct
      same-Hamiltonian variational gates.
- [ ] Broad regressions, independent probes, performance measurements, completion audit.

No completion claim is made by this initial checklist. Validation and remaining
limitations will be recorded as implementation progresses.

## Implementation checkpoint (work still in progress)

All twelve findings now have implementation changes. The completion audit is
still open: broad regression failures are being classified and repaired, and
heavy-atom runtime/stationarity need further checks. Do not treat the unchecked
list above as a final status report.

- Constrained generalized Dirac solves enforce the actual S metric. Orbital
  rotations with unequal populations are relaxed, closed-shell subspaces are
  canonicalized, and damping/DIIS retract to the orthonormal manifold. Frozen
  reference orbitals are not constrained by unoccupied virtuals. CI validates
  its input metric and refuses nonorthogonal inputs. Inner convergence is
  reported instead of being overwritten by an outer energy plateau.
- Exact cached CSF tensor terms replace population-factor transitions. For
  independent radial bases, overlap cofactors implement generalized one-body
  determinant matrix elements (including singular overlap matrices).
- Correct R-matrix phase matching, flux normalization, spin averaging,
  reciprocity, stable closed-channel startup, and converged Coulomb references.
- CRM directions and degeneracies repaired. Spontaneous Milne recombination is
  separated from explicitly field-dependent stimulated recombination; its
  analytical Kramers integral remains stable at low temperature.
- Grid/box controls are forwarded, shooting RR averaging is repaired, physical
  populations have no artificial floors, and radial density includes the
  off-diagonal one-body density matrix.
- Continuum eigensystems belong to the radial workspace, are bounded to eight
  entries, and use content fingerprints for in-place grid/potential invalidation.
  Default RR now uses the frozen residual Hartree core and records its local
  approximation; explicit potential models are honored by both engines.
- CI dimensions are counted before allocation. Fixed-M sectors are lazy,
  connections use common active-electron spectators (excluding frozen-core
  spinors), coefficient tables have connected-entry rows, and caches are bounded.
  Davidson uses projected sparse determinant operators, including every symmetry
  block, and only requested low eigenpairs are merged.
- A correlated orbital functional now differentiates the *same cell integrals*
  used in exact CI, including configuration interference. B-spline MCDHF uses
  its gradient in a fixed positive-energy nuclear Dirac subspace, with metric
  retraction, energy-lowering line search and stationarity diagnostics. This is
  undergoing broader validation; shooting/frozen protocols retain their explicit
  approximation. The gradient agrees with central differences to about 1e-9
  in a helium probe. A helium two-radial-orbital example lowers from -2.8620277
  to -2.8781328 Ha with orthonormality preserved.

Evidence so far (runs overlap, counts must not be added):

- 63 targeted scattering/CRM/RR tests passed before subsequent structure work.
- 89 passed, 1 xfailed, 1 failed in the first tensor/kernel run; the failure
  pinned the removed sqrt(2) normalization and was replaced with a filled-shell
  one-electron sum-rule assertion.
- 112 scattering/CI/first-principles/Breit tests passed.
- 23 structure/SCF tests passed after canonicalization.
- 35 RR and new review regression tests passed.
- Directional derivative and nonorthogonal cofactor tests passed.
- Original helium transition probe: exact same-wavefunction A values agree at
  floating-point precision (strong 2.6512242e9/s; weak 10.7916905/s).
- Rejected 21,718-determinant block: 0.42 ms, zero sectors enumerated, no measured
  increase in peak RSS (previous all-M allocation was 352,800 determinants).
- 80-energy RR run: 2 diagonalizations on first call, zero on repeated calls;
  approximately 0.27 s cold / 0.21 s warm on this machine. A controlled comparison
  against disabled production caching remains to be recorded.
- Broad non-slow regression run before subsequent fixes: 1166 passed, 15 failed,
  7 skipped, 10 xfailed, 24 deselected. Root-tracking, normalization, exact-tensor
  resolution expectations and same-basis variational assertions were repaired
  afterward; repeat regression still pending.
- A clean HEAD archive exists at /tmp/atomx-review-baseline for comparison.
  Seven stale surface-migration snapshots also fail there. The other tested
  baseline panel (hyperfine, Auger, exchange, FAC snapshots, isotope) passes:
  113 passed and 5 xfailed; changes against those must be investigated.

Reference theory: GRASP2018, DOI 10.1016/j.cpc.2018.10.032; GRASP manual,
https://uu.diva-portal.org/smash/get/diva2%3A1760743/FULLTEXT01.pdf ; Coulomb
reference expansions https://dlmf.nist.gov/33.11 . No external package output
is used as a substitute for the matrix-element, normalization and derivative
identity tests.

## Further validation and numerical fixes

- Corrected an additional mixed-component exchange error: P/Q exchange
  contractions now use both spline orders, and the PQ/QP blocks are mutual
  transposes rather than individually symmetrized. A finite-difference
  derivative of the HF energy independently checks the assembled operator.
- Same-symmetry occupied rotations use the exact quartic trigonometric HF
  energy; four analytic derivatives determine its stationary points. Closed
  subspaces are canonicalized during iteration. The quick argon case now
  converges in 22 iterations (previous constrained implementation: 82).
- Correlated optimization now scales the preconditioner by occupations,
  transports secant vectors between tangent spaces, and reuses the accepted
  line-search CI solution. Fe XVII reached the stated stationarity criteria
  in 109 iterations instead of failing the previous 160-iteration limit.
  Complete one-electron spaces use canonical nuclear eigenfunctions; missing
  lower reference shells are excluded for explicitly excited active spaces.
- Occupations and the radial density use the same degenerate-ground-state
  ensemble. Corrected levels, CI columns, and QED shifts are sorted together.
- Rydberg RR uses a common residual Hartree bound/continuum Hamiltonian,
  excludes occupied core states through Pauli vacancies, and removes the
  unbalanced-basis spurious positive-kappa bound root.
- CI radial evaluation reuses banded pair potentials. On a 40-spline lead
  example, the profiled energy evaluation fell from 0.536 to 0.225 seconds
  after preventing bounded-cache eviction (1,219 to 407 pair contractions).
- Controlled continuum benchmark: 80 energies, 278 splines; uncached median
  6.131 s, cached median 0.341 s, about 18x faster. Diagonalizations fall from
  160 to 2 and cross sections are bit-identical. See
  `benchmark_continuum.py` and `continuum_benchmark.json`.
- Fresh algebraic probes are in `remediation_probes.txt`: exact same-wavefunction
  line strengths, reciprocal collision strengths, unitary/symmetric S,
  correct tan(delta), spontaneous Milne balance, forwarded grids, dimension
  preflight and requested eigensolvers. The stringent tol=1e-10 helium orbital
  probe reports nonconvergence honestly despite a stable energy and ~1e-16
  overlap; this is not counted as a stationarity pass.
- Targeted runs: 76 structure/hyperfine/RR/Fe-line tests passed; 17 Auger/review
  tests passed with 3 existing xfails; 62 refreshed structure/isotope snapshot
  tests passed; 13 new structure regression tests passed. Counts overlap.
- The Auger fixtures now explicitly request frozen-core spectroscopic
  virtuals. Ground-state optimized compact correlation orbitals carrying a
  3p label are not spectroscopic 3p states. External FAC widths and accuracy
  bands are unchanged. Three strict expected failures for Fe XVII lines now
  pass and their obsolete xfail markers were removed.
- Self snapshots were refreshed after the operator and constraint fixes;
  external FAC columns were preserved. The record is in
  `snapshot_measurements.json` and `scattering_snapshot_measurements.json`.

## Heavy-atom and continuum completion checks

- The final correlated optimizer projects the initializer's screened Fock
  matrices into the fixed nuclear positive-energy space for preconditioning.
  The default Pb calculation converges in 19 optimizer steps: 71.715 s
  initialization, 26.165 s optimization, 97.881 s total. The earlier nuclear-gap
  preconditioner exhausted 160 steps after 817.558 s without converging.
  This is an optimizer comparison during remediation, not a same-physics
  speedup claim over the original nonorthogonal calculation.
- Pb final energy is -20913.797768480083 Ha; preconditioned orbital residual
  8.009e-6 against 1e-5 and estimated energy error 5.685e-11 against 1e-8.
  An independent five-point derivative under a same-kappa occupied rotation
  agrees with the analytic correlated gradient to 6.63e-10 Ha/radian on the
  40-spline calculation. See `benchmark_structure.py` and
  `structure_benchmark.json` for timing and diagnostics.
- The explicitly selected FAC-like residual-core potential had two additional
  defects: it flattened the nucleus below the first uniform-grid point and
  applied a bound-electron self-interaction correction after already removing
  the captured electron. Native Hartree quadrature, separate nuclear evaluation,
  and a single hole now preserve both the core field and residual ionic charge.
  The shooting-result path now builds its Hartree field rather than silently
  falling back to a bare nucleus; an analytic hydrogenic density checks it.
- FAC K-shell RR now differs by about 4%, passing its unchanged 15% gate.
  Each Li-like final level is separately optimized with its occupied jj orbital;
  the 2p3/2 channel no longer uses a zero-population ground-state spectator.
  Li-like discrepancies are below 17% and their external gates are unchanged.
  `rr_panel_measurements.json` records the refreshed self column, preserving
  external FAC data. Historical timing columns are not new measurements.
- The Tl test configuration omitted the six 5d5/2 electrons
  (kappa=-3); the corrected 81-electron fixture
  converges with orthogonal orbitals and both enhancement-scale gates pass.
  The obsolete strict xfails were removed. The wide enhancement interval is
  still a qualitative check, not a precision many-body benchmark.
- One former 0.2% excited-state cross-engine assertion was an accidental
  agreement between different constraint treatments. After the B-spline fix
  the 2p3/2 -> 3s rate differs by 0.424%. Shooting explicitly approximates
  open-shell Lagrange multipliers through sequential Schmidt projection.
  The test now applies its existing 3% open-shell model-comparison band to
  all such channels. This is a disclosed model limitation, not a claim that
  shooting's constrained orbital equations were repaired here. Closed-shell
  cross-engine and independent analytic-kernel tolerances were not changed.
- Old Atom-constructor migration snapshots already failed on the reviewed
  baseline after the Gaunt reporting change. Supported paths were regenerated;
  the statistical open-N fixture now explicitly disables a nonunique Gaunt
  correction. Independent DBSR_HF references are unchanged. See
  `surface_snapshot_measurements.json`.
- Latest targeted checks: 45 surface/potential/transition tests passed; 16
  RR/Tl tests passed. Counts overlap earlier runs. A fresh broad non-slow run
  and the final documentation/model audit are still in progress.

Remaining completion work: finish the fresh broad run, address any new failures,
and finish the model/validation audit. No claim of complete remediation or
universal basis convergence is made.


## Initialization and remaining regression follow-up

- The configuration-average initializer is capped at three outer passes for
  the B-spline correlated-gradient protocol. It solves a different fixed point
  and is not itself the stationarity criterion. The correlated optimizer can
  use up to eight steps per requested outer-iteration budget; convergence still
  requires both its stated orbital and estimated energy-error tolerances.
  87 targeted structure tests passed after the initialization change.
- DIIS trials with a singular orbital Gram matrix are rejected transactionally;
  the valid sweep is retained and the extrapolation history restarted. A
  deliberately collapsed extrapolation checks this recovery without weakening
  the CI metric validation.
- Carbon 2p²/3p² converges in 60 optimizer steps, about 25 s total. Carbon
  2p²/3d4d needs 189 steps and about 61 s. These are current-case timings,
  not controlled cross-version speedup ratios; see
  `correlation_space_measurements.json`. Their old self energies were refreshed,
  and the split-channel test measures the added-configuration weight directly
  instead of assuming a largest-CSF-coefficient bound independent of recoupling.
- The shared MBPT virtual-basis builder now removes the unbalanced P/Q
  positive-kappa spurious root and honors the SCF nuclear potential. A bare
  hydrogenic 2p energy and a constant-potential-shift identity independently
  check both balanced and equal-order bases. Zero-population spectator
  orbitals are no longer marked occupied. MBPT prose now correctly states
  the V^N reference and that truncated perturbation energies are not
  variational bounds.
- MBPT/polarizability/virtual-basis targeted validation: 42 passed, 1 existing
  xfailed, 6 slow tests deselected. Separate slow Tl tests are being checked.
- Default Pb recheck: 19 optimizer steps, energy -20913.79776848005 Ha,
  residual 8.274e-6, estimated energy error 6.260e-11. Total time was 123.727 s
  during concurrent regression work; the earlier isolated timing was 97.881 s.
  The energies differ by approximately 3e-11 Ha.
- Broad run before these final edits: 2,482 passed, 4 failed, 78 skipped,
  20 xfailed, 24 slow tests deselected in 953.18 s. The four failures were
  the now-documented open-shell engine comparison, DIIS rank collapse and
  two stale correlation-energy snapshots. The refreshed full run is pending.


The model/validation follow-up is documented in [MODEL_LIMITATIONS.md](MODEL_LIMITATIONS.md).
This records the fixed projector, local orbital optimization, approximate shooting
constraints, continuum models and the distinction between iteration convergence
and basis convergence. It also distinguishes the remaining expected failures
from the twelve reproduced defects corrected in this review.


Final targeted checks: all 35 review regression gates pass. With the final
virtual-basis changes, 19 DR/Auger tests pass with 3 existing expected failures;
all 12 Tl/EDM tests plus the corrected carbon split-channel test pass (13 total).
External FAC data were compared column-by-column with the reviewed HEAD and
are unchanged; see `external_reference_integrity.json`. Changed Python files
pass Ruff, and `git diff --check` is clean.
