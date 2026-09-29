# Excited-state orbital stationarity defect

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Reproduction

S I `1s2 2s2 2p6 3s2 3p3 5p1` and Si I
`1s2 2s2 2p6 3s2 3p1 5p1` converge from GSZ and TFDA with orbital-update
residuals below 4e-9, but retain different energies. Finite differences of the
actual Dirac–Coulomb energy under norm- and overlap-preserving occupied 3p/5p
rotations show nonzero first derivatives (roughly 0.01–0.03 Ha/radian).
The converged iteration is not stationary in all allowed orbital directions.

The cause is the blanket skip of occupied same-kappa rotations when the lower
principal shell has a vacancy. That safeguard prevents global energy
minimization from exchanging excited-state radial identities, but also freezes
an unnecessary degree of freedom. Independent-start minimum selection cannot
repair this stationarity defect by itself.

## Experimental repair

`DFSolver._optimize_same_kappa_rotations(..., core_hole_stationary=True)`
uses the existing exact quartic trigonometric interpolation of pair energy,
chooses the nearest stationary root, and limits a single rotation to 0.1 radian.
It permits a saddle rather than requiring a global minimum, appropriate to
following an excited state. This is inspired by the local stationary rotation
in `~/code/dbsr_hf/DBSR_HF/hf_rotate.f90`; the implementation uses atomx's
relativistic pair-energy polynomial, not a literal port of that Fortran routine.
After completion of the checks below, the option is **on by default**.
`core_hole_stationary=False` reproduces the historical frozen rotations.
Production revalidation is running; diagnostic records were not imported as
a substitute for fresh production solves.

For sulfur, both starts reach DC energy -398.247283216546 Ha, agreeing within
5e-13 Ha; for silicon, both reach -289.238950879421 Ha. Finite-difference
rotation gradients are below 6e-10 Ha/radian in both cases. These are numerical
stationarity checks, not claims of experimental or correlated-theory accuracy.

## Evidence and remaining checks

- `scripts/probe_first20_core_p_stationarity.py`, with and without
  `--local-rotation`, measures the energy derivative directly.
- `first20_core_p_stationarity*.jsonl` contain both original and repaired probes.
- `tests/test_dhf_pair_rotation_performance.py` includes an independent sulfur
  energy finite-difference regression and the existing closed-core lithium test.
- `core_hole_local_rotation_helium.jsonl` checks excited helium root identities.
- Two processes run `audit_first20_3p_branches.py --local-rotation` over the full
  53-state 3p-vacancy family. Their `first20_3p_local_rotation_*.jsonl` outputs
  include convergence, energy components, orbital energies and node diagnostics.

Next: review all family solutions and excited-state controls, then decide the
validated default scope and revalidate affected production rows with history.

Verification: both pair-rotation regression tests passed (58 s). Excited helium
1s2s and 1s4s converged from both starts to matching energies (within 1e-15 Ha);
1s4s retains zero significant 1s nodes and three 4s nodes. The 1s2s core has
an exchange-induced outer node; a universal local-potential node rule would
therefore be inappropriate. Wider family validation remains in progress.

The interactive comparison now includes all 16 sulfur/silicon finite-difference
records and explicitly labels affected excited-state production comparisons
provisional. The generated JavaScript passed syntax validation and the page
returned HTTP 200. The local-rotation coverage report is produced with
`review_first20_3p_branches.py --local-rotation`; the first seven completed
configuration pairs agree within 1e-7 Ha, with wider validation still running.

## Additional controls and explicit API

The option was introduced as `DiracFock(core_hole_stationary=True)` with an
opt-in default; after the full census passed it became the default, as recorded
in the production adoption section below.
The census and derivative probes use that explicit API; the regression checks
its propagation to the solver. Existing processes retain the implementation
loaded when they started, so no production worker changes method mid-run.

Both starts converged with ordered orbitals for Ne I 2p5 3p, Na VI 2p 3p,
Sc IV 3p5 4p and Kr IX 3d9 4d. The Sc IV total-energy difference between starts
is 1.46e-9 Ha, and the Kr IX difference is 4.02e-9 Ha. Their core/valence node
counts have the expected identities: Sc 3p/4p = 1/2; Kr 3d/4d = 0/1.
These independent controls address earlier displaced-root cases outside the
53-state census. They do not establish a global all-element convergence guarantee.
Run the controls with `audit_first20_3p_branches.py --local-rotation
--control-set early-p` or `--control-set heavier`; their output files carry those
suffixes. No candidate energy has yet been imported into production.

## Production revalidation scope and NIST comparison

`core_hole_rotation_revalidation_scope.json` enumerates **657** H–Ca
configurations with occupied same-l shells and a vacancy in a lower principal
shell. None is a ground configuration. This is the set requiring reassessment
when the new default is adopted, not an assertion that every energy is wrong.
The 53-state census is a validation sample rather than the full repair scope.

`scripts/compare_core_hole_rotation_nist.py` compares agreeing candidate pairs
against exactly the same complete evaluated NIST centroids, retaining the
production ground reference. It never writes production energies or chooses a
candidate using NIST agreement. An initial snapshot has 37 agreeing pairs and
10 eligible centroid comparisons: RMS error changes from 13960.99 to 6869.25
cm^-1. This subset result cannot be extrapolated to the full periodic table.
The website displays the snapshot alongside the finite-difference evidence.

## Production adoption

All 53 pairs passed the independent-start DC-energy agreement threshold of
1e-7 Ha, and their node diagnostics agree between starts. The default now uses
local stationary rotations. Five focused production/root-guard checks passed;
historical displaced-root tests explicitly disable the new rotation to keep
testing their original failure mode. The requeue regression passed and checks
that a live job is never reclaimed and original energies are preserved.

`prepare_core_hole_revalidation.py --apply` preserved and requeued all 657
affected rows in `core_hole_stationarity_history`. Four new workers consume
`repair_pending`; older workers finish the separate original pending queue.
The ledger audit now requires local-rotation provenance for completed records
in the repair history. Three original B I n=36–38 d jobs failed their ordinary
starts; their errors are retained in `boron_rydberg_retry_history` and a damped
retry pilot is running. These are separate from the core-hole rotation defect.

## Production comparison and separate Rydberg retries

The production before/after collector is
`scripts/compare_revalidated_core_holes_nist.py`. It joins each new energy to
its preserved history and scores only identical complete evaluated NIST
centroids. An initial 208-state repair snapshot has 39 eligible comparisons:
RMS drops from 36026 to 19741 cm^-1, but 9 residuals improve and 30 worsen.
Removing large numerical outliers can reduce RMS while many residuals grow;
this is not evidence that correlation is now included. The website displays
both counts and individual differences, refreshing the report at each build.

B I 36d converged from the damped hydrogenic start (0.35 mixing) in 210 s,
energy -24.241915047368217 Ha. The independent shooting-seed route failed
its final B-spline convergence gate after 143 s (residual 0.00999); its energy
is not accepted. The 37d–39d damped retries are running. Repair concurrency
was increased from four to eight workers with 97 GB RAM available.
