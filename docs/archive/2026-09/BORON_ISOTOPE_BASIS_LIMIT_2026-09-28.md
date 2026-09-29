# Boron-11 basis-domain validation

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The explicit isotope queue completed B-11 configurations through 62p but retained
seven `basis_limited` states, 63p–69p. Their ordinary production grids require
708–769 splines, exceeding the existing 700-spline validation cutoff. No SCF
attempt was made for these seven states by the ordinary worker.

`scripts/probe_boron_isotope_basis_limit.py` calculates the largest state,
`1s2 2s2 69p1`, with the same mass and v3 corrections on two grids. It does not
change the production cutoff or write results into the comparison ledger.

- First grid: radius 23,805 bohr, outer spacing 34.5 bohr, 769 splines.
- Converged from a hydrogenic seed with 0.35 damping in 61.8 seconds.
- Energy: -24.241634438845534 Ha; SCF residual 3.44398e-8.
- Second grid: same radius, outer spacing 28 bohr, 928 splines; converged in
  107.2 seconds, residual 2.46781e-8, energy -24.241635010736783 Ha.
- Grid difference: -0.125515621 cm^-1. This exceeds the quoted 0.022 cm^-1
  NIST level uncertainty and does not support promoting the coarse result.

The largest changing components were E_1e (-4.04871e-5 Ha) and E_2e
(+3.99151e-5 Ha), which partly cancel. A follow-up keeps the outer spacing
34.5 bohr and halves the exponential growth parameter from 0.25 to 0.125.
This tests whether radial resolution in the inner/middle region improves
accuracy more economically than adding uniform outer points. Its result is
kept separately; the production cutoff remains unchanged.

The middle-region refinement converged with 836 splines in 73.1 seconds,
residual 6.68057e-8 and energy -24.241634932704667 Ha. Its shift from the first
grid is -0.108389551 cm^-1, obtained with 67 extra splines rather than 159.
It differs from the outer-only refinement by +0.017126070 cm^-1. A fourth
probe combines hmax=28 with he=0.125; its result must be checked before
deciding on production admission.

The combined grid converged with 995 splines in 165.9 seconds, residual
2.04150e-8 and energy -24.24163492697373 Ha. It differs from the middle-region
refinement by only +0.001257781 cm^-1, much smaller than the first grid spread.

## Experimental Coulomb-phase grid

`scripts/rydberg_phase_grid.py` retains the finite nuclear/core mesh through
approximately 20 bohr and chooses outer intervals using the analytic integral
of the zero-energy charge-one Coulomb momentum, `2 sqrt(2 r)`. A relative-step
cap keeps the join gradual. This controls the oscillatory phase without using
NIST energies or fitting the desired eigenvalue. The 20 bohr join is specific
to this diagnostic; it is not validated for arbitrary atomic cores.

The phase-step-1 grid has 561 splines. It converged in 29.0 seconds with
residual 2.05122e-8 and energy -24.241634927963837 Ha, within about 0.0002173
cm^-1 of the 995-spline combined grid. A tighter phase step of 0.75 is being
checked independently. Production defaults have not changed.

Two tests verify core breakpoint preservation, monotonic knots, endpoint
multiplicity, B-spline partition of unity, phase-step bounds, refinement density,
and invalid-parameter rejection. The standard DBSR core grid rounds its box
upward to a full step; the test uses that actual core endpoint.

`boron11_69p_grid_diagnostics.json` estimates the semiclassical phase advance
per interval in a charge-one Coulomb tail. The largest estimated phase steps
are 3.74, 3.35 and 2.76 radians for the three grids respectively, near their
exponential-to-uniform transitions. Meanwhile 414–511 intervals lie beyond
the estimated outer turning point. This supports investigating a grid adapted
to the oscillatory region, rather than uniformly adding tail points. The
Coulomb-tail estimate is a diagnostic approximation, not a local representation
of the nonlocal DHF equation or a numerical error certificate.

The archived 63p and 69p levels each quote uncertainty 0.022 cm^-1 and J=1/2.
The DHF configuration averages remain unscored against those individual levels,
regardless of numerical convergence. Grid agreement alone is not experimental
accuracy or a justification for raising the cutoff for all other configurations.

The isotope collector now exposes `basis_limited`, `calculation_failed`, and
individual endpoint calculation statuses instead of collapsing these into a
generic missing-theory category. All seven affected observed lines and seven
levels were verified in the rebuilt site data. Three isotope tests passed,
including preservation of Ritz-only status and no fabricated prediction for a
basis-limited state.

Probe log: `/tmp/boron11-69p-basis-probe.log`.
Result artifact: `output/nist_spectra/boron11_69p_basis_probe.jsonl`.

## Phase-grid convergence confirmed for 69p

The phase-step-0.75 check completed with 701 splines in 53.9 seconds,
residual 2.02959e-8 and energy -24.241634927963972 Ha. It agrees with the
561-spline result to 1.35e-13 Ha (about 3.0e-8 cm^-1). This is strong
grid-convergence evidence for this state, not a total theoretical uncertainty.
The same two-grid validation has been started for each remaining B-11 state
63p–68p before those states can be admitted to the ledger.

For phase-grid probes the command's hmax argument does not define a constant
outer step. New records mark `hmax_bohr=null` and preserve `requested_hmax`
separately; the first two 69p phase records retain the older nominal field, and
their `phase_step` identifies the actual grid construction.

## Guarded result admission

`scripts/accept_boron_phase_results.py` validates each state independently:
matching B-11 identity and mass, the expected two phase grids, convergence,
finite energies, v3 nuclear profile and QED policy, exact correction accounting,
grid spread below 0.001 cm^-1, and a total energy below the matching B II core.
It refuses to replace worker-owned states. The old row is preserved in
`boron_phase_grid_history`, and the accepted components record both source-file
hashes, grid sizes, phase step and measured numerical spread. The nonuniform
grid has no scalar outer hmax, so that ledger column is explicitly NULL.

63p, 64p, 65p and 69p have passed and entered the isotope ledger. Their standard
ledger counterparts are absent, so no standard jobs were inserted by this
repair. The remaining three pairs are still being calculated. The isotope
ledger audit after initial admission passed with component-sum error below
7.2e-15 Ha. Four acceptance/geometry tests passed, covering excessive grid
drift, wrong isotopes, live-job protection, idempotence and preserved history.

All seven states 63p–69p have now passed and been admitted. Their individual
grid spreads range from 1.95e-8 to 4.20e-7 cm^-1. The B-11 ledger is complete
at 64 configurations, with seven original basis-limit records preserved in
history. Its final stored-record audit passed. These configuration averages
still do not resolve the individual J levels or supply correlation corrections.
