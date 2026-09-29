# Shooting correction numerics: occupied gauge and ordered Breit integrals

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Defects isolated

1. Fully occupied same-kappa orbitals can rotate without changing the DHF
   determinant or total energy. The shooting solve enforced orthogonality but
   did not diagonalize this occupied Fock block before assigning principal-n
   labels to the model self-energy corrections. The B-spline result did.
2. DBSR Breit ordered radial triangles used trapezoidal prefix integrals even
   when orbitals were represented on native SLAG6 or cellwise Gauss grids.
   This imposed a lower-order integration error on otherwise accurate states.

For Be-like sulfur, the old occupied Fock off-diagonal was -0.182986 Ha despite
a converged total energy. Canonicalization reduces it below 5e-15 Ha. Only
fully occupied same-kappa subspaces are canonicalized; physically significant
open-shell variational rotations are retained. The operation changes the
representation of an unchanged occupied state, not the energy functional.

## Implementation

- `ShootingHFSolver` now builds a shared occupied Fock matrix for diagnostics
  and canonicalization, orders canonical roots by principal n, and fixes signs
  deterministically before returning orbitals and energies.
- `numerics/radial_integration.py` owns cumulative native integration. Gauss
  cells use cardinal-polynomial partial integrals plus full preceding-cell
  quadratures; shooting uses SLAG6 with the complete Jacobian and origin.
- Native radial workspaces retain their integration rule and Jacobian. Breit
  provenance names the actual integration rule. Legacy arbitrary sampled
  workspaces retain explicitly identified trapezoidal integration.
- The shared workspace documentation now correctly distinguishes the SGRID
  Jacobian from the integration weights.

## Measured S XIII control

All values in Hartree. The nuclear model and isotope are matched.

| Contribution | Earlier shooting, 3200 points | Fixed shooting, 3200 points |
|---|---:|---:|
| Model self energy | 0.065789267975 | 0.065810942566 |
| Uehling VP | -0.004157122049 | -0.004157122049 |
| Static Breit | 0.060419009866 | 0.060419176268 |

Fixed shooting Breit at 1600/3200/6400 points is respectively
0.06041917626920, 0.06041917626827, 0.06041917626767 Ha.
The previous B-spline ledger Breit was 0.06041431981936 Ha. Fresh B-spline
native integration gives approximately 0.06041916291159 Ha, leaving a
1.34e-8 Ha cross-engine difference. Halving the tested B-spline inner/exponential
step controls did not materially reduce that last difference. It remains a
measured numerical limitation; no exact agreement is claimed.

26 focused tests passed: analytic polynomial partial integrals, manufactured
SLAG6 prefix, independent corrected B-spline/shooting controls, stored external
DBSR Breit references, open-shell constraints, native metric and correction
application contracts. The cross-engine correction test uses an explicit
2e-8 Ha absolute bound, retaining the observed Breit residual.

## Benchmark and website

The fresh 420-run checkpoint is
`output/nist_spectra/native_shooting_first20_canonical_native_integrals.json`.
It records source hashes and correction provenance separately from earlier
results. Prior ledger energies have not been silently replaced or relabelled.
Ground stages are only the first validation panel: excited-state/observed-line
coverage and all-element generation still need their own convergence gates.

`scripts/render_native_shooting_benchmark.py` renders the checkpoint into
`output/nist_spectra/native_shooting/index.html`, served at
`/comparison/native-shooting/` and linked from the atlas/comparison pages.
The interactive H–Ca periodic table filters charge-stage comparisons by
source kind and observable matching. Only evaluated, unique-ground-level
rows enter its RMS score. Each row exposes both endpoint specifications and
correction contributions. The page states successful/requested run counts,
so an incomplete checkpoint cannot masquerade as a completed campaign.

Completed rerun: 420 successful calculations, 210 ionization comparisons.
Maximum two-grid ionization change: 0.00042459758 cm^-1
(previously 0.195317 cm^-1). Evaluated/unique-ground-level RMS remains
11581.04120 cm^-1 across 29 comparisons; improving numerical
consistency has not supplied missing correlation. All recorded source hashes
matched the working source at completion. The HTML was regenerated from the
completed checkpoint. Client JavaScript syntax and the two existing atlas API
tests passed; no browser screenshot inspection was performed.
