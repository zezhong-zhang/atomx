# Third-grid and correction-expectation review

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

Selected the largest grid changes among the currently matched first20 centroids.
Recomputed F VII ground/5g and Ar VIII ground/6h/7h at 12800 points using the
frozen stagnation-recovery source, with the same physics and 1e-13 final SCF
tolerance. All five solves converged; bare checkpoints, original paired-grid
records, source hashes and correction ledgers are retained under
`normalization_diagnostic/third_grid_current/`.

| Excitation | 3200→6400 change (cm^-1) | 6400→12800 change (cm^-1) |
|---|---:|---:|
| F VII 5g | -0.000478811 | +0.000478811 |
| Ar VIII 6h | -0.000575341 | -0.000000237 |
| Ar VIII 7h | -0.000407132 | -0.000000137 |

The Ar controls settle on the larger grids. F's nonmonotonic change must not be
used for a Richardson extrapolation or a certified error bar. For F 5g, its
bare-state total energy changes only 1.22e-7 cm^-1 between the larger grids;
most of the corrected-energy change comes from Breit (+0.000338217),
self-energy (+0.000098092), normal mass (+0.000056171), and VP (-0.000013760).
These first-order expectation values are more sensitive to residual orbital
errors than the stationary total energy.

Refining both saved F 5g states to SCF tolerance 1e-15 converged in four and
three iterations, respectively. Final orbital-update norms are 1.82e-8 and
2.08e-8. The corrected *state-energy* grid difference decreases to
7.70436e-5 cm^-1. This is evidence of orbital-convergence sensitivity, not a
complete uncertainty bound; the ground states were not tightened in this
second diagnostic. The residual variation remains explicit. No campaign
result was silently replaced by this diagnostic.

## Result metadata fix

The direct warm-restart test exposed that `ShootingHFSolver.solve` returned
fractional occupations without retaining the explicit jj configuration
ensemble required by DBSR_HF Breit pair coefficients. Campaign wrappers had
manually restored the ensemble, masking the public-result gap. The solver now
returns an owned copy in `ShootingResult._jj_configs`; it survives persistence
and warm restart without caller repair. No physical coefficient changed.

The open-shell persistence regression now checks ensemble ownership and
applies Breit immediately after a warm restart. Six persistence/generation
tests pass. The frozen diagnostic driver explicitly restored the ensemble,
matching the established campaign wrapper; it did not alter its frozen solver.
