# He I 1s 16i eigenvector refinement repair

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The initial full first-20 queue completed 7525 configurations but left
He I `1s1 16i1` unconverged and 17 Ca I high-n configurations basis-limited.
Repeated ordinary starting fields did not resolve the helium failure:
reported residuals remained around 6.6e-5 to 1.6e-4.

## Cause and correction

Requested-root eigenvector refinement previously activated at bound index
10. The 16i orbital has l=6 and bound index 9, so it was excluded despite
being a diffuse Rydberg orbital. For an unconstrained channel the principal
number is `index+l+1`. The trigger now uses `index+l>=10`, retaining the
existing constrained-basis bordered-Newton algorithm, overlap guard, root
gap guard, and residual-improvement check.

The policy is recorded as `requested_bound_index_plus_l_ge10_newton_v2`.
Existing records retain their original numerical provenance; no completed
energies were silently replaced. Broader sensitivity checks of previously
completed high-l states remain appropriate before claiming universal accuracy.

## Evidence

Forcing refinement of the previously excluded index gives:

| Outer maximum spacing (bohr) | Converged residual | Corrected energy (Ha) |
|---|---:|---:|
| 8 | 2.28e-11 | -2.001768808906928 |
| 4 | 7.89e-11 | -2.001768808906912 |

The radial box is 1280 bohr in both cases. Energy agreement is about
3.5e-9 cm^-1. These are numerical checks, not a claim of matching NIST at
that precision; the archived 16i reference levels are semiempirical.

The synthetic eigenpair tests now exercise high-l low-index roots as well
as the existing high-index roots, with and without occupied constraints.
Together with the high-l physical tests: **12 passed**.

A fresh production worker with the corrected implementation completed the
helium record. Prior failed attempts are preserved in
`first20_final_retry_history`. The remaining first-20 unresolved states are
the Ca I `4s np` configurations n=63 through 79; two independent phase-grid
pilots for 79p are running and have not been imported.

Diagnostic data: `output/nist_spectra/he16i_refined_eigenvector_probe.json`.
