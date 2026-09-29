# Be-like excitation residuals after the stationarity repair

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The repaired 2s3s configuration centroids show a systematic deficit relative to
complete evaluated NIST manifolds. The independent two-configuration ground
calculation (1s2 2s2 + 1s2 2p2, six CSFs) supplies almost the same energy scale.
The collector verifies that its DHF ground baseline exactly matches the
production ground DC energy before making this comparison.

| Ion | Repaired DHF minus NIST (eV) | Ground mixing gain (eV) | Ground-only diagnostic remainder (eV) |
|---|---:|---:|---:|
| Be I | -1.200156 | 1.192262 | -0.007894 |
| C III | -1.988433 | 1.969879 | -0.018555 |
| O V | -2.676092 | 2.663562 | -0.012530 |
| Ne VII | -3.344495 | 3.323551 | -0.020944 |

This supports missing ground-state correlation as the dominant contribution
to these four 2s3s excitation deficits. It also explains why making an excited
DHF state numerically stationary can move its excitation farther from NIST:
the ground endpoint remains an uncorrelated configuration average.

The same ground-only diagnostic leaves larger residuals for 2p3p: -0.555 eV
(C III), -0.879 eV (O V), and -1.314 eV (Ne VII). Ground correlation alone
therefore does not explain all excited configurations. Ne VII 2s4s leaves
-0.200 eV. These are targeted diagnostic observations, not extrapolated error
bounds for the periodic table.

## Limits

No excited-state correlation is included in the diagnostic; the ground active
space is only two configurations, and Breit, QED and recoil have not been
recomputed on the correlated ground state. The shifted residuals are not
production predictions, not uncertainty estimates, and not experimental
accuracy scores. A consistent correlated spectrum requires treating both
endpoints and their corrections.

Artifacts: `scripts/diagnose_be_like_excitation_gap.py` and
`output/nist_spectra/be_like_excitation_gap_diagnostic.json`. The interactive
website plots the observed excitation deficit against the independent mixing
gain, with a line marking equal magnitudes and separate excited-configuration
series. All selection and energy gains are independent of NIST fitting.
