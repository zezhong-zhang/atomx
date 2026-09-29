# Deuterium numerical and spectral residual audit

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

`scripts/audit_deuterium_grid.py` independently refines 1s, 2s and 8s with
the production finite nuclear model, named deuterium mass and corrected
self-energy fallback. The enlarged outer box, halved outer spacing and
finer inner mesh leave the Hamiltonian and mass unchanged. Results are in
`output/nist_spectra/deuterium_grid_audit.json`; production energies are not edited.

| Excitation | Original residual (cm^-1) | Refined residual (cm^-1) | Grid shift (cm^-1) | NIST uncertainty (cm^-1) |
|---|---:|---:|---:|---:|
| 1s–2s | -0.0001135344 | -0.0001132693 | +0.0000002651 | 0.000000005 |
| 1s–8s | -0.0001338269 | -0.0001335256 | +0.0000003013 | 0.00000023 |

The finite-model grid shifts are about 400 times smaller than the spectral
residuals. The largest changing component is the one-body recoil expectation.
These spreads are convergence evidence, not rigorous numerical uncertainty bounds.

## Independent point-nucleus check

A separate uncorrected point-Dirac solve was compared with the analytic
hydrogenic energies on the refined mesh. Its very small inner spacing makes
low-lying eigenvectors sensitive to the full eigensolver's numerical error.
The existing requested-root Newton refinement, normally applied only to
bound indices >=10, was enabled for all requested roots **inside this diagnostic**.

| State | Default point solve error (cm^-1) | Refined requested-root error (cm^-1) |
|---|---:|---:|
| 2s | +0.0000733966 | +0.0000002686 |
| 8s | -0.0013187036 | -0.0000001794 |

The control demonstrates a numerical limitation in the point-nucleus path,
and a working remedy. It does not justify changing the production finite-nucleus
ledger or claiming that its remaining spectral discrepancy is numerical.

## Outstanding physical budget

The deuterium charge model still uses a 2 fm clamped Fermi radius with an
element-level reference value rather than a measured deuteron charge distribution.
The high-n self-energy fallback restores leading A41/A40 terms but omits higher
orders. Other missing contributions include consistent finite-mass scaling of
radiative terms, higher-order recoil, higher-loop QED and nuclear structure.
Neither 1s–2s nor 1s–8s agrees within the quoted NIST uncertainty. These controls
are one-electron tests and contain no electron-electron correlation.
