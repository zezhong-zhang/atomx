# Correct Dirac normalization of the Mohr screening factor

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Defect

The Mohr backend multiplies its tabulated hydrogenic self-energy by the ratio
of the calculated nuclear attraction to a hydrogenic reference. The numerator
used relativistic Dirac orbitals, but the denominator was the nonrelativistic
value −Z²/n². Even an unscreened point-Coulomb orbital therefore changed the
tabulated self-energy. For 1s the spurious factor was 1/sqrt(1−(Zα)²).

The corrected denominator is the exact Dirac-Coulomb expectation, derived by
differentiating the Sommerfeld energy with respect to nuclear charge:

```
a = Z alpha
gamma = sqrt(kappa² − a²)
N = n − |kappa| + gamma
<−Z/r> = Z dE/dZ = −Z² (N + a²/gamma) / (N² + a²)^(3/2)
```

It lives in `physics/dirac.py`; `corrections/qedmod/mohr.py` consumes it.
This restores the point-Coulomb unscreened limit without fitting to NIST.
The orbital-overlap screening prescription remains a model for many-electron
and finite-nucleus self-energy. This change does not make it full QED.

## Numerical and provenance changes

The derivative test exposed cancellation in the analytic binding-energy
expression `c²*(1/sqrt(1+x)−1)`. Its rationalized equivalent
`−c²*x/[sqrt(1+x)*(sqrt(1+x)+1)]` avoids subtracting nearly equal numbers.

Self-energy correction provenance now identifies the exact Dirac reference.
Grid-pair and transition comparisons reject mixing that revision with the
historical normalization. Existing frozen campaigns and records remain intact.
The changed Ne/Ar total-QED test pins are explicitly self-regression snapshots;
they were updated because the screening model changed, not as external
validation. The independent physical gates are the exact Coulomb limits.

Eighty-three analytic/QED/comparison tests passed after the normalization and
stable-energy changes. The analytic checks cover five states at Z=1,12,92
through the charge derivative and independently integrate the exact Dirac 1s
radial density at those charges. The backend recovers its unscreened table
value with these wavefunctions.

## Recalculation and remaining mass terms

A fresh two-grid, 218-state J-resolved benchmark completed using frozen source under
`/tmp/atomx-one-electron-dirac-se-20260929`. Outputs are separate in
`output/nist_spectra/shooting_one_electron_dirac_se`. Do not relabel the older
`shooting_one_electron_j` results as containing this correction.

All 436 attempts succeeded. Maximum line-interval change between grids is
9.67e-7 cm⁻¹. For Mg XII 1s → 6p₃/₂ the residual against the separately
labelled Ritz value improves from −8.966913 to +0.283912 cm⁻¹. Against the
shared observed wavelength it changes from +14,457.653753 to +14,466.904578
cm⁻¹; that discrepancy remains visible rather than being replaced by Ritz
agreement. H I 2p₁/₂ → 2s₁/₂ changes from +6.21375e-5 to +6.09320e-5 cm⁻¹
relative to its observed interval. The website exposes both source versions.

The historical recoil stack did not apply finite-mass changes to the
self-energy or vacuum-polarization operators. CODATA describes reduced-mass
scaling of self-energy, a distinct scaling of its magnetic-moment term, and
modified logarithm arguments. These cannot be replaced by one universal
multiplicative factor on the complete correction stack. Implementation and
verification of complete radiative mass terms remain outstanding. A subsequent
[explicit leading-order addition](RADIATIVE_MASS_LEADING_2026-09-29.md) now covers
the α(Zα)⁴ terms for one-electron states.
Primary reference: [CODATA 2018, Section VII](https://pmc.ncbi.nlm.nih.gov/articles/PMC9890581/).
