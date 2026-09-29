# Leading one-loop radiative mass terms — 29 September 2026

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

Implemented a separate finite-minus-infinite-nuclear-mass correction through
order α(Zα)⁴, including the leading self-energy logarithm. It is an explicit
addition, not a replacement for ordinary recoil or a claim of complete QED.

The kernel is `corrections/radiative_mass.py`. Its adapter,
`HydrogenicRadiativeMassLeadingCorrection`, integrates with `CorrectionStack`
for identified one-electron states, records separate self-energy and
vacuum-polarization contributions, and updates both total energy and the QED
ledger. It rejects many-electron states. This makes it available for energy
calculations generally, not just wavelength postprocessing.

## Physical scope

For μ=M/(M+1), leading self-energy terms scale as μ³ except the magnetic A40
term, which scales as μ²; the s-state logarithm also changes by log(1/μ).
The leading Uehling coefficient scales as μ³. Only the difference from the
infinite-mass terms is added. Stable `log1p`/`expm1` expressions preserve the
small difference for large nuclear masses. Bethe logarithms come from the
existing NIST TN 1467 table.

Reference: [CODATA 2018, Eqs. (14)–(16), pp. 10–11](https://pml.nist.gov/cuu/pdf/JPCRD2018CODATA.pdf).
Higher powers of Zα, two-loop mass effects, finite-size radiative mass effects,
additional radiative-recoil operators, and many-electron radiative recoil remain
omitted. These limitations are stored with the correction. The contribution is
explicitly selected with `--radiative-mass-leading` in the resolved benchmark;
it is not silently enabled in the full-table default.

## Measured result

The separate frozen run in `output/nist_spectra/shooting_one_electron_radiative_mass`
completed all 436 grid attempts for 218 states. It retains all 454 observed
records: 337 have classified J-resolved comparisons and 117 lack sufficient J.

| Diagnostic | Before addition | With leading radiative mass |
|---|---:|---:|
| H I 2p₁/₂ → 2s₁/₂ observed residual / cm⁻¹ | +0.00006093201 | +0.00000641749 |
| Mg XII 1s → 6p₃/₂ Ritz residual / cm⁻¹ | +0.283912 | +0.396527 |

For hydrogen the new interval contribution is −0.00005451452 cm⁻¹, composed
of −0.00005599141 from self-energy mass dependence and +0.00000147689 from
vacuum-polarization mass dependence. No NIST values enter those formulas.
The reduction is about 9.5-fold, but the remaining discrepancy exceeds the
two-grid change (~7.9e-9 cm⁻¹). The Mg result illustrates that adding a known
term need not improve a partially cancelled residual at every Z.

The immutable run manifest used the older generic scope wording “same isotope
and corrections as parent”; its explicit `radiative_mass_leading: true` and
per-state correction ledgers establish the additional term. The reviewed
report corrects the scope description while retaining and verifying the
original manifest hash. The runner's wording is fixed for future runs.

## Interface and verification

The website snapshot selector includes “J-resolved: leading radiative mass
terms”. Observed and Ritz residuals remain separate; shared-intensity and
repeated-feature metadata are preserved. Tests cover the analytic large-mass
limit, the distinct magnetic scaling, extreme-mass numerical stability,
invalid-state rejection, total/QED ledger updates and duplicate-application
prevention. Browser JavaScript syntax passed; no rendered-browser inspection
was performed.
