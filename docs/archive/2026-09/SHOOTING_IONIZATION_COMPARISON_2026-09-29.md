# Shooting ionization comparisons: first20

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The campaign comparison service now includes adjacent-charge ground-state
ionization energies. The interactive campaign viewer adds an **Ionization
limits** record mode, using the existing element/charge/status filters,
residual plot and record inspector.

Every comparison requires two accepted radial grids for each non-bare
endpoint. A fully stripped ion has exactly zero electronic energy; no fictitious
orbital calculation is supplied. Nuclear identity, shared correction requests
and self-energy screening conventions must match. The one-electron leading
radiative-mass addition remains restricted to its supported stages. The record
retains the bare energy difference and every post-SCF contribution separately.
NIST ground configurations must match both queued endpoints. Missing or
mismatched endpoints remain visible without scored residuals. Recovery source
identities are retained when comparing a historical campaign plus recoveries.

The NIST reference row is embedded intact, including uncertainty, raw notation,
classification and references. A theoretical or semiempirical ionization limit
is not relabelled as an experimental measurement. Configuration averages are
also distinguished from unique-level configurations. Even the latter are not
certified isotope-matched comparisons; that limitation is explicit in each
record and no global strict experimental RMS is computed here.

## Current-correction ground-state findings

All 210 stages from H through Ca have accepted ground-state pairs in the
current-correction snapshot. The NIST reference classifications are:

| Classification | Stages |
|---|---:|
| Evaluated, both configurations unique | 29 |
| Evaluated, at least one configuration average | 39 |
| Theoretical limit | 48 |
| Semiempirical limit | 94 |

Selected neutral-ionization residuals (Atomx minus NIST, cm⁻¹):

| Species | Residual | Comparison basis |
|---|---:|---|
| He I | −9230.311 | Unique configurations, evaluated limit |
| Li I | −403.496 | Unique configurations, evaluated limit |
| Be I | −10308.416 | Unique configurations, evaluated limit |
| Mg I | −8325.384 | Unique configurations, evaluated limit |
| Ca I | −7898.173 | Unique configurations, evaluated limit |
| H I | −0.0000624 | **Theoretical** NIST limit |

Large residuals cannot be interpreted as radial-grid errors solely because a
calculation is converged; differential correlation and resolved-state modelling
remain necessary. Neither the model corrections nor the new classification
changes the experimental database. These are current DHF diagnostics, not a
claim of full agreement with NIST.

Tests cover charge-state subtraction, the bare-nucleus endpoint, theoretical
reference separation, nuclear-mass mismatch, incorrect reference configuration,
pending endpoints and open-shell configuration averages. The related
comparison/route suite passes 13 tests. Viewer JavaScript syntax passes;
rendered-browser inspection remains outstanding.
