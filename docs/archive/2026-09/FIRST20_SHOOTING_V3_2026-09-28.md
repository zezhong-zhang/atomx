# H–Ca shooting validation with the current correction model

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The earlier shooting benchmark used the historical scalar-Breit ledger.
`scripts/benchmark_first20_shooting.py` now requires the current v3 ledger
and explicitly uses static DBSR Breit, model QED, relativistic NMS/SMS,
the one-electron mass remainder, and the selected isotope nuclear profile.
It checks that the neutral and ion use the same isotope and nuclear mass.
Historical benchmark files remain available separately.

## Results

- **20/20 neutral shooting seeds validated**, no failures or branch mismatches.
- Largest final-energy difference from direct B-spline DHF: **4.823e-9 Ha**.
- Largest shooting-to-B-spline projection L2 error: **2.296e-5**.
- Summed calculation time on this host: **36.83 seconds**, one BLAS thread.
- Current-model neutral ionization RMS error: **8.6249%** across 19 evaluated
  references (unique levels or complete configuration centroids). Hydrogen's
  theoretical NIST ionization is excluded from that experimental score.

Shooting supplies the initial orbitals; the final nonlocal DHF solution and
corrections are calculated with the B-spline solver. These results validate
that numerical route for the 20 neutral ground configurations. They do not
validate all excited states or ion stages, and do not establish that a pure
shooting treatment has equivalent open-shell constraints.

The remaining ionization discrepancy is much larger than the difference
between the two numerical starting routes. Missing many-electron correlation
and resolved-level physics remain essential; numerical agreement alone does
not justify a claim of experimental or high-order QED accuracy.

## Website and provenance

The shooting panel now loads `first20_shooting_v3.jsonl`. Each row includes
the physics version, neutral energy components, ion baseline, and correction
difference from the same model. The builder no longer joins the old scalar
Breit error budget onto these current rows.

The comparison builder completed, its nine tests passed, and the viewer
JavaScript passed Node syntax checking. Numerical data:
`output/nist_spectra/first20_shooting_v3.jsonl` and
`output/nist_spectra/first20_shooting_v3_summary.json`.
