# Corrected DHF controls across the table — 2026-09-27

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

Fresh neutral and first-ion calculations use shooting HF seeds followed
by converged B-spline DHF, static DBSR Breit, model QED, a Fermi nucleus,
and isotope-specific one- and two-body Breit recoil. The isotope identity
is identical for both endpoints. All runs retained the required correction
components and passed the solver's orbital-order/node checks.

| Element | Matched NIST reference | Theory IE (eV), base grid | NIST (eV) | Residual (eV) | First grid change (micro-eV) |
|---|---|---:|---:|---:|---:|
| Kr | Evaluated complete configuration centroid | 13.2341954230 | 14.2215413150 | -0.9873458920 | -0.098970 |
| Xe | Evaluated complete configuration centroid | 11.6638030108 | 12.5653179071 | -0.9015148963 | +0.111245 |
| Hg | Evaluated unique ground levels | 8.9124830948 | 10.437504 | -1.5250209052 | -5.501616 |

The initial repeat halves the inner-grid parameters to 0.125. Hg was
repeated at 0.0625: IE=8.9124761478 eV, a further -1.445418 micro-eV.
This trend does not constitute an extrapolated uncertainty bound. It does
show that the observed eV-scale discrepancy is much larger than these
grid changes. Numerical agreement alone does not establish the accuracy
of the physical approximation.

The DHF model omits electron correlation beyond exchange, full many-body
QED, finite-size recoil, radiative recoil and higher-order recoil. These
controls therefore do not justify a parts-per-billion claim. Kr and Xe
must be compared to complete ionic configuration centroids; replacing
those with the lowest fine-structure threshold would compare different
observables. The unique-level Hg reference needs no such centroid shift.

NIST reference identifiers retained in the JSON are L12196 (Kr), L12013,L9794,L6426 (Xe), and L11931/L11984 (Hg). Source values
come from the previously audited NIST records; no fit to them is used.

Reproduction: `benchmark_corrected_periodic_controls.py`, followed by
`refine_mercury_control.py`. Full component energies, isotope masses,
configurations, solver routes and per-grid results are preserved in
`output/nist_spectra/corrected_periodic_controls.jsonl`. These controls
remain separate from the all-table worker, which is still held.
