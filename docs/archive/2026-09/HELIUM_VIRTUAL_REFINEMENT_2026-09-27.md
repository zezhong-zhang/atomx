# Helium virtual-orbital refinement

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Result

Repopulating weak virtual orbitals and optimizing the same 30-CSF, J=0 even,
n≤4 Dirac–Coulomb problem lowers its energy from −2.9019746292548843 to
−2.9030429550720642 Ha. No experimental energy enters that optimization.

The initial optimization stopped after 212 iterations without meeting the
orbital tolerance. A restart from its saved state met the unchanged 1e−5
preconditioned orbital-residual tolerance in two iterations (9.5773e−6),
with energy unchanged to approximately 1e−14 Ha. The optimizer now reports
its termination reason and retries a failed secant line search using
preconditioned steepest descent. The successful restart did not need that
fallback; it reset the secant history and rebuilt initial scaling.

Three finite f-orbital replacements, evaluated with the original Hamiltonian,
raise the refined energy by 0.0105062, 0.0103201 and 0.00711908 eV. These are
the same trials that exposed the original false plateau. Passing these
limited trials does not prove a global minimum or active-space convergence.

## Corrections and NIST comparison

Corrections were recomputed on the refined orbitals and Gaunt-CI amplitudes,
including coherent local vacuum polarization, local model self-energy, and
one- and two-electron Breit recoil. The He II endpoint uses the same nuclear
mass and self-energy model.

| Quantity | Value |
|---|---:|
| Calculated ionization energy | 24.56532881583165 eV |
| Matched NIST reference | 24.587389011 eV |
| Theory − NIST | −0.02206019516835056 eV |
| Maximum radial overlap error | 1.78e−15 |
| Maximum nuclear-projector leakage weight | 1.18e−29 |

This diagnostic remains a finite active-space calculation. Neutral CI includes
Gaunt but lacks Breit retardation. Self-energy is a local model, not full
bound-state QED. Recoil is the point-Coulomb Breit operator at first order in
inverse nuclear mass; finite-size, radiative and higher-order recoil are absent.
The residual cannot be assigned entirely to correlation from this calculation.

## Reproduction and evidence

- `scripts/optimize_he_populated_virtuals.py --source he_populated_optimization.json --output-stem he_populated_refinement --max-iter 150`
- `scripts/audit_he_virtual_stability.py --source he_populated_refinement.json --output-stem he_refined_virtual_stability`
- `scripts/audit_he_refined_corrections.py`
- Outputs in `output/nist_spectra/`: `he_populated_refinement.json`,
  `he_refined_virtual_stability.json`, `he_refined_corrections.json`.
- Snapshot SHA-256:
  `fab1d012b149e0174dafc15064618fb5140c131436ca398e02e57bf7922111eb`.
- Real-helium regression covers energy lowering, projector preservation,
  forced failed-search recovery, and explicit failure without false convergence.

The comparison website includes this result separately from the historical
false-plateau calculation. Full-table generation remains held; this refinement
does not establish completion of the requested all-element database.
