# Resolving the heavy-ion shooting discrepancy

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Diagnosis

The previously reported uranium difference, about −0.00237 Ha, compared
different electron projectors. The enriched grid calculation constrained
its orbitals to the positive electron branch of the nuclear Dirac operator.
The separate HF solver solved a self-consistent occupied radial equation.
Those are not automatically the same projected Hamiltonian, even with
identical Dirac–Coulomb energy integrals and converged radial grids.

The diagnostic separates energy evaluation from the electron-space choice:

1. Evaluate the HF spinors using the grid CI functional on their original
   HF grid. For uranium the energy difference is −1.49e-10 Ha. The energy
   integrals do not explain the 0.00237 Ha discrepancy.
2. Keep the CI Hamiltonian, grid and configuration fixed. Change only the
   local reference that generates the electron-space basis.
3. For an occupied `1s2` HF spinor, direct `2J` minus exchange `J` gives the
   radial reference potential `V_nuc + J[1s]`. This places the converged HF
   orbital inside the reference electron space.
4. Repeat the constrained optimization. Uranium then differs from HF by
   −1.66e-10 Ha, consistent with the independently measured energy-evaluator
   difference. No total-energy offset or experimental data enters the fix.

The local `1s2` identity applies to the occupied equation. It is **not** the
nonlocal Fock operator acting on arbitrary virtual orbitals, nor an
exchange approximation for general open shells. A fixed nuclear projector
remains an explicit alternative model, rather than an invalid numerical
solver. Matching projector conventions is necessary when comparing models.

## Implementation

`bound_electron_space` and `continuum_enriched_electron_space` now accept
`reference_rv` and a mandatory `reference_label`. The potential is copied,
made read-only and fingerprinted with its radial grid. The reference is
used for bound roots, continuum roots, diffuse negative box roots and the
Green response preconditioner. It never replaces the CI Hamiltonian's
one-body potential or its Coulomb interaction.

`occupied_1s2_reference_potential` constructs the local occupied-equation
reference from a supplied normalized 1s spinor. It validates the
two-electron scope and normalization. The caller must establish that the
input spinor is the intended self-consistent reference; the function does
not silently assume convergence or infer a general open-shell operator.

Optimizer diagnostics now retain the electron-reference label and a
SHA-256 fingerprint. The continuum benchmark exposes `--reference` and
`--bound-roots`; comparisons using the nuclear projector are explicitly
labelled as model differences, not solely numerical errors.

## Matched controls

All controls use `1s2`, a Fermi nucleus, infinite nuclear mass and the
Dirac–Coulomb Hamiltonian. Breit, QED and recoil are excluded. The HF
reference uses 4,000 grid points and a 1e-13 energy-change tolerance. Both
electron spaces use the same independent 4,000-point CI grid, two localized
roots, the remaining negative box roots and 160 positive box roots.

| Z | Nuclear-projector energy minus HF (Ha) | Matched-reference energy minus HF (Ha) |
|---:|---:|---:|
| 2 | +1.58e-6 | −1.43e-12 |
| 6 | +3.47e-6 | −3.78e-12 |
| 26 | −1.47e-5 | −7.94e-11 |
| 54 | −2.96e-4 | −4.96e-10 |
| 92 | −2.37110e-3 | −1.66e-10 |

All optimizations pass the same 1e-6 projected-gradient tolerance.
For uranium, the HF orbital's norm outside the supplied space decreases
from 1.55e-4 to 1.98e-11 when the reference is matched. Using 48 instead of
160 positive roots in the matched control changes its energy by about
4e-12 Ha. The excellent agreement is a matched HF control, not a statement
of experimental or complete correlation accuracy.

The matched initialization requires a converged HF reference. Its zero to
two subsequent optimization iterations are not an overall speedup claim;
the reference computation still costs work.

The same explicit reference was also used for even-J=0 correlated He and
C V controls with `1s2`, `2s2`, `2p2` CSFs. All four 16/48-continuum-root
runs converge below the 1e-6 projected-gradient tolerance, with 127–318
iterations. The 48-root energies are −2.876371333100 Ha and
−32.391310682233 Ha respectively. These remain finite-space correlation
results, not complete-space benchmarks; the raw records are in
`output/correlation_benchmark/shooting_reference_correlated.json`.

## Tests and reproduction

**28 targeted tests passed**: matched He/Fe/U controls, unchanged physical
Hamiltonian, immutable reference data, provenance, normalization and scope
guards, plus continuum, full-grid derivative, Green-response and warm-start
regressions.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/benchmark_shooting_reference.py \
  --output output/correlation_benchmark/shooting_reference_match.json
```

The report includes energy evaluation on identical orbitals, projection
leakage, both optimizations, actual convergence status and source hashes.
The general continuum CLI also accepts:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/benchmark_shooting_continuum_space.py \
  --z 92 --nuclear fermi --reference occupied_1s2_hf --bound-roots 2 \
  --continuum 48 160 \
  --output output/correlation_benchmark/shooting_reference_cli.json
```

## Remaining scope

The measured `1s2` heavy-ion mismatch is resolved. General open-shell
nonlocal reference operators, correlation-space convergence and production
correction/result integration still require validation. Production defaults
remain unchanged; no full-table campaign has been launched from this work.
