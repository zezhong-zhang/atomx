# Cowan–Atomx–NIST pilot — 29 September 2026

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## What ran

Local source is `/home/zzhang/code/Cowan` (capital C). Built LANL RCN,
RCN2 and RCG from the local Fortran, then ran Mg I, Ca I, Ca II and Cs I:
three configurations per species, at CORRF=0 and CORRF=1. All eight runs
completed normally, with no RCN convergence warnings. This is a selected
centroid diagnostic, not a full-table or resolved-line validation.

- RCN: IHF=2, IREL=1 (Cowan HFR), no model QED.
- RCN2: all five integral scales exactly 1. No RCE fitting, empirical energy
  replacement, or fitted effective-operator parameters.
- RCG: the same explicit configurations; open s/p/d fractional-parentage
  decks only. Full unused f/g/h decks overflow the standard build's dimensions.
  The restricted deck contains every open-shell type in this pilot.
- NIST: existing local evaluated-level archive, complete statistical-weight
  manifolds required; references and uncertainties remain in that archive.
  This pilot does not newly download or independently verify all NIST data.
- Atomx: historical accepted first20 configuration centroids, plus the
  explicitly identified refined Ca I 4s6d diagnostic. Cs I is unavailable in
  that selected Atomx campaign. Historical predictions were not regenerated.

RCG levels and transition output are archived, but only configuration centroids
are scored in this first comparison. The models are not identical: Cowan HFR
uses approximate relativistic HF, while these Atomx records use Dirac HF with
Breit, model QED and recoil. A difference between codes is not automatically a
numerical error.

## Measured results

All entries below are excitation configuration centroids in cm⁻¹, CORRF=0.
“Exported” includes Cowan's perturbative correlation energy.

| Species / configuration | NIST | Atomx | Cowan before correlation | Cowan exported |
|---|---:|---:|---:|---:|
| Mg I 3s4s | 41773.886 | 34060.605 | 34638.582 | 38463.466 |
| Mg I 3s3p | 25180.957 | 20483.909 | 20515.390 | 21254.638 |
| Ca I 4s6d | 45035.754 | 37568.777 | 37627.826 | 42200.202 |
| Ca I 4s4p | 17360.393 | 13241.921 | 13300.162 | 13991.797 |
| Ca II 3d | 13686.604 | 16514.509 | 16643.858 | 14371.593 |
| Ca II 4p | 25340.103 | 23588.095 | 23657.169 | 24914.691 |
| Cs I 5d | 14557.808 | unavailable | 13802.759 | 14517.919 |
| Cs I 6p | 11547.627 | unavailable | 9502.154 | 11008.242 |

The printed bare ETOT totals have five decimals in Rydbergs: differences have
about 1.1 cm⁻¹ rounding resolution. RCN2's relative Eav listing has higher
precision. The pilot retains Cowan's own 109737.31 cm⁻¹/Ry conversion.

## Important finding: CORRF is not a total-correlation switch

The local RCN manual, Section II.I, explains that CORRF controls the approximate
correlation potential in the radial equations. RCN `OUTPT` always builds the
exported energy from `et4cr`; its new correlation term `ect` is included even
when CORRF=0. The seven printed ETOT columns are preserved separately by
`reference_data/cowan.py`.

For Ca I 4s6d, the exported correction raises the excitation by 4572.376 cm⁻¹,
reducing the Cowan residual from −7407.928 to −2835.552 cm⁻¹. Changing the
orbital potential to CORRF=1 gives 42227.647 cm⁻¹, an additional 27.445 cm⁻¹.
Calling either exported result “uncorrelated HF” would misidentify the physics.
For Cs I 5d, CORRF=1 worsens the exported agreement (14145.970 rather than
14517.919 cm⁻¹); the approximation is not uniformly beneficial.

## What this identifies in Atomx

1. **The present DHF campaign lacks differential many-electron correlation.**
   For several states the two uncorrelated mean-field descriptions have similar
   large residuals, and Cowan's approximate correlation energy explains a
   substantial part of the improvement. This is evidence for correlation work,
   not proof that Cowan's particular local model is the correct replacement.
2. **Use and converge the existing correlated machinery.** Atomx already has
   CI/MCDHF and second-order MBPT code; they are not equivalent to having
   converged state-specific correlation in this DHF campaign. Compare successive
   valence SD, core–valence and core–core spaces, checking both endpoint energies,
   state mixing and radial/projector cutoffs.
3. **Resolved spectra need resolved correlated states.** Configuration-average
   scores cannot diagnose singlet–triplet splitting or mixing. The next Cowan
   comparison should match RCG levels by configuration, J, parity and dominant
   LS content, retaining ambiguity instead of matching nearest energies.
4. **Some capabilities really remain incomplete.** `methods/coupled_cluster.py`
   is an explicit scaffold. The existing Breit–Pauli wrapper uses an HF–Slater
   base with hydrogenic shifts, so it is not a faithful Cowan HFR counterpart.
   These are separate from errors in the relativistic DHF implementation.
5. **Keep investigating the numerical/model residual.** Mg I 3s4s has a larger
   Atomx–bare-Cowan separation (~578 cm⁻¹) than the other light pilot cases.
   The historical Atomx run needs a strict orbital-stationarity/state audit and
   matched-model control before attributing that separation to a specific term.

## Reproduction and provenance

Run from the repository:

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python scripts/benchmark_cowan_nist.py --rcg
```

The runner accepts `--cowan`, `--binaries`, `--archive` and `--output`. It fails
on abnormal termination, RCN convergence warnings, incomplete energy ledgers,
or non-unit radial scales. Default binaries are in `output/cowan_benchmark/build`.
Inputs, raw output, total-energy ledgers, binary/source hashes, NIST archive hash,
and the original Atomx campaign identity are retained in
`output/nist_spectra/cowan/comparison.json` and its eight run directories.

Build used cached GNU Fortran 16.2 with `-O2 -std=legacy -fdec
-fallow-argument-mismatch -fno-align-commons`, then system GCC linking to the
cached libgfortran. Compiler tools were assembled under
`/tmp/atomx-cowan-toolchain`; the source installation was not modified. RCG
needed one portability change: close preconnected unit 5 before reopening it
as scratch. The patched source and hash are retained; no physics expression
was modified. Build commands and logs are under `output/cowan_benchmark/build`.
RCN and RCN2 compiled without source modifications.

Website: `/comparison/cowan/`, linked from the native shooting campaign page.
It filters by species and CORRF, plots signed residuals and exposes each raw
comparison record. Cs I's missing Atomx values stay blank. No visual browser
inspection was performed; JavaScript syntax and HTTP routes were checked.

## Re-run and common-configuration accuracy summaries

Re-ran all eight pilot RCN/RCN2/RCG jobs and all four ordering-panel
RCN/RCN2 jobs successfully against the current cached NIST archive. This
updates the archive hash after source-field enrichment; it is not a new
live NIST download. Atomx values remain the explicitly identified historical
and refined records above, not a new Atomx calculation.

The report now includes `common_cohort_summary`, separately for each CORRF.
Only excited configurations with finite predictions from all three models
enter these metrics. Ground reference zeros are excluded. Missing Atomx Cs I
predictions remain visible but do not enter the three-model score. The website
computes the same common-configuration MAE for the current species filter.

| Panel, CORRF=0 | Common excited configurations | Atomx MAE | Bare Cowan MAE | Exported Cowan MAE |
|---|---:|---:|---:|---:|
| Original pilot | 6 | 4762.615 | 4651.536 | 2425.215 |
| Mg / Al ordering panel | 4 | 5842.347 | 5828.569 | 1915.807 |

Errors are in cm⁻¹. These small selected samples are not full-periodic-table
statistics, and the models have different Hamiltonians. An improved mean
excitation error does not imply correct relative ordering: the Al ordering
failure persists and worsens with Cowan's approximate correlation correction.

### Work identified, in priority order

1. **Converge differential correlation for both transition endpoints.** Start
   with Mg and Al: explicit multireference CI/MCDHF, then enlarge valence and
   core–valence spaces with numerical convergence checks. Existing correlated
   code is not evidence that the present DHF data include those correlations.
2. **Add resolved RCG comparison.** Parse J, parity and eigenvector composition;
   match only unambiguous NIST states. RCG outputs are archived but currently
   unscored. Configuration centroids cannot substitute for this validation.
3. **Investigate the Mg 3s4s mean-field discrepancy.** Its approximately
   578 cm⁻¹ Atomx–bare-Cowan separation needs a matched-model and stationary
   orbital check; the current benchmark does not establish its cause.
4. **Fill implementation and coverage gaps explicitly.** Coupled cluster is
   still a scaffold; the Breit–Pauli wrapper is not Cowan HFR; this comparison
   has no Atomx Cs I values. None should be reported as validated capability.

Validation: four parser/metric/route tests pass. Tests cover exclusion of ground
references, missing and nonfinite values, empty common cohorts, and separation
of CORRF variants. Viewer JavaScript passes `node --check`; visual browser
inspection remains outstanding.


## Mg same-l orbital constraint audit

The local Cowan RCN manual, Section I.C, distinguishes RCN from HF8:
HF8 includes off-diagonal Lagrangian parameters for orthogonality of same-l,
different-n orbitals with unequal occupations. The present pilot uses RCN.
We read its original unformatted `tape2n` records using the explicit local
LANL layout (REAL*8, INTEGER*4, 20 shell slots, 1801 mesh slots); incompatible
record lengths are rejected. Only same-l overlaps are tested, because a radial
overlap between different-l orbitals is not a full orbital overlap.

For Mg I 3s4s, the normalized stored 3s–4s overlap is **0.047578982**.
Independent trapezoidal integration gives 0.047582301; Simpson orbital norms
are within 1.5e-7 of unity. The corresponding Atomx shooting spinors have overlap
1.83e-16 under the native radial metric. This is a measured constraint mismatch,
not a fitted explanation of the 557 cm⁻¹ bare-code energy difference.

The new independent B-spline control agrees with bare shooting excitation to
6.87e-8 cm⁻¹. Thus a shooting-specific discretization failure does not explain
this Mg discrepancy. A matched constrained Cowan/HF8 calculation or a common
functional evaluation is still needed to quantify the constraint contribution;
the absence of differential correlation still limits agreement with NIST.
All new Cowan reports include per-configuration radial-overlap diagnostics and
hash the binary radial archive. The website exposes these in each record.

## Explicit campaign selection and fresh Cowan verification

The runner now accepts `--atomx-comparison PATH`. An explicit snapshot disables
historical Ca I diagnostic substitution, records its absolute path and hashes
exactly the bytes parsed (even if another process subsequently refreshes the
snapshot). Missing or unaccepted Atomx configurations remain unavailable.

Re-ran all eight RCN/RCN2/RCG pilot calculations against
`native_shooting/corrected_checkpoint_comparison.json`, writing separate output
under `output/nist_spectra/cowan_current`. All runs completed normally. At the
snapshot used for this run, none of the eight excited pilot configurations had
an accepted Atomx counterpart; the common comparison count is zero, with null
accuracy metrics. This is a coverage limitation, not an accuracy result.
The website `/comparison/cowan/` now offers this explicitly labelled partial
snapshot alongside the historical pilot and ordering panel.

```sh
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/benchmark_cowan_nist.py --rcg \
  --atomx-comparison output/nist_spectra/native_shooting/corrected_checkpoint_comparison.json \
  --output output/nist_spectra/cowan_current
```

Six parser, metric and HTTP route tests pass. The main missing validation remains
converged differential correlation and resolved-state matching; this rerun does
not establish a full-table accuracy result or refresh the entire NIST archive.
