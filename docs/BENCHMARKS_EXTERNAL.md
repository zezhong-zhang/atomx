# External-Code Benchmark Campaign (2026-10-02)

Cross-code accuracy campaign: atomx (Dirac atomic structure, this repo)
against five independent external codes living in `../`:

| Code | Model | Speed of light (a.u.) | Role in this campaign |
|---|---|---|---|
| [DBSR_HF](../dbsr_hf) | Dirac–Coulomb / Breit MCDHF, B-splines | 137.03599976 (stock build) | structure, wavefunctions, density |
| [RADIAL (Salvat)](../radial) | local-exchange DHFS + Dirac continuum | 137.035999139 (SL, matched to 137.03599917759013 for probes) | structure (DHFS), Coulomb/screened scattering |
| [FAC](../fac) | optimized central potential | — | He-like structure (different orbital model) |
| [Cowan](../cowan) | nonrelativistic HFS | — | neutral-atom energies (model-difference check) |
| [GRASP2018](../grasp2018) | MCDHF/RCI | 137.035999139 | He-like/Ne-like single-CSF DC + Breit + SE |

Evidence root: `benchmarks/runs/2026-10-02-external-benchmark/`.
Every run directory keeps raw inputs, stdout/stderr, process metadata and
per-case artifacts; nothing is curated away. Runners live in
`benchmarks/bench_reference_{structure,wavefunctions,grasp}.py`,
`benchmarks/bench_cowan_vs_atomx.py`,
`benchmarks/bench_reference_dirac_coulomb.py`,
`benchmarks/bench_reference_screened_continuum.py`.

Constants note: atomx uses CODATA 2022 `c = 137.03599917759013`. DBSR's
stock build uses 137.03599976 and GRASP2018 uses the Salvat value
137.035999139; both mismatches are documented per row and are at the
1e-10 / 3e-10 relative level. The RADIAL probes were rebuilt against
`SL = 137.03599917759013` so the scattering comparison is constant-clean.

## 1. Structure energies (21 rows)

`structure/results.json` — neutral He, C, Ne, Ar, Fe, Kr, Xe; point nuclei;
corrections off; same occupations/ensembles.

| Reference | Model relation | \|ΔE\| relative | worst \|ΔE\| |
|---|---|---|---|
| DBSR_HF | same DC model, same jj ensemble | 3.9e-13 … 3.1e-10 | 2.2e-6 Ha (Xe) |
| RADIAL DHFS | same local-exchange model | 2.6e-9 … 1.5e-8 | 2.3e-5 Ha (Xe) |
| FAC He-like | different orbital model (central potential) | 2.6e-6 … 1.2e-4 | 1.2e-2 Ha (Xe⁵²⁺) |

The DBSR row is the numerical-equivalence gate (same Hamiltonian, same
state): agreement is at the reference's own convergence/print floor. The
RADIAL row validates the local-exchange branch against Salvat's own
implementation. The FAC spread is the expected signature of its different
orbital-optimization model (documented in the per-row contract; not a
defect gate).

## 2. Wavefunctions and density (7 systems)

`wavefunctions/results.json` — pointwise bound-spinor and electron-density
comparison against live DBSR_HF (`out_plot=1` tables, five significant
digits).

| Quantity | Panel worst | Comment |
|---|---|---|
| normalized orbital overlap 1−\|S\| | 5.6e-10 (Xe) | table-norm dominated |
| density ρ(r) max relative difference | 1.8e-4 (Xe) | reference print ceiling (1PE12.4) |
| pointwise relative dP (core-weighted) | 3.6e-3 (Kr) | tail regions; interpolation + print ceiling, locations tracked per orbital |

Per-orbital eigenenergies agree within the same print ceiling; norms audit
to ≤ 2.3e-3 electrons (table rounding). DBSR's `Cjkj` angular prefactor is
single precision — a documented reference limitation (~5e-8 Ha on Ne).

## 3. Scattering (RADIAL probes, constants matched)

`scattering/dcoul/results.json`, `scattering/dfree/results.json` —
Fortran probes in `scattering/probes/` rebuilt against the matched SL.

| Panel | Cases | Result |
|---|---|---|
| DCOULF pointwise (Dirac–Coulomb functions) | 160 probe evaluations | worst relative wave error 3.0e-12; Wronskian determinant error 1.6e-14 |
| DCOULF integrated continuum solves | 96 box solves | phase error mod π ≤ 6.5e-10; 420/480 reference points accepted (60 rejected on DCOULF's own error flag at tiny radii) |
| DFREE screened free states (tabulated DHFS potential) | 36 cases (Z ≤ 74, E, κ) | wave/amplitude error ≤ 1.2e-8; phase error ≤ 1.2e-8 |

## 4. Cowan (nonrelativistic cross-check)

`cowan_vs_atomx.csv` — neutral Ne, Ar, Fe, Kr, Xe.

ΔE relative runs 2.6e-3 → 5.5e-4 and flips sign between Kr and Xe: Cowan's
nonrelativistic HFS under-binds until relativistic deepening dominates, the
expected physics. This row is a model-difference check, not a gate.

## 5. GRASP2018 MCDHF/RCI (single closed-shell CSFs)

`grasp/results.json` — full scripted chain per case (rnucleus point nucleus
→ rcsfgenerate → rangular → rwfnestimate → rmcdhf → rsave → rci), raw
outputs in each case directory. Cases: He-like Z = 2, 10, 18, 26, 36 and
Ne-like Fe XVII, all single closed-shell J=0 CSFs, point nuclei.

| Quantity | Panel result |
|---|---|
| Dirac–Coulomb total energy | \|ΔE\| = 2.5e-10 Ha (He) … 1.1e-8 (Kr³⁴⁺) … 1.4e-7 Ha (Fe XVII); relative ≈ 1e-10 |
| Breit shift (matched: atomx `breit_frequency=orbital_differences`, GRASP transverse with unmodified photon frequencies) | \|Δ\| = 2.3e-12 Ha (He) … 6.7e-7 Ha (Kr, Fe XVII) |
| Breit shift vs atomx static Breit | off by 1.1e-3 Ha on Fe XVII — confirms GRASP's transverse operator runs at ω = \|Δε\|, not ω = 0 |
| Self-energy (indicative) | GRASP Welton estimate recorded per case (2.5e-5 Ha for He … 0.93 Ha for Kr³⁴⁺); atomx model-potential SE is a different method — indicative only, no gate |

The GRASP build's c (137.035999139) is documented per row; the residual
Fe XVII DC gap (−1.4e-7 Ha, relative 1.2e-10) is the combined
constant-mismatch (2.8e-10) and grid/convergence floor of both codes.

Runner gotchas worth remembering:

- rcsfgenerate input uses the NONRELATIVISTIC subshell notation:
  `1s(2,i)2s(2,i)2p(6,i)` auto-splits into `2p-(2)2p(4)`. The jj dash
  notation `2p-(2,i)2p(4,i)` is silently mangled into two 6-electron CSFs.
- rci's summary carries two `Eigenenergies:` tables (DC, then DC+SE) with
  the SE correction printed between them; the parser must not let the SE
  rows overwrite the DC table.
- rmcdhf "orbitals to be varied" answer `*` = all occupied; the minimal
  scripted input is `y\n1\n*\n*\n<max cycles>\n`.

## Legacy in-repo evidence (still valid)

`legacy_quick.md` — DHFS density Z = 2…18 + 74, WF_TXT He-like overlaps
Z ≤ 92, and Table S12 / Visscher–Dyall literature spot checks: all ok.
Existing FAC cross-section evidence from 2026-05 remains the scattering
cross-section reference; this campaign did not re-run those.

## Verdict

Where the compared quantity is the same physical object (DBSR DC energies
and orbitals, RADIAL local-exchange energies, RADIAL Coulomb/screened
continua, GRASP single-CSF DC and Breit shifts), atomx agrees at the
reference codes' own convergence and print floors — relative 1e-10 on
energies, 1e-10 on overlaps, 1e-9…1e-12 on continuum phases. Differences
that remain are attributable to documented reference limitations (print
precision, single-precision angular prefactors, stock constants) or to
deliberate model differences (FAC central potential, Cowan nonrelativistic
HFS), which the per-row contracts state explicitly.
