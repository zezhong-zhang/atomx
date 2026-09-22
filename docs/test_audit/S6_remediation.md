# S6 remediation — plasma / collisional-radiative / spectral synthesis

Scope: the S1 findings of `docs/test_audit/S6_plasma_kinetics.md` (F1–F6, F10, F11)
plus the cheap part of F7.  All 11 test files touched were run before and after
with `python -m pytest <files> -q -rsx -p no:randomly`; `ruff check` is clean on
every file listed below.

**Headline numbers (the 10 audit files touched, one new file added)**

| | before | after |
|---|---|---|
| collected | 142 | 140 + 15 (new `test_angular_harmonics.py`) |
| passed | 126 | 139 |
| skipped | **16** | **0** |
| strict xfail (carrying a measured failure) | 0 | 1 |

14 previously-skipped tests now execute (13 NLTE-9 cells + 1 W cooling
placeholder), and 10 angular physics gates were moved out from behind the
optional `plotly` import (verified below with `plotly` blocked at import).
Two permanently-dead skips were deleted.  No code under `src/atomx` was changed.

## 1. Finding-by-finding

| finding | file:line | change | measured before → after | new gate |
|---|---|---|---|---|
| **F1** PI⇄RR closed loop | `tests/test_plasma_physics.py:120` | deleted the tautology `alpha_rr * saha == pi_rate` (RR *is* `PI/Saha`, `crm.py:703`); added `test_hydrogenic_sigma0_matches_the_stobbe_threshold_closed_form` and the 7-point `test_radiative_recombination_matches_the_milne_planck_integral` | literal 6.3043e-18 vs closed form 6.304318e-18 = **2.87e-6** rel; Milne series vs module **3.47e-4 … 6.24e-3** | `(2^9π²/3)·α·a₀²·e^{−4}` at rel=2e-5 (`e` = **Euler's number**, stated in the test) and `[8πσ₀χ³/(h³c²)·Σ E₁(nχ/kT)]/Saha` at 1e-3…1e-2 |
| **F1** gates that could not fail | same file, `:177–227`, `:234–254`, `:279`, `:298`, `:442`, `:468` | added `abs=0.0` to 19 `pytest.approx` calls on sub-picoscale quantities | **all four DR-prefactor assertions were vacuous** (`abs` defaults to 1e-12, ~1e10× the 3.31e-22 prefactor); true deviations 2.8e-16, 1.4e-16, 1.2e-7, 2.9e-4 → all still pass at their stated `rel` | `abs=0` makes the rel tolerances real |
| **F2** NLTE-9 "published benchmark" | `tests/test_au_nlte9_workshop.py` (`:106`×12, `:122`) | published-benchmark claim removed from module docstring + `tests/data/nlte9_au/README.md`; the 13 skips replaced by 15 running rows (14 pass + 1 strict xfail); docstring's wrong "1.019" canonical ratio corrected; `source` column added to the CSV (uniform with the Pütterich CSV) | canonical model/stored ratios **1.0372 / 0.9660 / 0.9230 / 0.8507**, worst 1.2427 (0.5 keV/1e22); envelope is `mean ∓ 2.5` in **all 15 rows** (constructed ⇒ not a transcription) | `@pytest.mark.snapshot` per-row ratio at rel=1e-3 + the ±20 % envelope; the violating row is `xfail(strict=True, reason="panel-scope … 1.2427 …")` |
| **F2** Pütterich "published benchmark" | `tests/test_W_cooling_curve_vs_putterich2008.py:80` | skipped comparison deleted; renamed `test_stored_reference_…` (no "digitised" claim); new running placeholder-identity test | placeholder deviation **17.99 % at 5 eV** (2.62 % at 20 eV) against a 30 % gate; six candidate closed forms fitted to the 9 nodes, best 8.8e-2 ⇒ no committed generator | `tungsten_cooling_rate(T) == ref(T)·(1+0.18 sin ln T)` at rel=1e-12 + `max rel == 0.17987` |
| **F3** "approaches Boltzmann" asserted `pop[1] > 0` | `tests/test_plasma.py:242` | real high-density limit implemented (audit's specified ne set + analytic ratio) | `max|pop/boltz − 1|` = **1.540e-4 / 1.541e-7 / 1.5e-11** at 1e18/1e21/1e25 (falls as 1/ne) | monotone decrease + `< 1e-9` at 1e25 + `pop[1]/pop[0] == 3e^{−5/T}` rel=1e-9 |
| **F4** DR band rescaled by the bug factor | `tests/test_crm_satellites.py:18` | band `(0.025,0.40)·Ry^1.5` deleted; ratio gated on the Burgess closed form written out in the file | ratios 11.0477 / 5.3401 / 2.6253 reproduced to **1e-15**; prefactor headroom was ×0.478…×1.817 → **×1±1e-9** (×1.8 now fails 5/5 tests) | `pytest.approx(closed, rel=1e-9, abs=0) == ratio` (3 params + the wiring test) |
| **F5** UTA gate 3× the effect | `tests/test_crm_W_balance.py:142` | `rel <= 0.25` → `0.12` (measured 0.11617); added the sum rule + the analytic tail identity | explicit vs UTA **11.617 % / 11.166 % / 7.867 %** (1–200 / 1–2000 / 1–20000 nm); array power `ΣsᵢEᵢ` reproduced to 1e-12, `uta.integrate()/array_power` vs the erf tail to **1.6e-8** | `rel <= 0.12`, sum rule rel=1e-12, `captured == 0.88346` erf fraction rel=1e-6 |
| **F5** FAC copy presented as a comparison | `tests/test_crm_W_balance.py:69` | renamed `…_documents_the_fac_copy`; now asserts the documented copy explicitly (`atomx_fraction == fac_fraction`, `fraction_rel_error == 0`) and states there is no physics gate here | rms was identically 0 | placeholder-guard (a real balance trips it) |
| **F6** screening total shift ±9 % | `tests/test_atom_inputs_wired.py:100` | `≈5.5 ± 0.5 Ha` → `Σᵢ wᵢ⟨dV⟩ᵢ` over the unscreened orbitals (new `_first_order_total` helper, same construction as `:103`) | SCF **5.566119 Ha** vs first-order **5.566352 Ha** = **4.2e-5** rel | `rel=1e-3` (was ±9 %) |
| **F6** "merely different" models | `tests/test_atom_inputs_wired.py:154` | both SCF shifts checked against first-order theory and against each other, plus the analytic length ratio | u = 0.48801, λ_sp 14.312 > λ_D 14.048 a₀; SP shift **4.9762** vs Debye **5.5664 Ha**, ratio 0.89400 vs first-order 0.89391 | two rel=1e-3 first-order gates + ratio gate |
| **F10** dead no-op | `tests/test_viz.py:207-213` | deleted (bogus `skipif` on a bound method, `pass` body) | 1 permanent skip → 0 | — |
| **F10** fabricated "FAC clean-room" | `tests/test_viz_scattering.py:71` | `rc.alpha_cm3_s * 1.1` replaced by the Burgess closed form re-derived in the test; the plotted line is asserted against it | overlay was data×1.1; now independent and agrees with the plotted atomx curve to 1e-15 | `array_equal(plotted_reference, closed)`, `allclose(closed, rc.alpha_cm3_s, rtol=1e-9)` |
| **F10/F11** 8+ angular gates behind `importorskip("plotly")` | `tests/test_viz_plotly3d.py:32,47` → new `tests/test_angular_harmonics.py` | moved out of the plotly module, extended to 7 (l,m) + 6 (κ,μ); quadrature upgraded to Gauss-Legendre×periodic trapezoid | old 200×200 trapezoid grid gave `∫|Y₀₀|² = 0.99498` (5.0e-3 error) under a 5e-2 gate; new quadrature **9e-16** | `abs=1e-12` (was 5e-2) |
| **F7** weak closed forms (part) | `tests/test_plasma.py:24,31,54` | `0 < λ < 1e10` → the a.u. and cgs closed forms; `0.4 < R₂/R₁ < 0.6` → exact `8^{-1/3}`; "attractive" → `V_IS ≤ V_SP ≤ V_Debye` at two u | λ_D = 140.481 a₀, R_IS = 25.2563 a₀, u = 0.548 / 1.732 brackets measured | rel=1e-12 / exact / bracketing |

### The `uta.py:27` question (no code change)

`src/atomx/spectrum/uta.py:27` is **`intensity = total_strength * centroid_eV`**, and that *is* the
sum rule: `centroid_eV` is the strength-weighted mean transition energy
(`src/atomx/uta/moments.py:101`), so the rendered intensity equals
`Σᵢ n_u A_ul E_ul` exactly — confirmed against an independent recomputation from the
fixture's populations to 1e-12.  The measured 11.617 % "unconserved power" is **not** a
power defect: 100 % of it is the single broad Gaussian's tails falling outside the
wavelength window.  The array has centroid 50.81 eV (24.40 nm) and σ_λ = 19.62 nm, so the
erf fraction inside [1, 200] nm is 0.88346 and `uta.integrate()/array_power` reproduces
that to 1.6e-8.  Corrected form: none needed — the fix belongs in the test's
interpretation, and the audit's F5 diagnosis of a power-conservation defect is withdrawn.
(Recorded as a genuine refutation, not a deferral.)

## 2. Genuine failures that tightening exposed

1. **Every `pytest.approx` on a sub-picoscale quantity in `test_plasma_physics.py` was
   vacuous.** `pytest.approx(x, rel=r)` uses `tolerance = max(r·|x|, abs)` with `abs`
   defaulting to **1e-12**, so the four DR-prefactor assertions (3.31e-22), the Saha
   consistency check, the DR rate check (1.9e-11), the σ₀ literal (6.3e-18) and the
   collision-rate checks (1e-8…1e-11) could not fail.  This is why the audit's mutation
   "DR prefactor ÷ 50.186 → 5 failures" never listed this file's own prefactor gate.  With
   `abs=0.0` all of them still pass at their stated `rel` (2.8e-16 … 2.9e-4), so the fix
   costs nothing and restores ~19 real gates.
2. **The new F1 gates bite where the suite was blind** (in-memory mutation, no file
   changed): `_planck_photoionization_rate_s × 2` → 7 of 8 new tests fail;
   `_hydrogenic_sigma0_cm2 × 2` → 7 fail; Kramers exponent `(χ/E)³ → (χ/E)²` → 7 fail.
   Previously all three were 0 failures.
3. **The DR prefactor can no longer move by ×0.478…×1.817** — the old satellite band's
   headroom.  Patching the module prefactor by ×1.8, ×2 or ×0.5 now fails 5/5 satellite
   tests.
4. **NLTE-9's 0.5 keV / 1e22 cm⁻³ row genuinely violates the ±20 % band** (measured ratio
   1.2427 because the fixture is an Au50+…Au57+ panel while that cell needs
   Au35+…Au49+).  It is kept as a `strict=True` xfail carrying the measured number and the
   named cause rather than being dropped from the parametrisation.
5. **The audit's F6 expectation of the sign of the Stewart-Pyatt correction is wrong.**
   At the wired point (n_e = 1e22, T_e = 100 eV, u = 0.488) `stewart_pyatt_screening_length`
   is *longer* than λ_D (14.312 vs 14.048 a₀), so SP is the **less** attractive model
   (first-order 4.9762 vs 5.5664 Ha) — the opposite of "SP must exceed Debye at fixed
   (n_e, T_e)".  The test now asserts the measured sign and the first-order ratio.
6. **The old coordinate quadrature could not support a 1e-3 normalisation gate** (it gave
   0.99498).  Rather than widen the tolerance back to the old 5e-2, the quadrature was
   replaced (Gauss-Legendre in cos θ × periodic trapezoid in φ, exact for these
   integrands), which is what allows the 1e-12 gate.

## 3. Not fixed (and why)

| finding | status | reason |
|---|---|---|
| F7 (rest) | deferred | the six duplicated Stewart-Pyatt continuity tests (`test_plasma.py:94,109,126,138`, `test_plasma_physics.py:304,334`) are D-class coverage of a fixed bug; consolidating them changes no gate, and three of the six were already tightened here. |
| F8 | deferred | the six `C-degenerate` pairs need per-test annotation (`# C-degenerate: shares <kernel>`); no gate changes, and `test_plasma_physics.py:262` needs the de-excitation re-derived from the test's own collision strength (a real rewrite). |
| F9 / F11 (rest) | deferred | `test_crm_Au_band.py:39`, `test_crm_Au_cooling.py:20`, `test_crm_Fe_balance.py:52,58`, `test_crm_Fe_3C3D.py:29` are **5 remaining skips** in the slice. Unlike the two files fixed here, their skips are already loud, accurate and named, and their module docstrings already disclaim validation — the honest fix is a real atomx Au/Fe CRM balance plus `@pytest.mark.snapshot` on the live structural rows, which is out of this slice's budget. |
| F12 | deferred | `test_lineshapes.py:131` (Griem literal duplicated in `src/…/stark.py:43`, rel=1e-6 → 1e-3 + keep the n_e^{2/3} scaling), `test_average_atom.py` duplication, `test_cascade.py:75`, `test_rcfp.py` derivations, `test_convol.py:149` claim. All S2/S3, no false S1 claim. |
| F2 (provenance uniformity, reverse direction) | partially done | the NLTE-9 CSV gained a per-row `source` column (uniform with the Pütterich CSV) and its README now states the verification status; the Pütterich CSV's rows were left untouched because changing a stored external reference's metadata is a data edit the audit did not ask for. |

## 4. Skip accounting

* touched files, before: **16 skipped** (`test_au_nlte9_workshop` 13,
  `test_crm_W_balance` 1, `test_W_cooling_curve_vs_putterich2008` 1, `test_viz` 1).
* touched files, after: **0 skipped**, 1 strict xfail (the measured NLTE-9 panel-scope row).
* 14 previously-skipped tests now execute; 1 dead no-op and 1 dead duplicate skip deleted;
  10 angular physics gates moved out of the plotly-gated module.
* Plotly-absent evidence: with `plotly` blocked at import,
  `tests/test_angular_harmonics.py` still runs **15 passed**, while
  `tests/test_viz_plotly3d.py` collapses to its module-level skip.  Before the move those
  10 gates disappeared with it.
* Remaining in the slice (untouched files): 5 skips, see F9 above.
