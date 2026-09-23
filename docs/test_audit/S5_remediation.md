# S5 remediation — collision and continuum physics

Slice: the findings of `docs/test_audit/S5_scattering_continuum.md` (1–24) and
its annex `_S5_molecular_subreport.md` (F1–F14).  Measured 2026-09-22 with
`.venv/bin/python`, `MPLCONFIGDIR=/tmp/atomx_n1/mpl`, `-p no:randomly`.

Verdict: every S1 finding named in the remediation brief (a–k) is either fixed
with a tighter gate or converted into a loud `xfail(strict=True)` carrying the
measured number and the named cause.  **No tolerance was widened**: every band
that changed is a strict subset of the band it replaced.  Three genuine
physics failures were exposed by tightening and are now recorded as strict
xfails (DW partial-wave sum vs Born; the live PI/RR panel vs FAC; the EII
panel at E/I = 20 vs FAC) — none of them was hidden by a wider band.

## 1. Findings fixed

| # | finding | file:line | change | measured before → after | new gate |
|---|---|---|---|---|---|
| a/1 | Born band wider than the blend floor | `tests/test_dw_born_completion.py:66,170` | `BORN_TOL` 0.30 → **0.25** (floor 1−1/√2 = 0.2929 documented); Ω_DW now pinned explicitly; new strict xfail for the physics | \|rel\| 0.2284/0.2340/0.2322 (50/100/200×) unchanged; implied Ω_DW/Ω_Born **0.1094/0.0337/0.0091** (9×/30×/110× low) | `test_dw_partial_wave_sum_is_pinned` (`@pytest.mark.snapshot`, rtol 1e-3, 0.02321127/0.00958137/0.00381388); `test_dw_partial_wave_sum_reaches_the_born_limit` (`xfail(strict=True)`, requires Ω_DW ≥ 0.7 Ω_Born) |
| a/1b | Born reference never checked against the Bethe limit | `tests/test_dw_born_completion.py` (new) | new first-principles test: with `born_mod.ALPHA2 = 0` (the c→∞ limit) Ω_Born/ln[(k_i+k_f)/(k_i−k_f)] must be positive, increasing and bounded above by the exact closed form `4 g_i f/ΔE`, `f = 2⁸n⁵(n−1)²ⁿ⁻⁴/(3(n+1)²ⁿ⁺⁴) = 8192/19683` (Bethe & Salpeter 1957 eq. 63.5), `ΔE = 33.5897 Ha` → limit 0.049562 | ratio 0.03294 → 0.03855 at 20×→1000× (77.8 % of the limit at the top energy), i.e. the 1/ln E approach is intact | `test_born_reference_is_the_bethe_logarithm_at_alpha_zero` |
| b/2 | EII rate coefficient had no magnitude gate | `tests/test_eii.py:143-160` | added the closed-form + quadrature gates the brief asked for | constant σ=1e-16 cm²: code vs `σ√(8kT/πmₑ)` = 6.6924e-9 cm³/s at 1 eV (**rel ≤ 1e-6**); Ar BEB 7.660e-8 (20 eV) / 2.142e-7 (100 eV) cm³/s; code vs independent `scipy.quad` of the same cross section ≤ 3e-9, inside the rigorous bracket [σ_min, σ_max]·√(8kT/πmₑ) | `test_eii_rate_coefficient_matches_closed_form[3]`, `test_eii_rate_coefficient_of_argon_beb_is_bracketed_and_quadrature_exact` |
| c/3 | "`test_rr_vs_fac`" compared against no FAC data; the real panel was read by zero tests | `tests/test_rr_vs_fac.py:1-41`, new `tests/test_pi_rr_vs_fac.py`, `benchmarks/bench_fac_pi_rr.py`, `benchmarks/results_fac_pi_rr.csv` | docstring now states it is a hydrogenic class-C cross-check; the FAC panel got its own live test; the driver was made runnable and can no longer truncate the committed CSV; the atomx column was re-recorded with the **FAC column bit-identical** (verified: max change 0.0) | live atomx/FAC − 1 = **+0.587 / +1.137 / +1.301 / +2.022 / +2.628 / +3.521** (K-shell, 1→200 eV) and +0.577 / +0.233 / +0.259 (Li-like 2s / 2p₁/₂ / 2p₃/₂ maxima) | `test_panel_records_the_external_fac_column` (`external_fac`), `test_live_rr_reproduces_the_recorded_atomx_column` (`snapshot`, 1e-3), `test_live_li_like_rr_against_fac_is_inside_the_measured_envelope`, `test_live_k_shell_rr_agrees_with_fac_to_15_percent` (`xfail(strict=True)`) |
| d/4 | DR "kernel parity with FAC" is a same-kernel round-trip | `tests/test_dr_vs_fac.py:46,88-106` | renamed `test_kernel_reproduces_recorded_alpha`, marker `external_fac` → **`snapshot`**, `KERNEL_TOLERANCE` 0.01 → **1e-3** | ratio 0.99970901 at all four T (residual 2.90990e-4 = the `RYDBERG_EV**1.5` conversion), i.e. the old 1 % gate was 34× the residual | same test + the absolute scale now gated by the Saha test in `tests/test_eii_ea.py` |
| d/5 | DR CSV never recomputed atomx, frozen since 2026-07-16 | `tests/test_dr_vs_fac.py:129-163`, `benchmarks/results_fac_dr.csv` | CSV re-recorded (both columns) with the current kernel; new live-recompute gate calling `bench_fac_dr.run_atomx()` | ratios 0.486 / 0.978 / 1.074 / 1.158 at 1/5/10/50 MK, rms 0.2718; live-vs-recorded atomx ≤ 1e-4; the 2026-07-16 columns were 50.186× smaller because *both* go through the kernel and the DR prefactor fix scales both (the FAC **fixture** was not touched) | `test_live_atomx_dr_matches_the_recorded_panel` (`external_fac`) |
| e/6 | near-threshold EII gate permanently skipped; panel stopped at 1.5I | `tests/test_eii.py:388-424`, `benchmarks/bench_fac_dw_eii.py`, new `benchmarks/results_fac_dw_eii_extended.csv` | gate now runs (missing data ⇒ FAIL, not skip); extended panel (1.05I, 1.5I, 20I) added as a **separate CSV** because FAC's own value at a fixed E/I moves when the deck's grid changes (measured: Ne 1.5I = 1.6114e-20 in the 3-point deck, 1.5896e-20 in the extended one) — folding the rows in would have silently rewritten the FAC column gated at 3.5 % rms; the 3-point deck reproduces the committed CSV to 1e-11 (verified) | 1 skipped → 0 skipped; threshold vs FAC 2.5e-3 (gate 5e-3); σ(1.05I)/σ(1.5I) = 0.172; **at E/I = 20 atomx/FAC − 1 = −0.176 (Ne) / −0.165 (Fe) / −0.223 (Xe)** | `test_fac_near_threshold_shape_gate`, `test_fac_high_energy_eii_gate_is_inside_the_measured_band` (`external_fac`, band [−0.25, −0.10]), `test_fac_high_energy_eii_gate_within_10_percent` (`xfail(strict=True)`); `test_fresh_three_z_fac_k_shell_gate` now requests 1.5I too |
| f/8 | "detailed balance" composer is an algebraic identity | `tests/test_eii_ea.py:68-93` | replaced by the Saha statement between **distinct** processes: the capture coefficient `α_cap = (h²/2πmₑkT)^{3/2}(g_d/2g_i)Γ_a e^{−E_d/kT}` (CODATA by hand) is shared by DR and RE via their branching ratios | old: `a_RE == a_DR` to 1e-12 **by construction**; new: DR → α_cap as Γ_r→∞ (rel 1e-3), RE → α_cap as Γ_a^f→∞, each branch = α_cap·Γ_x/Γ_tot (rel 1e-6), and the two branches differ 3.1× for unequal widths | `test_detailed_balance_capture_flux_is_shared_by_dr_and_re` |
| g/7 | only working per-level Auger gates were ratio bands; both 10 % gates xfail | `tests/test_auger_fac_panel.py:68,205`, `tests/test_auger_kll_vs_fac.py:153`, `tests/test_jj_level_auger.py:157` | panel band [0.1, 30] → **[0.40, 6.5]**; jj_level band [0.1, 10] → **[0.1, 1.1]** (ceiling only — the floor is unchanged); both 10 % gates → `strict=True` with the measured deviation; new differential spread gate | **re-measured post-R7**: config-average panel 0.50 / 1.20 / 5.13 / 1.38 / 0.81 (the 1.13–14.82 in the audit is pre-R7); jj_level 0.097 / 1.007 / 0.236 / 0.361 / 0.795 (spread 10.4×) | `test_kll_auger_within_order_of_magnitude_of_fac`, both `..._10pct...` (`xfail(strict=True)`, worst \|ratio−1\| = 4.13), `test_panel_ratio_spread_is_small` (`xfail(strict=True)`, target < 2×) |
| h/10 | the two FAC-derived Fe XVII DR fixtures disagree 3.9× | `tests/test_dr_vs_fac.py` (new test) | adjudicated: `fe17_dr33_fac_cleanroom.json` (236 resonances, dr33, AICut 0) is the reference; the older `fe17_dr_fac.json` (194, dr33+dr34+KLM, AICut 1e-4) is not interchangeable | kernel(old)/kernel(cleanroom) = 0.2569 / 0.4499 / 0.5099 / 0.5693; median Γ_auger 1.826e13 vs 7.717e11 s⁻¹ (ratio 23.66) | `test_dr_fixture_pair_discrepancy_is_documented` (`snapshot`, rtol 2 %) |
| i/11 | "FAC" R-matrix thresholds are NIST under a 150× looser gate | `tests/test_r_matrix.py:23-29,152-169` | attribution corrected to NIST ASD (H I 13.598434599702 eV, He II 54.417760 eV); `FAC_THRESHOLD_RELTOL` 5e-4 → **1e-5**; new test that the tightened gate needs the reduced-mass factor | atomx 13.59846830 eV (rel 2.5e-6, was gated at 5e-4) and 54.41821051 eV (rel 8.3e-6); infinite-mass He II = 54.42567089 eV (rel 1.45e-4) now **fails** the gate | `test_hydrogen_target_threshold_matches_nist`, `test_heplus_target_threshold_matches_nist`, `test_heplus_threshold_gate_needs_the_reduced_mass` |
| j/19,21,22 | dead tests | `tests/test_dw_angular_coupling.py:166-183,303-320`; `tests/test_dr_density_suppression.py:30-35`; `tests/test_photoionization.py:38-43` | deleted `TestParityConservation` (2 tests) and `TestExchangePhase` (2 tests, all four did the arithmetic in the test body) and replaced them with one test on the kernel's own coefficients; the `None`-vs-`None` no-op became the Burgess closed form; `symmetric_limit` now asserts the symmetry it is named after | kernel: `_cc_direct(-2,-2,-1,-1,2,0) == 0.0` (parity-forbidden) vs `≠ 0` for k=1; zero-density rate equals the hand-written Burgess value to 1e-12; Fano F(+ε)+F(−ε) = 2 to 1e-12 | 37 → 34 cases in `test_dw_angular_coupling.py`; 5 → 5 (1 replaced) in `test_dr_density_suppression.py`; 14 → 14 in `test_photoionization.py` |
| k/16 | `external_fac` on tests asserting atomx's own recorded output | `tests/test_fac_dw_benchmarks.py:197,257`; `tests/test_dr_vs_fac.py:88` | those three retagged **`snapshot`** (the live-vs-FAC columns keep `external_fac`) | marker count: `-m external_fac` no longer selects any live-vs-recorded-atomx test | `-m external_fac` now selects only genuine external panels |
| k | `test_continuum_vs_fac.py` missing `external_fac` | `tests/test_continuum_vs_fac.py:73` | added `@pytest.mark.external_fac` to the four stored-dump cases | 4 stored-FAC cases now selected by `-m external_fac` | — |
| 14 | `facwave` dumps carried no in-file provenance; docstring claimed 100 eV–200 keV | `tests/data/facwave/*.txt`, `tests/test_continuum_vs_fac.py:10-15` | `# code = FAC 1.1.5 (sfac) WaveFuncTable dump`, `# deck = unrecorded …`, `# source = …` stamped into all four dumps (the parser ignores them: verified); range corrected to the four pinned energies (1/10/40/200 keV) | — | — |
| 12 | recorded-atomx drift band 2× its own measurement | `tests/test_fac_dw_benchmarks.py:83` | `EXC_ATOMX_REPRO` 0.05 → **0.03** | measured ≤ 2.6 % | still passes (7/7) |
| 17 | duplicated Auger coverage | `tests/test_auger_kll_vs_fac.py:127-150` | deleted the duplicated magnitude test (identical band, same solve, same five widths as `test_auger_fac_panel.py`, which additionally checks the fixture against the AITable record); the 10 % gate stays, now loud | 2 → 1 cases | — |
| annex F2,F12,F13 | fixture symmetry / assertions on deliberately-wrong code / constructor round-trip | `test_molecular_auger_hyp1066_regression.py:223-236`, `test_molecular_auger_spectrum.py:302,319` | F2 deleted (gates nothing: `mo_metadata` is read by no production path); F12's `rate_sum_intensity == 10.0` assertion replaced by a comment naming it as the ADR-0001 I3 counter-example (the property is documented as "incorrect, exposed only for tests"); F13 deleted | annex 148 → 146 collected cases, still 144 passed / 2 skipped | — |

### Code changes (all in `benchmarks/`, none in `src/`)

No `src/atomx/scattering/` change was needed: every S1 item in this slice was a
*gate* defect, and the three physics failures it exposed (Ω_DW 9×/30×/110× low,
K-shell RR 1.6–4.5× high vs FAC, EII −16…−22 % at 20I) need kernel work that
cannot be validated from the references available here, so they are recorded as
loud strict xfails instead of being papered over.

`benchmarks/bench_fac_pi_rr.py` (three real defects found while wiring the
panel, each independently verified):

1. `Atom(..., engine=None)` resolves to the **shooting** engine, which leaves
   `SCFResult._bsp_data = None`, so `rr_cross_section` raised
   `ValueError: SCFResult is missing _bsp_data` — the atomx column could not be
   recomputed at all.  Fixed by pinning `engine="bspline"` (verified: the
   recorded column now reproduces to 1.2e-9).
2. With the FAC residual-core continuum (a *shooting* model) and the default
   E1+M1+E2 multipoles the kernel raises `NotImplementedError`; the driver now
   pins `multipoles=("E1",)` and says so.  Side effect: the panel is E1-only,
   which is why the measured agreement moved from −0.63…−0.02 (stale column) to
   the current +0.52…+0.59.
3. On any case failure `main()` still wrote the CSV — measured: a failed run
   truncated the committed 24-row panel to a bare header (recovered from a
   backup).  It now refuses to write an empty panel and exits non-zero.

`benchmarks/bench_fac_dw_eii.py`: added `EXTENDED_ENERGY_FACTORS` and the
`--extended` flag writing `results_fac_dw_eii_extended.csv`; `ENERGY_FACTORS`
(the committed panel) is unchanged and re-verified to reproduce
`results_fac_dw_eii.csv` to 1e-11.

## 2. Genuine failures that tightening exposed (not fixed, recorded loudly)

| statement | measured | test |
|---|---|---|
| the DW partial-wave sum must reach the Born limit at 50–200× threshold | Ω_DW/Ω_Born = **0.1094 / 0.0337 / 0.0091** (9×/30×/110× low). `kappa_max=16/24` raises the sum (0.29/0.10/0.03 and 0.60/0.18/0.07) but makes the *shipped* value worse (−0.21/−0.26/−0.28 and −0.19/−0.39/−0.41), i.e. the completion is anti-correlated with the kernel. Cause: the unresolved high-l tail is replaced by `tail = max(geometric floor, Ω_partial)`, so the blend caps at w = 1/√2 | `test_dw_partial_wave_sum_reaches_the_born_limit` (`xfail(strict=True)`) |
| planewave RR (E1, B-spline) must be within 15 % of the FAC RRTable | K-shell **+0.587 / +1.137 / +1.301 / +2.022 / +2.628 / +3.521** at 1/15/37/70/122/200 eV (growing with energy ⇒ a matrix-element/continuum-normalisation problem, not a prefactor) | `test_live_k_shell_rr_agrees_with_fac_to_15_percent` (`xfail(strict=True)`) |
| the EII FAC panel must hold above the Born-completion point | at E/I = 20: **−0.176 (Ne) / −0.165 (Fe) / −0.223 (Xe)** vs FAC's pure-DW CI (max_kl = 36) while the 1.05–1.5I points agree to 0.5–2.2 % | `test_fac_high_energy_eii_gate_within_10_percent` (`xfail(strict=True)`) + the active band [−0.25, −0.10] |

Two more measured facts recorded in docstrings rather than tests: (i) the two
per-level Auger 10 % gates now fail by 4.13× (post-R7) instead of 13.8×
(pre-R7) — the panel improved, the claim is still not met; (ii) `dr_rate_coefficient`
and `resonant_excitation_rate` use different total widths for the same
resonance (`dr_rate_coefficient` ignores a `partial_auger_to_final` channel
that the RE function includes in Γ_tot), so the two are only mutually
consistent when the other channel is closed — documented in
`test_eii_ea.py`, not "fixed", because changing it is an API decision.

## 3. Findings not fixed (and why)

| finding | why deferred |
|---|---|
| 9 + annex F1, F4–F11, F14 | the molecular Auger slice needs a physics decision plus a real external reference (Koerber & Mehlhorn 1966 line energies, `build_core_hole_delta_scf_payload`) and `pyscf` is not installed, so F7's only real-physics path skips here.  F2/F12/F13 (the free E-class deletions) were done. |
| 13 | the remaining wide bands in `test_eii.py` (`1/3 < DW/BEB < 3`, `0.05 < DW/BEB < 15`, the 1e-19/1e-18 σ windows) need BEB's own peak/limit as the anchor, which the new high-energy EII gate shows is itself 16–22 % off FAC; anchoring a band to a quantity with a known external bias would enshrine the bias. |
| 15 | `REPORT_FAC_PARITY.md` vs the CSV: `docs/` — out of scope for this slice's remediation (only `benchmarks/results_fac_pi_rr.csv` was re-recorded). |
| 18 | the un-sourced He⁺ σ_th = 1.575e-18 cm² literal is superseded by the neighbouring σ ∝ Z⁻² ratio test; deleting it is a judgement call for the file's owner. |
| 20, 23, 24 | `_cc_direct` re-implementation, the `at_T` grid-point interpolation tautology and the `test_continuum.py` σ>0 / `factor < 10` bands were not in the a–k brief and are lower severity (S4). |
| 12 (high-energy excitation rows) | the excitation panel still stops at E/I = 5; extending it needs a live `sfac` run whose deck change moves the recorded FAC column (measured for the EII deck), so it needs its own separate artifact like `results_fac_dw_eii_extended.csv`. |

## 4. Verification

Run 2026-09-22 (`-p no:randomly`), touched files only:

| file | before | after |
|---|---|---|
| `test_dw_born_completion.py` | 18 pass | **20 pass, 1 xfail** (21) |
| `test_eii.py` | 22 pass, 1 skip (+1 slow) | **27 pass, 1 xfail** + 1 slow pass (29) |
| `test_eii_ea.py` | 4 pass | 4 pass |
| `test_pi_rr_vs_fac.py` | — | **3 pass, 1 xfail** (4) |
| `test_rr_vs_fac.py` | 1 pass (slow) | 1 pass (slow) |
| `test_dr_vs_fac.py` | 4 pass | 6 pass |
| `test_dr_density_suppression.py` | 5 pass | 5 pass |
| `test_photoionization.py` | 14 pass | 14 pass |
| `test_dw_angular_coupling.py` | 37 pass (+2 slow) | 34 pass + 2 slow pass |
| `test_r_matrix.py` | 16 pass | 17 pass |
| `test_continuum_vs_fac.py` | 4 pass | 4 pass |
| `test_fac_dw_benchmarks.py` | 7 pass | 7 pass |
| `test_auger_fac_panel.py` | 2 pass, 2 xfail(soft) | 2 pass, 2 **strict** xfail |
| `test_auger_kll_vs_fac.py` | 1 pass, 1 xfail(soft) | 1 **strict** xfail |
| `test_jj_level_auger.py` | 1 pass, 1 strict xfail, 1 KAT pass | 2 pass, 2 strict xfail |
| molecular trio | 146 pass, 2 skip | 144 pass, 2 skip |

Combined (`-m "not slow"`): **111 + 34 = 145 passed, 7 xfailed, 0 failed,
0 skipped** in the 15 non-molecular files; the 3 deselected `slow` cases (the
EII shooting-mesh convergence scan and the two DW integration tests) were run
separately and all pass, for **148 passed / 7 xfailed** in total.
`ruff check` (E4,E7,E9,F,I,B,SIM) is clean on all 17 touched `.py` files.

Artifacts touched: `benchmarks/results_fac_dr.csv` (both columns re-recorded,
same FAC fixture), `benchmarks/results_fac_pi_rr.csv` (atomx column only; FAC
column verified bit-identical), new
`benchmarks/results_fac_dw_eii_extended.csv`, and the four
`tests/data/facwave/*.txt` provenance headers.
