# S6 — plasma, collisional-radiative kinetics and spectral synthesis

Slice: `tests/test_plasma.py`, `test_plasma_physics.py`, `test_crm_satellites.py`,
`test_crm_W_balance.py`, `test_crm_Au_band.py`, `test_crm_Au_cooling.py`,
`test_crm_Fe_balance.py`, `test_crm_Fe_3C3D.py`, `test_crm_dw_rates.py`,
`test_au_nlte9_workshop.py`, `test_W_cooling_curve_vs_putterich2008.py`,
`test_cascade.py`, `test_lineshapes.py`, `test_convol.py`, `test_fitting.py`,
`test_average_atom.py`, `test_rcfp.py`, `test_viz.py`, `test_viz_plotly3d.py`,
`test_viz_scattering.py`. All 20 listed files exist.
**Added:** `tests/test_atom_inputs_wired.py` (the only file that gates the
`Environment(plasma_ne, plasma_Te, screening=...)` wiring, i.e. the plasma
question the brief asks about; `test_atom_input_surface.py` was *not* added — it
gates the input schema, not the physics).
222 tests collected; measured run: **201 passed, 21 skipped, 49.6 s**.

## 1. Verdict

The high-level physics in this slice is carried by **one** file.
`tests/test_plasma_physics.py` (22 tests) is genuinely first-principles and is
the only thing in the suite that would have caught the missing
radiative-recombination channel — mutation of the rate matrix to drop RR fails
exactly 4 tests there and in `test_plasma.py`, and nothing else. Everything
above it is hollow in a specific, measurable way: the two files that claim
published-benchmark validation (`test_au_nlte9_workshop.py`,
`test_W_cooling_curve_vs_putterich2008.py`) have **13 and 1 skipped tests
respectively** and no live comparison, because the "atomx" curves they compare
against are calibrated functions of the published numbers themselves (measured
"agreement" 3.7–14.9 % for Au ⟨Z⟩, 18.0 % max for W L_w — both numbers are
properties of the calibration, not of atomx); the remaining CRM files assert
fixture structure, normalisation-to-1, or positivity. The radiative sector
(PI ⇄ RR) is a closed loop: `radiative_recombination_rate_cm3_s` is *defined* as
`_planck_photoionization_rate_s / saha_ionization_density_cm3`
(`src/atomx/plasma/crm.py:703`), so doubling the photoionisation rate, doubling
the Kramers σ₀, or using the wrong Kramers exponent (χ/E)² leaves **63/63**
tests in this slice green (measured, three separate mutations) — the LTE ⇒ Saha
gate is exact *and* blind to any error shared by the two directions. Screening
is the one bright spot: the shift is gated against first-order perturbation
theory on unscreened orbitals to 2.4 % (measured 0.7079 → 0.6910 Ha for Ne²⁺
1s at 1e22 cm⁻³ / 100 eV), though the corresponding total-energy statement is
written as `5.5 ± 0.5 Ha` (±9 % on the best-measured quantity in the slice).
Class mix over 222 tests: **A 69, B 3, C 16 (7 non-degenerate), D 36, E 70,
F 28** — i.e. 32 % A/B, 47 % E/F, and 9.5 % of tests (21) never execute.

## 2. Files

| file | tests | class mix | worst | note |
|---|---|---|---|---|
| `test_plasma.py` | 24 | A7 C1 D5 E7 F4 | **S1** | best screening gates in the slice (`:68` analytic u→0/u→∞ limits); `:242` "approaches Boltzmann" asserts only `pop[1] > 0`; 5 continuity tests are 3× duplicated coverage |
| `test_plasma_physics.py` | 22 | A17 C3 D1 E1 | — | the slice's only real physics gate; `:97` LTE⇒Saha exact to 1e-9 and catches the RR bug; `:120` is a tautology (RR *is* PI/Saha) |
| `test_crm_satellites.py` | 4 | D4 | **S1** | ratio band = old band × RYDBERG_EV^1.5 (16× wide); measured ratios 11.0477 / 5.3401 / 2.6253; accepts DR prefactor ×0.48…×1.82 |
| `test_crm_W_balance.py` | 6 | C1 D1 E4 | **S2** | `:69` documents that the atomx abundance column is a verbatim copy of FAC (rms ≡ 0); `:142` UTA gate 25 % vs measured 11.6 % discrepancy |
| `test_crm_Au_band.py` | 3 | D1 E2 | **S1** | Bauche-Arnoult moment gate skipped (fixture reproduces the reference by construction); remaining 2 tests assert fixture structure |
| `test_crm_Au_cooling.py` | 3 | D1 E2 | **S1** | cooling gate skipped; the 2 live tests assert Σfraction = 1 (true by construction) |
| `test_crm_Fe_balance.py` | 4 | D2 E2 | **S1** | FAC/CHIANTI gates skipped as fabricated; live tests assert 3 fabricated columns each sum to 1 |
| `test_crm_Fe_3C3D.py` | 2 | D1 E1 | **S1** | Bernitt gate skipped (atomx column = reference + tiny log terms); live test is a grid-shape check |
| `test_crm_dw_rates.py` | 4 | A3 C1 | — | `:28` constant-Ω kernel exact to 1e-12; `:54` analytic 8.629e-6 rate and asserts the van Regemorter fallback differs by >5 % |
| `test_au_nlte9_workshop.py` | 14 | D13 E1 | **S1** | 13/14 skipped; live test asserts workshop_min < ⟨Z⟩ < workshop_max and 0 < ⟨Z⟩ < 79 |
| `test_W_cooling_curve_vs_putterich2008.py` | 2 | F1 D1 | **S1** | live test asserts the committed CSV is monotone and the interpolator reproduces its own nodes to 1e-12; the atomx comparison is skipped |
| `test_cascade.py` | 7 | A4 E1 F2 | — | exact branching ratios (1/6, 5/6) and probability conservation; `:75` duplicates `:52` |
| `test_lineshapes.py` | 18 | A14 B1 F3 | S2 | Voigt/FWHM/Doppler closed forms, all named; `:131` "Griem table 0.38 Å" is checked at rel=1e-6 against the same literal in `src/atomx/lineshapes/stark.py:43` |
| `test_convol.py` | 15 | A1 B2 C5 E7 | S2 | two real B oracles (He/Be vs DBSR_HF); Slater cell-vs-Yᵏ cross-checks at 1–5 %; 7 shape/definition no-ops |
| `test_fitting.py` | 7 | A3 E1 F3 | — | exact inverse-problem recovery + χ²/dof ∈ [0.5, 2.5] on seeded noise; well formed |
| `test_average_atom.py` | 9 | A7 E1 F1 | S3 | limits (cold-neutral, full-strip, H@50 eV) are correct but subsumed by `test_plasma_physics.py:368/395` at tighter tolerance |
| `test_rcfp.py` | 8 | C2 D5 E1 | S3 | literals ±√2, −1.0, 0.5 are exact RCFP algebra but no derivation/citation is named; wrappers over `mscheme` |
| `test_viz.py` | 27 | E18 F8 D1 | S4 | smoke tests only (returns a `Figure`); `:207` is a permanent skip with a bogus `skipif` and a `pass` body |
| `test_viz_plotly3d.py` | 21 | A8 E11 F2 | S4 | 8 real gates (∫|Y_lm|²dΩ = 1, spinor density normalised/positive) hidden behind a module-level `importorskip("plotly")`; the other 11 assert dropdown/trace counts |
| `test_viz_scattering.py` | 5 | E4 F1 | S4 | plot smoke; `:71` fabricates its "FAC clean-room" reference as `rc.alpha_cm3_s * 1.1` |
| `test_atom_inputs_wired.py` *(added)* | 17 | A5 C3 E6 F3 | S2 | `:63/:84/:103` gate the Debye/Stewart-Pyatt **shift** (not just the parameter) against first-order theory; `:100` pins the total shift at ±9 % |

## 3. Findings

### F1 — S1 — the radiative sector is a closed loop: every error shared by PI and RR is invisible
`src/atomx/plasma/crm.py:703` defines `alpha_RR := _planck_photoionization_rate_s(...) / saha(...)`.
Consequently `tests/test_plasma_physics.py:120` (`alpha_rr * saha == pi_rate`, rel=1e-12) is a
tautology: it re-checks the one-line definition. Measured by mutation (in-memory
patches, no file changed), running `test_plasma_physics.py test_plasma.py
test_crm_satellites.py test_crm_dw_rates.py test_average_atom.py` (63 tests):

| mutation | failing tests |
|---|---|
| drop the RR channel (`include_radiative_recombination=False`) | **4** (`plasma_physics:97,136,154`; `plasma:275`) |
| drop photoionisation | 2 |
| drop three-body | 3 |
| drop EII | 3 |
| **`_planck_photoionization_rate_s` × 2** | **0** |
| **`_hydrogenic_sigma0_cm2` × 2** | **0** |
| **Kramers exponent (χ/E)³ → (χ/E)²** | **0** |
| DR prefactor × 2 | 2 (only the 200 eV satellite case) |
| DR prefactor ÷ RYDBERG_EV^1.5 (the historical 50.186× bug) | 5 |

So channel *presence* is now gated, the RR/PI *ratio* is gated (RR × 2 fails 2 tests —
`plasma:275` and the satellite case correctly do not), but the absolute scale of
the photoionisation cross-section, the Planck photon density and the sharp-threshold
Kramers shape are gated by nothing in the slice. The constant itself is fine and
derivable: `6.3043e-18` vs the closed form `(2⁹π²/3)·α·a₀²·e⁻⁴ = 6.304318e-18 cm²`
(agreement 2.9e-6), and the sibling module pins the same NIST number at 1 %
(`tests/test_hydrogenic_cross_sections.py:18`) — the CRM simply duplicates it
ungated (`src/atomx/plasma/crm.py:779`).
**Remediation.** In `test_plasma_physics.py` replace `:120` with:
(a) `assert _hydrogenic_sigma0_cm2(RYDBERG_EV) == approx((2**9*pi**2/3)*ALPHA*a0_cm**2*exp(-4), rel=1e-5)`
— an A-class closed form, and cross-check it against
`atomx.methods.hydrogenic_cross_sections._STOBBE_PREFACTOR_H_CM2*exp(-4)`;
(b) an independent evaluation of the same integral: the Milne/Planck integral
`α_RR = [8π σ₀ χ³/(h³c²) · Σ_{n≥1} E₁(nχ/kT)] / Saha` (substitute x = E/kT and expand
1/(eˣ−1) = Σe^{−nx}). Measured against the module: rel. 3.5e-4 at (100 eV, 100 eV),
3.9e-4 at (20, 10) and (1, 0.5), 4.6e-4 at (13.6, 5), 6.3e-3 at (13.6, 0.86) — i.e.
the log-grid trapezoid is accurate to ≤0.05 % except at the hardest point. This gate
fails for a wrong prefactor, a truncated integration range, or a missing
(1−e^{−x}) stimulated factor, all of which are invisible today.

### F2 — S1 — the two "published benchmark" files gate nothing and their quoted agreement is a property of the fixture
`tests/test_au_nlte9_workshop.py`: 14 collected, **13 skipped** (`:106` ×12 parametrised,
`:122`). The model compared is `benchmarks/bench_fac_crm_Au.py:233-241`
`center = 50 + 7·(Te−0.5)/4.5 − 0.18·log10(ne/1e20)` — a Gaussian in *charge* fitted to the
same drive band. Measured model/CSV ratios across all 15 committed rows:
canonical cells (1, 2, 5, 10 keV at 1e21 cm⁻³) = **1.0372 / 0.9660 / 0.9230 / 0.8507**
(3.7 / 3.4 / 7.7 / 14.9 %), worst row **1.2427** (24.3 %, at 0.5 keV / 1e22). The
docstring's quoted "1.019 / 0.966 / 0.923 / 0.851" does not match the canonical set:
1.019 is the (1.0 keV, 1e20) row, which the fixture marks `canonical=False`. The CSV
carries `workshop_min/max` but **no `source` column**; the citation lives only in
`tests/data/nlte9_au/README.md`.
`tests/test_W_cooling_curve_vs_putterich2008.py`: 2 collected, the only atomx comparison
skipped (`:80`); the live test (`:56`) asserts the CSV is monotone/in-envelope and that
`putterich_2008_cooling_reference` reproduces its own nodes to rel=1e-12. The compared
quantity is `putterich_ref·(1+0.18·sin)`, measured max deviation **17.99 %** at 5 eV
(2.6 % at 20 eV) against a **30 %** gate — i.e. the gate's only content is the engineered
offset. The CSV rows do all carry `Putterich et al. 2008 PPCF 50 085016`, but its values
have 9 significant figures (`1.85782573e-31`), which cannot come from digitising a figure,
and no generator/fit is committed, so the provenance of the numbers is unverifiable inside
the repo.
**Remediation.** Do not leave these files named as validations. Either (a) delete the
skipped bodies and rename to `*_fixture_provenance.py`, or (b) make them real: for NLTE-9,
transcribe ≥4 rows of the workshop median table with page/table numbers into both the
README and the test (`assert row.mean_charge == 49.0` etc.) so the fixture is checked
against the paper rather than against itself, and drop `mean_charge` from the comparison;
for Pütterich, commit the analytic fit that generated the CSV and assert the CSV against it
to 1e-6, then state in the docstring that no atomx cooling curve exists.

### F3 — S1 — `test_high_density_approaches_boltzmann` asserts nothing about Boltzmann
`tests/test_plasma.py:242-261`. The test name and comment promise the LTE limit; the
assertion is `assert pop_crm[1] > 0` (line 261), and line 257 says explicitly
`crm.boltzmann_populations(Te)  # must not raise; not compared below`. Any model that puts
any population in the upper level passes, including one with the collisional channel
deleted. (Control: the neighbouring `test_steady_state_normalized` at `:228` is
`sum(pop) == 1`, true by construction of the solver at `src/atomx/plasma/crm.py:464`.)
**Remediation.** Replace with a convergence statement in ne: for
`ne ∈ {1e18, 1e21, 1e25}` assert `abs(pop / crm.boltzmann_populations(Te) − 1)` decreases and
is `< 5e-2` at 1e25 (the collisional rates are van Regemorter/DW with correct detailed
balance, so the limit is real), plus the analytic Boltzmann ratio
`pop[1]/pop[0] == (g1/g0)·exp(−5/Te)` at the high-density point.

### F4 — S1 — the DR satellite gate tolerates a 3.8× prefactor error; its band was rescaled in the fix commit
`tests/test_crm_satellites.py:18`: `_RATIO_BAND = (0.025, 0.40) · RYDBERG_EV**1.5 = (1.2546, 20.074)`
— the old band multiplied by exactly the factor (50.186) by which the DR prefactor was
corrected. Measured ratios and headroom (reproduced by running the fixtures):

| T_e | ratio | accepted prefactor multiplier |
|---|---|---|
| 200 eV | 11.0477 | ×0.1136 … ×1.817 |
| 400 eV | 5.3401 | ×0.235 … ×3.759 |
| 800 eV | 2.6253 | ×0.478 … ×7.646 |
| **all three** | | **×0.478 … ×1.817** |

Scan confirms: ×1.8 passes 4/4, ×1.82 fails 1, ×0.48 passes 4/4, ×0.45 fails 1, ×0.2 fails 3,
÷50.186 fails 4. The file's own docstring calls this "textbook range" with no citation.
**Remediation.** The ratio has a defensible first-principles target: with one resonance the
satellite emissivity is `n_u A_r` and the w-line is `n_u(A_w + …)`, so assert the analytic
branching form of `_fe17_fixture` (α_DR·branch vs A_w·pop ratio) at a stated value, or at
minimum replace the band with `pytest.approx(<measured>, rel=0.25)` **and** add the
prefactor identity gate that already exists at `test_plasma_physics.py:177` to this file's
docstring reference. A 16× band on a quantity that was just wrong by 50× is not a gate.

### F5 — S2 — the UTA-vs-explicit power gate is 3× wider than the effect it measures
`tests/test_crm_W_balance.py:142-170`: `rel <= 0.25`, measured
**11.617 %** (λ 1–200 nm), 11.166 % (1–2000 nm), 7.867 % (1–20000 nm) — the residual
survives a 100× wider window, so ~8 % of the bundled power is genuinely not conserved by
the UTA path, not merely moved out of band. `src/atomx/spectrum/uta.py:27` builds the array
as `intensity = total_strength * centroid_eV`, so Σgf·hν is expected to be conserved.
**Remediation.** Replace with the sum rule (`Σ_bundle gf·hν_ul` vs the explicit sum over the
same transitions — an A-class identity), keep the 11.6 % as a *shape* acceptance on the
rebinned spectrum, and state the measured number in the docstring.

### F6 — S2 — the screening shift is gated, but the best-measured number is pinned at ±9 %
`tests/test_atom_inputs_wired.py` (added) is the answer to "is the environment input gated?".
Yes, and it is measured rather than round-tripped: `:63` checks
`screening_shift == V_model − (−Z/r)` against the module's own potential; `:84` requires every
Ne²⁺ orbital shift to lie in `[0.5, 1.05]·Z/λ_D` (with `λ_D = 140.481 a₀` at 1e22 cm⁻³,
100 eV; measured shifts 0.685–0.691 Ha vs `Z/λ_D = 0.712 Ha`); `:103` compares the SCF shift
with the first-order perturbation `⟨dV⟩` on the **unscreened** orbitals and names the
derivation — measured **0.7079 → 0.6910 Ha (1s, 2.4 % below first order)** and
0.7078 → 0.6858 (2p3/2), asserted at rel=0.10. The weak link is `:100`:
`dense.total_energy − bare.total_energy == approx(5.5, abs=0.5)` — ±9 % on a 5.566 Ha
(151 eV) ionisation-potential depression whose physics the same file measures to 2.4 %.
`:154` (`stewart != debye`) and `:174` (`_screening_shift() is None`) are the only other
screening checks and are inequalities.
**Remediation.** Pin the total shift at the first-order value summed over occupied orbitals
(`Σ_i ⟨dV⟩_i` = 5.566 Ha, `rel=0.03`) instead of `5.5 ± 0.5`; assert the Stewart-Pyatt
potential exceeds the Debye one at fixed (ne, Te) by the analytic ratio from
`stewart_pyatt_screening_length` rather than merely differing.

### F7 — S2 — Stewart-Pyatt continuity is asserted 5 times with 3 different thresholds, and the weak tests in `test_plasma.py` pin no value
`test_plasma.py:94,109,126,138` and `test_plasma_physics.py:304,334` all sweep u through the
legacy branch points (thresholds 1.003 / 1.03 / 1.002 / 1.03 / 1e-6 rel / 0.05 rel). The
duplication is real coverage of a fixed bug (D-class), but it is 6 tests' worth of budget for
one silent-failure mode. In the same file, three tests in the screening block assert almost
nothing:
`:24` `debye_length(1e20, 100) > 0 and < 1e10` (the true value is **140.481 a₀ = 743.4·√(T_eV/n_e) cm**
= 7.434e-7 cm; the `< 1e10` bound is 13 orders of magnitude loose);
`:31` `0.4 < R₂/R₁ < 0.6` for an 8× density increase (exact **0.5**;
`ion_sphere_radius(1, 1e20) = 25.2563 a₀ = (3/4πn_e)^{1/3} = 1.3365e-7 cm`);
`:54` `test_stewart_pyatt_interpolates` asserts only `V(r) < 0`, i.e. "attractive" — the name
promises an interpolation check and compares nothing to either limit.
**Remediation.** Collapse the six continuity tests into one parametrised test over both
branch points, and spend the freed budget on the closed forms:
`debye_length(1e20, 100.0) == approx(743.4*sqrt(100/1e20)/a0_cm, rel=1e-6)`,
`ion_sphere_radius(1, 8e20)/ion_sphere_radius(1, 1e20) == approx(0.5, rel=1e-12)`, and for
`test_stewart_pyatt_interpolates` assert `debye_potential ≤ V_SP ≤ ion_sphere_potential` at
two values of u (the interpolation is what the name claims).

### F8 — S3 — C-class tests that share a kernel with the code they check (`C-degenerate`)
The following are labelled C but cannot fail for a physics reason, because the "independent"
side is the same expression or the same helper: `test_plasma.py:275` (compares `R` with
`three_body_recombination_rate_cm6_s`/`radiative_recombination_rate_cm3_s`, the functions
`_build_rate_matrix` itself calls; and
`pop[1]/pop[0] == R[1,0]/R[0,1]` is the 2-level steady-state solution by construction);
`test_plasma_physics.py:136` (same helper on both sides — its only real content is the
`include_radiative_recombination=False` control); `test_plasma_physics.py:262`
(detailed balance is the one-line construction `C_deexc = C_exc_reduced·(g_l/g_u)` at
`src/atomx/plasma/crm.py:354`); `test_crm_dw_rates.py:72` (same identity through the Ω table);
`test_convol.py:228` (`case1`/`case2` are transposes for symmetric rkb — an identity of the
implementation); `test_rcfp.py:19,37` (wrappers over `mscheme` returning 0 or re-reading the
convention). The genuinely independent C-pairs in the slice are
`test_plasma_physics.py:442` (rounded 8.629e-6 closed form vs explicit cgs Maxwellian
integration, 1.53e-5 rel.), `test_convol.py:149,166,176,187` (cell quadrature vs Yᵏ cumulative
sum, 1–5 %) and `test_atom_inputs_wired.py:125,213` (shooting vs B-spline).
**Remediation.** Rename or annotate the degenerate six (`# C-degenerate: shares <kernel>`),
as the rubric requires; for `test_plasma_physics.py:262` derive the de-excitation from the
*test's own* collision strength (the file already does this correctly at `:234` via the
van Regemorter large-x asymptote).

### F9 — S1/S3 — the balance/normalisation family: what it cannot see
These are the tests the brief's failure modes (a) and (b) describe, and all of them are
blind to a missing channel: `test_plasma.py:228` (Σn = 1, true by construction),
`:322` (a toggle adds a positive term — no value), `:350` (metadata round-trip),
`:263` (Boltzmann ground-state dominance: `boltmann_populations` *is* the closed form being
asserted); `test_crm_Au_cooling.py:35` and `:44` (the Au charge fractions — a hand-tuned
Gaussian — sum to 1 and cover the grid); `test_crm_Fe_balance.py:40` (three *fabricated*
columns, each normalised to 1); `test_crm_W_balance.py:69` (the atomx column is
`abund = dict(fac_abund)` at `benchmarks/bench_fac_crm_W.py:371`, so `fraction_rel_error`
is identically 0 — the test itself documents this and deliberately does not assert the RMS),
`:93` (mock call count), `:123` (`spec.integrate() > 0`), `:59` (fixture structure —
note `item.atom is None`: **no atom is solved in this file**); `test_crm_Au_band.py:27,55` and
`test_crm_Fe_3C3D.py:21` (fixture shape/window), `test_crm_Fe_balance.py:30`,
`test_au_nlte9_workshop.py:88` (fixture well-formedness).
An ionisation balance normalised to 1, a cooling curve asserted positive and finite, or a
charge-fraction table asserted to sum to 1 cannot see a missing recombination channel; the
only tests in this slice that can are the four listed in F1's first row.
**Remediation.** Delete the normalisation/positivity rows (they are re-asserted by the
constructors), and replace each with the one external number the fixture already contains:
for the Au panel assert `mean_charge(10.0, 1e21)` against the NLTE-9 CSV *and* against the
same quantity computed from a real CRM solve once one exists; for W assert
`fac_ref.ionization_potentials_eV[q]` against the NIST/FAC values quoted in the fixture.
Until then these files should carry the `snapshot` marker so they are not counted as
physics coverage.

### F10 — S4 — viz files are smoke tests; one is a permanent no-op and one fabricates its reference
`test_viz.py` (27 tests): every matplotlib/plotly test asserts `isinstance(fig, Figure)` or
an axis/trace count; the manim block asserts `hasattr` on symbols. `:207-213` is
`@pytest.mark.skipif(pytest.importorskip("importlib").import_module, reason="Always tested")`
with a `pass` body — a permanently skipped dead test whose condition is a bound method
object (always truthy). `test_viz_scattering.py:71-72` passes
`references={"FAC clean-room": (rc.T_eV, rc.alpha_cm3_s * 1.1)}` — the plotted "reference"
is the data times 1.1; the test then asserts only `get_xscale() == "log"`.
`test_viz_plotly3d.py` mixes 8 genuine gates (`:32` ∫|Y_lm|²dΩ = 1 within 5 % for 4 (l,m);
`:47` spinor density positive and normalised for 4 (κ,μ)) with 11 layout assertions.
**Remediation.** Delete `test_viz.py:207-213`; move the 8 angular-normalisation tests from
`test_viz_plotly3d.py` into `test_angular*` so they are not behind the optional `plotly`
import; keep the rest as explicitly-marked smoke tests (`@pytest.mark.smoke`) so the
physics count is not diluted. In `test_viz_scattering.py` plot the *actual* FAC numbers
(the file `tests/data/facwave/*` is already committed) instead of a scaled copy of the data.

### F11 — S1 (infrastructure) — 21 run-time skips + 23 tests behind an optional import = up to 20 % of the slice can vanish
Skips measured with `-rs` (21/222 = 9.5 %): `test_au_nlte9_workshop.py:106` ×12 and `:122`,
`test_crm_W_balance.py:109`, `test_crm_Au_band.py:39`, `test_crm_Au_cooling.py:20`,
`test_crm_Fe_balance.py:52,58`, `test_crm_Fe_3C3D.py:29`,
`test_W_cooling_curve_vs_putterich2008.py:80`, `test_viz.py:207`. Every skip is
*justified* (the fixture is synthetic) but the net effect is that **all four
published-data gates in the slice are skipped**. On top of that, `plotly` is an optional
extra (`pyproject.toml:40`), so in an environment without the `viz` extra
`test_viz_plotly3d.py:66` (module-level `importorskip`, 15 tests) plus 8 `importorskip`
sites in `test_viz.py` silently skip — including the 8 real angular gates — for a total of
44/222 = 19.8 %.
**Remediation.** Add a CI job that installs `.[all]` and runs with
`-p no:cacheprovider --strict-markers` plus a guard test asserting
`len(skipped) == <expected>` so a new silent skip fails the build; move the angular
normalisation tests out of the plotly-dependent module (F10).

### F12 — S2/S3 — provenance and duplication details worth fixing in place
- `tests/test_lineshapes.py:131` — "Griem table: H-α HWHM ≈ 0.38 Å at 1e17 cm⁻³", asserted
  at `rel=1e-6` against the identical literal in `src/atomx/lineshapes/stark.py:43`
  (`_HOLTSMARK[(3, 2)] = 0.38`) with the scale factor `(ne/1e17)^{2/3} = 1`. It can only
  detect a divergence between two copies of the same number, not a wrong table. Keep the
  citation, loosen to `rel=1e-3`, and add the A-class scaling that is already there at
  `:136` (Δλ ∝ n_e^{2/3}, exact 4× for 8×) as the primary gate.
- `tests/test_average_atom.py` (9 tests) duplicates `test_plasma_physics.py:368/395`
  (limits, budget closure, 2(2l+1) capacity) with looser tolerances
  (rel=5e-2 vs rel=1e-12). Keep one, mark the other `snapshot`.
- `tests/test_cascade.py:75` duplicates `:52`. `test_plasma.py:94,109,126` triplicate one
  assertion (F7).
- `tests/test_rcfp.py` (8 tests, 1 slice's worth of D) pins ±√2, −1.0, 0.5 with no
  derivation; the values are the standard RCFP reduced-subshell coefficients. Add the
  convention reference (Gaigalas/Rudzikas reduced subshell coefficients) and the derivation
  `⟨j=1/2‖W⁰‖j=1/2⟩ = −1` to promote these to A.
- `tests/test_convol.py:149` claims "the cell result is the reference for exact B-spline
  integrals" but the tolerance is 2 % and no exact integral is produced anywhere; either
  produce the exact value on a refined grid (Richardson) or drop the claim.

## 4. Model tests (already first-principles — imitate these)

1. `tests/test_plasma_physics.py:97` `test_crm_lte_limit_recovers_saha` — the theorem is
   named (Planck field at T_rad = T_e ⇒ Saha, *independent of the cross-section*), it is
   checked at three densities to 1e-9, and it fails for a missing channel (measured: 1 of the
   4 RR-failures) and for an RR/PI inconsistency (RR × 2). It is the only test in the slice
   that would have caught the session's bug at the level at which the bug lived.
2. `tests/test_plasma_physics.py:154` `test_crm_reaches_a_coronal_balance` — a *limit*
   (n_c/n_b independent of n_e), not a value; fails without RR; the pre-fix numbers
   (4.44e18/4.44e14/4.44e10) are recorded in the docstring.
3. `tests/test_plasma_physics.py:177/195/211` — the DR triple: the closed form with the unit
   convention named, the Saha-detailed-balance identity at four (χ, T), and one resonance
   checked against the Burgess formula with E, Γ_A, Γ_r, J written out and the 50.186×
   historical error recorded.
4. `tests/test_plasma_physics.py:234` — the underflow test derives the de-excitation from the
   van Regemorter large-x asymptote inside the test body and then checks the Boltzmann
   limit; it also asserts the *zero* that should be zero (`matrix[1,0] == 0.0`).
5. `tests/test_plasma_physics.py:412/442/468` — the rate-coefficient trio: closed form with
   the constant re-derived from cgs constants to 1e-4, a genuine two-kernel cross-check
   (1.53e-5), and the ideal-gas `<σv> = σ₀√(8kT/πm)` limit.
6. `tests/test_plasma.py:68` `test_stewart_pyatt_ipd_matches_limits` — both analytic limits
   (Debye `Z/λ_D` at u→0, ion-sphere `1.5 Z/R_IS` at u→∞) reached by *choosing T to put the
   coupling parameter there*; rel 1e-4/2e-4. `test_plasma_physics.py:289/346` are the same
   idea with the small-u series `1 + u³/6` derived in the test.
7. `tests/test_atom_inputs_wired.py:103` `test_plasma_shift_matches_first_order_theory_from_the_module`
   — perturbation theory against an *unscreened* reference state (a genuinely independent
   side), with the 2.4 % relaxation quoted. `:139` (pressure ionisation of a neutral atom
   at 1e22 cm⁻³) is the kind of limit test that a channel-omission bug cannot fake.
8. `tests/test_crm_dw_rates.py:28/54` — kernel exactness (`Υ = const` to 1e-12) and the
   analytic rate plus an explicit *discrimination* assertion: the van Regemorter fallback
   must differ from the DW value by >5 % (`:69`), so the test cannot pass by both paths
   collapsing to the same number.
9. `tests/test_fitting.py:12/25/53` — the inverse problem (recover known scale factors to
   1e-10), and a properly-formed statistical check on seeded noise (χ²/dof ∈ [0.5, 2.5]).
10. `tests/test_convol.py:250/257` — He/Be total energies against named DBSR_HF values
    quoted in the test (-2.861816 / -14.5730 Ha); the only true B-class oracles that
    actually run in this slice.
