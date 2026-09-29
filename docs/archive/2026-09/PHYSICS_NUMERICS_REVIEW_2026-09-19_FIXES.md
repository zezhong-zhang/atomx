# Physics & numerics review — remediation report

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

Companion to `docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`.  Every finding from
that review is listed below with what was done and the evidence.  Format:
**finding → status → evidence**.

Environment for all runs: macOS/arm64, Python 3.13.2, numpy 2.4.4,
scipy 1.17.1, numba 0.65.0, editable install of `src/`,
`MPLCONFIGDIR=/tmp/mpl`.  Runner: `.venv/bin/python -m pytest`.

Suite at the start of this pass: **1926 passed, 53 skipped, 4 xfailed**.
Suite at the end: **2109 passed, 53 skipped, 5 xfailed, 0 failed**
(+183 tests; the extra xfail is finding N1 below).

---

## 🔴 Critical

### C1 — Uehling vacuum polarisation used `K_1(2cr)` · FIXED

`src/atomx/corrections/qed.py` now evaluates the Uehling kernel
`I(x) = ∫_1^∞ e^{-xt}(1 + 1/(2t²))√(t²-1)/t² dt` from a cached spline of
`log I + x` against `log x`.

| quantity | before | after |
|---|---:|---:|
| `I(1)` | `K_1(1) = 0.6019` | `0.17793` (quad: `0.177934`) |
| H 1s `E_VP` | -844.42 MHz | **-215.10 MHz** (analytic -217.03) |
| U^91+ 1s `E_VP` | -517.5 eV | **-97.87 eV** (independent -98.0) |
| kernel vs `scipy.integrate.quad` | — | ≤ 1.13e-10 over `x ∈ [1e-9, 80]` |

* Lamb-shift budget closes: `8396.5 (SE) - 215.1 (VP) - 6.6 = 8174.8 MHz`
  against the measured `8172.87 MHz` (was `7552 MHz`).
* Default total energies move by +0.49 eV (Ne), +3.9 eV (Ar) — the VP is now
  3.9x smaller, as the ratio `I/K_1` requires.
* Gate: `tests/test_qed_uehling.py` (16 tests) — kernel vs quadrature, the
  `-4/15` leading-order coefficient, the H 1s MHz anchor, U^91+, sign/range,
  Z-linearity, scalar/array agreement.

### C2 / M1 — the self-energy table was the wrong function · FIXED

`src/atomx/data/mohr_se.py` is regenerated **programmatically** from FAC
1.1.x `faclib/coulomb.c` (`_qed_za`, `_qed_se0`, `_qed_se1`, `_zd`, `_sd`,
`_bethe`), not by hand.  `_interpolate_se_f` now implements FAC's selection
logic exactly, and `qedmod/indelicato.py` is a thin view of the same data, so
`qed_backend="auto"` has no discontinuity.

| Z | old `F(1s)` | new `F(1s)` | ratio |
|---:|---:|---:|---:|
| 1 | 10.3169 | 10.31679 | 1.00 |
| 10 | 11.6804 | 4.65413 | 2.51 |
| 36 | 14.0012 | 2.27932 | 6.14 |
| 92 | 21.6412 | 1.47251 | 14.69 |

* `F` now falls with `Z` (1.47 at Z=92) instead of rising; the `Z=36→37`
  default discontinuity is gone (`F(36)=2.27932`, `F(37)=2.24078`).
* Verification: the published expansion (Yerokhin & Shabaev 2015 Eqs. 7–19)
  with one calibrated remainder gives `F(2) = 8.52825` vs the table's
  `8.52833`; the old table would have needed `G = +9705`.
* Default QED shift: Ne `+0.0296 → +0.0115 Ha`, Ar `+0.3495 → +0.0932 Ha`.
  Per-system totals: Ne +0.49 eV, Ar +7.0 eV, Fe^16+ +35.3 eV, Kr +154 eV.
* M1: `n ≤ 5`, `|κ| ≤ 3` now come from FAC's `_sd` row (3d5/2 `+0.0400`,
  3d3/2 `-0.0440` at Z=1) instead of `F(1s)/n³`; `n > 5` uses FAC's `Klamaq`
  Bethe-logarithm form.  A `4f5/2` and a `4d3/2` no longer receive identical
  shifts.
* Gate: `tests/test_qed_self_energy_table.py` (13 tests) — FAC spot values,
  monotone fall to Z=54, the expansion at Z=1/2/5/10, `Z=37` continuity,
  backend agreement, `l`/`j` dependence, the U^91+ 460.2 eV ceiling, and the
  H 1s 8172.87 MHz reconstruction.

### C3 — the high-Z K shell cannot be resolved by the default grid · MITIGATED (knob added)

The default knot sequence is unchanged (it is the DBSR_HF contract), but the
grid shape is now reachable: `Atom(grid_hi=..., grid_he=..., grid_hmax=...)`
forwards to `make_physics_knots`.

| grid | hydrogen-like U^91+ 1s error |
|---|---:|
| default (`hi=0.25`) | **+97.74 eV** |
| `hi=0.05` | +9.85 eV |
| `hi=0.01` | +0.93 eV |
| `hi=0.005, he=0.10` | **+0.24 eV** |

* Gate: `tests/test_bspline_grid_accuracy.py` (14 tests) pins the measured
  error *profile* per Z (8e-5 eV at Z=10 → 97.7 eV at Z=92), the monotone
  improvement in `hi`, and that `None` reproduces the historical grid exactly.
* The review's finding stands as a documented limitation of the default; what
  changed is that a user can now fix it, and the size is pinned rather than
  unknown.

### C4 — silent wrong answers · FIXED

| path | before | after |
|---|---|---|
| unbound d/f subshell | `Atom(Z=10, charge=-1, config="1s2 2s2 2p6 3d1")` → `-37686.42 Ha`, `converged=True` | `UnboundSubshellError` |
| no bound state anywhere | `Atom(Z=1, config="7s1", numerics="bspline")` → `0.0 Ha`, empty orbitals, `converged=True` | `UnboundSubshellError` |
| shooting box too small | H 6s → `-0.0133333` (4 % off), 7s (31 % off), Na 9s–13s all pinned | `_OrbitalSolveError` with the ceiling and a `r_max` hint; `r_max` auto-sizes |

* The negative-continuum threshold moved from `-2 m_e c²` to `-m_e c²`
  (`dirac_assembly.NEGATIVE_CONTINUUM_EDGE`); the collapsed state measured
  `-1.99996 m_e c²`, which is how it slipped through.
* `dbound`'s `ier` is propagated (`ier_max`), the ceiling is taken from the
  converged potential rather than `-Z/r_max` (the bare-Z form wrongly flagged
  resolved core orbitals of neutral heavy atoms), and `total_sum` is
  initialised so the early-exit paths no longer diverge between the Python and
  numba builds.
* `r_max` is exposed and auto-sizes to `max(75, 5n²/Z)`, so the H Rydberg
  series 5s–8s now solves to ≤ 5e-6 relative (was 4.8e-6…31 % wrong).
* Gate: `tests/test_silent_failure_guards.py` (13 tests).

### C5 — plasma / CRM defects · FIXED

| defect | before | after |
|---|---|---|
| CRM had no radiative recombination | `n_c/n_b ÷ Saha/n_e` = 10.73 at `n_e=1e18` | **1.000000000000** at 1e18, 1e22, 1e24 |
| DR prefactor (kT in Ry vs eV) | `6.60305e-24` | `3.3128303850821445e-22` (×50.1712), closed form rel 2.8e-16 |
| `exp(-dE/kT)` underflow | `C_deexc = 0`, populations `[0.5, 0.5]` | `C_deexc = 1.176e-6 cm³/s`, `[1, 0]` = Boltzmann |
| Stewart–Pyatt discontinuity | 247 eV jump at `u = 10` | continuous (rel jump 6.5e-14 at ±1e-6), limits exact |
| average atom over-occupancy | Fe: `Σocc = 769.41` for `Z = 26` | `Σocc = 26.000000` (μ iterated by `brentq`) |
| `rates.py` K-vs-eV | 107.7x high at every T | closed form rel 8.3e-10 |

* Gate: `tests/test_plasma_physics.py` (22 tests) plus re-pinned
  `test_crm_satellites.py`, `test_dr_vs_fac.py`, `test_dr_fac_reference.py`,
  `test_autoionization.py`, `test_plasma.py`.  The circularity of the FAC DR
  fixture (its α equals the old kernel's output exactly) is now documented in
  those tests.

### C6 — DW Born completion never engaged · FIXED

| `E/threshold` | `Ω` before | `Ω` after | pure Born |
|---:|---:|---:|---:|
| 2.0 | 0.04487 | 0.04451 | 0.04012 |
| 5.0 | 0.08090 | 0.08156 | 0.08228 |
| 50 | 0.01049 | **0.16387** | 0.21238 |
| 100 | 0.00412 | **0.21776** | 0.28432 |
| 200 | 0.00159 | **0.31995** | 0.41671 |

`Ω` now rises with energy and is within 23.4 % of the plane-wave Born limit
(was 20x/69x/263x low and falling).  `_geometric_tail_estimate` reports
"not converged" instead of returning 0, and the monotone cap no longer
resets to zero.  The low-energy points moved < 1 %, so the FAC gates needed
no re-pin (worst live-vs-FAC 5.22 %, gate 10 %).  Gate:
`tests/test_dw_born_completion.py` (15 tests).

---

## 🟠 High

### H0 — `collision_strength_to_rate` dimensionally wrong · FIXED

Implemented in the dimensionally consistent atomic-unit form with the missing
`g_i` restored.  `T = 1 eV, Ω = 1`: `9.178e-16 → 8.010388e-08 cm³/s`
(the review's first "exact" column had a unit slip and was corrected in the
review document by a factor `E_h`; the defect itself stands).  Verified to
8.7e-9 against an independent first-principles quadrature at T = 1/10/100/1000
eV, and cross-checked against `crm.py`'s own 8.629e-6 constant.

### H1 — the "7 keV at Z=82" limitation was a stale artifact · FIXED

Flags matched to FAC's default (`breit=False, qed=True`, now the driver's
default), the atomx column regenerated, and a staleness gate added
(`tests/test_fac_structure_snapshot.py`, 22 tests).

| Z | iso | recorded | re-measured | FAC | old Δ | new Δ |
|---:|---|---:|---:|---:|---:|---:|
| 36 | Ar-like | -70 971.6 | -71 116.5 | -71 079.9 | +108 | **-36.6** |
| 54 | Ar-like | -173 801.5 | -174 726.8 | -174 596.1 | +795 | **-130.7** |
| 82 | Ar-like | -437 028.1 | -444 462.7 | -444 036.2 | **+7 008** | **-426.5** |

Whole panel now ≤ 9.6e-4 relative.  `README.md`, `docs/VERIFICATION.md` and
`benchmarks/REPORT.md` §4 were updated; the CSV now records its flags.

### H2 — UTA Einstein A used the lower state's degeneracy · FIXED

`calculator.py:484,486`: `tj_i → tj_f`.  Exact hydrogenic Ly-α through the
real single-config path:

| row | before | after |
|---|---:|---:|
| `1s1/2 → 2p3/2` | 1.253677e9 s⁻¹ | **6.268384e8** |
| `1s1/2 → 2p1/2` | 6.268313e8 | 6.268313e8 |
| ratio | 2.0000 | **1.0000114** |

Both components are within 0.056 % of NIST 6.2649e8.  Public API Ne:
`A(1s→2p3/2)/A(1s→2p1/2)` `1.9935 → 0.99673`.  Every row now satisfies
`A = 6.6703e15·gf/(g_u λ²)` to 7.2e-6.  Gate:
`tests/test_property_regressions.py`.

### H3 — normal mass shift sign · FIXED

`mass_shift.py:138`: `-kinetic/mass_me → +kinetic/mass_me`, docstrings
re-signed.  `E_NMS(H 1s)`: `-1.791704 → +1.791704 THz`.  H–D 1s–2s:
`-671.5554 → +671.5554 GHz` (measured +670.994; exact reduced mass
+671.5554).  `tests/test_isotope.py` re-pinned in 4 places (all encoded the
inverted sign).

### H4 — SCF orthogonality and variational consistency · PARTIAL (surfaced + MBPT fixed)

* **MBPT(2) partition — FIXED.**  Virtuals now built in `V^N`
  (`mbpt.build_virtual_basis`, default changed from `VN-1`) so the
  denominator uses one Hamiltonian.  He: `E2 = -0.049871 → -0.036968 Ha`,
  `E_tot = -2.911412 → -2.898782 Ha`.  The old value was 118 % of the exact
  correlation energy and 8 mHa *below* the exact ground state; the new one is
  within 0.7 % of the literature MP2/CBS value (-0.0372) and correctly
  bracketed by HF (-2.861680) and exact (-2.903724).  Gate:
  `tests/test_mbpt2.py::test_mbpt2_helium_matches_the_literature_value`.
* **Off-diagonal Lagrange multipliers — SURFACED, not implemented.**  A
  correct implementation is a change to the SCF fixed point, not a bug fix.
  What was added is visibility: every `SCFResult` now carries
  `max_same_kappa_overlap` (measured Li 5.69e-4, Na 2.67e-4, Kr 5.31e-5,
  Ne 4.60e-6) and the result is flagged `converged=False` with an explanatory
  `orthogonality_warning` above 1e-3.  The limitation and its measured size
  (~1e-4 Ha on Li versus the Schmidt-orthogonalised shooting engine) are
  recorded in `docs/VERIFICATION.md`.

### H5 — `compute_yk` double-counted the diagonal cell · FIXED

`numerics/bspline/evaluate.py::_compute_yk_flat`: each cumulative now
contributes the half cell it owns.

| metric | before | after |
|---|---:|---:|
| max `Y⁰` rel. error vs analytic | 2.9e-2 | **4.5e-4** |
| `F⁰(1s,1s)`, Z=10 | +1.75 % | **+5.0e-5** |
| `slater` Ne total | -128.639361 | **-128.623437** |

The independent refined-grid reference measured in the review was
`-128.623363` — the fixed kernel now matches it to 5e-7 relative.  The `gu`
model moved 0.72–1.24 eV; the `hf` path is bit-unchanged (it uses the cell
kernels).  DW/EII: rms vs FAC `2.0 % → 1.95 %`, and the atomx snapshot columns
were re-recorded (the FAC columns were left untouched).

### H6 — Wigner–Eckart phase convention · DOCUMENTED + PINNED (no physics change)

Investigation showed the convention is load-bearing and deliberate:
`reduced_C` carries Grant's `(-1)^(j_a+1/2)`, a test pins it, and switching to
the textbook phase changes `ckm`'s k=0 sign in a way the FAC-parity gates do
not support changing blind.  Every consumer in atomx uses the same layer, so
the sign cancels in squared observables.  What was wrong was that nothing said
so, and the non-relativistic `angular_cc` uses the opposite convention.  Both
`reduced_C` and `ckm` now document the convention explicitly, and
`tests/test_angular.py` gained three gates: the k=0 sign, the m-resolved
identity (`ckm(k,m,k,m,0,0) = -1`), and the convention-independent
Wigner–Eckart sum rule (≤ 1e-12 over 96 checks).

---

## 🟡 Medium

| # | finding | status | evidence |
|---|---|---|---|
| M1 | self-energy `n⁻³` fallback lost `l`/`j` | **FIXED** | see C2; FAC `_sd` rows for `n ≤ 5`, `Klamaq` beyond |
| M2 | config parser failed on `n ≥ 10` and implicit occupation | **FIXED** | shared `config/parsing.py`; `Atom(Z=3, config="1s2 2s")` and `config="10s1"` now parse |
| M3 | `grid_slater_rk` argument-order docstring contradicted the code | **FIXED** | docstring corrected; the 7 call sites were already consistent |
| M4 | Stewart–Pyatt discontinuity | **FIXED** | see C5 |
| M5 | UTA "g_u·g_l proxy fails" | **REFUTED, contract documented** | with every FAC 2p5 3d→2p6 line at J_u=1, J_l=0, no (2J+1) weighting can reproduce a strength-weighted moment — the proxy is a *degeneracy*-weighted (Unsöld-limit) UTA moment.  The strength-weighted reference and the documented Unsöld deviation are now both pinned, an optional `strengths=` argument gives exact moments, and the `total_strength` docstring no longer calls a degeneracy product a line strength |
| M6 | MBPT(2) virtuals in `V^{N-1}` | **FIXED** | see H4 |
| M7 | missing orbital in the CI silently gave the uncorrelated energy | **FIXED** | `MissingOrbitalError` + a `_check_orbitals_present()` gate at `build_ci_matrix` entry.  An *entirely empty* orbital map still means "no radial data supplied" (so the angular-only unit-radial probes in `tests/test_off_diagonal_j_ci_helike_1s2p.py` keep working unchanged), while the production entry point rejects it.  The literal Be symptom turned out to be latent in this tree (the driver already solves every CSF orbital), so the guard is what makes it loud |
| M8 | same-subshell doubles dropped by `generate_sd_excitations` | **FIXED** | the `i == j` branch decremented the occupation by 3 instead of 2, so validation rejected every a²→b² double; `generate_sd_excitations("1s2", ["2s","2p","3s"], 2)` doubles **0 → 6**, Be `correlation="sd"` configs **19 → 39** |
| M9 | hyperfine moments and a factor-2 docstring | **FIXED** | Cs-133 `2.5828327 → 2.58202503`, Cu-63 `2.2273456 → 2.2233`; the spurious factor 2 removed from the documented A formula; the reduced-mass term named in the error budget — measured **(1+m_e/m_p)⁻³ = −1632 ppm**, i.e. the review's original −0.054 % was 3x too small (corrected above) |
| M10 | `dipole_polarizability` crashed on the default engine | **FIXED** | clear actionable error instead of `TypeError: 'NoneType' object is not subscriptable` |
| M11 | `ls_jj_transformation` non-unitary | **RETRACTED — not a defect** | With the documented convention (arguments are twice-J) all 33 non-empty blocks for `l1,l2 ≤ 2, J ≤ 4` are square and orthonormal to 4e-15. The review's 22/45 figure came from passing `J` where the API expects `2J`. Both suggested parity guards are *vacuous* in the doubled convention. The docstring now states the convention (with the warning that passing `J` yields rectangular matrices) and a unitarity gate covers p²/d²/mixed |
| M12 | stale CODATA-2018 literals | **FIXED** | the six duplicated constants now import from `atomx.constants`, gated against CODATA-2022 |
| M13 | `slater_condon_f` disagreed with its own docstring | **FIXED** | one convention, stated once, gated |
| M14 | `total_sum` unbound on `dbound` early exits | **FIXED** | see C4 |
| M15 | continuum phase fit omitted the Coulomb log term | **FIXED** | `−η ln(2kr)` included; the returned `δ` no longer depends on the box radius |
| M16 | `coulomb_bethe_topup` dropped `g_i` | **FIXED** | `4 g_i f/ΔE·ln` restored and gated against the exact GOS |
| M17 | bare `except Exception` silently disabled the Born completion and continuum waves | **FIXED** | failures now propagate/report instead of degrading to a wrong number |

---

## 🟠 High-performance numerics

| # | finding | status | evidence |
|---|---|---|---|
| P1 | 86 % of wall time is inside njit kernels; the cProfile attribution saying otherwise was an artifact | **documented** | no code change needed; the review's first-pass profile was corrected in the review document |
| P2 | ~14 % of runtime was redundant RKB-bank rebuilds | **FIXED** | process-wide `_RKB_BANKS` registry keyed on grid *content* (`nv, ks, ksp, ksq, nsp, nsq, ns, t.tobytes(), tp.tobytes()`); bank builds per solve **100 → 20** (`_build_rkb_sym` 50→10, `_build_rkb_pqpq`/`_build_rkb_qpqp` 25→5); Fe Z=26 ns=113 solve **9.56 s → 7.32 s (23 %)**, total energy bit-identical.  `build_ci_matrix` no longer drops the stashed `_convol_cache` across `_reset_memo()`, which its own docstring forbade |
| P3 | `fastmath` neutral-to-harmful on 16 of 20 hot kernels, and two "bit-identical" docstrings were false | **FIXED** | kept on 4 kernels that earn it (`_convol_case1_t` 3.4x, `_sum_amb_full` 3.9x, `trapezoidal_integrate` 2.2x, `_accumulate_direct_density_batch` 1.2x), dropped from 16 where it was neutral or *slower* (`_convol_case1_mixed` 1.45, `_convol_case1` 1.30, `_sum_amb_banded` 1.30, `_convol_case4` 1.28, `_convol_case2_mixed` 1.27, `_convol_case2` 1.21, …); end-to-end **9.11 s → 8.58 s (5.8 %)**, energy drift 1.06e-14 relative.  `_density_sym` is now verifiably `array_equal` to the plain band construction and `_convol_case4_t` bit-identical to `_convol_case4` |
| P3b | scaling is quadratic (exponent 1.907) | **documented** | no defect: nothing is accidentally `O(N⁴)` |
| P3c | no BLAS speedup available; nothing pinned thread counts | **documented** | `scipy` is built against Accelerate; noted in the review, no code change |
| P4 | default grid ~3x more expensive than an explicit `ns` | **documented** | the knot families are not nested, so `ns` is not a convergence knob; `grid_hi`/`grid_he`/`r_max` are now the supported knobs |
| P5 | `perf_dashboard.py --check` is a freshness check, not a perf gate | **documented** | the review records what it does and does not prove; no false claim was added |

---

## Suite

```
$ .venv/bin/python -m pytest -m "not slow" -q
2109 passed, 53 skipped, 19 deselected, 5 xfailed in 474 s
```

`xfail`s: 4 pre-existing plus the 2 tests of `tests/test_sd_space_variational.py`
(non-strict count 5 in the default run; the SD case itself is additionally
marked `slow`).

New gate files added by this pass:

| file | tests | covers |
|---|---:|---|
| `tests/test_qed_uehling.py` | 16 | C1 |
| `tests/test_qed_self_energy_table.py` | 13 | C2, M1 |
| `tests/test_silent_failure_guards.py` | 13 | C4 |
| `tests/test_plasma_physics.py` | 22 | C5 |
| `tests/test_dw_born_completion.py` | 15 | C6, H0 |
| `tests/test_property_regressions.py` | 12 | H2, H3 |
| `tests/test_fac_structure_snapshot.py` | 22 | H1 |
| `tests/test_bspline_grid_accuracy.py` | 14 | C3 |
| `tests/test_rkb_bank_sharing.py` | 6 | P2 |
| `tests/test_mcdhf_ci_silent_zeros.py` | — | M7 |
| `tests/test_uta_fac_moments_gates.py` | 7 | M5 |
| `tests/test_review_m9_m13_gates.py` | 27 | M9–M13 |
| `tests/test_sd_space_variational.py` | 4 (2 xfail) | N1, new |

plus new gates inside `tests/test_qed.py`, `tests/test_qedmod.py`,
`tests/test_qed_indelicato_vs_fac.py`, `tests/test_mbpt2.py`,
`tests/test_angular.py`, `tests/test_data_loaders.py`,
`tests/test_isotope.py`, `tests/test_fac_dfs_shooting.py`.

Re-pinned tests (each with the reason inline, because each had encoded the
bug it was guarding): `test_qed.py`'s Ne/Ar QED shifts and VP components,
`test_qedmod.py`'s n=3 expectation, `test_qed_indelicato_vs_fac.py`'s
"mohr is 8x larger than FAC" assertion (which asserted the bug as a feature),
`test_data_loaders.py`'s table literals, `test_averaging_decoupling.py`'s Co
seam pin, `test_qed_orbital_shifts.py`'s Fe K-shift window, `test_mbpt2.py`'s
He bound, `tests/test_isotope.py`'s four sign assertions, `test_fac_dfs_shooting.py`
(matched flags, gate *tightened* 5e-3 → 2e-3), and the two DW/EII snapshot CSVs
(atomx columns only).

## Found during remediation

**N1 — `correlation="sd"` is not variational (ROOT-CAUSED, FIX PENDING).**
Enlarging the Be SD space to 39 CSFs moved the energy to **-15.0536 Ha,
0.386 Ha below the exact non-relativistic ground state (-14.6674 Ha)** —
impossible for a variational CI.  The defect predates the M8 fix (the 19-CSF
set already gave -14.9002).

The first diagnosis recorded here — "the singles run puts `2p` at the
unscreened hydrogenic -2.0005 Ha, so the MCDHF average over a large SD space
appears to collapse; **the CI matrix is not the suspect**" — was **wrong**.
The follow-up audit `docs/archive/2026-09/CORRELATION_SD_RESEARCH_2026-09-20.md` isolates it by
rebuilding the CI independently (Wigner–Eckart + sympy 3j, Slater–Condon in
the m-scheme determinant basis; no angular code shared with the repo) and
running it on **atomx's own converged orbitals and the same CSF space**:

| Be space | protocol | atomx | independent, same orbitals | Δ (Ha) |
|---|---|---|---|---|
| 1s²2s² | `fac_frozen` | −14.575892 | −14.575892 | 0 |
| singles (29 CSF) | `fac_frozen` | −15.400463 | −14.595092 | **−0.805** |
| singles (29 CSF) | `mcdhf` | −14.659475 | −14.578383 | **−0.081** |
| SD (399 CSF) | `fac_frozen` | −16.358105 | −14.654637 | **−1.703** |
| SD (399 CSF) | `mcdhf` | −15.052683 | −14.620800 | **−0.432** |

Two further checks, both clean: the 399 jj-CSFs are a *complete minimal basis*
of the intended determinant space (per-configuration CSF count == m-scheme
dimension, 39/39 configurations, 399 == 399), so this is not a rank-deficient
basis; and `⟨1s²2s²|Ĥ|1s²2s¹2p¹⟩` is exactly 0 as Brillouin requires, so the
error is entirely inside the open-shell J-blocks.

*Cause.* `build_ci_matrix` never evaluates `⟨CSF_r|Ĥ|CSF_s⟩`.  It uses a
configuration-average diagonal (`_diagonal_element`) plus two ad-hoc patches
(`_diagonal_J_correction`, `_off_diagonal_J_correction`) that are applied only
in a narrow envelope and are quantitatively wrong even there — for the He-like
1s2p J=1 block the off-diagonal correction contributes **+0.431282** on a base
element of −0.193857, moving the level to −14.711459 where the exact value in
the same space is −14.490933.  The enabler is structural: `CSF` carries only
`total_J2`, no coupling tree, so the exact jj-coupled element is not
representable (`matrix.py` says so itself).  `build_ci_matrix` also returns
non-zero elements *between different total J*; they are masked today by the
`(J, parity)` filter in `ci._solve_blocked`.  `_off_diagonal_J_correction`'s
docstring still claims "Always 0.0 in the current scaffold" while its body
returns a populated value.

A **second, independent defect** was found alongside it: low-weight
correlation orbitals are solved on the wrong branch.  In the `mcdhf` Be
`correlation="s"` run the 2p, 3p and 3d radial functions come out at *exactly*
the bare-nucleus hydrogenic `<r>` (2p: 1.250 a₀ = (1/2Z)(3n²−l(l+1)) for Z=4,
n=2 — completely unscreened); with `fac_frozen` the 3d floats to `<r>` = 25 a₀
with 0 nodes.  This is the collapse already described in
`MCDHFSolver._floored_occ_map`'s docstring.

`benchmarks/mscheme_ci_reference.py` (the oracle) and
`tests/test_ci_matrix_vs_mscheme_reference.py` (3 passing invariants + 5
strict `xfail`s carrying the measured numbers) are the durable artefacts.
`tests/test_sd_space_variational.py` also carries the SD and
unscreened-orbital cases as strict `xfail`s.  This is the only
review-adjacent item left open; the fix options are costed in §8 of the
research note.

## Corrections to the review itself

Two claims in `docs/archive/2026-09/PHYSICS_NUMERICS_REVIEW_2026-09-19.md` did not survive
reproduction and are corrected in place there:

* **H0's "exact" column** was dimensionally inconsistent (the formula is only a
  rate read in atomic units).  The defect stands; the magnitude was overstated
  by a factor `E_h ≈ 2.294e10`.  Corrected figure: 8.011e-8 cm³/s at 1 eV.
* **M5's premise** was a category error — a degeneracy-weighted UTA moment
  cannot reproduce a strength-weighted one, and the measured -1.2 %/+68.6 %
  is the Unsöld-limit deviation, not a weighting bug.  The review now says so.
* **M11 was retracted entirely.**  Its "22 of 45 non-unitary blocks" figure
  came from calling `ls_jj_transformation` with `J` where the API expects
  `2J`; with the documented convention every non-empty block is square and
  orthonormal, and the two parity guards the review proposed are identically
  true in the doubled convention (i.e. they would have changed nothing).  The
  review column now records the retraction and the real hazard — the caller
  convention — is spelled out in the docstring and gated.
* **M9(c)'s missing reduced-mass term** is -1632 ppm, not the -0.054 % the
  review first quoted; the figure goes as `(1+m_e/m_p)^-3` because the Fermi
  energy is defined with `m_r^3` (Antognini, Lin & Meißner,
  arXiv:2208.04025, Eqs. 2 and 7).
