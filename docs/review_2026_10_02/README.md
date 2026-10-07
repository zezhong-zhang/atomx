# Critical review — physics, numerics, performance, code (2026-10-02)

Reviewed commit: `5018fab` (`main`), plus the dirty working tree where noted.
Method: seven parallel review slices (structure/CI/corrections, angular+properties,
scattering, plasma/spectrum, numerics, performance, code/tests). Each slice
re-derived the 2026-09-23 review's claims from current source and probed the live
code; all probes are in [`probes/`](probes/) (subdirectories per slice where the
provenance is organized; flat `probe_*.py` files come from the
angular/properties, scattering, and performance slices, filenames indicate the
target). Probes were run with `.venv/bin/python` under
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; the repository was not modified by
the review.

**Caveat — live working tree.** A concurrent session edited
`methods/{rpa,mbpt,mbpt_transitions,mbpt_sigma2,sparse_ci,ci_mbpt,slater_scaling}.py`
and several tests *during* the review (~31 dirty files). Findings below refer to
committed HEAD unless marked "working tree"; the RPA items are called out
explicitly where HEAD and the in-flight rewrite diverge.

## Bottom line

Every one of the twelve 2026-09-23 findings is **fixed, probe-verified** (table
below), and the test-audit honesty remediation substantially landed. The
correctness risk has migrated to the newest layers: the committed RPA/TDHF
kernel is the wrong mathematical object, the multichannel R-matrix inner region
is measurably broken while being advertised as shipped, and a family of ungated
magnitude errors (missing `1/ΔE` in the CRM van-Regemorter fallback, a
`2·ΔE_Ry` factor in the Born dipole cross section, box-confined pseudo-Rydberg
states inflating RR totals, an `id()` identity bug disabling VN−1
polarizabilities on CI results) lives on secondary paths whose tests assert
shape or sign but never magnitude against an independent anchor.

## 1. Verdicts on prior findings (re-derived from current code)

| 2026-09-23 finding | Verdict | Evidence |
|---|---|---|
| F1 MCDHF nonorthogonal orbitals into orthonormal CI; inner failure suppressed | **Fixed** | `Atom(2, ['1s2','1s1 2s1'], …, tol=1e-10)`: `<1s|2s>` = 3.8e-16 (was 0.053); spurious 0.0062 Ha lowering gone; polar retraction `numerics/bspline/orbital_optimization.py:211-219`, Gram tripwire `physics/csf/matrix.py:1222-1250`, outer `converged` from the variational gradient (`methods/csf/solver.py:1229`), inner verdict separately reported |
| F2 MCDHF transition amplitudes ~4×10⁸× wrong | **Fixed** | `properties/transitions/calculator.py:496-623` contracts `Σ c_r c_s ⟨CSF_r‖T^L‖CSF_s⟩` with exact reduced CSF tensors. He `["1s2","1s1 2p1"]`: singlet A = 2.6511×10⁹ s⁻¹ = the review's independent-operator reference to 5 digits; intercombination A = 16.1 s⁻¹ (old code: 4.4×10⁹; physical He 2³P₁ ≈ 50.85 s⁻¹). An independent determinant-algebra reconstruction reproduces production to all printed digits |
| F3 multichannel R-matrix matching computed K⁻¹ | **Fixed** | `r_matrix/cc_outer.py:591-606` solves `(G−aRG′)K = aRF′−F`, flux-normalized reference waves (L583-588), hard `ArithmeticError` on K asymmetry. Probe (3 open channels): K symmetric to 0.0, S unitary to 2.2e-16 |
| F4 Ω divided by (2l_i+1), violating reciprocity | **Fixed** | `cc_outer.py:169-171` averages projectile spin only; Ω exactly symmetric, `g_i k_i²σ_ij = g_j k_j²σ_ji` to 1e-3 |
| F5 `CRM.from_atom` inverted emission direction, g=1 | **Fixed** | `plasma/crm.py:286-299` sorts by energy, rejects dE≤0; g = 2J+1 from CSFs / 2\|κ\| for orbital levels. Ground-state column of R at ne=0 is exactly zero |
| F6 spontaneous RR = Planck/Saha (implicit thermal bath) | **Fixed** | `crm.py:668-725` integrates the spontaneous Milne/Maxwell kernel `exp(x)E1(x)`; module/direct = 1.00000 at T = 0.01…1360 eV, incl. 7.63325e-13 cm³/s at 0.01 eV (old code returned 0) |
| F7 MCDHF ignored `r_max`/grid overrides | **Fixed** | U 1s point nucleus: string/list routes both honor refinement (−4861.189024 Ha, +0.24 eV; was stuck at +97.7 eV); forwarding `atom.py:937-938` → `methods/csf/solver.py:1066-1069`; inert overrides now raise (`atom.py:676-710`) |
| F8 shooting PI/RR divided by \|κ\| instead of 2\|κ\| | **Fixed** | both paths divide by 2\|κ\| (`radiative_recombination/__init__.py:249-251,297`); H 1s PI vs Stobbe 0.99–1.03 |
| F9 occupation floors leaking into reported occupations | **Fixed** | He `['1s2','1s1 2p1']` occupations sum exactly 2.0 (was 2.002); floors confined to the solver |
| F10 continuum scans re-diagonalized identical (H,S) per energy | **Fixed** | `numerics/bspline/continuum.py:113-180` caches the eigensystem per (κ, Z, content fingerprint), LRU-8. Re-measured: 80 energy points → exactly 1 `eigh` per κ, 0.046 s (was 160 eigh / 3.825 s) |
| F11 exact-CI size guard after combinatorial allocation | **Fixed** | `angular/csf_hamiltonian.py:1091-1104` counts dimensions before enumeration; lazy per-M sectors; bounded caches. Re-measured on the review's pattern: rejection in 0.01 ms, 0.00 MiB (was 0.43 s / 67.4 MiB) |
| F12 `solver='davidson'` silently dense-blocked | **Fixed** | `methods/ci.py:163-196` dispatches per (2J,parity) block, matrix-free `LinearOperator` route (`physics/csf/matrix.py:613-630`); dense path uses `subset_by_index` |

Related prior claims:

- **Breit scalar 3.03× defect — physics fixed, audit narrative stale.** String and
  list routes now agree to <0.0001 eV (was 0.90 eV); Ne `E_Breit` = 1.000015×
  DBSR_HF; the CI operator carries retardation. But `docs/test_audit/README.md`'s
  "the scalar is deleted" describes neither code nor reality:
  `corrections/breit.py` survives (with the 1/(2L+1) fix) as opt-in
  `scalar_legacy`, and the single-config default is the DBSR_HF port
  `dbsr_breit.py` (`stack.py:106`). Correct numbers everywhere probed; the
  documentation should be updated (P3).
- **Test-audit honesty patterns — mostly fixed.** 59 `@pytest.mark.snapshot`
  marks across 28 files (was 1); `xfail_strict = true` confirmed; pint/pyscf in
  the dev extra; the fabricated "Grant 2007" citation is now quoted as a
  documented defect; `benchmarks/mscheme_ci_reference.py` docstring states
  precisely what it does not certify; `tests/test_pi_rr_vs_fac.py` gates the real
  sfac panel; hyperfine tuned constants openly labeled, first-principles coverage
  at `test_hyperfine.py:407`.
- **S5 DW high-energy Born completion — partially fixed; physics still present.**
  `BORN_TOL` tightened to 0.25 (below the 0.2929 blend floor), but
  `pytest --runxfail` reproduces Ω_DW/Ω_Born = **0.109 / 0.034 / 0.009** at
  50/100/200× threshold. The shipped Ω is still propped up by
  `tail = max(tail, om_partial)` (`distorted_wave/solver.py:489`), structurally
  pinning the blend at 0.71–0.77×Born; the analytic Bethe tail named in the
  xfail reason is not implemented.
- **Auger KLL 10% gate — still strict-xfail** (average-of-configuration angular
  factor overcounts J-resolved resonances; live/FAC = 1.13–14.8). README's table
  is accurate about this.
- **CRM/plasma FAC benchmarks — still synthetic placeholders**, now honestly
  disclosed in README, benchmark docstrings, and skip reasons. No real
  atomx-vs-FAC CRM balance validation exists yet.

## 2. New P1 findings (wrong physics through public paths)

### P1-1 — RPA/TDHF A-kernel is the wrong mathematical object (committed HEAD; rewrite in flight)

Committed `methods/rpa.py` builds `A[i,j] = _pair_v_J(p, h'; p', h)` — the
particle-*particle* J-coupled antisymmetrized pair element — where TDHF requires
the particle-hole (Pandya-recoupled) tensor. Measured on He 1s²+2p canonical
virtuals: committed diagonal kernel +0.2676 Ha ≈ +F⁰(2p,1s) vs the CI-exact TDA
block (`H_CI[J=1,odd] − E(1s²)·I`) ≈ −0.036 Ha. At HEAD its own gates
`tests/test_rpa_tdhf.py::test_g3/test_g4` were red (anti-screening
α_RPA > α_DF). The uncommitted working-tree rewrite fixes the pole structure
(A matches the CI-exact TDA diagonal to all digits) but retains a
Pauli-forbidden `B[0,0] = 4.55e-7 Ha` for `<(2p1/2)²J=1|v|(1s1/2)²J=1>`
(exactly 0.0 through the validated pp kernel), and the rewrite **deleted the
screening-direction gate** (`alpha_rpa < alpha_df`) instead of repairing it —
the remaining residual checks are satisfied by any assembled A/B by
construction, so the suite no longer gates the response's sign physics.

### P1-2 — Multichannel R-matrix corrupts the elastic channel (exported as production)

`r_matrix/cc_inner.py` assembles the overlap channel-diagonal only (L251) and
never orthogonalizes projectile splines to target orbitals; the antisymmetrized
channel functions have O(1) inter-channel overlap while the Hamiltonian blocks
assume orthonormality. Measured (converged in basis and box): adding the closed
2s channel (5.5 eV below threshold) moves the elastic e+H ³S phase shift by
**−0.61 rad** — physically impossible; the 2p polarization channel shifts it
+0.10 rad (true shift ~0.01–0.02); even the 1s-only singlet (2.926) is far
above static exchange (~2.4–2.55). The He ¹Sᵉ(2s²) resonance does not exist in
the computed eigenphase, yet `extract_resonances` returns a confident fake
(E_r = −0.537 Ha, Γ = 0.063 Ha vs true −0.778/0.005). Production impact:
`r_matrix_excitation_cross_section` (docstring: "the production counterpart to
dw_excitation_cross_section") gives Ω(e+He⁺ 1s→2s, 60 eV) = 0.2517 vs the
FAC-verified DW 0.1871 (+35%). The maintainers' redesign spec
(`docs/RMATRIX_INNER_REGION_SPEC.md`, issue 7) already lists the missing
orthogonality as required work. **The module should not be listed as a shipped
capability, and its public entry points should warn until the redesign lands.**

### P1-3 — `rr_cross_section_rydberg` captures into box-confined pseudo-bound states

`radiative_recombination/__init__.py:703-718` takes every eigenpair with ε<0 as
a capture target with no containment check. In the default 50 a₀ box the 5th+
s-eigenstates are box-confined artifacts with inflated matrix elements.
Measured per-shell α_RR at 10⁴ K: n=1 ≈ 0.8× Storey–Hummer (box scatter), n=5
≈ 6× high, n=6 ≈ 30× high; **total Case A = 1.35e-12 vs S&H 4.18e-13 (3.2×
high)**, growing with `n_max_per_kappa`. Exported in `__all__`, silent; the only
gate uses `n_max_per_kappa=2`, blind to the artifact. Fix is small (containment
cut: require hydrogenic extent ≪ R_box or drop states deviating from the
Coulomb series).

### P1-4 — CRM van-Regemorter fallback missing 1/ΔE

`plasma/crm.py:381-384`: the constant 1.578e-5 already absorbs R_H, so the
standard form `C = 1.578e-5 · f · ḡ · e^(−x) / (ΔE_eV · √T_eV)` requires the
`1/ΔE_eV` that the code omits. Probe of the extracted rate matrix vs the closed
form: ratio exactly ΔE_eV (9.887 at ΔE=10 eV — residual is the Gaunt-floor
branch — 100.000 at 100 eV, 1.000 at 1 eV). The module's own Ω-path agrees
with the closed form (ratio 1.002), so the two collisional paths in the same
file disagree by ΔE. Impact: `from_atom` without DW tables (the default)
overestimates collisional excitation by ΔE in eV and underestimates sub-eV
fine-structure rates; LTE limits still pass because both directions share the
error — which is why the suite is green. `tests/test_plasma_physics.py:312-318`
and `tests/test_crm_satellites.py:157-161` pin the wrong formula as
"closed form".

### P1-5 — Same bug family in `born_excitation_dipole`

`scattering/distorted_wave/born.py:91-94`: σ = 8π²a₀²fḡ·ln(u)/(√3·E_Ha) vs the
Bethe-matched standard; ratio exactly **2·ΔE_Ry** (1.499 for Lyα, 14.70 at
ΔE=100 eV, 0.735 at 5 eV). Ungated — `tests/test_born_cb.py` checks shape and
scaling, never magnitude.

### P1-6 — `potential_mode="VN-1"` silently degrades to VN for MCDHF/CI results

`methods/mbpt.py:233,247`: the VN−1 reduction is `reduce_id = id(outer)` …
`if id(sh) == reduce_id`, but `VariationalCSFResult.subshells`
(`methods/csf/solver.py:163-171`) is a property rebuilding fresh `Subshell`
objects per access, so the identity test never fires and no electron is removed
from the core potential. Measured: H as `Atom(1, config=["1s1"])` → α(0) =
4.1828 vs exact 4.5 (virtuals contain no bound p-states); the same atom as
`config="1s1"` → 4.4998. He: VN−1 ≡ VN on the list route (identical, silently).
Public paths affected: `dipole_polarizability`, `sternheimer_edm_enhancement`
(both default `potential_mode="VN-1"`). One-line fix: match by `(n, kappa)`,
not `id()`.

## 3. Numerics findings

The core numerics verified solid (Section 6). New/standing issues:

- **P1 (narrow trigger) — B-spline engine ignores the auto-box.**
  `_resolve_r_max` (max(75, 5n²/Z), `atom.py:720-734`) is applied only on
  shooting routes; B-spline physics-grid routes pass the raw `self.r_max`
  (None → hardcoded 50 a₀, `bspline/knots.py:73-74`). Probe:
  `Atom(1, config='6s1', engine='bspline', …)` → E(6s) = −0.002733 Ha vs exact
  −0.013889 (**80% error**), `converged=True`, no warning. `Basis.r_max`'s
  docstring (`atom_inputs.py:437-440`) falsely claims the auto box is "used by
  both engines". Default engine resolution (shooting for single-config HF)
  limits the blast radius.
- **P2 standing limitation, now honestly documented — point-nucleus high-Z
  B-spline grid.** U 1s default grid error +97.7 eV, converging algebraically
  in the r^γ cusp (+2.9 eV at ns=400). Default `nuclear='fermi'` regularizes
  the origin; documented in `atom_inputs.py:450-452`.
- **P3 — Z≤2 special grid density is counterproductive** (`knots.py:81-90`):
  the premise is stale; the default density now gives H 1s to −5.8e-14 Ha vs
  the special-cased +3.6e-12 Ha at 1.6× cost.
- **P3 (latent) — `bsplvd` returns zeros for 2nd+ derivatives**
  (`bspline/basis.py:102-123`, dead loop). No production caller; a wrong-zero
  is worse than NotImplementedError.
- **P3 — legacy `green_solve` tail cancellation kept as default**
  (`shooting/dirac_green.py:98-101`); `stable_tail=True` exists but is off for
  "reproducibility controls".
- **P3 — legacy power-series Slater path accepts unconverged inner sums at
  r<1e-6** (`shooting/coulomb.py:159`); opt-in only, default SLAG6 is
  machine-accurate.
- **P3 — HYP-1590 transient-NaN root cause unattributed.** Tripwires convert
  load-correlated NaNs into loud raises, but the asserted BLAS/LAPACK
  corruption is unproven; a finite-but-wrong result would pass every tripwire.
  Worth an upstream reproducer or a pin.
- **Doc bug — README's determinism entry point `scripts/check_determinism.py`
  does not exist.** The real gate is `tests/test_determinism.py` (passes,
  including the NUMBA_BOUNDSCHECK=1 panel in `nightly.yml`). Production kernels
  run with bounds checking off; a new OOB would read heap garbage until the
  nightly catches it.

## 4. Performance findings

All three 2026-09-23 perf findings are fixed with measurements (F10/F11/F12,
table above). README timing claims are **conservative**: warm Ne DHF measured
0.157 s (claimed 0.86 s), Fe XVII 0.131 s (claimed 0.58 s), shooting+Gu 0.057 s
(claimed 50 ms) — single-threaded, so 4–5× margin. Cold→warm gap ≈ 18×
(one-time Coulomb bank build); `import atomx` 0.19 s. Numba hygiene: 113/114
`@njit` with `cache=True`, no object-mode risk, `parallel=True` on exactly 2
kernels, fastmath on 4 (matching the ledger).

Open items:

- **P2 — Cauchy–Schwarz integral screening not landed**
  (`angular/csf_hamiltonian.py:1173-1186` still iterates every distinct radial
  key): measured 33.8 s of an 80 s Fe XVII variational run; estimated 2–5× on
  that phase. The largest verified open lever.
- **P3 — continuum cache re-hashes 0.53 MB per hit** (327 µs = 92% of per-hit
  cost, `continuum.py:121-140`); memoize the geometry digest on the workspace.
- **P3 — `_BANK_CACHE_LIMIT = 4` bounds grid count, not bytes**
  (`convol.py:1384`); one bank can be hundreds of MB, so worst-case retention
  is multi-GB. The byte budget the 2026-09-27 B-spline performance review asked
  for is still absent.
- **P3 — no production thread pinning** (zero `OPENBLAS/OMP` references in
  `src/atomx/`); 2 numba `parallel=True` kernels × default OpenBLAS threading
  can oversubscribe on 32-core hosts.
- **P3 — `solve_symmetric_rpa` uses nonsymmetric `scipy.linalg.eig` on the full
  2n×2n matrix** (`rpa.py:113`): a structured route would be 4–8× cheaper, but
  the docstring documents a previously *wrong* Cholesky shortcut caught by the
  residual gate — a deliberate correctness-first choice. Revisit only if RPA
  spaces grow.
- HPC-note landing status (grep-level): §2.2 h0-carry landed as designed;
  §2.3 metric-check cadence landed (`ATOMX_CI_METRIC_CHECK_EVERY`) —
  **superseded 2026-10-05**: the env-tunable cadence was removed (finding
  E1 of `docs/code_review_2026_10_05/README.md`); the audit is now memoised
  per workspace on an orbital-and-radial-content fingerprint
  (`matrix._orbital_metric_fingerprint`), with no env knob; §2.4
  frozen-CI partially landed; §2.5 screening not landed (above); §2.6 convol
  prange intentionally not landed; §3.2/§3.3 Coulomb-tail continuation and
  batched cells not landed; §4.2 CI-line caching landed; §4.4 local-window
  deposition landed; §4.1/§4.3 not landed.

## 5. Code quality, docs, test harness

- **P2 — HEAD is red twice.** Pinned `ruff check --no-cache src tests scripts
  benchmarks` (ruff 0.15.6) reports **1100 errors** (565 benchmarks / 251 src /
  200 tests / 84 scripts: I001 ×375, E702 ×277, B023 ×126, E701 ×103, F401
  ×77…). `tests/test_scaffolds.py::test_rpa_scaffold_import_and_raises` fails:
  the RPA merge deleted the scaffold entry point but not its test. The lint
  gate was created because "the lint job could not fail" — the merge evidently
  landed without it passing.
- **P2 — PROJECT_INDEX.md is actively misleading** (stale since 2026-09-23):
  calls `r_matrix/`, `g_factor/`, `polarizability/`, `edm/`, `pnc/` "stubs / in
  progress" — all now have real implementations. README overstates in the
  opposite direction ("PNC" raises NotImplementedError for the actual
  amplitude; EDM's Sternheimer path self-reports a 42.85× mismatch at Z=1).
  Three documents disagree with each other and with reality.
- **P2 — remaining gate-quality gaps:** tolerance-justification policy
  half-followed in the newest tests (`test_slater_scaling.py` gates at
  abs=0.05 / 60–80 cm⁻¹ with no stated measured residual); 42 `skipif`/skip
  sites remain, including convergence regressions reported as skips;
  `test_high_density_approaches_boltzmann`-class honesty is fixed, but two CRM
  tests pin the wrong van-Regemorter formula as "closed form" (P1-4).
- **P3 cluster:** broken export (`methods/ci_mbpt.py:61` `__all__` names
  `CIMBPTResult`, defined nowhere; `import *` raises); dead-on-arrival
  committed scripts (`scripts/fk_series_validation.py` execs a `/tmp` scratch
  path; `scripts/eu3_tierA_anchor.py:70` references undefined `x`); stale
  docstrings (`results.py:603` "B not yet implemented" — it is;
  `properties/transitions/angular.py:9-11` inverts the E/M parity rule;
  `g_factor/__init__.py:95-104` `g_factor_qed` ignores κ and applies the 1s
  Grotch term to every shell; `two_photon.py:19` header quotes 8.22906 vs
  shipped 8.22943; `screening.py:329` Debye-shift direction reversed;
  `stark.py:32` says nm, returns Å; `uta/moments.py:16-18` advertises a
  nonexistent helper); `ls_jj_transformation` returns a non-square matrix for
  unphysical 2J instead of raising (`angular/recoupling.py:80-82`);
  `einstein_A_*` return silent 0.0 for ω≤0; `cascade.py` has no cycle guard;
  `steady_state` masks sign errors with `np.maximum(populations, 0)`;
  `atom.py` `_solve_fixed` remains a ~355-line branching dispatcher.
- **Verified good:** constants single-sourced (CODATA 2022; no hardcoded
  137.03/27.21 outside docstrings); no bare `except:` (28 `except Exception`
  sites warn rather than swallow); lazy `__init__` gated by an AST-walking
  architecture test; numerics upward-import ban enforced by AST;
  3976 tests collect in 3.7 s; README's ten-argument table matches the `atom.py`
  signature; the two radial engines share kernels (`numerics/yk.py`,
  `physics/slater.py`) rather than drifting; repo hygiene clean (no tracked
  `__pycache__`/artifacts).

## 6. Verified correct (coverage of what passed)

- **Angular algebra:** Wigner 3j/6j/9j + CG vs sympy ≤ 5.8e-16 over ~54k
  sampled cases; Condon-Shortley consistent; LS term tables (p², d³, f²)
  reproduce; spinor harmonics normalized.
- **Transitions:** H 2p→1s E1 A = 6.2683×10⁸ vs exact 6.2649×10⁸ s⁻¹; E2 3d→1s
  matches closed forms to 4e-5; M1 2s→1s matches Drake's Z¹⁰ law within 0.6%;
  length/velocity gauge agree to 8 digits on exact eigenfunctions, and tests
  gate gauge equality only where it is a theorem.
- **Structure:** Ne DHF total −128.69192584306 vs DBSR_HF −128.69192584304
  (2e-11 Ha); He end-to-end −2.8617218 Ha; MBPT(2) He E2 = −0.03696 Ha
  (literature-consistent); Hylleraas-Undheim nesting holds under CSF-space
  growth; CI+MBPT σ⁽¹⁾ injected-shift identity exact to 2e-16.
- **Corrections:** Breit operator Ne = 1.000015× DBSR_HF (retardation landed);
  QED auto-backend (Mohr ≤ Z36 / Indelicato ≥ Z37) consistent on both routes;
  Fermi potential V(r)·r → −Z to 12 digits; hyperfine DHF baseline A(H 1s) =
  1421.272852 MHz = exact Dirac point-nucleus value; NMS sign correct (H–D
  anchor); g(H 1s) = 2.002283762 vs CODATA 2.002283845 (documented
  leading-order).
- **Numerics:** knots clamped with correct multiplicities; n_gauss=ks
  quadrature exact for overlaps (2e-15); Gram cond ≤ 8e5 at production sizes;
  SLAG6 Slater integrals 1e-13…1e-15 vs mpmath 40-digit; TRK sum rule on the
  RKB pseudospectrum 1.49987/1.5; SCF convergence measures the pre-mixing
  fixed-point defect on both engines; same-κ overlap veto at 1e-9.
- **Scattering (where gates reach):** R-matrix outer region to machine
  precision (Coulomb functions ≤ 2e-14, K a-invariant, S unitary, Ω
  reciprocal); single-channel driver vs direct Numerov 1e-4; DW excitation/EII
  FAC gates green (35 passed, 1 xfail); DR Burgess prefactor exact;
  two_electron F⁰(1s,1s) = 5Z/8 exactly; polarisation anchors exact.
- **Plasma:** Saha constant vs CODATA 1.00000000; coronal limit n_c/n_b =
  S/α_RR exactly; high-density approach to Boltzmann+Saha first-principles
  gated; free-free constant vs CODATA 4.5e-5; UTA moments exact with power
  sum rule at 1e-12; Voigt/Gaussian/Lorentzian area-conserving on uniform
  grids.
- **Performance infrastructure:** F10/F11/F12 fixes re-measured; numba cache
  hygiene; README timings conservative.

## 7. Recommended order of work

1. **Do not merge the RPA rewrite as-is**: restore a sign-physics gate
   (screening direction), zero the Pauli-forbidden B residue, and fix the red
   `test_rpa_scaffold_import_and_raises` (delete or update the scaffold test).
2. **Unlist R-matrix as a shipped capability** (README) and add a runtime
   warning on its public entry points until the channel-orthogonalization
   redesign (`RMATRIX_INNER_REGION_SPEC.md` issue 7) lands.
3. **Fix the four magnitude bugs** (each small): CRM van-Regemorter `1/ΔE`;
   `born_excitation_dipole` `2·ΔE_Ry`; `rr_cross_section_rydberg` containment
   cut; VN−1 identity match by `(n, κ)`. Add magnitude gates against the
   closed forms used in the probes.
4. **B-spline auto-box parity** with the shooting routes
   (`_resolve_r_max` on the B-spline paths), fixing the `Basis.r_max`
   docstring.
5. **Green the gates**: the pinned ruff command (1100 errors) and the red
   scaffold test; then reconcile README / PROJECT_INDEX /
   ARCHITECTURE_CAPABILITIES with reality (R-matrix, PNC, EDM, polarizability).
6. **Perf**: land Cauchy–Schwarz integral screening (est. 2–5× on variational
   MCDHF); memoize the continuum digest; byte-bound the R^k bank cache.
7. **Close the standing physics debts**: DW analytic Bethe tail (S5), Auger
   J-resolved angular factors (strict-xfail), real (non-synthetic) CRM balance
   validation.

## 8. Reproduction

Probe scripts: [`probes/structure/`](probes/structure/) (structure/CI/corrections,
incl. the RPA kernel series `p_rpa_*.py`), [`probes/plasma/`](probes/plasma/)
(p1…p10), [`probes/numerics/`](probes/numerics/) (probe_a…i), and flat
`probes/probe_*.py` (angular/properties, scattering, performance slices;
filenames indicate the target). Run with:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/atomx-mpl \
  .venv/bin/python probes/<script>.py
```

Environment: Linux x86-64, Python 3.10.21, NumPy 2.2.6, SciPy 1.15.3, Numba
0.65. Focused test files were run per slice (all green except the HEAD-red
items noted above); the full suite was not run.

## 9. Coverage limits

Not examined: `autoionization/molecular.py` internals; DR enumeration details;
`csf_recoupling.py`/`csf_hamiltonian.py` angular internals beyond end-to-end
validation; `rcfp.py` vs published CFP tables; crystal-field/f-ion pipeline
(`slater_scaling.py`, `crystal_field.py`); hyperfine at the many-electron
F-level (not implemented anywhere — absent scope, not a bug); PNC γ5 sign
convention beyond structural reading; molecular Auger; multi-threaded scaling;
full-suite run; benchmark fixture provenance. Working-tree edits in flight
during the review (RPA/MBPT family) are assessed only where noted.
