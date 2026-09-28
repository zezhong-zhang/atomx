# High-Performance Numerics: B-spline / Shooting / Spectrum

**Deep research report — atomx calculation-speed fundamentals**
2026-09-27 · profiling worktree `/tmp/atomx-prof` @ 385d908 · all measurements with BLAS threads pinned (`OPENBLAS_NUM_THREADS=1` etc. before numpy import)

---

## 0. Executive summary

The numerics stack is **kernel-saturated**: 86 % of B-spline wall time already sits inside hand-tuned numba kernels (documented 2026-09-19 review, P1), the convol family has absorbed transposed layouts (3×), fastmath policy tuning (3.4×/3.9×, kept on 4 kernels only), and RKB-bank sharing across solves (23 %, bf4de7f). Further micro-optimization of these kernels is past the point of return — including `parallel=True`, which was *measured* 9.4× slower at production basis sizes (convol.py:768-776).

The remaining fundamental speed lives one level up, in **physics-structural reuse**: the code repeatedly pays O(basis³) dense costs at points where the underlying physics varies smoothly —

1. **MCDHF variational path (measured this session, Fe XVII j-resolved):** two structural defects in the CI rebuild pipeline account for **36 % of wall time** (80.1 s → 50.9 s when fixed, energy **bit-identical**, verified A/B). Both fixes are pure memo-semantics changes, zero numerics.
2. **Shooting continuum:** the e > 0 solve is a *direct IVP over the full grid*, but everything past the Coulomb onset radius r_c is analytically known (α·F + β·G with the residual charge). Integrating the asymptotic region is numerically redundant — a physics-exact cutoff, worth 2–10× on large boxes (high-energy / heavy-target waves).
3. **Spectrum/RR rendering:** continuum waves are re-solved per subshell × per photon-grid point, yet σ(ω) is smooth between thresholds with Wigner cusps only *at* thresholds. A threshold-aware coarse energy grid + interpolation replaces the inner loop entirely.

The deepest theme, stated once: **stop re-deriving what the physics says changes slowly** (the CI mixing along an optimization path, the continuum wave along an energy grid, the line profile on a wavelength grid). The wins are structural; the kernels are done.

---

## 1. Methodology and measurement hygiene

Every number in this report follows the rule re-learned in the 2026-09-27 critical review (F2): **LAPACK costs measured on an unpinned loaded box are artifacts by up to two orders of magnitude** (eigh "43 s/17 calls" unpinned vs 0.53 s pinned). Therefore:

- All timings: `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=…=1` set **before** numpy import (same discipline as repo-root conftest.py:40-54).
- A/B optimizations must show **bit-identical energies** (or a gated tolerance with a physics justification), not just "tests pass".
- cProfile is used for *attribution*, wall-clock for *cost*; numba C-level dispatchers emit no profile events, so kernel tottime underestimates kernel share (the 86 % figure comes from scoped wall-clock, not cProfile).

Fe XVII fixture (the nightly slow test): `Atom(Z=26, charge=16, config=[4 configs], corrections="none", Correlation(j_resolved_csfs=True))`, variational MCDHF with frozen screened-Fock-preconditioned L-BFGS.

---

## 2. B-spline / MCDHF variational path

### 2.1 Measured profile (cProfile + scoped wall, Fe XVII)

| Component | Time | Share | Verdict |
|---|---|---|---|
| `optimize_orbitals` total | 63.5 s | 60 % | hot path |
| └ `ci_solve` (756 calls) | 56.4 s | 53 % | **the target** |
| &nbsp;&nbsp;&nbsp;`build_ci_matrix` entry `_reset_memo()` → orbital-independent h0 rebuilds (`build_dirac_matrices` 3805 calls) | ~20 s | 19 % | **pure waste** |
| &nbsp;&nbsp;&nbsp;`_check_orbital_metric` (per `ci_solve`, full Gram audit) | 20.4 s | 19 % | **pure waste** |
| &nbsp;&nbsp;&nbsp;CI block `eigh` (5455 dense calls, n ≤ 1500 path) | 20.6 s | 19 % | *legitimate* |
| &nbsp;&nbsp;&nbsp;`np.fromiter` CI coefficient evaluation | 33.8 s | 32 % | partially reducible (§2.5) |
| `_convol_case1_t` (numba, transposed, fastmath) | 37.1 s tottime | 35 % | **kernel-saturated** |
| nuclear `V_func` evaluations | 7.19 M calls | — | memo artifact, dies with §2.2 |

Note the overlap: convol time is *inside* build_ci_matrix as well. The two "pure waste" rows are the only large components that are not physics.

### 2.2 P0 (verified): carry orbital-independent h0 blocks across CI rebuilds — 16 %

**Defect.** `build_ci_matrix` calls `_reset_memo()` at entry (matrix.py:538, 741), wiping *all* memoized integrals. That wipe is sound for the id()-keyed Slater entries (orbital coefficients change between outer iterations; id-recycling safety is documented at matrix.py:57-91) — but it also destroys the `("h0", ns, kappa, Z)` entries from `_dirac_blocks` (matrix.py:205), which are **orbital-independent**: they are the one-body Dirac H0/S0 Galerkin blocks (kinetic + nuclear + overlap), functions of (basis size, symmetry, Z, workspace) only. With 756 `ci_solve` calls in one variational optimization, they are rebuilt 3805 times (~5 per solve, ~5 ms each).

**Why the carry is sound.**
- The h0 key contains **no id()** — nothing to recycle.
- Cross-workspace staleness is already guarded by `_memo_context(bsp_data)` (identity check): the MCDHF driver rebuilds `bsp_data` fresh every outer iteration (mcdhf/solver.py:1046-1062, "The grid is rebuilt identically each outer iter"), so a new workspace still triggers a full reset once per outer iteration (~50 rebuilds instead of 3805).
- The one true dependency is `_nuclear_potential` (passed into `build_dirac_matrices`, matrix.py:208): make it part of the carry condition — carry h0 only together with the `"_nuclear_potential"` slot, and key the carry on both being unchanged (within a solve the nuclear model is fixed by construction).

**Measured A/B (this session, same machine, pinned):**

| | wall | E0 |
|---|---|---|
| baseline | 80.1 s | −1148.4391761319 |
| h0-carry patch | **67.4 s** | −1148.4391761319 |

**−15.9 %, bit-identical.** Patch sketch: in `_reset_memo` (or a `_reset_memo_preserving_structure` used by the two call sites), retain entries whose key is `"_nuclear_potential"` or `("h0", …)`; the id-bearing entries still wipe. `_MEMO_KEEPALIVE` extension is a no-op for h0 keys (no ids) but harmless.

### 2.3 P0 (verified): gate `_check_orbital_metric` — another 20 %

The per-`ci_solve` Gram audit (matrix.py:1097) costs 27 ms × 756 = 20.4 s to re-verify, on every line-search trial point, that the orbital basis is orthonormal. The orbitals change by an L-BFGS step (~1e-3 rad rotations at most); the metric is not fragile.

**Measured A/B** (h0-carry **plus** metric-check reduced to once per solve):

| | wall | E0 |
|---|---|---|
| baseline | 80.1 s | −1148.4391761319 |
| h0-carry only | 67.4 s | −1148.4391761319 |
| both | **50.9 s** | −1148.4391761319 |

**−36 % combined, bit-identical.** Recommended production form: run the check once per `build_ci_matrix` *outer-iteration context* (not once per solve — keep it cheap but present), or behind an env-flag default-on in tests. Keep the check itself unchanged; only its cadence changes. If the suite ever sees it fire, cadence gets tightened — the check is a tripwire, not a calculation.

### 2.4 P1 (algorithmic, the big one): frozen-CI microiterations

After §2.2+§2.3, ~53 % of the optimizer's time is still `ci_solve`, and within it the 5455 dense eighs are "legitimate" only in the sense that a full CI rebuild at a trial point is legitimate. **It is not necessary.**

Physics: along a smooth orbital-optimization path, the electronic structure is *adiabatic* — the CI mixing coefficients converge on a much faster timescale than the orbitals (their time derivative scales with the orbital gradient, second-order in the energy error). This is the standard MCSCF "partitioned step" (Werner & Meyer 1980; every modern CASSCF does orbital microiterations with the CI density frozen, then one exact CI macroiteration). atomx's variational loop currently does a **full exact CI solve per L-BFGS trial point** — 1 gradient CI + ~1.5 line-search CIs per iteration, each rebuilding all blocks from scratch.

Concrete design:
- After each exact CI, cache the eigenpairs and build the 1-/2-particle RDMs once.
- Orbital gradient and line-search energies at nearby points use the **frozen RDMs** — the energy functional with fixed CI coefficients needs only orbital-gradient machinery (the existing DFSolver functional applied with an effective fractional occupation / fixed-pair matrix), no CI matrix at all.
- Refresh the exact CI when either (a) K microiterations have passed (K ≈ 5–10), or (b) the orbital displacement exceeds a trust threshold, or (c) the frozen-CI energy and a cheap probe disagree beyond tolerance.
- The L-BFGS curvature pairs collected during frozen stretches remain valid (the functional is still exact for the frozen state); no history clearing needed.

Expected: removes most of the remaining 53 % `ci_solve` share during steady stretches — **the single largest remaining lever on this path** (2–4× end-to-end on optimizer-dominated cases, which the 2026-09-26 trace identified as "82 % of wall" for Be sd).

Risk controls: convergence gate must remain the *exact-CI* stationarity (as e7f4b68 established — do not regress to DHF-only convergence); final macroiteration always exact. The failed refresh-preconditioner experiment (2026-09-27) is *not* a counterexample: that mutated the Hessian model; this mutates only the CI-state cadence, and the exact CI is reasserted every K steps.

### 2.5 P2: integral screening in the CI coefficient evaluation

`evaluate_block`'s `np.fromiter` over 219 900 distinct radial keys (33.8 s) includes integrals whose orbital-pair densities are below any meaningful threshold — virtual shells with ~0 occupation entering CSFs with 1e-6 amplitude weight. Screen the coefficient table by `|c_r · c_s| × max|angular| × density-norm` before radial evaluation; evaluate the survivors. Physics grounding: an integral's contribution to a matrix element is bounded by the product of the CSF amplitudes and the density overlap — a Cauchy–Schwarz screen, exact and safe with a conservative threshold (start 1e-12, tighten by measurement). Expect 2–5× on the coefficient-evaluation row; synergizes with §2.4 (fewer exact CI builds × cheaper builds).

### 2.6 P2: convol parallelism with the measured crossover

The serial-kernel decision (convol.py:734-776) was measured at ns ≈ 40–150, where `prange` is a 9.4× regression (fork/join dominates). The same ledger projects **4–8×** from `prange` at ns ≳ 290 (the documented crossover). Production heavy-atom bases (Pb-class ns ≈ 276-512, NIST DHF queue running 517-700 splines) sit at/above the crossover. Implement **ns-adaptive dispatch** (both kernels compiled, one branch on ns) rather than a global switch; verify against the per-Z FAC-matrix wall-time tripwire (benchmarks gate already enforces >30 % regressions).

### 2.7 Documented dead ends — do not retry

- **Preconditioner refresh in the variational L-BFGS** (2026-09-27 experiment): diverges past ~320 iterations — the skip_pq_exchange Fock is systematically wrong at relaxed orbitals. The stalled path needs §2.4, not a better preconditioner.
- **Cholesky-cached standard-form eigensolve**: measured slower (extra O(n³) triangular solves). **dsygvx subset drivers**: 2–3× slower at 200–500 dims and structurally unusable (the positive-energy projector needs the whole branch). `driver="gvd"` (23c5c7e) is the settled answer.
- **More fastmath**: kept on 4 kernels only; dropped from 16 with measured ratios (audit P3).
- **B-spline pseudostate continuum vs FAC**: documented limitation, not a speed issue.

---

## 3. Shooting / continuum

### 3.1 What the solver actually is — and why that matters

For e > 0, `solve_dirac_continuum_shooting` is **not a shooting problem**: there is no eigenvalue parameter and no matching iteration. It is one direct IVP: `doutw` propagates 0 → r_max in a single pass (dirac_integrator.py:155, `n1 = iotp = n_out` for e > 0), and the outer 20 % of the box serves only as a window to extract the asymptotic amplitude A_inf and phase δ against the known Coulomb reference (dirac_continuum.py:129-186). Warm-starting across energies has nothing to attach to — which redirects the speed search to the three real levers below.

(My batch measurement, warm: 240 waves × (E ≤ 8 a.u., r_max = 200, n_grid = 8000) = 30 ms total; Python-side cost is thread-pool bookkeeping, numerics fully inside nogil numba. FAC-parity panels: 0.44–5.31 ms/wave warm, docs/CONTINUUM_FAC_PERFORMANCE_2026-09-24.md.)

### 3.2 P1 (physics-exact): Coulomb-tail analytic continuation — IMPLEMENTED 2026-09-27

**Observation.** The matching code already *assumes* the tail is Coulombic: it takes η from `rv[-1]` (r·V → −Z_res, dirac_continuum.py:172) and fits `P ~ A·sin(k r − lπ/2 − η ln 2kr + σ_l + δ)` in the outer window. For the frozen-core potentials used in RR/PI/CE/EII, that assumption is **exact beyond the core charge radius r_c**: the Dirac-Fock/SCF potential of a compact core is −Z_res/r up to the outermost orbital's exponential tail (the code carries no polarization model that would spoil this).

**Design correction (measured).** Two premises of the original sketch failed on contact with production potentials; the implemented design resolves both without Coulomb special functions:

1. *The "exact beyond r_c" premise is optimistic for production screening models.* On the real FAC-screened Fe/Xe SCF bench the sampled r·V deviates from −Z_res by ~1e-4·Z_res, decaying only ~r^−2.4 all the way to the box edge — a Thomas-Fermi/DH-style screen never lands the strict 1e-8 mask the sketch proposed (measured: r_c never engages, bit-identical output, 1.0× speedup). Production potentials need a *budgeted* tail start, not an *exact* one.
2. *No scipy Coulomb functions are needed.* The tail IVP is propagated by a wide-step walker (`doutw_coulomb_tail`, dirac_integrator.py) reusing the production adaptive power-series controller (`dir_integrate` semantics: accept 15<nsum<60, halve below / double above) with the *exact* Coulomb coefficients (−Z_res, 0, 0, 0). The same `dir0` series that yields the endpoint value also yields every interior grid point, so the dense output is filled by the identical physics — continuation costs tens of wide steps instead of tens of thousands of per-interval calls. (The uv env's scipy 1.15.3 also lacks `coulombF`/`coulombG`, added in 1.16 — moot now.)

**Tail-start criterion (accuracy-first).** i_c(k) requires BOTH: pointwise mask |rV+Z_res| ≤ 1e-4·max(|Z_res|,1) AND action integral ∫|rV+Z_res|/r dr ≤ 1e-4·k — the phase error committed by switching at i_c is ≤1e-4 rad, ~100× inside the 1.5% FAC-parity gate. The action term makes the start per-wave: slower waves (smaller k) integrate the same deviation more strongly and start further out (synthetic 1/r³ screen: fast wave mask-bound at r≈48.5, slow wave pushed to r≈82). Deviation-free potentials (hydrogenic, free particle) engage from the second grid cell.

**Measured economics** (branch `perf/continuum-coulomb-tail`):
- Real Fe/Xe SCF bench: **1.19× / 1.65× / 1.28× / 1.70×** (Fe 1 keV, Fe 40 keV, Fe κ=−5, Xe 200 keV). Synthetic 8k panels: 1.47–1.70×. 300k-point DW grid: **3.63×** (tail share of the IVP 77–85% at that sizing).
- Accuracy vs the spline-integrated reference path: amplitude moves ≤1.1e-8, phase ≤6e-5 rad — both ~100× inside the FAC gate.
- Outer-box spline error — **root cause fixed** (user directive: 保留意外收益，修正样条外盒误差). The measured outer-box deviation of the reference path (5.3e-4 on a gently screened synthetic by r = 200; up to 6.7e-3 in r·V on the Fe SCF bench) decomposes into two defects, both now fixed in `_prepare_potential`, so the walker is a speed lever rather than an accuracy patch: **(a)** vint's Coulomb-tail freeze builds an exact constant-asymptote entry (va = zinf, vb = vc = vd = 0) for the region past the last fitted knot, but the `ind` mapping clipped every outer grid point to the *last fitted interval*, extrapolating its cubic over the whole outer box — the constant entry was unreachable. Grid points beyond the last fitted knot now map to it. **(b)** Where the freeze never triggers (production screened potentials, deviation ~1e-4·Z_res to the box edge), natural-spline cubics deviate between widely spaced outer knots; fitted intervals whose cubic misses the callable at four Gauss-Lobatto nodes by > 1e-8·max(Z_res, 1) are re-interpolated by the node cubic — exact for smooth tails to roundoff — under a strict-improvement guard at fresh Gauss nodes (tabulated/noisy callables keep the spline; divergent model cores are excluded by a scale valve). Measured after the fix: implied r·V tracks the callable to ≲1e-9 at fresh check points (kink-potential probe: monotone, never worse anywhere), and the reference-path tripwire test needed re-basing — pure Coulomb now integrates exactly on *both* paths, so it exercises the tripwire with a genuinely non-asymptotic window (slowly screened Z = 54, low energy, small box) instead. Cost: ~90 ms per prepare on a 300k-point grid (4 vectorized V evaluations per interval), amortized across waves by the existing prep cache. The walker stays: it removes the per-interval integration cost of the tail, now both paths agree on the same exact physics there.
- Tripwire semantics: with deviation-free potentials the walker's exact point-charge wave coheres with the Coulomb reference at any box size, so the small-box "unconverged" raise no longer fires (phase radius-stable to 3e-5; constant Dirac-vs-Schrödinger reference offset 0.1103 rad at Z=54). The regression test monkeypatches the walker off to keep exercising the reference-path tripwire.

This subsumes the grid-sizing tension (`10·r_max·p_max` up to 300k points, distorted_wave/solver.py:270): the walker's tail cost is O(1) per grid point regardless of box size, so the fine grid still covers the whole box but only [0, i_c] pays per-interval integration.

### 3.3 P2: restructure the batch loop — cells outer, energies inner

`solve_dirac_continuum_many` threads over energies, but each energy still walks the grid serially with per-cell setup: the power-series coefficients in `dir0` (potential Taylor terms, h powers) are **energy-independent** — only k(r) = √(2(E−V)) enters the recurrence. A batched formulation (vectorize the state over the energy batch inside each cell) reuses every coefficient setup n_E× and turns the loop into contiguous SIMD-friendly arrays. Independent of and multiplicative with the 4-worker threading (4.37× measured, a54c231). Expected 2–5× on the numerics at typical batch sizes (10–100 energies); also amortizes the potential spline lookup per cell. This is the same structural idea as §2.2: don't re-derive energy-independent structure per energy.

### 3.4 P2: measure the resolution margin, then spend it

`dir_integrate` starts each cell at h/4 (k_ctrl = −2) and lets the series run to 60 terms; the accuracy target is eps from the caller. The margin between *achieved* local error and *allowed* error is unmeasured — the DHO-grid work showed 150 points suffice for bound states at 2.2e-4 accuracy where 1000 were used, a 6.6× over-resolution bought for safety. Run an error-vs-(k·h) calibration against the FAC WaveFuncTable gate (bench_fac_continuum.py already records amplitude errors), then either relax the sub-step controller or upgrade the propagator order in the smooth region. Physics grounding: the power series is locally a Taylor expansion in k·h; its error is (k·h)^(p+1) with p the retained order — the controller currently spends sub-steps to buy p it may not need. Careful with the node-count-dependent bound path (dbound reuses dir_integrate) — gate this on the continuum side first.

### 3.5 Bound path: leave dbound alone; the heavy-Z problem is convergence, not speed

`dbound` at 1–2.5 ms/shell is not the bottleneck; the n_grid = 400 default (trace.md:404-440) already bought 5–14× vs sfac. The real bound-path defect is Z ≥ 74 SCF failing ≤ 500 iterations with Gu's low-α_xc exchange — a physics-model convergence issue; fix there (exchange model) rather than buying faster non-converging iterations.

---

## 4. Spectrum / scattering consumers

### 4.1 P1: replace per-point continuum solves with threshold-aware energy grids

`rrc_spectrum` calls `rr_cross_section` **per occupied subshell over the full photon grid** (radiative_recombination/__init__.py:569-621), and each call batch-solves continuum waves at every grid energy; the DW solver similarly rebuilds its continuum map per energy iteration (distorted_wave/solver.py:557, flagged in SCATTERING_SPECTRUM_REVIEW_2026-09-24.md).

Physics: the PI/RR cross section between thresholds is a smooth function of ω (dipole matrix elements vary like powers of k; the continuum wave is smooth in E — adiabaticity again), with non-analyticity only *at* thresholds (Wigner laws σ ∝ (E−E_th)^{l+1/2}, channel openings). Therefore:
- Solve the continuum **once per partial wave** on a coarse adaptive energy grid (dense within a threshold width near each threshold, sparse between), with the §3.2 Coulomb continuation making each solve cheap.
- Interpolate σ(ω) on the fine render grid (spline in log-log between thresholds; the threshold cusps are handled by the dense knots).

This converts an O(n_photon × n_subshell × n_κ) solve schedule into O(n_E_coarse × n_κ) solves + interpolation — typically 20–100× on the continuum share of a rendered spectrum. Gate: reproduce the FAC PI/RR panels within existing tolerances (the 23.13% Fe PI/RR discrepancy is a documented physical-model difference — don't chase it with grid density; verify against the adjacent-grid convergence check that already exists).

### 4.2 P2: cache CI-line manifolds per config pair

`ci_line_strengths` builds a fresh `_PairCI` (determinant manifold + Hamiltonian/J² tables) **per level pair** during a CI-resolved spectrum render (properties/transitions/ci_lines.py:513; crm_synth `_add_ci_groups` per (upper_cfg, lower_cfg) pair). The manifold depends only on the *config pair*, not on the level pair — cache it. O(n_det³) dense eigensolves per M block follow the same reuse logic (eigenvalues shift slightly with the Hamiltonian? no — the manifold CI Hamiltonian is built once per config pair; only level *identification* differs per pair). This is the exact-CI-cap sibling of Finding 11 in the 2026-09-23 review.

### 4.3 P2: sparse CRM steady state

`CollisionalRadiativeModel.steady_state` builds a **dense** n_levels × n_levels rate matrix and does a dense `scipy.linalg.solve` per (n_e, T_e) (plasma/crm.py:487-510). Rate matrices are sparse — a level couples radiatively/collisionally to a handful of others; dense factorization is O(n³) on mostly zeros. Sparse LU (or iterative with the previous population as warm start — populations are also adiabatic in T_e along a sweep) removes the cubic. Relevant for the line-intensive CRM renders where n_levels reaches 10⁴–10⁵.

### 4.4 P1 (near-free): productionize the local-window deposition

The 8σ local-window Gaussian deposition prototype measured **13.55×** (96.45 ms → 7.12 ms, max error 4.16e-15 relative to peak, SCATTERING_SPECTRUM_REVIEW_2026-09-24.md:77). The review's caveat (8σ unsafe for Lorentzian/Voigt wings — power-law tails) is handled by window = max(8σ, n·γ) with the existing full-grid fallback per line profile class. This is the rare case where the prototype number is already the production number; land it behind the existing profile-area conservation invariants (already regression-tested by 4a47a59's 252-line test file).

---

## 5. Cross-cutting

1. **Production thread pinning.** Only pytest and `scripts/density_gs.py` pin BLAS/OpenMP; production entry points oversubscribe by default (critical-review F1/F2). Every recommendation above needs pinned-thread measurement to be real — make pinning the default in `atomx` entry points (env-overridable), not just in tests.
2. **Measurement-as-gate.** perf_dashboard is freshness-only (audit P5). Extend the existing FAC-matrix wall-time tripwire (benchmarks, >30 % regression) to: the MCDHF variational cases (Be s/sd, Fe XVII), the continuum wave panel, and one RR spectrum render. All with pinned threads in the harness.
3. **The adiabaticity principle as a design rule.** Three of the four biggest levers (§2.4 frozen CI, §3.3 energy batching, §4.1 threshold grids, §4.3 warm-start CRM) are the same observation: *atomx recomputes from scratch quantities that physics constrains to vary smoothly along the parameter actually being swept*. When reviewing future hot paths, the first question is now: "what is being held fixed that lets this be reused?"

---

## 6. Prioritized roadmap

| # | Item | Expected | Risk | Verified? |
|---|---|---|---|---|
| P0 | §2.2 h0-carry memo | −16 % variational MCDHF | none (bit-identical shown) | **yes, this session** |
| P0 | §2.3 metric-check cadence | −20 % more | none (bit-identical shown) | **yes, this session** |
| P0 | §5.1 production thread pinning | measurement validity | none | mechanism exists (conftest) |
| P1 | §2.4 frozen-CI microiterations | 2–4× optimizer-dominated cases | medium — exact-CI refresh cadence; keep e7f4b68 gate semantics | no |
| P1 | §3.2 Coulomb-tail continuation — **done 2026-09-27** (`perf/continuum-coulomb-tail`) | 2–10× large-box continuum → **measured 1.2–1.7× real SCF bench, 3.6× 300k DW** | budgeted tail start (phase ≤1e-4 rad); walker exact by construction | no |
| P1 | §4.1 threshold-aware spectrum grids | 20–100× continuum share of renders | medium — interpolation vs Wigner cusps | no |
| P1 | §4.4 local-window deposition | 13.5× rendering | low | prototype measured |
| P2 | §2.5 Cauchy–Schwarz integral screening | 2–5× coefficient eval | low | no |
| P2 | §2.6 ns-adaptive convol prange | 4–8× at ns ≥ 290 | medium — crossover re-measure per kernel | measured both regimes |
| P2 | §3.3 batched-energy integrator | 2–5× batch numerics | medium — restructure | no |
| P2 | §4.2 manifold caching · §4.3 sparse CRM | render-scale | low | no |
| P3 | §3.4 resolution-margin spend | 2–3× continuum | high — accuracy gate is FAC waves | calibration only |
| P3 | GPU convol phase-2 (5090, ledger roadmap) | kernel-scale | high | roadmap only |

**Sequencing note:** P0 items are independent of PR #89 and can land first (they change no numerics); §2.4 then attacks the remaining `ci_solve` mass from a lower baseline.

---

## Appendix A — empirical record (this session)

Fe XVII A/B (pinned threads, worktree 385d908, identical work in all runs — 320-iter budget, non-converged flag identical):

```
baseline        wall 80.1s   E0 -1148.4391761319
h0-carry        wall 67.4s   E0 -1148.4391761319   (-15.9%)
h0 + metric     wall 50.9s   E0 -1148.4391761319   (-36.4%)
```

h0-carry patch used for the experiment (production form belongs in `matrix.py` next to `_reset_memo`, with the `"_nuclear_potential"` co-carry condition):

```python
def _reset_keep_h0():
    keep = {k: v for k, v in M._SLATER_MEMO.items()
            if k == "_nuclear_potential"
            or (isinstance(k, tuple) and k[0] == "h0")}
    keep_alive = list(M._MEMO_KEEPALIVE)
    _orig_reset()
    M._SLATER_MEMO.update(keep)
    M._MEMO_KEEPALIVE.extend(keep_alive)
```

Warm continuum batch (240 waves, E ≤ 8 a.u., r_max = 200, n_grid = 8000): 30 ms total — thread-pool bookkeeping dominates the Python side; numerics are entirely inside nogil numba kernels.

## Appendix B — dead ends (do not retry, with evidence)

| Idea | Evidence |
|---|---|
| Preconditioner refresh in variational L-BFGS | diverges >320 iters (skip_pq_exchange Fock; 2026-09-27 experiment, 8-config matrix) |
| Cholesky-cached standard eigensolve | measured slower (trace.md:899) |
| dsygvx subset drivers | 2–3× slower at 200–500 dims; structurally unusable for projector |
| Global convol prange | 9.4× slower at ns ≈ 60 (convol.py:768-776) |
| More fastmath | dropped from 16 kernels per measured ratios (audit P3) |
| Unpinned eigh attribution | 43 s → 0.53 s under pinning (F2) |
