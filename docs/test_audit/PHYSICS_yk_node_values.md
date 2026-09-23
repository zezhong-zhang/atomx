# Slater `R^k` / `Y^k` accuracy defect — root-cause diagnosis

> **Status: fixed 2026-09-23.**  The fix is the interpolatory
> partial-interval rule in `src/atomx/numerics/bspline/evaluate.py`
> (`_yk_partial_matrices` + rewritten `_compute_yk_flat`), i.e. fix
> direction 1 below, but computed from the node values alone (degree
> `n_gauss-1` interpolant on the straddled cell) rather than from the exact
> B-spline density — the production kernel only receives node values.  On
> the analytic cases of §1 the corrected rule measures at machine precision
> (2.2e-16…6.3e-12), five orders of magnitude *better* than the ≤1.2e-7
> headroom demonstrated here with exact partials; on the physical Ne case
> (§6) it reproduces the exact-partial arbiter to all its digits.  Gates
> and the full before/after table: `CHANGELOG.md` (Unreleased → Fixed —
> numerics) and `tests/test_slater_integrator.py`.  Note: the rule-B
> transcription in the scratch `exp1_rule.py` (`halfv * (g @ Th.T)`) has a
> scaling error and measures 7–27 % wrong — do not copy it; the correct
> normalisation is `mo = Theta / wi` (see `_yk_partial_matrices`).

Scope: read-only diagnosis of the S1 follow-up ("the Slater kernel is externally
accurate to only ~3e-5 … systematic in the `Y^k` construction, not peak
resolution"). No file under `src/atomx/`, `tests/`, `docs/` was modified.
Scratch scripts: `/tmp/atomx_n1/yk_*.py`; arbiters: mpmath (40 digits) and F.
Salvat's own `SLATER` compiled from `~/code/ASAP/radial/DHFS/DHFS.f:1191`
(`/tmp/atomx_n1/salvat/`, gfortran, driver `dri.f`).

## 0. Headline

The ~3e-5…6e-4 loss is **not** a branch/index/weight bug and **not** a
resolution wall of the node set. It is one specific, removable approximation:

> `src/atomx/numerics/bspline/evaluate.py:104-117` (`_compute_yk_flat`) builds
> the truncated cumulative integrals `I_<(r_i) = ∫_0^{r_i} ρr'^k dr'` and
> `I_>(r_i) = ∫_{r_i}^∞ ρ/r'^{k+1} dr'` as *"full Gauss weight for every node
> strictly inside the limit, plus **half** the Gauss weight of the node at the
> limit"*. That half-weight is a first-order estimator of the partial knot
> interval the node straddles; the B-spline density on that interval is exactly
> a degree-`2ks-2 = 16` polynomial sampled at only `n_gauss = ks = 9` nodes, so
> the partial-interval integral is **not determined by the data the rule uses**.
> The resulting bias has the same sign at every node (systematic), is
> `O((dr/r)²)`, and is the whole of the observed error.

Everything else is exonerated by measurement (weights exact to 1e-16, outer
contraction exact given exact `Y`, pure `h²` convergence, no `k` anomaly).
The error is provably removable **on the same grid, same nodes, same weights**
(§7: 3 orders of magnitude recovered), and the shooting/SLAG6 engine does not
have it at all (§5).

## 1. Reproduced numbers (S1's own grid)

Grid = the `ne_grid` fixture of `tests/test_slater_integrator.py`, rebuilt
without the SCF: `make_knots(60, 9, r_min=0.05/10=0.005, r_max=40)` →
`prepare_bspline_data(t, 60, 9, kinetic_balance=False)`;
**nv = 52 knot intervals × n_gauss = 9 = 468 nodes**, `dr/r = 0.17622`
(log-step), `r ∈ [7.96e-5, 39.90]`. mpmath reference = 40-digit adaptive
quadrature of the exact double integral (splitting at `r`).

| case | atomx (`slater_rk`, poisson) | exact | relative error |
|---|---|---|---|
| hydrogenic 1s, Z=1 (`P=2Z^{3/2} r e^{-Zr}`, ρ=P²), k=0 | 0.6250177251 | 5Z/8 = 0.625 | **2.8360e-5** |
| same, Z=10 | 6.2501772709 | 6.25 | **2.8363e-5** |
| same, Z=26 | 16.2504657722 | 16.25 | **2.8663e-5** |
| ρ = e^{-r}, k=0 | 1.3864839738 | ln 4 = 1.38629436112 | **1.3678e-4** |

* mpmath reproduces `5Z/8` to 0.0e+00 relative at all three Z → the closed form
  is right, and the mpmath arbiter is exact for it.
* The exact value for ρ = e^{-r} **is** `ln 4`: `R^0 = ∫_0^∞(e^{-r}-e^{-2r})/r dr
  + ∫_0^∞ e^{-r}E_1(r)dr = ln2 + ln2`. Verified numerically by mpmath.
* `∫ρ dr` on this grid = `1 - 1.1e-16` (Z = 1, 10, 26) and `1 - 1.1e-16` for
  e^{-r} — the quadrature is exact for the *density*, as S1 said.
* `slater_rk(..., "poisson")` vs `slater_rk(..., "direct")`: `|Δ| ≤ 2.3e-13`
  on every one of these cases.
* These reproduce S1's 2.84e-5 and 1.37e-4 exactly.

## 2. The two kernels do **not** share a construction — they are one rule

`slater_rk_direct` (`numerics/bspline/slater.py:47-72`) sums
`Σ_i Σ_j w_i w_j ρ13_i ρ24_j K(r_i,r_j)`. The Poisson/`compute_yk` path sums
`Σ_i w_i ρ13_i Y_i` with `Y_i = r_i^{-(k+1)}[Σ_{j<i} w_j ρ24_j r_j^k + ½w_iρ24_ir_i^k]
+ r_i^k[Σ_{j>i} w_j ρ24_j r_j^{-(k+1)} + ½w_iρ24_ir_i^{-(k+1)}]`.
Expanding, the `j≠i` terms reproduce all off-diagonal pairs and the `j=i` terms
give `w_i²ρ13_iρ24_i/r_i` — exactly the diagonal of the direct sum, since
`K(r_i,r_i) = 1/r_i`. **The two "interchangeable kernels" are algebraically the
same quadrature rule**, which is why they agree to 1e-13 and why S1's
`rel=1e-10` cross-check (`tests/test_slater_integrator.py:56-66`) can never see
this error: it is not an independent path.

The genuinely independent paths are the SLAG6 shooting kernel and Salvat's code
(§5) — and those do **not** show the defect.

## 3. Decisive experiment 1 — grid refinement (error vs resolution)

**Family A** — add knots, `ks=9, n_gauss=9, r_min=0.005` fixed, `ns = 60…1040`
(nv = 52…1032, log-step 0.176 → 0.0087):

| ns | 60 | 80 | 120 | 160 | 240 | 360 | 520 | 760 | 1040 |
|---|---|---|---|---|---|---|---|---|---|
| hydro k=0 | 2.836e-5 | 1.462e-5 | 5.983e-6 | 3.234e-6 | 1.383e-6 | 6.010e-7 | 2.852e-7 | 1.338e-7 | 7.25e-8 |
| ρ=e^{-r} k=0 | 1.368e-4 | 1.286e-4 | 1.235e-4 | 1.218e-4 | 1.207e-4 | 1.203e-4 | 1.201e-4 | 1.200e-4 | **1.199e-4** |

Observed order (h ≡ log-step): hydro **p = 2.00, 2.00, 2.00, 2.00, 1.99, 1.98,
1.97, 1.93**; ρ=e^{-r} **p = 0.19 → 0.00, plateau at 1.1994e-4**.

**Family B** — fixed knots (`ns=60`), refine `n_gauss` = 9, 11, 13, 17, 21, 25,
33: hydro p = 1.91→1.97; e^{-r} p = 1.47→1.66 (errors 1.368e-4 → 1.764e-5).

**Family C** — self-similar refinement (halve the log-step; `r_min` shrinks with
it so *every* interval, including the first, is refined):

| log-step | 0.176 | 0.0881 | 0.0441 | 0.0220 | 0.0110 | 0.00551 |
|---|---|---|---|---|---|---|
| hydro k=0 | 2.836e-5 | 7.082e-6 | 1.769e-6 | 4.425e-7 | 1.105e-7 | 2.765e-8 |
| e^{-r} k=0 | 1.705e-5 | 4.257e-6 | 1.064e-6 | 2.660e-7 | 6.647e-8 | 1.665e-8 |
| e^{-r} k=1 | 1.155e-4 | 2.885e-5 | 7.208e-6 | 1.803e-6 | 4.504e-7 | 1.128e-7 |

**p = 2.000 for every case** (ratio 4.00 per halving, to three digits).
Verdict: a **genuine second-order quadrature error**, not a plateau, when the
grid is refined in the way that refines the whole grid. The Family-A plateau for
ρ(0)≠0 is an artefact of the *fixed first knot interval* `[0, r_min]`
(`basis.py:242-247` maps the first span `[t[ks-1], t[ks]] = [0, 0.05/Z]`, which
no addition of knots subdivides — log-step refinement there goes as `(he-1)`).

## 4. Decisive experiment 2 — k-dependence, and where the error lives

Relative error vs k on the production grid (mpmath references):

| case | k=0 | k=1 | k=2 | k=3 | ratio |
|---|---|---|---|---|---|
| hydro 1s | 2.836e-5 | 1.418e-4 | 3.402e-4 | 6.287e-4 | 1 : 5.00 : 12.0 : 22.2 |
| ρ=e^{-r} | 1.368e-4 | 1.032e-3 | 2.407e-3 | 4.167e-3 | 1 : 7.5 : 17.6 : 30.5 |
| physical Ne 1s–1s (vs §6 ref) | 2.804e-5 | 1.410e-4 | 3.390e-4 | 6.272e-4 | 1 : 5.03 : 12.1 : 22.4 |

* The hydrogenic and the *physical* B-spline densities share the ratio
  `1 : 5 : 12 : 22 ≈ (k+1)(3k+2)/2` — a smooth quadratic in `k`, the signature
  of the two-term kernel (`ρr^k` inner piece and `ρr^{-(k+1)}` outer piece)
  entering a quadrature error term. An index slip in `r^k`/`r^{-(k+1)}` would
  give an O(h)≈1e-2…1e-1 error, not 1e-5…6e-4, and would not refine as `h²`.
* Under Family A every `k` refines at the *same* rate (ratios 8.77 and 11.34 for
  k=0,1,2,3 of the hydrogenic case) → the error is one mechanism with a
  k-dependent constant, not a k-specific defect. There is no `k=0` special case
  anywhere in `_compute_yk_flat`/`slater_rk_direct`/`_yk_kernel_rho`.

**Localisation (which stage of the algorithm).** With the *exact* `Y^k` values
evaluated at atomx's own nodes and atomx's own weights:

* hydrogenic Z=10, k=0,1,2 at ns = 60/160/520:
  `Σ_i w_i ρ_i Y^k_exact(r_i) = 6.2500000000` — i.e. **the weights, the
  contraction and the grid all reproduce `5Z/8` to <1e-10**; 100 % of the
  2.84e-5 sits in the `Y^k` *node values*, spread over the density-peak region
  (max |ΔY| at r = 0.092 for k=0; first interval contributes 3.2e-9).
* ρ = e^{-r}, k=0: atomx 1.3864839738; `ΣwρY_exact = 1.3862593100` (its own
  error −2.5e-5, the log-singular contraction in the first interval);
  `Σwρ(Y_atomx − Y_exact) = +2.247e-4 = +162 %` of the total error.
  Per interval: `[0, 0.005]` alone contributes +2.013e-4; `Y(r_0)` is 5.25 % high
  (10.3790 vs 9.8613).

So the weights/contraction/first-interval bookkeeping are clean; the node values
of `Y^k` are the defect — and the rule that produces them is the half-straddling
weight of §0.

## 5. Do the other kernels share it? No

Same test integrals, independent schemes / grids:

| kernel | grid | hydro k=0 | ρ=e^{-r} k=0 |
|---|---|---|---|
| `compute_yk` + contract (`slater_rk` poisson) | bsp Ne, 468 pts | 2.836e-5 | 1.368e-4 |
| `slater_rk_direct` (same rule, §2) | bsp Ne | 2.836e-5 (Δ=2.3e-13) | 1.368e-4 |
| `_slater_integral_slag6` (`shooting/solver.py:327`, SLAG6) | `sgrid(75,1e-7,0.5,1000)` | **2.28e-12** | 4.37e-7 |
| same, n_grid = 2000 / 4000 | | 2.51e-14 / **1.11e-16** | 4.37e-7 (plateau) |
| Salvat `SLATER` (Hartree ODE + spline, external) | same sgrid, 1000/2000/4000 | 5.62e-10 / 2.51e-11 / 1.18e-12 | 4.40e-7 |
| `yk_on_grid` + trapezoid dot (`numerics/quadrature.py`) | `make_fine_grid(30,k_max,10)`, k_max=1/4/20 | 1.630e-4 / 1.630e-4 / 1.499e-4 | 1.254e-4 / 9.94e-5 / 2.63e-5 |

* The SLAG6 kernel (used by the shooting engine and by
  `methods/mcdhf/matrix.py` on `GridCIWorkspace`) is **machine precision** on
  the hydrogenic case and 300× better on the smooth case — it does **not** share
  the defect. atomx's SLAG6 is a faithful port of Salvat's `SLATER` (same
  numbers to its own accuracy, better at large `n_grid`); both plateau at
  ~4.4e-7 on ρ=e^{-r} because that density is 1 at r=0 (the shooting grid drops
  the r=0 node), which is an artefact of the test density, not of the kernel.
* **Pointwise `Y^k` vs Salvat** on atomx's *own* Ne nodes (Salvat `IWR=1`
  `slater.dat` dump, `difr` = trapezoid measure): Salvat's `Y(r)/r` agrees with
  the exact `Y^0(r)` to −5e-8 … −2e-8 for r ≥ 0.006, while atomx's `Y` is
  +4.7e-6 (r=0.006), +4.6e-6 (r=0.0245), +4.0e-5 (r=0.21). The node set
  therefore carries enough information for a ~1e-8 `Y`; atomx's rule throws it
  away. (Salvat's own total on this clustered grid is biased ~9.3e-5 by the
  trapezoid `difr` fed to it and its Y is reset for r<1e-4, so it is used here
  only as the *pointwise* arbiter, not as a total-value arbiter.)
* `numerics/quadrature.py`'s trapezoid family (`yk_on_grid`, `_yk_kernel_rho`,
  `_slater_rk_kernel`) is the same class of rule with a different constant; on
  its own fine grid it is ~5× worse than the B-spline path for the hydrogenic
  case. Note that on `make_fine_grid` `∫ρ dr` is **not** machine-exact
  (1.00007, trapezoid at dr/r = 1/48) — the 1e-16 claim holds only on the
  B-spline Gauss grid, where the 9-node rule is exact for the degree-16
  polynomial density.

## 6. Physical densities (the case that matters) and an exact reference

Ne DHF orbitals (`DFSolver(Z=10, ns=60, ks=9, exchange="hf")`, the test's own
fixture). The orbital is exactly evaluable at *any* r via
`scipy.interpolate.BSpline` on atomx's own `t`/coefficients (verified against
`eval_orbital_on_grid` on the production nodes: max |Δ| = 1.3e-15). Reference =
the same physical density evaluated at 60 001 self-similar nodes
(h/r = 2.9e-4) and integrated with the same cumulative scheme — validated on the
analytic hydrogenic case to 3.3e-8…1.2e-7 (its own residual).

| pair | k | atomx (production grid) | fine-grid reference | relative error |
|---|---|---|---|---|
| 1s–1s | 0 | 5.9681691575 | 5.9680018233 | 2.804e-5 |
| 1s–1s | 1 | 3.5645377959 | 3.5640353993 | 1.410e-4 |
| 1s–1s | 2 | 2.4713669219 | 2.4705295352 | 3.390e-4 |
| 1s–1s | 3 | 1.8702258518 | 1.8690535474 | 6.272e-4 |
| 1s–2p | 0 | 0.2343109754 | 0.2343036953 | 3.107e-5 |
| 1s–2p | 1 | 0.1469098282 | 0.1468879727 | 1.488e-4 |
| 1s–2p | 2 | 0.1041488546 | 0.1041124265 | 3.499e-4 |
| 1s–2p | 3 | 0.0797212412 | 0.0796702436 | 6.401e-4 |

So the defect is present, at the same magnitude, on **real production
densities** — not just on analytic test input. (The same test grid's `2p` is
used; `∫ρ_1s dr = 0.99877` because the DHF 1s is not normalised to 1 in this
fixture, which is irrelevant to a relative error.)

The density is exactly a degree-16 polynomial inside each knot interval
(Chebyshev fit of Ne 1s² on `[0.8287, 0.9883]`: residual 2.0e-18 at degree 15,
1.3e-18 at 16, 9.8e-19 at 17) — which is *why* `∫ρ dr` is exact, and *why* 9
node values cannot determine the partial-interval integrals the half-weight rule
needs.

## 7. The defect is removable on the same grid (measured headroom)

Keeping the production knots, all 468 nodes and all weights, and replacing only
the half-straddling-weight partial pieces with **exact** partial-interval
integrals (12-node Gauss on `[a, r_i]` for the inner piece — exact for degree
≤ 23 ≥ 16+k — and adaptive quad for the outer piece, both on the exact B-spline
density):

| pair | k | atomx Y^k path | exact-partial Y, same nodes/weights | error after |
|---|---|---|---|---|
| 1s–1s | 0 | 5.9681691575 | 5.9680016294 | 3.2e-8 |
| 1s–1s | 1 | 3.5645377959 | 3.5640352233 | 4.9e-8 |
| 1s–1s | 2 | 2.4713669219 | 2.4705293400 | 7.9e-8 |
| 1s–1s | 3 | 1.8702258518 | 1.8690533191 | 1.2e-7 |
| 1s–2p | 0…3 | — | — | 3.3e-8 … 1.2e-7 |

The residual equals the fine-grid reference's own error (3.25e-8 … 1.22e-7 on
the analytic case) → the fixed values are exact to the reference's precision.
**3 orders of magnitude recovered on the same grid** by changing only the
partial-interval treatment.

## 8. Hypothesis scoreboard

1. **Cumulative differencing / one rule reused** — *partly the mechanism*: the
   rule is a cumulative (differencing) scheme and it does reuse one Gauss rule
   for the truncated limits; but there is **no cancellation loss** (both pieces
   are positive here, the sum is well conditioned; the error is a systematic
   quadrature bias, not roundoff). Reusing the same rule truncated at the node
   is wrong *only* through the ½-weight partial-interval estimate.
2. **Inner/outer split placed at a grid node rather than at r → one-node
   inconsistency** — **eliminated**. The split *is* at the node by construction,
   and the `direct` kernel, which has no split at all, gives the same number to
   2.3e-13; a one-node inconsistency would be an O(h) plateau and does not refine
   (measured p = 2.000 for every case in Family C).
3. **`r^k`/`r^{-(k+1)}` at the wrong index** — **eliminated**. Errors are
   1e-5…6e-4, refinable as `h²`; the k-scaling is the smooth
   `(k+1)(3k+2)/2 = 1:5:12:22` for two unrelated densities; `slater_rk_direct`'s
   independently written `ri**k/rj**(k+1)` branch agrees to 1e-13.
4. **Weights not scaled by the Jacobian / first or last interval double-counted
   or dropped** — **eliminated**. `setup_gauss_points` (`basis.py:242-247`)
   stores `half*wi` with *no* `r²` because the B-spline radial functions are
   normalised as `∫(P²+Q²)dr = 1` — correct here; `Σwρ = 1 ± 1.1e-16` for all
   three Z and for the physical orbitals; `Σ w ρ Y_exact` reproduces `5Z/8` to
   <1e-10.
5. **`k=0` special case** — **eliminated**. No k-branch exists in
   `_compute_yk_flat`, `slater_rk_direct`, `slater_rk_poisson`,
   `_yk_kernel_rho`; k=0…3 all show the same mechanism and the same `h²`.

## 9. Root cause, file:line

**Primary (the offending code):**
`src/atomx/numerics/bspline/evaluate.py:104-117` —
```python
i_out = np.empty(n); s = 0.0
for i in range(n):
    diag = w_flat[i] * rho_flat[i] * rk[i]
    s += diag
    i_out[i] = s - 0.5 * diag          # <- half of the straddling node's weight
i_in = np.empty(n); s = 0.0
for i in range(n - 1, -1, -1):
    diag = w_flat[i] * rho_flat[i] * rkm[i]
    s += diag
    i_in[i] = s - 0.5 * diag           # <- same, for the outer tail
```
This fixes the earlier O(h) diagonal double-count (the comment at
`evaluate.py:96-103` documents the +1.75 % it removed) but leaves a systematic
`O((dr/r)²)` bias. It is the *only* place the R^k value is formed for the
Galerkin path; the twin consumers `slater_rk_poisson` /
`slater_rk_direct` (`numerics/bspline/slater.py`) and
`slater_rk` inherit it.

**Contributing grid facts (not bugs, but they set the constant and the
plateau):**
* `src/atomx/numerics/bspline/basis.py:219-249` — `n_gauss = ks = 9` by default:
  9 samples of a degree-16 polynomial per interval. Enough for the *total*
  (`∫ρ dr` exact) — not enough to determine partial-interval integrals.
* `src/atomx/numerics/bspline/basis.py:242-247` — the first knot span is
  `[0, r_min]` with `r_min = 0.05/Z` (`methods/dirac_fock.py:618`); it is never
  subdivided by adding knots, which is why the ρ(0)≠0 test case plateaus at
  1.1994e-4 in Family A while everything else refines as `h²`.
* The same class of rule in `numerics/quadrature.py` (`_yk_kernel_rho`,
  `yk_on_grid`, `_slater_rk_kernel`) — trapezoid variant, no Jacobian bug, but
  the same truncated-cumulative approximation, ~5× worse on the hydrogenic case
  on its own fine grid.

**Why the test suite could not see it:**
`tests/test_slater_integrator.py:56-66` compares `poisson` against `direct`,
which §2 shows is the *same* quadrature rule (measured Δ = 2e-14) — a vacuous
cross-check; and the closed-form gate at `:78-105` is `rel=1e-4`, 3.5× the
measured 2.84e-5, so it passes by construction.

## 10. Fix directions (not applied — read-only task), with measured support

1. **Exact partial-interval cumulative** (best): the B-spline density is
   exactly a piecewise polynomial of degree ≤ `2ks-2`; compute the straddling
   partial integrals from the B-spline coefficients (or by Gauss on the partial
   interval) instead of ½·w_i. Measured: 2.8e-5…6.3e-4 → ≤1.2e-7 on the same
   grid (§7); expected ~1e-12 with an exact polynomial evaluation.
2. **Grid-only alternative**: `n_gauss ≥ 2ks-1 = 17` makes the per-interval
   polynomial (hence the partial-interval integrals) exactly determined by the
   node values. Caveat: this changes every grid in the package and would need
   re-pinning of all snapshot values; the §7 experiment shows the error is not
   a shortage of *interior* resolution.
3. **Route through the shooting kernel** for the consumers that can use it:
   `_slater_integral_slag6` is already machine precision on the same physical
   problem.
4. Any fix must add a **non-shared** reference gate (mpmath closed form /
   fine-grid refinement), because the existing poisson-vs-direct gate cannot
   detect this.

## 11. Files / commands

* Repro: `/tmp/atomx_n1/yk_repro.py`, refinement: `yk_refine.py`,
  localisation: `yk_localise.py`, `yk_hydro_local.py`, k-table: `yk_kdep.py`,
  other kernels: `yk_other_kernels.py`, Salvat: `yk_salvat.py` +
  `/tmp/atomx_n1/salvat/*.f`, physical + exact reference: `yk_physical.py`,
  `yk_physical_ref2.py`, fix demo: `yk_fixdemo.py`, sanity: `yk_sanity.py`,
  high-Z: `yk_highz.py`.
* All runs: `MPLCONFIGDIR=/tmp/atomx_n1/mpl /Users/zhangzz/code/ASAP/atomx/.venv/bin/python <script>`.
