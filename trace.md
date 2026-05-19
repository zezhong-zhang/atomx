# atomx trace log

### EARS — Progress (2026-05-19 12:30)
<!-- concepts: mcdhf-csfs, fe17-3c3d, j-resolved-coupling -->
- Started HYP-182: Fe XVII 3C/3D root-cause + fix (the gate HYP-175 xfailed).
- Issue: `Atom(Z=26, charge=16, j_resolved_csfs=True).solve()` returns 27 states
  / 111 E1 transitions, but only 5 from State_0 (the 2p⁶ ¹S₀ ground), and
  none in the 14.5–15.5 Å region where 3C/3D live.
- Suspect surface: `enumerate_j_resolved_csfs` (`src/atomx/methods/mcdhf/config.py`)
  generates J-resolved CSF placeholders by m-scheme counting, but the
  `_compute_mcdhf_transitions` path filters by `multipole_allows_state_transition`
  on the dominant CSF *only* — that's fine for J/parity, but the underlying
  `one_body_csf_coupling` then only checks occupation diffs, not whether the
  bra/ket parentages actually couple. Likely the 7 J=1 odd-parity 2p⁵3l
  CSFs that share the same configuration occupation get the wrong angular
  coupling factor (population_factor only) → only a handful end up nonzero.
- Plan: build a minimal repro, dump per-CSF angular tables, then add the
  missing parentage/spin-coupling weights so we span the full J=1⁻ manifold.

### EARS — Progress (2026-05-19 12:57)
<!-- concepts: ci-hamiltonian, slater-condon-2diff, angular-sign -->
- Extended fix to `_two_diff_element` (analogous bug: hardcoded `k=0`
  direct integral with no angular check on the orbital-pair κ's).  The
  2-diff direct now requires `κ_a = κ_c AND κ_b = κ_d`; exchange
  intersects (a,d) and (c,b) k-ranges and uses geometric-mean angular.
- After both fixes the 5 J=1 odd-parity states cluster as expected
  (3s: 716/952 eV; 3d: 761/963/1161 eV) — the 3C analog state[6] sits
  at 16.29 Å with f=2.83 (in band [2.2, 3.0]).  But wavelength is just
  outside the Bernitt 14.5–15.5 Å window, and state[24] still picks up
  spurious f=1.42 at 10.68 Å, making the f-ratio = 2.0.
- Adding a derived sign factor (-1)^{(tj_a+tj_c)/2 - 1} for the
  different-κ exchange angular product — see whether it brings the
  3 J=1 3d states closer in energy and pushes the spurious f down.

### EARS — Progress (2026-05-19 12:45)
<!-- concepts: ci-hamiltonian, slater-condon-1diff, jj-angular -->
- Found two real bugs in `_one_diff_element` (`src/atomx/methods/mcdhf/matrix.py`):
  1. Direct integral hardcodes `R^0(ac, bc)` with NO angular check, so
     3s↔3d / 2p_{1/2}↔2p_{3/2} promotions get a spurious huge off-diagonal
     coupling that scrambles the J=1 odd-parity manifold (the 5 J=1 states
     end up at 635, 676, 896, 1017, 1332 eV — should be 730–840 eV).
  2. `sqrt_factor = sqrt(|Δq_a|·|Δq_b|) = 1` for every 1p1h promotion,
     missing the standard jj weight `sqrt(q_source^{ket} · h_dest^{ket})`
     which is 2 for 2p₁/₂↔2p₃/₂ swaps and sqrt(6) for 3d₃/₂↔3d₅/₂.
- Applied fix: zero direct when κ_a ≠ κ_b, use proper sqrt factor, and
  geometric-mean exchange angular √(λ^k(a,c) λ^k(c,b)) (collapses to
  λ^k(a,c) for the same-κ case = existing behaviour).
- Next: rerun repro, check 3C/3D ratio.

## 2026-04-16

<!-- concepts: atomic-structure, dbsr-hf, hdf5-schema -->
- Added `scripts/generate_density_gs.py` — CLI tool that runs B-spline
  Dirac–Hartree–Fock (DBSR_HF-style jj-averaged `term=AV`) with Breit and
  QED via `Atom(Z, breit=True, qed=True).solve()` and writes per-element
  HDF5 files matching `/Users/zhangzz/code/coredensity/database/density_gs/Ag.h5`.
- Took a wrong turn defaulting to `correlation="sd"` (true MCDHF with CSF
  excitations); this was orders of magnitude slower than DBSR_HF and not
  needed for ground-state densities. Switched default to single-config
  DHF (matches DBSR_HF); `--correlation sd` still available as opt-in.
- Ag schema gotchas discovered by direct inspection:
    * root has *no* attrs; metadata lives in `/metadata` group.
    * `small_component_ratio` = integrated small-component density
      (nq·∫Q²dr), *not* the unit-normalized ∫Q²/∫(P²+Q²).
    * `speed` = sqrt(2·⟨T_NR⟩)/c with ⟨T_NR⟩ = -½⟨P|d²/dr²|P⟩ +
      ½l(l+1)⟨P|r⁻²|P⟩ (non-rel kinetic of the large component only).
      Matches reference <0.5% on all subshells.
- Default `--max-iter=300` bumped from 80 — Cu (3d10 4s1) needs ~122.

### EARS — Progress (2026-04-16 23:37)
<!-- concepts: cli-design, hdf5-schema, periodic-table -->
- Extended `generate_density_gs.py` CLI so it can drive whole-periodic-table
  runs. Positional tokens now accept inclusive ranges in three forms: `1-20`,
  `1..20`, `1:20`, and symbol ranges like `H-Ne` or `La-Lu`. Descending
  ranges (`10-6`) are allowed. All forms can be freely mixed with single
  tokens: `python scripts/generate_density_gs.py Ag Cu 26 50-52 H-He`.
- Refactored `parse_elements` to delegate single-token resolution to a
  helper `_resolve_single`, so the range path and scalar path share
  validation (Z ∈ 1..118, symbol ↔ Z via `atomx.constants.atomic_numbers`).
- Bad inputs still raise cleanly: `'0-5'` → `Z=0 out of range`,
  `'Xx'` → `Unknown element token`, etc.
- Moved default output dir to `./output/` (relative to cwd) to make batch
  runs self-contained; `--out` still overrides.

### EARS — Debug (2026-04-16 23:58)
<!-- concepts: bspline-eval, grid-sampling, phase-convention -->
User reported generated waves/densities look "wavy not smooth" compared
to DBSR_HF and the pre-refactor output. Root-caused two independent issues:

1. **Gauss-Legendre node sampling.** `result.waves.grid` is
   `bsp_data["gr"].ravel()` — the raveled Gauss quadrature nodes inside
   each B-spline interval. These cluster at knot boundaries (dr drops to
   ~1e-7 at each knot, widens into the interval by the Legendre rule).
   Plotted with any line-segment connection this looks zigzag even though
   the underlying wavefunction is smooth. Fix: cubic-spline interpolate
   (P, Q) from the dense Gauss grid onto a geometric log grid
   (r = geomspace(r_min, r_cut, 400)). r_cut from the existing
   density_large<1e-17 tail threshold; r_min clamped at 1e-8 bohr.

2. **Sign convention.** Atomx's SCF returns orbitals with an arbitrary
   overall phase (±1). For 6/15 Ag subshells this was opposite to the
   reference. Fix: enforce P(r→0⁺) > 0 by probing `P_g` at the first
   sample above 1e-6·|P|_max and flipping both P and Q if negative.
   (Density is sign-invariant; only wave_large/wave_small needed this.)

Result: body L2 error vs reference Ag.h5 is 3e-4 (1s) to 2e-2 (5s).
Remaining deviation is physics (we include QED+Breit, reference doesn't),
not numerics.

### EARS — Session Start (2026-04-19 23:14)
<!-- concepts: repo-indexing, project-structure -->
- Task: Run /sc:index-repo to generate PROJECT_INDEX.md / PROJECT_INDEX.json for the atomx repo.
- Why: User wants a compact index (≤3 KB) so future sessions can orient without reading the full source tree (~58 K tokens).
# Trace: inspiring-jepsen-3590b5

<!-- concepts: fac-integration, local-exchange-dfs, shooting-scf -->

## 2026-04-20 · FAC local-exchange DFS port — performance phase

**Session start.** Wrapping up a multi-commit feature branch that ports
Gu's FAC local-exchange SCF into atomx as `exchange='fac'`. All 7 plan
tasks complete, 488/489 tests green (1 pre-existing `test_ci` failure
unrelated). User now wants performance work before deciding merge/PR
path: **be faster than FAC at every Z while keeping the 10⁻⁵ accuracy**.

Known status going in:
* Z ≤ 54: atomx is **2–5× faster** than sfac already
  (e.g. Z=54, 0.11 s vs 0.31 s).
* Z ≥ 74: atomx shooting SCF with Gu's low α_xc (~1.0 at Z=74 vs
  1.5 for the Salvat DHFS default) fails to converge in ≤ 500 iterations —
  we capped the test at Z ≤ 54 to keep the suite fast.

Plan: (1) profile across panel to confirm warm timings ≤ FAC, (2) trace
Z=74 to see whether the bottleneck is iteration count or per-iter cost,
(3) identify targeted fixes, (4) reverify on the Task 0 panel at the
10⁻⁵ gate.

# Trace: trusting-hellman-f45deb

<!-- concepts: shooting-dhfs, slater-integral, heavy-z-expval, dirac-power-series -->

## 2026-04-20 · Heavy-Z total-energy diagnosis + SLAG6 Slater-integral toggle

**Goal (from user).** Replace `_slater_integral`'s power-series ODE
with direct SLAG6 forward+backward Hartree-Y quadrature in
`_expval`, keep the old backend for comparison, tighten the
Z ≥ 74 gate in `tests/test_fac_dfs_shooting.py` from 0.30 to 10⁻³.

**What shipped.**
* New `_slater_integral_slag6(f1, f2, L, r, difr, np_)` in
  `src/atomx/numerics/shooting/solver.py`.
* `slater_method` parameter plumbed through `_expval` →
  `_dhfs_scf` → `ShootingSolver` → `Atom`, default `'slag6'`,
  opt-in `'power_series'`.
* 18 tests in `tests/test_slater_integral_slag6.py` — backend
  agreement ≤ 10⁻⁷ across Z_eff ∈ {9, 30, 70} × L ∈ {0, 1, 2},
  plus neutral-atom DHFS.f regression at Z ∈ {10, 26, 82} for
  both backends.
* `benchmarks/REPORT.md` §6.2 / §6.2.2 (new) / §6.4 / §7 item 6
  (new) rewritten with the actual diagnosis.
* Full suite: 513 passed / 3 skipped, no regression.

**Key finding — the task's root-cause hypothesis was wrong.**
Both Slater-integral backends agree to ~4 × 10⁻⁹ on the exact
Z = 74 / 92 densities that were supposed to stress the power
series, yet they give *identical* heavy-Z totals.  The power
series does NOT lose precision for sharp inner peaks.

**Real root cause (newly discovered).** `dbound` / `dir0` (the
κ > 0 "Case A" small-r power series in
`src/atomx/numerics/shooting/dirac_power_series.py:125`) returns
a ~3× too-deep eigenvalue for 2p₁/₂ (κ = +1) orbitals at
Z ≳ 50 in bare or weakly-screened potentials.  Concretely:
bare Coulomb Z = 74 n = 2 κ = +1 → atomx −2394.7 Ha,
exact Dirac −758.7 Ha.  Fortran DHFS.f on the same Ne-like
W⁶⁴⁺ input gives 2p₁/₂ = −17.0 keV and total = −301.3 keV
(matches sfac to 4 eV); atomx gives 2p₁/₂ = −57.9 keV and total
= −382.7 keV.  Neutral atoms are unaffected because outer
screening moves the 2p₁/₂ root out of the broken regime.

**Consequently.** The gate in
`tests/test_fac_dfs_shooting.py` stays at 0.30 for Z ≥ 74 with
an updated comment pointing to the real `dbound` bug.  Tightening
it to 10⁻³ requires fixing `dir0`'s κ > 0 small-r series, which
is tracked as `benchmarks/REPORT.md` §7 item 6.

### EARS — Session End (2026-04-20 wrap-up)
<!-- concepts: shooting-dhfs, slater-integral, dbound-bug -->
- Accomplished: SLAG6 toggle ships, DHFS.f bit-match preserved
  for neutrals, real root cause of heavy-Z ion over-binding
  diagnosed and documented (dbound κ > 0 small-r power series,
  not the Slater integral).
- Next session: fix `dirac_power_series.dir0` Case A branch so
  κ > 0 2p₁/₂ at Z ≳ 50 matches exact Dirac hydrogenic.  Then
  re-verify with `bench_local_exchange.py` and tighten the
  `test_fac_dfs_shooting.py` gate to 10⁻³.
- Open issues: `dbound` / `dir0` κ > 0 Case A bug (see
  `benchmarks/REPORT.md` §7 item 6).

### EARS — Follow-up (2026-04-20 post-wrap-up)
<!-- concepts: dbound-eigenvalue-search, b-spline-vs-shooting, fac-comparison -->
User asked for the real `dir0` κ > 0 fix with step-by-step FAC
comparison, and to check if the B-spline path has the same issue
against `../radial` and `../dbsr_hf`.

**Further localisation of the dbound bug.** Ruled OUT:
1. `dir0` Case A power-series coefficients — character-by-character
   match with Fortran radial.f DIR0 (ME-4.30..4.34).
2. `doutw` TCONV early-exit — disabling it doesn't change the
   wrong-answer behaviour; finite-difference TCONV and Fortran
   analytical `2·CA(3)·PPI` both detect the classically forbidden
   region correctly when iotp is set right.
3. Premature `ier=3` exit in dbound when turning point leaves the
   grid — tried removing; still converges to spurious E.

The actual bug: `dbound`'s **shoot loop itself** converges on a
non-physical matching condition when e_guess is too shallow.  At
bare Coulomb Z=74, `dbound(e_guess=-758.67)` correctly returns
−758.68 Ha (ier=0), but any shallower guess (including the solver's
default −0.1) gives the spurious −2394.95 Ha.  The "too many nodes"
branch drives E below physical min, search range expands into an
unphysical region, and a spurious E satisfying the Q-discontinuity
matching is found.

**Cross-solver check (user's direct question).**
* `exchange='shooting'` / `'fac'` via `ShootingSolver`: matches
  Fortran `../radial/DHFS.f` to 10⁻⁸ for neutral atoms Z = 2..92,
  but fails by 20–30 % for Ne-like / Ar-like ions Z ≥ 74.
* `exchange='slater'` / `'fac'` via B-spline `DFSolver`: matches
  Fortran DHFS.f for neutrals AND heavy-Z ions.  Verified Z=74
  W⁶⁴⁺ Ne-like atomx=−301,310 eV, DHFS.f=−301,327 eV (rel
  1.7 × 10⁻⁵); Z=82 Pb⁶⁴⁺ Ar-like 4.1 × 10⁻⁵; Z=92 U⁸²⁺ Ne-like
  1.6 × 10⁻⁴.  So the B-spline path does **not** share the bug.
* `../dbsr_hf` is a Hartree-Fock code (exact exchange), not local
  DFS, so not directly comparable to either `exchange='shooting'`
  or `exchange='slater'` — the relevant atomx equivalent is
  `exchange='hf'` which uses a separate `DFSolver` path already
  validated via `scripts/generate_density_gs.py`.

**Interim fix (not yet applied, awaiting user decision).**
One-liner in `src/atomx/atom.py:258` to route
`Atom(exchange='fac')` through the B-spline `DFSolver` for
highly-charged ions.  The B-spline FAC wiring already exists
(tests/test_dfsolver_fac.py).  This would close the heavy-Z
facade gap without needing to fix `dbound` itself.  `dbound`
remains a standalone follow-up.

### EARS — Bug FIXED (2026-04-20 post-merge)
<!-- concepts: doutw-tconv, dirac-power-series, ca2-coefficient -->

Real root cause finally localised AND fixed.  Debugging went three
layers deep:

1. Ruled out `_slater_integral` (Slater integral quadrature).
2. Ruled out `dir0` / power-series coefficients (character-by-
   character match with Fortran radial.f DIR0).
3. Ruled out `dbound`'s outer eigenvalue search logic (it matches
   Fortran's emin/emax/bisection exactly).
4. Ruled out `doutw`'s early-exit `ier=3` (Fortran has it too).

**Actual bug:** `doutw`'s past-turning-point blow-up check
``TCONV = P''·P`` used a 3-point finite-difference proxy.  For
sharp heavy-Z 2p₁/₂ wavefunctions, this proxy mis-detected the
sign of P'' near the turning point (finite difference gave
NEGATIVE TCONV where analytical power-series gave +1.6 × 10⁵ at
Z=74).  Result: TCONV never fired past the turning point,
integration continued into the classically-forbidden region and
picked up irregular-solution contamination that registered as a
spurious radial node, driving the "too many nodes" branch 3× too
deep.

**Fix:** thread the analytical power-series ``(r-r0)²`` coefficient
``ca[2]`` out of ``dir_integrate`` and use ``TCONV = 2·ca[2]·PPI``
exactly as in Fortran radial.f DOUTW line 1780.  Two callers
updated (doutw, dinw) — dinw only to consume the new return
value, TCONV not used in inward integration.

**Cross-solver sanity.**

| path | vs `../radial/DHFS.f` (Salvat)                         | vs `sfac` (FAC) |
|------|--------------------------------------------------------|-----------------|
| `exchange='shooting'` / `'fac'` (shooting) | neutral Z=2..92 : 10⁻⁸ ✓; Ne-like W⁶⁴⁺ **now 2.2e-4** ✓ | same 2.2e-4 |
| `exchange='slater'` / `'fac'` via DFSolver (B-spline) | neutral Z=2..92 : 10⁻³ ✓; Ne-like W⁶⁴⁺ : 5.7 × 10⁻⁵ ✓ | same |
| `exchange='hf'` (B-spline DHF) | — | — (different physics; compared against ../dbsr_hf in generate_density_gs.py, not here) |

**Verification — Fortran demorad on the exact same grid.** Compiled
Salvat's `demorad.f` from source on arm64 with atomx's grid params
(NGP=1000, RN=75, R2=1e-7, DRN=0.5), ran Dirac bound-state solver
on bare Coulomb Z=74 and Z=92 2p₁/₂ — both return the correct
Dirac hydrogenic eigenvalue.  Before fix: atomx returned
3× too-deep spurious value for the same inputs.  After fix:
atomx returns −758.68 Ha (Z=74) and −1257.40 Ha at correct initial
guesses.  End-to-end: Ne-like W⁶⁴⁺ via `Atom(exchange='shooting')`
drops from −383,544 eV (27 % off) to −301,389 eV (2.2 × 10⁻⁴ off).

**Tests.**
* Tightened `tests/test_fac_dfs_shooting.py` Z ≥ 74 gate from
  0.30 to 1 × 10⁻³.  All 14 panel rows now pass the tighter gate.
* New `tests/test_dbound_heavy_z.py` pins bare-Coulomb 2p₁/₂ at
  Z ∈ {10, 30, 50, 74, 82} from shallow + hydrogenic guesses.
* Full suite: 526 passed / 3 skipped (up from 513 — 13 new tests).

**Residual edge.** Bare-Coulomb Z = 92 2p₁/₂ from a shallow
initial guess still hits a spurious root (too-aggressive emin
expansion in "too many nodes" branch when sign flip happens BEFORE
the turning point).  Does NOT affect any atomic SCF (screening
keeps the eigenvalue out of this regime).  Tracked but
low-priority.

### EARS — Second dbound patch: turning-point-scan fallback (2026-04-20)
<!-- concepts: dbound-scan-fallback, fortran-do-loop-semantics -->

Compiled a verbose variant of Salvat's radial.f (with
``DATA IWR/1/``) and ran it against the "residual" Z = 92 bare
Coulomb 2p₁/₂ case.  The verbose output showed Fortran's shot
loop DOES jump to E = −3701 Ha on shot 0 (same as atomx), but then
at shot 1 Fortran gets ``NZERO=0`` with ``IOTP=269`` (TCONV fires
early in the forbidden region).  atomx on the other hand hits
``ier=3`` immediately on shot 1 because its turning-point scan
can't find a classically-allowed region.

**Why:** Fortran's scan
```fortran
DO J=2,NRT
  IOTP=NRT+2-J
  ...
  IF(EKIN.GT.0) GOTO 3
ENDDO
```
**assigns** ``IOTP`` on every iteration.  If the loop completes
without break (no classically-allowed region), ``IOTP = 2``
(innermost, 1-indexed) from the last iteration.  After
``IOTP+1=3`` the gate ``IOTP > NRT-1`` doesn't fire and DOUTW
proceeds from the inner end.

atomx's scan initialised ``iotp = nrt - 1`` **outside** the loop
and only reassigned inside the break, so the Python for-loop's
"fall-through" left ``iotp = nrt - 1``.  After ``iotp+1 = nrt``,
the gate ``iotp > nrt - 2`` tripped and ``ier=3`` fired before the
shot loop could bisect.

**Fix:** mirror Fortran's scan exactly — initialise
``iotp = 1`` (0-indexed fallback equivalent to Fortran ``IOTP=2``),
loop ``jj=2..nrt`` visiting ``idx=nrt-1`` down to ``idx=1``,
and relax the post-scan gate to ``iotp > nrt - 1`` so it only
fires when the turning point genuinely lands past the last valid
grid index.

**Impact.**  Bare-Coulomb Dirac-hydrogenic 2p₁/₂ now converges
from any shallow initial guess up through Z = 100 (verified by
``tests/test_dbound_heavy_z.py`` at Z ∈ {10, 30, 50, 74, 82, 92,
100} × shallow & hydrogenic guesses, plus the n=1..3 shell at
Z = 74, 92 — 35 tests total).  No change to heavy-Z ion SCF
accuracy (already correct after the TCONV patch).  Full suite:
526 → 548 passed / 3 skipped.

**Nothing left on the shooting path.**  The three independent bugs
unmasked by this investigation (``_slater_integral`` SLAG6 option,
``doutw`` TCONV analytical formula, ``dbound`` turning-point scan
fallback) are all fixed and regression-locked.  The residual claim
in the prior trace entry ("Z = 92 shallow-guess bare Coulomb still
hits a spurious root") is **no longer true**.

### EARS — Performance pass: n_grid = 400 default (2026-04-20 evening)
<!-- concepts: adaptive-integration, grid-sizing, performance -->

After finishing all three correctness fixes, revisited performance.
Profile showed 99%+ of ``Atom(exchange='fac').solve()`` time lives
inside the numba-JITed ``_dhfs_scf``, and ``dbound`` per shell was
1–2.5 ms with most of the cost coming from per-sub-interval
``dir_integrate`` calls.

**Key empirical finding.** Sweeping ``n_grid`` on the Z = 74 Ne-like
panel from 150 to 2000 showed the total binding energy was
**identical** (rel = 2.2 × 10⁻⁴) across the full range, with
runtime scaling linearly in ``n_grid``.  Reason: Salvat's
``dir_integrate`` adaptively sub-divides each grid interval to hit
the ``EPS`` tolerance, so the static grid only determines where we
sample the wavefunction at the end, not the integration accuracy.

Dropping the ``Atom`` facade default from the DHFS.f historical
1000 to 400 cuts wall time by 2.2× across the full Z = 2..92 panel
with no loss of end-to-end accuracy:

| Z  | old (n=1000) | new (n=400) | sfac  |
|----|-------------:|------------:|------:|
| 10 | 124 ms       | **59 ms**   | 400 ms |
| 26 | 105 ms       | **48 ms**   | 270 ms |
| 54 |  99 ms       | **45 ms**   | 310 ms |
| 74 |  94 ms       | **42 ms**   | 310 ms |
| 92 |  95 ms       | **42 ms**   | 330 ms |

atomx is now **5–14× faster than sfac** (was 2–5× at n=1000).

Neutral atoms still match Fortran DHFS.f to ≤ 3 × 10⁻⁷ relative
(vs ≤ 2 × 10⁻⁸ at n=1000).  Users who need the tighter bound can
pass ``n_grid=1000`` directly to ``ShootingSolver``.  The three
regression test files (`test_slater_integral_slag6.py`,
`test_dbound_heavy_z.py`, `test_fac_dfs_shooting.py`) all pass
unchanged at the new default; full suite 548 passed / 3 skipped.

### EARS — Session Start (2026-05-01 13:02)
<!-- concepts: ci-solver, mcdhf, test-tolerance -->
- Task: Investigate and fix the failing CI test
  `test_ci_solve_matches_first_mcdhf_macroiteration` in `tests/test_ci.py`,
  where pure `ci_solve` on frozen DHF orbitals (-2.8451 Ha) deviates
  ~0.18% from MCDHF first macro-iteration ground energy (-2.8401 Ha),
  exceeding the `rel=1e-10` assertion.
- Why: Restore green CI for atomx; ensure CI/MCDHF code paths are
  consistent (Phase-6 split invariant) before further MCDHF work lands.

### EARS — Progress (2026-05-01 13:06)
<!-- concepts: ci-solver, mcdhf, bspline-knots -->
- Root cause for `test_ci_solve_matches_first_mcdhf_macroiteration` failure:
  `MCDHFSolver.solve()` swaps in the DBSR-HF semi-exponential physics knots
  grid (`make_physics_knots`) post-construction (mcdhf/solver.py:246-256),
  overriding the user's `ns`. The test's standalone DHF reproduction used
  the default `DFSolver` knot grid with `ns=50`. For a single CSF (1s2 He)
  the CI matrix is 1×1, so CI energy *equals* the DHF energy — and the two
  different bases gave -2.8401 (physics grid) vs -2.8451 Ha (default grid),
  a 5 mHa discrepancy that swamped the rel=1e-10 assertion.
- Verified empirically by toggling `use_physics_grid` on the inner MCDHF
  call: with both paths on the default grid, CI vs MCDHF first-macro agree
  to 1.3e-10 Ha; with both on the physics grid they would also agree.
- Fix (test-only): pass `use_physics_grid=False` to `MCDHFSolver` in the
  test so both code paths share the default knot grid. Production behavior
  unchanged — the test was just comparing apples to oranges.

### EARS — Progress (2026-05-15 11:25)
<!-- concepts: mcdhf-csf-enumeration, jj-coupling, fac-parity -->
HYP-165 (M3): User asked to "systematically compare intermediate results and fix". Root cause for Fe XVII 27 vs 37 gap: FAC's `Config('n3', '2*7 3*1')` expands to 7 non-relativistic configs (2p-hole AND 2s-hole patterns), while atomx's benchmark only lists 4 (2p-hole only). Compounded by `enumerate_jj_csfs` leaving `total_J2=0` for every CSF — no J-resolution at all. Counted: 7 NR configs → 1 + 4 + 10 + 12 + 2 + 4 + 4 = 37 J-resolved CSFs (matches FAC). Implementing: (1) seniority-allowed J enumeration per subshell, (2) cascaded CG triangle to total J, (3) multi-config helper, (4) Atom() multi-config path uses J-resolved, (5) block-diagonal CI by (J,π) in solver, (6) extend Fe XVII benchmark configs to 7.

## 2026-05-15 17:43

<!-- concepts: charge-density, benchmark-reference, wfn-formats -->
- User asked to compare atomx ground-state charge density with
  `/Users/zhangzz/code/ASAP/WF_TXT/<Z>.txt` (Z=2..108).
- WF_TXT file format reverse-engineered: each file has *multiple* sections.
  First section is `R, <orbital>LC, <orbital>SC, ...` for every occupied
  relativistic subshell (with `*` = j-, no-star = j+); sections continue
  with more orbitals (variable column count, padded with empty cells at
  the tail of each orbital's support). Final section is `R, Density`
  with `int rho(r) dr = Z` exactly — i.e. *radial* density, already
  weighted by occupations, same convention as atomx's `density` field.
- Existing atomx outputs in `scripts/output/` cover only H..Ca (Z=1..20),
  so the cross-check is limited to that range unless we regenerate.
- Wrote `scripts/compare_wf_txt.py`: sums per-subshell `density` arrays
  (each on its own tail-trimmed grid) onto a common log grid via linear
  interp + zero-pad, then compares against the WF_TXT density on the
  support intersection. Emits PNGs + REPORT.md in `scripts/wf_txt_compare/`.
- Pyright "could not be resolved" warnings on h5py/numpy/matplotlib are
  the usual VSCode-vs-uv interpreter mismatch — not a real problem; the
  script runs under `uv run`.

### EARS — Progress (2026-05-15 17:53)
<!-- concepts: charge-density, periodic-table, atomx-constants -->
- Extending WF_TXT comparison to Z=79 (Au), 92 (U), 103 (Lr), 108 (Hs).
  Spawned `scripts/generate_density_gs.py 79 92 103 108 -v` in background
  to produce the missing `.h5` outputs.
- Refactored `scripts/compare_wf_txt.py` to use `chemical_symbols` from
  `atomx.constants` instead of a hard-coded H..Ca map, so the comparison
  table auto-extends to whatever h5 files exist in `scripts/output/`.

### EARS — Progress (2026-05-15 18:12)
<!-- concepts: refactoring, script-merge, density-comparison -->
- Merged `generate_density_gs.py` and `compare_wf_txt.py` into a single
  `scripts/density_gs.py`. New script: per element it generates the h5
  (skip if present, --force to overwrite) then compares against
  `/Users/zhangzz/code/ASAP/WF_TXT/<Z>.txt` (skip if absent or --no-compare),
  emitting per-element PNG + a final REPORT.md.
- Cut the `--verify-ag` shim (one-shot sanity check against a frozen Ag
  reference) and the long inline schema docstring — both still in git
  history.
- About to delete the two old scripts and re-run the regression on the
  23 existing elements (H..Ca + Au, U, Lr, Hs) to confirm the merged
  script reproduces the same REPORT.md.

### EARS — Progress (2026-05-16 12:39)
<!-- concepts: dielectronic-recombination, FAC-benchmark, autoionization-driver -->
- Working on HYP-146 DR driver `dr_calculate()` + FAC `SetDR` parity (Pojo re-trigger).
- Scaffold from codex (`c8ba72e`) was already on top of `origin/main` — no rebase needed.
- Key discovery: local sfac (`/Users/zhangzz/code/ASAP/fac/bin/bin/sfac`, FAC 1.1.5) has **no `SetDR` or `DRTable` symbols** (removed in FAC 0.9.3 per codex's earlier note). Confirmed via `strings | grep`. Available primitives: `AITable`, `TRTable`, `Structure`, `OptimizeRadial`, `RecStates`, `MemENTable`. Canonical FAC §6.3 protocol with available primitives = `Structure → AITable → TRTable`, then combine via the same Burgess detailed-balance kernel `DRTable` would have used internally (Gu 2003, ApJ 590, 1131).
- Wrote working AI+TR sfac script for Fe XVII → Fe XVI (Δn=0 KLL + Δn=1 KLM). Produces 194 autoionising resonances and 3190 stabilising TR rows in ~1 sec; parsed cleanly with new `parse_ai` + existing `parse_levels`/`parse_transitions`.
- Discovery on FAC ILEV bookkeeping: the recombining-ion ground is *not* always ILEV 0 (depends on `E0` reference). Robust rec-idx pick: `Counter(r['lower'] for r in ai_rows).most_common(1)[0][0]` — the level appearing most often as the AI lower-target is by definition the continuum-anchor.
- Tension: atomx `dr_calculate` defaults overshoot FAC α_DR by 100-1000×. Two combined causes — (i) `FeXVIIModel` orbital energies in the bench were placeholder; updated to FAC-derived screened values (helped ~10×). (ii) The default `_default_auger_width` and `_default_radiative_width` are admittedly screened-hydrogenic and not tuned for highly-charged ions; the per-resonance branch is dominated by inflated Γ_A. This is a *documented limitation* of the scaffold ("hooks accept ab-initio matrix elements once HYP-171/M3 lands").
- Dead-end avoided: tempted to retune the default prefactors to chase the 20% gate — backed off because that's overfitting placeholders. Cleaner test design: split the gate into (a) kernel parity using FAC widths fed through `dr_rate_coefficient` (should pass tight), (b) end-to-end atomx-driven with current defaults (xfail with HYP-171 follow-up rationale).
- Fixed broken master test `tests/test_fac_runner_parsers.py` (from 74ff5ce) by adding `parse_ci` to `benchmarks/fac_runner.py`.

### EARS — Progress (2026-05-18 20:32)
<!-- concepts: QED-self-energy, FAC-Indelicato-table, high-Z-spectroscopy -->
- HYP-173 re-dispatch: prior codex run finished the Indelicato backend +
  tests locally on `agent/codex-review/79a2389e` (54 tests passing) but
  never pushed to origin. Pojo is asking for push + PR + Z=54/74/92
  H/He/Li-like FAC parity rows in the regression.
- Discovery: the `tests/test_qed_indelicato_vs_fac.py` left in 79a2389e
  *already* covers Z=54/74/82/92 1s F-values, Z=54/74/92 H-/He-/Li-like
  ground levels via `GROUND_LEVELS`, and the leading-1s residual
  <=0.5% gate. So the file mostly satisfies Pojo's "extend the FAC
  regression" ask out of the box; just need to make sure those Z=54/74/92
  residual % numbers show up explicitly in the PR body.
- Approach: copy the 8 modified + 4 new files into the current
  up-to-date `agent/claude/1c98aef6` worktree (none of the upstream
  commits since 5cefde7 touched these files, so no rebase conflict).
  Apply the REPORT.md gap-closed status change inline at the existing
  "Outstanding gaps" item 4 rather than the trailing append (the
  79a2389e copy used a `<<<<<<< HEAD` conflict marker which we must not
  ship).
- Next: run `uv run --extra dev python -m pytest ...` to confirm the
  4-file QED suite still passes, capture residual %, then commit and
  push.

### EARS — Progress (2026-05-19 13:12)
<!-- concepts: MCDHF-CI, J-coupled-CSF, Fe-XVII-spectroscopy -->
- HYP-182 re-dispatch (push-or-invalid). Previous turn (commit 46dd02d) fixed three configuration-averaged Slater–Condon bugs in `src/atomx/methods/mcdhf/matrix.py`: (1) k=0 direct-integral angular check, (2) sqrt(q·h) jj weight, (3) geometric-mean exchange angular factor with k-range intersection. Gate `tests/test_fe17_3c_3d_vs_bernitt2012.py` still xfails because two physics terms remain.
- Current state confirmed: 5 E1 transitions from ground; State_0→State_6 f=2.84 @ 16.307 Å (in [2.2, 3.0] but ~1.3 Å redward of Bernitt window); ratio of top-2 lines (State_24 picks up f=1.40 @ 10.7 Å) = 2.03. 157 passed / 1 skipped / 1 xfailed on the CI/MCDHF/transition subset.
- Remaining gaps are multi-day per the partial-delivery clause: (a) off-diagonal Grant §8.3 J-coupled CI between same-(J, parity) CSFs sharing open subshells (1-diff and 2-diff cases — the diagonal `_diagonal_J_correction` works; the analog for `_one_diff_element` / `_two_diff_element` is the missing piece); (b) DODS / EOL orbital relaxation (the 1.3 Å red-shift is the canonical signature of single Guerra-averaged orbital set vs per-block).
- Decision: per the explicit partial-delivery clause, opened **HYP-184** (Fe XVII 3C/3D wavelength residual + off-diag J-CI + DODS) as the targeted follow-up, updated the test's xfail reason from HYP-183 (which is Au CRM) → HYP-184, and will open a PR documenting the partial-delivery scope. Not chasing the off-diag CI in this session — the diagonal Grant correction took a full session in HYP-171, and DODS is its own multi-day driver change.
