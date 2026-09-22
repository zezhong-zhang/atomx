# Slice S1 — structure / SCF / basis (remediation)

Remediation of `docs/test_audit/S1_structure_scf_basis.md` (20 findings).

Scope: 19 test files under `tests/` plus this report. **No `src/atomx` change**
was made or needed — every finding was closed by tightening a gate, replacing
a self-referential reference with a closed form or an external number, or by
converting an unsatisfiable gate into a `strict=True` xfail that carries the
measured number and the named cause. `docs/VERIFICATION.md`, `CHANGELOG.md`
and `docs/test_audit/**` were not touched. `tests/test_atom_surface_equivalence.py`
is owned by another agent and was not touched (its share of findings 7 and 14
is referenced by name only).

`xfail_strict` is **not** set in `pyproject.toml`, so both xfails added here
pass `strict=True` explicitly.

## 1. Finding-by-finding table

`before → after` is the *gate* (the number the assertion admits), and
`measured` is what the code actually produces on this host (2026-09-22).

| # | file:line (after) | what changed | measured before → after | new gate |
|---|---|---|---|---|
| 1 | `test_fac_dfs_shooting.py:105-182` | run is now `corrections="none"`, matching the reference's operators (`extract_AHX.py:56-59`: `SetSE/SetVP/SetMS/SetBreit = 0`); stale "worst 4.10e-3 at Z=36" paragraph deleted; gate is now per-row, 1.5x the matched-flag measurement | Z=10 1.016e-4→**1.212e-5**; Z=18 1.770e-4→**7.23e-7**; Z=26 2.81e-4→**3.27e-6**; Z=36 3.85e-4→**9.26e-6**; Z=54 6.95e-4→**4.67e-5**; Z=74 1.005e-3→**2.225e-4**; Z=82 9.84e-4→**3.60e-4**; Z=92 9.99e-4→**9.44e-4** (QED shift alone was 370.0/960.7 eV) | per row `1.5 x MEASURED_REL[tag]`, worst 1.42e-3 (Z=92) vs the replaced 1e-3/2e-3 tiers; **every** row is strictly tighter |
| 2 | `test_scf_potential.py:76-118` | `-0.2 < V_hx(10) < -0.05` (passed 1 ULP under the Latter cap) replaced by the two exact asymptotes; the Latter convention (`rvinf = -(1+Z-N_eff) = -2`) is now documented at the assertion instead of changed in `src` | Hartree twin `V·r` = **-0.9999877** (band admitted -1 and -2); Latter branch `V·r` = **-1.9999999999999998 … -2.0** exactly | Hartree `V(r)·r == -1` atol **2e-5** (r = 20/30/50); exchange `V(r)·r == -2.0` atol **1e-9** |
| 3 | `test_dfsolver_slater.py:96-141` | cross-engine gate converted from 1e-3 *relative* to absolute, and a Salvat `radial/DHFS` external row added | Ne 1.049e-4 Ha vs a 0.129 Ha gate (**1226x** slack); Ar 1.538e-5 Ha vs 0.529 Ha (**34000x**) | Ne `abs < 2e-4` Ha, Ar `abs < 5e-5` Ha; **new B-class** `test_neon_slater_matches_salvat_dhfs`: atomx -128.6234601 vs Salvat point-nucleus α=1.5 -128.623390, measured 7.0e-5 Ha, gate **1.5e-4 Ha** |
| 4 | `test_atom_numerics_exchange.py:131-155` | gate tightened; docstring names the shared kernel (α_xc table, jj-average angular coefficients) and the non-shared one (Galerkin vs shooting) | 1.204e-4 Ha (rel 1.050e-7) vs a 5.73 Ha gate (**47600x** slack) | `rel < 1e-5` (~95x the measurement) |
| 5 | `test_bspline_grid_accuracy.py:75-108` | tautology `test_error_grows_strongly_with_z` **deleted** (it asserted the test's own `MEASURED` dict was monotone and never called atomx); profile converted from a two-sided ±25 % band to a one-sided bound and marked `snapshot`; the deleted test replaced by an A-class default-grid bound against the exact Dirac eigenvalue | Z=92 profile 97.737 eV (band also accepted *improvements*); tautology could not fail | profile `err(Z) < MEASURED[Z]*1.25`; new `err(Z=92, default) < 100 eV` |
| 6 | `test_scf.py:1-27, 96-130` | wrong citation fixed: "-128.547 Desclaux 1973 DHF" was the non-relativistic HF limit (it fails the :120 assertion by 0.145 Ha = 145x the tolerance). Docstring now names DBSR_HF -2.86181332 (He) and -128.69192580 (Ne) with sources; the 4 D-class energy pins marked `snapshot` | He DHF -2.86181334 (3.2e-9 from DBSR_HF); Ne DHF -128.69196949 (3.05e-5 from DBSR_HF) | unchanged literals, now correctly attributed and marked |
| 7 | `test_scf.py:142-155`, `test_slater_integral_slag6.py:144-155`, `test_exchange_model_is_honoured.py:11-21` | cross-reference comments at each of the three sites I own, naming nucleus model + exchange model + engine per value; the fourth (Fermi/HF -128.6299783) is referenced by name only (file owned by another agent) | four values, previously indistinguishable | no numeric change (as instructed) |
| 8 | `test_slater_integrator.py:56-105` | false "~1e-3 diagonal product-quadrature error" comment deleted; gates tightened; **new** closed-form gate for the kernels | k=0 2.19e-14, k=1 4.68e-14, k=2 2.72e-14, k=3 3.06e-14 vs a 5e-2 gate (Auger 6.86e-14 vs 1e-1) | kernels `rel=1e-10`; Auger `rel=1e-10`; new `R0(1s,1s) = 5Z/8` (Slater–Condon, independently re-integrated with scipy) at `rel=1e-4`, measured 2.84e-5 |
| 9 | `test_dfs_frozen_orbitals.py:42-90` | gates tiered per orbital from the measurement; `z_asym == approx(17.0, abs=0.2)` → exact | 1s 1.310e-4, 2s 1.758e-3, 2p1/2 1.521e-3, 2p3/2 1.572e-3; virtuals 9.10e-4/1.389e-3/3.576e-3; `z_asym` = **17.000000000000** | 1s **5e-4** (10x tighter), 2s/2p **5e-3**; virtuals **2e-3 / 3e-3 / 5e-3**; `z_asym` abs **1e-9** |
| 10 | `test_continuum_knots.py:42-90` | `ns_cont >= 0.7*ns_phys` (a 30 % coarsening passed) replaced by strict inequalities; the `if len(inner_knots) >= 3` guard that could void the spacing test replaced by an assert | ns 89 vs 78, inner knots 19 vs 14; spacing median 9.615e-3 vs λ/4 | `ns_cont > ns_phys` and `inner_cont > inner_phys` (strict); `len(inner) >= 3` asserted; spacing gate unchanged |
| 11 | `test_potentials.py:104-155` | the `<=` test that passed on 12 vs 12 split into (a) a C-class check that the start does not move the converged energy, and (b) a `strict=True` xfail for the acceleration claim | `starting_potential=None/"tfda"/"gsz"` all give **12 iterations** and the **bit-identical** E = -128.6299783505843 Ha | (a) `E_tfda == E_default`; (b) xfail reason carries 12 vs 12, the bit-identical energy and the named cause |
| 12 | `test_bspline_local_potential.py:1-110, 386-420, 440-500` | C-degenerate tests labelled as API contracts; the `-Z/r` case given an **independent closed form** (Dirac hydrogenic), plus an exact `-Z/r` check on the `nuclear` channel | solver on a `-Z/r` table: -50.0667410 vs the Dirac closed form -50.0667420 (**2.0e-8**), -12.5208598 vs -12.5208597 (1.2e-8), -12.5041633 vs -12.5041630 (2.3e-8); the Schrödinger value -Z²/2 is 1.3e-3 away, so the test now also proves the relativistic kinematics | new `rel=1e-7` closed-form gate (was: only a 1e-10 comparison against the shared `build_dirac_matrices` path); `nuclear == -Z/r` at rtol 1e-13 (measured 1.8e-16) |
| 13 | `test_scf_potential.py:46-74, 120-215` | ±100 % / `<0.05` bands replaced by the exact asymptotes `V·r → N-Z` and `V·r → -Z`; new test pinning the documented clamp below the first tabulation point | He `V_HF(20)·20` = 2.456e-5; Ar = 2.293e-3; Ne `V_HF(0.01)/V_bare` = 0.7599 inside the clamped region | He atol **5e-5**, Ar atol **3e-3** (both: was 0.05 on `V` = ~1.0 in `V·r` units); Ne `V·r(5e-4..1e-3) = -Z ± 0.05`; `n_radial=400000` required and explained |
| 14 | `test_scf.py` (4), `test_atom_numerics_exchange.py:79`, `test_bspline_grid_accuracy.py:76`, `test_shooting_guerra.py:69`, `test_fac_branch_inventory.py` (module) | `@pytest.mark.snapshot` added to every literal-pin function in the files I own; `test_no_jj_configs_bit_identical` renamed (the pins are relative, only the repeat check is bit-exact) | 20-odd D-class functions were unmarked | markers only |
| 15 | `test_dirac_hydrogenic.py` (whole file) | the 7 self-referential comparisons replaced by `_dirac_coulomb`, the Sommerfeld closed form written out in the file with its derivation; `test_hydrogen_1s` tightened; both exact properties (κ-parity, n=1 collapse) documented in the helper (already gated in `test_first_principles_invariants.py`, deliberately not duplicated) | `hydrogenic_dirac_energy` vs the in-test closed form: 1.7e-12 (H 1s), 1.5e-11 (H 2s), 2.4e-10 (U91+ 1s) — the residual is atomx's `C` differing from CODATA by 6.8e-10 relative | closed form in-test; package-vs-closed-form `rel=1e-9`; H 1s literature -0.50000665 at `rel=1e-7` (was 1e-6) |
| 16 | `test_ladder_and_sampling.py:167-222` | the misnamed no-op (`E < 0` only, comment abandoning monotonicity) replaced by the ladder's real content; renamed | measured rungs -4.0, -4.0002130, -2.8616799953898364, -2.861893000807627, -2.861813320655833 — **not monotone** | HartreeFock rung `== -2.8616799956 ± 1e-6` (NR-HF limit, Froese Fischer; measured 2.1e-10 away); DiracFock `-2.903724 < E < -2.8616799956`; hydrogenic rungs `-Z² = -4` |
| 17 | `test_fac_asymptotic.py:134-171` | the clamp identity (true whether or not the envelope ran) replaced by a pure-clamp assertion plus an N=1 vs N=2 discriminator | N=1: **every** point is exactly -5 (rv_inf = -(5-1+1)); N=2: last point -4.0 and a different curve | `out_N1 == -5.0` atol 1e-12; `out_N2[-1] == -4.0` abs 0.01; `not allclose(N1, N2)` |
| 18 | `test_dfsolver_fac.py:1-90` | FAC provenance cited (calibration.csv `AHX_fac`, FAC `radial.c`: 0.514651052 at Z=10, 0.504451201 at Z=26); the constant-only pin replaced by an α_xc→**energy** consequence | gu vs slater(α = AHX_fac·(4π²/3)^⅓): Ne **1.85e-13**, Fe XVII **1.39e-15** relative | new `rel < 1e-9` per Z (a silent fall-back to the Slater default 2/3 would move the energy by ~10 Ha) |
| 19 | `test_fac_branch_inventory.py:1-45` | module marked `pytestmark = pytest.mark.snapshot` and its docstring now says these counts are a coverage budget for *unimplemented* physics, not validation of the FAC port | unchanged counts (210/330 rows, 192 branch-9 pending, 12 + 6 fallbacks) | markers + wording only. The end-to-end Fe XVII `fac-angular` level gate against live `sfac` was **not** added — see §2 |
| 20 | — | no action required by the audit ("none required for the two marker cases") | `guard-page` oracle still `skipif(not linux)`; `test_determinism.py:47` still `slow` (nightly) | — |

## 2. Findings NOT fixed (and why)

* **19 (partly) — the live-`sfac` Fe XVII level gate.** The audit's suggested
  addition needs either the `sfac` binary or a frozen FAC level dump for the
  `fac-angular` branches; neither is in this slice's fixtures
  (`benchmarks/fac_dumps/*.lev` belongs to `test_fac_io_roundtrip.py`, which is
  `slow` and outside S1). The existing end-to-end FAC gate
  (`test_fac_dfs_shooting.py`, now QED-matched) covers the total energy. The
  gap is written into the module docstring so it is visible.
* **20 — platform visibility.** No remediation was required. The Linux-only
  `mmap` guard-page oracle and the `slow` numba bounds-check gate are recorded
  here as still-conditional; making them portable is a new-test task, not a
  remediation of an existing gate.
* **1 and 6 — the documentation sites outside my write scope.**
  `docs/VERIFICATION.md:39` ("Z = 2..92, 5e-4 structure parity") and
  `docs/VERIFICATION.md:25/28` (SCF values "pinned as regression"; the
  hydrogenic file advertised as "known-answer, literature" while 7 of its 16
  tests compared atomx to atomx) still need the edits the audit asks for. The
  test-side halves are done: the FAC panel docstring now states the measured
  envelope and the hydrogenic file is now a real literature gate.
  `docs/VERIFICATION.md` is explicitly off-limits for this task.
* **7 — no shared `tests/references.py`.** The audit proposed one module-level
  reference table imported by all four files. The instruction for this slice
  was to add short per-site comments naming the nucleus model and the exchange
  model instead, so four files were left import-independent (and one of the
  four is owned by another agent).
* **14 — `tests/test_atom_surface_equivalence.py` markers.** Owned by another
  agent (it already carries `@pytest.mark.snapshot` on three functions).

## 3. Genuine failures that tightening exposed

1. **The audit's own Z ≤ 10 gate for the FAC panel cannot hold (finding 1).**
   With matched flags the light rows measure 1.344e-4 (Z=2), 4.16e-5 (Z=3),
   5.32e-5 (Z=4), **6.46e-4 (Z=6)**, 1.21e-5 (Z=10) — not the 1e-5 the audit
   proposed (that would fail on four of the five rows), and the audit's "5e-4
   for the rest" would fail at Z=82 (3.60e-4) and Z=92 (9.44e-4). The residuals
   are a *model* difference, not a numerics artefact: Z=6 is unchanged at
   `tol=1e-11` (6.4561e-4 both ways, 25 vs 46 iterations) and it is the one
   open-shell light row (2p², jj-average vs FAC configuration average).
   Resolution: per-row gates at 1.5x the measured residual, which are strictly
   tighter than the replaced tiers on **every** row.
   Also measured: at Z=92 the QED-off run is 466.79 eV *more* bound than FAC
   and the QED-on run 493.88 eV *less* bound — QED-off only buys 5 % of the
   gate there (9.99e-4 → 9.44e-4), so that row's gate cannot go below ~1.4e-3.
2. **The neutral-atom asymptote is not exact (findings 2/13).** The audit's
   `|V_HF(20)·20| < 1e-4` fails by 23x for Ar (2.293e-3) and 15x for Ne
   (1.487e-3); He passes at 2.456e-5.
   Measured cause: `static_hartree_potential` linearly re-grids the B-spline
   density onto a uniform `n_radial=4000` mesh (Δr = 1.3e-2 a₀, narrower than
   the Ar 1s peak) and integrates it with the trapezoid rule, so the enclosed
   charge is off by +2.6e-3 e (Ar) / +1.5e-3 e (Ne). At `n_radial=400000` the
   same code gives the physically correct behaviour (`r·V(5e-4) = -10.018`,
   `r·V(0.02) = -9.385`).
   Resolution: the passing gates were set at the measured level (He 5e-5,
   Ar 3e-3, both still 17-20000x tighter than the replaced 0.05 band), and the
   audit's exact target was added as
   `@pytest.mark.xfail(strict=True)` carrying the numbers and the quadrature
   cause — **not** weakened into the passing gate. No `src` change was made:
   switching that helper to the B-spline quadrature weights is a numerics
   change I cannot verify against an independent reference for every caller.
3. **The TFDA starting potential is silently ignored on the `Atom`/`Basis`
   route (finding 11).** `starting_potential=None`, `"tfda"` and `"gsz"` all
   give 12 iterations and the bit-identical total energy -128.6299783505843 Ha.
   Named cause: `DFSolver` consumes `starting_potential` only in its first
   hydrogenic eigensolve (`dirac_fock.py:1726-1777`), which the MCDHF/`Atom`
   route skips because it warm-starts from `seed_orbitals`; the option is live
   only for a directly constructed `DFSolver`. Recorded as a `strict=True`
   xfail; the old assertion (`<=`, 12 vs 12) is gone.
4. **The audit's "≥ 3x inner knots" for continuum grids does not hold
   (finding 10).** Measured 19 inner knots vs the physics grid's 14 = 1.36x.
   Resolution: used the audit's alternative — strict `ns_cont > ns_phys` and
   `inner_cont > inner_phys` — which is still a real gate (the old
   `ns_cont >= 0.7*ns_phys` admitted a 30 % coarsening).
5. **The audit's variational sandwich does not apply to the DiracFock rung
   (finding 16).** The rungs are `-4.0, -4.000213, -2.8616800, -2.861893,
   -2.8618133` — not monotone (the two hydrogenic rungs are non-interacting and
   sit ~1.14 Ha below), and `E_DiracFock = -2.8618133` is *below* the
   non-relativistic HF limit -2.861680, so "`-2.903724 < E < -2.861680`" is
   false for it (relativity lowers it by 1.33e-4 Ha).
   Resolution: gate what is exact — the HartreeFock rung against the published
   NR-HF limit -2.8616799956 (measured 2.1e-10 away), the DiracFock rung as
   `-2.903724 < E < -2.8616799956` (strictly below the non-relativistic limit),
   and the hydrogenic rungs as `-Z²` exactly.
6. **The Slater `R^k` kernel is externally accurate to ~3e-5, not machine
   precision (finding 8).** Independent check: for a hydrogenic 1s,
   `R⁰(1s,1s) = 5Z/8`; scipy gives 6.250000127 (Z=10) while atomx gives
   6.2501773 — 2.84e-5 relative, the *same* relative offset at Z=1, 10 and 26.
   For a smooth ρ = e^{-r} the kernel's error is 1.37e-4 relative (exact value
   ln 4 = 1.38629436112, atomx 1.38648397) even though ∫ρ dr is exact to 1e-16.
   So the deviation is systematic in the `Y^k` construction, not peak
   resolution. The new closed-form gate therefore uses `rel=1e-4` (3.5x the
   measured 2.84e-5) and the observation is recorded here for follow-up; the
   two-kernel agreement gate is `rel=1e-10` (measured 2e-14) and the two
   kernels share that systematic, so it cannot see it.
7. **`test_bspline_grid_accuracy` profile: tightening the delivered fix.** The
   refined-grid bounds went from `< 1.0 eV` / `< coarse/50` to `< 0.5 eV` /
   `< coarse/200`; measured 0.2416 eV vs 97.737 eV (404x), so both hold with 2x
   margin. The default-grid Z=92 bound (97.737 eV) is carried at `< 100 eV`
   exactly as the audit specified — a 2.3 % margin, i.e. it fails on any real
   degradation but would also fire on a deliberate default-grid change; that is
   the point of the gate.

## 4. Verification

All 19 touched files, one run:
`228 passed, 2 xfailed, 0 failed` (`-p no:randomly`).
`ruff check` (repo config E4,E7,E9,F,I,B,SIM; E741 ignored) is clean on all 19.
Per-file counts are in §5 of the hand-off message.

Per-file summary (latest run of each file):

```
test_fac_dfs_shooting            15 passed
test_scf_potential               10 passed, 1 xfailed
test_dfsolver_slater              9 passed
test_atom_numerics_exchange      20 passed
test_bspline_grid_accuracy       14 passed
test_scf                         16 passed
test_slater_integral_slag6        18 passed
test_exchange_model_is_honoured   8 passed
test_dirac_hydrogenic            16 passed
test_slater_integrator           11 passed
test_continuum_knots              6 passed
test_potentials                  15 passed, 1 xfailed
test_dfs_frozen_orbitals          5 passed
test_bspline_local_potential     16 passed
test_ladder_and_sampling         20 passed
test_fac_asymptotic               9 passed
test_dfsolver_fac                 6 passed
test_fac_branch_inventory         7 passed
test_shooting_guerra              7 passed
```
