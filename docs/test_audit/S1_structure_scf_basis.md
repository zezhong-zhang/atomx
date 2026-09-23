# Slice S1 — structure / SCF / basis (test audit)

Rubric: `docs/test_audit/RUBRIC.md`. Slice files: the 31 listed, all present. Two
files were **added** because they are plainly structure/SCF gates and were not on
the list: `tests/test_atom_numerics_exchange.py` (engine × exchange matrix, α_x
default, engine-agreement gates) and `tests/test_atom_surface_equivalence.py`
(Atom-surface SCF totals pinned at `abs=1e-12`). Files inspected for
cross-file pin consistency but out of slice: `test_atom_numerics_exchange.py`'s
sibling `test_averaging_decoupling.py`, `test_fac_structure_snapshot.py`,
`test_mbpt2.py`, `test_first_principles_invariants.py`.

## 1. Verdict

This slice contains the best and the worst of the suite, and they are easy to
tell apart only by reading the assertion in isolation. The genuinely
first-principles core is strong and worth imitating: the exact Dirac–Coulomb
closed form is re-derived inside `test_dbound_heavy_z.py`,
`test_silent_failure_guards.py`, `test_bspline_grid_accuracy.py` and
`test_dirac_continuum_shooting.py` (32 + 4 + 2 + 5 assertions at 1e-6…1e-2 of an
independently written formula), the Latter asymptote `r·V → −(Z−N+1)` is pinned
at `atol=1e-9` in `test_fac_asymptotic.py`, the B-spline algebra theorems
(partition of unity, Gram symmetry, positive definiteness) are pinned exactly in
`test_bspline.py`, and one external Fortran oracle is used honestly
(`benchmarks/results_shooting.csv`'s DHFS.f column, quoted in
`test_slater_integral_slag6.py`, gate 5e-8). Against that, the *flagship* SCF
gates are D-class literals that atomx printed itself — the repo even labels them
so in `docs/VERIFICATION.md:25` ("atomx values pinned as regression") — and the
tolerances around them are sized to whatever the code happened to produce.
Measured slack is routinely 10²–10⁶×: Ne Slater bspline-vs-shooting agrees to
1.05e-4 Ha but is gated at 0.129 Ha (`test_dfsolver_slater.py:87`); bspline-gu
vs shooting-gu Fe XVII agrees to 1.2e-4 Ha but is gated at 5.73 Ha
(`test_atom_numerics_exchange.py:142`); the two Slater R^k kernels agree to
2e-14 relative but are gated at 5e-2 (`test_slater_integrator.py:66`). Two
gates are outright non-gates: `test_continuum_knots.py:59` claims "high k_max →
more inner knots" while allowing 30 % *fewer* (measured ratio 1.141), and
`test_potentials.py:123` claims the TFDA start accelerates convergence with
`<=` while its own comment records 12 iterations both ways. One comparison mixes
physics models: `test_fac_dfs_shooting.py` runs atomx with `corrections="qed"`
against a calibration panel generated with `SetSE(0)/SetVP(0)/SetMS(0,0)/SetBreit(0)`
(`extract_AHX.py:56-59`), so up to 960 eV (Z=92) of the residual is a flag
mismatch, and the gate was re-tuned in the same commit that flipped the flag.
One assertion passes by a single ULP: `test_scf_potential.py:81` bounds
`V_hx(10)` above by `−0.2`, which is exactly the Latter cap the code applies
(measured `−0.19999999999999996`), while the same file's Hartree twin returns
the physically correct `−0.1` — the band contains both answers.

Class mix over 259 test functions in 33 files:
**A 66 · B 10 · C 27 · D 20 · E 47 · F 89** (A 25 %, B 4 %, C 10 %, D 8 %,
E 18 %, F 34 %). Of the 47 E's, ~15 are sign/shape/NaN guards of a real
silent-failure mode and should be kept with a comment saying so; the rest are
bookkeeping identities, one-line re-implementations and shape checks.

## 2. File table

Counts are test **functions** (parametrisations noted in the note column).

| file | tests | class mix | worst | note |
|---|---|---|---|---|
| test_scf.py | 16 | A3 B0 C0 D4 E6 F3 | S2 | Docstring cites "Desclaux 1973 DHF: Ne −128.547" (the NR-HF limit); the pin requires −128.6920±1e-3, so the cited reference fails by 0.145 Ha (145× the tolerance). He/Ne pins are atomx literals (margin 1.3e-5 / 3.1e-5 Ha), unmarked. |
| test_scf_potential.py | 9 | A7 B0 C0 D0 E1 F1 | **S1** | Bands (±100 %, [0.5,1.5] ratio, <0.05) on quantities that are analytically −1/r, →1 and 1.2e-6; `V_hx(10)` passes 1 ULP under a Latter cap of −2/r while its Hartree twin returns −1/r. |
| test_scf_nan_sentinel.py | 12 | A0 B0 C2 D0 E0 F10 | S4 | Good HYP-1590 tripwires; the only byte-exact guard-page oracle is `skipif(not linux)` and is **skipped on this host** (confirmed). |
| test_potentials.py | 11 | A7 B0 C0 D0 E3 F1 | **S1** | Latter-tail/sub-1e tests are exact (atol 1e-8, rtol=0); but `test_starting_potential_tfda` cannot fail (12 vs 12 iterations, `<=`), and the TF/GSZ asymptotes are gated at 5–50 % where the model is exact. |
| test_shooting_hf.py | 8 | A0 B0 C4 D0 E1 F3 | S2 | Real cross-engine check, but gates 1e-4/5e-4 vs measured 1.06e-10…5.7e-8 total and ≤3.9e-5 orbital (10²–10⁶ slack); both engines share the jj-average angular coefficients, so an angular error cancels. |
| test_shooting_guerra.py | 6 | A1 B0 C2 D1 E0 F2 | S3 | `test_no_jj_configs_bit_identical` is a self-pin at rel=1e-13 (relaxed from `==` in a894c55 with an in-test justification + same-process bit check); name still says "bit identical", no `snapshot` marker. |
| test_shooting_injection.py | 4 | A0 B0 C0 D0 E1 F3 | S4 | Injection wiring only; `!=` for "changes physics". |
| test_shooting_corrections.py | 7 (19 items) | A1 B0 C2 D0 E0 F4 | S4 | Honest gate sizing (gate 2e-3, measured worst 2.28e-4 reproduced here); `E_QED_total == E_SE+E_VP` and `total == bare+Δ` are bookkeeping. |
| test_dfsolver_slater.py | 8 | A0 B0 C3 D0 E1 F4 | **S1** | The headline gate is 1e-3 **relative** = 0.129 Ha (Ne) / 0.529 Ha (Ar); measured agreement 1.05e-4 Ha / 1.5e-5 Ha. The Salvat DHFS tables reach ~1e-4 Ha absolute on this quantity. |
| test_dfsolver_fac.py | 4 | A0 B0 C0 D0 E2 F2 | S4 | `expected = 0.51465105·(4π²/3)^(1/3)` is the source's own line (dirac_fock.py:656); the literal equals `get_hxs(10,10)` to 1e-11, so the FAC provenance is not exercised, only re-copied. |
| test_dfs_frozen_orbitals.py | 3 (6 items) | A1 B2 C0 D0 E0 F0 | S2 | FAC 1.1.5 numbers are quoted (good B) but at `rel=5e-3`: ±1.39 Ha on the 1s where the measured difference is 0.0365 Ha; the 3d5/2 virtual passes at 3.58e-3 vs a 5e-3 gate (1.4× margin). `z_asym == 17.0, abs=0.2` where the value is exactly 17.0. |
| test_dirac_hydrogenic.py | 16 | A4 B0 C7 D0 E4 F1 | S3 | Only `test_hydrogen_1s` (−0.50000665, rel 1e-6) pins the closed form externally; 7 tests compare the B-spline solver to atomx's own `hydrogenic_dirac_energy` — advertised in `VERIFICATION.md:28` as "known-answer, literature". |
| test_dbound_heavy_z.py | 3 (35 items) | A2 B1 C0 D0 E0 F0 | S2 | Model file: exact Dirac formula re-derived in the test at 1e-6 over 32 states. The sfac end-to-end gate is 1e-3 relative (301 eV at Z=74) with quoted eV literals. |
| test_exchange_model_is_honoured.py | 4 | A0 B0 C0 D0 E1 F3 | S4 | Refusal contracts; the single-config check is `!=` only (the 0.0685 Ha difference it found is quoted in the docstring, not asserted). |
| test_local_exchange.py | 17 | A6 B4 C0 D0 E7 F0 | S4 | Real B core: four FAC `AHX` values (out.pot) quoted to 1e-7 and the analytic LDA constant; the rest are clips/fallbacks and one blend test that uses the function's own endpoints. |
| test_silent_failure_guards.py | 8 (11 items) | A4 B0 C0 D0 E0 F4 | S3 | Exact Dirac H Rydberg 5s–8s at 1e-6 (excellent); but the sea guard is `e > −0.5·m_ec²` (half the rest energy) and `_resolve_r_max` pins the heuristic `5.0*36`, 75.0, 123.0 with no principle. |
| test_bspline.py | 7 | A7 B0 C0 D0 E0 F0 | — | All first-principles: clamped knot vector, partition of unity (1e-14), non-negativity, Gram symmetry (1e-14), positive definiteness. |
| test_bspline_grid_accuracy.py | 6 (14 items) | A2 B0 C1 D2 E1 F0 | **S1** | `test_error_grows_strongly_with_z` (:88) never calls atomx — it asserts the test's own `MEASURED` dict is monotone. The same class pins a known 97.7 eV (Z=92) default-grid error with a 25 % band. |
| test_bspline_local_potential.py | 16 | A3 B0 C0 D0 E4 F9 | S3 | Three tests re-run the wrapper's own two lines (the wrapper *is* `radial_density_from_orbitals → local_central_potential_from_radial_density`) and one asserts `delta_v ≡ 0` for a −Z/r table; only the density normalisation and Σq tests carry physics. |
| test_bspline_radial_phase.py | 2 | A0 B0 C0 D0 E0 F2 | S4 | Phase-convention plumbing. |
| test_slater_integral_slag6.py | 5 (21 items) | A0 B1 C2 D0 E0 F2 | S2 | The only true Fortran-DHFS gate in the slice (quoted from `results_shooting.csv`, 3 atoms, 5e-8). Z=82 reference carries 3 decimals while the gate is 1.05e-3 Ha; docstring says "bit-for-bit" for a 2.2e-8 residual. |
| test_slater_integrator.py | 5 (8 items) | A0 B0 C2 D0 E1 F2 | **S1** | Poisson-vs-direct gated at rel=5e-2 (Auger 1e-1) with a comment claiming a "~1e-3 rank-dependent residual"; measured 2.2e-14…4.7e-14 for k=0…3. |
| test_rkb_bank_sharing.py | 6 | A0 B0 C0 D0 E0 F6 | S4 | Pure cache/performance; correct and useful, no physics. |
| test_dirac_continuum_shooting.py | 8 (12 items) | A5 B0 C0 D0 E2 F1 | S3 | `TestCoulombPhaseShift` (δ=0 for a pure Coulomb field, box-independent) and the √(2/πk) envelope are model tests; `phase_shift_finite` and shape checks are noise. |
| test_continuum_knots.py | 6 | A2 B0 C0 D1 E1 F2 | **S1** | :59 asserts "more inner-region knots" as `ns_cont ≥ 0.7·ns_phys` (measured 1.141 → a 30 % *loss* passes); the spacing test is guarded by `if len(inner_knots) >= 3`. |
| test_density_tail.py | 3 | A2 B0 C0 D0 E0 F1 | S4 | Clean-tail identity and constant `d lnρ/dr` to 1e-12; third test is payload plumbing. |
| test_ladder_and_sampling.py | 20 | A2 B0 C0 D0 E5 F13 | **S1** | `test_climb_total_energies_decrease_monotonically_on_he` (:167) asserts only `E<0` and finite — the ladder's physics claim is ungated; the density sum rules (∫ρ=N) are the A content. |
| test_determinism.py | 2 | A0 B0 C0 D0 E0 F2 | S4 | Process-level determinism; the bounds-check one is `slow` (nightly only, not skipped locally). |
| test_fac_dfs_shooting.py | 3 (15 items) | A0 B1 C0 D0 E1 F1 | **S1** | Reference is QED-free, run is `corrections="qed"`: residual is 1.02e-4 (Z=10), 3.85e-4 (Z=36), 1.005e-3 (Z=74), 9.99e-4 (Z=92) gated at 1e-3/2e-3; with matched flags 1.2e-5/9.3e-6/2.2e-4/9.4e-4. |
| test_fac_asymptotic.py | 9 | A7 B0 C0 D0 E2 F0 | S4 | Latter/envelope tests are exact (atol 1e-9, interior no-op); `test_disabled_envelope_when_N_too_small` asserts a clamp identity that holds either way. |
| test_fac_branch_inventory.py | 7 | A0 B0 C0 D5 E1 F1 | S3 | Pins an inventory script's counts, including 210/330 rows "without production FAC angular coefficient" — documentation of unimplemented physics, unmarked, no atomx solve. |
| **test_atom_numerics_exchange.py** (added) | 14 (26 items) | A0 B1 C2 D3 E2 F6 | **S1** | :142 gates "same physics, different engines" at rel<5e-3 (5.73 Ha); measured 1.2e-4 Ha. `-1148.4141858935911` pinned exactly; α_x=1.5 default *is* gated here (:186). |
| **test_atom_surface_equivalence.py** (added) | 4 (11 items) | A0 B0 C0 D4 E0 F0 | S3 | 11 atomx literals at abs=1e-12, including a row the file admits is **not converged** (`fe17_slater_bspline`); honest D, unmarked. |

## 3. Findings

**1. S1 — `test_fac_dfs_shooting.py:103-138` compares two different physics
models and sizes its gate to the mismatch.** The reference `E0_eV` column of
`docs/plans/2026-04-20-fac-calibration/calibration.csv` was produced by
`extract_AHX.py:56-59` with `SetSE(0)`, `SetVP(0)`, `SetMS(0,0)`, `SetBreit(0)`
("pure local exchange, exactly the physics we're porting"), but the test builds
`Atom(..., corrections="qed")` (:121) on the stated grounds that "FAC's default
structure run has QED on and Breit OFF" — true of the *other* FAC panel
(`results_fac_structure.csv`, used by `test_fac_structure_snapshot.py:63`), not
of this one. Measured residual (`|ΔE|/|E|`): Z=10 1.016e-4 (qed on) vs 1.212e-5
(off); Z=36 3.846e-4 vs **9.260e-6**; Z=74 1.005e-3 vs 2.225e-4; Z=92 9.991e-4
vs 9.443e-4. The atomx QED shift itself is 0.31/28.0/370.0/960.7 eV — the same
size as the gate (1.0e-3×494345 = 494 eV at Z=92). The flags and the gate were
changed in one commit (f77be88: `5.0e-3 → 2.0e-3`, `numerics="shooting"` →
`corrections="qed"`), and the module docstring (:16-23) still claims "worst
4.10e-3 at Z=36" while the code comment says "worst 1.01e-3 (Z=74)".
*Remediation:* use `corrections="none"` (the reference's actual operators) and
set the gate from the matched-flag measurement — 1e-5 for Z ≤ 10, 5e-4 for the
rest — then the panel tests atomx rather than the flag mismatch. Delete the
stale `4.10e-3` paragraph and fix `docs/VERIFICATION.md:39`.

**2. S1 — `test_scf_potential.py:76-81` passes by one ULP and its band contains
both the right and the wrong asymptote.** With `exclude_label="1s1/2"` on He the
residual core holds one electron, so `V(r→∞) = −(Z−1)/r = −1/r`; the
pure-Hartree twin returns exactly that (measured `V_h(10) = −0.09999881`), while
the FAC-exchange path applies the **Latter cap with `N_eff` in the neutral-atom
convention** (`scf_potential.py:275`: `rvinf = −(1.0 + Z − N_eff) = −2`) and
returns `V_hx(10) = −0.19999999999999996` — i.e. the escaping electron sees a
2+ ion. The assertion `−0.2 < V_hx(10) < −0.05` passes only because
`scf_potential.py:288-291` nudges the capped value by one ULP toward zero
(`np.nextafter`); without that nudge the test fails. The band spans the physical
answer (−0.1) and the capped one (−0.2), so it gates nothing.
*Remediation:* assert the asymptote against the residual charge,
`V(r)·r == −(Z − N_core) = −1` to 1e-6 for `r ≥ 20`, and either drop the `+1`
in `rvinf` on the exchange branch or document that the two branches return
different asymptotic charges.

**3. S1 — `test_dfsolver_slater.py:82-104` gates the flagship Slater SCF at
1e-3 relative.** That is 0.129 Ha (3.5 eV) on Ne and 0.529 Ha on Ar. Measured
bspline-vs-shooting agreement on the same quantity: Ne 1.049e-4 Ha (rel
8.16e-7), Ar 1.538e-5 Ha (rel 2.91e-8) — 1226× and 34000× inside the gate. The
independent reference for the same physics is never used here: Salvat's
`radial/DHFS` tables reproduce atomx `slater` at α=1.5 to ~1e-4 Ha (their
density deviation is 6e-6…1e-4, `docs/VERIFICATION.md:263`), and FAC/sfac
agrees with atomx to 1.5e-4…6.0e-4 (`docs/VERIFICATION.md:262`). The docstring's
own bug history is a 2× error (−225 vs −128.6 Ha), i.e. the gate only catches
catastrophes.
*Remediation:* gate `|E_bspline − E_shooting| < 2e-4 Ha` (≈2× the measured
1.05e-4) and, better, add the Salvat DHFS.f row for Ne with α=1.5 at
`abs=1e-4 Ha` — the external number that makes this a B-class gate.

**4. S1 — `test_atom_numerics_exchange.py:131-142` gates "same physics,
different numerics" at 5.73 Ha.** `rel < 5e-3` on Fe XVII gu (−1146.7 Ha) =
5.73 Ha = 156 eV; measured bspline-vs-shooting = 1.204e-4 Ha (rel 1.050e-7),
47600× inside the gate. A sibling gate pins a *non-converged* bspline gu energy
(`test_atom_surface_equivalence.py`, `NON_CONVERGED` set), so the two engines are
known not to solve the same equations to machine precision yet.
*Remediation:* gate at 1e-5 relative (≈100× the measured) and state which
kernel is shared (the α_xc table) and which is not (Galerkin vs shooting).

**5. S1 — `test_bspline_grid_accuracy.py:88-92` never runs atomx.**
`test_error_grows_strongly_with_z` builds `errs` from the test's own `MEASURED`
literal dict and asserts it is monotone and that `errs[-1]/errs[3] > 100`; it
cannot fail for a physics reason, and it duplicates the physics already covered
by the parametrised profile test above it. The profile test itself pins a
**known 97.7 eV (Z=92) default-grid 1s error** with a 25 % band (:82), i.e. the
suite's highest-Z accuracy failure is a frozen expectation.
*Remediation:* delete the tautology; convert the profile into a bound
(`err(Z) < measured(Z)·1.25`) kept as a `snapshot`-marked xfail-document, and
gate the *fix* (already done by `test_grid_hi_refines_the_k_shell`, which
compares against the exact Dirac eigenvalue — promote that pattern to the
default grid, e.g. `err(Z=92, default) < 100 eV` with the review's reference).

**6. S2 — `test_scf.py:1-8` cites the wrong number for its own assertion.** The
module docstring gives "Reference values (Desclaux 1973, Dirac-Hartree-Fock):
… Ne total energy: −128.547 Hartree". −128.5471 is the **non-relativistic HF
limit**, not DHF; the assertion at :120 requires −128.6920 ± 1e-3, so the
docstring's cited reference fails the test by 0.145 Ha (145× the tolerance).
Measured atomx Ne DHF = −128.69196949 (margin 3.05e-5 Ha); He = −2.86181334
(margin 1.33e-5 Ha) against a docstring value of −2.862.
*Remediation:* put the real external reference in the docstring (Desclaux 1973
Ne DHF −128.6919, He −2.86168) and quote the source; keep the atomx literal but
mark the D-class files `@pytest.mark.snapshot`.

**7. S2 — the same Ne number is pinned four ways in four files with no
cross-reference.** −128.69197 (point nucleus, HF, `test_scf.py:120`),
−128.6299783 (Fermi nucleus, HF, `test_atom_surface_equivalence.py:32/105`),
−128.623390 (point, Slater α=1.5, shooting, `test_slater_integral_slag6.py:146`),
−128.623437 (point, Slater, bspline, quoted in
`test_exchange_model_is_honoured.py:9`). A reader cannot tell which physics each
is; the rubric's motivation #1 is literally this row.
*Remediation:* one module-level reference table (`tests/references.py`) naming
nucleus/exchange/engine/source per value, imported by all four files.

**8. S1 — `test_slater_integrator.py:56-66` (and :104) gate two R^k kernels at
5e-2 (10 % for the Auger rate) with a comment that is not true.** The comment
claims "the residual is the direct method's diagonal (r₁≈r₂)
product-quadrature error … here 1s×2p … grows mildly with rank k". Measured on
the test's own densities: k=0 rel 2.19e-14, k=1 4.68e-14, k=2 2.72e-14,
k=3 3.06e-14 — machine precision, no diagonal error at all. A 5 % gate on a
1e-14 agreement cannot see a wrong R^k.
*Remediation:* tighten to `rel=1e-10`, delete the false comment, and add the
first-principles value for one case (for hydrogenic 1s, `R⁰(1s,1s) = F⁰(1s,1s) =
5Z/8`, Slater–Condon; for 1s–2p use the tabulated hydrogenic `R¹`) so the kernels
are compared to an external number rather than to each other.

**9. S2 — `test_dfs_frozen_orbitals.py:42-68` uses 0.5 % on FAC values whose
measured agreement is 1.3e-4…3.6e-3.** Gates: ±1.39 Ha on the 1s (measured
difference 0.0365 Ha), ±0.249 Ha on 2s (0.0874 Ha), and the 3d5/2 virtual passes
at rel 3.576e-3 against a 5e-3 gate — a 1.4× margin, i.e. the gate is as loose
as it can be while still passing. `z_asym == approx(17.0, abs=0.2)` (:56) where
the returned value is exactly 17.000000.
*Remediation:* tier the gates by the measured numbers (1s 5e-4, 2s/2p 5e-3 is
acceptable *if* the 3d5/2 discrepancy is explained as model difference, not
noise); assert `z_asym == 17.0` to 1e-9 since the Latter cap makes it exact.

**10. S1 — `test_continuum_knots.py:51-59` gates the opposite of its name.**
"High k_max → more inner-region knots than the physics default" is asserted as
`ns_cont >= ns_phys * 0.7`; measured `ns_phys = 78`, `ns_cont(k_max=50) = 89`
(ratio 1.141), so a 30 % *coarsening* passes. `test_low_k_max_close_to_physics_grid`
(:48) pins `|ns_cont − ns_phys| < 5` with measured difference 0.
*Remediation:* assert the inner-region knot count directly (`#knots with
r < 5/Z` for k_max=50 ≥ 3× that of the physics grid) or `ns_cont > ns_phys`,
and drop the `if len(inner_knots) >= 3` guard that can silently void (:34).

**11. S1 — `test_potentials.py:104-125` cannot fail for the claim it names.**
"The TFDA starting potential should accelerate HF convergence" is asserted as
`ne2.iterations <= ne1.iterations`, and the comment records the measurement:
"12 iterations both ways, so the claim 'not worse' holds with equality". A
starting potential that is silently ignored passes.
*Remediation:* assert the first-iteration energy residual (or iteration count)
strictly improves, `ne2.iterations < ne1.iterations`, and if the measurement is
12 vs 12 then the claim in the docstring must be dropped to "no worse" and the
test renamed.

**12. S2 — `test_bspline_local_potential.py:398-477` and :356-395 are
C-degenerate.** `fac_local_central_potential_from_orbitals` *is*
`radial_density_from_orbitals(...)` followed by
`local_central_potential_from_radial_density(...)`
(`src/atomx/numerics/bspline/local_potential.py:294-303`), and the test performs
the same two calls and compares (:437, :477); the matrix test re-runs
`_assemble_integral` with the source's own `grw*delta_v` and 0.5(A+Aᵀ)
expressions (:383-395). `test_local_potential_solver_matches_nuclear_only_dirac_matrix`
(:17-54) feeds an exact `−Z/r` table so `delta_v ≡ 0` and re-runs the same
`build_dirac_matrices`/`solve_dirac` kernel — it can only detect an indexing
change, at 1e-10.
*Remediation:* keep them as API-contract tests but label them so, or replace the
physics with an independent number: the FAC `V_HF` dump on the same grid, or the
analytic hydrogenic 1s energy in the `−Z/r` case (`−Z²/2` to 1e-9).

**13. S2 — `test_scf_potential.py` bands on analytically exact quantities.**
`V_HF(20)` for neutral He is 1.2278e-6 and for Ar 1.1464e-4, gated at `< 0.05`
(4e4× and 436× slack, :55, :109); `V_excl(10)` is exactly −1/r = −0.1, gated at
`−0.2 < V < −0.05` (±100 %, :66); the Ne inner ratio `V_HF(0.01)/V_bare(0.01)`
is 0.75993 — 24 % away from the "no screening at r→0" premise — and is gated at
`0.5 < ratio < 1.5` (:96). The same measurement shows `V_HF(0.005) ==
V_HF(0.01) == −759.92646191`, i.e. below the first quadrature point the returned
potential is a constant, so the "r → 0" claim is tested in a region where the
interpolator does not vary.
*Remediation:* assert the exact asymptotes — `|V_HF(20)·20| < 1e-4` for the
neutral cases, `V_excl(10) == −1/10` to 1e-6, and for the inner region the
correct statement `V_HF(r)·r → −Z` as `r → 0` measured at r ≥ the first grid
point, plus a test that the tabulation is not clamped below it.

**14. S3 — D-class pins are unmarked across the slice.** `@pytest.mark.snapshot`
is used by exactly one file in the whole repo (`test_fac_parity_gates.py`), yet
this slice contains ~20 D-class functions: `test_scf.py` (4 energy literals),
`test_shooting_guerra.py` (:75/:80, rel 1e-13), `test_atom_surface_equivalence.py`
(11 literals at `abs=1e-12`, one of them a non-converged SCF energy),
`test_atom_numerics_exchange.py:79` (`-1148.4141858935911`), and the
`MEASURED` error profile. `docs/VERIFICATION.md:25` already describes the SCF
files as "atomx values pinned as regression", so the classification is known —
it just is not visible to a reader of the tests or to `pytest -m snapshot`.
*Remediation:* add `@pytest.mark.snapshot` to every literal-pin function; the
marker's documented meaning ("asserts nothing external") is exactly right.

**15. S3 — `test_dirac_hydrogenic.py` is advertised as a literature gate but 7
of its 16 tests compare atomx to atomx.** `docs/VERIFICATION.md:28` lists it
under "Hydrogenic/Sommerfeld energies … known-answer, literature". In the file,
only `test_hydrogen_1s` pins a value not computed by atomx (−0.50000665,
rel=1e-6); `test_hydrogen_1s_bspline`, `2s_bspline`, `2p32_bspline`,
`he_plus_1s`, `fe26_1s` and the two `TestKineticBalance` hydrogenic comparisons
(`test_h_1s_accuracy`, `test_he_plus_1s`) all use `hydrogenic_dirac_energy(...)`
— the same package's closed form — as the reference. A scale error in that
function that
happens to preserve `n=1, κ=−1, Z=1` (e.g. a wrong `n_eff` for n>1) would pass.
*Remediation:* pin 2s, 2p1/2, 2p3/2, 3s for Z=1 and 1s for Z=26, 92 to quoted
values of the Sommerfeld formula (e.g. H 2p3/2 = −0.1250004160, H 2s1/2 =
−0.1250020802, H 3s1/2 = −0.0555562952, U91+ 1s = −4861.1979 Ha — all recomputed
here from `γ = √(κ²−(Zα)²)`), not to `hydrogenic_dirac_energy`.

**16. S4/E — `test_ladder_and_sampling.py:167-176` is misnamed.**
`test_climb_total_energies_decrease_monotonically_on_he` checks `E < 0` and
`isfinite` for each rung; the comment explicitly abandons monotonicity. The
ladder's real physics claim (successive rungs approach the exact energy and the
endpoint respects the variational sandwich) is ungated — the file's own
`test_density_native_default_unchanged` shows how to do it with a sum rule.
*Remediation:* assert the rung energies are non-increasing after the first rung
and that the final DiracFock rung satisfies `−2.903724 < E < −2.861680`
(the `test_mbpt2.py` sandwich), then rename.

**17. S4/E — `test_fac_asymptotic.py:134-144` asserts a clamp identity.** "N ≤ 1
→ envelope disabled, only Latter" is asserted as
`out[-1] == approx(min(rv[-1], -5.0))`, which holds whether the envelope ran or
not; the in-test comment is "…(here: nothing clips because rv is always ≤ -5?
Let's check.)". Same class: `test_grid_shape_preserved` (:132).
*Remediation:* compare the N=1 output against a *N=2* run on the same synthetic
`rv` (the envelope must differ where the raw `rv` is too weak), or delete.

**18. S4/E — `test_dfsolver_fac.py:22-50` re-implements the source line.**
`expected = 0.51465105 * (4π²/3)^(1/3)` is character-for-character
`dirac_fock.py:656`; the literal equals `get_hxs(10,10)` to 1e-11 (verified:
0.51465105249, 0.50445120123), so the test pins atomx against atomx. The FAC
provenance (`docs/plans/2026-04-20-fac-calibration/calibration.csv`, `AHX`
column printed by `radial.c:2804`) exists but is not cited in the test.
*Remediation:* cite the FAC `AHX` value and the commit that recorded it, and pin
one α_xc→energy consequence (e.g. Fe XVII gu total) rather than the constant.

**19. S3 — `test_fac_branch_inventory.py` pins unimplemented physics.**
The tests assert counts for rows *without* production FAC angular coefficients:
210/330 for the Fe XVII panel (:80), 12 closed-spectator fallbacks, 6 one-diff
branch-5 fallbacks, 192 branch-9 "pending" two-electron terms (:67), and one test
matches the printed summary text (:89-96). These are D-class script snapshots
that read like validation of the FAC port.
*Remediation:* keep one count as a coverage budget, mark it `snapshot`, and add
the missing gate the script implies: an end-to-end Fe XVII level energy against
live `sfac` for the branches that *are* production (`fac-angular`).

**20. S4 — platform-conditional and marker-conditional gates.**
`test_scf_nan_sentinel.py:251` (`guard-page` OOB oracle for the HYP-1590 kernel)
is `skipif(not linux)` and **is skipped on this host** (pytest: "SKIPPED …
requires linux mmap/mprotect"); the identical-array reference test
(`:224-248`) does run and is the only other oracle for that fix.
`test_determinism.py:47` is `slow`, so the numba bounds-check gate runs only in
`nightly.yml` (the default CI job is `pytest -m "not slow"`, test.yml:58); the
CI does run it in the nightly job, so this is a visibility issue, not a dead
test. `test_fac_dfs_shooting.py:64-70` returns `[]` when the reference CSV is
missing — now guarded by `test_calibration_reference_present_and_spans_z2_to_z92`
(:90), which is the right pattern (it self-skipped silently until a894c55).
*Remediation:* none required for the two marker cases; port the guard-page trick
to a portable check (allocate the density in a 1-page `mmap` and compare against
the pure-Python reference with a sentinel row of NaNs past the end), or add a
Linux-only marker to the docstring so the local gap is visible.

## 4. Model tests (imitate these)

1. `test_dbound_heavy_z.py:35-126` — the exact Dirac–Coulomb eigenvalue is
   *written out in the test* (`γ = √(κ²−(Zα)²)`, `N = √((n−|κ|+γ)²+(Zα)²)`) and
   compared at `rel < 1e-6` for 32 (Z, n, κ) combinations up to Z=100, from a
   deliberately bad guess (−0.1 Ha). This is what "first-principles" means: the
   test would fail if the source and the formula were wrong together.
2. `test_silent_failure_guards.py:32-36 + :89-123` — same pattern (H 7s, 5s–8s at
   1e-6), attached to a real silent-failure mode (the `−Z/r_max` bracketing
   ceiling).
3. `test_dirac_continuum_shooting.py:105-163` — `TestCoulombPhaseShift`: for a
   pure `−Z/r` potential the extracted phase shift *must* be zero once the fit
   includes `−η ln 2kr` and `σ_l`; four (Z,k,r_max) combinations at 0.05 rad,
   plus box-independence. An exact theorem, named, with the failure mode
   (plane-wave fit) written down.
4. `test_fac_asymptotic.py:32-67` — Latter asymptote `r·V → −(Z−N+1)` at
   `atol=1e-9` for neutral and ionic cases, **and** a no-op assertion in the
   interior; the model's exact content, no SCF needed.
5. `test_potentials.py:33-58` — the TFDA tail at `atol=1e-8` and the
   sub-one-electron case at `rtol=atol=0` (bit-exact bare nuclear). Exact-limit
   gates with zero slack.
6. `test_bspline.py:39-85` — partition of unity to 1e-14, non-negativity,
   `S == Sᵀ` to 1e-14, positive definiteness. Basis theorems, independent of any
   physics.
7. `test_slater_integral_slag6.py:141-176` — the only place in the slice where an
   external Fortran code's number is quoted in the test (`E_radial_Eh` from
   `dhfs.dat`, reproduced in `benchmarks/results_shooting.csv`); 5e-8 relative
   for three atoms. The pattern to copy (just fix the "bit-for-bit" wording and
   the 3-decimal Z=82 reference).
8. `test_shooting_corrections.py:132-142` — an honest cross-engine gate: the
   docstring states the measured slack (≤2.3e-4 Breit, ≤4e-6 SE, ≤2e-5 VP), the
   gate is 2e-3, and I reproduced 2.28e-4 worst. Gate sizing with a stated
   measurement is the discipline the rest of the slice lacks.
9. `test_local_exchange.py:44-63` — FAC `AHX` values (Z=26 N=10 → 0.50445120;
   Z=10 N=10 → 0.51465105; Z=82 N=18 → 0.37229885) quoted to 1e-7, i.e. the
   external oracle for the whole Gu α_xc port. (The surrounding clip/fallback
   tests are noise; keep the four numbers.)
10. `test_ladder_and_sampling.py:245-250` and `:234-242` — `∫ρ = N` (10.0 e)
   at 1e-3 native / 0.2 on a log grid; a sum rule, with the log-grid slack
    explained rather than hidden.
11. `test_density_tail.py:35-47` — the repaired tail must be a pure exponential:
   `std(d lnρ/dr) < 1e-12` over the noisy segment, plus monotone decrease and
   positivity. An exact property of the replacement model.
