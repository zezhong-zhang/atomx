# Test-suite physics audit — synthesis

This directory holds a full audit of the `atomx` test suite (165 files,
~33 000 lines, 2 326 collected tests) against a single question: **would this
test fail if the physics were wrong?**

It exists because the answer was, for a large part of the suite, no. Two
concrete symptoms started it:

1. The same physical quantity is pinned to four different numbers in four
   different files — Ne appears as `-128.6299783505843`, `-128.6920`,
   `-128.623437` and `-128.623390000` — and nothing at any of those sites says
   which nucleus model and which exchange model each belongs to. A reader
   cannot tell whether the four agree or disagree.
2. `assert he_plus.total_energy == pytest.approx(-2.0, abs=0.5)` is a 25 %
   band on a hydrogenic eigenvalue known in closed form to twelve digits.

[S1](S1_structure_scf_basis.md) [S2](S2_ci_mcdhf_correlation.md)
[S3](S3_qed_breit.md) [S4](S4_properties_transitions.md)
[S5](S5_scattering_continuum.md) [S6](S6_plasma_kinetics.md)
[S7](S7_api_units_harness.md) — slice reports.
[RUBRIC.md](RUBRIC.md) — the classification vocabulary.
`S*_remediation.md` — what was changed, in each slice.

## The vocabulary

Every test is classified as exactly one of:

| class | meaning |
|---|---|
| **A** | first-principles — the expected value is derivable without atomx and without any other atomic code: a closed form, a symmetry, a sum rule, a variational bound, a known limit, a unit identity |
| **B** | external oracle — NIST, DBSR_HF, FAC, GRASP, Salvat, a cited paper table |
| **C** | internal cross-check between two atomx paths — legitimate only if the paths share no kernel |
| **C\*** | **C-degenerate** — the "cross-check" compares two callers of the same kernel, so it is self-regression wearing a disguise |
| **D** | frozen self-regression — a literal only known because atomx printed it earlier; change detection, not validation |
| **E** | tautology — cannot fail for a physics reason (asserts a shape, a mock, its own data structure, or nothing at all) |
| **F** | plumbing — serialisation, CLI, error surfaces, provider contracts; legitimate, but not physics |

A gate is only physics-bearing if it is A or B. C is worth having. D is worth
having *and must be labelled*. E is noise. F is not evidence about physics.

## Scorecard

A+B is the fraction of the suite that could detect a wrong physical result.

The counting unit differs slightly between slices — S4, S6 and S7 counted test
*functions*, S1 and S3 items, S5 parametrised *cases* (because one parametrised
physics test is a different amount of evidence from one plumbing assertion) — so
the "all slices" row aggregates roughly-comparable units rather than being an
exact census. The per-slice figures are the ones to quote.

| slice | files | items | A | B | C (C\*) | D | E | F | **A+B** |
|---|---|---|---|---|---|---|---|---|---|
| S1 structure / SCF / basis | 33 | 259 | 66 | 10 | 27 | 20 | 47 | 89 | **29 %** |
| S2 CI / MCDHF / correlation | 24 | 258 | 71 | 19 | 55 (≈36) | 35 | 25 | 53 | **35 %** |
| S3 QED + Breit | 13 | 164 | 55 | 20 | 21 (17) | 10 | 33 | 25 | **46 %** |
| S4 properties / transitions | 21 | 371 | 171 | 30 | 17 | 24 | 76 | 53 | **54 %** |
| S5 scattering / continuum | 26 (+3 molecular) | 260 (+148) | 94 | 23 | 24 | 7 | 63 | 49 | **45 %** |
| S6 plasma / kinetics / synthesis | 20 | 222 | 69 | 3 | 16 (7) | 36 | 70 | 28 | **32 %** |
| S7 public API / units / harness | 29 | 393 | 64 | 23 | 18 | 23 | 45 | 220 | **22 %** |
| **all slices** | **165** | **1 927** | **590** | **128** | **178** | **155** | **359** | **517** | **37 %** |

Two numbers matter more than the rest:

* **The lowest A+B is the public API slice (22 %)**, and its 220 F-class tests
  are dominated by `test_atom_input_surface.py` — 60 nodes, zero physics, 26 of
  them matches on retired-keyword *error message text*. That file is not
  worthless (it is the clean-break migration's only guard), but it must not be
  counted as coverage of anything physical.
* **A large share of the A/B tests never run.** In S6, 21 of 222 skip at
  runtime and *every published-data gate is among them*; a further 23 sit
  behind `plotly`'s `importorskip`, including 8 genuine angular-normalisation
  gates. In S7, 48 nodes skip because `pint` and `pyscf` are absent from the
  `dev` extra — all 32 `test_units.py` nodes, which are the slice's most
  likely first-principles tests. A gate that does not execute is not a gate.

## Cross-cutting patterns

The same eight failure modes recurred independently in slices audited by
different agents, which is what makes them worth fixing as policy rather than
as 200 individual edits.

**1. Tolerances orders of magnitude looser than the code's own accuracy.**
This is the single most common finding. The gate cannot see the error it
exists to bound.
- S1: Slater SCF total energies gated at 1e-3 relative against a measured
  8.16e-7 — 1 226x slack for Ne, 34 000x for Ar. "Same physics, different
  engines" gated at 5.73 Ha against a measured 1.204e-4 Ha: 47 600x.
- S7: `test_fac_parity_gates.py:69-98` gates at 1e-3 against a measured
  4.264e-9 — 234 523x, i.e. +-445 eV at Z=82, which is *larger than the 426 eV
  atomx-vs-FAC gap the same file records*.
- S4: the length/velocity gauge theorem is gated at 1e-3 while the measured
  residual is 2e-11 — seven to eight orders of slack.
- S6: the UTA-vs-explicit gate is 25 % against a measured 11.6 %.

**2. Gates the bug passes.** A tolerance chosen so wide, or a quantity chosen
so insensitive, that the historical defect sails through.
- S3: `tests/test_qed.py:135` pins the scalar Breit at `approx(0.0504, rel=0.05)`,
  a band of [0.04788, 0.05292] that **excludes the correct value**. DBSR_HF's
  Ne Breit is 0.016643630840490 Ha (verified by running
  `dbsr_hf an=10 term=AV mbreit=1`; log in `/tmp/atomx_n1/dbsr_ne/Ne.log`).
  The test does not merely fail to catch the 3.03x error — it *blocks the fix*.
- S4: `docs/VERIFICATION.md:29` names `tests/test_transitions.py` as the gate
  for the H 2p->1s A coefficient. That file has no such assertion; its
  strongest check is `assert tr.A_rate_per_s > 0`, which the buggy value
  1.253677e9 and the correct 6.268384e8 both pass.
- S6: `test_high_density_approaches_boltzmann` asserts `pop_crm[1] > 0` while
  its own comment admits the Boltzmann comparison is not made. It passes for a
  model with no collisional channel.
- S5: `BORN_TOL = 0.30` is *wider than the floor the Born blend mathematically
  imposes*. At 50-200x threshold the solver forces `w >= 1/sqrt(2)`, so
  `|Omega/Omega_Born - 1| <= 0.293 < 0.30` automatically. The audit's
  counterfactual run, with `Omega_DW` 114x/419x/1571x below the Born value,
  **passes at rel = -0.153**. The gate cannot fail for the reason it was
  written; the underlying `Omega_DW/Omega_Born` is still 0.112/0.034/0.009.
- S7: `test_nist_levels.py:66-86`, the newest external-verification file, has
  exactly one passing physics test and it asserts `max|dE| > 100 cm^-1` — a
  one-sided magnitude threshold with no anchor, green while the same file
  records the scalar Breit as 3.03x wrong.

**3. Circular and fabricated references.** Worse than no gate, because they
launder a number into looking verified.
- S4: `data/hyperfine_presets.py:31-36` holds coefficients "empirically tuned
  to reproduce ... 1420.403 MHz", and `test_hyperfine.py:279` then asserts
  `== 1420.403 +- 0.002` and calls it "<1 ppm from Karshenboim". The test
  asserts what the code was tuned to produce.
- S6: `test_viz_scattering.py:71` plots a "FAC clean-room" reference that is
  the data multiplied by 1.1. `test_crm_W_balance.py:69` documents that
  `atomx_fraction` is a verbatim copy of FAC, so its rms is identically 0 —
  and it is presented as a comparison.
- S3: the comment at `test_qed.py:117-119` attributes +0.05 Ha of Ne Breit to
  "Grant 2007 tables for Ne I". The true Ne Breit is 0.0166 Ha. The
  attribution is fabricated and it is why nobody looked.
- S5: `test_dr_vs_fac.py:88-106`'s "kernel parity with FAC" is a same-kernel
  round-trip — the ratio is 0.99970901 at *all four* temperatures because the
  prefactor cancels — while `test_rr_vs_fac.py` compares against an internal
  hydrogenic chain and never touches the real `sfac` PI/RR panel that sits
  unused in `benchmarks/results_fac_pi_rr.csv`.
- S1: `test_bspline_grid_accuracy.py:88-92` never calls atomx at all; it
  asserts that the test's own `MEASURED` dictionary is monotone.
- S2: the **oracle itself**. `benchmarks/mscheme_ci_reference.py` was
  described in `docs/VERIFICATION.md`, `docs/README.md` and the N1 research
  note as "an independent m-scheme Slater–Condon CI ... which shares no code
  with the repo". It shares the angular kernel:
  `exact_ci.two_body_angular` (`src/atomx/methods/mcdhf/exact_ci.py:609-656`)
  is a line-by-line transcription of `mscheme_ci_reference._angular_terms`
  including the phase exponent; the determinant sign convention is shared by
  design; the CSF patterns come from the repo's own `enumerate_jj_csfs`; and
  `_h_one_body` / `_norm_sq` / `_rk` are shared. The history confirms it: the
  oracle file does not exist at the fix commit `04cd274` (`git ls-tree`) even
  though that commit's tests import it, and its only commit is `2007d36`, a
  descendant of the fix. It gates the *assembly* — patterns, fermion signs,
  CSF orthogonality, the `C† H_det C` reduction — faithfully, but it cannot
  arbitrate the angular algebra, which is the thing its docstrings implied it
  arbitrated. Corrected in all three documents on 2026-09-22.

**4. C-degenerate cross-checks.** Comparing two callers of one kernel and
calling it validation. S3 found 17, S6 found 7 of 16, S4 found
`test_transitions_engine_uniform.py` (`multipole.py:35` states one radial
kernel serves both engines, so its rate agreement gates the two SCF solvers,
not the transition machinery), S1 found `test_bspline_local_potential.py:398-477`
re-implementing the wrapper's own two-line body. These are not worthless — but
they must say what they actually gate.

**5. D-class pins that are not labelled.** `@pytest.mark.snapshot` is
documented in `pyproject.toml` as "frozen atomx SELF-regression against its own
recorded output; asserts nothing external" and is used by **exactly one test in
the entire suite** (`test_fac_parity_gates.py:49`) against 120 unmarked D-class
pins counted across the slices. The marker already exists; it is simply not
used, so the suite's own documentation overstates what is verified.

**6. False precision.** S7 measured the `abs=1e-12` Ne pin in
`test_atom_surface_equivalence.py:99,105` in detail. It is bit-reproducible
along the exact default path, but the method's own convergence error is ~1e-6 Ha
— six orders of magnitude larger — and *tightening the SCF is rejected by the
pin*: `solve(tol=1e-9)` moves it by +7.84e-6, `Basis(r_max=120)` by +1.55e-6,
`Sampling(n_grid=1500)` by -4.67e-6, bspline `ns=113` by -4.54e-5. Worse, the
pinned number contains `E_Breit = 0.0504371` whose error against DBSR_HF is
0.0338 Ha, i.e. 3.4e10 times the stated tolerance, while the correction-free
value -128.69193378732467 *is* the externally verified one and lives in a
different file.

**7. Documentation that points at gates which do not exist, or contradicts
them.** `docs/VERIFICATION.md:29` (H 2p->1s, above), `:39` (still quotes "worst
4.1e-3 at Z=36" while the test gates 2e-3), `:28` (lists
`test_dirac_hydrogenic.py` as a literature known-answer gate, but 7 of its 16
tests used atomx's own `hydrogenic_dirac_energy` as the reference).
`src/atomx/corrections/adapters.py:138` still describes the Uehling potential
with `K1(2cr)` — the wrong Bessel function that was fixed this session — and
`:249` still documents the old, inverted normal-mass-shift sign. A wrong
description is how the bug got written.

**8. Physics claims attached to tests that never execute.** Covered in the
scorecard above; the pattern is that the *published-benchmark* gates are
disproportionately the skipped ones, because they are the ones with an external
data dependency to hang a skip on.

## Policy

Adopted so the suite does not decay back:

1. A new gate states its class in the docstring. If it is not A or B, say so.
2. A D-class pin carries `@pytest.mark.snapshot`. No exceptions.
3. A C-class gate names the two paths and which kernels differ.
4. Every tolerance is justified as a stated multiple of a *measured* residual,
   written next to the assertion ("measured 4.264e-9; gate 1e-8, 2.3x margin").
   A tolerance with no measured number behind it is a guess, and the audit
   shows guesses run 10^3-10^5 too wide.
5. An `xfail` that carries a physics claim is `strict=True` and its reason
   holds the measured number and the named cause. `xfail_strict = true` is set
   repository-wide in `pyproject.toml` so this cannot be forgotten.
6. A test that skips must be visible in the summary line as a skip, never as a
   silent `try: import` inside the body.
7. A reference that is a copy of the data, or a constant the code was tuned to
   reproduce, is removed rather than gated.

## What the audit found that the suite could not

Four defects that had been shipping past a green suite, three of them found
only by taking the suite's own claims seriously:

| defect | measured | would have been caught by |
|---|---|---|
| `build_ci_matrix` never evaluated `<CSF_r|H|CSF_s>` | Be `sd` = -15.0527 Ha, **0.385 Ha below the exact ground state** | the Hylleraas-Undheim-MacDonald bound, now in `tests/test_first_principles_invariants.py` |
| scalar Breit is 3.03x DBSR_HF | 0.0504371 vs 0.016643630840490 Ha | `test_nist_levels.py`, as a strict xfail |
| Uehling kernel used `K_1(2cr)` instead of `I(2cr)` | H 1s VP -844.42 MHz instead of -215.10 MHz | `tests/test_qed_uehling.py` |
| MCDHF on H I is 45x worse than DHF | rms 2 132 vs 47.5 cm^-1 over 9 levels | `test_nist_levels.py`, as a strict xfail |

The first is the important one: an energy 0.385 Ha *below the exact ground
state* is not a subtle error, it is a violation of the variational principle,
and no amount of pinning atomx's own output could have seen it.

### And one the audit itself only half-saw

Chasing the scalar-Breit pin turned up a reporting defect that is arguably
worse than the 3.03x error, because it applies to the number a user reads:

* On the **MCDHF/CI route** the correct operator shift *is* applied
  (+0.017340451403925 Ha, `breit_provenance()` reports `mode: operator`,
  `scalar_shift_applied: False`) — but `energy_components["E_Breit"]` reports
  **+0.050448592992794**, the scalar that was never applied. A reported number
  that is not the number used.
* On the **single-configuration** route (`Atom(10, "1s2 2s2 2p6")`, a *string*
  config) there is no operator at all, so the applied Breit *is* the scalar
  **+0.050437102577** — 3.03x DBSR_HF. Note it is the *string-vs-list spelling*
  that selects this, not the engine — the real discriminator is the **CSF
  count** (`atom.py:824-836 use_ci_avg`): a configuration whose jj CSF space has
  `len(csfs) == 1` (every closed shell) takes DHF + the correction stack and
  gets the scalar, while `len(csfs) > 1` (every open shell) already routes
  through `MCDHFSolver` and gets the operator. So `engine="bspline"` with a
  string config also lands on the scalar path (**+0.050448590840**), and the
  blast radius is exactly the single-CSF configurations — which includes
  `Atom(Z)` with no config, since `ground_state_config()` returns a `str`.

  Measured consequence on the same physical system: the two spellings differ by
  the whole Breit disagreement, **0.033096651 Ha = 0.9007 eV = 7264.6 cm^-1**,
  while their correction-free energies agree to 7.9e-6 Ha. So the *only*
  substantial difference between `Atom(10, "1s2 2s2 2p6")` and
  `Atom(10, ["1s2 2s2 2p6"])` is that one of them has a Breit 3.03x too large.

So the same key meant "a value that was not used" on one engine and "a wrong
value that was used" on the other, and nothing distinguished them. The
external anchor for the operator is solid, incidentally: the MCDHF route's
applied He shift is 6.377915809485657e-05 against my own DBSR_HF run's
`Breit: 0.000063777432919` — **+0.0028 %**; Ne is +4.19 %, the documented
omitted retardation.

**Resolution (2026-09-22): the scalar is deleted.** The decision taken was
that the Gaunt operator becomes the *only* Breit in atomx — the scalar is
removed rather than deprecated or relabelled, and the single-configuration
(string config) route is being moved onto the operator, evaluated on the
orbitals the SCF already produced rather than by re-running the MCDHF SCF
(that alternative costs 4-7x more wall time: Fe goes from 8.6 s to >60 s).
The scalar was a systematic **2.91x / 2.92x** the operator at Z=10/18, and its
own error against DBSR_HF is +0.0028 % for He but 3.03x for Ne — it happened to
be adequate for the two-electron case and failed badly for a real closed shell.

### Root cause of the systematic factor: the omitted `(2L+1)`

The 2.9-3.1x was not random. Measured against DBSR_HF (`mbreit=1`, run by hand
for each element):

| Z | scalar as-is | ÷ (2L+1) on `v = L` | operator (Gaunt only) | DBSR_HF |
|---|---|---|---|---|
| 2 He | 1.9156e-04 (**3.004x**) | 6.3857e-05 (+0.13 %) | 6.3777e-05 (+0.000005 %) | 6.377743e-05 |
| 10 Ne | 5.0437e-02 (**3.030x**) | 1.6585e-02 (-0.35 %) | 1.7340e-02 (+4.15 %) | 1.664363e-02 |
| 18 Ar | 4.1004e-01 (**3.098x**) | 1.3320e-01 (+0.63 %) | 1.4054e-01 (+6.2 %) | 1.323646e-01 |

The signature is textbook: the ratio creeps 3.003 → 3.030 → 3.098 exactly as
`2L+1 = 3, 5, 7` predicts once the higher multipoles gain weight with Z. In the
multipole expansion `P_L(cos θ₁₂) = (4π/(2L+1)) Σ_q Y_L^q(1) Y_L^{q*}(2)` the
`(2L+1)` sits in the denominator; DBSR's `SMU` coefficients are written for the
**unnormalised** `Y_L` contraction, while atomx's radial integrals are built
with the **Racah-normalised** `C^L = √(4π/(2L+1)) Y_L`, which already absorbs
that denominator. Applying the former to the latter multiplies the `v = L`
(Gaunt) channel by `(2L+1)` — 3 for the dominant `L = 1`.

Where the omission physically lives, read from the reference source: DBSR's
caller `hf_energy_BR.f90:29-32` does `if(coef(i,j,k).eq.0.d0) Cycle` then
`S = S * coef(i,j,k)`. **`SMU` itself matches atomx's transcription
bit-for-bit** (`SMU = -(ka+kc)*(kb+kd); SMU=SMU/L/(L+1)`, line 82). What is
missing is `coef(i,j,k)`, the configuration angular coefficient built from the
determinant expansion at `dbsr_lib_zconfjj.f90:4098`. DBSR also loops
`Do k = 0,kmax` (atomx starts at `L = 1`) and uses the transposed configuration
for `i<j` (`k3=kj, k4=ki`).

This settles the disagreement recorded earlier in this file: dividing `v = L` by
`(2L+1)` was rejected as "agreeing with DBSR only by cancelling the missing
retardation", but that argument fails on He — the Gaunt-only operator equals
DBSR there to 5e-7 %, so retardation is zero and there is nothing to cancel,
yet the corrected scalar lands 0.13 % away.

The two defects are independent, which is why the corrected scalar and the
operator **bracket** DBSR: the `v = L` fix removes the factor of ~3, while the
`v ≠ L` channels remain ~50x too small (that is the 0.35 %/0.63 % shortfall at
Ne/Ar), and the Gaunt-only operator omits retardation entirely (the 4-6 %
excess). GRASP's equivalent factor was **not** extracted — `breit.f90` is a
driver over generated `genintbreit1/2` angular integrals and the generator read
carries no explicit factor — so the DBSR source is the only sibling
implementation this was confirmed against.
