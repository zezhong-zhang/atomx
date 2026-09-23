# S7 — public API, units, providers, and the external-verification harness

Slice: 28 listed files + 1 added (`tests/test_ladder_and_sampling.py`) under
`/Users/zhangzz/code/ASAP/atomx/tests/`, 393 test functions, 592 collected
nodes. Every listed file exists. Method: every function body read, the code it
exercises read, `git log -S'<literal>'` + `git log -1` for literal provenance,
and targeted runs of individual files (never the whole suite). Reproducibility
probes were run in fresh processes; the repo was not modified.

**Added file.** `tests/test_ladder_and_sampling.py` (250 lines, 20 tests) is not
in the slice list but is the only coverage for four top-level exports
(`Ladder`, `LadderRun`, `RungOutcome`, `Sampling`); it is included and marked
with `*` in the table.

---

## 1. Verdict

This slice is where "the API works" is routinely mistaken for "the physics is
right". Of 393 test functions, **64 are first-principles (A) and 23 are
external-oracle (B) — 22 % — while 265 (67 %) are plumbing (F) or tautology
(E)**; another 23 (6 %) are frozen self-regression (D). The surface-freeze file
that the migration produced as its main artifact,
`test_atom_input_surface.py`, contains **zero physics in 60 nodes**: 26 of them
are error-message text matches on retired keyword names, and the rest pin the
constructor's routing tables. The genuinely good physics in this slice is real
but concentrated and often un-executed: `test_book_preface.py:35` gates the H 1s
Dirac eigenvalue against the closed-form Sommerfeld formula to 1e-11, and
`test_msub.py`, `test_pyscf_embedding_model.py`, `test_pyscf_adapter.py` and
`test_atom_inputs_wired.py` contain clean A-class gates (selection rules, Evjen
lattice neutrality, `CᵀSC = I`, ERI 8-fold symmetry, plasma first-order theory,
box independence) — but 31 of the slice's A/B functions **never execute in a
plain `pip install -e .[dev]`** because `pint` and `pyscf` are optional extras
absent from the `dev` group: all 32 units tests, 17 of 18 PySCF adapter tests
and the whole PySCF embedding module skip silently and non-strictly. The newest
external-verification gate, `test_nist_levels.py`, is the sharpest illustration
of the whole problem: its four physics gates are correctly `xfail(strict=True)`
and document real defects (C V ¹P–³P 40481.6 vs NIST 28172.3 cm⁻¹; scalar
Breit 3.03× DBSR_HF; H-like MCDHF rms 2132 vs DHF 47.5 cm⁻¹; Ar MBPT(2)
E2 = −1.057787 Ha), but of the two tests that *pass*, one only asserts that
something moved by more than 100 cm⁻¹ and the other parses a cache — so the
harness currently certifies no physics at all. And the 1e-12 total-energy panel
in `test_atom_surface_equivalence.py` is bit-reproducible but physically
meaningless as a precision claim: it pins a number that contains the very
Breit scalar the NIST gate says is 3.03× too large (error 0.0338 Ha =
3.4 × 10¹⁰ × its own tolerance), and it breaks if the SCF is converged more
tightly (+7.8e-6 Ha at `tol=1e-9`) or the radial box is widened (+1.55e-6 Ha at
`r_max=120`).

---

## 2. File table

`defs/nodes` = test functions / collected test nodes. Worst severity is the
highest S-level found in the file.

| file | defs/nodes | A | B | C | D | E | F | worst | note |
|---|---|---|---|---|---|---|---|---|---|
| test_atom_api.py | 12/12 | 2 | 0 | 0 | 3 | 7 | 0 | **S1** | He⁺ `-2.0 ± 0.5 Ha` = ±13.6 eV (25 %); four tests assert only `< 0` |
| test_atom_input_surface.py | 30/60 | 0 | 0 | 0 | 0 | 5 | 25 | S3 | clean-break surface freeze: 60 nodes, 0 physics; 26 are retired-name message matches |
| test_atom_inputs_wired.py | 17/17 | 4 | 0 | 2 | 0 | 3 | 8 | S4 | the 4 wired inputs measured properly; two `!=`-only tests |
| test_atom_surface_equivalence.py | 3/13 | 0 | 0 | 0 | 2 | 0 | 1 | **S1** | 11-value 1e-12 panel incl. a known-3×-wrong Breit; unmarked D |
| test_atom_unified_api.py | 27/31 | 6 | 0 | 0 | 0 | 9 | 12 | S2 | CI-norm occupation identity is good; `<r> = 0.75 ± 0.2`; stale `numerics=`/`configs=` docstring |
| test_atom_numerics_exchange.py | 14/20 | 0 | 0 | 2 | 1 | 0 | 11 | S3 | unmarked Fe XVII bit-identity pin `-1148.4141858935911` |
| test_protocols.py | 7/7 | 0 | 0 | 0 | 0 | 1 | 6 | S4 | duck-typing protocol on a hand-built mock |
| test_units.py | 32/32 | 18 | 1 | 0 | 3 | 3 | 7 | **S2** | **all 32 SKIPPED here** (`pint` absent, non-strict); 20 `approx()` calls, 17 derived |
| test_constants.py | 8/8 | 2 | 4 | 0 | 1 | 0 | 1 | — | best-in-slice: CODATA numbers quoted, `c = 1/α` exact |
| test_data_loaders.py | 6/6 | 0 | 1 | 1 | 3 | 0 | 1 | S3 | FAC table values quoted; 3 pins are "the copy still equals the copy" |
| test_native_provider.py | 5/5 | 0 | 0 | 0 | 0 | 2 | 3 | S4 | mock-as-subject |
| test_scaffolds.py | 3/3 | 0 | 0 | 0 | 0 | 1 | 2 | S4 | scaffold contract |
| test_sos_basis.py | 4/4 | 0 | 0 | 1 | 3 | 0 | 0 | S3 | `eps == result.orbital_energies` is self-consistency (C-degenerate) |
| test_msub.py | 8/8 | 4 | 0 | 0 | 0 | 2 | 2 | — | **model**: axial selection rule, Σw = 1, A₂ = √(1/6) |
| test_book_preface.py | 4/4 | 1 | 1 | 0 | 0 | 2 | 0 | S4 | **model A gate**; the tungsten "claim" is a 40 s smoke test |
| test_nist_levels.py | 6/6 | 0 | 4 | 0 | 0 | 1 | 1 | **S1** | 4 × `xfail(strict=True)`; the 2 passes gate nothing |
| test_ladder_and_sampling.py `*` | 20/20 | 2 | 0 | 0 | 0 | 2 | 16 | S4 | added; `test_climb_total_energies_decrease_monotonically_on_he` asserts no monotonicity |
| test_providers_contract.py | 92/160 | 9 | 2 | 7 | 0 | 3 | 71 | **S1** | 142/249 nodes assert only "non-finite raises"; tolerances copy the module constants |
| test_qa_provider_finite_contract.py | 24/89 | 4 | 0 | 0 | 2 | 0 | 18 | **S1** | 2 non-strict xfails; conjugation test is a model sanity-first pattern |
| test_pyscf_adapter.py | 18/18 | 4 | 0 | 0 | 0 | 2 | 12 | **S2** | 17/18 SKIPPED (no `pyscf`); `CᵀSC=I`, 8-fold ERI, MOM argmax are the real gates |
| test_pyscf_embedding.py | 14/0 | 5 | 0 | 0 | 0 | 0 | 9 | **S2** | whole module skipped at import; zero-scale limit + linearity are A-class |
| test_pyscf_embedding_model.py | 9/9 | 3 | 0 | 0 | 0 | 0 | 6 | — | **model**: Evjen neutrality, Mg₁₃O₁₄ arithmetic |
| test_fac_runner_parsers.py | 3/3 | 0 | 0 | 0 | 0 | 0 | 3 | S4 | sfac path/error plumbing |
| test_fac_io_roundtrip.py | 10/10 | 0 | 5 | 5 | 0 | 0 | 0 | **S2** | genuine sfac 1.1.5 dumps (B) + round-trips; the Fe XVII "1e-6 A gate" is a length check |
| test_fac_structure_snapshot.py | 3/22 | 0 | 0 | 0 | 2 | 0 | 1 | **S2** | 21 self-snapshot nodes, **no `snapshot` marker**; 2e-4 = 88.9 eV at Z=82 vs 3.76e-9 measured |
| test_fac_dw_benchmarks.py | 7/7 | 0 | 2 | 0 | 2 | 1 | 2 | **S2** | live-vs-stored-FAC rms gates are the strongest external gates here; self-regression bound 5 % > the 2.85 % it protects |
| test_fac_hamiltonian_parser.py | 2/2 | 0 | 0 | 0 | 0 | 0 | 2 | S3 | parsers have no caller outside the test; `HPAIR` exists nowhere in FAC |
| test_fac_parity_gates.py | 1/1 | 0 | 0 | 0 | 1 | 0 | 0 | **S2** | honestly labelled + `snapshot`-marked, but tolerances 9 × 10³–2 × 10⁵ × loose |
| test_fac_angular_conventions.py | 4/15 | 0 | 3 | 0 | 0 | 1 | 0 | S3 | "FAC reference" 3j/6j literals are textbook values, 10/10 = sympy ≤ 2.2e-16 |
| **total** | **393/592** | **64** | **23** | **18** | **23** | **45** | **220** | | A+B = 22 % of defs, E+F = 67 % |

Executed coverage is worse than the table: 48 nodes (31 of them A/B) skip in
this environment — 32 units, 17 PySCF adapter, 1 PySCF embedding module,
1 `total_energy_as("eV")`, 2 pint-guarded provider nodes. **Executed A+B =
56/393 defs (14 %).**

---

## 3. Findings

### S1

**1. The 1e-12 total-energy pin is false precision on the physics axis and a
fragile change-detector on the numerics axis.**
`tests/test_atom_surface_equivalence.py:99,105` (also :120).
`Atom(10, config="1s2 2s2 2p6").solve().total_energy == approx(-128.6299783505843, abs=1e-12)`.
Provenance: introduced by `3567c5e` *"refactor(atom)!: reduce the Atom input to
ten physically-named axes"* — a **refactor** commit; the file docstring claims
"A failure means the rename changed physics", which is false. Measured
sensitivity (fresh processes, this work):

| knob | E (Ha) | Δ vs pin | survives 1e-12? |
|---|---|---|---|
| default, 3 repeats, OMP_NUM_THREADS 1 and 8 | −128.6299783505843 | 0 | yes |
| `solve(tol=1e-9)` | −128.6299705068449 | +7.84e-6 | no |
| `solve(tol=1e-12)` | −128.62997038732755 | +7.96e-6 | no |
| `Basis(r_max=120.0)` | −128.62997679723262 | +1.55e-6 | no |
| `Basis(r_max=60.0)` | −128.6299786533976 | −3.03e-7 | no |
| `Sampling(n_grid=1500)` | −128.62998302272942 | −4.67e-6 | no |
| `Sampling.publication` | −128.6299781081566 | +2.42e-7 | no |
| `Basis(ns=120/200)` (shooting ignores `ns`) | unchanged | 0 | yes |
| bspline, default | −128.62995892582862 | 0 | yes |
| bspline, `ns=113` | −128.63000430017655 | −4.54e-5 | no |
| Be multiconfig, `ns=150` | −14.612851922468955 | −9.88e-7 | no |

Verdict: **defensible as a bit-exact change-detector, indefensible as a
precision claim.** The number reproduces only along the exact default code path
(it is the fixed point of the *default* SCF tolerance, 12 iterations); the
method's own numerical convergence error is ~1e-6 Ha, i.e. 6 orders of magnitude
larger than the tolerance, and a *tighter* SCF (`tol=1e-9`, 18 iterations) is
rejected. Worse, the pinned quantity contains `E_Breit = 0.0504371 Ha`
(the default `corrections` is breit+qed, `test_atom_input_surface.py:279-280`)
whose error against the DBSR_HF value 0.016643629692418 Ha quoted at
`test_nist_levels.py:190` is **0.03379 Ha = 3.4 × 10¹⁰ × the 1e-12 tolerance**;
the correction-free number is `-128.69193378732467`, which *is* the externally
verified value in `benchmarks/EXTERNAL_VERIFICATION.md:15`
(−128.69192580 vs DBSR_HF −128.69192584). So the file pins the
correction-contaminated value to 12 digits while the physics-correct comparison
sits 0.062 Ha away in a different file, and nothing in the test says so.
Remediation: mark the file `@pytest.mark.snapshot`; loosen the panel to
`abs=1e-6` (the solver's own convergence scale); keep one bit-exact regression
in a `snapshot`-marked test; and add the missing A/B gate on the
*correction-free* energy at 1e-6 against the DBSR_HF number already quoted in
`EXTERNAL_VERIFICATION.md`. Delete the docstring's "failure means the rename
changed physics".

**2. The newest external-verification gate certifies nothing that passes.**
`tests/test_nist_levels.py:66-86` — the only passing physics test asserts
`np.max(np.abs(with_breit - without)) > 100.0` cm⁻¹ between Breit on/off. It is
a one-sided magnitude threshold with no external anchor: it passes today while
the same file documents the scalar Breit as 3.03× too large
(0.0504371 vs DBSR_HF 0.016643629418 Ha, error 0.0338 Ha) and the C V rms vs
NIST as 10651.963 cm⁻¹. The other passing test (`:121`) parses the NIST cache.
Result: 4 of 6 tests are `xfail(strict=True)` and the 2 that pass assert
"something moved" and "the cache parses". Remediation: give the passing test an
absolute external target — assert the C V rms against the NIST levels
(`≤ 3000 cm⁻¹` after the G¹ fix, or record the current 10652 with an
`xfail(strict=True)` until it improves) and assert the differential Breit shift
against a quoted DBSR_HF/GRASP split rather than `> 100`.

**3. The Breit scalar that the NIST gate calls wrong is the default and is not
gated anywhere that passes.** `test_nist_levels.py:179-193` (xfail strict) vs
`test_atom_surface_equivalence.py:99` (passes, 1e-12). Measured:
`energy_components['E_Breit'] = 0.05043710257669187` Ha with
`breit_mode='scalar'` (the default) versus DBSR_HF `0.016643629692418` Ha —
ratio 3.03; the per-(2J,parity) Gaunt operator gives 0.017340 Ha. Any user
reading "Ne total energy pinned to 1e-12" would conclude the corrections are
under control; they are wrong by 0.034 Ha. Remediation: promote
`breit_ci` to the default only after the two documented coefficient errors
(v = L ~3× large, v ≠ L ~50× small) are fixed, and add a passing scalar-vs-operator
cross-check at the documented 4 % retardation level so the defect cannot hide
behind a strict xfail with no owner.

**4. He⁺ is pinned with a 25 % band on an exactly known hydrogenic energy.**
`tests/test_atom_api.py:36` — `he_plus.total_energy == approx(-2.0, abs=0.5)`.
Measured: `-2.0000102440959777` Ha; the closed-form Dirac value is
`C²(√(1-(2α)²) − 1) = -2.0002664` Ha. The band is ±13.6057 eV, i.e. ~91,000×
the physics. `tests/test_atom_api.py:14` (`-2.86 ± 0.1 Ha` = ±2.72 eV, 3.5 %)
and `:21` (`-128.5 ± 2.0 Ha` = ±54.4 eV) are the same defect (the correct Ne
value is in the same tree). Remediation: replace with the Sommerfeld formula,
exactly as `test_book_preface.py:43` already does for H 1s —
`assert abs(E/exact - 1) < 1e-9` — and for He use the HF limit −2.86168 with
`abs=1e-5`.

**5. Self-anchored tolerances in the provider contract.**
`tests/test_providers_contract.py:974,1008,1182,1205` retype the implementation's
own tolerance formula `1e-10 + 1e-12*abs(real)`; `src/atomx/providers/contract.py:166-169`
holds `_ELECTRON_COUNT_IMAG_ATOL = 1e-10`, `_..._RTOL = 1e-12` (same pair for
the ERI). `git log -S'1e-10 + 1e-12'` puts the test literal in the *same*
commits as the constants (`bc5a9cf`, `8a63b35`). A wrong tolerance and the test
agree by construction. Remediation: derive the bound from a round-off budget for
real symmetric `P, S` (`n·eps·‖P‖‖S‖ ≈ 1e-14`), assert accept at 1e-14 and
reject at 1e-8 (2e-7 eV), with the bound built from fixture norms.

**6. 92 % of the provider-contract nodes assert nothing about values — and the
contract is fail-open for three interior invariants.** Measured by an
independent probe: `validate()` accepts a **non-symmetric overlap `S`**,
`electron_count()` returns **3.99** for a 4-electron density (Tr(PS) is only
checked for finiteness), and `TwoElectronIntegrals` accepts an ERI that breaks
the 8-fold `(ij|kl) = (kl|ij)` symmetry. The suite's own adapter test asserts
that symmetry (`test_pyscf_adapter.py:115-117`, atol=1e-10) and Tr(PS) = N
(`:104`), so the invariants are known and simply absent from the contract file.
`tests/test_providers_contract.py` 92 defs = A9 B2 C7 E3 F71; 142 of 249 nodes
assert only "non-finite input raises" (all through `contract.py:281`); 30
`pytest.raises` carry no `match=`. Remediation: one table-driven non-finite
test; add S-symmetry, ERI permutation symmetry, and `Tr(PS) = N_elec`
consistency gates (all A-class, all independently derivable).

### S2

**7. `xfail_strict` is not configured.** `pyproject.toml [tool.pytest.ini_options]`
sets only `addopts = ["--strict-markers"]` and four markers — no
`xfail_strict`. Default is `False`. The four NIST xfails are *explicitly*
`strict=True` (`test_nist_levels.py:90,140,197,237`) — the prompt's suspicion
that they were non-strict is **not** borne out — but the two other xfails in
the slice are `strict=False` (`test_qa_provider_finite_contract.py:214,608`), so
they can never force a revisit and a fix would pass silently. Correct form:
add `xfail_strict = true` to `[tool.pytest.ini_options]` (making non-strict
xfails an explicit, reviewable opt-out) and re-mark the two QA xfails with
`strict=True` + a tracking reference. The correct *shape* of the four NIST
xfails is otherwise right: they assert the desired behaviour, quote the external
number, and named the fix path.

**8. Three different values of `1 Ha → eV` inside one slice.**
`test_atom_api.py:94` uses `27.21` with `abs=1.0` eV; `test_constants.py:33`
quotes CODATA 2022 `27.211386245981` (the library's `HARTREE_EV` is
`27.211386245981167`); `test_atom_unified_api.py:88,233` and
`test_providers_contract.py:85,94` use the CODATA **2018** value
`27.211386245988`. At `rel=1e-6` the 2.5e-13 relative difference is invisible, so
a corrupted CODATA table passes. `tests/test_first_principles_invariants.py:458`
already documents this in another slice. Remediation: import `HARTREE_EV` from
`atomx.constants` everywhere and assert the conversion identity at `rel=1e-12`;
keep exactly one literal (in `test_constants.py`) as the external oracle.

**9. Optional-extra skips are silent and non-strict: 48 nodes of coverage
disappear in a `[dev]` install.** `test_units.py:17` `pytest.importorskip("pint")`
skips all 32 nodes; `test_pyscf_adapter.py` 17/18 nodes skip on
`importorskip("pyscf")` (`:38,48,130,164,197,242,284`);
`test_pyscf_embedding.py:16-20` skips the whole module;
`test_providers_contract.py:82,90` and `test_atom_unified_api.py:227` add 3 more.
Measured: `199 passed, 2 skipped, 4 xfailed` for my run of 16 files without the
extras. `pint` and `pyscf` are **absent from the `dev` extra**
(`pyproject.toml:22-31`) and are only installed by CI's
`uv sync --locked --all-extras` (`.github/workflows/test.yml:43`). This is
exactly the silent-coverage-loss mode that `--strict-markers` was added to
prevent, one level up. Remediation: add `pint` (and optionally `pyscf`) to the
`dev` extra, and add a session-scoped assertion — e.g. a `conftest.py` hook that
fails when the collected-skip count exceeds an expected constant in CI — so a
future lockfile/hardware change cannot quietly drop the units and provider
gates.

**10. Loose windows where an analytic value exists.**
`test_units.py:222` asserts `abs(iso.FS_as("kHz").magnitude) < 50000`; measured
`E_FS_MHz = -5.972050736135407` → the window is 8.4 × the value and has no sign
check (a +50 MHz error passes). `test_atom_unified_api.py:275`
`orb.expectation_r(grid) == approx(0.75, abs=0.2)` — 27 % on the hydrogenic
`⟨r⟩ = 3/(2Z)`. `test_pyscf_embedding.py:130` `5.0 < shift_ev < 40.0` for the
MgO Madelung lifting of the Mg 1s core level (an 8× window on an analytic
lattice sum, α_M = 1.747565). `test_fac_structure_snapshot.py:40`
`SNAPSHOT_REL = 2e-4` = 88.9 eV at Z = 82 while the measured scatter is 3.76e-9
(53,247× margin). Remediation: `abs<10 kHz` with a sign assertion;
`⟨r⟩ = 3/(2Z)` at `rel=2e-2`; Madelung window ±20 % of the computed lattice sum;
`SNAPSHOT_REL = 1e-7`.

**11. The "FAC parity" file is a self-regression whose tolerances cannot see
the gap it records.** `tests/test_fac_parity_gates.py:69-98` with
`@pytest.mark.snapshot` and an exemplary docstring ("NOT a validation against
FAC … fac_* columns are never asserted"). Measured myself over the 34 golden
rows: `max_relE = 4.264e-9` against a bound of `1e-3` (234,523×), `max_relA =
3.415e-9` vs `5e-2` (1.46e7×), `max_relG = 5.38e-6` vs `5e-2` (9,294×); the
E gate's `1e-3` is **445.07 eV** at Z=82 (`max |atomx_E_eV| = 445,066`), i.e.
larger than the 426 eV atomx-vs-FAC disagreement the same benchmark records;
`TIMING_MULTIPLIER = 4.0` versus a measured per-Z ratio of 0.04–0.053 (75× dead).
Because the file is honestly named and marked, this is S2 (a tolerance that
cannot distinguish the physics at issue) rather than S1 — but anyone counting
this file as FAC coverage is misled. Remediation: E → 1e-6 (0.45 eV),
A/gauge → 1e-4, `TIMING_MULT` → 1.5 or delete the timing assert, and assert
`fac_*` once the units/definition mismatch in `docs/CRITICAL_REVIEW_2026-07-09.md`
§V1 is settled.

**12. The advertised Fe XVII 1e-6 A-rate gate is a length check.**
`tests/test_fac_io_roundtrip.py:195-197`: the test's own docstring promises the
transition-rate round trip at 1e-6, but the TR half only asserts
`len(rows_tr) == len(ref_trs)` (19 == 19); the omitted comparison measures
max rel 3.85e-9, so it passes trivially today. Remediation: copy the Li body at
`:149-154` (`gA/(2j+1)` vs `ref A`, `rtol=1e-6`) — that comparison *does*
discriminate.

**13. Two recorded-value round trips in the self-regression band cannot fail
for a physics reason.** `test_fac_dw_benchmarks.py:198,258` compare live Ω/σ to
recorded columns with 5 % and 1e-6; the 5 % bound is *looser* than the 2.85 %
live-vs-FAC rms the same file protects (measured 1.32 %) and the
threshold sub-gate is 1e-3 vs a measured 2.36e-10 (4.2e6× vacuous).
`:301` compares the CSV's recorded `rel_diff` against gates written by the same
run — no atomx call. Remediation: derive `rel_diff` from the live fixtures; set
`EXC_ATOMX_REPRO = 0.03`, `THRESHOLD_REPRO = 1e-6`; delete `:301` or make it
recompute.

### S3

**14. `@pytest.mark.snapshot` is used in 1 of 166 test files.** The marker is
documented as "frozen atomx SELF-regression … asserts nothing external", yet
only `test_fac_parity_gates.py` carries it, while at least 23 D-class functions
in this slice (`test_atom_surface_equivalence.py` ×12,
`test_atom_numerics_exchange.py:79`, `test_fac_structure_snapshot.py` ×21,
`test_fac_dw_benchmarks.py` ×2, `test_sos_basis.py` ×3, `test_data_loaders.py`
×3) carry none. `test_fac_structure_snapshot.py:37-39` even says in a comment
"the atomx column is a *self* snapshot" and still omits the marker.
Remediation: add the marker, and add a collection-time check that every file
containing a `<1e-9`-style frozen literal of a computed quantity is either
marked `snapshot` or has an external citation in its docstring.

**15. A test that reads as validation but only re-reads the record.**
`test_fac_structure_snapshot.py:81` `test_panel_agrees_with_fac_at_the_documented_level`
asserts the *recorded* `max|rel_diff| < 1e-3` (recorded 9.605e-4) and
`max|diff_eV| < 500` (recorded 426.487). No atomx and no FAC is invoked; only a
hand edit of the CSV within 4 %/15 % of the bounds can fail it, and the
`E_fac_eV` column has no regeneration path (no CI sets `FAC_*`, and
`~/code/fac/bin/bin/sfac` fails to load `libgfortran.5.dylib`).
Remediation: recompute FAC in a gated job when `sfac` works, or widen to 2e-3
and rename to `test_recorded_panel_deviation_is_below_the_documented_bound`.

**16. `test_fac_angular_conventions.py` "FAC reference" literals are textbook
values.** 10/10 of the 3j/6j/reduced-C literals agree with exact sympy to
≤ 2.2e-16 and 8/10 are bit-identical to atomx's own output; no FAC dump is
checked in. They duplicate `tests/test_angular.py:48-105`, which already
verifies the same quantities against sympy. Only
`test_reduced_cl_j2_matches_fac_reference` (`:43`) encodes a genuine FAC
convention (`faclib/angular.c:529-535`). Remediation: delete the duplicated
3j/6j tuples, keep one convention test with a checked-in FAC fixture.

**17. `test_atom_input_surface.py` is a 60-node plumbing file with zero
physics, and 26 of its nodes are error-message text matches.**
Measured: 30 defs → 60 nodes (`RETIRED` has 26 entries; `Basis()/Corrections()/…`
frozenness 5; wrong-type 2). Class E per the rubric's "error-message *text*
matching where the error type alone would do" — they do guard one real
silent-failure mode (a stale name must not bind a *different* physical
meaning), and `test_only_config_is_positional` (`:97`) is the load-bearing part
of that guarantee. But `test_correction_presets_map_to_the_booleans` (`:275`)
iterates the module's own `CORRECTION_PRESETS` dict and asserts `Atom` agrees
with it — self-anchored, unfailable. Remediation: keep the positional/keyword-only
and signature-freeze tests, collapse the 26 message tests into one
parametrized table checking type + the replacement *token* only, and drop the
`MIGRATION`/`retired` prose assertions; move the preset test's expectation to
literal booleans. (The prompt's "65 tests" is actually 60 nodes / 30 defs.)

**18. C-degenerate pairs.** `test_data_loaders.py:31` (interpolator reproduces
its own `TABLE` — same dict, so it can only detect a code edit);
`test_sos_basis.py:52-58` (`build_sos_virtual_basis` vs `build_virtual_basis`
length — the former calls the latter); `test_atom_numerics_exchange.py:52,59`
(the "default" and "explicit" calls both resolve to the shooting engine);
`test_atom_inputs_wired.py:288` (`mss == auto` — the default *is* mss);
`test_atom_unified_api.py:105,116,124` (`x == approx(x, rel=1e-12)`).
Rubric-compliant use requires naming the differing kernel; none do.

**19. Bare-raise and order-coupled provider tests.**
`tests/test_providers_contract.py` has 30 `pytest.raises(ProviderContractError)`
with no `match=`; `:1226` passes only because block validation happens to run
before the leading-dimension check for an ERI shaped `(2,2,2,2)` with
`n_ao = 3`. Remediation: discriminating `match=` on the stable field label.

**19b. A 38-test first-principles file exists in the working tree but is not in
git, and the NIST gate's reason text says it is absent.**
`git status --porcelain` reports `?? tests/test_breit_operator.py` (457 lines,
38 tests, `38 passed in 2.02s` — measured) and
`?? tests/test_first_principles_invariants.py`. `tests/test_breit_operator.py`
is a genuinely A-class file (Dirac κ bookkeeping, `σ_μ† = (−1)^μ σ_{−μ}`,
`Tr(σ_μ σ_ν†) = 2δ_{μν}`, the closed-form Racah–Gaunt coefficient, a
completeness sum rule, `M` conservation, and a retardation-omission provenance
check) and its docstring says `breit_ci`'s angular algebra is checked there.
But `test_nist_levels.py:159-160` (inside the `test_breit_scalar_matches_dbsr_hf`
reason) states that `tests/test_breit_operator.py` is "claimed by the breit_ci
docstring but **ABSENT** from the tree" — false for the working tree, true for
git and therefore true for CI, which checks out from git. Consequences: (a) CI
never runs those 38 gates; (b) every `git log -S` provenance search in this
audit is blind to both files, which is how the false "ABSENT" claim survived;
(c) any slice reporting the Breit angular algebra as untested is wrong about the
working tree. Remediation: `git add` both files (and add a CI check that the
working tree has no untracked `tests/*.py`), then correct the NIST xfail text.
The 38 tests were not classified here — they belong to the Breit slice.

### S4

**20. A test whose name asserts a physics claim its body deleted.**
`test_ladder_and_sampling.py:167` `test_climb_total_energies_decrease_monotonically_on_he`
asserts only `E < 0` and `isfinite(E)` for each rung — **no monotonicity check
exists**, and the body's comment concedes the first rung is *less* bound.
Remediation: assert `E_rung[k+1] < E_rung[k]` for the rungs after the
non-interacting one, or rename to `test_climb_energies_are_negative_and_finite`.

**21. Stale retired-API names in test docstrings after a clean break that
deleted them.** `test_atom_unified_api.py:5-7` (`Atom(Z, numerics='bspline')`,
`configs=[...]`), `test_atom_api.py:52` (`Atom(configs=[...])`),
`test_atom_numerics_exchange.py:5,49,67` (`numerics=None`),
`test_ladder_and_sampling.py:7` (`Atom(Z, sampling=...)` — now
`basis=Basis(sampling=...)`). The migration updated call sites but left the API
documentation inside tests pointing at names that raise `TypeError`. This is the
one place where a reader learns the API. Remediation: search-and-replace the
docstrings in the same commit style as the call-site migration.

**22. Assertions that cannot fail for a physics reason.**
`test_atom_api.py:28,74,101,103,110` (`< 0`, `< 0`, `< 0`, `< -1000`, `< -450`),
`test_atom_api.py:42` (`"E_VP" in components`), `test_atom_api.py:78,81`
(`"He" in repr`), `test_native_provider.py:63` (isinstance of the test's own
mock), `test_atom_unified_api.py:250` (`fig is not None`),
`test_book_preface.py:113-114` (`result.converged`; `len(result.subshells) > 0` — **40.33 s of the
47 s runtime**, and the preface claim "the call that solves hydrogen solves
tungsten unchanged" deserves an external DHF number for W, e.g. −16115 Ha).
`test_book_preface.py:104` `assert u_sys_sr > 0.0` is explicitly labelled "NOT
an atomx verification" — good disclosure, but it is still an E-class node;
keep as documentation only.

**23. Count/dtype-only tests in the provider files.**
`test_qa_provider_finite_contract.py:353` asserts only `dtype == complex128`;
`test_providers_contract.py:1059` asserts isinstance of a `_Prov` mock defined
in the test file (an unused monkeypatch); `:207,583,614` assert
`validate() is not None` on a method that returns `self` and therefore cannot
fail. Remediation: `assert payload.validate() is payload`, and point the
protocol test at `native_adapter`/`pyscf_adapter` output.

**24. The ERI 8-fold symmetry is asserted in the *adapter* but not in the
*contract*.** `test_pyscf_adapter.py:115` (atol=1e-10) vs
`test_providers_contract.py` (no symmetry assertion). The contract file is the
one that runs everywhere; the invariant belongs there.

---

## 4. Model tests (imitate these)

1. **`test_book_preface.py:35`** `test_hydrogen_dirac_1s_eleven_digits` — H 1s
   against the closed-form Sommerfeld value `C²(√(1-α²)−1)`, both engines,
   `rel < 1e-11`, with the high-accuracy sampling tier explicitly selected and
   corrections disabled. This is what a physics gate should look like.
2. **`test_msub.py:12,19,35,41`** — J = 0 → 1 axial photo-absorption populates
   only |M| = 1; weights sum to 1 at 1e-12 for four (J_a, J_b); Σσ(M) = σ; and
   A₂ = √(1/6) from Inal–Dubau. Pure angular algebra, no code self-reference.
3. **`test_pyscf_embedding_model.py:53,70,80`** — Evjen summation: the truncated
   rocksalt cube is neutral, so `net_mm_charge == -q_cation` at 1e-12, and the
   Mg₁₃O₁₄ count under `quantum_shells=1` gives +2. Derivable lattice
   arithmetic, stated in the comments.
4. **`test_pyscf_adapter.py:90,101,107`** — `CᵀSC = I` at 1e-8 preceded by a
   *sanity* assertion that the AO overlap is **not** orthonormal
   (`max_deviation > 0.1`), `Tr(PS) = N` at 1e-8, and ERI 8-fold symmetry at
   1e-10; plus `:296` MOM tracking = `argmax|ref·S·C|` with `max_overlap == 1.0`
   exactly for a column-swapped orthonormal basis.
5. **`test_pyscf_embedding.py:215,226`** — the zero-charge-scale **limit**
   recovers the isolated atom to 1e-8, and doubling the MM charge scale doubles
   the core shift to 5 % (linear response). Limits and scalings, not pins.
6. **`test_atom_inputs_wired.py:103`** — the SCF plasma shift vs first-order
   perturbation theory `∫dV·ρ_unscreened dr` at `rel=0.10`, with the *sign* of
   the relaxation predicted in advance (`scf_shift < first_order`).
7. **`test_atom_inputs_wired.py:213`** — box independence: widen the B-spline box
   until H 6s matches the box-free shooting eigenvalue to 1e-6, having first
   shown the narrow box misses by 0.29 eV.
8. **`test_atom_inputs_wired.py:139`** — a neutral atom in a Debye plasma must
   have **no bound state** (`UnboundSubshellError`); a model-asymptotics claim.
9. **`test_atom_unified_api.py:143,309`** — Guerra occupation asserted as
   `2(1 − Σ|mixing|²)`, the CI normalisation identity, instead of the previous
   "≈ 2.0" pin.
10. **`test_ladder_and_sampling.py:234,245`** — `∫ρ dr = N_electrons = 10.0` on
    the log grid (`abs=0.2`, quadrature-limited) and on the native grid
    (`rel=1e-3`).
11. **`test_constants.py:31,46`** — CODATA 2022 values quoted *with their
    source and uncertainty* (`rel=2e-10` against ±1e-10) and the exact
    atomic-unit identity `SPEED_OF_LIGHT == 1/ALPHA` at 1e-14.
12. **`test_fac_dw_benchmarks.py:232,281`** (per the FAC sub-audit) — live
    recompute of the atomx quantity at the recorded grid against the **stored
    external FAC column**, with per-point *and* rms gates whose docstrings quote
    the measured margin (≤ 2×). The best external-oracle pattern in the slice.
13. **`test_qa_provider_finite_contract.py:385`** — asserts the fixture *can*
    fail (`max|CᵀSC − I| > 0.1` for the naive transpose) before asserting the
    production value; the pattern that keeps an A-class test from being a
    tautology.

---

## Annex A — public API surface and coverage

45 names are visible from `src/atomx/__init__.py` (8 submodules + 37 names).
"Refs" counts `\bname\b` matches under `tests/**/*.py` (AST-verified export
list; `.pyc` caches excluded).

| symbol | refs | files | status |
|---|---|---|---|
| Atom | 477 | 73 | covered |
| Basis | 65 | 11 | covered |
| Corrections | 21 | 5 | covered |
| Correlation | 38 | 10 | covered |
| Environment | 18 | 3 | covered |
| Exchange | 42 | 10 | covered (the `atomx.atom_inputs` dataclass; the `atomx.types` enum is shadowed and tested once at `test_atom_input_surface.py:394`) |
| Sampling | 42 | 5 | covered |
| compare | 7 | 6 | covered |
| Ladder / LadderRun / RungOutcome | 12 / 5 / 2 | 1 each | single-file coverage (added file) |
| MCDHF | 51 | 21 | covered |
| BreitPauli | 16 | 2 | covered |
| DiracFock | 30 | 4 | covered |
| HartreeFock | 3 | 2 | covered |
| HydrogenicDirac | 4 | 1 | single-file |
| HydrogenicSchrodinger | 6 | 1 | single-file |
| AtomicResult | 3 | 1 | covered (protocol isinstance only) |
| HyperfineResult | 6 | 2 | 1 live file |
| OrbitalWavefunction | 9 | 3 | covered |
| Waves | 10 | 4 | covered |
| ContinuumModel / PotentialModel | 7 / 4 | 2 / 1 | covered |
| fitting, lineshapes, properties, scattering, spectrum, uta, viz | 4–108 | 1–44 | covered |
| **theory_ladder** | **0** | **0** | **ZERO COVERAGE** — `src/atomx/compare.py:134` |
| **what_changed** | **0** | **0** | **ZERO COVERAGE** — `src/atomx/compare.py:249` |
| **Method** | **0** | **0** | **ZERO COVERAGE** — the `atomx.methods` base class |
| **CouplingScheme** | **0** | **0** | **ZERO COVERAGE** |
| **EnergyUnit** | **0** | **0** | **ZERO COVERAGE** |
| **Gauge** | **0** | **0** | **ZERO COVERAGE** (3 apparent hits are the English word in comments: `test_fac_parity_gates.py:84`, `test_transitions.py:7`, `test_transitions_engine_uniform.py:68`) |
| **Multipole** | **0** | **0** | **ZERO COVERAGE** |
| **NuclearModel** | **0** | **0** | **ZERO COVERAGE** |
| **RelativisticCorrection** | **0** | **0** | **ZERO COVERAGE** |
| **StartingPotential** | **0** | **0** | **ZERO COVERAGE** |
| **TheoryLevel** | **0** | **0** | **ZERO COVERAGE** — only `atom.theory_level is not None` is asserted (`test_atom_unified_api.py:92,103`) |
| codata_version | 3 | 1 | **only in the skipped `test_units.py`** |
| pint_available | 3 | 1 | **only in the skipped `test_units.py`** |
| ureg | 15 | 2 | live only in `test_providers_contract.py:83-94`, both inside `importorskip("pint")` tests → **zero executed coverage** |

Summary: **11 symbols with zero references anywhere** (`theory_ladder`,
`what_changed`, `Method`, `CouplingScheme`, `EnergyUnit`, `Gauge`, `Multipole`,
`NuclearModel`, `RelativisticCorrection`, `StartingPotential`, `TheoryLevel`),
and 3 more (`codata_version`, `pint_available`, `ureg`) with zero *executed*
coverage in a `[dev]` install. The 7 legacy `atomx.types` enums are exported but
never used by any test; the results protocol they feed is only checked for
`is not None`. `atomx.units`' converter API (`to_hartree`, `to_bohr`,
`to_kelvin`, `to_MHz`, `energy_quantity`, …) has coverage in exactly one file,
which does not run without the `units` extra.

## Annex B — skipped, xfailed, deselected

| where | count | reason (verbatim / verbatim-equivalent) | strict? |
|---|---|---|---|
| `test_units.py:17` | 32 nodes | `could not import 'pint': No module named 'pint'` | n/a (skip) |
| `test_atom_unified_api.py:227` | 1 | `could not import 'pint'` | n/a |
| `test_providers_contract.py:82,90` | 2 | `could not import 'pint'` | n/a |
| `test_pyscf_adapter.py:38,48,130,164,197,242,284` | 17 of 18 | `could not import 'pyscf': No module named 'pyscf'` | n/a |
| `test_pyscf_embedding.py:16-20` | whole module (14 defs, 0 nodes collected) | "PySCF is an optional dependency (atomx[pyscf]); the whole module is skipped on the minimal dependency set" | n/a |
| `test_nist_levels.py:90` | 1 | `test_cv_1p_3p_splitting_matches_nist` — 40481.6 vs NIST 28172.3 cm⁻¹ (1.44×); G¹ 0.2762 vs NIST-implied 0.1925 Ha | **YES (`strict=True`)** |
| `test_nist_levels.py:140` | 1 | `test_breit_scalar_matches_dbsr_hf` — scalar 0.050449 vs DBSR_HF 0.016643629692418 Ha (3.03×); its reason text additionally (and now wrongly) claims `tests/test_breit_operator.py` is "ABSENT from the tree" — the file exists untracked, 38 tests pass (finding 19b) | **YES** |
| `test_nist_levels.py:197` | 1 | `test_h1_mcdhf_orbitals_are_screened` — MCDHF rms 2132 vs DHF 47.5 cm⁻¹ (45×), 2p −0.361624 vs −0.125 Ha | **YES** |
| `test_nist_levels.py:237` | 1 | `test_ar_mbpt2_correlation_energy_is_physical` — E2 = −1.057787 Ha, rms 217,223 cm⁻¹ | **YES** |
| `test_qa_provider_finite_contract.py:214` | 1 | "HYP-1380 QA residue: UnitError from the orbital_energies mapping does not name the offending key" | **NO (`strict=False`)** |
| `test_qa_provider_finite_contract.py:608` | 1 | "nested numeric values are not finite-gated (fail-open)" | **NO** |
| `test_nist_levels.py:44` `_needs_data` | 0 (cache present) | skipif if `benchmarks/nist_asd/levels/asd.006004` is missing | n/a |
| `test_book_preface.py:107` `@pytest.mark.slow` | 1 | deselected by CI's `pytest -m "not slow"` (`test.yml:58`); run nightly (`nightly.yml:41 -m "slow"`, `:43 -m "book"`) | n/a |
| `test_fac_io_roundtrip.py:178` `@pytest.mark.slow` | 1 | same CI deselection (per FAC sub-audit) | n/a |
| `test_fac_parity_gates.py:33` | 0 | `FAC_PARITY_TIMING_MULT` env override (default 4.0) | n/a |

No test in the slice is deselected by default in a plain local run; the
`-m "not slow"` deselection is CI-only. `xfail_strict` is unset in
`pyproject.toml` (finding 7).
