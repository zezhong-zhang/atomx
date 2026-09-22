# S7 remediation — public API, units, providers, external-verification harness

Remediation of `docs/test_audit/S7_api_units_harness.md`.  Scope: the five
priority items (a–e) of the remediation brief, plus the S1/S2 findings in this
slice that could be closed without touching `src/atomx` or files owned by other
slices.  All numbers below were measured in this working tree on 2026-09-22;
every gate that does not carry the `snapshot` marker names an external oracle
(DBSR_HF, NIST ASD, Clementi–Roetti) or a closed form written out in the test.

**Tree-state caveat.**  `src/atomx/corrections/breit.py`,
`src/atomx/methods/mcdhf/breit_ci.py`, `src/atomx/methods/mcdhf/solver.py`,
`src/atomx/corrections/adapters.py` and `.../qedmod/mohr.py` are modified in the
working tree by another agent while this slice was remediated: the C V panel's
`E_Breit` moved from 0.007345078435653687 Ha (the 3.03x scalar) to
0.002447606634596866 Ha (the DBSR_HF-consistent scale) between two measurements
of this session.  Gates here that would move with that fix are written as
*patterns* (sign, ordering, unit identities), not as pins.

## 1. Changes

| # | finding | file:line | change | measured before → after | new gate |
|---|---|---|---|---|---|
| a | S1-1: 1e-12 total-energy pin is false precision | `test_atom_surface_equivalence.py:99,105` | panel tolerance `abs=1e-12` → `abs=1e-6` (`CONVERGENCE_SCALE_HA`); every frozen-literal test marked `@pytest.mark.snapshot`; one bit-exact regression kept (`==`, `test_the_ne_panel_keeps_every_orbital`) | 1e-12 (3.4e10 × the 0.0338 Ha Breit error) → 1e-6 (solver's own convergence scale); bit-exactness preserved in the marked test | new A/B gate `test_ne_correction_free_energy_matches_dbsr_hf`: `corrections="none"`, `solve(tol=1e-11)` → −128.69192580015647 vs DBSR_HF −128.69192580 (log −128.691925796129510), `abs=1e-6`, measured 1.6e-10 (4.0e-9 vs the full-precision log) |
| a | S1-1: false docstring claim | `test_atom_surface_equivalence.py:1-14` | "A failure means the rename changed physics" **deleted**; docstring now states the panel is the fixed point of the *default* SCF tolerance (tol=1e-5), that `solve(tol=1e-9)` (+7.84e-6) and `Basis(r_max=120)` (+1.55e-6) are rejected, and that the pinned energy contains the 3.03x Breit | claim "physics changed" → "default code path moved" | — |
| b | S1-2: the only passing NIST test asserted `max|Δ| > 100` cm⁻¹ | `test_nist_levels.py:66-86` | replaced by `test_breit_shifts_excitation_energies_differentially`: per-(2J, energy-rank) pairing (the CI block order is not stable, the old elementwise difference mixed levels), four excited levels must all move **down**, spread > 100 cm⁻¹, and `max|Δ| ≤ E_Breit+E_QED` | one-sided `>100` (no anchor) → −529.99/−770.65/−629.03/−693.38 cm⁻¹, spread 240.7, bound 909.6 | A-pattern gate (repulsive e–e Breit; the compact 1s² ground carries the largest Breit; not a uniform scalar) with the 3.03x magnitude defect named in the docstring and carried by the strict xfail |
| b | S1-2/S3-19b: false "ABSENT from the tree" claim | `test_nist_levels.py:159-160` (now :200-207) | reason text now states `tests/test_breit_operator.py` **exists** (38 tests: Racah–Gaunt closed form, hermiticity pairing, completeness sum rule, M/L selection rules) and passes; the red item is the element-level external anchor | false claim → true statement | — |
| b | S1-3/S2: DBSR_HF number provenance | `test_nist_levels.py:190` | DBSR_HF Ne Breit 0.016643629692418 → **0.016643630840490** (the run whose log line 338 the orchestrator pointed at); atomx scalar 0.050449 → **0.050448592973526926** measured on the gate's own solver path; He data point added: live `dbsr_hf an=2 term=AV mbreit=1` 0.000063777429574 vs atomx 0.00019155548661222858, ratio 3.003 (structure-independence) | documentation only; gate unchanged (`rel=0.05`, still strict xfail) | — |
| c | S2-9: 48+ nodes of coverage vanish silently | `test_units.py:17` | `pytest.importorskip("pint")` → module-level `HAS_PINT` + `pytestmark = pytest.mark.skipif(...)`, reason names the `units` extra and the fix (`pip install -e .[dev]`) | 1 anonymous module skip / 32 nodes → 32 individually reported skips; 32 run on a fresh `.[dev]` | — |
| c | S2-9 | `test_pyscf_adapter.py:38,48,130,164,197,242,284` | 7 in-body `importorskip("pyscf")` removed; module-level `HAS_PYSCF` + `@needs_pyscf` on the 17 tests that really build a mean field (verified: the two "spurious" guards at :164/:197 do need pyscf — `PySCFProvider.__init__` imports it — so they were re-marked after an initial failure) | 1 passed / 17 skipped with 5 different anonymous sites → 1 passed / 17 skipped, all 17 named | — |
| c | S2-9 | `test_pyscf_embedding.py:16-20` | module-level `importorskip` → `HAS_PYSCF` + `pytestmark` skipif; 3 in-body guards removed; `HARTREE_TO_EV = 27.211386245988` (CODATA 2018) → `from atomx.constants import HARTREE_EV` (finding 8) | 1 module skip, 0 nodes collected → 14 collected and reported skips | — |
| c | S2-9 | `test_atom_unified_api.py:227,248`, `test_providers_contract.py:82,90` | in-body `importorskip` → module-level `needs_pint` / `needs_plotly` markers with the extra named | 3 anonymous skips → 3 named skips; 2 unit-coercion nodes + 1 eV node unchanged (pint absent), plotly node still runs | — |
| c | S2-9: dev extra cannot run the gates | `pyproject.toml:23-31` | `pint>=0.21` and `pyscf>=2.3` added to the `dev` group | 66 nodes un-runnable from `pip install -e .[dev]` → runnable; **0 move to run in this venv** (both are genuinely absent — `importlib.util.find_spec` returns None) | — |
| d | S3-17: surface file reads as validation | `test_atom_input_surface.py:1-14` | docstring now opens with "**This file validates no physics**", classifies every node as F/E, names where the physics is gated, and states that message *prose* is deliberately not asserted | implied validation → explicit surface-freeze | retirement coverage kept intact (26 nodes) |
| d | S3-17: error-message text matches | `test_atom_input_surface.py:131-139` | dropped `"retired" in message` and `"MIGRATION" in message`; kept `TypeError` + the offending name + the replacement-keyword token(s) from the `RETIRED` table | prose matching → type + replacement token | — |
| d | S3-17: self-anchored preset test | `test_atom_input_surface.py:275` | iteration over the module's own `CORRECTION_PRESETS` replaced by the literal mapping `{none:(F,F), breit:(T,F), qed:(F,T), "breit+qed":(T,T)}` | dict == itself → literal expectation | — |
| e | S4/Annex A: zero-coverage public API | new `test_public_api_ladder_and_compare.py` (8 tests) | `theory_ladder`, `what_changed`, `Method` (structural), `TheoryLevel`, `CouplingScheme`, `EnergyUnit`, `NuclearModel`/`StartingPotential` (value spellings) | 0 references → 8 gates | Z=1 rungs = `−Z²/(2n²)` and Sommerfeld `C²(γ−1)` (measured rel 7.6e-9); DHF rung vs DBSR_HF He −2.86181332 / Ne −128.69192580 at `abs=1e-6` (measured 4e-8); `what_changed` Ha→eV/cm⁻¹ identities + antisymmetry; `compare(unit=cm-1)` prints `E*HARTREE_CM1`; finite-nucleus energy above point-nucleus, shift growing with Z |
| 7 | S2-7: `xfail_strict` unset | `pyproject.toml:107-116` | `xfail_strict = true` added | non-strict xfails could never force a revisit → xpass fails the run | — |
| 7 | S2-7: two silent xfails | `test_qa_provider_finite_contract.py:209,602` | `strict=False` → `strict=True` (both currently XFAIL, verified: 87 passed, 2 xfailed) | — | — |
| 4 | S1-4: 25 % band on a hydrogenic energy | `test_atom_api.py:36` | `He+ ≈ −2.0 ± 0.5` → `corrections="none"` and the exact Sommerfeld value `C²(γ−1)`, `|E/exact − 1| < 1e-7` | ±13.6057 eV (25 %) → measured 7.6e-9 relative | closed form in-test |
| 4 | S1-4: same defect, He/Ne | `test_atom_api.py:14,21` | He `−2.86 ± 0.1` → `corrections="none"` vs DBSR_HF −2.86181332 `abs=1e-6` (measured 3.0e-10); Ne `−128.5 ± 2.0` → DBSR_HF −128.69192580 at `tol=1e-11`, `abs=1e-6` (measured 1.6e-10) | ±2.72 eV and ±54.4 eV → sub-µHa | DBSR_HF (`EXTERNAL_VERIFICATION.md` §1) |
| 22 | S4: assertions that cannot fail | `test_atom_api.py:28,74,103,110` | Li `< 0` → variational bracket `−10.125 < E < −7.432727+1e-3` (hydrogenic sum / Clementi–Roetti HF limit; measured −7.4327904516039); Fe `< −1000` → DBSR_HF Fe −1271.414194275546834 `abs=0.5` (measured −1271.5441, ns=60 basis truncation); Ar `< −450` → DBSR_HF −528.68377257 `abs=5e-4` (measured 5.8e-5, grid-limited); `test_chaining` now asserts the identity `solve() is atom` | 271 Ha / 78 Ha of slack → 0.5 Ha / 5e-4 Ha | three external anchors |
| 8 | S2-8: three values of 1 Ha→eV | `test_atom_api.py:94`, `test_atom_unified_api.py:108`, `test_pyscf_embedding.py:36`, `test_providers_contract.py:96,105` | all literals (`27.21`, `27.211386245988`) replaced by `HARTREE_EV` from `atomx.constants` | `abs=1.0` eV / `rel=1e-6` → `rel=1e-12` (exact identity, measured ratio 1.0) | constant identity |
| 10 | S2-10: ⟨r⟩ = 0.75 ± 0.2 (27 %) | `test_atom_unified_api.py:295` | replaced by the derived bracket `3/(2Z) < ⟨r⟩ < 3/(2(Z−1))` → `0.75 < ⟨r⟩ < 1.5` (measured 0.9272 — the audit's suggested 0.75 reference is wrong for a *screened* two-electron 1s); new `test_hydrogenic_ion_expectation_r_matches_the_closed_form` for the one-electron He+ where `3/(2Z)` **is** exact (measured 0.7499549913620823 vs 0.75, rel 6e-5, gate rel=1e-3) | window [0.55, 0.95] (26 % below the physics) → bracket + exact closed form | closed form |

No file under `src/atomx/` was modified.  `pyproject.toml` was modified only for
the two items above (dev extra, `xfail_strict`).

## 2. Verification

Per-file, `.venv/bin/python -m pytest <file> -q -p no:randomly`, before → after:

| file | before | after |
|---|---|---|
| `test_units.py` | 1 skipped (module; 32 nodes invisible) | **32 skipped**, each naming the `units` extra |
| `test_pyscf_adapter.py` | 1 passed, 17 skipped | 1 passed, 17 skipped (all named) |
| `test_pyscf_embedding.py` | 1 skipped (module; 0 nodes collected) | **14 skipped**, each named |
| `test_atom_unified_api.py` | 30 passed, 1 skipped | **31 passed**, 1 skipped |
| `test_providers_contract.py` | 158 passed, 2 skipped | 158 passed, 2 skipped (named) |
| `test_atom_surface_equivalence.py` | 13 passed | **14 passed** (new DBSR gate) |
| `test_atom_input_surface.py` | 60 passed | 60 passed |
| `test_atom_api.py` | 12 passed (43 s) | 12 passed (16 s) |
| `test_qa_provider_finite_contract.py` | 87 passed, 2 xfailed | 87 passed, 2 xfailed (both strict) |
| `test_nist_levels.py` | 2 passed, 4 xfailed | 2 passed, 4 xfailed (reasons corrected) |
| `test_public_api_ladder_and_compare.py` | (new) | **8 passed** |

Combined run of all eleven files above (including `test_nist_levels.py`, whose
four xfails cost ~64 s):
`.venv/bin/python -m pytest <the 11 files> -q -p no:randomly` →
**373 passed, 66 skipped, 6 xfailed in 99.7 s**.  The same eleven files before
the remediation: 363 passed, 66 skipped (reported as 22 entries), 6 xfailed —
i.e. the executed-node count rises by 10 (two new tests in
`test_atom_surface_equivalence.py`, one in `test_atom_unified_api.py`, the eight
new public-API tests) and the un-executed node count is unchanged at 66.
`ruff check` (E4,E7,E9,F,I,B,SIM; E741 ignored) is clean on every file touched,
including `test_nist_levels.py` and the new file.

## 3. Skip → run count

`pint` and `pyscf` are **genuinely absent** from this venv
(`importlib.util.find_spec` → `None`), so the skip logic was not wrong, only
silent.  Consequently:

* **tests that move from skip to run in this environment: 0.**
* **tests that move from skip to run on a fresh `pip install -e .[dev]`: 66**
  — 32 `test_units.py` + 17 `test_pyscf_adapter.py` + 14
  `test_pyscf_embedding.py` + 1 `test_atom_unified_api.py` (eV accessor) + 2
  `test_providers_contract.py` (pint coercion) = **66 nodes**, the whole of
  Annex B's optional-extra block.
* **reported skip entries: 22 → 66.**  The 66 nodes were always skipping; before
  this change 44 of them were hidden behind one anonymous module-level skip
  each (32 units + 14 embedding), so a reader counting the summary saw 22.
* `test_nist_levels.py`'s four xfails are unchanged (4 xfail, 0 xpass); the two
  QA xfails are now strict and still xfail.  All 16 `xfail`s in the suite were
  checked for XPASS before turning `xfail_strict` on: `test_M1_E2_rates.py:498`,
  `test_auger_fac_panel.py:212`, `test_auger_kll_vs_fac.py:153` and the two QA
  ones all currently XFAIL, so the global flag changes no other file's outcome
  today (it only makes a future fix fail loudly instead of passing silently).

## 4. Public-symbol coverage (Annex A)

| symbol | before | after |
|---|---|---|
| `theory_ladder` | 0 refs | **covered** — closed-form Z=1 rungs + DBSR_HF DHF rung for He/Ne |
| `what_changed` | 0 refs | **covered** — Ha→eV/cm⁻¹ identities, argument antisymmetry, keyed explanation |
| `Method` | 0 refs | **structural** — `issubclass(...)`, ladder rungs are the documented method objects |
| `TheoryLevel` | 0 refs | **covered** — HYDROGENIC rungs, last rung `DIRAC_HARTREE_FOCK` |
| `CouplingScheme` | 0 refs | **covered** — LS on the non-relativistic rung, JJ on the Dirac rung |
| `EnergyUnit` | 0 refs | **covered** — `compare(unit=EnergyUnit.CM1)` prints `E*HARTREE_CM1` |
| `NuclearModel` | 0 refs | **partially** — the *value* strings are gated (`test_nuclear_and_starting_potential_enum_values_are_accepted_spellings`, finite-nucleus shift grows with Z); the **member** is not accepted: `Atom(10, nuclear=NuclearModel.POINT)` raises `ValueError` (see §5) |
| `StartingPotential` | 0 refs | **partially** — value spellings gated; `Basis(starting_potential=StartingPotential.THOMAS_FERMI)` raises `ValueError` (see §5) |
| `RelativisticCorrection` | 0 refs | **indirect** — consumed by the ladder's Breit-Pauli rungs; no direct assertion (it is only read inside `compare.theory_ladder`) |
| `Gauge`, `Multipole` | 0 refs | **still zero** — no live consumer anywhere in `src/atomx` (`Gauge`/`Multipole` appear only in `types.py`); a test could only pin the enum member values, which is E-class |
| `codata_version`, `pint_available`, `ureg` | only in the silently-skipped `test_units.py` | unchanged in *this* venv (still skipped, now loudly); they execute on a fresh `.[dev]` install |

## 5. Not fixed, and why

1. **S1-3 (promote `breit_ci` to the default).**  Requires a `src/atomx` change
   and the coefficients are being rewritten in the working tree right now by
   another agent (`breit.py`, `breit_ci.py`, `solver.py` are dirty; the C V
   panel `E_Breit` moved 0.007345 → 0.002448 Ha during this session).  Reported
   instead: the defect is documented in the strict xfail with both DBSR_HF data
   points.
2. **S1-5, S1-6, S3-19, S3-24 (provider contract: self-anchored tolerances,
   fail-open interior invariants, bare `raises`, ERI symmetry missing from the
   contract).**  All require *new* gates in `test_providers_contract.py` built
   from a round-off budget (`n·eps·‖P‖‖S‖ ≈ 1e-14`) — out of the a–e priority
   set and not verifiable without a longer pass over that 160-node file.  Not
   weakened, not touched.
3. **S2-10 in `test_units.py:222`** (`abs(FS_as("kHz")) < 50000`).  Deferred: the
   file does not run without `pint`, so a tightened bound cannot be verified
   here; the audit's own remediation ("`abs<10 kHz`") contradicts its own
   measurement (5.972 MHz = 5972 kHz), so applying it blind would break the
   gate.  Left at 50000 rather than guessed.
4. **S2-10 in `test_pyscf_embedding.py:130`** (`5.0 < shift_ev < 40.0`) and
   **S2-12/S2-13 (`test_fac_io_roundtrip.py`, `test_fac_dw_benchmarks.py`)** —
   outside the a–e set (pyscf absent; FAC files belong to the FAC sub-audit).
5. **S3-14 (add `snapshot` to the remaining ~23 D-class functions).**  Labelling
   only; the marker was added to the file item (a) names
   (`test_atom_surface_equivalence.py`, 3 of 4 tests; the fourth is the new
   external gate and must *not* be marked).  The rest
   (`test_atom_numerics_exchange.py:79`, `test_fac_structure_snapshot.py`,
   `test_data_loaders.py`, `test_sos_basis.py`) are owned by other slices — a
   sweep would collide with their authors.
6. **S4-21 (stale retired names in test docstrings).**  Documentation-only
   sweep across files outside the a–e set.
7. **`test_custom_config`'s `Li` bracket** uses the Clementi–Roetti HF limit
   −7.432727 Ha from memory of the published table; the number is standard, but
   it was not re-read from the paper in this offline environment (the measured
   DHF value −7.4327904516039 sits 6.3e-5 below it, and the gate's slack is
   1e-3, so a ±3e-5 uncertainty in the quoted limit cannot flip it).

## 6. Code changes

None in `src/atomx`.  Two API inconsistencies found while gating the exported
enums, reported rather than fixed (both would be `src` changes):

* `Atom(10, nuclear=NuclearModel.POINT)` → `ValueError: nuclear must be 'fermi',
  'point', or 'uniform' (got <NuclearModel.POINT: 'point'>)`; only the value
  string is accepted, although `atomx/types.py` documents member/value
  equivalence.
* `Basis(starting_potential=StartingPotential.THOMAS_FERMI)` → same shape
  (`starting_potential must be None, 'tfda', or 'gsz'`).

`pyproject.toml` changes (verified by `tomllib` parse and by running the suite
with `--strict-markers`): `dev += ["pint>=0.21", "pyscf>=2.3"]`,
`[tool.pytest.ini_options] xfail_strict = true`.
