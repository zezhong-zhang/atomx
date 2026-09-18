# What atomx actually verifies

This file exists because the README's verification table described *results*
without saying what was *gated*.  Every row below states the gate, whether the
gate recomputes atomx or merely reads a recorded reference, and what is not
covered.  If a claim has no executing gate, it is listed as such — do not move
it back into the README table without adding one.

## Markers

| marker | meaning |
|--------|---------|
| `external_fac` | the test asserts atomx against **stored external reference data** (clean-room FAC fixtures in `tests/data/`, FAC dumps in `benchmarks/fac_dumps/`, recorded FAC columns in `benchmarks/results_*.csv`). A pass means "atomx now agrees with a recorded external number". Regenerating the reference needs a live `sfac` (see below). |
| `snapshot` | the test asserts atomx against **atomx's own recorded output** — a drift detector, not validation. |
| `slow` | expensive; excluded from the default `pytest` run (`-m "not slow"`). CI runs these in `.github/workflows/nightly.yml`. |
| `book` | executable claims from `docs/book`; run with `pytest -m book`. |

`pytest --strict-markers` is on, so a typo'd marker/selector fails instead of
silently selecting nothing.

## Current gates

| claim | gate | recomputes? | notes |
|-------|------|-------------|-------|
| Ground-state DHF/DFS energies (Ne, Ar, Fe XVII, Kr…) | `tests/test_scf.py`, `tests/test_dfsolver_fac.py`, `tests/test_shooting_hf.py`, `tests/test_fac_dfs_shooting.py` | **yes** (atomx values pinned as regression) | `external_fac` where a stored FAC value is asserted |
| Numerical reproducibility (identical results in fresh processes) | `tests/test_determinism.py`, `scripts/check_determinism.py` | **yes** | Added 2026-09-18 after a nondeterministic NaN (see CHANGELOG) |
| No out-of-bounds reads in the numba kernels | `scripts/check_determinism.py --boundscheck` (`pytest -m slow`) | **yes** | Runs with `NUMBA_BOUNDSCHECK=1`; converts an undefined read into a deterministic `IndexError` |
| Hydrogenic/Sommerfeld energies, CODATA constants | `tests/test_dirac_hydrogenic.py`, `tests/test_constants.py` | **yes** | known-answer, literature |
| Hydrogen 2p→1s A coefficient, 1s hyperfine | `tests/test_transitions.py`, `tests/test_hyperfine.py` | **yes** | NIST/CODATA reference values |
| Continuum waves (shape, √(2/π) normalisation) | `tests/test_continuum_vs_fac.py` | **yes**, against stored FAC `WaveFuncTable` dumps | |
| DW collisional excitation Ω (He-like Z=10/26/54) | `tests/test_fac_dw_benchmarks.py` | **yes** — recomputes the grid and asserts both the recorded FAC column and the recorded atomx column | Live-measured 2026-09-18: rms 3.3 %, max 5.2 % (gates 5 %/10 %); live-vs-recorded-atomx ≤2.6 % (the Z=54 column of the CSV is 2.2–2.6 % stale). Regenerating the FAC column needs `sfac` |
| DW electron-impact ionisation (K-shell) | `tests/test_fac_dw_benchmarks.py` | **yes**, same structure | Live-measured: rms 2.0 %, max 3.1 % (gates 3.5 %/6 %); live-vs-recorded-atomx ≤2e-8 |
| Fe XVII first 10 levels vs live `sfac` | `tests/test_ci_J_block.py` | **yes** — runs `sfac` at test time | Skips with an actionable message when no *runnable* `sfac` is configured (`SfacUnavailable` distinguishes a dyld/loader failure from a physics failure); set `FAC_SFAC_PATH` + `FAC_DYLD_LIB` to arm it |
| Auger widths, Fe XVII KLL per level | `tests/test_auger_fac_panel.py`, `tests/test_auger_kll_vs_fac.py` | **yes** — recomputes atomx live for the five strongest KLL resonances | Measured live/FAC = **1.13–14.8** (2026-09-17), so the strict 10 % per-resonance gate is `xfail` and the "1.005–1.023 vs FAC" figures in `docs/scattering_fac_verification_2026-07.md` are a development note, not a verified claim. Four further fixture levels cannot be recomputed (no atomx channel decomposition / outside the modelled channel) and are asserted for presence only |
| DR radiative widths and α(T) (Fe XVII dr33) | `tests/test_dr_manifold_e2e.py` (`slow`), `tests/test_dr_vs_fac.py` | **yes**, against the clean-room `tests/data/fe17_dr33_fac_cleanroom.json` fixture | Kernel parity is exact (1 %); the end-to-end ratio carries the documented ~40 eV E_res offset |
| J-resolved CI line strengths | `tests/test_ci_line_strengths.py` | **yes**, against a clean-room FAC `TRTable` fixture with recorded provenance | |
| Angular conventions (3j/6j/9j, FAC `angular.c` literals) | `tests/test_fac_angular_conventions.py`, `tests/test_angular.py` | **yes** | |
| Shooting DHFS vs Salvat `radial`/DHFS (≤2e-8) | `benchmarks/bench_shooting_vs_radial.py` | live, but requires the external code | Set `RADIAL_DHFS_DIR`; the script errors clearly if unset. No CI gate (external binary) |
| Local-exchange DFS vs `sfac` (shooting, `exchange="gu"`) | `tests/test_fac_dfs_shooting.py` (`external_fac`) | **yes** — recomputes atomx against the recorded `sfac` calibration panel | `docs/plans/2026-04-20-fac-calibration/calibration.csv`, 13 rows sampled over Z = 2…92. Measured 2026-09-17: worst 6.3e-4 for Z ≤ 10, **4.1e-3 at Z = 36** — the "≤ 1e-3 over the full panel" claim in `benchmarks/REPORT.md` is not met and is now flagged there |
| Local-exchange DFS vs `sfac` (benchmark script) | `benchmarks/bench_local_exchange.py` | live, requires `sfac` | Set `FAC_SFAC_PATH` / `FAC_DYLD_LIB` |
| Structure vs FAC, Z=2…92 | `benchmarks/bench_fac_structure.py` | live, requires `sfac` | **Read the caveats**: the parity figure is relative to *total* energy (7 keV absolute at Z=82) and the comparison runs atomx with `breit=False, qed=False` by default while FAC's default has QED on. No CI gate |

## Recorded-reference data in the repository

| path | used by | regenerated by |
|------|---------|----------------|
| `tests/data/fe17_*.json` | Auger/DR/transition tests | `benchmarks/bench_fac_dr.py`, `bench_fac_levels_transitions.py` (need `sfac`) |
| `tests/data/facwave/*.txt` | continuum shape tests | FAC `WaveFuncTable` dump |
| `benchmarks/fac_dumps/*.{lev,tr,ce}` | `tests/test_fac_io_roundtrip.py` (`slow`) | `sfac` |
| `benchmarks/reference/fac_matrix_2026-05-26.csv` | `tests/test_fac_parity_gates.py` (`snapshot`) — **atomx columns only** | `benchmarks/fac_matrix.py`. The `fac_*` columns disagree with atomx by up to ~9 orders in `first_A_s` (unreconciled units/definition) and are **not asserted** |
| `benchmarks/results_fac_dw_*.csv`, `results_fac_dr.csv` | `external_fac` gates | the corresponding `bench_*.py` + `sfac` |

## Not verified by anything

* **CRM / plasma benchmarks** (Au, Fe, W cooling curves, charge-state balance,
  NLTE-9 ⟨Z⟩): the committed "atomx" curves are synthetic placeholders
  (documented as such in `benchmarks/bench_fac_crm_*.py`), and the tests that
  asserted agreement with them are skipped with the algebra of why the
  assertion was vacuous.  A real CRM validation needs published CHIANTI /
  NLTE-9 reference data plus a per-stage atomx CRM run.
* **Self-energy total shift**: the Mohr backend is anchored analytically at
  Z=1 (`tests/test_qedmod.py`), but nothing pins the total QED shift for a
  many-electron atom against an external value.  `tests/test_qed.py::
  TestQEDTotalShiftPin` now pins atomx's own total shift — Ne +0.027863 Ha,
  Ar +0.330528 Ha, i.e. self-energy + Uehling VP with **no** Wichmann–Kroll
  in the default stack — as a self-regression pin.  The same quantity differs
  from FAC's model by ~0.85 eV in `benchmarks/results_fac_structure.csv`; one
  of the two numbers is substantially wrong and no external reference is
  available here to settle it.
* **g-factors**: `tests/test_book_preface.py` now computes the Si13+ g factor
  with atomx and compares to the Sturm-2011 theory value at 1.5e-6 relative
  (gate 5e-6).  Claims of "a few parts in ten billion" are literature-only —
  atomx's g accuracy is ~1.5e-6, not 1e-10.
* **Absolute total energies at high Z** (7 keV vs FAC at Z=82) — see the
  structure row above.

## Two-electron (double-replacement) CI coupling

`Atom(configs=[...])` couples configurations that differ by a two-electron
replacement.  All three patterns and every total J are handled by one exact
m-scheme element (`methods/mcdhf/matrix.py::_two_electron_replacement_element`):

| pattern | example |
|---------|---------|
| a² ↔ b² | 2s² ↔ 2p² (the near-degeneracy pair), 6p½² ↔ 6p3/2² (a jj pair *inside* one configuration), at J = 0 and J = 2 |
| a² ↔ c¹d¹ | 2p² ↔ 3d¹4d¹ (a double excitation into two different orbitals) |
| a¹b¹ ↔ c² | the transpose of the above |

Validation (`tests/test_two_electron_replacement.py`): the element is compared
against the FAC-ported m-scheme determinant machinery (`ResonanceManifold`, the
engine validated against FAC AITable) on identical radial data and the same
grid, for s/p/d/f pairs with equal and different κ at J = 0 and J = 2, for
both the pair and the split pattern.  The two sides agree to the radial
quadrature difference (≤3e-4 for d/f trial functions, ≤5e-5 for s/p); the
determinant side additionally self-checks that each state it builds is a J²
eigenstate.  End-to-end pins cover Be 2s²/2p² (mixing and the 0.041 Ha drop
below HF), a J = 2 pair channel in C-like 2p²/3p² and the split channel in
2p²/3d¹4d¹.

Guarded, deliberately: the element is applied only when every *spectator*
subshell is closed, so the two active electrons are in a definite J state.  With
open spectators the parentage is unknown and the helper returns 0 rather than
guessing.

**Pairing convention of `matrix._rk`** (fixed 2026-09-18): `_rk(a, b, c, d)`
pairs its first two orbitals and its last two, ``R^k(ab, cd)``; callers pass
the roles they mean, e.g. ``_rk(Pa, Pc, Pb, Pd)`` for ``R^k(ac, bd)``.  The
grid branch paired ``(arg1, arg3)``/``(arg2, arg4)`` instead, so the two
engines evaluated different integrals for the same call — the grid-flavour CI
was wrong wherever an off-diagonal element went through it.  Both flavours are
now pinned by tests (`test_ci_grid_kernel.py::test_rk_pairing_matches_gk`,
`test_two_electron_replacement.py::test_rk_pairing_convention_bspline`); note
that the earlier version of the former asserted the buggy pairing.

The CI still should not be described as "complete correlation": a
two-configuration MCDHF recovers part of the correlation energy (Be: 0.041 of
the 0.094 Ha), not all of it.

## Regenerating external references

```bash
export FAC_SFAC_PATH=/path/to/sfac        # FAC_BIN / FAC_BIN_DIR also accepted
export FAC_DYLD_LIB=/path/to/gcc/runtime  # macOS only, if sfac needs it
export RADIAL_DHFS_DIR=/path/to/radial/DHFS   # Salvat DHFS benchmark
python benchmarks/bench_fac_dw_excitation.py   # etc.
```

Without these variables the benchmark scripts stop with an actionable message
rather than silently skipping or using a path from someone else's machine.

## CI coverage

| job | what runs |
|-----|-----------|
| `.github/workflows/test.yml` | `pytest -m "not slow"` on Python 3.10 + 3.13, `uv sync --locked --all-extras`, plus `scripts/check_determinism.py --repeat 4 --panel full` |
| `.github/workflows/lint.yml` | `ruff check --no-cache src tests scripts benchmarks` (pinned ruff) |
| `.github/workflows/nightly.yml` | `pytest -m slow`, `pytest -m book`, the `NUMBA_BOUNDSCHECK=1` gate and a 12-process determinism run |
