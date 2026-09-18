# Critical Review — atomx

Date: 2026-09-17
Scope: whole repo (`src/atomx` 216 files / 55.4k LOC; `tests/` 134 files / 22.1k LOC;
`benchmarks/`, `scripts/`, `docs/`, CI, packaging).
Method: independent adversarial read + **execution**. Every claim below is either a
file:line citation or a command output reproduced on this machine
(macOS/arm64, Python 3.13.2, numba 0.65, numpy/scipy current, editable install).
The full non-slow suite was run (321 s). Nothing in the repo was modified
(`git status --porcelain` → empty).

Prior reviews exist and were read (`CRITICAL_REVIEW.md`,
`docs/CRITICAL_REVIEW_2026-07-09.md`, `docs/FEYNMAN_REVIEW_2026-07-15.md`). This
review is deliberately **not** a re-run of them: it (a) verifies their `[FIXED]`
claims, and (b) reports what they missed. Where a prior claim does not survive
verification, that is stated explicitly.

---

## Verdict

The physics core is real, and several of the July fixes are genuine. The problem is
no longer "the code is wrong everywhere" — it is that **the evidence layer is much
weaker than the documentation implies, and the code is not reproducible**:

1. 🔴 **The public multi-configuration API returns NaN in ~30 % of fresh processes**
   (23 failures / 77 fresh processes measured), and the non-slow suite is **red on the
   documented dev workflow**. Traced to the B-spline Slater cell kernels returning NaN
   from provably finite inputs.
2. 🔴 **The headline "verified against FAC" claims are graded on frozen records, not on
   live computation.** The DW-excitation/EII "parity gates" never import `atomx` at
   all; the FE XVII Auger "parity panel" compares the FAC fixture against constants
   copied from that same fixture; the README's Z=2..92 gate points at a file that does
   not exist.
3. 🟠 **Accuracy is reported in the metric that hides errors.** The repo's own committed
   FAC benchmark disagrees by **7 keV** at Z=82 (shown as "~1.5 %"), the comparison runs
   with different physics flags on each side, and the single largest correction applied
   by default (self-energy, +0.82 eV for Ne) has no value assertion beyond the
   analytic Z=1 anchor.
4. 🟠 Structural debt the July review did not close: ghost package trees, 832 LOC of
   unreachable modules, **9** independent R^k implementations, a 3-cycle in the import
   graph, and an `id()`-keyed memo cache that is unsound by construction.

Severity: 🔴 correctness / reproducibility / integrity · 🟠 design · 🟡 hygiene.

---

## 🔴 Critical

### C1. `Atom(configs=[...])` produces NaN nondeterministically; the suite is red

**Reproducer** (README's own quick-start example, unmodified):

```python
from atomx import Atom
a = Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"])   # README.md:37
a.solve()
```

* Failure rate: **23 of 77 fresh interpreter processes (~30 %)** across 8 measurement
  batches. It fails **serially, in isolation, with a warm JIT cache** (3/12, 1/12 with
  `PYTHONHASHSEED=0`, 5/15 with instrumentation). Ten repeats *inside* one process never
  failed → the nondeterminism is **per process**, not per call.
* Failure mode: an opaque `ValueError: array must not contain infs or NaNs` raised by
  `scipy.linalg.eigh` five frames deep
  (`atom.py:486 → methods/mcdhf/solver.py:631 → methods/ci.py:158 → ci.py:233`).
* The NaN enters the CI Hamiltonian: instrumenting `scipy.linalg.eigh` shows a 3×3
  (J, parity) block whose **whole diagonal is NaN** while the off-diagonals are fine.
* Localisation: wrapping the matrix builders names the producer —
  `matrix._fk(...) → convol.compute_slater_fk_cell(...)` returns NaN for `k = 0`.
* Cause is *inside* that kernel, not upstream: I verified every input is finite —
  the four orbital arrays `Pa, Qa, Pb, Qb` (all shape `(113,)`) and **every float array
  in `bsp_data`** (`bsp_p, bsp_p_d, bsp_q, bsp_q_d, gr, grw, t, tp`) are all-finite, yet
  the integral comes back NaN. Process-dependent garbage from finite inputs, with
  identical inputs, points at an out-of-bounds / uninitialised read (or a
  thread-scheduling-dependent path) inside the numba kernels; the cell kernels carry
  **34 `fastmath=True`** decorators and many `np.empty` work buffers
  (`numerics/bspline/convol.py:54,89,146-155,240,280,381`).
* **The suite is red.** `pytest -m "not slow"` (the CI command) on this machine:

  ```
  1 failed, 1571 passed, 34 skipped, 15 deselected, 1 xfailed, 1 xpassed, 3 errors in 321.31s
  ```

  The 3 errors are `tests/test_ci_J_block.py` with **exactly this ValueError** (they pass
  3/3 when the file is run alone → the flake is load/ordering sensitive). The failure is
  `tests/test_shooting_guerra.py::test_no_jj_configs_bit_identical[Ne]`:

  ```
  Ne 1s1/2: -32.8174667451201 != -32.81746674512012
  ```

  i.e. a **1-ULP** mismatch against a hardcoded baseline — reproducible here, and either
  (a) the "bit-identical" pin is machine/BLAS-specific, or (b) the last-bit instability
  is the same nondeterminism in another suit. Both readings are bad for a tool whose
  README promises bit-reproducible accuracy.

Why it matters: this is the documented README entry point, it corrupts the *flagship
multi-configuration feature*, and the observable symptom depends on luck — a fixable
crash here, but the same uninitialised memory could equally yield a **wrong number with
no exception** elsewhere. Anyone running atomx as an HPC job array
(`srun`, `xargs -P`, `pytest-xdist`, an embarrassingly-parallel scan) will see ~1 in 3
jobs die or, worse, silently emit garbage. `docs/CRITICAL_REVIEW_2026-07-09.md` §A5
diagnosed an intermittent NaN as a "cold first-ever numba-cache compile race" and
concluded "warm operation is clean"; **that diagnosis does not hold** — the failure
reproduces warm, serially, and without compile contention.

Fix order: (1) `NUMBA_BOUNDSCHECK=1` / `numba.objmode` bisect on `compute_slater_rk_cell`
to find the read; (2) delete `fastmath=True` from every kernel that feeds a physics
observable (it also makes bit-reproducibility unattainable); (3) make the CI/CI-matrix
path fail loudly with a domain error instead of leaking numpy's message; (4) add a
30-process determinism test to CI (`for i in $(seq 30); do python -c ...; done`).

### C2. The "verification against FAC" claim is backed by frozen records, not computation

README.md:123-141 asserts point-by-point FAC verification and says
"`benchmarks/` regenerates these live … `pytest -m fac_parity` gates them". What the
gates actually do:

* **`tests/test_fac_dw_benchmarks.py` contains no `atomx` import at all** (`grep -c 'from
  atomx\|import atomx'` → 0). It reads
  `benchmarks/results_fac_dw_excitation.csv` / `..._eii.csv` and asserts properties of
  those files (`rel_diff ≤ 0.10`, row coverage, positivity). The docstring says "the
  physics parity gate is the test below" — the test below is also a CSV reader. So the
  README's **"Ω rms 3.7 %, every point ≤ 6.7 %"** and **"EII ≤ 6 %"** rows cannot fail
  if `dw_excitation_cross_section` is deleted. They are edit-detectors for a CSV.
* **`tests/test_auger_fac_panel.py` is a tautology.** `REFERENCE_PANEL`
  (`:16-22`) is compared against `tests/data/fe17_dr_fac.json` — I checked all five
  values: relative error **0.000 %** for every one (they were copied from the fixture).
  The test never calls atomx; its assertion message prints the FAC number labelled
  `atomx=…` (`:58`). The README's **"Auger widths (Li-like KLL, per level) 1.005–1.023"**
  therefore has no executing gate. The real comparison is `xfail(strict=False)`
  (`test_auger_kll_vs_fac.py:155`) and its live gate is a **4-decade** band
  `1e13 … 1e17 s⁻¹` (`:59-60,148`) against FAC ≈ 3e14 — i.e. the honest state of Auger
  accuracy is "right order of magnitude", not "0.5 %".
* **A headline gate is permanently dormant.** `tests/test_fac_dfs_shooting.py:20-27`
  loads `docs/superpowers/plans/2026-04-20-fac-calibration/calibration.csv`; that path
  does not exist (the real directory is `docs/plans/...`), so `_load_reference()` returns
  `[]` and the test self-skips. The "≤ 1 × 10⁻³ over the full Z = 2..92 panel"
  structure-parity claim that this gate was supposed to enforce lives in
  `benchmarks/REPORT.md` (TL;DR table row 6) and is backed by nothing while the
  path is wrong. *(Corrected 2026-09-18: the review first attributed this claim to the
  README; it is `benchmarks/REPORT.md`. With the path fixed the gate runs and measures
  worst 6.3e-4 for Z ≤ 10 and 4.1e-3 at Z = 36 — so the ≤ 1e-3 claim does not hold on the
  sampled panel either, and `REPORT.md` now says so.)*
* **A calibrated tautology is still asserted.** `tests/test_au_nlte9_workshop.py:87-93`
  skips the Au NLTE-9 gate and *admits* `charge_fractions` is a hand-tuned Gaussian — but
  `:109-123` asserts the same ±20 % band with no skip against
  `bench_fac_crm_Au.py:232-235`'s `center = 50 + 7·(Te−0.5)/4.5 − 0.18·log10(ne/1e20)`.
  Measured ratios at the four canonical points: 1.019 / 0.966 / 0.923 / 0.851 — all pass
  by construction.
* The July "honesty pass" was real (the synthetic CRM models and the self-regression
  snapshot are now prominently labelled), but it converted *fraud* into *skip*, and
  left the README's parity table unchanged. `fac_parity` still names a marker whose
  own description says it "does NOT assert the fac_* columns"
  (`pyproject.toml:52`; `benchmarks/reference/fac_matrix_2026-05-26.csv` records
  `atomx_first_A_s = 3.59e6` vs `fac_first_A_s = 6.85e-3`, ~9 orders).

Why it matters: the project's central selling point is external validation. Right now
the validation is a set of committed CSVs and fixtures that nothing regenerates, plus
one test that compares a file to itself. A green `pytest -m fac_parity` currently means
"some CSVs still contain the numbers they contained in July".

Fix: (a) delete `test_auger_fac_panel.py` or make it call `auger_rate_csf_average` and
compare to the fixture; (b) move the frozen-CSV gates into a `nightly` job that
**regenerates** the CSV first (sfac or a pinned container), and mark the CSV-reading
variant `snapshot`; (c) rename the marker to `fac_reference` / `self_regression` and
split it; (d) delete or fix `test_fac_dfs_shooting.py`; (e) rewrite README §Verification
to state, per row, whether the number is live-recomputed or frozen, and drop the rows
with no live gate.

### C3. Accuracy is denominated in total energy — the metric that hides the errors

`benchmarks/results_fac_structure.csv` (committed) vs README's "Structure: Z≤36 ≤ 0.3 %;
Z=82 ~1.5 % (QED-model differences)":

| Z | ion | atomx (eV) | FAC (eV) | Δ (eV) | Δ/atomx |
|---|-----|-----------:|---------:|-------:|--------:|
| 82 | Ar-like | −437 028.1 | −444 036.2 | **7 008** | 1.6 % |
| 74 | Ar-like | −347 789.4 | −351 765.9 | 3 976 | 1.1 % |
| 54 | Ar-like | −173 801.5 | −174 596.1 | 795 | 0.46 % |
| 36 | Ne-like | −63 137.6 | −63 246.5 | 109 | 0.17 % |
| 10 | Ne-like | −128.664 | −128.635 | 0.85 | 0.02 % |

Three problems:
1. A relative-to-total metric turns a **7 keV** disagreement into "1.5 %". The entire
   QED correction for Pb-like ions is ~1 % of that; the README's explanation
   ("differences in how the two codes parameterise high-Z self-energy — a physics
   choice, not a bug") cannot carry 7 keV.
2. The comparison is **not like-for-like**: `bench_fac_structure.py:129` defaults
   `breit=False, qed=False`, while FAC's default is QED-on. The row is atomx-DHF
   (+QED when the flag is passed) vs FAC-default, and the difference is dominated by the
   missing Breit, not by QED.
3. Total energy is the wrong figure of merit for atomic data. Nothing in the repo
   benchmarks **energy differences** (transition energies, ionization potentials,
   Kα shifts) against FAC across a panel, although that is the product users consume.

Related: the largest correction atomx applies **by default** is unpinned. For Ne,
`Atom(Z=10).solve()` = −128.6136 Ha, built from:

| term | value (Ha) | value (eV) | known-answer test? |
|------|-----------:|-----------:|--------------------|
| Breit | +0.050437 | +1.372 | pinned ±5 % (`test_scf.py`) |
| Uehling VP | −0.002355 | −0.064 | pinned ±5 % |
| Wichmann–Kroll | +5.7e−7 | ~0 | none |
| **self-energy** (`mohr`, the Z=10 default) | **+0.030218** | **+0.822** | Z=1 analytic anchor only (`test_qedmod.py`) |
| *(self-energy, `shabaev_local` backend)* | *+0.028161* | *+0.766* | none |

*(Corrected 2026-09-18: the default stack for Ne resolves to the `mohr` backend
— `resolve_se_method("auto", 10) == "mohr"` — which gives +0.030218 Ha; the
+0.028161 Ha figure first quoted here is that adapter's own default backend,
`shabaev_local`. The stack total, SE+VP+WK = +0.027863 Ha, reproduces the
measured `Atom(breit=False, qed=True)` shift exactly.)*

The self-energy is 12× the VP term and has no value assertion anywhere: its
only anchor is the analytic Z=1 case (`test_qedmod.py:56-62`, F(α)=10.3169);
`tests/test_qed.py:71-87` passes **all-zero wavefunctions** (`np.zeros(50)`) and asserts
only `E_SE > 0` and `10 < ratio < 25`; `test_qed.py:93-101` asserts He Breit `< 0.01 Ha`
(an order-of-magnitude band). The FAC structure CSV says the two codes' QED models
differ by 0.85 eV at Ne — 104 % of atomx's own QED shift. One of those two numbers is
substantially wrong, and the repo has no test that would notice. (Minor, same file:
`test_qed.py:73`'s docstring says a positive self-energy "increases binding", but the
measured sign convention is the opposite — +0.0302 Ha makes the total *less* negative.)

---

## 🟠 High

### H1. `_SLATER_MEMO`: a cache keyed by `id()` that cannot be sound

`methods/mcdhf/matrix.py` memoises R^k / one-body integrals in a module-global dict keyed
by Python object identity: `:67` `_fk`, `:84` `_gk`, `:100` `_rk` (four ids), `:255`
`_grid_pair_density`, `:293` `_h_one_body`. Only `_grid_ds` (`:244-248`) keeps the keyed
array alive — with the comment `# keep y alive: id-reuse safety`, which proves the hazard
was known and fixed at exactly one of six sites. `methods/quadrature.py:123,186` do the
same with `(id(r_grid), k)`.

Consequences: (a) a freed array's address can be reused by a different orbital and the
stale entry is returned as a finite, plausible, **wrong** integral; (b) the memo is
global and never cleared, so entries leak across `Atom` instances in one process
(cross-contamination between a Ne solve and a later Be solve is exactly the ordering in
which I first saw C1's NaN); (c) unbounded growth in long-lived processes. I probed
`_grid_pair_density` for stale hits across 15 processes and found none **in that
particular run** — this is a latent soundness bug, not a demonstrated wrong answer today.
Fix: key on `(k, kappa, n, array_id)` with a strong reference, or key on the orbital
labels/coefficient arrays, or drop the memo and use `functools.lru_cache` on a
hashable key. Never key a physics cache on `id()`.

### H2. The README's multi-configuration example does not do what it says

Beyond crashing ~1/3 of the time (C1), when it does run the advertised mixing is
invisible: `["1s2 2s2"]` → −14.573175213282465 Ha and
`["1s2 2s2", "1s2 2p2"]` → −14.573174629472758 Ha, a **5.8e−7 Ha** difference. Either the
example is meaningless (the two CSFs are orthogonal, so nothing mixes) or the CI is not
coupling them. `README.md:37` sells this as "Multi-configuration (Be with 2s² and 2p²
mixing)". Same section: `Atom(Z=26, charge=16, …, numerics="shooting", exchange="gu")`
is advertised at "~50 ms" and measures **139 ms** here (2.8×) — acceptable drift, but
it shows the performance table is not re-measured anywhere.

### H3. The only static-analysis gate cannot fail

`pyproject.toml:45` `select = ["T201"]` (flake8-print) and CI runs `ruff check src`;
there are **zero** T201 violations, so the lint job is a constant `All checks passed`.
With real rules the same tree has ~1600 findings: F401 unused imports 210, UP 725,
E501 177, I001 158 — including a genuine **`F821 Undefined name 'TwoElectronTerm'`** at
`src/atomx/scattering/r_matrix/target_states.py:46` (a quoted forward reference to a class
that is never imported). `line-length = 88` (`pyproject.toml:41`) is dead
config (no enabled rule uses it, `ruff format` never runs), and `tests/`, `scripts/`,
`benchmarks/` are never linted. This is the mechanism that let the ghost trees and dead
modules accrue. `.pre-commit-config.yaml` has one hook (the perf dashboard), is not
installed here, and is not run by CI.

### H4. Ghost trees, dead modules, 9 R^k implementations, a 3-cycle

* `src/atomx/prop/` (8 subdirs) and `src/atomx/analysis/` contain **no `.py` files** —
  untracked residue of the `prop → properties` rename, invisible to `.gitignore`, and
  shipped by hatchling because it packages `src/atomx` wholesale. `import atomx.prop.stark`
  resolves as an empty namespace package (`ModuleNotFoundError` on the submodule).
* **832 LOC of modules with no reachable reference** anywhere in `src/`, `tests/`,
  `scripts/`, `benchmarks/`: `scattering/autoionization/jj_level.py` (272),
  `dielectronic_recombination/fac_reference.py` (212), `two_electron/coulomb.py` (153),
  `radiative_recombination/{rydberg.py,milne.py}` (69+67 — third and fourth copies of
  physics that is also inline in `radiative_recombination/__init__.py` and
  `methods/hydrogenic_cross_sections.py`), `electron_ionization/rates.py` (59).
* **R^k / Slater integrals exist in 9 places** (`numerics/bspline/slater.py`,
  `slater_yk.py`, `convol.py ×2`, `numerics/quadrature.py`, `numerics/shooting/solver.py
  ×2`, `scattering/two_electron/coulomb.py`, `scattering/autoionization/auger.py`) plus
  `_rk` re-implementations in `methods/mbpt.py`, `methods/mcdhf/matrix.py`,
  `scattering/autoionization/mscheme_ci.py`. They diverge physically: the orphaned
  `coulomb.py:88` hardcodes `r_max = 50.0` with `n_points=2000` for all Z. `mcdhf/matrix.py:26`
  imports the *private* `_slater_integral_slag6` across a package boundary. This matters
  for C1: nine kernels means nine places an uninitialised read can hide.
* **Import graph has a 3-cycle**: `methods → properties` (`methods/edm.py:30`,
  `methods/pnc.py:31` — thin pass-through shims whose only purpose is to launder a type),
  `properties → scattering`, `scattering → methods` (`distorted_wave/solver.py:34`,
  `dielectronic_recombination/enumerate.py:25`); `providers ↔ scattering` likewise, which
  contradicts ADR-0001's "boundary" claim. 289 function-local imports hide this from
  static tooling.
* `Atom.__init__` takes **24 parameters, none keyword-only** (`atom.py:162`), validates
  string enums at runtime (16 `raise ValueError` in one file), and tracks
  `self._averaging_explicit` / `self._numerics_explicit` to distinguish "default" from
  "explicitly passed the default" — a design smell with a body count.

### H5. Packaging and metadata drift

* **Two versions, both live**: `pyproject.toml:8` `0.1.0`, `src/atomx/__init__.py:14`
  `__version__ = "0.2.0"`; the installed wheel metadata says 0.1.0. **0 git tags**, no
  CHANGELOG, no LICENSE file and no license field (README says "see pyproject.toml for
  project metadata" — it isn't there).
* **The sdist ships a local AI-agent artifact**: `uv build` → `atomx-0.1.0.tar.gz`
  contains `.code-review-graph/graph.db` (160 KB, absolute paths) because that directory
  self-ignores via an *untracked* nested `.gitignore` that hatchling does not honour.
  The wheel itself is clean (219 files, all `.py`).
* Heavy runtime deps that nothing imports: `nbformat`, `ipython`, `ipykernel` (0 hits in
  `src/`), plus `plotly`/`matplotlib`/`manim`, which are lazily imported and guarded
  (`viz/__init__.py:35-42`). `manim` needs system pangocairo, which is why CI installs
  `libcairo2-dev libpango1.0-dev` — so the documented `pip install -e .` is never tested
  as written. The extras are *weaker* than the base requirements (`plotly>=5.0` extra vs
  `>=6.7.0` base), so `[viz]` can never raise a floor.
* `.gitignore` gaps verified with `git check-ignore`: `.benchmarks/`, `.mypy_cache/`,
  `.coverage`, `htmlcov/`, `.ipynb_checkpoints/`, `*.orig`, `*.rej`, `nohup.out`,
  `.code-review-graph/`. `.gitignore:17` protects
  `docs/superpowers/plans/.../fac_data/` — a path that does not exist (the real one is
  `docs/plans/...`), so the intended protection is void. `.gitignore:18` ignores all of
  `benchmarks/*` with 4 negations, so 40+ tracked benchmark files cannot be re-added
  (`git add` is a silent no-op).
* `.git` is **82 MB with 2210 loose objects** (pack is 4.85 MiB) — never `gc`'d. No
  binary ever entered history (the largest blob is `uv.lock`); this is pure hygiene.

---

## 🟡 Medium

* **Tolerances sized to the systematic.** `tests/test_auger_mscheme_ci.py:37` hardcodes
  `FAC_TRACE_J72 = 5.357e14` "from the fixture" while the fixture's J=7/2 subset sums to
  6.405e14 (19.6 % higher) — and the gate is `rel=0.20`, so "different AI-cut subset" and
  "number copied from a previous atomx run" are indistinguishable.
  `test_ci_line_strengths.py:121` accepts A ratios in [0.85, 1.20] while its own docstring
  says the basis difference is 10–25 %. `test_dr_manifold_e2e.py:143` allows 0.35–1.45
  where its docstring claims ≤17 %. No *pathological* tolerance exists (loosest
  `pytest.approx` is 0.35), but several gates are sized to the defect they should catch.
* **A test that always skips.** `tests/test_eii.py:388` calls `pytest.skip()` as the last
  statement, so the test reports "skipped" even after asserting.
  `tests/test_W_cooling_curve_vs_putterich2008.py` is an entire file whose only test is
  skipped, so `tests/data/putterich2008_W_cooling_digitized.csv` is asserted by nothing.
* **34 skipped + 15 deselected tests, and 15 `slow` tests are never run by any
  automation** (`.github/workflows/test.yml:29` is `pytest -m "not slow"`; no `schedule:`
  trigger exists). The excluded set includes `test_M1_E2_rates.py`, `test_mcdhf.py`,
  `test_eii.py`, `test_rr_vs_fac.py` and `test_fac_io_roundtrip.py` — the only consumer of
  the `benchmarks/fac_dumps/` fixtures.
* **CI hardening**: no `permissions:`, `concurrency:`, `timeout-minutes:`, uv cache,
  `--locked`/`--frozen` on `uv sync` (so CI may re-resolve away from the committed lock),
  no coverage/type-check/`pip-audit`/`pytest --strict-markers`, actions pinned to mutable
  tags, single Python version (3.10) although the local dev stack is 3.13 — the version
  where I reproduced C1.
* **Environment-specific paths in tracked files** (10 files): `benchmarks/fac_runner.py:21`,
  `bench_local_exchange.py:31`, `bench_shooting_vs_radial.py:32`, `scripts/density_gs.py:13`,
  `scripts/energies_gs.py:45`, `docs/plans/2026-04-20-fac-calibration/extract_AHX.py:24`,
  plus `/home/zzhang/miniconda3/...` in `benchmarks/RESULTS_fac_dw_eii.md:13,21`.
* **AI-process artifacts as first-class tracked root files**: `trace.md` (42 KB session
  log), `PROJECT_INDEX.json` (stale since 2026-04-21), `PROJECT_INDEX.md`,
  `CRITICAL_REVIEW.md`, `docs/CRITICAL_REVIEW_2026-07-09.md`,
  `docs/FEYNMAN_REVIEW_2026-07-15.md`. None is a build input or a test fixture.
* Internal ticket IDs (`HYP-####`) appear 21× in `src/`, 40× in `tests/`, 51× in `docs/`.
* `docs/book/PLAN.md:376` promises per-chapter notebooks and a `figures/` directory gated
  by `pytest -m book`; the real gate is 4 tests in one file that call no atomx code
  (`tests/test_book_preface.py:36-53` assert relations among hardcoded literals).

---

## What is genuinely good (verified, not assumed)

* **Real external validation does exist** — it is just narrower than the README implies:
  `test_ci_line_strengths.py` (clean-room FAC TRTable fixture with recorded provenance and
  the full sfac deck), `test_dr_manifold_e2e.py` (α_DR + ΣA_r vs the clean-room dr33 set),
  `test_continuum_vs_fac.py` (continuum shape constant to 1.5 %, √(2/π) normalisation vs
  raw FAC dumps), `test_fac_angular_conventions.py`, `test_constants.py` (CODATA).
* **Strong internal invariants**: CI sum rule vs determinant trace at rel=1e-10
  (`test_ci_line_strengths.py:183,191,220`), cross-engine CI agreement at 1e-6
  (`test_ci_grid_kernel.py`), `convol.py` cross-validated against the independent Y^k path
  (`test_convol.py:149-186`), Sommerfeld/hydrogenic known answers, H 1s hyperfine
  = 1420.4 MHz, H 2p→1s A = 6.27e8 s⁻¹ against NIST/FAC.
* **The July remediation is largely real**: `prop → properties` (except the ghost
  directories), single-sourced `c`/`a0`/`HREV`/`REV` (`numerics/shooting/constants.py:14-25`
  now derives from `atomx.constants`), the Auger Slater O(N²)→O(N) change, the
  `prange`-on-tiny-work fix in `bspline/convol.py:695-703`, and the honest labelling of the
  synthetic CRM benchmarks and the self-regression snapshot.
* **`numerics/` has no upward imports** (`numerics → constants` only); `angular/` is
  dependency-free and single-sourced; the `Correction`/`CorrectionResult` and
  `scattering/protocols.py` seams are real.
* **4 TODO / 0 FIXME / 0 XXX / 0 HACK** in 55k LOC, all four TODOs being honest prose
  caveats. `NotImplementedError` is used as fail-closed, never as a silent zero.
* **No secrets anywhere** (tree or history); no PDF was ever committed; `uv lock --check`
  is in sync; the wheel is clean and installable; 1625 tests collect with zero collection
  errors.

---

## Remediation plan (ordered by damage per unit effort)

**Do first — correctness & reproducibility (blocks every accuracy claim)**

1. **Kill C1.** Bisect `compute_slater_rk_cell` with `NUMBA_BOUNDSCHECK=1`; add a
   30-fresh-process determinism gate to CI; drop `fastmath=True` from physics kernels.
   Acceptance: 30/30 identical results for the README Be example, no NaN in 200 runs.
   *(Effort: hours-to-days; without it, every number in the repo is provisional.)*
2. **Fix H1** (`id()`-keyed memo) before it turns into a silent wrong answer.
3. **Make the suite green on the documented dev stack** (3.13) and pin the bit-identity
   test to a tolerance that survives platform/BLAS differences — or delete it and keep a
   *relative* reproducibility check instead of an exact-float one.

**Then — integrity (cheap, high credibility gain)**

4. Delete `test_auger_fac_panel.py` or make it compute; delete the calibrated Au NLTE-9
   assertion; delete or fix `test_fac_dfs_shooting.py`; rename the `fac_parity` marker and
   split `external_fac` from `snapshot`; add a scheduled job that **regenerates** the
   DW/EII/DR CSVs before gating them; add `--strict-markers` and a scheduled `-m slow` job.
5. Rewrite README §Verification as a table with a "live-recomputed / frozen / no gate"
   column, and drop the parity rows that have no executing gate. Add the absolute-eV
   column to the structure claim, or replace it with transition-energy benchmarks.
6. Give the self-energy a known-answer test (H-like 1s, Z=1/10/54/92 vs a published
   table) and pin the *total* QED shift for Ne and Ar.

**Then — structure (do these only after 1-3)**

7. `rm -rf src/atomx/prop src/atomx/analysis`; delete the 832 LOC of unreachable modules;
   add `ruff select = ["E4","E7","E9","F","I","B","SIM"]` + `ruff format --check` to CI
   **before** refactoring, and lint `tests/` and `scripts/` too.
8. Consolidate the 9 R^k paths onto two sanctioned entry points
   (`numerics/bspline/slater.py` for basis pairs, `numerics/quadrature.py` for callables).
9. Collapse `methods/{edm,pnc}.py` into `properties/`, then break the
   `methods ↔ properties ↔ scattering` cycle and revisit ADR-0001 (whose boundary the code
   violates).
10. Replace `Atom.__init__`'s 24 positional parameters with typed config objects; remove
    the `_*_explicit` flags. Single-source `__version__`, cut a tag, add LICENSE, fix
    `.gitignore`, `git gc`.

**Guardrails to add so this does not recur**

* CI: `--strict-markers`, `--locked`, coverage floor, a "no empty directory under
  `src/atomx`" check, and a no-network packaging test (`uv build` + install + import in a
  clean venv).
* A `docs/VERIFICATION.md` that lists every external claim, its gate, and whether the gate
  recomputes or reads a frozen record — the README table should be generated from it.

---

## Appendix — reproducers

```bash
# C1: NaN in the README multi-config example (~30% of fresh processes)
for i in $(seq 20); do
  python -c "from atomx import Atom; a=Atom(Z=4, configs=['1s2 2s2','1s2 2p2']); a.solve(); print(a.total_energy)"
done

# C1: pinpoint the producer (wrap matrix._fk / convol.compute_slater_fk_cell
# and assert np.isfinite on the inputs — inputs are finite, output is NaN)

# C2: the DW/EII "parity gates" never touch the library
grep -c "from atomx\|import atomx" tests/test_fac_dw_benchmarks.py     # -> 0

# C2: the Auger parity panel is self-comparison
python - <<'PY'
import json; d=json.load(open('tests/data/fe17_dr_fac.json'))
ref={24:2.7544e13,32:4.1004e14,62:6.4310e13,135:9.8863e13,150:1.2721e13}
for r in d["resonances"]:
    lvl=int(r["label"].split(":")[0].removeprefix("FAC ILEV "))
    if lvl in ref: print(lvl, abs(float(r["gamma_auger_per_s"])-ref[lvl])/ref[lvl])  # all 0.0
PY

# C2: the Z=2..92 gate points at a nonexistent path
python -m pytest tests/test_fac_dfs_shooting.py -q      # -> 1 passed, 1 skipped

# C3: absolute FAC disagreement at Z=82 (from the committed CSV)
python -c "import csv;r=list(csv.DictReader(open('benchmarks/results_fac_structure.csv')));\
print(sorted(((abs(float(x['diff_eV'])),x['Z'],x['iso']) for x in r),reverse=True)[:3])"

# C3: the default QED breakdown for Ne
python - <<'PY'
from atomx import Atom
from atomx.corrections.adapters import (BreitCorrection, UehlingVPCorrection,
                                        WichmannKrollVPCorrection, SelfEnergyCorrection)
a=Atom(Z=10, breit=False, qed=False); a.solve()
for c in (BreitCorrection(), UehlingVPCorrection(), WichmannKrollVPCorrection(), SelfEnergyCorrection()):
    print(c.name, c.evaluate(a._result).energy_shift)
PY

# Suite status on the documented dev workflow (Python 3.13 venv, warm cache)
python -m pytest -m "not slow" -q
# -> 1 failed, 1571 passed, 34 skipped, 15 deselected, 1 xfailed, 1 xpassed, 3 errors
```
