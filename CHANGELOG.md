# Changelog

All notable changes to atomx are recorded here.  Versions follow the value of
`atomx.__version__` (`src/atomx/__init__.py`), which is the single source of
truth for the package version (`pyproject.toml` reads it via hatchling).

## Unreleased — correctness, validation-integrity and hygiene pass

Driven by `docs/CRITICAL_REVIEW_2026-09-17.md`; see
`docs/CRITICAL_REVIEW_2026-09-17_FIXES.md` for the finding-by-finding report.

### Fixed — validation integrity

* The He-like Z=10 M1 benchmark measured the wrong line.  It took
  `max(A_rate)` over the CI space, which in that CSF set is a transition near
  1010 eV, not the 2³S₁ → 1¹S₀ tracer at 904.7 eV.  So the "6.19e6 = 0.88 ×
  Lin et al." reported in July was never the tracer: selected by transition
  energy, the tracer reads 3.94e6 s⁻¹ = **0.56 × Lin** and has been there all
  along (3.970e6 before the 2s²/2p² configurations were added, 3.940e6 now).
  The gate is split into a passing regression pin on the energy-selected
  tracer and an explicit `xfail` for the Lin comparison, which needs the full
  LS-coupled Breit–Pauli M1 operator.

### Fixed — physics

* **Two-electron replacements now couple in the CI — all three patterns, any
  J.**  The CSF-based CI handled only the two-distinct-replacement
  Slater–Condon pattern; a single-subshell pair move (a² ↔ b²) fell through to
  zero, so `Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"])` returned the *un-mixed*
  2s² state at the HF energy (−14.5731746 Ha) and the classic 2s²/2p²
  near-degeneracy correlation — and the jj pair 6p½² ↔ 6p3/2² *inside* one
  configuration — was missing.  One exact m-scheme element now covers every
  total J and all three patterns:
  a² ↔ b² (2s²/2p², jj pairs), a² ↔ c¹d¹ (2p² → 3d¹4d¹) and a¹b¹ ↔ c².
  Be becomes −14.6139143 Ha with mixing [0.944, 0.282, 0, 0, 0, 0.171] (0.041
  of the 0.094 Ha correlation energy); Pb 6p² converges in 8 s with
  [0.94, 0.342, 0] and no limit cycle — the stability concern that had
  deferred this channel does not reproduce.  Validated element-by-element
  against the FAC-ported m-scheme determinant engine on identical radial data
  (`tests/test_two_electron_replacement.py`), at J = 0 and J = 2, for the pair
  and split patterns.  Guarded: applied only when every spectator subshell is
  closed (otherwise the active pair's J is not determined and the element
  returns 0).
* **`matrix._rk` computed different integrals in its two engine flavours.**  The
  grid branch paired its arguments as `(arg1, arg3) × (arg2, arg4)` while the
  B-spline branch (and all nine call sites, whose docstrings state
  `R^0(ac, bd)`) pair `(arg1, arg2) × (arg3, arg4)`.  Any off-diagonal CI
  element built through `_rk` on the grid flavour therefore used the wrong
  radial integral — and `tests/test_ci_grid_kernel.py::test_rk_pairing_matches_gk`
  had pinned the buggy pairing.  Both flavours now agree; the test asserts the
  callers' convention.
* The MCDHF Guerra occupations now reflect that admixture; two tests that had
  frozen "2s occupation ≈ 2.0" were replaced by the consistency check
  `occ(2s½) = 2·(1 − Σ|c_p²|)`.

### Fixed — numerics

* **Out-of-bounds reads in the B-spline cell integrals** (the cause of the
  intermittent `NaN`).  `_convol_case2_mixed` contracted a density with
  `ns = max(nsp, nsq)` rows while the density only had `nsp` (or `nsq`) rows,
  so the mixed P/Q Slater terms read one row past the end of the array: with
  numba bounds checking off that read returns heap memory, which is then
  multiplied by a structurally zero `rkb` entry — harmless when the garbage is
  finite (which is why most runs were correct, and why no recorded energy
  changed) and a poisoned `nan` integral when it is not.  That site is fixed
  by HYP-1590; this change fixes the **remaining** sites with the same defect —
  `_density_sym`, `_density_full` and their `_accumulate_*` twins read past the
  shorter of their two components, which the callers hit whenever the P/Q
  basis orders differ under kinetic balance (`P[:ns_full]` is shorter than
  `ns_full`) — so out-of-range reads are now zero, i.e. the shorter component
  is zero-padded.
  Symptom before the fix: `Atom(Z=10, numerics="bspline")` and the README's
  `Atom(Z=4, configs=[...])` example returned `nan` or raised
  `ValueError: array must not contain infs or NaNs` in ~30 % of fresh
  interpreter processes (with bounds checking on, 100 % of runs raised
  `IndexError`).  No recorded energy changed: the pinned self-regression
  snapshot and every known-answer test reproduce bit-identically, because the
  offending term was multiplied by zero.
* **`_SLATER_MEMO` was unsound by construction.**  The CI integral memo keys on
  `id()` of the orbital coefficient arrays but only pinned them at one of six
  sites, so a freed array's address could be reused and a stale entry returned.
  Keyed arrays are now retained for the memo's lifetime and the memo is bound
  to a single radial workspace (`_memo_context`), which also drops entries if
  the same orbitals are re-used on a different grid.
* `_density_full` / `_density_sym` / `_convol_case2_mixed` docstrings now state
  the extent conventions that the callers rely on.

### Added — reproducibility gates

* `scripts/check_determinism.py` — runs a physics panel in N fresh interpreter
  processes and compares digests, plus a `--boundscheck` mode that runs the
  panel with `NUMBA_BOUNDSCHECK=1` (turning an undefined read into a
  deterministic `IndexError`).
* `tests/test_determinism.py` — pytest gates wrapping the above (the
  bounds-check gate is marked `slow` and runs in the nightly workflow).
* `.github/workflows/nightly.yml` — runs the `slow` suite, the `book` claims,
  the bounds-check gate and a 12-process determinism check; none of these ran
  in any automation before.

### Changed — validation honesty

* "FAC parity" gates are split by what they actually assert: `external_fac`
  (stored external reference data) and `snapshot` (atomx self-regression).
  `fac_parity` is gone — it named a self-regression snapshot as parity.
* The FE XVII Auger "parity panel" no longer compares the FAC fixture against
  hard-coded copies of its own values.
* Calibrated/synthetic CRM assertions are skipped or removed rather than
  presented as agreement; the Au NLTE-9 band is no longer asserted without the
  disclosure that its model is hand-tuned.
* README §Verification was rewritten to say, per row, what is gated, whether
  the gate recomputes or reads a recorded reference, and which claims have no
  executing gate (`docs/VERIFICATION.md`).
* Lint: `ruff` now selects real rules (`E4`,`E7`,`E9`,`F`,`I`,`B`,`SIM`, with
  `E741` ignored because `l` is the orbital angular-momentum quantum number),
  covers `tests/` and `scripts/` as well as `src/`, is pinned to one ruff
  version in CI and pre-commit, and runs with `--no-cache` (a stale cache was
  observed reporting "All checks passed!" without checking anything).
  Two `F821` undefined-name defects were fixed.
* The shooting "bit-identical" pin used exact float equality against a baseline
  recorded on another environment and failed by 1 ULP (~6.5e-16 relative) on
  every run here; it is now a documented 1e-13 relative pin plus an exact
  same-process repeat assertion, which is the property the pin was for.

### Changed — packaging, dependencies and CI

* Version is single-sourced: `pyproject.toml` is `dynamic = ["version"]` from
  `src/atomx/__init__.py` (previously `0.1.0` in metadata vs `0.2.0` in code).
* Only genuine runtime imports remain hard dependencies (`numpy`, `scipy`,
  `numba`, `h5py`, `matplotlib`); `manim`, `plotly`, `nbformat`, `ipython` and
  `ipykernel` moved to the `viz*` / `notebook` extras (nothing in `src/`
  references the last three).  Extras no longer carry floors below the base
  requirement they overlap with.
* The sdist no longer ships local agent caches (`.code-review-graph/graph.db`,
  which contains absolute paths); build outputs and caches are excluded.
* CI: Python matrix (3.10 floor + 3.13, the version used in local
  development), `uv sync --locked` so CI tests the committed lock, timeouts,
  `permissions`, `concurrency`, uv caching, and `--strict-markers` so a typo'd
  `-m` selector cannot silently select nothing.
* `.gitignore`: the blanket `benchmarks/*` rule (which made `git add` a silent
  no-op for the 40+ tracked files under `benchmarks/`) is replaced by targeted
  ignores for generated outputs, and caches/coverage artefacts are covered.  It
  also pointed at a `docs/superpowers/...` path that does not exist.
* Machine-specific absolute paths (`/Users/...`, `/home/...`) in benchmarks and
  scripts moved behind environment variables (`FAC_SFAC_PATH`, `FAC_DYLD_LIB`,
  `RADIAL_DHFS_DIR`, `ATOMX_MCDHF_REF_CSV`, `ATOMX_WF_TXT_DIR`) with actionable
  error messages; `fac_runner` no longer falls back to a private checkout path.
* `.pre-commit-config.yaml` runs ruff (same selection as CI) plus the standard
  whitespace/large-file/merge-conflict hooks, not just the perf dashboard.

### Removed

* Leftover empty package trees `src/atomx/prop/` and `src/atomx/analysis/`
  (residue of the `prop` → `properties` rename, shipped by the wheel builder).
* Dead modules with no reachable reference anywhere in `src/`, `tests/`,
  `scripts/` or `benchmarks/`: `scattering/radiative_recombination/rydberg.py`,
  `scattering/radiative_recombination/milne.py` (both third and fourth copies
  of physics already implemented in `radiative_recombination/__init__.py` and
  `methods/hydrogenic_cross_sections.py`) and
  `scattering/electron_ionization/rates.py`.
