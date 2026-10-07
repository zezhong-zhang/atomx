# Critical code review — 2026-10-05 (engineering + physics epistemology)

Scope: `src/atomx` (~91,310 lines, 332 files) and `tests/` (~320 files, ~2,624 test
functions). Method: four parallel audits — full reads of the six largest core modules
(~12k lines), repo-wide mechanical grep audit, physics-kernel reuse cross-check, and
test-suite honesty audit. Every claim below was traced to file:line; the three most
severe (`shabaev.py` cached fallback, `matrix.py` global memo + env-tunable audit
cadence, `dirac_fock.py` undeclared `virtual_orbital_keys`) were re-verified by hand
after the audit returned.

Yardstick: the project's own contract in root `AGENTS.md` — fail-closed, no silent
`getattr` fallbacks swallowing physics, shared kernels, measure-don't-assert.

Requested framing: two voices — Linus (engineering bluntness) and Feynman
(first-principles "what do you actually know"). The verdicts are evidence, not style.

---

## Delivery status (2026-10-05)

The findings below record the original dirty-tree audit, including observations
of a separate scientific-development stack. This delivery covers today's review
fixes and documentation. The unfinished XFEL/state/complex-CI stack is excluded.
Line references below identify the reviewed source, not the current layout.

| Finding | Current disposition |
|---|---|
| E1, E3, matrix/cache parts of E8 | Metric audit bound to orbital **and radial** content; nuclear-context-safe reuse; legacy CI removed; shared teardown explicit |
| E2, duplicated portions of E4 | Declared solver controls, shared grid installation and radial-state access; mirrored legacy functional removed |
| E5, E6 | Variational diagnostics declared; numeric energies separated from metadata; legacy archive migration |
| E7, provider part of E8 | Failures propagate; approximations require explicit selection; required rates/weights and nuclear provenance enforced; duplicate ERI scans removed |
| P3–P5 | Convention-bearing names, shared angular/radial/rate machinery, kappa/label helpers and canonical conversion constants |
| P1 | Six kernels on one normalized density, independently anchored; discretization and first-cell limitations retained |
| P2 | Added live frozen-table Monte Carlo survival/conservation gate; historical fixture replay remains historical; independent physical reference coverage is not expanded |
| P6 | Cheap kernel comparison performed; Ne satellite discrepancy attribution remains unresolved |
| E4/E8 remaining design suggestions | Coherent drivers and existing `Atom.solve`/`Atom.result` semantics retained with reasons; B-spline functional/iteration separation stays longer-term work |

Details: [shared physics](SHARED_PHYSICS.md), [solver contracts](SOLVER_CONTRACTS.md),
[test performance](TEST_PERFORMANCE.md) and [validation/source identity](VALIDATION.md).
The original prioritization is historical; this table is the current review status.


## Part I — Engineering findings (ranked)

### Critical

**E1. Process-global mutable caches decide numerical behavior.**
`physics/csf/matrix.py:86-92`: module-level `_SLATER_MEMO`, `_MEMO_KEEPALIVE`,
`_MEMO_CONTEXT`. At `matrix.py:1197`:

```python
_METRIC_CHECK_EVERY = int(os.environ.get("ATOMX_CI_METRIC_CHECK_EVERY", "20"))
```

The orbital-metric validation tripwire is skipped on 19 of 20 CI builds by default,
tunable by an env var read at import time; nothing in a result records whether the
audit ran. In a library whose gates demand bit-reproducibility, correctness auditing
is history- and environment-dependent. (The memo itself is keepalive-pinned and
context-guarded — the cache key discipline is fine; the *audit cadence knob* is the
problem.) Related: `numerics/bspline/convol.py:1404` module-global `_RKB_BANKS`;
`ConvolCache.clear()` (1655-1657) wipes banks shared with other caches on the grid.

*Resolution (2026-10-05): the env knob and count cadence are removed. The audit
is memoised per workspace on an orbital-and-radial-content fingerprint
(`matrix._orbital_metric_fingerprint`): first build and every content change
audit; byte-identical validated content skips. Recorded only after a passed
audit. Gated by `tests/test_ci_memo_reuse.py::TestMetricCheckMemo`.*

**E2. `DFSolver` physics depends on attributes absent from its contract.**
`methods/dirac_fock.py:1692, 2031, 2179` read
`getattr(self, "virtual_orbital_keys", set())`; the attribute is never declared in
`__init__` — it is injected externally by `methods/csf/solver.py:1103`
(`dhf.virtual_orbital_keys = {...}`); same for `core_hole_stationary`
(`dirac_fock.py:2636` writes, `2340/2389` read via getattr default `True`). A direct
`DFSolver(...)` user silently gets default physics. This is the exact
AGENTS.md-documented failure mode ("result 类型动物园" / swallowed `+J[ρ_full]`
radial bug), still live in the flagship solver. Verified: three read sites, zero
declarations, one external injection.

**E3. Known-wrong legacy CI kernel (~1,500 lines) ships in the physics layer.**
`physics/csf/matrix.py:805-2316`: `_legacy_build_ci_matrix_approximate` + ~20
helpers, docstrings admit "quantitatively wrong" / "LANDED, UNVALIDATED". Verified
zero production callers (tests and an opt-in flag only). Contains decoy stubs:
`_chain_one_diff_spectator_exact_scalar` returns `None` "placeholder" (2276-2278),
`_fac_one_diff_open_spectator_exchange_coef` always returns `0.0` as a "reachability
probe" (2284-2300). Documented-as-wrong is not a lock on the door; delete or
quarantine to `tests/oracles/`.

*Resolution (2026-10-05): deleted. `_legacy_build_ci_matrix_approximate` and the
25 legacy-only helpers (including both decoy stubs) are gone from
`physics/csf/matrix.py` (2316 → 945 lines); the shared helpers
(`MissingOrbitalError`, `_get_orb`, `_check_orbitals_present`,
`_check_orbital_metric`, metric memo) were retained. `matrix.py:659`'s
getattr-defaulted J/parity block assignment (E2's matrix.py item) is now direct
attribute access, and the empty-CI branch (E8's second item) returns a (0, 0)
`LinearOperator` when `as_operator=True`. The dead-code element-level tests in
`tests/test_two_electron_replacement.py` were removed with the kernel; its
production-path tests remain.*

### Major

**E4. God methods with string-flag dispatch.** `DFSolver.solve` ~640 lines
(1780-2419, three SCF algorithms interleaved with DIIS/damping/orthonormalization);
`VariationalCSFSolver.solve` ~648 lines (943-1591); `Atom._solve_fixed` ~358 lines;
`Atom.__init__` ~349 lines. `DFSolver.__init__` takes 20 parameters,
`VariationalCSFSolver.__init__` ~22. Predictable consequence — hand-mirrored
duplication: SCF energy functional twice (`dirac_fock.py:1500-1608` vs
`matrix.py:1263-1316`, docstring admits manual sync); physics-grid override block
copy-pasted 4× (`atom.py:1161-1175`, `dirac_fock.py`, `csf/solver.py:1078-1092`,
`csf/solver.py:1813-1828`), each mutating another object's internals; `radial_state`
property verbatim in 3 files (`results.py:501-507`, `dirac_fock.py:150-156`,
`csf/solver.py:200-205`).

*Partial resolution (2026-10-05): the physics-grid override block is now a single
`DFSolver.install_physics_grid(...)` method (`methods/dirac_fock.py`), called by
`atom.py`, both `csf/solver.py` sites and `DiracFock.solve` (which keeps its
`log.basis_info` call on the returned basis data) — the four hand-mirrored
mutations of solver internals are gone. The `radial_state` property now lives
in `RadialStateMixin`, and the mirrored SCF energy functional vanished with the
legacy kernel. Coherent orchestration remains in its owning modules; the planned
B-spline functional/iteration separation remains longer-term work.*

*Further disposition: [solver contracts](SOLVER_CONTRACTS.md) records why the
existing return/accessor contract is retained.*

**E5. `__dict__` metadata bus on result objects.** `methods/csf/solver.py:1471-1590`:
21 `result.__dict__[...] =` writes (`_E_Breit`, `_qed_provenance`,
`_variational_diagnostics`, ...), read back via `self.__dict__.get(...)` (110-126).
`atom.py:1115` and `dirac_fock.py:2468` write private state into result objects they
didn't construct. Undeclared, untyped, undocumented data flow.

**E6. Strings smuggled into a numeric-typed dict.** `dirac_fock.py:2388-2394` writes
`"_core_hole_rotation_mode": "local_stationary_v1"` (and a `nuclear_profile` object)
into `SCFResult.energy_components: Dict[str, float]` — while `atom.py:1183-1188`
documents that string entries in that dict *previously broke consumers*. The rule is
written in one module and violated in the other.

*Addendum (2026-10-05): the leftover `corrections/stack.py` write of the
`_mohr_high_ns_policy` string into `energy_components` now lands in
`result.solver_diagnostics`, and `io/results.py` migrates that key out of legacy
`components_json` archives on load, so the float-only invariant holds for old
files too.*

**E7. Silent fallbacks that change physics — concentrated in the corrections layer.**
Worst site in the codebase, `corrections/qedmod/shabaev.py:209-220` (verified
verbatim): `@lru_cache(maxsize=128)` over

```python
try:
    return hydrogenic_r_exp_integral_dirac(n, kappa, Z)
except Exception:
    return _hydrogenic_r_exp_integral(n, l, Z)   # non-rel analytic
```

A transient solver failure silently substitutes a non-relativistic QED self-energy
reference, and the `lru_cache` makes the degraded answer permanent for the session.
Same class: `corrections/qedmod/mohr.py:79` (PCHIP failure → silent linear
interpolation of the Mohr F(Zα) table);
`scattering/dielectronic_recombination/solver.py:514` (silent screened-hydrogenic
widths); `plasma/crm.py:205` (getattr default silently zeroes a radiative rate);
`calculation_spec.py:78` (missing atomic weight silently selects a point nucleus).
These violate the "unsupported paths refuse loudly" rule at the worst locations.

*Addendum (2026-10-05): two silent-default leftovers swept. `enumerate.py::
_initial_statistical_weight` narrows its config-parse guard from `except Exception`
to `except ValueError` — the only failure its strict NR grammar raises; anything
else now propagates instead of silently deferring to the subshell-occupation
derivation.  `plasma/crm.py::add_dr_satellites` reads `resonance.g` directly: the
documented-required statistical weight previously defaulted to 1.0 via getattr,
silently mis-setting the satellite level's degeneracy.  `channels` stays optional
by design (`AutoionizationState` defaults it to `{}`), with the empty-mapping
fallback onto the required `a_radiative_per_s` now documented in the docstring.*

### Minor

**E8.** `Atom.solve()` returns `self` or an `AtomicResult` depending on arguments
(`atom.py:874-876` vs `1218`). `build_ci_matrix` returns dense `np.zeros((0,0))`
instead of a `LinearOperator` for the empty case (`matrix.py:585-587`).
`ns = len(...) if orbitals_dict else 50` magic fallback (`matrix.py:1491`, `1050`).
In-function import sprawl (atom.py ×47, dirac_fock.py ×41, csf/solver.py ×39) — half
are circular-import workarounds signaling misplaced layering seams; dirac_fock
re-imports `convol` symbols inside four methods despite a module-level import.
`contract.py:665-676` has dead shape checks and runs the ERI finite scan twice
(641, 678).

*Resolution (2026-10-05, matrix.py items only): the empty-CI branch now returns a
(0, 0) `LinearOperator` with `.diagonal()`/`.blocks` when `as_operator=True`;
both `ns = ... else 50` fallbacks vanished with the legacy helpers (E3 deletion).*

---

## Part II — Physics epistemology findings

**P1. Six radial Slater kernels, never cross-checked against each other.**
`numerics/bspline/convol.py` (DBSR cell), `numerics/yk.py` (Poisson),
`numerics/quadrature.py:365` (trapezoid), `numerics/shooting/coulomb.py:235` (SLAG6),
`scattering/r_matrix/cc_potentials.py:28` (sixth Yᵏ), and
`scattering/two_electron/coulomb.py:23-106` (standalone Gauss-Legendre mini-module,
cross-validated against *nothing*). Each kernel is self-checked and energies are
checked end-to-end, but no test pins two kernels against each other on a shared
density. The project's own archive records a +74 Ha error from exactly this class of
integrator substitution (`matrix.py:80-85` docstring). One cheap decisive experiment
— all kernels, one density, same answer — either buys confidence or finds a carried
bug.

**P2. The largest CI cross-check shares its formula with the kernel it checks.**
The m-scheme oracle (`benchmarks/mscheme_ci_reference.py`) is a transcription of the
kernel; a sign error baked into both is invisible to it (honestly documented in
`tests/test_ci_matrix_vs_mscheme_reference.py:7-45`). The genuinely independent
anchor is one layer thick: GRASP native full-matrix comparison at atol 3.7e-10
including off-diagonal interference signs (`tests/test_breit_open_shell_reference.py`),
plus FAC CI matrices and Condon-Shortley closed forms. Additionally, three XFEL test
files gate archived JSON fixture replays — a code regression today would not fail
them.

**P3. Convention collisions are documented but unenforced.** Two functions named
`reduced_C` with different phases: `angular/coefficients.py:35` (Grant jj,
`(-1)^(j_a+1/2)`) vs `scattering/r_matrix/angular_cc.py:52` (LS, `(-1)^l`) — hazard
H6 pinned in `tests/test_angular.py:163-175`, but nothing enforces which convention
a matrix element came from. Same pattern reproducing: two angular factors in one
file (`properties/transitions/calculator.py:441` `angular_eL` vs `:685` `reduced_C`),
verified numerically equal *today* but unpinned. Inline re-derivations of the
Einstein prefactor (`scattering/autoionization/mscheme_ci.py:732`,
`xfel/coherent.py:658-673`) and the static dipole radial integral
(`properties/transitions/fluorescence.py:~57`, `xfel/coherent.py`) instead of the
shared `einstein_A_eL` / `radial_eL`. Each inline copy is a fresh chance to lose a
(2J+1) factor — the failure mode the project already paid six audit rounds for.

**P4. Constants leak with divergent precision.** `27.211386245988` hardcoded ×4 in
`xfel/cascade.py:257,265,277,330` instead of `constants.HARTREE_EV`;
`219474.63` — a *truncated* copy of `HARTREE_CM1` — in
`methods/csf/crystal_field.py:46` and `methods/csf/slater_scaling.py:124`; `hc` in
eV·nm ×4 at different precisions (`uta/moments.py:70`, `spectrum/uta.py:36`,
`spectrum/crm_synth.py:219`, `viz/mpl.py:336`); `HA_CM1` duplicated in
`reference_data/nist.py:15`. Individually harmless; collectively the "convention
before bug" class at the module edges.

**P5. Micro-duplication around the one true kernel.** The Wigner symbols, signed jj
CSF algebra, and B-spline cell integrator are genuinely single-source and well
oracled — but the seams re-implement: CG-from-3j ×3 (`angular/clebsch.py:16` vs
`methods/mbpt_sigma2.py:68` vs inline in `physics/csf/breit_ci.py:130-134` — the
latter in the exact area that produced the open-shell Breit sign bug); fermion
bitmask sign ops ×4 (`angular/mscheme.py:58`, `angular/csf_hamiltonian.py:909`,
`physics/csf/matrix.py:1721`, `scattering/autoionization/mscheme_ci.py:315`); κ→(l,2j)
helpers ×7, with the de-facto shared pair misplaced in
`scattering/distorted_wave/solver.py:40,45`; `"spdfghiklmno"` label tables ×~12 with
one divergent alphabet in `scattering/ci_continuum.py:227`; `_csf_twice_J`
(`calculator.py:137-179`) duplicating `csf_twice_j` (`angular/csf.py:45-70`) as a
fallback in a file that already imports the shared one.

**P6. Strategic observation.** The Ne⁷⁺ q/r campaign has layered increasingly
elaborate machinery (complex scaling, DKB, identity tracking) on top of the
common-orbital CI while P1's cheap decisive experiment has never run. The ~0.22 eV
discrepancy may be profound physics or a mundane kernel discrepancy; it cannot be
attributed until the boring comparisons exist. Do the boring experiment first.

---

## What is genuinely strong (recorded for balance)

- One canonical Wigner implementation (`angular/wigner.py`, doubled-j, exact-integer
  fallback) with sympy + FAC oracles; ~25 production callers, no second
  implementation in `src/`.
- `pyproject.toml:125` `xfail_strict = true` globally — silent xpass impossible by
  construction; every xfail found carries a measured panel; no skip without reason.
- 3 `assert` statements in 91k library lines; prints are deliberate `# noqa: T201`
  user-facing tables; KNOWN GAP banners declare stubs at module top; most broad
  `except` sites re-raise with context or warn-and-record; radial caches are
  content-fingerprinted.
- Test suite: ~2,624 tests, ratio ~0.62 lines-of-test per line-of-source; anchors
  are independent (Sommerfeld closed form re-derived in-file to avoid circular
  comparison, CODATA constants deliberately re-typed in `test_dw_born_completion.py`
  independent of `atomx.constants`). A silent sign error in a two-body matrix
  element would go red through 5-8 files of independent physics. Docstrings record
  rejected circular designs and staleness dates — rare and valuable.
- Root `conftest.py` is clean: BLAS thread pinning, sys.path workaround, no seeds or
  monkeypatching. `benchmarks/` is an oracle harness, zero pytest duplication.

## Test-suite reservations (from the honesty audit)

- 773 private-attribute accesses in tests; `atom._result` alone ×174 — the result
  zoo forces the suite to couple to internals; a rename breaks ~174 sites.
- The three XFEL fixture-replay tests gate past measurements, not current code.
- Several strict xfails mark open physics debts (KLL 10% gate, DW-sum-reaches-Born)
  — compliant with discipline, but they are debts, not coverage.

---

## Prioritized fixes

1. **Kill the `except Exception` + `lru_cache` fallback in `shabaev.py:209-220`** —
   fail loud, never cache a fallback. Then fix `mohr.py:79`,
   `dielectronic_recombination/solver.py:514`, `plasma/crm.py:205`,
   `calculation_spec.py:78`.
2. **Delete or quarantine the legacy CI kernel** (`matrix.py:805-2316`) including
   the decoy stubs.
3. **Declare `virtual_orbital_keys` / `core_hole_stationary` in
   `DFSolver.__init__`**; remove physics-bearing getattr defaults
   (`matrix.py:659` J/parity block assignment included).
4. **One test: all six Rᵏ kernels, one density, same answer.** Cheap, decisive.
5. **Enforce constants/conventions mechanically**: ban `HARTREE_EV`/`HARTREE_CM1`
   literals by lint; rename the dual `reduced_C` so the convention is in the name.
6. Make `_METRIC_CHECK_EVERY` per-build by default (or record the audit decision in
   the result); remove the env-var knob on a correctness gate.
7. Longer term: collapse the hand-mirrored duplicates (energy functional ×2, grid
   override ×4, `radial_state` ×3) behind the shared layer; give results a public
   accessor so tests stop touching `_result`.

---

*Review conducted 2026-10-05. Concurrent-session edits to AGENTS.md during the
review were re-read and accounted for; findings verified against the working tree at
that date. The original audit changed no production code; the delivery status and linked
records above describe the subsequent reviewed fixes and their validation.*


## Lasting knowledge disposition

Trigger → action → reason → scope: when reusing a successful orbital metric
check or one-body block, include radial representation content and the nuclear
context; object identity alone misses in-place changes. This strengthens the
existing cache/metric rule rather than introduces a second physical contract.
When a test needs only algorithmic correctness, use the smallest complete space
that retains phases, interference and symmetry; keep large external/transfer
claims in explicitly named campaigns. This is the user's test-design decision,
not a performance measurement. No new root rule or independent physics insight
is promoted. Failed checks and the kernel limitations remain registered below.
