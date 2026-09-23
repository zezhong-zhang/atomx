# Changelog

All notable changes to atomx are recorded here.  Versions follow the value of
`atomx.__version__` (`src/atomx/__init__.py`), which is the single source of
truth for the package version (`pyproject.toml` reads it via hatchling).

## Unreleased — correctness, validation-integrity and hygiene pass

Driven by `docs/CRITICAL_REVIEW_2026-09-17.md`; see
`docs/CRITICAL_REVIEW_2026-09-17_FIXES.md` for the finding-by-finding report.

### Changed (breaking) — the `Atom` constructor has ten arguments, one per physical question

* **One canonical surface, no aliases.**
  `Atom(Z, config=None, *, charge=0, nuclear="fermi", exchange="hf",
  corrections=None, correlation=None, engine=None, basis=None,
  environment=None)`.  The 28 flat keywords are gone, and so are the six
  grouped dataclasses (`System`, `ElectronicStructure`, `Basis`,
  `Interaction`, `CIProtocol`, `Environment`) that briefly sat on top of
  them.  Each physical question now has exactly one spelling:

  | argument | carries |
  |----------|---------|
  | `Z` | nuclear charge (required, positional) |
  | `config` | one reference configuration, or a *list* for the MCDHF/CI path (was `config=`) |
  | `charge`, `nuclear` | net ion charge; nuclear charge distribution |
  | `exchange` | model name, or `Exchange(model=..., alpha_x=..., slater_method=...)` |
  | `corrections` | preset `"none"`/`"breit"`/`"qed"`/`"breit+qed"`, or `Corrections(breit=, qed=, qed_backend=, stack=)` |
  | `correlation` | `"s"`/`"sd"`, or `Correlation(expansion=, averaging=, radial_optimization=, j_resolved_csfs=, per_block_orbital_sets=)` |
  | `engine` | `"bspline"`/`"shooting"` (renamed from `numerics`) |
  | `basis` | `Basis(ns=, ks=, r_max=, grid_hi=, grid_he=, grid_hmax=, starting_potential=, sampling=)` |
  | `environment` | `Environment(plasma_ne=, plasma_Te=, screening="debye"/"stewart_pyatt")` |

* **Retired names fail helpfully.**  `Atom(10, configs=[...])`,
  `Atom(10, numerics="shooting")`, `Atom(10, breit=False)`, `ns=`, `alpha_x=`,
  `averaging=`, `plasma_ne=`, … each raise a `TypeError` naming the
  replacement and pointing at `docs/MIGRATION.md`; an unknown keyword lists
  the ten valid ones.  Gated name-by-name in
  `tests/test_atom_input_surface.py`.
* **`Z` and `config` are the only positional slots.**  The old signature had
  28 positional-or-keyword parameters appended after `Z`, so a stale
  positional call would have silently bound `charge` as `ns`.
* **`alpha_x` and `slater_method` moved onto `Exchange`**, `breit`/`qed`/
  `qed_backend`/`correction_stack` onto `Corrections`, the CI protocol
  (`averaging`, `radial_optimization`, `j_resolved_csfs`,
  `per_block_orbital_sets`) onto `Correlation`, and the basis onto `Basis` —
  so a knob that only exists for one model or one engine travels with it
  instead of sitting next to it.
* `atomx.Exchange` / `atomx.Correlation` now name the constructor value
  objects; the legacy enums used by the *method* API stay available as
  `atomx.types.Exchange` / `atomx.types.Correlation` and are still accepted
  as `Atom(exchange=...)` values.
* `Atom.provenance` reports `{"engine": ..., "exchange": ..., "averaging":
  ...}` (the first key was `"numerics"`), plus `"plasma"` when the atom sits
  in one.
* Migration: 236 `Atom(...)` call sites rewritten in 83 files under `src/`,
  `tests/`, `benchmarks/` and `scripts/`, five more in the runnable `docs/`
  scripts, and five sites reached through `**kwargs` dicts
  (`CO`, `kwargs`, `params`, `kw`) by hand.  `tests/test_atom_input_surface.py`
  was rewritten from scratch (it pinned the retired design).  The full
  old→new table and one worked example per concern are in
  `docs/MIGRATION.md`; `docs/gallery/atom_inputs_example.py` is a runnable
  tour of the new surface.  Left as historical records (their snippets show
  the API of their date): `docs/CRITICAL_REVIEW_*.md`,
  `docs/PHYSICS_NUMERICS_REVIEW_*.md`, `docs/CORRELATION_SD_RESEARCH_*.md`,
  `docs/plans/`, `trace.md` and the past-release entries of this file.

### Fixed — constructor inputs that were accepted and then ignored

Four arguments were recorded on the atom and never reached a solver.  Each
now changes the physics; the numbers below are measured by
`tests/test_atom_inputs_wired.py`.

* **`plasma_ne`/`plasma_Te` (now `environment=Environment(...)`) screen the
  nucleus inside the SCF.**  The Debye (or Stewart–Pyatt) potential from
  `atomx.plasma.screening` is added to the nuclear potential as
  `delta_V(r) = V_model(r) - (-Z/r)`: finite at the origin, tending to
  `Z/lambda_D` there, so every orbital moves up by the ionisation potential
  depression.  Measured on Ne²⁺ at `ne = 1e22 cm^-3`, `Te = 100 eV`
  (`lambda_D = 14.05` Bohr, `Z/lambda_D = 0.712` Ha): every orbital energy
  rises by 0.685–0.691 Ha, the total energy by **+5.566 Ha (151 eV)**, and
  the shift agrees with first-order `⟨delta_V⟩` on the unscreened orbitals
  (0.708 Ha) to 2.4 %.  Both engines agree on the shift to 1e-5 Ha, and the
  wired shift is bit-identical to `debye_potential(...) + Z/r`.  Before this
  change a "plasma" run returned the vacuum number exactly.  A *neutral*
  atom at these densities is predicted to be pressure-ionised (nuclear
  screening leaves `V_eff -> (N - Z e^{-r/lambda})/r > 0`); the B-spline
  solver reports that instead of returning the core.
* **`Sampling.r_max` reaches the radial box.**  `Basis(r_max=...)`, and
  `Basis(sampling=Sampling(r_max=...))` for a tier that sets one, now decide
  the box the orbitals live in (a tier field left at its schema default still
  means "use the engine's own auto box").  Measured: the hydrogen 6s orbital
  does not fit the default B-spline grid (50.7 Bohr) and comes out 0.29 eV
  too high at −0.003078 Ha; `Basis(r_max=120.0)` reproduces the
  box-independent shooting eigenvalue −0.0138886 Ha to 2.6e-7 Ha.
* **`Sampling.n_grid` reaches the shooting grid.**  `Basis(sampling=
  Sampling(n_grid=...))` sets the SGRID point count; measured 1.39e-8 Ha on
  He between `n_grid=500` and the engine default 1000.  `solve(sampling=...)`
  keeps its documented role (tol/max_iter only).
* **`qed_backend` reaches multi-configuration runs.**  `MCDHFSolver` used to
  resolve the default `"auto"` policy regardless, so
  `Corrections(qed_backend="welton")` was silently the MSS backend on every
  CI run.  Measured on the Ne `1s2 2s2 2p6 + 1s2 2s2 2p5 3s1` expansion:
  `E_SE` 0.012119 → 0.017502 Ha, total energy −128.663067 → −128.657684 Ha
  (5.38e-3 Ha = 0.146 eV).  The default is unchanged (`auto` still resolves
  to MSS/Mohr for Z ≤ 36), and the backend now reaches the shooting flavour
  of the Guerra-CI route as well.
* **`slater_method` is rejected on the B-spline engine** instead of being
  ignored: the Galerkin path evaluates its local-exchange Slater integrals
  with its own quadrature and never read the knob.  It remains available on
  the shooting engine (`Exchange(model="slater"|"gu", slater_method=...)`),
  where it selects the Slater-integral backend.  No call site in the tree
  passed a non-default value on the B-spline engine.
* Everything else is bit-identical: the canonical form reproduces the
  pre-change numbers exactly for the closed-shell (Ne), open-shell
  CI-averaged (N), multi-configuration (Be), shooting-engine and
  Breit+QED panels.
* `docs/gallery/atom_inputs_example.py` is a runnable tour of the new
  surface, including five rejected combinations and the error each raises.

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

* **`mbpt2_energy` summed every cross-κ virtual pair twice, doubling that
  channel of the MP2 sum.**  The virtual double loop iterated the full
  Cartesian product over κ blocks; the antisymmetrised amplitude obeys
  |A_J(ab;sr)|² = |A_J(ab;rs)|² with Δε symmetric, so the ordered channels
  (κ_r,κ_s) and (κ_s,κ_r) contribute identically.  Only κ_s at or after
  κ_r in the block list is now summed (same-κ blocks keep their
  `orb_r_idx ≤ orb_s_idx` guard).  Measured on Ar I: E2 −1.057787 →
  −0.682274 Ha — verified by kept-side reconstruction (same-κ −0.306761 +
  2× cross-κ −0.375513 = −1.057787 Ha exactly, matching the recorded
  pre-fix number to all six digits).  −0.682 Ha is ~97 % of argon's
  ~−0.7 Ha total correlation, a conventional MP2-overshoot-scale term.
  The `|E2| < 0.35` gate in
  `tests/test_nist_levels.py::test_ar_mbpt2_correlation_energy_is_physical`
  stays `xfail` — the heuristic itself now predates the fix and replacing
  it is left to a later pass.  A `term_recorder` diagnostic hook was added
  so any second-order sum can be attributed to its dominant channels; see
  `docs/test_audit/PHYSICS_ar_mbpt2.md`.  Side benefit: the virtual-channel
  loop work is halved (3.2 M of the 4.0 M pre-fix terms were duplicates).

* **`build_ci_matrix` now evaluates exact `⟨CSF_r|Ĥ|CSF_s⟩`.**  The
  configuration-average diagonal plus two ad-hoc "J-correction" patches are
  gone; the CI matrix is built by an exact m-scheme determinant construction
  (`src/atomx/methods/mcdhf/exact_ci.py`) projected back onto the CSF basis as
  `C† H_det C`, with `CSF` carrying the coupling tree it needs.  This closes
  finding N1 of `docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`.  Measured on
  atomx's own orbitals, against the independent oracle
  `benchmarks/mscheme_ci_reference.py`:

  | Be space | protocol | before | after |
  |---|---|---|---|
  | singles (29 CSF) | `fac_frozen` | −15.400463 | −14.575893 |
  | SD (399 CSF) | `fac_frozen` | −16.358105 | −14.632497 |
  | SD (399 CSF) | `mcdhf` | −15.052683 | −14.619112 |

  The oracle used as the reference had its own bug - it dropped the exchange
  integral whenever the direct one vanished by angular selection, proven by a
  rotational-invariance violation.  Its first-published targets (−14.595092
  singles, −14.654637 SD) were artefacts and are corrected above.
  `correlation="sd"` previously fell 0.385 Ha *below* the exact
  non-relativistic Be ground state (−14.66736 Ha) under the default protocol
  and 1.69 Ha below it under `fac_frozen`; both now respect the variational
  bound.  Elements between CSFs of different `(J, parity)` are now exactly
  zero instead of non-zero-but-masked.

  Two new independent gates: an orbital-free Slater–Condon identity,
  `E(¹P₁) − E(³P₁) = 2G¹(1s,2p)/3`, whose ratio was a constant **3.533** at
  Z = 2 / 6 / 10 and is now **1.0000 / 1.0019 / 1.0119**
  (`tests/test_ci_analytic_slater_condon.py`), and a NIST-anchored Be I level
  comparison (`tests/test_ci_levels_vs_nist.py`).  Fe XVII against a live
  `sfac`: rms 73.2 → 51.4 eV, max 135.0 → 87.2 eV.

  **The orbitals are still wrong** — see the R7 limitation in
  `docs/VERIFICATION.md`; excited-state absolute energies remain off until
  that is fixed.

* **The CI matrix is now the Hamiltonian (`correlation="s" | "sd"` is
  variational).**  `build_ci_matrix` evaluated a configuration-average
  diagonal plus two hand-rolled "J-correction" patches, which is not a
  representation of the Hamiltonian: Be `correlation="sd"` returned
  −15.05 Ha, 0.39 Ha *below* the exact non-relativistic ground state
  (−14.66736 Ha), and the builder returned non-zero elements between
  different total J.  It now expands every CSF over the Slater determinants
  of its own configuration, builds the determinant Hamiltonian by explicit
  Slater-Condon rules, and projects: `H_CSF = Cᵀ H_det C`
  (`src/atomx/methods/mcdhf/exact_ci.py`, §8 option A of
  `docs/CORRELATION_SD_RESEARCH_2026-09-20.md`).  CSFs now carry their
  sequential coupling tree (`CSF.coupling`), which `enumerate_jj_csfs`
  emits one-per-coupling-path — the CSF count and the per-`2J`
  multiplicities are unchanged.  After the fix (fac_frozen, breit/qed off):
  Be ref+2s2p −14.5758915, Be singles −14.5758932 (Brillouin: a singles
  expansion on near-HF orbitals buys nothing), Be SD fac_frozen −14.6324974,
  Be SD mcdhf −14.6191124 — all equal to the independent m-scheme oracle to
  ≤2e−14 on identical orbitals, and all above the exact ground state.
  `Atom(Z=4, correlation="sd")` now returns −14.616668 Ha (default flags
  include Breit + QED) instead of −15.052683.  The old assembly survives as
  `matrix._build_ci_matrix_approximate` for comparison; there is no
  fallback to it.
* **The m-scheme oracle was dropping exchange integrals.**
  `benchmarks/mscheme_ci_reference.py` skipped the exchange integral
  whenever the *direct* one vanished by angular selection rules
  (`if v == 0.0: continue`).  That broke rotational invariance — the Be
  `1s2 2s1 2p1` J=1 level moved between the 2M=0 and 2M=2 blocks — and
  made every open-shell off-diagonal number in the N1 document 19–22 mHa
  too low.  Fixed, and the invariant is now an explicit test.
* Gates whose pinned numbers came from the old kernel were updated, not
  widened: Be/C pair-channel totals, the Co `averaging="ci"` baselines, the
  DW excitation threshold record, and the Fe XVII / He-like-Z=2 gates (the
  latter two are now strict xfails carrying the measured numbers and
  pointing at the still-open orbital defect, §3.7).

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

* **The `Y^k` Poisson cumulatives used a half-weight estimator for the
  straddled knot interval, biasing every `Y^k` node value.**  The outward/
  inward truncated integrals in `evaluate._compute_yk_flat` added "full Gauss
  weight for every node strictly inside the limit, plus half the weight of
  the node at the limit" — a first-order stand-in for the partial-interval
  integral, which for a degree-`2ks-2` B-spline density sampled at
  `n_gauss = ks` nodes is not determined by the data the rule uses.  The
  bias was systematic, `O((dr/r)^2)`, and was the whole of the observed
  ~3e-5…6e-4 Slater error (root-cause diagnosis preserved in
  `docs/test_audit/PHYSICS_yk_node_values.md`).  Replaced by the
  **interpolatory partial-interval rule** (`evaluate._yk_partial_matrices`):
  within each knot interval the truncated integrals are those of the
  degree-`n_gauss-1` interpolant of the node values (precomputed `ng×ng`
  matrices, built once per `n_gauss`), across intervals the exact full-cell
  Gauss sums — same grid, same nodes, same weights, so no snapshot is
  re-pinned by the rule change itself.  Measured on the same grid:

  | case | before | after |
  |---|---|---|
  | hydrogenic 1s, k=0, Z = 1/10/26 (closed form 5Z/8) | 2.84e-5 | ≤2.2e-16 |
  | ρ = e⁻ʳ, k = 0…3 (mpmath 40-digit / ln 4) | 1.37e-4 … 4.17e-3 | ≤6.3e-12 |
  | Ne DHF 1s–1s, k = 0…3 (fine-grid reference) | 2.80e-5 … 6.27e-4 | 3.2e-8 … 1.2e-7 |

  The last row equals the exact-partial arbiter of the diagnosis (§7) to
  all its digits — the residual is the fine-grid reference's own error.
  All `compute_yk` consumers (DHF Hartree potential, Slater integrals,
  distorted-wave, autoionization, local potentials) shift at the
  1e-5…1e-4 level; that shift is the defect leaving.  The poisson-vs-direct
  agreement is no longer machine-tight — the direct double sum keeps its
  own 2D-quadrature error, now measured at 1.2e-5…5.0e-4 on the Ne fixture —
  so the cross-kernel gate drops to rel=1e-3 and the absolute anchor moves
  to non-shared gates: the hydrogenic closed form tightened from rel=1e-4
  to rel=1e-8, plus a new 40-digit mpmath gate for ρ = e⁻ʳ k = 0…3 and
  matrix-construction invariants, all in
  `tests/test_slater_integrator.py`.  Kernel cost rises ~9× inside the
  partial loops (O(nv·ng²) vs O(n)) but the measured end-to-end effect is
  noise: a full Ne DHF solve takes 0.220 s with the fix vs 0.214 s without.
  The `numerics/quadrature.py` trapezoid family (`yk_on_grid`,
  `_yk_kernel_rho`, `_slater_rk_kernel`) shares the rule class and is
  deliberately left for a follow-up.
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

### Changed — performance

* **The B-spline RKB banks are built once per solve instead of five times.**
  `ConvolCache` was per-`DFSolver` instance and `mcdhf/solver.py` builds a new
  solver on every MCDHF outer iteration, so all 20 `(k, itype)` banks were
  rebuilt 5x with bit-identical content — 13.8 % of the wall time.  A
  process-wide registry keyed on the *grid content* (not `id()`, since the
  five `bsp_data` dicts are distinct objects with identical content) takes the
  builds per solve from **100 to 20** and a Fe-like `ns=113` solve from
  **9.56 s to 7.32 s (23 %)**, bit-identically.  `build_ci_matrix` also no
  longer drops the stashed `_convol_cache` across `_reset_memo()`, which its
  own docstring four lines above forbade.
* **`fastmath=True` is now applied only where it measurably pays.**  It was
  on 20 kernels; on 16 of them it was neutral or *slower* (up to 1.45x slower)
  while silently breaking bit-reproducibility — and two docstrings claimed
  bit-identity that the flags contradicted.  Kept on the four that earn it
  (3.9x, 3.4x, 2.2x, 1.2x), dropped from the rest: end-to-end **5.8 %
  faster**, energy drift 1.06e-14 relative, and `_density_sym` is now
  verifiably `array_equal` to the plain band construction.
* **A missing orbital in the CI build now raises** (`MissingOrbitalError`)
  instead of zero-filling the matrix element and silently returning the
  uncorrelated energy, and **same-subshell doubles are generated**
  (`generate_sd_excitations` removed three electrons from a doubly-occupied
  shell, so every `a²→b²` double failed validation and was dropped).

### Added — cross-verification against the external codes

* `benchmarks/verify_external_codes.py` + `benchmarks/EXTERNAL_VERIFICATION.md`
  run atomx against every runnable reference in `~/code/ASAP` and record the
  measured agreement: **DBSR_HF** (live) 2e-10…8.5e-7 on closed-shell total
  energies, **FAC/sfac** (live) 1.5e-4…6.0e-4 on He-like and Ne-like ions,
  **Salvat DHFS** tabulated densities to 6e-6…1e-4 median, the **WF_TXT**
  He-like radial dumps to a 0.99993–0.99999 overlap, and the published
  **Table S12** to 1e-10…7e-7 for closed shells.  Open-shell rows are
  annotated as configuration averages, and each Salvat table's own `ALPHA`
  header is read to select the matching exchange — assuming one value
  inflates the apparent deviation from 2e-5 to 4e-2.

### Fixed — models that were reported but not used

* **The multi-configuration path silently ignored `exchange`.**  `Atom` built
  its `MCDHFSolver` without an `exchange` argument, so
  `Atom(configs=[...], exchange="slater")` returned a Hartree-Fock energy
  while `provenance` reported `'exchange': 'slater'` — measured, all three of
  `hf`/`slater`/`gu` gave -128.691926 Ha where the single-configuration path
  gives -128.623437 for `slater`.  The MCDHF/CI solver implements non-local
  HF exchange only, so the combination is now refused with a clear error
  instead of running a different model than the one reported.  The
  single-configuration path, which does implement all three, is unaffected.

### Changed — validation honesty

* **`correlation="s"` / `correlation="sd"` are not variational; documented and
  gated rather than silently returning numbers.**  The open N1 finding was
  root-caused on 2026-09-20 (`docs/CORRELATION_SD_RESEARCH_2026-09-20.md`) and
  the earlier guess recorded in the review's fixes doc — "the CI matrix is not
  the suspect" — is retracted there.  `build_ci_matrix` never evaluates
  `⟨CSF_r|Ĥ|CSF_s⟩`; it uses a configuration-average diagonal
  (`_diagonal_element`) plus two ad-hoc patches (`_diagonal_J_correction`,
  `_off_diagonal_J_correction`) because `CSF` carries no coupling tree.  For
  the He-like 1s2p J=1 block the off-diagonal patch contributes +0.431282 on a
  base element of −0.193857.  A new independent oracle
  (`benchmarks/mscheme_ci_reference.py`: Wigner–Eckart with sympy 3j,
  Slater–Condon in the m-scheme determinant basis, no angular code shared with
  the repo) run on atomx's **own orbitals and the same CSF space** disagrees by
  0.081 Ha (`"s"`/`mcdhf`), 0.805 Ha (`"s"`/`fac_frozen`), 0.432 Ha
  (`"sd"`/`mcdhf`) and 1.703 Ha (`"sd"`/`fac_frozen`).  Two alternative
  explanations were tested and rejected: the CSF list is a complete minimal
  basis (per-configuration CSF count == m-scheme dimension, 39/39, 399 == 399)
  and the reference-to-single coupling is exactly zero as Brillouin requires.
  A second, independent defect is recorded: low-weight correlation orbitals are
  solved on the wrong branch (the `mcdhf` Be singles run puts 2p/3p/3d at
  *exactly* the bare-nucleus hydrogenic `<r>`; `fac_frozen` puts 3d at
  `<r>` = 25 a₀ with 0 nodes).
  Gated by `tests/test_ci_matrix_vs_mscheme_reference.py` (3 passing
  invariants + 5 strict `xfail`s) and `tests/test_sd_space_variational.py`.
  **The physics fix is not in this change.**
* `matrix.py`'s scaffold status is now reported honestly in
  `docs/VERIFICATION.md`: three helpers in the production CI path are pure
  stubs (`_fac_one_diff_open_spectator_exchange_coef` → always `0.0`,
  `_chain_one_diff_spectator_exact_scalar` → always `None`,
  `_fac_two_diff_branch9_element` → a "reachability probe"), the Fe XVII
  J-block gate compares against FAC with a 200 eV tolerance, and
  `_off_diagonal_J_correction`'s docstring still claims "Always 0.0 in the
  current scaffold" while its body returns a populated value.
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

### Added — a first-principles invariant layer, and an audit of whether the suite can fail

* `tests/test_first_principles_invariants.py` (47 tests) gates physics that
  holds for *any* correct implementation, with expected values derived without
  consulting atomx or any other atomic code: the exact Dirac–Coulomb
  κ-degeneracy `E(n,κ) == E(n,−κ)` (the point-Coulomb spectrum depends on `j`
  only, which is why the Lamb shift is observable), the independently derived
  `1s₁/₂` closed form `mc²(√(1−(Zα)²) − 1)` (agrees to 8.7e-13 at Z=1 and
  1.2e-6 at Z=92), the point-Coulomb critical charge at `Zα = 1`, the textbook
  H 2p interval **0.365 cm⁻¹ = 10.95 GHz**, the finite-nucleus sign theorem
  (a finite charge distribution is less attractive inside the nucleus, so
  `E(finite) > E(point)`; measured +2.07e-8 Ha for He and +4.37e-5 Ha for Ne,
  with the predicted `Z^{14/3}` scaling) and the Rayleigh–Ritz bound.  The
  last section is the **Hylleraas–Undheim–MacDonald** ladder:
  `E(SD) ≤ E(singles) ≤ E(1-CSF)` and `E(SD) > E(exact)`, which is precisely
  the statement the old configuration-average `build_ci_matrix` violated by
  0.385 Ha for Be with a green suite.
* `tests/test_breit_operator.py` (38 tests) — the file
  `methods/mcdhf/breit_ci.py`'s docstring had been claiming as its validation
  while it did not exist.  It gates the rank-`L` angular element against the
  closed-form Racah–Gaunt coefficient (0 mismatches over `l,l' ≤ 3`, `L ≤ 4`),
  the `σ_μ` Hermiticity pairing and trace orthonormality, the completeness sum
  rule `Σ_q Σ_{l'm'} |⟨lm|C^L_q|l'm'⟩|² = 1` (3.996e-15), `M` conservation,
  the multipole triangle ceiling, reality of the coefficients, and the
  declaration contract that keeps the omitted retardation term from decaying
  into a silent zero.  It also records that Hermiticity of the assembled CI
  matrix is gated nowhere, rather than implying otherwise.
* `docs/test_audit/` — a full audit of the 165 test files against the question
  *would this test fail if the physics were wrong?*, with a classification
  vocabulary (`RUBRIC.md`), seven slice reports, a synthesis scorecard and
  per-slice remediation reports.  Measured A+B (first-principles or external
  oracle) ranges from **22 %** in the public-API slice to **54 %** in
  properties; `@pytest.mark.snapshot` was used by exactly **one** test in the
  whole suite against 120 unmarked frozen self-regression pins.

### Changed — validation honesty

* `xfail_strict = true` repository-wide: an `xfail` that carries a physics
  claim must fail loudly when the defect is fixed, instead of passing silently
  either way.
* Tolerance policy: every gate now states a *measured* residual next to the
  assertion and the margin it allows.  The audit found gates running 10³–10⁵
  times looser than the code's own accuracy — 234 523x in
  `test_fac_parity_gates.py`, 47 600x in `test_atom_numerics_exchange.py`,
  1 226x/34 000x in `test_dfsolver_slater.py`, 7–8 orders of magnitude in the
  length/velocity gauge theorem — which is how the defects below survived.
* Removed or corrected references that were copies of the data or values the
  code had been tuned to reproduce: a "FAC clean-room" curve that was the data
  times 1.1, a `atomx_fraction` that was a verbatim copy of FAC with rms
  identically 0, a DR "kernel parity with FAC" that was a same-kernel
  round-trip (ratio 0.99970901 at all four temperatures because the prefactor
  cancels), and a hyperfine preset asserted against the value it was tuned to
  reproduce.

### Fixed — the scalar Breit's systematic 3x, and a test that blocked its own fix

* `tests/test_qed.py` pinned the scalar Breit at `approx(0.0504, rel=0.05)`,
  a band that **excludes the correct value**.  DBSR_HF's Ne Breit is
  **0.016643630840490 Ha** (verified by running `dbsr_hf an=10 term=AV
  mbreit=1`); atomx's scalar component was 0.05043710257669187, 3.03x too
  large.  The test did not merely fail to catch the error — it made the fix
  look like a regression.
* **Root cause of the 3x: an omitted `(2L+1)`.**  Not a random error — the
  ratio creeps 3.003 → 3.030 → 3.098 (He → Ne → Ar), exactly as
  `2L+1 = 3, 5, 7` predicts once the higher multipoles gain weight with `Z`.
  In the addition theorem `P_L(cos θ₁₂) = (4π/(2L+1)) Σ_q Y_L^q(1) Y_L^{q*}(2)`
  the `(2L+1)` sits in the denominator; DBSR's `SMU` coefficients are written
  for the **unnormalised** `Y_L` contraction, while atomx's radial integrals
  use the **Racah-normalised** `C^L = √(4π/(2L+1)) Y_L`, which already absorbs
  it.  Applying one to the other multiplied the `v = L` (Gaunt) channel by
  `(2L+1)` — 3 for the dominant `L = 1`.  Reading the reference source
  (`hf_energy_BR.f90:29-32`) shows DBSR multiplies `SMU` by a configuration
  angular coefficient `coef(i,j,k)` built from the determinant expansion
  (`dbsr_lib_zconfjj.f90:4098`); `SMU` itself matched atomx bit-for-bit, but
  atomx used 1 for `coef`.  Fixed by dividing the `v = L` entry by `(2L+1)`:

  | Z | before | after | DBSR_HF | ratio before → after |
  |---|---|---|---|---|
  | 2 He | 1.9156e-04 | 6.384971e-05 (+0.11 %) | 6.377743e-05 | 3.004x → 1.001 |
  | 10 Ne | 5.0437e-02 | 1.658861e-02 (−0.33 %) | 1.664363e-02 | 3.030x → 0.996 |
  | 18 Ar | 4.1004e-01 | 1.332345e-01 (+0.66 %) | 1.323646e-01 | 3.098x → 1.006 |

* **The scalar is kept, not deleted, and the two implementations are
  complementary.**  The Gaunt operator omits retardation (its +4.19 %/+6.22 %
  excess at Ne/Ar); the fixed scalar carries the `v ≠ L` channels even though
  they remain ~50x too small, so it is the *more accurate closed-shell total*.
  Every single-CSF run now reports **both** — the scalar is applied and the
  operator is stashed as `gaunt_operator_cross_check` with
  `operator_vs_scalar_ratio` (He 0.99877, Ne 1.04533, Ar 1.05519 = the omitted
  retardation, not a disagreement).  Since the scalar (DBSR `SMU` transcription
  plus `S^v` radial integrals) and the operator (3j angular algebra plus Gaunt
  multipoles) **share no angular kernel**, that ratio is genuine independent
  evidence — which this suite notably lacked, having been found to contain a CI
  "oracle" that was a transcription of the code it was meant to arbitrate.
  The pre-existing CSF-count split is retained: single-CSF closed shells take
  DHF + the correction stack with the fixed scalar, multi-CSF/open shells take
  the operator through `MCDHFSolver`.  `breit_mode="scalar"` remains available
  as an explicit comparison mode.
* Refuted and recorded so it is not retried: no coefficient tweak fixes the
  `v ≠ L` channels, and dividing the other channels by `(2L+1)` changes nothing
  because they are already negligible.  A correct scalar for those channels
  needs DBSR's full `coef` determinant expansion.

### Removed

* Leftover empty package trees `src/atomx/prop/` and `src/atomx/analysis/`
  (residue of the `prop` → `properties` rename, shipped by the wheel builder).
* Dead modules with no reachable reference anywhere in `src/`, `tests/`,
  `scripts/` or `benchmarks/`: `scattering/radiative_recombination/rydberg.py`,
  `scattering/radiative_recombination/milne.py` (both third and fourth copies
  of physics already implemented in `radiative_recombination/__init__.py` and
  `methods/hydrogenic_cross_sections.py`) and
  `scattering/electron_ionization/rates.py`.

## Unreleased — physics & high-performance numerics pass

Driven by `docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`; see
`docs/PHYSICS_NUMERICS_REVIEW_2026-09-19_FIXES.md` for the finding-by-finding
report.  The suite grew from 1926 to ~2030 tests; every fix below carries a
gate that did not exist before.

### Fixed — physics

* **The Uehling vacuum-polarisation kernel was the wrong special function.**
  `corrections/qed.py` evaluated `K_1(2cr)` where the Uehling potential needs
  `I(2cr) = ∫_1^∞ e^{-2crt}(1 + 1/(2t²))√(t²-1)/t² dt`.  `I/K_1` runs from
  0.039 at `x = 0.01` to 0.30 at `x = 1` and never crosses 1, so `V_VP` was
  too large by 3.3–160x depending on radius: the 1s shift came out
  **-844.42 MHz for hydrogen where the correct value is -215.10 MHz** (the
  leading-order analytic result is -217.03 MHz), and -517.5 eV for U^91+
  instead of -98.0 eV.  That error alone exceeded the *measured* total 1s
  Lamb shift of U^91+ (460.2 ± 4.6 eV).  The kernel is now tabulated once by
  adaptive quadrature and evaluated by a spline in `log I + x` vs `log x`
  (≤ 1.1e-10 relative against `scipy.integrate.quad` over 12 decades).
  New gate: `tests/test_qed_uehling.py`, anchored on the H 1s Lamb-shift
  budget (`8396.5 (SE) - 217.5 (VP) - 6.6 = 8172 MHz` against the measured
  `8172.87 MHz`).

* **The tabulated self-energy function was not the function the formula
  needs.**  `data/mohr_se.py` agreed with the published hydrogenic `F(Zα)`
  only at `Z = 1` and then diverged with the wrong `Z`-dependence — 2.5x too
  large at `Z = 10`, 6.1x at `Z = 36`, **14.7x at `Z = 92`**.  Because the
  default `qed_backend="auto"` switched to a second table at `Z = 37`, the
  default self-energy was also *discontinuous* there (a 6.3x drop, -65 eV per
  1s electron).  Verified three ways: the published `Zα` expansion
  (Yerokhin & Shabaev 2015, Eqs. 7–19) reproduces FAC's table with a single
  calibrated remainder and rejects the shipped one; FAC's `coulomb.c` uses
  the identical prefactor; and the shipped table's U^91+ 1s self-energy
  (5218 eV) was 11.3x the measured total Lamb shift.  The table is now
  generated programmatically from FAC's ``_qed_se0``/``_qed_se1``/``_sd``/
  ``_bethe`` arrays, both backends read it, and `auto` has no discontinuity.
  Default total energies move by **+0.49 eV (Ne), +7.0 eV (Ar), +35 eV
  (Fe^16+), +154 eV (Kr)**.  New gate:
  `tests/test_qed_self_energy_table.py`.

* **The self-energy lost all `l`/`j` dependence outside the K/L shells.**
  The old fallback was `F(1s)/n³`, so a `4f5/2` and a `4d3/2` orbital
  received *identical* shifts and the sign was wrong for high `l`.  FAC's
  `_sd` table (every `n ≤ 5`, `|κ| ≤ 3`) and its `Klamaq` Bethe-logarithm
  form for `n > 5` are now used instead.

* **Hyperfine-adjacent: `qed_orbital_shifts` divided an already per-electron
  self-energy by the occupancy**, halving the 1s shift of any two-electron
  subshell (Fe^16+ 1s: 4.28 eV reported as 2.14 eV).

* **The single-configuration (UTA) Einstein A used the lower state's
  degeneracy.**  `A = (4/3)α³ω³S/(2j_upper+1)` needs the *upper* state;
  the code passed the lower one, so every rate with `j_upper > 1/2` was wrong
  by `(2j_u+1)/(2j_l+1)`.  The H Ly-α doublet came out with
  `A(2p3/2)/A(2p1/2) = 2.00` where both components must have
  `A = 6.2649e8 s⁻¹`; it is now `1.0000`.  The two other transition engines
  were already correct, so `A` and `gf` in the same result also violated
  `A = 6.6703e15·gf/(g_u λ²)` by exactly the same factor.

* **The normal mass shift had the wrong sign**, so every isotope shift built
  from it was reversed: `H-D 1s-2s` came out as -671.5554 GHz against a
  measured +670.994 GHz and an exact reduced-mass value of +671.5554 GHz.

* **The distorted-wave Born high-energy completion never engaged.**  The
  blend weight is `(tail/Ω)^½` and `_geometric_tail` returns exactly 0 when
  the partial-wave profile is still rising at `kappa_max`; the monotone cap
  then propagated that single zero *downward*, so `w = 0` at every energy.
  Collision strengths above ~5x threshold were 20x/69x/263x too small and
  *fell* with energy where they must rise.  Ω at 50–200x threshold now rises
  and is within 23 % of the plane-wave Born limit.

* **`collision_strength_to_rate` was dimensionally wrong** — missing the
  `(kT)^{-3/2}` Maxwellian normalisation, the electron velocity and `g_i`;
  it returned `9.18e-16 cm³/s` for `Ω = 1, T = 1 eV` where the correct value
  is `8.01e-8` (≈1.1e-8 too small at 1 eV, and the residual scaled as `T`
  rather than `T^{-1/2}`).

* **The CRM had no radiative recombination at all**, so its ionisation
  balance could not reach either the coronal or the LTE limit: with a Planck
  field at `T_rad = T_e` it sat 98.3x above Saha at `n_e = 10^18 cm^-3` and
  never converged as the density rose.  Added via Milne
  (`α_RR = PI_Planck/Saha`); the LTE ratio is now 1.0 to machine precision at
  10^18, 10^22 and 10^24.  In the same pass: the **dielectronic-recombination
  prefactor was 50.17x too small** (it used `kT` in Rydberg where the rest of
  the module uses eV), `exp(-dE/kT)` underflow silently zeroed *both*
  collisional directions above `dE/kT ≈ 745`, the Stewart–Pyatt potential was
  discontinuous by 247 eV at its branch switch, the average atom returned
  `Σocc = 769` electrons for `Z = 26`, and `plasma/rates.py` passed
  `sqrt(T_eV)` where the 8.629e-6 constant needs `sqrt(T_K)` (107.7x, at every
  temperature).

* **`compute_yk` double-counted the diagonal cell of its cumulative
  integrals**, biasing the Hartree potential by `+w_iρ_i/r_i` (an O(h) error).
  Measured +1.75 % on `F⁰(1s,1s)`; the `slater` and `gu` exchange models,
  which route through this kernel, moved by 0.44–4.5 eV.  Each side now
  contributes the half cell it owns; the `slater` Ne energy matches an
  independent refined-grid reference to 5e-7 relative.

* **MBPT(2) built its virtuals in the `V^{N-1}` potential while taking the
  occupied ε from the N-electron SCF**, mixing two Hamiltonians in the energy
  denominator.  He came out at `E2 = -0.0496 Ha` — 118 % of the exact
  correlation energy — and `E_tot` 8 mHa *below* the exact ground state,
  which is impossible for MP2 on a HF reference.  With `V^N` virtuals
  `E2 = -0.036968 Ha`, within 0.7 % of the literature MP2/CBS value and
  correctly bracketed between HF and exact.

### Fixed — silent wrong answers

Three paths returned a plausible number with `converged=True` instead of
raising.  All three now raise:

* an unbound (virtual) d/f subshell collapsed into the negative-energy
  continuum (`Atom(Z=10, charge=-1, config="...3d1")` returned
  `-3.8e4 Ha`); the sea threshold was `-2 m_e c²` where the continuum starts
  at `-m_e c²`;
* a requested subshell with no bound state produced an all-zero orbital, so
  `Atom(Z=1, config="7s1", numerics="bspline")` returned `total_energy == 0.0`
  with an empty `orbital_energies`;
* the shooting engine returned its own bracketing ceiling `-Z/r_max` (4 %
  error for H 6s, 31 % for 7s, and the whole Na I Rydberg series 9s..13s
  pinned to one value).  `Atom` now also auto-sizes `r_max` to contain the
  requested orbitals, so the H Rydberg series 5s–8s solves to ~1e-10
  relative.  `ier` is propagated from `dbound`/`sbound` and `total_sum` is
  initialised on the early-exit paths (pure Python raised
  `UnboundLocalError` while the JIT build silently returned garbage).

### Fixed — numerics and interface

* The B-spline inner grid step is now reachable from the public API
  (`Atom(grid_hi=..., grid_he=..., grid_hmax=...)`).  Without it the
  hydrogen-like U^91+ 1s eigenvalue was **+97.7 eV** off at the default grid;
  `grid_hi=0.005, grid_he=0.10` brings the same eigenvalue to **0.24 eV**.
* `Atom(..., r_max=...)` is exposed for both engines.
* One shared configuration parser (`config/parsing.py`) replaces two copies
  that failed on a two-digit `n` (`"10s1"` raised `KeyError: '0'`) and on the
  implicit unit occupation (`Atom(Z=3, config="1s2 2s")` raised
  `ValueError: invalid literal for int()`).
* `qed_orbital_shifts` no longer divides by occupancy; the DR per-resonance
  rows (`enumerate.py`) use the same prefactor as the rate coefficient
  instead of a second hard-coded copy; `NormalMassShiftCorrection` passed a
  nonexistent keyword and then `float()`-ed a dataclass.
* The Wigner–Eckart layer's **Grant phase convention is now documented and
  pinned**.  `reduced_C` carries `(-1)^(j_a+1/2)`, so `ckm(k,m,k,m,0,0) = -1`
  rather than `+1`; that is deliberate and consistent across CI/DFS,
  distorted-wave and Auger, and a new gate pins both the sign and the
  convention-independent Wigner–Eckart sum rule so it cannot drift.

### Changed — performance

* **The B-spline RKB banks are built once per solve instead of five times.**
  `ConvolCache` was per-`DFSolver` instance and `mcdhf/solver.py` builds a new
  solver on every MCDHF outer iteration, so all 20 `(k, itype)` banks were
  rebuilt 5x with bit-identical content — 13.8 % of the wall time.  A
  process-wide registry keyed on the *grid content* (not `id()`, since the
  five `bsp_data` dicts are distinct objects with identical content) takes the
  builds per solve from **100 to 20** and a Fe-like `ns=113` solve from
  **9.56 s to 7.32 s (23 %)**, bit-identically.  `build_ci_matrix` also no
  longer drops the stashed `_convol_cache` across `_reset_memo()`, which its
  own docstring four lines above forbade.
* **`fastmath=True` is now applied only where it measurably pays.**  It was
  on 20 kernels; on 16 of them it was neutral or *slower* (up to 1.45x slower)
  while silently breaking bit-reproducibility — and two docstrings claimed
  bit-identity that the flags contradicted.  Kept on the four that earn it
  (3.9x, 3.4x, 2.2x, 1.2x), dropped from the rest: end-to-end **5.8 %
  faster**, energy drift 1.06e-14 relative, and `_density_sym` is now
  verifiably `array_equal` to the plain band construction.
* **A missing orbital in the CI build now raises** (`MissingOrbitalError`)
  instead of zero-filling the matrix element and silently returning the
  uncorrelated energy, and **same-subshell doubles are generated**
  (`generate_sd_excitations` removed three electrons from a doubly-occupied
  shell, so every `a²→b²` double failed validation and was dropped).

### Added — cross-verification against the external codes

* `benchmarks/verify_external_codes.py` + `benchmarks/EXTERNAL_VERIFICATION.md`
  run atomx against every runnable reference in `~/code/ASAP` and record the
  measured agreement: **DBSR_HF** (live) 2e-10…8.5e-7 on closed-shell total
  energies, **FAC/sfac** (live) 1.5e-4…6.0e-4 on He-like and Ne-like ions,
  **Salvat DHFS** tabulated densities to 6e-6…1e-4 median, the **WF_TXT**
  He-like radial dumps to a 0.99993–0.99999 overlap, and the published
  **Table S12** to 1e-10…7e-7 for closed shells.  Open-shell rows are
  annotated as configuration averages, and each Salvat table's own `ALPHA`
  header is read to select the matching exchange — assuming one value
  inflates the apparent deviation from 2e-5 to 4e-2.

### Fixed — models that were reported but not used

* **The multi-configuration path silently ignored `exchange`.**  `Atom` built
  its `MCDHFSolver` without an `exchange` argument, so
  `Atom(configs=[...], exchange="slater")` returned a Hartree-Fock energy
  while `provenance` reported `'exchange': 'slater'` — measured, all three of
  `hf`/`slater`/`gu` gave -128.691926 Ha where the single-configuration path
  gives -128.623437 for `slater`.  The MCDHF/CI solver implements non-local
  HF exchange only, so the combination is now refused with a clear error
  instead of running a different model than the one reported.  The
  single-configuration path, which does implement all three, is unaffected.

### Changed — validation honesty

* **The "7 keV / 1.6 % at Z=82" limitation was an artifact.**  The committed
  `benchmarks/results_fac_structure.csv` atomx column had been stale for five
  months (it corresponded to an under-converged basis) *and* the comparison
  ran atomx with `qed=False` against a FAC default that has QED on.  With the
  flags matched the whole `Z = 10…82` panel agrees to **≤ 9.6e-4 (0.096 %)**,
  the worst absolute row being **426 eV at Z=82** — the sign of the residual
  reverses and the gap is 16x smaller than documented.  The driver's default
  is now the matched pair, the CSV records which flags produced it, and
  `tests/test_fac_structure_snapshot.py` recomputes the atomx column on every
  run so it cannot go stale again.
* The FAC DFS/shooting parity gate now compares like for like
  (`breit=False`, matching FAC's default) and its gate was *tightened* from
  5e-3 to 2e-3 as a result — the old gate was measuring the Breit-on-vs-off
  mismatch.
* Several gates that encoded the bugs above were re-pinned with the reason
  recorded inline: the QED total shifts, the Ne/Ar VP components, the n=3
  self-energy expectation, the "Mohr table is 8x larger than FAC" assertion
  (which asserted the bug as a feature), the hyperfine-adjacent Co seam pin,
  and the MBPT He sanity bound.
