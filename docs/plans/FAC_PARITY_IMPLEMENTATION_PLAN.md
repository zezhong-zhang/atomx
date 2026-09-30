# AtomX FAC-Parity Implementation Plan

Goal: implement FAC-equivalent physics in AtomX while preserving AtomX's modern
Python structure, explicit data models, process-specific calculators, and testable
benchmark gates.

This plan is intentionally dependency ordered.  FAC-level radiative,
collisional, ionization, autoionization, and DR accuracy all depend on the same
core angular and continuum machinery.  Implementing process wrappers before that
core only produces runnable but non-quantitative code.

## Definition of Done

AtomX reaches FAC parity when:

1. Every major FAC physics table has an AtomX equivalent:
   `Structure`, `TRTable`, `PITable`, `RRTable`, `CETable`, `CITable`,
   `AITable`, `DRTable`, `RMatrix`, `StructureEB`, `TRTableEB`,
   `CETableEB`, M-substate tables, cascades, CRM, average atom.
2. Shared physics conventions match FAC:
   configuration groups, CSF enumeration, J/pi block structure, CFP/RCFP
   angular reductions, continuum normalization, table units, `A/gA/gf`
   conventions, statistical weights, and top-up options.
3. Benchmarks run reproducibly from one entry point and write process-level
   speed and accuracy summaries.
4. Existing public APIs keep working; FAC-parity behavior is exposed through
   explicit model/backend options rather than hidden behavior changes.

## Accuracy Targets

- Local-exchange DFS totals: median relative error `< 1e-3`.
- Ground-state structure: median relative error `< 1e-3` through mid-Z;
  high-Z accepted at `< 1e-2` unless FAC QED/Breit options are exactly matched.
- Strong radiative lines: `< 5-10%` after basis matching.
- Weak/intercombination lines: correct cancellation pattern and order of
  magnitude first, then tightened after full RCFP validation.
- PI/RR/CE/CI: `< 10-30%` in resolved benchmark regimes; separate near-threshold
  tolerances until direct continuum is complete.
- AI/DR: factor `< 2` first, then tighter after full two-body CSF coefficients.

## Phase 0: Benchmark and State-Matching Infrastructure

Deliverables:

- Stable runner: `benchmarks/run_fac_parity.py`.
- Per-process benchmark scripts with FAC command, AtomX command, elapsed time,
  output CSVs, and return codes.
- Matching by `(2J, parity, dominant configuration, energy)` where available;
  rank-only matching is only a fallback.
- Explicit conversion for FAC table conventions, especially `gA` vs `A`.
- Optional FAC availability detection for CI.

Acceptance:

- `uv run python benchmarks/run_fac_parity.py --quick` runs successfully.
- Summary CSV records command, return code, and elapsed time.
- No benchmark compares mixed units or mixed statistical conventions.

## Phase 1: Shared CSF Angular Core

Deliverables:

- `atomx.angular.csf` with:
  - explicit `J=0` handling,
  - J/parity tensor selection rules,
  - FAC UTA one-body population factors,
  - one-body CSF coupling metadata,
  - two-body Coulomb coupling metadata,
  - direct/exchange allowed multipole ranks.
- CFP/RCFP layer equivalent to FAC `cfp.c`, `rcfp.c`, `recouple.c`,
  `angular.c`.
- Cached reduced one-body and two-body CSF coefficients.
- A single angular interface used by MCDHF, transitions, CE, CI, AI, and DR.

Acceptance:

- J-resolved CSF counts match FAC for representative s/p/d/f open shells.
- One-body CSF selection and population factors are process-independent.
- Two-body rank selection is process-independent.
- Full RCFP coefficients reproduce FAC angular matrix elements for small shells.

## Phase 2: FAC-Level Structure Hamiltonian

Deliverables:

- J/pi block CI Hamiltonian using full one-body and two-body CSF reductions.
- Slater integral cache keyed by orbital quartet and k.
- Breit and QED correction stack with FAC-compatible options.
- Rich level metadata: energy, `2J`, parity, dominant CSFs, mixing vector,
  parentage labels, configuration group IDs.

Acceptance:

- He/Ne/Ar-like structure remains at current accuracy.
- Fe XVII 37-level benchmark improves materially from current simplified
  Hamiltonian results.
- Runtime improves from block diagonalization and angular/radial caches.

## Phase 3: Radiative Transitions

Deliverables:

- FAC-equivalent transition table for E/M multipoles of arbitrary rank.
- Full CSF one-body tensor reduction for resolved lines.
- UTA and resolved-line modes.
- Length/velocity gauges.
- Correct `A`, `gA`, `gf`, `f`, line strength, wavelength conventions.
- Two-photon rates wired into transition/cascade workflows.

Acceptance:

- He-like M1/E2/M2 benchmarks populated against FAC.
- Li-like 2s-2p rates match FAC across Z series.
- Fe XVII strong and weak E1 lines show correct angular cancellation behavior.

## Phase 4: Direct Energy-Normalized Dirac Continuum

Deliverables:

- Shooting/direct continuum backend at arbitrary energy.
- FAC-style asymptotic normalization and phase shifts.
- Adaptive inner radial grid for high-Z K-shell and near-threshold electrons.
- Backend selector: `continuum_backend="direct" | "bspline"`.

Acceptance:

- PI/RR high-Z K-shell benchmark no longer fails from B-spline resolution.
- Near-threshold PI/RR/CE/CI curves are smooth and reproducible.

## Phase 5: PI/RR/DR Recombination Stack

Deliverables:

- FAC-compatible `PITable`, `RRTable`, UTA variants.
- Milne relation with exact statistical conventions.
- DR built from full AI + radiative branching.
- Maxwellian rate tables and Rydberg/top-up handling.

Acceptance:

- He-like and Li-like PI/RR FAC benchmark median errors are within target.
- DR benchmark is quantitative rather than topology-estimate based.

## Phase 6: CE Electron-Impact Excitation

Deliverables:

- Distorted-wave CE using shared CSF angular core.
- Born and Coulomb-Born high-energy modes.
- Partial wave convergence controls and top-up.
- Collision strengths, cross sections, and Maxwellian rates.
- M-substate `CETableMSub`.

Acceptance:

- He-like Fe CE benchmark no longer differs by orders of magnitude.
- High-energy Born limits agree with FAC options.

## Phase 7: CI/EII Ionization

Deliverables:

- Direct ionization with DW, Coulomb-Born, and explicit BEB model modes.
- Channel/subshell-resolved ionization tables.
- Continuum-continuum Coulomb integrals and top-up.
- M-substate `CITableMSub`.

Acceptance:

- Fe XVII CI benchmark improves beyond current moderate agreement.
- Production mode does not rely on hidden BEB clamping.

## Phase 8: AI/Auger and DR Coupling

Deliverables:

- Autoionization rates from full two-body CSF Coulomb matrix elements.
- Inner-shell Auger and Coster-Kronig channels.
- `AITable` and `AITableMSub`.
- DR pipeline using AI and TR branching.

Acceptance:

- Ne-like Se AI benchmark becomes quantitatively comparable to FAC.
- DR rates match simple FAC cases within the AI/DR target tolerance.

## Phase 9: R-Matrix Close Coupling

Deliverables:

- R-matrix basis, channels, surface amplitudes, boundary conditions.
- Target-state construction from AtomX structure levels.
- Low-energy resonance scattering and excitation.
- Conversion/expansion utilities comparable to FAC workflows.

Acceptance:

- R-matrix module is no longer a stub.
- Low-energy resonance benchmarks are separate from DW benchmarks.

## Phase 10: External Fields and M-Resolved Physics

Deliverables:

- Joint Stark + Zeeman Hamiltonian.
- `StructureEB`, `TRTableEB`, `CETableEB` equivalents.
- Magnetic-sublevel populations and polarized emission.
- Asymmetry parameter beta.

Acceptance:

- Hydrogenic/simple-field limits pass analytic tests.
- Small FAC EB/MSub benchmarks pass.

## Phase 11: Plasma, Cascade, Average Atom, and Scaling

Deliverables:

- FAC-like cascade and X-ray cascade workflows.
- CRM benchmark and FAC-compatible rate ingestion.
- Self-consistent average atom solver.
- Parallel benchmark execution and large-table production mode.

Acceptance:

- Cascade/CRM examples reproduce FAC-style workflows.
- Average atom has EOS/opacity-oriented validation cases.
- Full FAC parity suite can run in parallel.

## Required Execution Order

1. Phase 0 benchmark runner and state matching.
2. Phase 1 CSF angular core and full RCFP coefficients.
3. Phase 2 structure Hamiltonian.
4. Phase 3 radiative transitions.
5. Phase 4 direct continuum.
6. Phase 5 PI/RR/DR.
7. Phase 6 CE.
8. Phase 7 CI/EII.
9. Phase 8 AI/Auger.
10. Phase 9 R-matrix.
11. Phase 10 external fields and M-resolved physics.
12. Phase 11 plasma/cascade/average atom/scaling.

## Current Status

- Phase 0: in progress.
  - Quick runner exists: `benchmarks/run_fac_parity.py`.
  - Fe XVII level benchmark now carries AtomX/FAC `2J` and parity and uses
    J/pi-aware level matching.
  - Fe XVII transition benchmark now maps FAC upper levels through the level
    matcher instead of rank-sorting unrelated transitions.
  - Corrected 2026-09-29: FAC prints `A` directly. The former
    `gA/(2J_upper+1)` interpretation was wrong; current callers compare
    `A_per_s` without another weight division and explicitly request E1.
    Historical numbers need regeneration; see `docs/VERIFICATION.md`.
  - Quick/full parity runner includes RCFP primitive and J-resolved CSF
    benchmarks.
  - Benchmark-facing FAC parsers for RR, CE, CI, and AI ASCII tables are
    restored in `benchmarks/fac_runner.py`.
  - CI parsing now follows FAC `PrintCITable` column order
    (`E, strength, sigma`) instead of reusing the CE column order; this fixes
    the apparent Fe XVII CI scale error in the benchmark.
  - Public compatibility exports/APIs needed by the parity harness are
    restored: `ContinuumModel`, `count_j_resolved_csfs`, MCDHF result
    `subshells`/`orbitals_raw`, RR `continuum_model`, and DW threshold
    override support.
  - Fe XVII levels benchmark now opts into J-resolved CSF enumeration and
    includes both `2p`-hole and `2s`-hole members of FAC's compact
    `2*7 3*1` model space, restoring the 37-level comparison.
  - CI diagonalisation now removes cross-`J/pi` couplings before solving,
    matching FAC's symmetry-block structure and preventing unphysical
    mixed-symmetry eigenvectors.
  - Fe XVII transition benchmark now filters AtomX E1 lines to ground-state
    to excited-state transitions before comparing to FAC's `n2 -> n3` table.
  - RR now honors shooting `ContinuumModel` inputs by solving direct
    continuum waves in the requested residual-core potential; the FAC RR
    parser now compares against the energy-dependent RR cross-section column.
  - PI/RR now uses the same Milne `1/(2E)` convention as the hydrogenic FAC
    helper and uses reduced E1 line-strength angular factors instead of the
    old continuum-state degeneracy shortcut.
  - PI/RR fine-structure channels now average the reduced E1 angular strength
    consistently with the Milne bound-state statistical factor, fixing the
    Li-like Fe `2p3/2` RR channel scale.
  - MCDHF transition accumulation now applies shared CSF one-body
    J/parity/occupation gates and uses the upper-state statistical weight in
    the Einstein-A denominator.
  - Distorted-wave CE active-orbital discovery now uses occupation changes
    rather than common-orbital energy shifts; optically allowed DW-zero cases
    fall back to a Coulomb-Bethe high-partial-wave floor.
  - B-spline continuum solver now has an explicit delta-energy normalization
    option. CE/AI scattering call sites request it, avoiding accidental use of
    raw box-normalized pseudo-continuum states.
  - Continuum diagnostics show current B-spline pseudo-continuum eigenvalues
    can miss requested CE energies by roughly 5-20%, so direct shooting
    continuum is preferred for FAC-sensitive rates when available.
  - The public Auger compatibility wrappers now execute a coarse named-channel
    two-electron Slater-integral calculation with continuum waves and
    state/channel occupation factors, replacing the previous deterministic
    zero-rate placeholders.
  - Coarse AI now uses the direct shooting continuum when `shooting_n_grid` is
    supplied and compares topology-level AtomX rates to the dominant FAC row in
    the matching active-shell/final-J pool.
  - Fe XVII transition benchmark now writes a fresh transition-summary CSV so
    the parity runner no longer reports stale recoupling metrics.
- Phase 1: in progress.
  - Shared CSF one-body and two-body metadata exists.
  - Exact m-scheme backend exists for antisymmetric `j^q` determinant states,
    highest-weight `J` terms, and reduced creation matrix elements.
  - Public reduced-subshell API exists in `atomx.angular.rcfp`; it currently
    uses the exact m-scheme backend and is ready for FAC-table validation and
    optimized table backends.
  - `W = [a† x atilde]^K`, scalar `[W^K x W^K]^0`, mixed
    `[A^j x W^K]^R` / `[W^K x A^j]^R`, and pair creation/annihilation reduced
    primitives are implemented and covered by tests.
  - `benchmarks/bench_rcfp_primitives.py` records stable primitive coefficients
    for future FAC/Gaigalas table parity checks.
  - Initial multi-subshell recoupling data model exists in
    `atomx.angular.csf_recoupling`, with exact scalar same-subshell `W` and
    `WxW0` coefficients and spectator/J/pi blocking.
  - Two-subshell coupled-state scaffold exists, with validated allowed total-J
    values for the next recoupling step.
  - One-body `W^K` reduced matrix elements acting on the left or right member
    of a two-subshell coupled state are implemented with Wigner-6j recoupling.
  - Scalar two-subshell `[W_left^K x W_right^K]^0` matrix elements are
    implemented by explicit coupled-product expansion.
  - A validated left-associated multi-subshell CSF chain exists, with
    recursive Wigner-6j reduced `W^K` recoupling for arbitrary active
    subshell positions.
  - General scalar chain `[W_i^K x W_j^K]^0` coefficients are implemented by
    explicit coupled-projection expansion for arbitrary active subshell pairs.
  - Explicit CSF coupling chains now have a scalar two-body discovery API that
    scans active subshell pairs and returns nonzero chain `WxW0` coefficients.
  - A conservative CSF-level two-body coefficient API exists for simple
    two-active-subshell scalar `WxW0` products.
  - Automatic discovery of simple scalar two-body coefficients across active
    subshell pairs exists for same-shape recoupled CSFs.
  - Recoupled CSF occupation changes are classified as diagonal, single,
    double, or unsupported, providing the gate for occupation-changing
    coefficient discovery.
  - A conservative single-excitation coefficient path exists using source
    annihilation, destination creation, and scalar spectator `W` factors.
  - A conservative paired double-excitation coefficient path exists using pair
    annihilation/creation primitives and scalar spectator `W` factors.
  - A conservative mixed two-source/two-destination double-excitation
    coefficient path exists for simple recoupled CSFs.
- Phase 10: initial line-polarisation coefficient support exists.
  - `steffen_alder_F2` is implemented and covered by regression tests for
    rank-2 E1 alignment factors and selection-rule blocking.
- General multi-subshell recoupling, FAC-table validation, and production
  Hamiltonian/scattering/rate wiring remain the main blockers for FAC-level
  weak-line, Hamiltonian, AI, and DR parity. CE is no longer a zero-rate
  failure in the quick benchmark, but still only has a Coulomb-Bethe fallback
  for the optically allowed high-partial-wave contribution. AI is no longer a
  placeholder and the shooting continuum puts several channels in the right
  order of magnitude, but FAC's CSF-resolved Auger angular factors are still
  missing.
- RR/PI continuum normalization is improved but still not FAC-parity: current
  quick benchmark residuals are approximately max/median `0.626/0.053` for
  He-like Fe RR and `0.121/0.026` for Li-like Fe RR.
- Latest quick CE/CI/AI residuals after the CE and coarse-Auger updates are
  approximately:
  CE max/median `0.641/0.409`, CI max/median `0.242/0.209`, AI max/median
  `16.6/0.967` for the coarse topology-level AI comparison.
