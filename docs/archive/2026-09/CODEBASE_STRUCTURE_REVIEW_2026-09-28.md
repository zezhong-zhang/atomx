# Full codebase structure and design review

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

Date: 2026-09-28. Scope: the current working tree, including ongoing changes.
Companion target design: [ARCHITECTURE_DESIGN.md](../../ARCHITECTURE_DESIGN.md).

Implementation follow-up: [repairs on 29 September](ARCHITECTURE_REPAIRS_2026-09-29.md)
addresses transition substitution/fallback, canonical method names and controls,
shared subshell ownership, two numerical extractions, and packaged NIST parsing.
The follow-up also extracts nonlocal SCF iteration, shared fields/workspaces,
operator-stage guards, packaged comparison services and lazy public imports.
The snapshot and original findings below describe the tree before those repairs;
the follow-up distinguishes completed work from remaining migration.

## Current working-tree recheck

The latest review re-parsed **249 source modules / 71,962 lines**, inventoried
**244 test files and 136 top-level Python scripts**, and resolved both absolute
and relative static imports. All source modules parsed successfully. Counts
describe this working tree, including uncommitted work, rather than a release.

**F1 and F2 below are historical defects now repaired in this tree.**
`io/results.py` writes schema-2 physical metadata and restores the calculation
specification and correction accounting. The native provider consults the
declared equation before its explicitly identified legacy heuristic. Persistence
is still SCF-specific; this does not establish complete shooting/MCDHF archival.
The original evidence is retained below to explain the repairs.

Fresh verification: **37 tests passed** across `test_scfresult_save_load.py`,
`test_native_provider.py`, `test_numerical_design_contracts.py`, and
`test_correction_application_contract.py`, plus
`test_native_campaign_comparison.py` and `test_native_shooting_generation.py`.
The local pytest environment lacks
the timeout plugin and reported two configuration warnings. No full suite or
new external physics benchmark was run for this recheck.

This pass changed review/design documentation only. Existing repairs and
campaign outputs are not newly implemented or revalidated physics results.

### Repository inventory

Counts include comments, tables and package initializers. Test files are counted
recursively by `test_*.py`; script counts are immediate `scripts/*.py` files.

| Source area | Modules | Lines |
|---|---:|---:|
| Root API, contracts, results and utilities | 13 | 4,973 |
| Angular algebra | 11 | 3,223 |
| Configuration | 3 | 399 |
| Corrections | 18 | 3,584 |
| Data tables | 6 | 3,601 |
| Fitting | 2 | 223 |
| IO | 3 | 1,162 |
| Line shapes | 3 | 282 |
| Methods | 35 | 15,815 |
| Numerics | 47 | 9,956 |
| Physics | 6 | 558 |
| Plasma | 6 | 1,987 |
| Potentials | 4 | 668 |
| Properties | 29 | 6,032 |
| Providers | 5 | 2,578 |
| Scattering | 45 | 13,774 |
| Spectrum | 5 | 976 |
| UTA | 2 | 257 |
| Visualization | 6 | 1,914 |

The tree is already organized into useful scientific subsystems. The design
problem is responsibility crossing their interfaces, especially around state
identity, numerical iteration, and application orchestration. Preserve coherent
areas such as angular algebra, line shapes, and integral kernels during migration.

### Newer campaign code: good foundations and remaining ownership work

The native campaign uses a single database writer, source identity checks,
explicit attempts, and two radial resolutions. Its comparison code checks
endpoint model consistency, retains unmatched records, and separates resolved
lines from configuration averages. These are useful contracts to preserve when
extracting `workflows/` and `reference_data/`.

However, `compare_native_shooting_campaign.py` imports physical classification
and wavelength conversion from `generate_full_dhf_nist.py`, centroid matching
from another script, and `Subshell` from a concrete DHF solver. These should
be shared package services rather than dependencies between executables.
Successful attempts and grid agreement also need to remain distinct from state
acceptance: the existing crossed-orbital-label check is a review flag, not a
general proof of eigenstate identity. A comparison record should retain the
state acceptance evidence and its policy version.

### Performance implications of the design

Keep physical policy outside repeated radial loops. Resolve angular coefficients,
operator selection, and capabilities before iteration where their inputs permit.
Keep representation-specific compiled kernels, banded B-spline structures and
prepared Green response contexts inside numerics. Avoid a Python callback per
radial point; functional interfaces should operate on whole arrays or blocks.

Cache lifetimes should follow their actual dependencies: geometry and quadrature
may survive an SCF iteration; potential-dependent Green data may not; CI-density
contractions must track the current density; finalized observables must track the
state and operator specification. Existing `ShootingRadialProblem` ownership and
`ConvolCache` machinery provide foundations. This review does not establish a
cache bug or a measured runtime bottleneck. Profile initialization/JIT separately
from steady-state propagation, integrals, response, CI, allocation and IO before
claiming an optimization.

### Open issues to address first

1. **F3: transition model substitution and fallback.** The current CI path still
   creates a separate pure-Coulomb local-exchange atom and converts generic
   exceptions into fallback. This needs an explicit requested/actual model and
   failure policy before its results can be treated as the original calculation.
2. **F4/F6: method naming and ignored controls.** The HF and Breit–Pauli wrappers
   hardcode `n_grid=1000`, `r_max=75` and do not forward their inherited nuclear
   option. `Atom._solve_method` passes only tolerance/iteration controls to them
   (`atom.py:1181`). Thus the `Sampling` documentation's claim that the same
   discretization controls drive these methods is not implemented consistently.
   This is a traced execution path, not a newly measured energy discrepancy.
3. **F5/F7: ownership and shared state.** Complete the functional/engine split;
   define result finalization and cache ownership before moving consumers.
4. **F8/F9: operator stages and application packaging.** Carry the resolved
   calculation and matching record through jobs, archives and the website.

### Complete numerical-boundary scan

These are all seven direct static imports from `numerics` into methods,
properties or scattering in this snapshot. Local imports are included.
Compatibility exports explain some edges but still need a migration endpoint.

| Import site | Higher-level dependency | Resolution |
|---|---|---|
| `numerics/analytic/__init__.py:16` | `methods.hydrogenic_reference` | Keep orchestration with methods; remove its numerical re-export after compatibility migration |
| `numerics/analytic/hydrogenic_dirac.py:2` | `methods.hydrogenic_reference` | Same; the implementation uses a numerical B-spline solve, so “analytic” is misleading |
| `numerics/bspline/continuum.py:337` | `properties.transitions.angular` | Move photoionization orchestration to scattering; retain continuum eigenproblems in numerics |
| `numerics/shooting/__init__.py:30` | `methods` | Remove upward lazy exports once canonical numerical drivers exist |
| `numerics/shooting/hf_shooting.py:2` | `methods.shooting_hf` | Extract physical functional and numerical SCF driver |
| `numerics/shooting/scf_potential.py:185` | `scattering.protocols` | Consume a shared radial contract below scattering |
| `numerics/shooting/solver.py:2` | `methods.shooting_local` | Extract local-exchange definition and numerical SCF driver |

The current architecture test visits a selected list of shooting files and
does not resolve relative import targets. Extend it across all numerical modules
with a named, shrinking compatibility allowlist. An import graph is evidence of
ownership, not a runtime-performance profile or proof of an import cycle.

### Decisions for the migration

- Keep existing `potentials/` as the owner of nuclear/environment potential
  construction. Do not create another implementation in `physics/` merely to
  match a directory diagram.
- Move `Subshell` out of the B-spline solver module into the shared state/config
  contracts. Its current home causes configuration and persistence consumers to
  depend on a concrete solver. Keep an import alias during migration.
- Place `SCFResult`, `ShootingResult` and MCDHF physical result contracts outside
  solver implementations. Preserve representation-specific payloads rather than
  forcing coefficients and sampled spinors into one array convention.
- Freeze finalized state arrays and resolved inputs; keep mutable iteration
  buffers private to engines. Lazy radial snapshots otherwise require explicit
  invalidation when a result changes. This remains a lifecycle risk, not a
  reproduced stale-cache energy defect.
- Treat named sampling tiers as presets. Report measured SCF residual, radial
  refinement, box/projector/correlation cutoffs separately; a preset name is not
  an accuracy certificate.
- Profile integral construction, propagation, constrained response, CI and
  allocation/copying separately. Preserve compiled kernels and banded/batched
  operations through extraction. No new speedup is claimed by this review.

The remaining sections retain the original review evidence. Where historical
line numbers changed, the module and named function are the stable references.

## Summary

Atomx has substantial reusable numerical and physical machinery, but its
architectural boundaries are only partly enforced. The highest priority is
preserving the meaning of a calculation across API dispatch, downstream
observables, provider conversion and persistence. Moving shooting files is
necessary, but insufficient on its own.

The original review found two directly reproduced defects (now repaired): HDF5 round trips lose
nuclear charge and calculation specification; the native provider labels a
Schrödinger payload with zero Q arrays as relativistic. Other findings below
are traced code paths or design risks, explicitly distinguished from measured
numerical failures. No solver implementation was changed during this review.

## Scope and evidence

- Parsed all **246 Python modules under `src/atomx`**, totaling **71,567 lines**
  including comments and embedded tables, and inspected their static import edges.
- Inventoried **237 test files** and **128 Python scripts** immediately under
  `scripts/`; reviewed packaging, CI and the existing provider ADR.
- Read representative implementations and traced interfaces in every major
  subsystem. This is a repository-wide architecture review, not a line-by-line
  proof of every physical formula or a fresh experimental benchmark.
- Ran numerical-boundary, scaffold and correction-application tests:
  **17 passed**. The local environment warned that pytest-timeout configuration
  options were unknown; these checks finished normally.
- Used isolated in-memory/scratch-file probes for the two defects above.
  No production archive or calculation database was modified.
- Static imports include local imports; dynamic imports and runtime call graphs
  are not exhaustively captured. Import cycles are not inferred merely from
  package-level mutual use.

## Findings, in priority order

### F1 — Repaired: persisted results lost physical identity and provenance

Evidence: `src/atomx/methods/dirac_fock.py:218` (`SCFResult.save`) and
`:313` (`load`). The format writes energies, orbitals and basis information,
then reconstructs the result without `Z` or `calculation_spec`. New correction
application records and full model provenance are not serialized there either.

Reproduced with a minimal SCFResult carrying Z=1 and a CalculationSpec:

```text
saved specification: True
loaded specification: None
loaded Z: None
```

Consequence: a result advertised as usable after reload loses information
needed for nuclear-dependent properties, correction support and matched-model
comparisons. A numerical restart is not an auditable scientific result archive.

Recommendation: move persistence to `io/results.py` with a versioned schema.
Round-trip physical specification, radial representation/metric, state identity,
correction stages and diagnostics. Old files must report missing provenance
explicitly; do not reconstruct it from guesses.

### F2 — Repaired: native provider inferred physics from array layout

Evidence: `src/atomx/providers/native_adapter.py:31` returns relativistic=True
whenever Q is not None; it does not inspect the declared equation. Its docstring
says “non-trivial” Q, but zero arrays also qualify. `build_payload` at line 84
also labels every native basis `atomx-radial-grid` and supplies an empty
approximation list.

Reproduced: a payload explicitly marked `_equation='schrodinger'` with Q=zeros
is classified relativistic. This is especially problematic because the shared
scattering protocol expressly permits zero Q for nonrelativistic orbitals.

Recommendation: derive equation, representation and approximation provenance
from resolved metadata. Preserve a distinction between sampled orbitals and
B-spline coefficients, and between orbital orthonormality and the basis metric.
Do not infer spin-orbit treatment from array presence. Reuse provider capability
contracts rather than introducing another competing registry.

### F3 — High: transition calculation can change the requested model

Evidence: `src/atomx/properties/transitions/calculator.py:239` falls back from a
requested CI path to UTA when CI returns None; `:296` catches ValueError along
with unsupported/size errors. `_ci_ground_result` at `:323` constructs a new
Atom with shooting, Gu local exchange and corrections disabled, then catches
all Exception and caches None on failure. Its cache key is configuration only.

The UTA fallback is documented and rows carry `resolution`; it is not wholly
unlabelled. However, the failure reason and the model change are not represented
by that one label. Generic ValueError can hide an invalid input or actual bug.

Recommendation: require an explicit fallback policy, preserve structured reasons,
and pass a declared reference-calculation specification. Property kernels should
consume states; a separate orchestration service may request missing states.
Cache by complete model/state identity. A strict CI request must fail if it
cannot be fulfilled.

### F4 — High: names overstate implemented physical methods

Evidence: `methods/hartree_fock.py:38` invokes the local Slater-exchange shooting
solver. `methods/relativistic_hf.py:99` uses that same kind of base calculation;
`:147` sets Z_eff=Z for hydrogenic perturbative shifts. Its inherited nuclear
option is not forwarded in this call. The formulas in `physics/breit_pauli.py`
are explicitly documented as hydrogenic expressions.

Recommendation: canonical module `methods/breit_pauli.py`; explicit model labels
for the existing limited implementation; a separately named HF–Slater method.
Reserve nonlocal HF labeling for nonlocal Fock exchange. Validate/forward nuclear
options. Keep legacy imports without changing old calculations silently.
A rename must not be represented as implementation of a complete Hamiltonian.

### F5 — High: physical and numerical responsibilities are mixed in both engines

Evidence:

- `methods/shooting_local.py` contains compiled SCF/angular kernels, a numerical
  driver and ShootingResult; `methods/shooting_hf.py` contains both physical
  exchange construction and numerical iteration.
- `numerics/shooting/solver.py:2`, `hf_shooting.py:2` and package `__init__.py:30`
  import implementations upward from methods.
- `numerics/shooting/scf_potential.py:185` imports a result adapter from scattering.
- `methods/dirac_fock.py` combines a physical method, B-spline matrix construction,
  SCF iteration and result definition in 2,497 lines at recheck. Persistence has
  since been extracted to IO; result ownership and iteration still need splitting.
- `methods/mcdhf/grid_shooting.py` combines electron-space construction and
  constrained optimization; `shooting_preconditioner.py` is a numerical kernel.
- `methods/mcdhf/shooting_solver.py:43` constructs an HF solver just to obtain
  its nuclear/environment potential.

Recommendation: perform the splits in the target design for **both** engines.
Numerics owns integration, assembly and optimization. Methods owns occupations,
CSF objectives, density contractions and physical functional choices. Potentials
must be constructible without creating an unrelated solver. Compatibility
facades are explicit temporary exceptions, not the dependency direction.

### F6 — High: calculation resolution is not centralized yet

Evidence: `atom.py:761` begins a large branching solve dispatcher, while method
wrappers and direct solvers also resolve options. `atom_inputs.py` and `types.py`
both define Exchange and Correlation with different API roles.
`calculation_spec.py:75` reconstructs a specification by inspecting solver
attributes; it is a result-recording boundary rather than the sole input resolver.

Recommendation: translate legacy inputs once into a validated resolved request,
then derive the result record from that exact request plus resolved numerical
choices. Expose capabilities by method, operator stage and engine. Reject ignored
options before solving. Keep result provenance separate from convergence outcomes.

### F7 — Medium: overlapping result interfaces preserve private-layout coupling

Evidence: `results.py:536` defines AtomicResult, `scattering/protocols.py:75`
defines AtomicResultLike, `providers/contract.py:764` defines a broader payload,
and `radial_state.py` supplies another native data boundary.
`scattering/protocols.py:145` still obtains grid data exclusively via `_bsp_data`.
`methods/_result_adapters.py:26` detects `_result`, may solve an Atom on demand,
and accepts bare results by `_bsp_data` presence.
`corrections/workspace.py` also owns generic orbital evaluation used elsewhere.

Recommendation: keep the distinct purposes, but define one shared radial-state
contract and compose it into atomic/provider contracts. Move shared radial
workspace evaluation below corrections and scattering. Isolate old-layout
translation in one adapter. Make “obtain or calculate a state” a visible workflow
operation, not a side effect of coercing a result.

Native snapshots are a useful improvement. Their lazy caching against mutable
legacy results still needs a documented lifecycle: finalize and freeze results,
or explicitly invalidate snapshots after allowed mutations. No stale-snapshot
numerical failure was measured in this review.

### F8 — Medium: correction orchestration and operator implementations overlap

Evidence: correction formulas reside in `corrections/breit.py`, `qed.py`,
`recoil.py` and `specific_recoil.py`; CI interaction contractions reside in
`methods/mcdhf`. `CorrectionStack` has useful support checks and optional strict
preflight, but defaults to recording skipped terms. Repeated finalization is
blocked; that is distinct from general operator-stage overlap detection.

Recommendation: separate operator definitions, representation-specific
contractions and application policy. Extend the existing CorrectionApplication
record instead of replacing it. Record requested/applied/unsupported/already
included terms and their stages. Production requests should explicitly select
strict capability enforcement. A QED Boolean must not stand for unspecified
operator content or accuracy.

### F9 — Medium: NIST application logic lives outside the installable package

Evidence: `scripts/generate_full_dhf_nist.py:38` modifies sys.path and imports
configuration parsing from `benchmarks/nist_asd.py`. It owns job selection,
physics/basis policies, execution and database updates.
`view_nist_spectra.py:26` imports benchmarks and scripts, then mixes queries,
scientific derived quantities and HTTP routing. `compare_theory_nist.py:28`
imports a network of audit/generation scripts and also renders plots.
The wheel packages only `src/atomx` (`pyproject.toml`).

Recommendation: package reusable archive/schema/normalization/state-matching code
under a reference-data subsystem; package calculation jobs and audit services
under workflows. Keep scripts thin command-line entry points. Separate HTTP
handlers and static assets from scientific queries. Version source and derived
schemas independently, with calculation hashes and matching reasons.

This finding is about reproducibility and deployment boundaries. It does not
establish that the current archive is complete, or that every comparison is
correct; those require a separate record-level audit.

### F10 — Medium: capability documentation and architecture gates lag the code

Evidence: `scattering/r_matrix/__init__.py:7` calls compute a scaffold although
`:35` delegates to implemented multichannel code. Coupled-cluster and RPA really
are explicit scaffolds. `NUMERICAL_ARCHITECTURE_REVIEW.md` includes superseded
method-location and MCDHF status statements. The earlier provider ADR is still
marked proposed while provider implementations exist.
`tests/test_numerical_design_contracts.py:46` checks a selected list of shooting
kernels, not all numerical modules or relative-import paths comprehensively.

Recommendation: publish a tested capability table, label historical review
sections, and enforce package dependency rules with resolved absolute/relative
imports plus a small compatibility allowlist. Keep unsupported methods explicit;
a present module or importable class is not a claim of implemented physics.

### F11 — Lower: eager top-level imports broaden runtime coupling

Evidence: `atomx/__init__.py:24` imports visualization and several downstream
subsystems; `viz/__init__.py:31` eagerly imports Matplotlib and attempts optional
backends. Packaging explicitly makes Matplotlib a core requirement for this reason.

Recommendation: retain convenient public names through lazy imports and load
renderers on use. Test minimal core installations independently of optional
provider/visualization extras. No startup performance gain was measured here.

## Subsystem assessment

| Area | Current strengths | Main design action |
|---|---|---|
| Physics, nuclear potentials, constants | Equation objects, shared constants, central Fermi parameter helper | Give operator/nucleus construction clear ownership; preserve explicit port conventions |
| Angular/configuration | Dedicated coupling, CSF and parsing modules | Keep pure angular algebra independent of radial engines; audit method-local duplicates during extraction |
| Shooting | Separated integrator/Green/response kernels and owned problem contexts | Complete SCF/optimizer split; eliminate upward imports |
| B-splines | Dedicated basis, quadrature, convolution and integral modules | Extract remaining assembly/iteration from the DHF monolith without losing band structure |
| CI/MCDHF/MBPT | Explicit correlated density and variational machinery | Separate numerical optimizers from state objectives; declare projector and supported spaces |
| Results/providers/IO | Native metric snapshots and validated external payloads | Repair persistence and provider provenance; reconcile shared contracts |
| Corrections | Support protocol, strict option, application records | Unify operator-stage accounting and generic radial evaluation |
| Properties/transitions | CI tensors, gauge metadata and result types | Make model/fallback policy explicit and stop catching generic failures as unsupported physics |
| Scattering | Dedicated process modules, continuum models and protocols | Move shared contracts below process packages; relocate large implementations out of package initializers |
| Plasma/CRM | Separate population/rate model and spectrum bridge | Preserve provenance of approximate rates and populations through synthesis |
| Spectrum/line shapes/UTA | Generally useful separation of broadening and synthesis | Carry state/model/resolution metadata end to end; preserve explicit statistical models |
| Fitting | Dedicated Slater-parameter fitting area | Keep fitted parameters and training/reference records distinct from ab-initio predictions |
| Visualization | Result-oriented renderer APIs | Lazy dependencies; keep scientific policy out of renderers |
| NIST/web/workflows | Extensive audits, generation and interactive views | Extract reusable application services from scripts and benchmarks |
| Tests/CI/build | External vs snapshot markers, strict xfail, multiple Python versions, nightly bounds/determinism gates | Add persistence/provider/model-equivalence and whole-package architecture gates |

Large files are signals for inspection, not automatic defects. For example,
`data/nist_bethe_log.py` is mostly tabulated data, while the large convolution
module has a coherent numerical responsibility. Split by ownership and lifecycle,
not an arbitrary line limit.

## Revisions needed to the proposed design

1. Add **versioned persistence** as an explicit boundary; it cannot remain a
   convenience method inside a solver implementation.
2. Integrate existing provider/capability contracts; do not create a parallel
   capability system. Shared radial contracts must be below scattering.
3. Add **workflows/reference-data/application** ownership for NIST ingestion,
   job execution and the website. Benchmarks must consume production services.
4. Specify result mutability and cache lifetimes, including native snapshots.
5. Give every fallback a requested model, actual model and structured reason.
6. Treat HF–Slater/Breit–Pauli labeling and ignored nuclear options as physical
   API correctness work, separate from relocation-only changes.

## Implementation order

**A. Protect scientific meaning.** Preserve the completed persistence and provider
repairs and their regression tests. Expose transition
fallback policy and preserve failure/model records. Clarify method names and
unsupported options.

**B. Establish contracts.** Single request resolution; shared radial/result
interfaces; explicit correction stages and capability checks. Maintain legacy
adapters at the public boundary.

**C. Move implementation by responsibility.** Shooting kernels and optimizers
into numerics; physical functionals into methods; B-spline assembly/SCF split;
result persistence into IO. Enforce import rules at each step and preserve
matched-model numerical controls.

**D. Package the application.** Extract NIST archive/matching and campaign
services, then HTTP endpoints and CLI wrappers. Carry the same resolved model
record into database rows and interactive comparisons.

**E. Consolidate and benchmark.** Update capability documentation, remove
internal uses of compatibility paths, measure representative runtime/memory
by numerical phase, and publish fresh artifacts. Do not combine relocation,
new physics and performance algorithm changes in one unreviewable patch.

## Verification limits

The 17 passing tests establish selected current contracts, not correctness of
the entire 71k-line package. No full suite, external GRASP/FAC comparison,
NIST completeness audit or full-table production calculation was run for this
architecture review. Performance recommendations identify work boundaries and
potential duplication, not measured hot spots. Those require phase profiling
on representative closed-shell, open-shell and correlated cases.
