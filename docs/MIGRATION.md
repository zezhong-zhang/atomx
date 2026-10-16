# Migration Notes

Changes are grouped by the version that introduced them. The package version
is `atomx.__version__` (`src/atomx/__init__.py`); see `CHANGELOG.md` for the
per-version summary.

## Unreleased

### Optional declared-mask sparse row index (2026-10-16)

`numerics.canonical_sparse.canonical_body_sparse(..., row_index="hash")`
adds an exact declared-mask index for selected fixed-N spaces. The default
`"colex"` representation remains unchanged. Hash collisions compare complete
uint64 determinant masks; row order, signed values, accumulation order and
complete CSC output are preserved. Exact prefix-population support from all
declared rows rejects only impossible matches before full-key lookup.
Both modes retain explicit work/memory
admission. Large CSC outputs account for SciPy's int64 index promotion.
This is a numerical representation option, not a new physical approximation,
core-correlation method or experimental accuracy certificate.


### Explicit native tensor CCSDT/EA consumer (2026-10-15)

`methods.tensor_coupled_cluster.CCSDTTensorOperator(integrals, memory_MiB=512)`
adds an opt-in real packed-native-integral consumer. `solve_core(retained_rank=2|3)`
returns core amplitudes, total/correlation energies and the projected equation
residual. `attachment_action(core, vector)` applies the complete directed R3
operator, including a final batch axis; `attachment_matrix` explicitly admits
its dense projected storage. `resource_plan` performs scalar tensor planning.
Existing finite CC/EA and external FSCC interfaces/defaults remain unchanged;
these result types are not interchangeable. Full physical states/residuals,
J-reduced production, valence-universal FSCC and dressed properties are not
inferred from the new core result. See [equations, provenance and live scope](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_tensor_ccsdt001/README.md).

`projection_storage='canonical'` now opts into complete independent output
accumulation; the default is still `'dense'`. `solve_core` adds opt-in
`scalar_sector=False`, `preconditioner=None|'fock_diagonal'` and
`initial=None|TensorCoreResult`. Exact complete-multiplet M/parity admission,
finite/nonzero unshifted denominators, same-operator initial ownership, exact
forbidden-coordinate zeros and final complete retained-residual checks apply.
An already-converged initial point avoids an unnecessary root call. These
options change numerical storage/solving, not retained equations or declared
ranks. See [complete4f physics, failures and matched timing](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_tensor_4f002/README.md); the local
J-projected Davidson driver is not an added public eigensolver API.

### Explicit analytic real-CI response ingredients (2026-10-14)

`numerics.variational_response.polar_path_derivatives(point, direction)` returns
the polar path's zero/first/second derivatives, retaining finite Gram and
tangency defects. `real_eigenstate_response(H0, H1, H2, energies, vectors, root)`
requires a complete real ordinary-metric finite eigensystem, checks residual/
Gram, rejects target degeneracy and enforces an explicit dense dimension cap.
It returns a read-only first coefficient response, first/second energy
derivatives and the **CI-response portion** of E'''/6; it supplies no H3 term,
total FD bound or state/physical-input ownership.

`physics.csf.radial_increment.bspline_dc_radial_derivatives(reference, first,
second, workspace, Z, ns, ...)` binds actual first/second P/Q derivatives to
native real B-spline DC integral keys. `components(key)` returns actual first/
second integral derivatives, including raw-h ket normalization and all Coulomb
cross terms. Breit/QED, grid and generalized-metric CI response are outside
these interfaces. These explicit lower-level ingredients change no existing
default, CI result/owner, finite-increment behavior or optimizer. See the
[analytic qualification and retained failures](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision012/README.md).

### Explicit streamed DC radial density and batched bilinears (2026-10-14)

`physics.csf.radial_density.CIRadialDensity` accepts
`angular_storage='streamed'`; `CSFOrbitalFunctional` selects this with
`density_storage='streamed'`. Both defaults remain cached. The new
`contract_bilinears(left, right)` accepts matching real CSF-by-column arrays
and returns one unnormalized signed radial-key dictionary per column pair.
Streamed contraction shares one complete connection traversal across columns,
without retaining angular tables or pair-by-state arrays. Bilinears remove
exact zeros only; the legacy ensemble `contract` retains a magnitude cutoff.
This does not change CI solver storage or create a full-spectrum state owner.
See [qualification and physical scope](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision010/README.md).

### Explicit numeric CI assembly without retained angular tables (2026-10-14)

`physics.csf.matrix.build_ci_matrix` and `angular.csf_hamiltonian.exact_block_matrix`
accept `angular_storage='streamed'`. The default remains `'cached'`. Streamed
assembly contracts the same signed angular terms directly into numeric
determinant CSR, with a fresh radial-key value cache for each operator build.
It preserves complete declared spaces, static Breit and complex c-products;
it does not add sparse full-spectrum state ownership. Density storage is
selected separately as described above. Both modes now evaluate projected
diagonals from exact occupation-pattern principal blocks, avoiding a global
`Hdet @ projection` temporary. Full cross-pattern actions are retained.
See [source-bound qualification and physical-run status](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision009/README.md).

### Exact angular connection and CSR storage (2026-10-14)

Internal `_connected_pairs` now returns an ordered iterator. Diagnostic callers
that need a list must materialize it explicitly. Singleton spectator buckets
are discarded after compact first-occurrence indexing; their exact diagonals
are emitted explicitly. `_coefficient_table` builds typed CSR buffers directly
and preserves imaginary callback values through complex promotion. Native
real-angular coefficients remain FP64, with unchanged keys, coordinates and
CSR results in the complete A/B controls. Public CSF/Hamiltonian entry points
and physical spaces are unchanged. See [measured scope](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision008/README.md).

### Sparse internal orbital-density projection (2026-10-14)

`CIRadialDensity.blocks` now stores the existing CSR CSF projection instead of
a dense array. `contract` and `contract_bilinear` keep their signatures and
coefficient policies. Internal diagnostic consumers that need an array must
explicitly call `.toarray()`; ordinary density/gradient consumers use `@`.
The physical CSF space and public full-spectrum state contracts are unchanged.
See [qualification and limits](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision006/README.md).

### Explicit generalized eigenblock refinement (2026-10-14)

`numerics.bspline.dirac_assembly.refine_generalized_eigenblock` adds an opt-in
full-coordinate leading-band refinement with joint metric controls and full
residual reporting. It retains all supplied columns and the caller's real
H/S. No DF/virtual factory default or MCDHF frozen-projector policy changes.
Existing `_refine_eigenpair` callers keep their Euclidean selection objective;
the new block consumer explicitly supplies the original S-dual objective.
See [scope, resource limits and live evidence](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/eigenblock_native001/README.md).

### Result-metadata channel, solve() return, and CI metric audit (2026-10-10)

`Atom.solve(method=...)` now always returns the `Atom` with the result on
`.result`; previously the method path returned the raw result object.

The `ATOMX_CI_METRIC_CHECK_EVERY` environment knob was removed. The CI
orbital-metric audit now runs content-fingerprinted: on the first build per
workspace and again on any orbital content change.

Corrections-layer provenance (`_breit_provenance`, `_nms_provenance`,
`_sms_provenance`, `_hydrogenic_mass_remainder_provenance`) and the
`_E_Breit_operator*` expectation values moved from loose result attributes
into the declared `solver_diagnostics` dict. Archives written with the
legacy top-level metadata keys migrate into `solver_diagnostics` on load;
an explicit `solver_diagnostics` entry takes precedence.

### Native scalar linearized SD (2026-10-09)

`methods.linearized_sd` is a separate lower-level closed-core linked linearized
equation solver and scalar resource planner. Existing nonlinear CCSD/FSCC and
finite NR-CISD APIs keep their equations and defaults. No Atom dispatch or
automatic substitution is added. [Scope and caps](ATOMIC_LINEARIZED_SD.md)
distinguish the finite J=0 DC/DCB anchor from the direct reduced DC core path
in `methods.core_sd`. The latter requires an isolated source-bound pCI build;
it does not automatically replace FSCC or add valence screening. SD and FSCC
now share `methods.atomic_space.build_atomic_space` for complete native VN
virtual bands, without a per-kappa count cutoff.

The optional exact `core_single_to_doubles_blas` kernel now has matching
planner, `CoreSD`, and core/valence/screening-builder options (CLI
`--core-single-to-doubles-blas`). Its default remains false; explicit positive
workers and LP64 BLAS are required, with a bound OpenMP runtime and one inner
thread for concurrent calls. Complete scratch is included in preflight.
`CoreSD.solve(initial_amplitudes=...)` accepts a complete signed iterative seed,
records its hash, and freshly validates the current full equation and energy;
it does not import a prior result's identity or residual claim.

`methods.valence_sd` adds a separate fixed-energy CI+all-order interface on a
solved `CoreSD`. Plan with `plan_valence_sd`, build a fresh combined backend
with `tools/build_pci_valence_backend.py`, then use `ValenceSD` with the matching
core result and explicit column energies. Full scalar noncanonical Fock,
signed correlation Sigma1 rows and implicit energy response are retained.
Core amplitudes and physical/source identities must match. Results do not
represent a solved excitation spectrum or a complete CI+all-order Hamiltonian;
no default method substitution or CI effective-Hamiltonian consumer is added. See the
[fixed-energy usage and finite validation](ATOMIC_LINEARIZED_SD.md#fixed-energy-native-ciall-order-valence-equations).

`methods.sigma2_sd` separately adds complete directed reduced screening and
joint energy response from source-bound `ValenceSDResult` columns. Plan with
`plan_sigma2_sd`, compile exact dimensions with `build_pci_sigma2_backend.py`,
then use `Sigma2SD(valence).evaluate(columns)`. Its public magnetic query restores
original S1 normalization. This does not inject raw directed results into a
Hermitian CI matrix: the current finite consumer audit measures nonzero adjoint
defects. Source/core identity, energy and effective-metric contracts must be
preserved; [scope and gates](ATOMIC_LINEARIZED_SD.md#complete-fixed-energy-reduced-sigma2)
remain distinct from physical excitation accuracy.

### Separate tensor CCSD/FS-CCSDT backend

`methods.fscc.compute` now source-binds the numerator-division engine and
includes the known legacy1e-14 numerator-skip floor in its raw residual bound.
A zero legacy fixed-point update can therefore return a positive bound.
Known legacy source manifests remain admitted; unknown/mislabeled engine
versions reject before export. The optional builder flag
`--exact-nonzero-division` preserves every nonzero numerator without changing
denominators or physical equations. It requires the qualified source SHA,
capability and binary marker; defaults and sector coverage are unchanged.
Provenance adds `numerator_skip_threshold_Ha`,
`numerator_skip_residual_floor_Ha` and `denominator_kernel_source_sha256`.
Historical saved bounds are unchanged; see the [source contract](ATOMIC_FSCC.md#numerator-division-source-contract).

For `sector="0h1p"`, `FSCCResult.model_space_matrix_Ha` and its saved NPZ now
use the qualified physical ket convention (raw formatted HEFF adjoint),
relative to the correlated core. `read_heff` and `effective_matrices` remain
raw. Do not reinterpret older saved matrices under the new convention without
checking their source/provenance. Every 0h1p request now rejects missing or
changed producer-convention signatures before export. Closed energies and the
raw hole-sector convention remain unchanged. No Hermitian averaging, property
metric or omitted-rank certificate is added. See [source-bound complex and
live validation](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/public_heff_convention001/README.md).

Noncanonical `methods.fscc.compute(..., sector="0h1p")` now raises `ValueError`
before export when any actual occupied–virtual Fock element is nonzero and the
backend lacks the source-bound `ATOMX_FOV_T2_001` attachment equation repair.
Rebuild the pinned backend with the existing `--noncanonical-attachment-fix`
flag; do not delete Fov to bypass the check. The historical builder default,
exactly zero-Fov attachment and other sectors retain their separate contracts.
No result fields are renamed. Request provenance adds
`occupied_virtual_fock_max_Ha`, `noncanonical_attachment_equation` and
`residual_scope`. See the [same-H public-entry validation](atomic_physics_correctness_2026_10_11/README.md).

The native factories now accept `representation="on_demand"` and an explicit
`radial_pair_cache_size`. Packed remains the default. `export_integrals` streams
the same ATXCC001 signed file order and now rejects a logical file request above
2 GiB before creating output (override `max_integral_file_bytes` explicitly when
appropriate). Resource estimates continue to include the backend's dense ERIs
and tensors. See [current contracts](ATOMIC_FSCC.md); native streaming is not an
angular-reduced CC amplitude solver.

`methods.fscc` is the new external integral-driven interface; it does not expand
the finite `methods.coupled_cluster` API or its determinant caps. Explicit source-
bound EXP-T builds, native signed DC integrals, frozen-core normal ordering and
complete declared virtual bands are required. Full triples use DIIS for triples;
unsupported physics and over-budget spaces fail explicitly. See [current usage
and limits](ATOMIC_FSCC.md).

### Explicit finite CCSD/CCSDT, attachment and SD/DCB interfaces

The former `methods.coupled_cluster.compute` scaffold now requires aligned
`hamiltonian_Ha`, complete fixed-N `determinants`, `reference_mask`, and explicit
`spinor_count` / `reference_identity`. It returns a finite nonlinear CCSD or
connected CCSDT result, including total energy, immutable amplitudes/coefficient
vector, separate projected and full-state residuals, and
`n_function_evaluations`. Higher ranks and FSCC reject. The separate
`methods.eom_attachment.compute` consumes owned native integrals and declares
its retained 1p/2p1h/3p2h space; omitted-rank full-H defects remain separate.
Both finite solvers now accept `action_representation="state_action"` and
`cluster_term_application_cap=2_000_000` without increasing determinant or
amplitude caps. Default dense remains. The option removes generator banks;
complete sectors/dense Hamiltonians remain and the current small case runs
slower. See [matched finite integration](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/state_action_integration001/README.md).
Neither interface has Atom facade dispatch or production atomic scaling.

`methods.ci_sd.plan_sd_ci_space` materializes complete declared reference-union
NR single/double configurations with resource guards. The finite P/Q factory
accepts `include_breit=True` for static Breit in every matrix block; default DC
behavior is unchanged. Do not append its restricted second-order controls to an
already solved DCB state. See [contracts and measured limits](yb_sd_breit_completion_2026_10_09/README.md).

`FiniteCIModel.bloch_reduction(root_indices)` explicitly selects a complete
finite invariant subspace and returns its owned wave operator, physical metric,
directed/orthonormal Hamiltonians and biorthogonal states. Transform full-space
properties with `effective_operator(..., representation="bloch"|"orthonormal")`.
It rejects split degeneracy, singular projection and excess resource requests;
it supplies no inferred metric for the separate SD/EXP-T effective matrices.
See the [current metric contract](ATOMIC_LINEARIZED_SD.md#exact-finite-bloch-wave-operator-and-physical-metric).

### Independent DF orbital controls and explicit-size basis correction

`Atom.solve(orbital_tolerance=...)` and `Sampling(orbital_tolerance=...)`
set a dimensionless L2 orbital-update threshold, independently of the energy
criterion. The explicit solve argument wins. Direct Hamiltonian DF supports
the control on both engines; unsupported methods reject it. Requested
controls and measured updates are recorded. See [solver controls](DF_SOLVER_CONTROLS.md).

Hamiltonian DF with an explicit `Basis(ns=...)` now uses the balanced P/Q
B-spline representation, as the automatic physics grid already did. The old
unbalanced representation could falsely converge to a spurious positive-kappa
root; affected calculations must be regenerated. Basis size remains explicit,
and basis orders are recorded in the calculation specification. B-spline CSF
results now report final optimizer sweeps as `iterations`, rather than the
initializer's outer-iteration count.

### Radiative multipole and FAC conventions

- FAC transition dictionaries now use `A_per_s` and
  `reduced_matrix_element`, replacing the incorrect `gA_per_s` and `gf_alt`
  meanings. No extra upper-state degeneracy division belongs on FAC's printed
  rate. Aggregated `MULTIP=0` rows expose `aggregate_gf` and no reduced amplitude.
- `write_tra` explicit rows require the corrected rate and amplitude fields.
  Atom-generated rows preserve A directly and export a positive amplitude
  magnitude; coherent phases cannot be recovered from a line strength.
- Higher-multipole oscillator strengths include their photon-frequency factors.
  Magnetic angular normalization is corrected, and bound transitions and
  B-spline recombination pass photon energy to `radial_mL(..., omega=...)`.
  Calling that low-level kernel without omega explicitly retains its static
  limit. `TransitionResult.photon_approximation` records the operator choice.
- Historical FAC rate tables using the old gA interpretation require
  regeneration. See [current verification evidence](VERIFICATION.md).

### Optional shooting refinement

`Sampling(radial_refinement=RadialRefinement(...))` opts into repeated grid
and automatic-box checks for single-configuration shooting. Import
`RadialRefinement` from `atomx` and pass the sampling through `Basis`.
The point count becomes the starting count; explicit radial boxes stay fixed.
`result.radial_convergence` records outcomes and limits. Ordinary fixed-grid
calls keep their existing controls. See [solver controls](DF_SOLVER_CONTROLS.md).

### Direct Slater reference integration

`numerics.bspline.slater.slater_rk_direct` now takes a two-dimensional
cell Gauss grid and weights, matching `bsp_data["gr"]` and `bsp_data["grw"]`.
Flattened grids cannot identify cell boundaries and are rejected. Density
arrays may still be flat. The public `slater_rk(..., method="direct")`
signature is unchanged. Its independent reference integration now splits
coincident cells along the Coulomb cusp; the default Poisson method is
unchanged.

### Removed forwarding modules and misleading aliases

Imports now name the implementation directly. These old modules and names
have been removed; there is no compatibility forwarding layer.

| Removed path/name under `atomx` | Use instead |
|---|---|
| `methods.hartree_fock.HartreeFock`, public `HartreeFock` | `HartreeFockSlater` (local Slater exchange, not nonlocal HF) |
| `methods.relativistic_hf.BreitPauli` | `methods.breit_pauli.BreitPauli` or public `BreitPauli` |
| `scattering.protocols` | `state_protocols` |
| `corrections.workspace` | `numerics.radial_workspace` |
| `numerics.shooting.scf_potential` | `potentials.static_field` |
| `properties.hyperfine.qed_corrections` and its constant exports | `data.nuclear_structure.TABLE[isotope]` |
| Yk imports through `numerics.bspline.evaluate` | `numerics.yk.compute_yk`, `yk_flat`, `yk_partial_matrices` |

For example, the proton Zemach radius is
`TABLE["h1s"].zemach_radius_fm`; its polarizability contribution is
`TABLE["h1s"].polarisability_fraction`. This is tabulated nuclear data.
The removed `qed_corrections` module contained no hyperfine QED operator
implementation. The existing hyperfine correction ladder retains its
documented empirical coefficients and limitations.

The `methods.mcdhf.solver` entry point remains: its distinct method/result
types select explicit multiconfiguration calculations. Package `__init__`
exports remain the public API.

### Default DHF Hamiltonian averaging

Single-configuration `Atom(..., exchange="hf")` now selects
`Correlation(averaging="hamiltonian")`: ground-state intermediate coupling
within the CSFs of the specified nonrelativistic configuration. Coefficients
come from Hamiltonian diagonalization; occupations are `sum_j |c_j|² n_ij`
(Guerra et al., *Relativistic calculations of screening parameters and atomic
radii of neutral atoms*, 2016 revision, Eqs. 5 and 10). Orbital optimization
uses the full `c†Hc` derivative, including off-diagonal CSF interference.
This does not generate excited configurations or add dynamical correlation.

`statistical` remains the distinct fixed degeneracy ensemble; `equal` gives
each integer jj distribution equal weight. `fractional` uses products of
mean occupations. `generate_jj_average` accepts only the fixed `statistical`
and `equal` choices: it cannot compute Hamiltonian eigenvector weights.

Single-CSF configurations retain the ordinary DHF solver. Other Hamiltonian
DF cases use direct, self-consistent radial source equations on both engines.
They share Hamiltonian contractions and result assembly with MCDHF, but do not
enter its finite-continuum orbital optimizer. The numerical engines follow the
same self-consistent electron branch. Explicit MCDHF retains its fixed no-pair
variational space. The orbital functional is Dirac–Coulomb;
Breit enters the subsequent state Hamiltonian, and QED/recoil retain their
recorded correction stages. No claim of self-consistent Breit/QED orbital
optimization is made. Direct DF requires radial grid/box refinement independently
of SCF convergence. `Basis(shooting_space=...)` now requires explicit MCDHF;
use `Basis(sampling=Sampling(n_grid=...))` to refine direct shooting DF.
See [DF solver controls](DF_SOLVER_CONTROLS.md) for routing and timing evidence.

An old single-configuration `Correlation(averaging="ci")` request now raises
with migration instructions. Use `averaging="hamiltonian"` for DHF. To request
MCDHF explicitly, pass a configuration list (even a one-item list), a correlation
expansion, or an explicit `MCDHF` method. No silent theory switch is retained.

The density CLI follows this default; `--correlation s/sd` selects MCDHF.
The demo begins with a source-path cell to avoid importing an editable
installation of another checkout. Restart a kernel that already loaded it.


### Shared CSF implementation

The general state machinery is no longer owned by MCDHF. Internal imports
must use the canonical modules; the removed modules have no forwarding shims.

| Responsibility | Canonical location |
|---|---|
| CSF definitions/enumeration (formerly `methods.mcdhf.config`) | `config.csf` |
| Exact angular Hamiltonian tables (formerly `methods.mcdhf.exact_ci`) | `angular.csf_hamiltonian` |
| Hamiltonian matrix elements, Breit/retardation, densities, one-body contractions | `physics.csf` |
| Shared state functional and variational driver | `methods.csf` |
| Monoconfiguration intermediate-coupling DF entry point | `methods.hamiltonian_df` |
| Expanded configuration-space entry point | `methods.mcdhf` |
| B-spline constrained orbital optimizer | `numerics.bspline.orbital_optimization` |
| Shooting constrained orbital optimizer | `numerics.shooting.orbital_optimization` |

State-consuming properties now use `Atom._csf_result` for both DF and MCDHF;
public `Atom.result`, `state_energies`, `waves` and `density` remain the accessors
for users. The DF result records that no configuration expansion was generated.

### Shooting engines and result ownership

Update downstream callers using this table. Repository source, scripts,
benchmarks and tests now use the canonical paths. No forwarding modules remain
at the removed locations. Physical contractions previously accessed through HF
solver private methods now live under `solver.functional`, for example
`solver.functional.two_electron_energy(...)`.

| Previous import (under `atomx`) | Canonical import |
|---|---|
| `methods.shooting_local.ShootingSolver`, `numerics.shooting.solver.ShootingSolver`, or the shooting package export | `atomx.numerics.shooting.local.ShootingSolver` |
| `methods.shooting_hf.ShootingHFSolver` or `numerics.shooting.hf_shooting.ShootingHFSolver` | `atomx.numerics.shooting.hartree_fock.ShootingHFSolver` |
| `ShootingResult` through a shooting solver or package | `atomx.results.ShootingResult` |
| `dfs_frozen_orbital` through the old shooting solver | `atomx.numerics.shooting.local.dfs_frozen_orbital` |
| Slater integrals and `_yk_over_r` through the shooting solver | `atomx.numerics.shooting.coulomb` |
| Inhomogeneous orbital helpers through `hf_shooting` | `atomx.numerics.shooting.dirac_orbital` |
| Homogeneous pair and Green solve through `hf_shooting` | `atomx.numerics.shooting.dirac_green` |
| `methods.mcdhf.grid_shooting` electron-space builders | `atomx.numerics.shooting.electron_space` |
| `methods.mcdhf.grid_shooting.optimize_grid_shooting` | `atomx.numerics.shooting.orbital_optimization.optimize_grid_shooting` |
| `methods.mcdhf.grid_shooting.occupied_1s2_reference_potential` | `atomx.methods.csf.grid_functional.occupied_1s2_reference_potential` |
| `methods.mcdhf.shooting_preconditioner` | `atomx.numerics.shooting.preconditioner` |
| `methods.mcdhf.shooting_solver.ShootingSpace` | `atomx.numerics.shooting.electron_space.ShootingSpace` (public `atomx.ShootingSpace` is unchanged) |
| `methods.mcdhf.shooting_solver.solve_variational_shooting` | `atomx.methods.csf.grid_solver.solve_variational_shooting` |
| `numerics.analytic[.hydrogenic_dirac]` hydrogenic helpers | `atomx.methods.hydrogenic_reference` |
| `numerics.bspline.continuum.photoionization_cross_section` | `atomx.scattering.photoionization.photoionization_cross_section` |

The old `numerics/analytic` namespace contained numerical B-spline reference
calculations, not closed-form wavefunctions. Its forwarding files are removed.
Historical design and audit documents may retain the old paths as evidence of
the tree they reviewed.

#### Saved results

`ShootingResult` retains its fields, native arrays, metric and physical metadata.
Schema-1 shooting HDF5 files use a format identifier rather than a Python module
path, so they require no rewrite. Python pickle import compatibility is not
provided by this change; the versioned archive is the supported persistence path.

Local Slater results explicitly have no jj configuration ensemble. The archive
writer and validator now accept that `None` value and preserve it as JSON null;
previously they attempted to iterate it. Nonlocal HF ensembles continue to be
validated and preserved for open-shell correction replay.


Numerical modules now reject upward imports without compatibility exceptions.
See [migration verification](archive/2026-09/NUMERICAL_BOUNDARY_MIGRATION_2026-09-29.md).


### The `Atom` constructor is ten arguments (breaking — every call site changes)

`Atom.__init__` used to take 28 flat keyword arguments mixing six unrelated
physical concerns, plus a short-lived "grouped dataclass" layer on top
(`System`, `ElectronicStructure`, `Basis`, `Interaction`, `CIProtocol`,
`Environment`).  Both are gone.  There is now **one** spelling per physical
question, and every retired keyword raises a `TypeError` naming its
replacement.

```python
# before
ne = Atom(Z=10, config="1s2 2s2 2p6", numerics="bspline", ns=200,
          grid_hi=0.02, grid_he=0.10, breit=False, qed=False,
          averaging="statistical", j_resolved_csfs=False,
          plasma_ne=1e22, plasma_Te=100.0)

# after
from atomx import Atom, Basis, Corrections, Correlation, Environment

ne = Atom(
    10,                                   # Z (positional: the atom's identity)
    config="1s2 2s2 2p6",                 # or a list for multi-configuration
    charge=0,
    nuclear="fermi",
    exchange="hf",                        # or Exchange(model=..., alpha_x=...)
    corrections="none",                   # or Corrections(...)
    correlation=Correlation(averaging="statistical"),   # or "s" / "sd"
    engine="bspline",                     # was numerics=
    basis=Basis(ns=200, grid_hi=0.02, grid_he=0.10),
    environment=Environment(plasma_ne=1e22, plasma_Te=100.0),
)
ne.solve()
```

`Z` and `config` are the only positional slots; everything after them is
keyword-only.  The old signature accepted 28 positional-or-keyword
parameters, so a stale positional call would have silently bound `charge` as
`ns`; now it raises.

#### Every retired name and its replacement

| retired | replacement | notes |
|---------|-------------|-------|
| `configs=[...]` | `config=[...]` | a *list* still selects the MCDHF/CI path; a plain string is one configuration |
| `numerics=` | `engine=` | the argument selects an engine, not "numerics" |
| `ns=` | `basis=Basis(ns=...)` | B-spline count |
| `ks=` | `basis=Basis(ks=...)` | B-spline order |
| `r_max=` | `basis=Basis(r_max=...)` | radial box (Bohr) |
| `grid_hi=` | `basis=Basis(grid_hi=...)` | DBSR_HF knot-shape override |
| `grid_he=` | `basis=Basis(grid_he=...)` | DBSR_HF knot-shape override |
| `grid_hmax=` | `basis=Basis(grid_hmax=...)` | DBSR_HF knot-shape override |
| `starting_potential=` | `basis=Basis(starting_potential=...)` | SCF initial guess |
| `sampling=` | `basis=Basis(sampling=...)` | accuracy tier; `solve(sampling=...)` still exists |
| `alpha_x=` | `exchange=Exchange(model="slater", alpha_x=...)` | only the Slater model has a coefficient |
| `slater_method=` | `exchange=Exchange(model=..., slater_method=...)` | only the local models, and only on the shooting engine |
| `breit=` | `corrections=` | `Corrections(breit=False)` or a preset |
| `qed=` | `corrections=` | `Corrections(qed=False)` or a preset |
| `qed_backend=` | `corrections=Corrections(qed_backend=...)` | now also reaches multi-configuration runs |
| `correction_stack=` | `corrections=Corrections(stack=...)` | the stack stays authoritative |
| `averaging=` | `correlation=Correlation(averaging=...)` | Guerra CSF weights |
| `radial_optimization=` | `correlation=Correlation(radial_optimization=...)` | MCDHF outer loop |
| `j_resolved_csfs=` | `correlation=Correlation(j_resolved_csfs=...)` | CSF enumeration |
| `per_block_orbital_sets=` | `correlation=Correlation(per_block_orbital_sets=...)` | DODS/EOL scaffold |
| `plasma_ne=` | `environment=Environment(plasma_ne=..., plasma_Te=...)` | both are required together, and are now applied |
| `plasma_Te=` | `environment=Environment(plasma_ne=..., plasma_Te=...)` | as above |
| `system=System(...)` | `charge=`, `nuclear=` | the interim group is gone |
| `electronic=ElectronicStructure(...)` | `config=`, `correlation=` | |
| `interaction=Interaction(...)` | `exchange=`, `corrections=` | |
| `ci=CIProtocol(...)` | `correlation=Correlation(...)` | |
| `basis=Basis(numerics=..., grid_hi=...)` | `engine=`, `basis=Basis(grid_hi=...)` | `Basis` no longer carries the engine |
| `environment=Environment(plasma_ne=...)` | unchanged spelling | but the fields now do something |

`Z`, `config`, `charge`, `nuclear`, `exchange`, `correlation` and
`environment` keep their names; `environment` and `basis` keep the keyword
but their objects changed (see the last two rows).

#### Worked examples, one per concern

**1. System — `charge`, `nuclear`**

```python
# before
he_plus = Atom(Z=2, charge=1, nuclear="point")
# after
he_plus = Atom(2, charge=1, nuclear="point")
```

**2. Electronic structure — `config` / `configs` / `correlation`**

```python
# before
be = Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"])
be = Atom(Z=4, correlation="sd", averaging="statistical")
# after
be = Atom(4, config=["1s2 2s2", "1s2 2p2"])
be = Atom(4, correlation=Correlation(expansion="sd",
                                     averaging="statistical"))
# `correlation="sd"` alone is still accepted (the string form):
be = Atom(4, correlation="sd")
```

**3. Radial basis and engine — `numerics`, `ns`, `ks`, `r_max`, `grid_*`,
`starting_potential`, `sampling`**

```python
# before
ne = Atom(Z=10, numerics="bspline", ns=200, ks=9, r_max=90.0,
          grid_hi=0.02, grid_he=0.10, starting_potential="tfda",
          sampling=Sampling.high)
# after
from atomx import Atom, Basis, Sampling

ne = Atom(10, engine="bspline",
          basis=Basis(ns=200, ks=9, r_max=90.0,
                      grid_hi=0.02, grid_he=0.10,
                      starting_potential="tfda",
                      sampling=Sampling.high))
```

**4. Interaction model — `exchange`, `alpha_x`, `slater_method`, `breit`,
`qed`, `qed_backend`, `correction_stack`**

```python
# before
fe = Atom(Z=26, charge=16, config="1s2 2s2 2p6", numerics="shooting",
          exchange="slater", alpha_x=2 / 3, slater_method="slag6",
          breit=False, qed=True, qed_backend="mohr")
# after
from atomx import Atom, Corrections, Exchange

fe = Atom(26, charge=16, config="1s2 2s2 2p6", engine="shooting",
          exchange=Exchange(model="slater", alpha_x=2 / 3,
                            slater_method="slag6"),
          corrections=Corrections(breit=False, qed=True,
                                  qed_backend="mohr"))
```

**5. Correlation / CI protocol — `averaging`, `radial_optimization`,
`j_resolved_csfs`, `per_block_orbital_sets`**

```python
# before
n = Atom(Z=7, config="1s2 2s2 2p3", averaging="ci",
         radial_optimization="fac_frozen", j_resolved_csfs=True)
# after
from atomx import Atom, Correlation

n = Atom(7, config=["1s2 2s2 2p3"],
         correlation=Correlation(averaging="ci",
                                 radial_optimization="fac_frozen",
                                 j_resolved_csfs=True))
```

**6. Environment — `plasma_ne`, `plasma_Te`**

```python
# before: accepted and then ignored — the run was an ordinary vacuum atom
ne = Atom(Z=10, charge=2, plasma_ne=1e22, plasma_Te=100.0)
# after: the Debye-screened potential is added to the nucleus inside the SCF
from atomx import Atom, Environment

ne = Atom(10, charge=2,
          environment=Environment(plasma_ne=1e22, plasma_Te=100.0,
                                  screening="debye"))
```

#### Behaviour changes that come with the migration

Four of the retired keywords used to be accepted and *dropped*.  They now do
what they say (measured sizes in `CHANGELOG.md`):

* `plasma_ne`/`plasma_Te` (now `environment=`) screen the nuclear potential
  inside the SCF.  A run that passed them and relied on the numbers being
  the vacuum ones will move — by design; that is the point of the argument.
* `Sampling.r_max` and `Sampling.n_grid` (now `Basis(sampling=...)`) reach
  the solvers: a tier with a non-default `r_max` now changes the box, and a
  non-default `n_grid` changes the shooting grid.
* `qed_backend` (now `Corrections(qed_backend=...)`) reaches
  multi-configuration runs instead of being replaced by the default `auto`.
* `slater_method` on the B-spline engine now **raises**: the Galerkin path
  evaluates its Slater integrals with its own quadrature and never read it.

Two more names changed meaning:

* `atomx.Exchange` / `atomx.Correlation` are now the constructor value
  objects from `atomx.atom_inputs`.  The legacy enums used by the *method*
  API (`DiracFock(exchange=Exchange.SLATER)`, `MCDHF(correlation=...)`) are
  still available as `atomx.types.Exchange` / `atomx.types.Correlation` and
  are still accepted as values by `Atom(exchange=...)`.
* `Atom.provenance` now reports `{"engine": ..., "exchange": ...,
  "averaging": ...}` (the key was `"numerics"`), plus a `"plasma"` entry
  when the atom sits in one.

### Grouped `Atom` constructor inputs, interim design (withdrawn)

The six grouped frozen dataclasses (`System`, `ElectronicStructure`,
`Basis`, `Interaction`, `CIProtocol`, `Environment`) and the
`system=`/`electronic=`/`basis=`/`interaction=`/`ci=`/`environment=`
keywords were an additive layer on top of the 28 flat keywords, and
`tests/test_atom_input_surface.py` pinned the pair.  They were superseded by
the single canonical surface above: `Basis` and `Environment` survive with
new fields, the other four are gone.

### Package version is single-sourced

`pyproject.toml` no longer carries a literal version — it declares
`dynamic = ["version"]` and hatchling reads `src/atomx/__init__.py`. The two
values had drifted (`0.1.0` in the metadata that `pip`/`uv` reported,
`0.2.0` in `atomx.__version__`); the code value wins. Nothing to change in
downstream code, but `importlib.metadata.version("atomx")` now agrees with
`atomx.__version__`.

### Heavy visualization/notebook packages moved to extras

`pip install atomx` no longer pulls `manim`, `plotly`, `nbformat`, `ipython`
or `ipykernel`. Install the back-end you use:

```bash
pip install "atomx[viz]"          # plotly
pip install "atomx[viz-manim]"    # manim (also needs system pangocairo)
pip install "atomx[notebook]"     # nbformat/ipython/ipykernel
pip install "atomx[all]"
```

`matplotlib` stays a base dependency because `import atomx` imports
`atomx.viz.mpl` unconditionally. Code that relied on a transitive plotly/manim
import should import it explicitly.

### Test markers renamed

`pytest -m fac_parity` no longer selects anything. The marker conflated two
different things and was renamed accordingly:

| old | new | meaning |
|-----|-----|---------|
| `fac_parity` (stored external data) | `external_fac` | asserts atomx against stored external FAC reference data |
| `fac_parity` (self-regression) | `snapshot` | asserts atomx against its own recorded output |

`pytest` also runs with `--strict-markers` now, so a stale selector fails
loudly instead of silently collecting nothing.

### Benchmark scripts take their reference code from the environment

`benchmarks/fac_runner.py`, `bench_local_exchange.py`,
`bench_shooting_vs_radial.py`, `scripts/energies_gs.py`,
`scripts/density_gs.py` and the FAC calibration helper no longer contain
machine-specific absolute paths. Set `FAC_SFAC_PATH` (or `FAC_BIN` /
`FAC_BIN_DIR`), `FAC_DYLD_LIB`, `RADIAL_DHFS_DIR`, `ATOMX_MCDHF_REF_CSV`,
`ATOMX_WF_TXT_DIR` as needed; each script stops with an actionable message
when its reference is missing rather than silently skipping.

## 0.2.0 — QED backend default

`Atom(..., qed=True)` now defaults to `qed_backend="auto"`.

`auto` keeps the historic Mohr/MSS self-energy backend for `Z <= 36` and
uses the FAC/Indelicato-Mohr K/L-shell backend for `Z >= 37`.  This closes
the documented high-Z residual against FAC.

Users who need bit-stability against the old MSS-only output should pass:

```python
Atom(82, config="1s1", corrections=Corrections(qed_backend="mss"))
```

## Finite-frequency electric radiation

One-photon transition, photoionization, RR and orbital DR radiative-width
calculations now use the finite-frequency electric operator. Both electric
gauges retain the spherical Bessel dependence. Static response calls remain
static. See [operator conventions](TRANSITION_OPERATORS.md).

For direct radial calls, `radial_eL` accepts `kappa_i`, `kappa_f` and signed
`omega=E_f-E_i`; omitting omega selects its static limit. The velocity kernel
now requires both kappas and signed nonzero omega and raises when missing;
the former arbitrary fallback is removed. Its sign convention now agrees
with length gauge. For momentum/static diagnostics explicitly request
`long_wavelength=True`.

`ci_line_strengths` defaults to `photon_approximation="finite_wavelength"`.
Its `CILineSet.trace` is `None` because each line has a different operator.
Request `photon_approximation="long_wavelength"` for static lines and the
mixing-invariant determinant-trace identity.

## Direct-momentum specific mass shift

`compute_sms_abinitio(orbitals, workspace, *, Z=None)` no longer accepts
orbital energies. It evaluates direct Vinti integrals and uses occupation
fractions in the reduced-angular sum. Supply `Z` for the point-Coulomb Breit
SMS2/SMS3 terms; omit it for the leading momentum operator. Open shells now
raise instead of being silently treated as closed shells. The old neon SMS
value near -117.84 is invalid; see [recoil conventions](RECOIL_OPERATORS.md).
The unequal-j relativistic SMS2 phase is also corrected in the common
recoil implementation, affecting the SMS correction stack as well as properties.

## Finite-nucleus basis resolution and orbital accuracy

Fixed-size Fermi B-spline bases now use the shared nuclear surface scale for
the first knot, replacing `0.05/Z`, which failed to resolve light nuclei.
Increasing `ns` alone previously left this error unchanged. Energies and
properties from those under-resolved fixed bases must be regenerated.

`ShootingHFSolver.solve(orbital_tolerance=...)` can independently converge
orbital shapes for first-order properties while retaining the requested
energy stopping criterion. The default remains unchanged; this native
control is not yet an `Atom.solve`/`Sampling` option. See
[accuracy and timing evidence](DF_SOLVER_CONTROLS.md#independent-orbital-accuracy-for-properties).

## Radial collision quadrature and stable Coulomb kernels

`dw_eii_cross_section`, `eii_cross_section` (DW portion) and
`dw_excitation_cross_section` now accept `radial_refinement=1`. A positive
integer subdivides every observable integration interval, retaining the
same endpoints and original nodes. This control is separate from the
continuum propagation grid (`ContinuumModel.shooting(n_grid=...)` or
`shooting_n_grid`) and outgoing-energy quadrature. EII provenance records
the refinement, mesh size and trapezoidal rule. Refinement does not certify
convergence automatically.

The general numerical entry point is
`make_fine_grid(..., refinement=2)`. Resource caps raise instead of silently
coarsening the requested mesh. `RadialMetric.trapezoid` owns the nonuniform
weights, including the two half-weight endpoints; the CE, EII and determinant
Auger callers no longer use `np.gradient(r)` as integration weights.

The arbitrary-grid Coulomb kernels use bounded powers of radii relative to
local block anchors. They rescale cumulative moments exactly at block
boundaries and retain the same trapezoidal Green function. This avoids the
overflow of separate absolute `r**k` and `r**(-k-1)` factors. Raw `pow_tables`
and its private caches have been removed; use `yk_on_grid`, `grid_slater_rk`
or `yk_pair_dot_batch_on_grid`. Only immutable meshes are cached.

Scalar Slater integrals skip only exactly zero outer-density support,
retaining the last nonzero sample's half interval. EII integrates both spinor
components on the full mesh. The former pointwise relative-tail cutoff had
no general bound on signed matrix-element errors and is no longer applied.

## Shared Coulomb angular coefficients

The coupled two-electron Coulomb coefficients now live in
`atomx.angular.coefficients.coupled_coulomb_direct` and
`coupled_coulomb_exchange`. They use an exact Racah reduction of the former
magnetic-sublevel sums, with the same `reduced_C_jj` (Grant) phase convention. Their
arguments are `(kappa_c, kappa_d, kappa_a, kappa_b, twice_J, rank)`; the
antisymmetrized amplitude remains `D*R_direct - E*R_exchange`.

The private DW helpers `_cc_direct`, `_cc_exchange`, `_ck_m` and `_cg2` have
been removed. Import `ckm` from `atomx.angular.coefficients` and
`clebsch_gordan` from `atomx.angular.clebsch` (the canonical CG kernel;
the former `coefficients.cg` wrapper is gone).
CE, EII, Auger and CI photon callers have been migrated; no forwarding aliases
remain. The radial integrals and physical channel truncations are unchanged.

The shared Wigner implementation also now uses its precomputed log-factorial
table. Long or strongly cancelling Racah sums are evaluated with exact Python
integers before the final floating-point square root. Compiled callers reach
this bounded-cache fallback through Numba object mode; the short, well
conditioned path stays compiled. This fixes cancellation errors at high
angular momentum without changing angular conventions.

## Explicit finite-frequency transverse exchange

`ci_solve(..., include_breit=True, breit_orbital_energies=...)` now replaces
static Breit by the real frequency-dependent transverse operator. Energies
are an explicit `(n, kappa)` mapping in Hartree, fixed during diagonalization.
Without the mapping, the existing static operator is retained.
Public calculations can now request
`Corrections(breit_frequency="orbital_differences")`. Ordinary DHF uses
converged orbital energies; Hamiltonian DF/MCDHF additionally require an
explicit `breit_orbital_energies` map for every active orbital. The same
fields are available on `CorrectionStack`, `DiracFock` and `MCDHF`.
Static Breit remains the default. Finite-frequency exchange replaces the
static contribution, and is evaluated on fixed orbitals. See
[operator, numerical method and FAC evidence](TRANSVERSE_INTERACTION.md).

### Native frozen-core integrals

Bound CI levels and photons retain their original radial representation.
Closed-core contraction is `physics.csf.frozen_core.closed_core_one_body`;
the scattering-specific `build_core_folded_one_body` and
`build_bspline_core_folded_one_body` implementations were removed.
`bspline_bound_integrals` adapts a native result and its virtual coefficients
for the shared Coulomb callback. Low-level `compute_exchange_matrix_cell`
returns all four blocks `(PP, QQ, PQ, QP)`; direct and exchange matrices use
the padded `bsp_data['ns']` layout.

Shooting checkpoints optionally include `_rv_nuc`, the sampled nuclear plus
environment field actually used in SCF. Existing archives still load. A bound
CI reconstruction needing this field rejects archives that lack it and asks
for a new solve, rather than assuming a point nucleus.

### DR photons and continuum results

`ResonanceManifold.dr_levels` now uses resolved final-state CI energies and
finite-frequency E1 stabilization. `ManifoldLevel.radiative_channels` records
the final level, photon energy and rate. Low-level manifolds must supply the
bare-plus-frozen-core `one_body` Hamiltonian for photon calculations; dressed
orbital energies are insufficient. Use `manifold_from_configs` to retain the
native bound representation. `photon_approximation="long_wavelength"` requests
the static diagnostic explicitly.

Regenerate continuum-dependent results produced with the incorrect VINT tail
interval or shared Auger radial cache. The continuum solver now propagates in
the actual Coulomb tail, and Auger integrals are scoped to one continuum
energy/potential. These corrections can substantially change widths; former
slow execution was also a symptom of the wrong extrapolated potential. See
[verification evidence](VERIFICATION.md#dr-final-state-ci-and-continuum-repair-2026-09-30).

`dr_calculate` now shares the native manifold constructor. Local-potential
shooting results can be used directly; B-spline results retain their native
Fock basis. Nonlocal shooting HF virtuals are not yet supported and raise
explicitly, without substituting a local model. Continuum sampling expands
to cover a bound on the CI resonance energies. Resonance labels now contain
the hole/parity group, J and level index, avoiding collisions in thermal
rate breakdowns. The `channels['ground']` entry retains the ground Auger rate.
The computed default complex remains dr33, with the limitations documented
on `dr_calculate`; larger capture spaces are not implied by this migration.


### Exact Coulomb matching for continuum shooting

`solve_dirac_continuum_shooting` now obtains normalization and phase from the
regular/irregular Dirac–Coulomb spinors. The returned phase is relative to
Salvat DCOULF's relativistic Coulomb convention, modulo an overall real wave
sign; consumers using the former nonrelativistic phase convention must update.
`match_phase=False` still returns NaN for phase and uses the same accurate wave
normalization. Two tail points check the matching coefficients when available.
Callers must still choose a box that reaches the physical Coulomb tail; the
check does not certify that truncating a non-Coulomb field is valid.

Regenerate continuum-dependent rates where the former WKB normalization bias
matters. The new `mpmath>=1.3` runtime dependency supplies a private precision
context only when the fast Coulomb asymptotic expansion cannot meet tolerance.
The new numerical module is `numerics/shooting/dirac_coulomb.py`; the existing
`coulomb.py` still supplies Slater integrals. See the
[external normalization evidence](VERIFICATION.md#diraccoulomb-continuum-normalization-2026-09-30).


### Continuum sampling on an observable grid

Both scalar and batch continuum solvers accept `interp_grid`. Requested
samples now use the Dirac integrator's converged power series and adaptive
propagation where subdivision is needed, including the regular origin
solution. Scattering's `_continuum_on_grid` delegates this work to numerics.
Linear interpolation of the wave is removed from these paths.

The batch API's returned radius array now corresponds to the sampled waves
when `interp_grid` is supplied. Output radii must be finite, nondecreasing and
inside the solved box; endpoint clamping is no longer accepted. A failed
batch wave raises a contextual `RuntimeError` with the original exception as
its cause, instead of returning an unexplained `None` value. Global box and
point-count defaults remain unchanged pending broader adaptive validation.
See [screened continuum evidence](VERIFICATION.md#screened-continuum-and-dense-radial-sampling-2026-09-30).

### Shared mean Hartree field for collisions

The collision-specific `_distorting_potential_fn` has been removed. DW
excitation, ionization and the affected Auger paths now use
`potentials.static_field.static_hartree_potential(result, screening="mean")`.
This scales the complete target density by `(N-1)/N`, retains the solved
nuclear field, and interpolates electronic screening separately. It avoids
flattening the nuclear potential below the first B-spline quadrature point.
The one-electron point-nucleus limit is exactly `-Z/r`, without density
integration. Finite-nucleus results require their stored nuclear field.

`PotentialModel.static_hartree(screening="mean")` exposes the same choice.
Mean screening cannot be combined with a named subshell hole. EII now honors
`ContinuumModel.potential`; previously that input could be ignored while
provenance incorrectly reported local exchange. The result reports the
actual screening and residual charge used for Coulomb matching. Explicit
bare fields cannot request local exchange. This correction can change
results that previously supplied an ignored potential model.

DW excitation and ionization also evaluate bound B-spline expansions
directly on their integration meshes. `eval_P` and `eval_Q` accept optional
`radii`; omitting it retains the cached native Gauss-grid evaluation. Explicit
radii must lie inside the spline box. Excitation handles different initial
and final boxes by zero extension outside each box. This removes linear
interpolation and the artificial origin clamp in ionization.

### Full energy-sharing integration and explicit EII tail choice

DW ionization now integrates the full half-sharing interval `[0, (E-I)/2]`.
The reusable numerical rule `sinh_squared_quadrature` applies Gauss-Legendre
integration under `E1 = I*sinh(x)^2`; its Jacobian is included in the weights.
It resolves the binding-energy scale and square-root endpoint behavior
without calling the continuum solver at zero energy. The previous fixed
endpoint cutoffs and fixed gap between its two sampling regions are removed.
The default `n_outgoing_energies` is 16. This is a quadrature order, not a
convergence guarantee; refine it for the target and incident energy.

`eii_cross_section` and `dw_eii_cross_section` now accept
`partial_wave_tail="none"` (default) or `"geometric"`. The default returns
only the computed partial-wave sum. The former EII-specific tail switched
discontinuously to zero when the last terms began rising, which introduced
large false energy-quadrature sensitivity. The optional estimate now uses
the same continuous capped estimator as excitation. It remains approximate;
the result reports the computed sum, applied tail, estimated tail and the
number of nodes outside the estimator's decay regime. Increasing
`kappa_max` remains necessary to establish physical partial-wave convergence.
Despite its historical name, that control bounds orbital `l` and includes
both Dirac partners, through `kappa=-(kappa_max+1)`.

The shared estimator is `numerics.extrapolation.geometric_tail_estimate`.
The two private tail helpers in `scattering.distorted_wave.solver` were
removed, with all internal callers migrated. No import aliases remain.
Continuum cache keys now retain full floating-point energies rather than
rounding to ten decimal places, and repeated incident energies reuse solved
waves. A near-threshold case with a 1.27e-12 Ha outgoing node now remains
positive instead of rounding to zero.

The stored helium-like benchmark raises the computed partial-wave cutoff
from 10 to 16 following removal of its automatic tail. Only native values
and derived ratios were refreshed; external FAC values and agreement gates
remain unchanged. New FAC runs must state their multipole cutoff explicitly:
FAC's `IONMAXK=6` means rank 3 in its doubled-index convention.

## Ionization continuum domain independent of the bound box

`ContinuumModel.shooting(r_max=...)` in distorted-wave ionization may now
exceed the native bound B-spline box. This radius controls both continuum
propagation and the collision integrals. The fixed bound expansion is zero
outside its own box; no extrapolation or new bound-state solve is performed.
Continuum-continuum Coulomb sources span the full collision domain even when
the consuming bound-continuum density has compact support. Only exactly zero
consumer tails are skipped, with both spinor components included.

Results record the bound and continuum radii separately and state that radial
domain convergence requires an independent check. The collision operator is
explicitly recorded as `instantaneous_coulomb`, with `transverse_exchange=False`:
atomic Breit/QED settings do not add a transverse collision amplitude.
Default radii and physical models are unchanged.

## High-angular-momentum continuum dense output

Dense Dirac sampling now evaluates middle-interval power series using a
locally normalized spinor, then restores its amplitude. When the left
endpoint cannot represent the requested relative accuracy and the right
endpoint is larger, it uses the right endpoint of the same potential interval.
This prevents a tiny normalized wave from exhausting the relative residual
test. Exactly zero middle-interval initial data return the homogeneous zero
solution; the regular Frobenius origin boundary remains distinct. Adaptive
integration raises if its step can no longer advance at machine precision.

This fixes the high-partial-wave EII hang reproduced at kappa=-65. It does
not discard weak partial waves or impose an amplitude cutoff. The shared
coupled Coulomb coefficient caches are bounded to 131072 entries each;
eviction changes memoization only, not the calculated coefficients.

## Bounded continuum storage in ionization

EII now evaluates one outgoing energy pair at a time and retains only that
pair and the incident continuum waves. Stored wave arrays no longer grow
with energy-sharing quadrature order or the incident energy grid. Repeated
incident energies reuse the completed shell result, including diagnostics.
Radial solves, exact floating-point energy keys, quadrature and contraction
order are unchanged. Missing waves still raise unless diagnostic mode is
explicitly requested. The unused scalar-wave fallback was removed.

The measured He+ panel reduces peak process memory by 52%, with a roughly
4% runtime cost. This enables larger numerical convergence studies; it is
not a claimed throughput improvement. See the EII streaming section of
`VERIFICATION.md` for the matched comparison and current FAC reference limits.

## Reusable continuum numerical batches

`numerics.shooting.dirac_continuum.DiracContinuumBatch` provides a context
for successive wave batches with common numerical controls:

```python
with DiracContinuumBatch(potential, r_max=50., n_grid=1000,
                        Z_residual=17., interp_grid=radii) as solver:
    grid, waves = solver.solve_many([(-1, 10.), (1, 10.), (-2, 10.)])
    grid, waves = solver.solve_many([(-1, 20.), (1, 20.), (-2, 20.)])
```

The context reuses worker threads and checks the potential each time. It
does not retain returned waves. Closing releases workers, including after
exceptions. The standalone `solve_dirac_continuum_many` entry remains
available for individual batches. EII now uses one context per incident
energy and reuses its outgoing P/Q buffers between sharing nodes.

## Shooting propagation workspace

Public DIR0 and DIR integration calls retain their signatures. Radial sweeps
use internal kernels accepting caller-owned coefficient buffers; the buffers
are reused within that sweep and are not global state. The zero-state path
clears the active coefficients and the turning-point coefficient. Adaptive
grid checks and numerical tolerances remain independent of this allocation
change. See `DF_SOLVER_CONTROLS.md` for live Salvat RADIAL comparisons.

The Breit configuration-average evaluator copies its radial mesh before
marking that copy immutable for geometry reuse. Caller-owned grids remain
mutable and cannot silently stale the integration geometry.

## 2026-10-05 review: explicit operator conventions

Use `atomx.angular.coefficients.reduced_C_jj` for the retained Grant jj phase
and `atomx.scattering.r_matrix.angular_cc.reduced_C_LS` for orbital LS coupling.
The ambiguous `reduced_C` imports are removed. `reduced_cl_j2` retains the separate
GRASP/FAC phase used by one-body transitions; do not substitute a renamed function
without checking its convention.

CG coefficients come from `angular.clebsch.clebsch_gordan`; kappa helpers from
`angular.recoupling`; label formatters from `config.subshell`. Private DW/local
copies and `coefficients.cg` are removed. Use `properties.transitions.rates.
einstein_A_prefactor` only with explicit magnetic-state strength;
`einstein_A_eL` accepts reduced strength and applies initial-state averaging.

`ConvolCache.clear()` is replaced by `release_shared_banks()`. This releases
shared banks process-wide for the grid. The known-wrong private legacy CI builder
has been deleted; use `physics.csf.matrix.build_ci_matrix`.

## 2026-10-05 review: solver metadata and failures

Pass `virtual_orbital_keys` and `core_hole_stationary` to `DFSolver` explicitly.
Drivers use `install_physics_grid` to install a consistent representation.
The numeric `energy_components` mapping no longer contains `se_method` or
solver-policy/nuclear objects; read those from `solver_diagnostics`. Loading
older archives migrates the known legacy metadata keys automatically.

QED reference/interpolation failures propagate. DR screened-hydrogenic fallback
widths require explicit `allow_default_widths=True`. Resonances consumed by CRM
must supply their total radiative rate and statistical weight. Calculation
provenance cannot guess a nuclear model when the solver supplies neither model
nor atomic weight. [Review disposition](code_review_2026_10_05/SOLVER_CONTRACTS.md)
records the retained contracts and limitations.

The optional EXP-T `--axial-parity` build enables `fscc.compute(..., tensor_symmetry="axial")`; C1 remains the default. Complete magnetic multiplets, rank-six irrep range and exact signed forbidden-integral zeros are required. `FSCCResult.model_space_matrix_Ha` restores all directed model blocks in ascending global active order; `effective_matrices` preserves original block keys. The current resource estimate remains uncompressed. Valence/Sigma2 builders now bind their own complete valence sizes and compile serial large-array units separately from OpenMP core workers.

Valence/Sigma2 planners and builders optionally accept
`valence_term2_blas=True, valence_term2_threads=8` (CLI
`--valence-term2-blas --valence-term2-threads 8`). The exact heap-only term2
unit compiles separately with OpenMP; original wrappers/diagrams remain serial.
The default still uses original term2 with one worker. Fresh build manifests
bind the complete source, worker count and per-worker memory; `ValenceSD`
derives those options and rejects unbound or incompatible parallel BLAS.
Use identical options for valence and screening plans. Historical retained
images keep their archived source; rebuild to exercise the current optional
path. See the [current contract](ATOMIC_LINEARIZED_SD.md#fixed-energy-native-ciall-order-valence-equations).

Rebuild the combined valence/Sigma2 image for the homogeneous value-only ABI.
`solve` uses that action for Krylov iterations and homogeneous response values;
`evaluate`, full root checks and the implicit-response source still return or
consume the complete derivative. The optional streamed kernel has one explicit
response policy, with three full-response call sites and one value-only site
bound in the manifest. Original diagrams remain the default; full worker
scratch remains in the preflight for both modes.

Native fixed-energy valence `solve` optionally accepts complete real `initial_numerators` and `initial_numerator_derivative`; they are numerical guesses, always re-certified with the current operator. No saved result automatically acquires current source/state identity. The caller must bind physical input/energy/channel/denominator identity before archive reuse; see [the current contract](ATOMIC_LINEARIZED_SD.md).


Native fixed-energy valence `solve` also accepts an optional
`iteration_callback`. It receives complete read-only uncertified guesses at
GMRES restart boundaries with current source/coordinate hashes; exceptions
propagate. Caller persistence must bind physical and source identity and repeat
all current solution gates. Two callback scratch vectors count toward the
library cap; retained snapshots and disk storage are the caller's responsibility.
Existing calls and iteration-budget semantics are unchanged.

Fresh `ValenceSD` binding reuses a same-instance `CoreSD` residual certificate
only for that owner's most recent unchanged immutable result. The certificate
is private and transient, with full input/operator/provenance/amplitude/scalar
identity; archive reconstruction and changed results still receive the original
complete raw check. No constructor argument, saved format, equation or residual
threshold changes. See the [current certificate gates](atomic_pci_sd_native_2026_10_09/valence_core_certificate_public021/pytest.log).

The proposed native valence GCROT subspace-reuse option was withdrawn after
an independent strong-reference4s raw-gate failure. It is absent from the
current public API and production driver; GMRES remains the certified path.
[Source restoration and failure](atomic_pci_sd_native_2026_10_09/valence_recycling_public022/withdrawal.json)
are preserved separately from the60 passing small live candidate checks.
