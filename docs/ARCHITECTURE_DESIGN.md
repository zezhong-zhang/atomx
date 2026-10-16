# Atomx architecture

Methods select the physical problem. Numerical engines solve it. Shared result
objects carry native radial data, physical identity and convergence evidence.

The external FSCC interface source-binds the denominator engine when translating
raw fixed-point updates into retained-equation bounds. Known legacy numerator
skips contribute an explicit residual floor. The optional builder patch for
exact nonzero division changes only numerical division, with its own source
capability; it adds no Hamiltonian or sector. See [the contract](ATOMIC_FSCC.md#numerator-division-source-contract).

`numerics.axial_eri.AxialPairLayout` owns only exact coordinate packing and
signed/conjugated random access for supplied integer M/parity labels. Physical
selection and full multiplet/Fock/ERI validation remain in `methods.fscc`.
Its optional compact backend integral storage preserves every allowed value;
versioned sidecars, source binding and explicit replay/decompression budgets
belong to the FSCC interface. It does not change amplitude equations or default
packed storage. See [the storage contract](ATOMIC_FSCC.md).

`numerics.fermion.canonical_body_matrix` projects supplied canonical zero-
through-three-body coefficients on explicit finite fixed-N masks. Exact
spectator sums retain complex directed entries; work/output/index budgets
are admitted before projection. It constructs no Hamiltonian approximation,
orbital reference or physical state. The [core-hole research controls](EUV_SOURCE_MODEL.md#finite-second-order-core-hole-operator-controls)
use it for a bounded native projection. The separate `numerics.canonical_sparse`
engine applies the same supplied coefficients through occupied annihilator
subsets. Default colexicographic row indices retain the full fixed-N
combinatorial map; explicit `row_index="hash"` stores only declared masks
with exact uint64 keys, collision comparisons and load at most1/2. Row and
accumulation order stay identical. An exact prefix-population support derived
from every declared row rejects only impossible row hits before full-key
lookup; it introduces no physical screen. It counts every reachable directed
coordinate before admitting CSC output, including SciPy int64 index
promotion above the int32 coordinate limit, retains arbitrarily weak words
and does no Hermitian averaging. Physical coefficient generation and complete
Sn CI+MBPT qualification remain with the consuming research workflow.

`MagneticCSFProjectionCache.project_active` owns the existing filled-shell
contraction and signed exterior-product conversion into an explicit active
spinor order. Native spinor CI and supplied effective operators share this
angular mapping; neither needs a fabricated SCF/reference object to transform
a CSF basis. Consumers still preflight construction memory and bind all states.

## Ownership

| Area | Responsibility |
|---|---|
| `atom.py`, `atom_inputs.py`, `calculation_spec.py` | Public requests, validation, dispatch and resolved calculation identity |
| `config/`, `state_protocols.py` | Configurations, shared subshells and radial state protocols |
| `physics/`, `angular/`, `potentials/` | Equations, field/energy expressions, CI level/ensemble identity, radial snapshots, angular coefficients and nuclear/environment fields |
| `methods/` | Physical method selection, CI/state assembly and correlated objectives |
| `numerics/` | Radial geometry, integration, matrix assembly, eigensolvers, shooting, constraints and optimization |
| `results.py`, `radial_state.py` | Result views, native metric and radial-state access |
| `corrections/` | Capability checks, physical corrections and operator inclusion accounting |
| `io/` | Versioned result persistence |
| `properties/`, `scattering/` | Observables and process calculations consuming solved states |
| `xfel/` | Pulse-driven populations, Monte Carlo cascades, coherent magnetic-state dynamics and transient observables |
| `reference_data/`, `workflows/` | Source parsing, matching and comparison services |
| `viz/`, `spectrum/`, `plasma/` | Rendering, spectral synthesis and population/rate models |
| `scripts/` | Supported command-line workflows |

The [reduced EUV source](EUV_SOURCE_MODEL.md) composes shared CRM physical
rates (dense/CSR), `spectrum.slab_transfer` angular transport and
`plasma.laser_absorption` deposition in `plasma.euv_source`. The native
`workflows.tin_source` binds input identity; `tools/run_tin_source.py` is its
explicit local CLI. Fixed-charge LTE and small supplied-rate non-LTE modes
share one energy ledger. The separate `charge_mixture`/`mixture_radiation`
composition supplies finite Saha EOS and explicit LTE continuum data to
`lagrangian_source`: conservative smooth-flow cells, layered radiation, Drude
waves and conduction. `reference_data.euv_experiment` owns measured-count
calibration. The finite `nonlte_source` composes the same CRM with explicit
partial-Auger/inverse-capture and shared Milne photon moments, retaining electron
number and ionization energy in an optically thin isochoric closure. It rejects
effective synthetic satellites as physical resonances. No atomic solver owns
hydrodynamic policy; complete real Sn kinetic data, nonLTE spatial coupling,
neutral/solid EOS and shock/ablation remain absent.

`scattering.photoionization.ExplicitCIPhotoionization` consumes explicitly supplied
adjacent blocked CI sectors, including closed spectators, instead of generating
a full residual Fock cascade. It binds common DC/nuclear/radial identity and
all roots by default, with explicit degenerate-cluster-safe root selection;
shared signed Dyson removal feeds final-level projected Hartree
continuum E1 PI and direct-node Milne rate/photon moments. Optional
`numerics.fock_removal` kernels operate on supplied finite bitsets/coefficient
arrays; angular J-support meaning stays in the Dyson consumer. Explicit
`dipole_mode="reduced-dyson"` reconstructs the full magnetic trace from signed
highest pure-J components without changing true g or radial interference;
the complete-M default rejects incomplete initial/final partners. The complete4d Sn
pilot supplies partial rates only, with distinct radial/continuum/quadrature
checks; it is not data for the full n=4 source or experimental CE acceptance.
`scattering.autoionization.ci.ExplicitCIAutoionization` reuses that snapshot and
actual thresholds for full signed h_D+Coulomb widths. Complete magnetic sums
and exact coupled scalar reduction are independently checked. A highest-M
scalar density and shared streamed root projection avoid redundant lowering;
`numerics.continuum_actions` owns the bounded compiled signed Fock action,
including exact common-core contraction. Selection and scalar representation
never screen weak amplitudes. The active full-n4 Sn resonance campaign has
balanced daughter CI but does not yet certify complete rates or experiment;
The separate `numerics.acyclic_rates` owns bounded positive topological
residence/edge/terminal integration and an explicit positive finite-time
backward-Euler triangular consumer. Caller-owned time grids require independent
refinement; callers also own physical channels and isolation. Neither
eventual nor finite-time one-capture flow supplies complete nonLTE kinetics.
See the [owning contract](EUV_SOURCE_MODEL.md#finite-capture-time-window).

`scattering.electron_ionization.ci.ExplicitCIElectronIonization` reuses the
adjacent CI snapshot for coherent removal, selected final-ion fields and
actual thresholds. Immutable explicit radial-channel inputs feed the same
DW driver as its SCF adapter; the generic driver receives fields and radial
arrays, never a fabricated solved state. Complete unpolarized magnetic-trace
reduction preserves same-kappa radial interference and all supplied Pauli
projectors. The [source guide](EUV_SOURCE_MODEL.md#explicit-coherent-ci-direct-electron-impact-ionization)
records the spectator-ion approximation, preserved initial partial-wave
failures and later passed numerical controls for three partial cross sections.
Direct-node Maxwellian averaging evaluates every quadrature node; thermal
observable convergence and complete physical channel coverage remain separate.

The explicit `methods.csf.blocked_functional.BlockedCSFOrbitalFunctional`
selects a fixed symmetry/root orbital ensemble on complete DC H blocks.
Owned selected-root trials carry the complete-H residual, point binding and
objective identity; they are not complete plasma/continuum spectra.
`physics.csf.spinor_density` maps exact finite directed RDMs to shared radial
keys, including closed-core radial derivatives and phases. The compiled
`numerics.fock_density` owns only supplied finite masks/coefficient algebra
and numerical resource admission. No physical state or orbital policy lives
there. `angular.projection_cache.MagneticCSFProjectionCache` reuses bounded
canonical signed bases for H and RDM with fresh root/radial data; complete
frozen-core phase and row-order conversion remains in the physical builder.
The blocked functional owns its cache and admits its budget alongside matrix
scratch. Complete blocked radial weights use canonical key order across
process hash seeds. An opt-in ground-only variational comparison composes the
existing full finite-density/constraint math; it still requires both complete
CI solutions and does not supply an FP error bound or stationarity result.
The existing constrained optimizer retains projector, S metric and
stationarity gates. The [state guide](ATOMIC_STATE_DESIGN.md#explicit-complete-block-orbital-objective-15-october-2026)
records limits; the current160-check combined panel passes with3 optional
external EXP-T skips, while the actual full8543-CSF Sn DC orbital objective now passes its original
stationarity gate; the subsequent static-DCB optical experiment still fails.

## Shooting ownership

Shooting is a numerical technique, not a theory level. Its implementations live
in `numerics/shooting/`:

- `local.py`: local-potential SCF and frozen-orbital solves.
- `hartree_fock.py`: nonlocal HF driver and numerical occupied rotations.
- `scf.py`, `local_scf.py`: iteration, mixing and convergence.
- `dirac_*`, `schrodinger_*`: boundary-value solves and radial propagation.
- `electron_space.py`: bound/continuum electron-space construction and controls.
- `orbital_optimization.py`, `preconditioner.py`: constrained optimization and Green response.

The HF driver composes `physics.hartree_fock.RadialHartreeFock`, which supplies
pair moments, angular weights, Fock action, energy and allowed rotation policy.
The engine performs the rotation search. `physics/slater.py` supplies local
exchange and energy expressions. Neither physical operator launches an SCF solve.

DF and MCDHF share state assembly in `methods/csf/solver.py`, with their
configuration-space policies defined by their method entry points. State
energy and gradient assembly remains in `methods/csf/grid_functional.py`,
composing the physical contractions in `physics/csf`. The drivers call
numerical electron-space and optimization routines.
`ShootingResult` belongs to `results.py`; loading it requires no method
implementation.

## Dependencies

1. Numerical modules do not import methods, correction policy, properties,
   scattering, the Atom facade, workflows or web code. The architecture test
   checks all numerical modules, including relative and function-local imports.
2. Physical operators can consume low-level representation-specific integration
   kernels. Those kernels cannot import the drivers or physical policy using them.
3. Methods pass explicit operators, arrays or functional callbacks to engines.
   Callbacks operate on complete arrays or blocks, not Python calls per radial point.
4. Results and persistence do not start calculations. Converting an unsolved or
   unconverged object into a solved result fails explicitly.
5. Rendering imports are lazy. Importing the core API does not load plotting or
   optional electronic-structure backends.

## Scientific contracts

Single-configuration DHF defaults to explicit `averaging="hamiltonian"`
(Hamiltonian-derived intermediate coupling within one NR configuration).
Ground-state occupations follow Guerra Eq. 10, while orbital variation retains
the full coherent state energy. Fixed statistical averages are a separate model.
Dynamical correlation requires an explicit configuration expansion. Hamiltonian
DF follows direct stationary radial equations on the self-consistent electron
branch for both engines; it does not construct a fixed continuum basis.
MCDHF retains its explicit variational electron-space controls. See
[DF solver controls](DF_SOLVER_CONTROLS.md) for the argument/mechanism table.

The shared method driver and state functionals live in `methods/csf`; DF and
MCDHF entry points define the configuration space. CSF definitions belong to
`config/csf.py`, angular Hamiltonian tables to `angular/csf_hamiltonian.py`,
and matrix elements/density contractions to `physics/csf`. Constrained radial
optimization algorithms belong to `numerics/{bspline,shooting}`; occupied
rotation algebra is shared in `numerics/orbital_rotations.py`. Physical
operators must not depend on either DF or MCDHF method entry points.

### CI state identity and orbital objectives

Configuration space, orbital optimization, observable state, dynamical
preparation, and energy zero are separate contracts. `physics/csf/levels.py`
owns `CILevel` and `CILevelPopulation`; `state_average.py` owns fixed single-
and multi-charge objectives. Public inputs pass these objectives through
`Correlation(state_average=...)` to the B-spline MCDHF solver. The joint-charge
path currently requires Dirac–Coulomb and common orthonormal orbitals.

`methods/csf/variational.py` and `charge_variational.py` combine CI energies,
signed density contractions, and radial gradients. They can explicitly bind
the lower-level `CIBasisTracking` contract to a frozen physical reference;
ambiguous or lost subspace identity rejects a trial. Sorted public roots alone
do not supply this tracking. `CIRadialSnapshot` shares immutable radial data and
evaluators within a declared Hamiltonian snapshot.

The [atomic-state design](ATOMIC_STATE_DESIGN.md) distinguishes these implemented
pieces from the unfinished unified state-view and independent-orbital interfaces.

Density storage is a separate explicit choice from eigensolver storage.
`CIRadialDensity` defaults to cached angular tables; its streamed real DC
bilinears share the Hamiltonian's signed pair/core algebra and sparse ordered
CSF projection without retaining the angular table. Batched columns share a
complete connection traversal. `CSFOrbitalFunctional(density_storage='streamed')`
only selects this density representation. Raw bilinears carry no inferred
eigenstate, normalization, Hamiltonian or radial-snapshot ownership.

Analytic directional response is also explicit. `numerics/variational_response.py`
checks a complete real finite eigensystem and forms an isolated-state first
coefficient response, energy second derivative and the CI-response portion of
the third Taylor coefficient. It rejects partial/target-degenerate spectra;
it does not supply the fixed-vector H''' term or a physical ownership object.
Its polar-path derivatives include the finite input Gram and constraint
acceleration. `physics/csf/radial_increment.py` separately binds actual first/
second radial jets to native ket-normalized one-body and full four-slot real
DC cell integrals. Callers retain metric, operator, state and parameter identity.
These ingredients do not alter the optimizer or provide a matrix-free response
for the large selected-root spaces. See the
[source-bound analytic qualification](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision012/README.md).

`methods/ci_blocks.py` is an explicit complete-real-DC J/parity storage
consumer, with all roots and full H/Gram gates. Its streamed builder shares
the ordinary radial snapshot. Its optional `physics/csf/spinor_ci.py` builder
consumes the shared native `h_eff/g/E_frozen` export and compiled numerical
Slater–Condon values, without consuming the exporter's reference Fock or
assuming its reference particle count is the target CI count. Exact common
closed-core contraction retains opened active holes. It explicitly maps
the sorted magnetic layout and fermionic phase into the integral layout;
real active layouts are limited to63 spinors. The [tin consumer](TIN_EUV_BASELINE.md)
selects the physical space and emission endpoints; this lower-level path
does not add MBPT or Atom facade dispatch.

Native spin-orbital tensor CC/EA follows the same dependency boundary.
`methods/tensor_coupled_cluster.py` owns native-integral consumption, complete
SDT/attachment coordinates, resource admission and nonlinear solving.
`physics/csf/ccsdt_terms.py` owns attributed spin-orbital equations;
`numerics/tensor_recipes.py` applies caller-supplied signed contractions and
shape-only paths, including optional indexed accumulation that preserves nested
free-index subtraction. Methods choose the complete canonical coordinates and
exact M/parity admission; numerics does not screen physical terms. Fock inverse
preconditioning and owned warm starts remain method controls. This lower-level
consumer enumerates no full particle
sector. Dense orbital tensors and projected EA matrices retain separate
limits; projected roots do not supply full-H states or dressed observables.
The [independent Yb qualification](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_tensor_ccsdt001/README.md)
constructs complete target-sector states only in its external diagnostic code.
The [complete4f continuation](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_tensor_4f002/README.md) uses only polynomial retained coordinates
for18 correlated electrons. Its diagnostic signed highest-weight projection and
matrix-free Davidson remain outside the production API. All original equations
and magnetic partners are rechecked; no full-H state is inferred.

### Channels and time evolution

`scattering/ci_continuum.py` and `multielectron_ci.py` construct common-orbital
ionic sectors, signed Dyson photoionization, and full one-/two-body Auger
amplitudes. Complete residual sectors mean complete Fock spaces over the
retained orbital pool. Continuum waves use the frozen final-ion direct Hartree
Dirac field with retained-bound-space projection and energy normalization;
nonlocal continuum exchange and coupled Fano channels remain outside this model.

XFEL drivers consume rate graphs or coherent magnetic-state operators and
prepared channels. The bound MCDHF driver explicitly excludes ionization;
the two-/many-electron drivers add finite-space CI loss channels with separate
reservoir applicability gates. Reusing a snapshot changes cost, not the
preparation, orbital objective, or physical approximation.

Complex-scaled Dirac and CI kernels are lower-level representations with a
c-product and explicit projectors. Target eigensolvers do not themselves
certify resonance poles. See [capabilities](ARCHITECTURE_CAPABILITIES.md) and
[XFEL design/evidence](XFEL_GAP_ANALYSIS.md) for boundaries and verification.

### Shared invariants

- Keep configuration averages, term/level energies, orbital energies and
  transition energies distinct. Preserve occupations, coupling and state identity.
- Keep each representation's native quadrature and overlap metric. Do not
  silently replace a shooting or B-spline metric with trapezoidal sampling.
- Record the nuclear model, isotope/mass, Hamiltonian, electron projector,
  numerical controls and correction inclusion stages with the result.
- Do not add a correction already included in orbital or CI optimization.
  Diagnostic reevaluation must not change the finalized energy.
- Treat SCF convergence, orbital identity, radial refinement and correlation-space
  convergence as separate checks. None alone establishes experimental accuracy.
- Unsupported physics raises explicitly. A requested fallback records the
  original request, actual model and reason.
- NIST source rows and uncertainty fields remain source data. Matching and
  comparison must not rewrite them to improve a score.

## Change and transition policy

Change a boundary when ownership or behavior is demonstrably wrong. Keep
coherent modules intact regardless of size. Prefer explicit data and direct
calls; add abstractions when actual consumers need them.

Complete each migration: update callers, tests and current documentation, remove
the obsolete entry point, and document breaking imports. Keep saved-data
compatibility independent of Python module names. Existing aliases outside the
migrated area do not justify unrelated renames. See [migration notes](MIGRATION.md).

Validate structural changes with dependency checks and matched physical controls.
Keep equation changes and numerical algorithm changes reviewable separately.

## Remaining work

- Complete the B-spline functional/iteration separation in the large DHF module.
- Resolve numerical requests and defaults consistently at one public boundary.
- Complete correlated-state archival and result mutability/cache-lifetime contracts.
- Package remaining reusable script services as their supported use cases require.

These are identified boundaries, not claims of completed implementations. See
[capabilities](ARCHITECTURE_CAPABILITIES.md) and [historical design reviews](archive/README.md).

## Reviewed shared contracts (2026-10-05)

Angular convention names distinguish Grant jj (`reduced_C_jj`), orbital LS
(`reduced_C_LS`) and GRASP/FAC one-body (`reduced_cl_j2`) phases. Callers reuse
shared CG, fermion signs, kappa/label helpers, electromagnetic angular/radial
operators and rate normalization. Native result classes share radial-state
access; solver diagnostics carry policy/provenance separately from numeric
energy components, including across persistence.

`DFSolver.install_physics_grid` owns installation of a consistent basis and its
derived matrices/cache. Virtual-orbital and core-hole stationarity policies are
declared controls and enter the resolved calculation identity. CI audit reuse
requires unchanged orbital and radial content; successful validation is the
only event that records a reusable metric pass. See the
[review disposition](code_review_2026_10_05/SOLVER_CONTRACTS.md).


`physics.csf.one_body_composition` composes a complete point-bound scalar
operator with existing packed or native on-demand spinor integrals and normal-ordering ledgers;
it generates no physical correction. Operator producers retain approximation
and source ownership, including the separately qualified native QEDMOD matrix.
`plasma.rates.maxwellian_rate_direct` owns physical thermal averaging of a
complete callable cross section on explicit shifted quadrature nodes. CI EII
owns state/threshold/field selection and records those nodes; generic numerical
rules do not decide physical coverage or experimental acceptance.
