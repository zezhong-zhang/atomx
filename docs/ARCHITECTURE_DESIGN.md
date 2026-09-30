# Atomx architecture

Methods select the physical problem. Numerical engines solve it. Shared result
objects carry native radial data, physical identity and convergence evidence.

## Ownership

| Area | Responsibility |
|---|---|
| `atom.py`, `atom_inputs.py`, `calculation_spec.py` | Public requests, validation, dispatch and resolved calculation identity |
| `config/`, `state_protocols.py` | Configurations, shared subshells and state contracts |
| `physics/`, `angular/`, `potentials/` | Equations, field/energy expressions, angular coefficients and nuclear/environment fields |
| `methods/` | Physical method selection, CI/state assembly and correlated objectives |
| `numerics/` | Radial geometry, integration, matrix assembly, eigensolvers, shooting, constraints and optimization |
| `results.py`, `radial_state.py` | Result views, native metric and radial-state access |
| `corrections/` | Capability checks, physical corrections and operator inclusion accounting |
| `io/` | Versioned result persistence |
| `properties/`, `scattering/` | Observables and process calculations consuming solved states |
| `reference_data/`, `workflows/` | Source parsing, matching and comparison services |
| `viz/`, `spectrum/`, `plasma/` | Rendering, spectral synthesis and population/rate models |
| `scripts/` | Supported command-line workflows |

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
