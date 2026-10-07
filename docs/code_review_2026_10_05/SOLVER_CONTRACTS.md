# Explicit solver, result and failure contracts

`DFSolver` declares `virtual_orbital_keys` and `core_hole_stationary` in its
constructor. Callers supply these at construction; direct users get the same
contract. Resolved calculation specifications retain both controls. The shared
`install_physics_grid` installs knots, basis, dimensions, box and derived
matrices/cache together; callers no longer mirror mutations of solver internals.

`SCFResult`, `VariationalCSFResult` and `ShootingResult` share native radial-state
access. Variational provenance and diagnostics use the declared
`solver_diagnostics` mapping. Numeric `energy_components` contains energy shifts;
`se_method`, policy strings and nuclear metadata belong in diagnostics. Old HDF5
component payloads migrate those known metadata keys on load. Explicit result
fields record calculation identity and the nuclear potential where consumed.

The correction seams now preserve the selected model when computation fails:

- Shabaev's Dirac reference failure raises with its cause and is not cached;
  it cannot substitute a nonrelativistic reference.
- A failed Mohr PCHIP interpolation propagates its error.
- DR width defaults require `allow_default_widths=True`; failed structure builds
  propagate. Statistical weights and CRM total radiative rates are required.
- A solver with neither atomic weight nor nuclear model cannot record a guessed
  point nucleus.

The provider ERI validation performs one finite scan and retains its resource
preflight. Complex inputs are checked before real projection, so a nonfinite
imaginary part is still rejected.

## Scope retained deliberately

`Atom.solve()` preserves its existing default-facade / explicit-method-result
return contract; `Atom.result` is the public solved-result accessor. This review
adds no breaking facade redesign. The deleted CI kernel also removed the mirrored
SCF functional. The coherent SCF/orbital/CI drivers remain where they belong;
separating the B-spline functional and iteration remains the architectural
follow-up already documented. Method length alone does not justify splitting
physical state across new abstractions.

## Documentation impact and evidence

Updated: constructor/result/failure docstrings, [migration notes](../MIGRATION.md)
and [architecture contracts](../ARCHITECTURE_DESIGN.md). Retained: capabilities
and provider ADR, since numerical operators and provider shape/budget contracts
are preserved. Archive loading, correction injection, live solver/grid controls,
failure injection, provider checks and public DR paths are exercised; see
[validation](VALIDATION.md). Self-consistency does not establish experimental
accuracy or resolve the Ne satellite discrepancy.
