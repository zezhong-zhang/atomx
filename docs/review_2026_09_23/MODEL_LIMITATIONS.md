# Accuracy and model limits after remediation

The fixes establish consistency of the implemented operators, normalization,
constraints, state identities and solver dispatch. They do not make every
finite atomic expansion or approximate collision model an exact calculation.

## Structure

- B-spline MCDHF differentiates its correlated Dirac–Coulomb CI functional in
  a **fixed nuclear positive-energy subspace**. This specifies its finite-basis
  no-pair model. Breit/Gaunt and model QED are subsequent corrections.
- The radial target is the ground-state energy, with an equal ensemble at exact
  degeneracy. Optimization is local and does not certify a global minimum across
  initial orbital choices. Compact correlation orbitals labelled “3p” need not
  be spectroscopic excited 3p orbitals. Excited-state observables need suitable
  orbital optimization and configuration-space convergence; frozen reference
  protocols serve a different, explicitly selected purpose.
- Shooting MCDHF retains configuration-average orbital optimization. Shooting
  open-shell HF uses sequential Schmidt projection, an approximation to the
  full constrained equations. The optional per-block orbital sets are still a
  scaffold, not a fully state-specific EOL optimization.
- `converged` concerns the implemented iteration and stated stationarity
  tolerances. It does **not** certify radial-grid, box, angular or configuration
  convergence. The point-nucleus U91+ grid probe gives -4857.606126 Ha on the
  coarse automatic grid and -4861.189024 Ha with `grid_hi=0.005`,
  `grid_he=0.1`, `r_max=30`. Both scalar and list-configuration paths now
  honor those controls. High-Z precision work needs further refinement checks.
- Gaunt is the implemented magnetic part of Breit; frequency-independent
  retardation is absent. QED corrections are model approximations. The result's
  correction provenance distinguishes the applied operator from comparison-only
  scalar estimates.

## Continuum and kinetics

- Default RR uses a frozen residual Hartree core. Nonlocal exchange and core
  relaxation are absent. The explicit FAC-like option uses local exchange;
  agreement with FAC is an external model comparison, not an algebraic identity.
- Rydberg RR uses a common local bound/continuum potential and statistical
  Pauli vacancies. It is an independent-particle capture model.
- A finite-box B-spline continuum is a pseudostate representation. Caching
  reuses precisely the same eigensystem and does not remove discretization
  error. Threshold and resonance calculations require box/mesh convergence.
- Shooting continuum normalization still uses an exterior WKB amplitude fit.
  RR's displayed Milne conversion uses nonrelativistic electron momentum;
  very high electron energies require a relativistic kinematic treatment.
  These remain declared approximations beyond the corrected statistical factor.
- CRM's analytic hydrogenic/Kramers rate helpers are approximate atomic rates.
  Spontaneous recombination is now integrated against Maxwellian electrons;
  stimulated recombination requires an explicitly supplied radiation field.

## Validation scope

Independent checks cover CI operator identities, radial Gram matrices,
Hellmann–Feynman gradients, hydrogenic roots, constant-potential shifts,
Coulomb functions, scattering boundary conditions, reciprocity and Milne
integrals. Self snapshots only detect changes; external FAC/DBSR_HF/Bernitt
references remain separate. No claim is made that all molecular, MBPT,
open-shell scattering or high-Z QED implementations have been exhaustively
validated. Existing expected failures remain visible in the test results.
