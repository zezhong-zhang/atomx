# Architecture capabilities and limits

This table describes implemented interfaces and their verification scope. An
implemented interface does not establish experimental accuracy for every atom.

| Area | Implemented contract | Remaining limit |
|---|---|---|
| HF–Slater and Breit–Pauli | Explicit method names; grid, nuclear and screening controls forwarded; unsupported coupling rejected | Breit–Pauli uses hydrogenic perturbative shifts, not a complete many-electron Hamiltonian |
| Nonlocal shooting DHF | Native exchange, coupled occupied constraints, shared nuclear/environment field | Adaptive nonlocal recovery is available; grid, box and state-identity convergence remain separate requirements |
| Correlated shooting | CI-gradient orbital optimization in a declared positive-energy space | Finite-space stationarity is not projector, box or correlation-space convergence |
| CI level selection and preparation | `CILevel` resolves symmetry-block energy roots; `CILevelPopulation` specifies total unpolarized populations in supported coherent drivers | Sorted roots do not track character across bases; a unified `atom.state(...)` view remains a proposal |
| Common-orbital state averages | `CIStateAverage` and `CIChargeStateAverage` reach B-spline MCDHF through `Correlation(state_average=...)`; joint-charge objectives use complete retained-orbital daughter sectors | Joint-charge optimization requires Dirac–Coulomb; independent charge-specific orbitals and dynamic relaxation are unfinished |
| CI identity and radial reuse | Lower-level functionals accept explicit frozen-reference `CIBasisTracking`; `CIRadialSnapshot` binds shared radial evaluators to owned data | Ambiguous/lost identity rejects trials; neither sorted roots nor cache reuse certifies convergence |
| Common-orbital CI loss channels | Two-/many-electron providers construct signed Dyson PI and full `h_D + V` Auger in complete residual Fock sectors over retained orbitals | Projected frozen-ion direct Hartree Dirac DW; no nonlocal continuum exchange or coupled Fano channels |
| XFEL dynamics | Native rate tables, population equations, Monte Carlo cascades, bound and joint-sector coherent drivers, transient configuration observables | `MCDHFCoherentModel` is bound-only; joint CI drivers enforce pulse/space/reservoir gates, and conditional reservoir bounds do not certify microscopic bath physics |
| Complex scaling and target spectra | Point-nuclear complex Dirac/DKB, Coulomb/static-Breit CI, and target eigensolvers in lower-level modules using the c-product | No QED in this CI entry point or automatic `Atom.solve()` width mode; physical-pole and representation convergence require separate validation |
| Transition resolution | Explicit CI reference and explicit UTA fallback policy | Configuration averages cannot resolve individual multiplet lines |
| Correction inclusion | Explicit operator stages; duplicate addition rejected before evaluation; diagnostic reevaluation allowed | Historical results without inclusion records cannot be reconstructed from guesses |
| Result conversion | Conversion requires an explicitly solved, converged result | The PNC/EDM entry points still require their B-spline payload |
| Native radial access | Shared state contracts below scattering; native quadrature workspace exposed | Complete result immutability and correlated-state serialization remain unfinished |
| Persistence | Versioned SCF and native shooting metadata, orbitals, calculation specification and correction accounting | Native checkpoints do not imply complete correlated-state archival |
| NIST comparison | Packaged configuration matching, centroid evaluation and read-only native campaign comparison | Ingestion, production queue orchestration and HTTP routing still partly live in scripts |
| Providers | Native adapter and provider contracts; optional PySCF adapter | No claim of implemented ABACUS/DeePTB adapters or every proposal in ADR-0001 |
| R-matrix | Single-channel and multichannel drivers are implemented | Supported target/channel models are those documented by the actual drivers |
| Coupled cluster | Explicit `NotImplementedError` entry point | Remains a scaffold |
| RPA / TDHF | Closed-shell particle-hole response with a metric-orthogonal V^N virtual basis | Finite basis and partial-wave convergence required |
| Visualization | Renderers loaded on demand; core imports do not load Matplotlib, Plotly or Manim | Matplotlib remains a declared base dependency for installation compatibility |

## Verification map

- API model fidelity: `test_architecture_scientific_fidelity.py`.
- Import and state boundaries: `test_architecture_boundaries.py`,
  `test_numerical_design_contracts.py`, `test_protocols.py`.
- Physical field construction: `test_scf_potential.py`,
  `test_shooting_guerra.py`, `test_shooting_core_hole_stationarity.py`.
- Correlated derivatives: `test_shooting_correlated_grid.py`.
- CI selectors, objectives and identity: `test_ci_level_selection.py`,
  `test_ci_state_average.py`, `test_ci_charge_state_average.py`,
  `test_ci_basis_overlap.py`, `test_ci_basis_tracking.py`, `test_ci_radial_snapshot.py`.
- CI continuum and coherent dynamics: `test_xfel_ci_continuum.py`,
  `test_xfel_multielectron_ci.py`, `test_xfel_mcdhf_coherent.py`,
  `test_xfel_radiative_reservoir.py`, `test_xfel_residual_certificate.py`.
- Complex representations: `test_complex_scaled_dirac.py`,
  `test_complex_radial_ci.py`, `test_complex_breit.py`, `test_complex_target_spectrum.py`.
- Correction accounting: `test_correction_application_contract.py`,
  `test_corrections_stack.py`, `test_scfresult_save_load.py`.
- Source matching: `test_configuration_centroids_nist.py`,
  `test_nist_comparison_workflow.py`, `test_full_dhf_nist.py`.
- Provider and scaffold contracts: `test_native_provider.py`,
  `test_providers_contract.py`, `test_scaffolds.py`.

See [the repair log](archive/2026-09/ARCHITECTURE_REPAIRS_2026-09-29.md) for the
earlier architecture checks, and the [agent reference index](AGENT_REFERENCE.md)
for CI/XFEL measurements and subsequent corrections. The map lists relevant
tests, not a claim that the full suite has been run.
