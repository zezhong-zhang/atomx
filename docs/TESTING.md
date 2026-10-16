# Testing Atomx

The default suite targets about **three minutes** and contains a reviewed set of
essential tests listed in `pyproject.toml` under `testpaths`. Plain `pytest -q`
and GitHub Actions use the same selection. An explicit path overrides that
selection, so run affected subsystem tests as well when changing their code.
Do not use `pytest tests` as the routine command: it collects the extended suite.

Use the existing `.venv`. For a fresh clone:

```bash
uv sync --locked --extra dev --python 3.10
.venv/bin/pre-commit install
export PYTHONPATH=src:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
.venv/bin/python -m pytest -q --durations=15
```

The essential suite does not require optional rendering backends or their Cairo
and Pango system libraries. Development dependencies include the units/PySCF
extras and `pytest-timeout`; the lock remains authoritative. Serial BLAS,
OpenMP and Numba avoid oversubscription and reduce scheduling variability.

## GitHub Actions and local equivalents

| Workflow | Trigger | Gates |
|---|---|---|
| [Test](../.github/workflows/test.yml) | Main pushes and pull requests | Essential suite on Python 3.10 and 3.13, including two fresh-process determinism repeats |
| [Lint](../.github/workflows/lint.yml) | Main pushes and pull requests | Ruff 0.15.6, repository documentation links and project-index freshness |

The recorded 2026-10-10 hosted run passes **452 tests** per Python version:
job time including setup/cleanup is **3m13s (3.10)** and **2m48s (3.13)**.
These are source-bound observations; [logs and identities](ci_repair_2026_10_10/README.md)
record the conditions. A local warm-cache default run passed in 37.83 s.

The former nightly slow/book/bounds/full-determinism workflow was removed on
2026-10-10 at the user's request. The three-minute goal is a turnaround target,
not a scientific tolerance or a hardware-independent timing guarantee; each CI
job has a five-minute safety timeout including environment setup.

```bash
# Focused checks explicitly select files or nodes outside the default list.
.venv/bin/python -m pytest -q tests/test_sternheimer_edm.py -m 'not slow'
.venv/bin/python -m pytest -q tests/test_numerical_bounds.py
.venv/bin/python -m ruff check --no-cache src tests scripts benchmarks
.venv/bin/python scripts/check_documentation.py
.venv/bin/python scripts/build_project_index.py --check
```

The explicit bounds diagnostic uses the same C/Ne/Be inputs with
`NUMBA_BOUNDSCHECK=1`. It forces a fresh compilation and is outside routine CI.
The Cf addition and four/twelve-repeat campaigns have been removed. Two fresh
processes still compare energies, every orbital energy and signed wave bytes.
They reuse compiled kernels from the ordinary suite; only the optional bounds
diagnostic requires an isolated compilation cache.

The link check covers root and top-level documentation entry points and requires
tracked repository targets. It excludes external URLs, heading anchors, fenced
examples and archived report bodies. It cannot judge scientific claims; review
the [owning guide](DEVELOPMENT.md#code-to-documentation-entry-points) with each change.
Historical selected-panel counts live in [validation history](VALIDATION_HISTORY.md).

If Numba raises `ReferenceError: underlying object has vanished` while loading
or saving generated cache entries, retain the failed log and replay with a
fresh `NUMBA_CACHE_DIR` before attributing the failure to physics. The
[16 October cleanup record](repository_cleanup_2026_10_16/README.md) includes
the failed run, unchanged-input cache replay and affected regression results.
`test_mixed_ci_finite_increment.py` retains its sealed historical thirteen-point
campaign and original derivative gate. Its live fixture skips without
`ATOMX_MIXED_INCREMENT_OUTPUT`; the standalone response checks still run.
Executing the campaign requires its source-bound resource driver and local
archived assets, and is not part of an ordinary affected-test run.

## Essential coverage and removed campaigns

The explicit source panel is `test_euv_source_model.py` and
`test_tin_source_loader.py`, with `test_charge_mixture_transport.py` and
`test_euv_experiment_contract.py` for the spatial multi-charge increment, and
`test_nonlte_source.py` for18 explicit resonance/kinetic energy controls. It checks analytic CR/detailed balance, sparse
conservation, exact angular transport and blackbody limits, laser energy
partition, excitation heat capacity, adiabatic expansion, coupled non-LTE
excitation/radiation exchange, preheat control and native input admission.
Run the CRM/plasma/spectrum panels after shared generator changes. The
[source guide](EUV_SOURCE_MODEL.md) distinguishes these gates from experimental
CE and records missing resonance-safe atomic/rate coverage and solid-target
shock/ablation physics. The current affected panel passes191 tests; independent
Saha/EOS derivatives, Milne reciprocity, layered heating/Planck baths, critical
wave optics and explicit absolute calibration are included. Manufactured spatial
controls and conservative ledgers are not experimental acceptance. The finite
nonLTE panel independently checks partial-channel phase volume, emergent DR
branching, RR photon moments and captured-electron energy, EII/Auger electron
count, Saha/energy relaxation, global energy gauge, source ownership and rejection
of charge-changing channels by the original fixed-charge source.

`test_explicit_ci_photoionization.py` contains fifteen checks of signed removal against
independent sorted-list fermion algebra, tiny amplitudes, coherent radial
bright/dark spin-angular limits, full Kr-core d2->d1 strength, legacy native PI
normalization, analytic Milne moments, an independent scalar-density oracle
and identity/resource admission. Four of these check the explicit highest-J
Dyson E1 trace against signed complete-M mixed radial and closed-core
controls, darkness/phase/forbidden sectors and native PI/Milne values. The
PI/Auger/acyclic/architecture panel passes33 tests with an isolated Numba
cache. `test_explicit_ci_auger.py` adds eight checks
of signed h/V/interference, inverse phase volume, degenerate-cluster-safe root
selection, full magnetic versus exact coupled scalar widths and closed-core
normal ordering. `test_continuum_fock_action.py` checks five independent full
second-quantized signed actions and input admission/ownership. The
expanded operator/density/gradient/architecture panel passes160 tests in26.29s, with3
optional external EXP-T checks skipped; actual Sn partial PI/RR
has separate quadrature, shooting-grid and radial-basis refinements in the
[source guide](EUV_SOURCE_MODEL.md#native-adjacent-ci-partial-pi-and-rr). Numerical
convergence of a complete4d model does not certify full Sn rates or experimental CE.
The [balanced full-n4 resonance campaign](EUV_SOURCE_MODEL.md#explicit-signed-ci-auger-and-balanced-sn-thresholds)
has separate omitted-threshold, shooting refinement, resonance-isolation and
radiative-coverage gates. Its measured optical fine splitting fails experiment;
small operator controls do not change that failed scientific gate.

`test_fock_density.py` checks every directed one-/two-body density coordinate
against independent sorted-list Fock actions, including high bits and tiny
coefficients. `test_spinor_radial_density.py` compares every canonical radial
key, closed-core exterior phases, independent Fock energies and actual native
radial gradients. `test_blocked_orbital_functional.py` checks full-matrix
ensemble energy/gradient, owned trial binding, complete degenerate clusters,
admission and actual constrained-optimizer consumption. With the existing
streamed-density and independent state-derivative controls these contribute
41 tests to the combined160-test panel. The actual full8543-CSF Sn probe
passes its separate energy/gradient gates; stationarity and experiment are
not inferred from finite controls.

`test_magnetic_projection_cache.py` adds seven controls for exact signed basis
reuse, fresh tiny/complex root columns, immutable buffers/wrappers, oversized
nonretained entries, full DC/Breit H with opened core holes and orbital phases,
and every actual blocked gradient coordinate against the uncached path.
The affected cache/CI/density/orbital panel passes29 tests. Actual full Sn
H hashes, energy and all gradient coordinates replay bitwise; see the
[measured reuse contract](EUV_SOURCE_MODEL.md#measured-exact-angular-reuse-in-the-full-ci-objective).

`test_fock_removal.py` adds six numerical sorted-list fermion-sign, high-bit,
tiny-amplitude, exact integer bit-linear-form and admission controls. The
45-test affected removal/PI/Auger/cache/architecture panel passes, including
both explicit Python and compiled signed Dyson paths. Compilation changes
neither physical selection rules nor true magnetic statistical averaging.
The later combined operator/density/gradient/resume/cache/architecture panel
passes233 tests in11.91s. These counts include only checks actually executed,
with the earlier Numba-cache reporting failure retained separately.

The blocked ground comparison adds independent complete-H/Rayleigh controls
with both orbital phases, resolved positive/negative steps and coherent CI
interference. Ownership, boundary, excited-root and increment guards reject
unsupported use. A subprocess control compares every gradient coordinate
across two Python hash seeds. The one-CSF stable-difference panel remains
unchanged; the combined current panel passes246 checks. The distinct current-
source mixed-CI diagnostic regression has42 passes and2 failures: strict
bitwise parity with its frozen older H source and a small signed central-
difference precision gate. These failures and the first missing execution-
driver invocation are preserved; this is not an all-green mixed-CI claim or
permission to alter its sealed historical qualification. The13 complete
native/independent Decimal increment controls pass. See the
[source guide](EUV_SOURCE_MODEL.md#measured-exact-angular-reuse-in-the-full-ci-objective)
for actual Sn full-H/metric/canonical-gradient replay and the still-active
original stationarity gate.

`test_explicit_ci_eii.py` adds independent full magnetic/Fock removal-density
reduction with two d_j and interfering radial components, actual native
helium threshold/closed-pair DW normalization, Pauli overlap and immutable
input/full-work rejection. The affected EII/CI/operator panel passes124 tests
with2 retained strict high-energy FAC EII/DW-Born expected failures. Actual
old/new SCF arrays and sharing integrands replay bitwise; the Sn pilot's
partial-wave gate fails despite sharing/shooting/box refinement passes. See
[the complete scope](EUV_SOURCE_MODEL.md#explicit-coherent-ci-direct-electron-impact-ionization).

`test_acyclic_rates.py` adds three independent positive topological-flow
controls against an augmented matrix exponential, full photon/terminal energy
bookkeeping, exact dark terminals, empty graphs and cycle/order/resource
rejections. No renormalization or weak-rate deletion is performed. The
[current conditional cascade](EUV_SOURCE_MODEL.md#conditional-capture-and-complete-window-radiative-cascades)
is not a collision or experimental certificate.

The explicit Sn12+ panel includes `test_one_body_tensor_batch.py`,
`test_tin_euv_workflow.py`, `test_ci_blocks.py` and `test_slater_values.py`.
It covers complete signed complex tensors, early root projection, bounded
projection-cache reuse and changed-coefficient invalidation, high-bit and
parentage/resource guards, native complete-H blocks with opened core holes
and orbital phase reversal, independent creation/annihilation products,
published space counts and distinct gA/Boltzmann weights. After exact-M
coupling reuse, the affected CI/integral/property/architecture panel passes
177 checks with3 optional external-runtime skips; the separate angular/
complete-m-scheme panel passes123 with1 existing strict `fac_frozen` helium
LS-limit expected failure. Its independent radial2p partners violate that
test's common-radial prerequisite. The initial shared-cache HFS serialization
failure and isolated-cache rerun remain in the [baseline record](TIN_EUV_BASELINE.md).

| Essential contract | Selected coverage |
|---|---|
| Public inputs, units, method ownership | Input surface, configuration/method validation, constants, units and architecture boundaries |
| Radial mathematics | Native metric, B-spline identities, analytic Dirac energies, shooting operators, Gauss moments and six-kernel crosschecks |
| Signed CI and eigensolvers | Double-excitation signs, complete traces, dense closure, Davidson, radial snapshots and Hermitian clusters |
| Independent physical anchors | Full signed GRASP open-shell Breit blocks and stored FAC CI operator matrices |
| Public live behavior and corrections | Canonical Atom energy panel, correction ledgers/stack and native checkpoint round trips |
| Rates and dynamics | Tracked native Ne table, complete signed multi-electron CI channels and analytic XFEL kinetics |
| Reproducibility and maintenance | Two-process C/Ne/Be comparison, NIST workflow, documentation links and staged project-index behavior |

Selection is explicit and reviewable; no runtime-based skipping, relaxed
thresholds or truncated production models are used. Extended subsystem tests
remain available by explicit path, but their results are not implied by a green
essential run. Add or replace a default test only when its contract earns the
runtime cost; inspect `--durations` and update this guide in the same change.

Hosted timings also moved `test_ci_analytic_slater_condon.py` and the two
absolute-reservoir campaigns in `test_xfel_reservoir_campaign.py` outside the
default selection. Their models and assertions are unchanged. The small live
three-electron closure, signed channels and pulse checks remain essential.

The multi-minute Pr/Eu/Nd transfer, ns=256/320 DHF bank-comparison and Fe XVII
LMM campaigns were removed from `tests/`. Their unchanged source and SHA-256
manifest are retained in the [historical campaign archive](archive/2026-10/long_running_tests/README.md).
The small complete 91-determinant 4f² Slater gates remain essential; they check
antisymmetry, signed interference, J sectors and parameter recovery, without
claiming free-ion stationarity or experimental agreement. Earlier
[measurements](code_review_2026_10_05/TEST_PERFORMANCE.md) describe the original
workloads and do not measure this new suite. Current measurements and source
identity are in the [CI repair record](ci_repair_2026_10_10/README.md).

## Fixtures and numerical regressions

Tests must read fixtures from tracked paths. `tests/data/ne_native_1000eV.json`
is the unchanged native 134-configuration table, SHA-256
`17d9126593cda0a18a7e47f1d62833b125e57d5110e603eea351402765c008ae`;
it differs from the older `ne_1keV_rate_table.json` model. Its recorded origin
and [repair evidence](ci_repair_2026_10_10/README.md) identify the producer.

`test_ci_dense_orthogonality.py` replays a native clustered CI block at three
energy zeros with the original closure and residual gates. Complete dense CI
uses LAPACK `evd`; partial dense spectra retain `evr`. Both remove the nearest-integer
Hartree origin of the first real block diagonal and restore it to the eigenvalues.
Independent signed 2×2 tests at a large negative energy zero check complete and
subset eigenprojectors against an analytic angle formula, residuals against the
original unchanged H, and preservation of its storage. The native complete-space
[route qualification](yb_mcdhf_rci_2026_10_11/correlated29_radial_reuse_qualification003_preparation/RESULTS.md)
keeps all 1046 signed cluster spans and the old failed epochs separate.
The [original-Yb saved-H diagnostic](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision011/README.md)
rechecks all50 complete symmetry blocks of ten frozen005 Hamiltonians with the
existing current centered helper. Its10-test panel and full residual/Gram
controls pass; the ordinary derivative replay is38/40 and still unqualified.
Exact signed Rayleigh quadratics distinguish root/reporting effects from
finite CI response. It does not rebuild a current-source H/gradient or pass
the two remaining large-step derivative gates.
The nitrogen self snapshot
includes the independently anchored open-shell Breit phase already present in
`233fec1`; it is not an experimental energy reference. The explicitly selected
Sternheimer test retains the Na HF model/tolerance with a 100-iteration budget;
the 40-iteration control failed before convergence at 46. It is outside the
default selection because its extended scaling/heavy-atom work is unnecessary
for every commit.

## Organization

Tests remain grouped by subsystem through their filenames. Shared physical
behavior is tested against package APIs. CLI tests belong to commands that still
exist; removed campaigns do not leave skipped or failing test placeholders.

| Contract | Representative tests |
|---|---|
| Ownership and imports | `test_architecture_boundaries.py`, `test_numerical_design_contracts.py` |
| Public requests and physical identity | `test_architecture_scientific_fidelity.py`, `test_method_request_controls.py` |
| Shooting HF, open shells and constraints | `test_shooting_hf.py`, `test_shooting_guerra.py`, `test_shooting_open_shell_constraints.py` |
| Radial branch identity | `test_dhf_radial_identity.py`, `test_shooting_core_hole_stationarity.py` |
| Dirac radial operator accuracy | `test_shooting_radial_operators.py`: analytic moments, boundary stencils, symmetry and refinement |
| Joint generalized eigenblocks | `test_generalized_eigenblock.py`: independent constructed real H/S with omitted-span perturbations, complete retained columns, physical projectors/energies, joint metric orthogonality, unchanged lower-H convention and pre-factorization resource rejection. `test_rydberg_eigenvector_refinement.py` protects existing single-root behavior. The [86-check affected panel and frozen-Yb g application](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/eigenblock_native001/README.md) retain original residual thresholds; high-energy roots outside the requested band are not newly certified |
| Electron projector and correlated gradient | `test_shooting_electron_reference.py`, `test_shooting_correlated_grid.py` |
| Sparse signed orbital-density projection | `test_ci_complete_trace.py`: complete magnetic-trace anchors plus an independent full-H bilinear check with reversed CSF order, signed vectors, forbidden dense projection and an interference-destroying negative control. The affected derivative/response/cap and public single-/joint-charge state-average panels pass 84 tests; [complete Yb A/B controls](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision006/README.md) separately compare every radial key and native sparse H/Davidson against owned full dense spectra |
| Explicit streamed signed DC density | `test_ci_density_streaming.py`:21 new independent full-Fock bilinear, every-key cached comparison, core/parentage/reordering, tiny signed coefficient, batch single-traversal, native full radial gradient and early resource/input rejection tests pass. The affected113-test derivative/ensemble/response/numeric-H panel passes. [Current source-bound Yb qualification](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision010/README.md) preserves the failed ordinary-FD gates and separates selected-state binding from full-spectrum ownership |
| Explicit numeric CI streaming | `test_ci_numeric_streaming.py`: 30 independent complete Fock-action, signed core/repeated-parentage, DC/static Breit, complex c-product/adjoint, reciprocal/nonreciprocal-key, fresh-cache, resource-rejection and native RKB public-builder checks pass. The affected 232-test panel also passes. [Complete current A/B qualification](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision009/matched001/RESULTS.md) passes 523 live checks and 222 independent saved-result controls. Selected roots and fixed-orbital comparisons do not certify MCDHF stationarity or full-spectrum ownership |
| Streamed exact angular construction | `test_ci_connection_streaming.py` and `test_csf_sparse_projection.py`: independent complete Slater–Condon graphs, high-bit/frozen/isolated/duplicate/vacuum cases, ordered and triangular streams, early-yield memory, signed DC/Breit full-H oracles, CSR explicit-zero/cancellation and complex-phase/promotion controls. Together with the affected public state/gradient/response/snapshot panel, 232 tests pass. [Matched A/B measurements and full4f12 graph evidence](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision008/README.md) distinguish exact structure, angular-stage performance and its then-uncomputed large-class energies |
| Five-state B-spline derivatives | `test_mcdhf_state_resolved_derivative.py`: independent exact-CSF forward differentiation of two-slot ket-normalized one-body and four-slot Coulomb integrals, all five UID components, actual CI-norm/signed-rescaling invariance, rediagonalized polar finite difference and an interference-destroying negative control. Four tests pass. This finite three-electron anchor does not certify Yb stationarity or experimental accuracy; the [complete Yb measurement](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision005/STATUS.md) preserves three failed line-FD gates |
| Analytic constrained CI-response curvature | `test_variational_response.py`:15 independent closed-form/recursive third-order, signed-projector, input/full-spectrum/gap/cap, polar-path and native radial first/second derivative tests pass. With existing polar/state tests25PASS. The [complete original Yb analytic control](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/physical_next_decision012/README.md) passes155 live controls and all16 response predictions. Full affected panel67PASS/2FAIL; the same two legacy failures reproduce in the actual before source with39 bitwise equal arrays. No total-FD or optimizer certificate follows |
| Closed-core attachment self-energy, Dyson roots and one-body CI injection | `test_mbpt_matrix.py`: independent nonzero off-diagonal magnetic-state contractions, signed phase covariance, analytic derivative and auxiliary-Hamiltonian eigenvectors/residues; `test_mbpt_transitions.py`: occupied/external on/off-shell diagonal anchors, immutable snapshots and pole/branch rejection; `test_ci_mbpt_sigma1.py` supplies the legacy diagonal CI consumer; `test_ci_effective_one_body.py` anchors signed full-matrix injection, sqrt(2) two-electron weights, phase covariance, core-hole/producer identity rejection and finite nuclear binding |
| Analytic QED fallback coverage | `test_qed_approximation_coverage.py`: warnings reach scalar/default entry points; `test_qed_self_energy_table.py` supplies the separate numerical anchors |
| Finite core-hole P/Q propagation | `test_ci_effective_qspace.py`: independent full-matrix eigenstates/P weights, signed phase covariance, full residual/derivative, no-Q and uncoupled pole controls, native complete small Be+ spaces, pre-build disjoint identity and resource guards; native full PP/PQ/QQ static Breit and strict C–B second-order controls with independent mixed-eigenvalue derivatives. Exact Bloch wave-operator/metric and full complex signed properties, degenerate/singular/cap rejection now pass nine tests with no skips; finite Q is not atomic all-order SD or generic MBPT screening |
| Cross-sector finite effective electric lines | `test_ci_effective_transitions.py`: rectangular complex signed operators versus dense states, independent endpoint phase covariance, two-metric reverse adjoints, native Be+ complete lowest-M fermionic oracle, length/velocity and finite/long-wavelength controls, normalized full-root matrix elements/rates, second-root photon/statistical weights and pre-work identity/resource rejection. Normalized multiplicity-two 5g3 rotation retains a 2.78e-17 component; weak complex properties use zero absolute tolerance. Four shared angular magnitude cutoffs removed. Together with finite Q-space tests: 21 checks; broader 14-file affected panel: 240 passed, no skips. Finite CI only; no inferred production SD metric or experimental accuracy |
| Complete declared NR-SD space | `test_ci_sd_space.py`: independent occupation-distance enumeration, all jj coupling trees, reference canonicalization, N/J/parity/frozen-shell validation and preallocation resource rejection |
| Finite standalone monovalent SD | `test_monovalent_sd.py`: live native core/result binding; primary Eq.8/9/13 versus full signed canonical/noncanonical action; independent nonlinear feedback root; normalized physical SD wave, separate complete eigenstate residual and expectation; real/complex one-body bilinears, weak nonzero values, source/energy/root/resource rejection. Ten checks pass without skips with the pinned pCI environment. Finite anchor only; [contract and retained failures](ATOMIC_LINEARIZED_SD.md#finite-standalone-monovalent-sd) |
| Noncanonical external attachment capability | `test_fscc_backend_physics_capability.py`: reject original/missing/wrong source capability before export for every nonzero Fov, preserve verified and separately scoped sector paths. `test_fscc_equation_audit.py` and `test_fscc_scope_projection.py` independently distinguish signed equation defects from omitted excitation ranks. The [current audit](atomic_physics_correctness_2026_10_11/README.md) also re-runs the actual public corrected backend against complete finite Bloch/HEFF/FCI controls |
| External attachment matrix convention | `test_fscc_heff_convention.py`: complete saved complex tensor-wave projection and all four physical projectors distinguish adjoint from raw/transpose/conjugate despite equal roots; missing/changed producer signatures reject before export. A live current-public seven-spinor replay checks the immutable physical matrix, unchanged raw blocks and serialized convention. The [54 distinct affected checks](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/public_heff_convention001/README.md) protect C1/axial covariance, core/hole paths and full-property/omitted-rank gates |
| External numerator division and residual floors | `test_fscc_numerator_division.py`: independent analytic two-spinor wave/amplitude controls below/above the old numerator threshold, actual legacy/exact binaries, positive legacy bound at zero update, unknown/mislabeled source rejection before export and byte-exact qualified C patch. Set `ATOMX_EXPT_EXECUTABLE` to the actual public exact-division build and `ATOMX_EXPT_LEGACY_EXECUTABLE` to the pinned legacy binary for all live cases. The nine-file affected panel passes74 without skips on the actual fresh public factory binary (22.73s, `--runxfail`); Ruff passes. This qualifies numerical division/provenance, not omitted-rank or Sn experiment accuracy. [Contract](ATOMIC_FSCC.md#numerator-division-source-contract) |
| J-reduced standalone monovalent SD | `test_reduced_monovalent_sd.py`: 84 live original/streamed/noncanonical checks for four complete valence columns and all ten M; signed actions, fixed-core feedback, independent roots and J², unchanged fixed-energy columns, parent/energy/constructor/solve resource guards. Five-file panel147 pass/3 missing-runtime skips, complemented by3 pass with serial BLAS:150 distinct checks.40 exact-J source-bound replay states/824 arrays pass; stronger unrestricted-M normalized-state negatives are retained. [Full contract and boundaries](ATOMIC_LINEARIZED_SD.md#j-reduced-standalone-monovalent-sd) |
| Finite nonlinear CCSD | `test_finite_ccsd.py`: two-electron exact CI, four-electron noninteracting-pair additivity with nonzero disconnected quadruples, native Be²⁺ complete CSF anchor and strict input/convergence/resource guards; `test_scaffolds.py` checks the explicit-input CC interface |
| Native tensor CCSDT/EA | `test_tensor_coupled_cluster.py`: complete arbitrary off-shell SDT equations and every directed/batched EA entry versus independent full fermion exponentials, three-electron FCI solving, nonzero T3, four-electron omitted T4, physical orbital-phase/energy-zero covariance, owned amplitudes and rejection before integral allocation. [Actual16-spinor Yb qualification](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_tensor_ccsdt001/README.md) adds196 live and353 saved controls, all2/4/6 target partners and full-H defects. The [complete4f extension](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_tensor_4f002/README.md) adds both canonical/dense storage, overlapping tiny-complex signed permutations, exact scalar-sector and Fock/warm-result controls (12 tensor cases);59PASS/2 external-only SKIP in current affected panels,514 live/96 saved controls. Initial exact-cancellation test failure, prior shared-cache failure, SR resource cap and small Davidson J2 failure are preserved. Complete18-core/50-frozen SDT/R3 is qualified, with fixed-kernel matched timing; no full-H state, paper or experimental convergence certificate. The [complete6p continuation](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/native_virtual_p003/README.md) replays12 current tensor cases at unchanged production source and independently qualifies the local DIIS solver against finite FCI/old-Yb amplitudes; new36-spinor core/targets and a22-spinor frozen4f corner pass526/288 live gates. It measures conditional model increments, not portable corrections or a new paired speedup |
| Finite native connected triples and attachment | `test_finite_ccsdt.py`: independent signed two-body FCI/log-CI cumulants, nonzero T3, disconnected T3×T1 size consistency, omitted connected T4 defects, exact Jacobian versus independent full exponential derivatives and pre-allocation caps. `test_eom_attachment.py`: complete N/N+1 FCI spectra and physical states, nonsymmetric EOM operator, omitted 3p2h physical residual, phase/energy-zero/ownership controls, pre-work resource rejection and live native Be²⁺→Be⁺ 1s/2s/3s integrals versus independent complete sparse CI. Finite equation/operator validation, not the 69-electron paper reproduction |
| Finite cluster state actions | `test_cluster_actions.py`: independent complex fermion matrices/exponentials, complete core and N+1 actions, every unit coordinate, inverse/BCH controls, nonzero log-CI T3, disconnected quadruples, tiny amplitudes and pre-work bounds. `test_cc_state_action.py`: real CCSDT/EA solvers versus independent FCI and dense physics, forbidden generator allocation, complete physical projectors, retained omitted-rank residuals and admission before operator calls. The [50-check integration panel and bounded storage/time comparison](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/state_action_integration001/README.md) retain default dense behavior; no production speedup follows |
| Projection between radial representations | `test_shooting_seed.py` |
| Checkpoints and correction replay | `test_shooting_result_persistence.py`, `test_scfresult_save_load.py`, `test_correction_application_contract.py` |
| NIST matching and uncertainty semantics | `test_nist_comparison_workflow.py`, `test_shooting_ionization_comparison.py` |
| Source parsers | `test_nist_bibliography.py`, `test_nist_spectrum_notes.py`, `test_nist_reference_annotations.py` |

The [fixed-virtual complete5p alignment discriminator](yb_mcdhf_rci_2026_10_11/physical_budget31_after_gradient/paper_reproduction001/paper_alignment004/README.md)
replays12 tensor cases at unchanged production source (12PASS/0SKIP), then
passes1171 old-coordinate/sign/off-shell controls and531 new live controls.
All1292379 SDT equations and each of12 complete191880-coordinate R3 vectors
pass1e-10Ha, with original J2/norm gates. This qualifies one22-core/18-virtual
model, whose experimental line errors worsen; it is not full-paper convergence.

Tests of retired archive importers, campaign commands and HTTP routes are
preserved in [the historical test archive](archive/2026-09/retired_script_tests/README.md),
outside the current test collection.

A source parser test does not establish that a removed importer or HTTP route is
still available. CLI schemas and options follow the retained command versions.

## Scientific evidence

- Compare matched Hamiltonians, state identities and numerical controls.
- Keep independent references distinct from Atomx snapshots and cross-engine checks.
- Preserve native metric, correction ledger and convergence evidence through IO.
- Do not loosen a scientific tolerance to accommodate a file move.
- Keep benchmark experiment setup in its owning test when no reusable production
  service needs it. A physics regression must not depend on a retired CLI.

`slow`, `external_fac`, `snapshot` and `book` markers are defined in
`pyproject.toml`. In particular, `external_fac` uses stored reference data;
regenerating it requires the external code. See [verification](VERIFICATION.md)
for the interpretation and provenance of each gate.

A passing collection checks imports, not numerical correctness. Focused tests
validate the affected contracts; only a complete run establishes full-suite
status. Historical test counts and campaign results live in the
[archive](archive/README.md).

The multi-electron PI check compares every signed coordinate with dense Fock
annihilation matrices, including zero amplitudes and interference; a floating
nonzero channel count is not a completeness or convergence gate.

## Commit-time project navigation

Install `.venv/bin/pre-commit install` once per clone. The `project-index` hook
runs `python3 scripts/build_project_index.py --staged --stage` before every
commit. It reads staged blobs, including partially staged files, and stages only
`PROJECT_INDEX.md` and `PROJECT_INDEX.json`; unstaged and untracked scientific
work is excluded. Preserve local index edits before rebuilding. Verify staged
navigation with `--staged --check`; a working-tree preview can legitimately differ.

`tests/test_project_index.py` checks deterministic AST/query output, staged
renames/deletes, partial staging, exclusion of unrelated work, refusal to overwrite
local index edits, invalid-source failure and the installed Git hook in a separate
repository. These are navigation checks; they do not prove physical correctness.

## Correlated-method validation

See [correlated-method checks](CORRELATED_METHOD_TESTING.md) for live optional
backends, model-specific gates and their source-bound validation records.


`test_bound_one_body_composition.py` compares full native signed CI matrices
with an independent bound one-body CI evaluator, including frozen-core trace,
phased cross-radial entries, unchanged ERIs and point/duplicate/payload guards.
Two controls pass; the affected finite composition/transverse/CI panel passes41
with5 optional backend skips. The explicit FSCC export-cap guards retain the
2GiB default and test admitted/malformed budgets before export. On the actual
source-qualified backend,56 live checks pass with one strict historical-marker
XPASS; that same full-FCI case passes normally with `--runxfail`. No tolerance
is changed to suppress the original backend's failure.

`test_direct_maxwellian_rate.py` independently checks analytic thresholded
flux, adaptive integration and finite/shape/positivity guards for the new
callable thermal API. With the native CI thermal entry controls,11 tests pass;
the affected EII/thermal panel passes43 with one retained high-energy expected
failure. These do not replace node/partial-wave convergence or close ionic
rate/population/experimental coverage. See the
[owning numerical evidence](EUV_SOURCE_MODEL.md#explicit-coherent-ci-direct-electron-impact-ionization).


The finite-time acyclic extension adds `test_finite_acyclic_rates.py`: independent
dense CTMC exponentials, exact two-state survival, every repeated-edge reward,
energy ledgers, extreme stiffness and complete resource/geometry rejection.
Together with the unchanged eventual solver14 checks pass. The source/kinetic/
architecture panel passes67 with a private `NUMBA_CACHE_DIR`; its first shared-
cache ReferenceError and mistyped test path are preserved. The [source guide](EUV_SOURCE_MODEL.md#finite-capture-time-window)
records actual15ns temporal refinements and the continuing experimental failure.


`test_axial_resource_admission.py` independently enumerates every allowed ordered
rank/sector/quartic coordinate, tests explicit high-j range/export persistence,
and checks mixed-parity raw HEFF numbering against the generated irrep table.
With512 and original256 source-qualified backends, the affected live panels
each pass50 checks. Seven additional transverse tests pass with `--runxfail`,
preserving the original strict-FCI target/marker. Independent native38-spinor
high-j CCSD/CCSDT C1/512-axial and483/454-determinant FCI controls pass;
the initial hardcoded g/u-offset failure remains archived. On-demand scalar
composition also passes both signed-phase full-CI controls. These establish
representation correctness, not spectroscopy or controlled performance.

`test_compact_axial_eri.py` independently enumerates all complex signed
coordinates, reconstructs full packed files across chunk boundaries, and
checks decompression/resource/source/sidecar rejection. With
`ATOMX_EXPT_COMPACT_ERI_EXECUTABLE` pointing at a separately built
`--compact-axial-eri` backend, complete native CCSD/CCSDT matrices and a
direct fermion three-electron FCI anchor run under a complex multiplet phase.
The combined affected live panel passes80 checks; seven transverse checks
also pass with `--runxfail`, retaining their original numerical targets.
Complete real Sn n4/n5/n6 file replay is bitwise; n6 complete CCSD Heff
agrees with original C1 within6.138e-16Ha. These certify exact storage and
declared finite operators, not physical convergence or controlled speedup.

`test_canonical_body_matrix.py` compares every signed0..3-body matrix entry
against independent exhaustive word application for0..4 electrons, including
complex non-Hermitian coefficients, selected-space order and rejection before
over-budget projection. Eleven checks pass. This finite numerical anchor
does not replace native P/Q, angular, physical-space or experimental gates;
the [source guide](EUV_SOURCE_MODEL.md#finite-second-order-core-hole-operator-controls)
records those separate research controls.

`test_canonical_body_sparse.py` independently qualifies occupied-subset CSC
action against exhaustive signed words and the finite spectator engine.
Complex directed entries, cross-tile selected layouts, exact tiny values,
integer accumulation and separate index/work/counted-output admission are
covered. Current19 checks pass with explicit colex/hash coverage, independent
complex N0..4 word actions, exact CSC byte equality, tiny directed values,
a selected N18/40-spinor case that rejects the full colex map but fits the
hash index, and index-width admission against the installed SciPy consumer.
The first hash slot typing failure is retained; explicit signed slot
indices repair compilation without changing arithmetic. The historical
original eight together with affected finite, angular, native CI, Slater
and Fock-action checks passed61 tests. The added
counted-output check passes in the20-test finite/sparse panel. Native Sn
p6d2 operator matrices at all three hierarchies replay bitwise. Shared
`project_active` angular mapping also matches an independently enumerated
occupied-spinor permutation, while native cached/uncached DC and Breit
matrices remain bitwise equal. These gates precede large selected-sector use.


The exact-hash native qualification also replays every stored coordinate,
value, row index and pointer of the1086326-state single5g Sn matrix:
507892914 entries match the qualified colex result byte for byte. The
row map requires50331648 bytes instead of1509395976; this is an index
storage count, not a whole-solver speedup. Native48/N8 and40/N18 physical38-
electron Slater/complex-phase controls remain separate physical anchors.
The [source guide](EUV_SOURCE_MODEL.md) records source epochs and failures.
