# Public accepted-point optimizer continuation

The lower-level native `optimize_orbitals` now accepts a validated immutable numerical state. It restores the original ordered no-pair projectors, coordinate point and scaled inverse metric, recomputes the supplied full CI objective/gradient, and starts with empty secant history. The existing fresh default trajectory is unchanged: its complete diagnostics and physical coefficients matched the frozen pre-change optimizer bitwise over 12 independent-particle optimization steps.

This is a numerical continuation interface. It has not performed a new mixed531 Yb solve, implemented a stable mixed-CI energy increment, or established expanded-space MCDHF stationarity or experimental accuracy. The [prior mixed-CI probe](../mixed_ci_stationarity_probe/README.md) and [prospective successor derivation](../mixed_ci_stationarity_probe/PUBLIC_OPTIMIZATION_SUCCESSOR.md) remain unchanged historical evidence. The latter's absent-interface statement describes its recorded source epoch; this implementation replaces that particular interface gap.

## Using the interface

Import `OrbitalOptimizationState` and `optimize_orbitals` from [orbital_optimization.py](../../../src/atomx/numerics/bspline/orbital_optimization.py). Construct a state from complete archived arrays:

```python
state = OrbitalOptimizationState.from_arrays(
    functional,
    layout=original_layout,
    coordinates=original_coordinate_point,
    inverse_diagonal=original_inverse_diagonal,
    inverse_blocks=original_inverse_blocks,
    projector_source="nuclear",
    objective_identity=original_objective_sha256,
    occupation_floor=0.05,
)
waves, ci, diagnostics = optimize_orbitals(
    functional,
    resume_state=state,
    objective_identity=new_objective_sha256,
    allow_objective_change=True,
    tolerance=1e-7,
    energy_tolerance=1e-8,
    checkpoint_callback=save_accepted_point,
    max_wall_time=830,
    return_state=True,
)
final_state = diagnostics.pop("optimizer_state")
```

`layout` is a tuple/list of `(ordered_keys, projector, (rows, columns), contiguous_slice)` entries with exactly one complete group per κ. All saved arrays must be finite real float64, including nested tuple/list inverse blocks. The coordinate vector is authoritative. The optional radial `orbitals` argument must equal its exact unpacked float64 coefficients; the API never reconstructs coordinates through PᵀSC or repairs them with a seed polar map. When only saved radial coefficients existed, the consumer must disclose its recovered coordinate point rather than claim restoration of unsaved original coordinates. The [prepared Yb consumer](../public_mixed_resume_run/README.md) uses the already archived recovered point and states its approximately 2.987e−13 coefficient discrepancy.

`objective_identity` is a nonempty caller-declared string. The numerical API does not infer or authenticate signed CSF order, state matching, EOL weights or physical operators from a string. The scientific consuming contract must bind those inputs, old and new targets, and source epochs. A changed string requires `allow_objective_change=True`; both changed and unchanged resumes recompute CI/gradient and discard all old secant history. The state contains no old CI, gradient, history, tolerance, or convergence certificate. Explicit call tolerances remain necessary.

The state exposes `layout`, `coordinates`, `inverse_diagonal`, `inverse_blocks`, `residual_metric_sha256`, `geometry_sha256`, `state_sha256`, `projector_source`, `occupation_floor` and `objective_identity`. The legacy residual SHA preserves the original list-key/shape/P/inverse byte algorithm, including original005 SHA aae3a596009a671fd905f3c97f935cf17408a344a3d08cced5fbfb3ab2dcb6fe. The new fully framed geometry hash additionally binds actual S, the complete numerical B-spline workspace, Z/ns, nuclear one-body operators, sampled nuclear potential and projector-policy operators. The point hash adds coordinates and the caller objective declaration.

## Validation and lifecycle

Array import owns bytes-backed immutable copies; caller mutation cannot alter the state, and `setflags(write=True)` cannot expose mutable storage. Import validates complete ordered same-κ groups, contiguous nonoverlapping slices, exact shapes, finite geometry, active S positivity, S-orthonormal projectors and a complete selected positive-energy span. Inactive padded S rows/columns and P rows must be exactly zero. The supplied point must satisfy both coordinate and actual radial S-Gram defects ≤1e−10. Inverse diagonals must be positive; full blocks must be symmetric and strictly positive. Validation does not clip, symmetrize, drop imaginary components, rotate projectors or polar-normalize the point.

The import eigensolve verifies the saved positive subspace against the actual supplied operator and S; its eigenvectors never replace the archived P. A Fock-space import requires the original complete κ-indexed projector operators through `projector_operators`. Original inverse blocks cannot reconstruct a unique original Fock operator. Nuclear-space imports default to the actual functional nuclear one-body operator.

Resume checks current ordered keys, complete physical geometry and the sealed residual/payload hashes before the first objective evaluation. It performs no spectrum solve, gap reconstruction, occupation rescaling, or seed orthonormalization. Resume explicitly rejects new preconditioner matrices/factories, shooting updates, a changed projector policy or occupation floor. The private geometry dataclass is an implementation detail; this nonadversarial Python API and its deterministic hashes do not authenticate an untrusted pickle or malicious private-object forgery.

`checkpoint_callback(iteration, state, orbitals, ci, diagnostics)` receives iteration0 and every actual accepted point. Its diagnostics include the original residual/energy indicator, explicit tolerances, objective identity and geometry/metric SHA. The old four-argument progress callback remains unchanged. Subsequent snapshots share the same sealed P/S/inverse payload; only small point coordinates are copied and checked. The complete workspace hash and spectral validation are not repeated at each checkpoint. Ordinary calls without state/checkpoint export do not build expensive geometry payloads or add typed objects to their old JSON diagnostics.

`max_wall_time` includes initialization. It stops at the last complete accepted point; if time expires during a trial, that trial is discarded and its CI/energy cannot replace the accepted x/g/CI. A budget return has `termination_reason="wall_time_limit"` and `converged=False`, including a zero-budget call. A single objective evaluation cannot be interrupted by this soft check; the consumer still needs a separate hard resource stop and durable checkpoint writes. A successful line-search step is not stationarity. No convergence gates are relaxed.

The fresh factory/refresh parameters remain accepted for compatibility, but the current implementation does not perform automatic refreshes and each search starts at unit step. Existing inaccurate refresh/previous-step docstrings were corrected; no uncontracted numerical refresh path was added.

## Evidence actually obtained

[contract.json](contract.json) was written before implementation/tests and binds the 353-file starting source manifest, relevant archived inputs, mathematical guards and CPU1/small-model resource plan. Its hypothesis's phrase about CI/history reconstruction is clarified here: original CI/history are never restored, and the acceptance clauses require a full new CI/gradient evaluation. The frozen pre-change module is [source_before/orbital_optimization.py](source_before/orbital_optimization.py); [source_after_manifest.json](source_after_manifest.json) identifies the actual new source and [optimizer.patch](optimizer.patch) records only this increment against that pre-change module.

[test_orbital_optimizer_resume.py](../../../tests/test_orbital_optimizer_resume.py) passed **40 controls in 0.50 s** with numerical threads pinned to1; [focused.log](focused.log) records that tool result. Both changed files passed Ruff. Independent ns35 one-body Rayleigh problems check diagonal and full SPD inverse actions, projected gradients and actual accepted energies/Gram matrices; changing occupation does not rescale a restored inverse. An explicit new objective recomputes CI and changes the gradient consistently. Two genuine accepted continuation steps retain physical orthogonality and share the sealed geometry. Default old/new 12-step trajectories and all legacy diagnostics agree bitwise.

Reject controls cover incorrect order/groups/slices/shapes, negative-energy contamination, nonorthogonal points/projectors, complex/NaN data, nonpositive/asymmetric inverses, caller mutation, wrong workspace/S/nuclear operator/nucleus/Z, incompatible controls and inconsistent radial seeds. Framed-hash controls distinguish previously colliding scalar sequences and nested mappings. A combined radial-Gram control rejects a point whose two individually allowed defects would compose above the physical guard. Safe zero-budget and trial-budget controls preserve the accepted point. These are interface and analytic manufactured-model tests, not a real mixed-CI stationarity proof.

[focused_attempt001.json](focused_attempt001.json) preserves three initial test-harness failures: two exact float equality assertions differed by 2.22e−16 Ha because the independent strided matrix contraction changes operation order; a third referenced a nonexistent test-only negative-eigenvector attribute. The tests now use a 2e−15 Ha independent Rayleigh allowance and a separate full generalized eigensolve for the negative branch. No production physical or scientific gate changed.

No controlled performance benchmark or observed peak-memory measurement was taken in this subtask. The focused process used one-thread settings and tiny ns35 models; its 3.95 GiB plan was not enforced through a process address-space limiter. Root separately ran the prospectively declared affected panel: **115 tests passed in 95.18 s**, with unchanged353 source and unchanged test hashes, CPU affinity1, numerical threads1, a300s timeout and isolated compiler cache; no regression memory cap was applied. Its [verification](../public_resume_integration/verification.json), [execution binding](../public_resume_integration/execution_binding.json) and [pytest log](../public_resume_integration/pytest.log) are separate obtained evidence. Root owns independent audit integration, calling-layer source seal and any later Yb computation. [verification.json](verification.json) records the exact checks/hashes and these limits.

The separate read-only [final physics/interface audit](../public_optimizer_resume_audit/final_review.json) passed13 checks at the exact frozen source/test hashes with no unresolved material finding. Its [original004 metric control](../public_optimizer_resume_audit/legacy_metric_control.json) independently reproduced legacy metric SHA aae3a596… and the saved-gradient inverse action with maximum element error0; [typed hash resolution](../public_optimizer_resume_audit/typed_hash_resolution.json) retains the hash-framing failure and repair controls. This audit does not duplicate the numerical test execution or certify mixed531 physics.

## Documentation impact

[documentation_impact.json](documentation_impact.json) records source docstring corrections and this current authoritative interface account. Root owns the shared capabilities/reference/navigation changes and the separately prepared consumer; no old scientific results, sources or traces were edited here. Source-before successor proposals are retained with source-epoch qualification and incoming links. The full `MCDHFSolver.solve` still initializes a new solve; this lower-level interface does not add a solver-dispatch resume backend. EARS is recorded in ignored `trace.md`.
