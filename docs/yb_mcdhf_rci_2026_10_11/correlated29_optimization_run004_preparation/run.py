"""Source-only preparation for one bounded native reported-EOL MCDHF call.

No numerical imports precede exact root admission. Historical derivative failures remain unchanged. Narrow new same-point
controls admit only a separately sealed bounded early experiment. The stored-H
guard qualifies checkpoints; it never changes the optimizer's objective.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


class QualificationStop(RuntimeError):
    """Keep the forensic candidate and the previous qualified checkpoint."""


class ResourceStop(RuntimeError):
    """A declared producer cap stops without silently shrinking the model."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def target_bindings(targets, tracking):
    """Match by physical UID; symmetry insertion order is not label order."""
    matched = {}
    for binding in tracking:
        if len(binding["reference_levels"]) != 1 or len(binding["selected_levels"]) != 1:
            raise QualificationStop("Expected singleton physical target")
        uid = tuple(binding["reference_levels"][0])
        if uid in matched or tuple(binding["selected_levels"][0][:2]) != uid[:2]:
            raise QualificationStop("Duplicate or changed target UID")
        matched[uid] = binding
    expected = {(t["twoJ"], t["parity"], t["models"]["DC"]["local_root"]) for t in targets}
    if len(expected) != 5 or set(matched) != expected:
        raise QualificationStop("Original five-target matching incomplete")
    return matched


def physical_rows_by_label(rows):
    """Select physical lower/upper states independently of input row order."""
    expected = {"6s_1/2": (1, 1, 0), "5d_3/2": (3, 1, 0),
                "5d_5/2": (5, 1, 0), "6p_1/2": (1, -1, 0), "6p_3/2": (3, -1, 0)}
    by_label = {row["label"]: row for row in rows}
    if len(rows) != 5 or set(by_label) != set(expected):
        raise QualificationStop("Physical five-state labels incomplete or duplicated")
    if any(tuple(by_label[label]["UID"]) != uid for label, uid in expected.items()):
        raise QualificationStop("Physical label and original UID disagree")
    return by_label


def nominal_four_gap_rms(rows):
    physical_rows_by_label(rows)
    upper = [row for row in rows if row["label"] != "6s_1/2"]
    if len(upper) != 4:
        raise QualificationStop("Expected exactly four physical upper states")
    return math.sqrt(math.fsum(row["nominal_gap_cm1"]**2 for row in upper) / 4)


def prerequisite_controls(contract, evidence):
    """Closed terminal point only; no global derivative admission inherited."""
    result = evidence["prior_result"]
    receipt = evidence["prior_receipt"]
    completion = evidence["prior_completion"]
    terminal = evidence["terminal_manifest"]
    if not (result["status"] == "bounded_public_optimizer_returned" and result["error"] is None
            and result["source_map_sha256"] == contract["source"]["candidate356_map_sha256"]
            and receipt["status"] == "foreground_closed" and receipt["exit_code"] == 0
            and receipt["actual_files_sha256"]["actual/result.json"] ==
                contract["pending_prerequisites"]["prior_result"]["sha256"]
            and completion["all_completion_metadata_checks_passed"] is True
            and completion["all16_finite_stored_H_acceptances_qualified"] is True
            and completion["stationarity_passed"] is False):
        raise ValueError("Prior same-source bounded run is not closed and qualified")
    if not (len(result["controls"]) == 1175 and all(c["passed"] for c in result["controls"])
            and [t["iteration"] for t in result["trajectory"]] == list(range(17))
            and result["last_qualified_checkpoint"] == contract["pending_prerequisites"]["terminal_manifest"]["path"]
            and result["trajectory"][-1]["manifest_sha256"] ==
                contract["pending_prerequisites"]["terminal_manifest"]["sha256"]
            and terminal["iteration"] == 16 and terminal["checkpoint_qualified"] is True
            and terminal["array_sha256"] == contract["input_roles"]["continuation_baseline_arrays"]["sha256"]
            and terminal["metric_sha256"] == contract["metric"]["residual_metric_sha256"]
            and terminal["geometry_sha256"] == contract["metric"]["geometry_sha256"]
            and terminal["source_map_sha256"] == result["source_map_sha256"]
            and terminal["stationarity_passed"] is False
            and terminal["exact_acceptance"]["qualified"] is True):
        raise ValueError("Continuation must use exact terminal16, not a selected earlier gap")
    expected = {"H": 27, "CI": 27, "gradient": 17, "optimizer": 1, "validation_Dirac": 7}
    if any(result["completed_calls"][key] != value for key, value in expected.items()):
        raise ValueError("Wrong prior actual scientific scope")
    guard = evidence["exact_guard_qualification"]
    if guard["all_passed"] is not True or guard["no_Yb_gradient_or_optimizer_qualification"] is not True:
        raise ValueError("Exact finite stored-H arithmetic guard unqualified")
    old = evidence["original_metadata_source_identity"]
    new = evidence["corrected_source_identity"]
    name = "src/atomx/numerics/bspline/orbital_optimization.py"
    if new["all_atomx_python_sha256"][name] != old["all_atomx_python_sha256"][name]:
        raise ValueError("Untested optimizer metadata implementation change")
    if evidence["prior_objective_identity"]["sha256"] != terminal["objective_sha256"]:
        raise ValueError("Changed original physical objective")


def metadata(contract):
    """Pure standard-library readiness; execution still needs an exact root seal."""
    if contract.get("execution_ready") is not True:
        raise ValueError("Bounded optimizer prerequisites are pending")
    if contract["status"] != "prepared_requires_exact_root_execution":
        raise ValueError("Preparation pending; no scientific execution admission")
    if any(v is None for v in contract["pending_prerequisites"].values()):
        raise ValueError("Missing qualified prerequisite")
    for role in (*contract["input_roles"].values(), *contract["pending_prerequisites"].values(),
                 *contract["hold_evidence"].values(), *contract["prospective_successor_of"].values()):
        if sha(REPO / role["path"]) != role["sha256"]:
            raise ValueError("Changed input, prerequisite or historical evidence")
    evidence = {name: load(REPO / role["path"]) for name, role in
                contract["pending_prerequisites"].items() if role["path"].endswith(".json")}
    prerequisite_controls(contract, evidence)
    failure = load(REPO / contract["hold_evidence"]["rotation002_failure"]["path"])
    if failure.get("error") != "Scientific gate failed: direction1_NEW_Rayleigh_derivative":
        raise ValueError("Historical failed gate was changed")
    identity = load(REPO / contract["input_roles"]["candidate_source_identity"]["path"])
    if identity["source_map_sha256"] != contract["source"]["candidate356_map_sha256"]:
        raise ValueError("Wrong complete source epoch")
    source = REPO / contract["source"]["candidate_archive"]
    for name, expected in identity["all_atomx_python_sha256"].items():
        if sha(source / name.removeprefix("src/")) != expected:
            raise ValueError("Frozen source mismatch")
    return source


def admit(contract, execution):
    if execution.get("execution_authorized_by_root") is not True:
        raise ValueError("Wrong exact root execution binding")
    source = metadata(contract)
    resources = contract["resources"]
    expected = {"execution_authorized_by_root": True, "driver_sha256": sha(__file__),
                "contract_sha256": sha(HERE / "contract.json"),
                "preparation_sha256": sha(HERE / "preparation_verification.json"),
                "source_map_sha256": contract["source"]["candidate356_map_sha256"],
                "CPU": 31, "wall_seconds": 1200, "address_space_GiB": 3.95,
                "max_H": 257, "max_CI": 257, "max_full_gradient": 65,
                "max_optimizer_calls": 1, "max_new_accepted_steps": 64}
    if any(execution.get(k) != v for k, v in expected.items()):
        raise ValueError("Wrong exact root execution binding")
    proof = load(HERE / "preparation_verification.json")
    if (proof.get("all_source_only_checks_passed") is not True
            or proof.get("runner_sha256") != expected["driver_sha256"]
            or proof.get("contract_sha256") != expected["contract_sha256"]):
        raise ValueError("Wrong prepared consumer proof")
    if any(os.environ.get(k) != "1" for k in (
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")):
        raise ValueError("Single-thread environment required")
    accounting = sorted(os.sched_getaffinity(0))
    load_average = os.getloadavg()
    if 31 not in accounting or load_average[0] > execution["maximum_system_load1"]:
        raise ValueError("Host admission failed")
    if any(n == "atomx" or n.startswith("atomx.") or n == "numpy" or n.startswith("numpy.") for n in sys.modules):
        raise ValueError("Preimported physical source is forbidden")
    resource.setrlimit(resource.RLIMIT_AS, (int(resources["address_space_GiB"] * 2**30),) * 2)
    os.sched_setaffinity(0, {31})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(ResourceStop("1200s process cap")))
    signal.alarm(resources["wall_seconds"])
    cache_path = Path(os.environ["NUMBA_CACHE_DIR"]).resolve()
    if not cache_path.is_relative_to(HERE.resolve()) or any(cache_path.iterdir()):
        raise ValueError("New empty corrected-source process cache required")
    sys.path.insert(0, str(source))
    return source, {"original_accounting_CPUs": accounting, "affinity": [31],
                    "load_average": load_average, "threads": 1,
                    "inner_wall_seconds": 1200, "outer_wall_seconds": 1215,
                    "address_space_GiB": 3.95, "performance_claim_valid": False}


def execute(contract, execution):
    source, admission = admit(contract, execution)
    import numpy as np
    from scipy import linalg as scipy_linalg

    import atomx.methods.csf.variational as variational
    import atomx.numerics.bspline.dirac_assembly as dirac
    import atomx.numerics.bspline.orbital_optimization as optimization
    import atomx.physics.csf.matrix as matrix
    import atomx.physics.csf.tracking as tracking_module
    from atomx.angular.csf_hamiltonian import csf_signature
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.constants import HARTREE_CM1
    from atomx.methods.ci import CIResult
    from atomx.physics.csf.levels import ci_level_blocks
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.physics.csf.tracking import CIBasisTracking
    from atomx.potentials.nuclear import make_nuclear_potential

    for name, module in tuple(sys.modules.items()):
        if ((name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None)
                and not Path(module.__file__).resolve().is_relative_to(source.resolve())):
            raise ValueError("Undeclared physical import")
    helper_path = REPO / contract["pending_prerequisites"]["exact_guard_helper"]["path"]
    spec = importlib.util.spec_from_file_location("exact_reported_guard", helper_path)
    guard = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = guard
    spec.loader.exec_module(guard)
    class ContractExactGuardBudget(guard.ExactGuardBudget):
        """Explicit run004 resource successor; all exact arithmetic unchanged."""
        def reserve(self, roots, pairs):
            if roots < 0 or pairs < 0:
                raise ValueError("Negative exact consumer work")
            if self.points + 1 > 65 or self.quotients + roots > 325 or self.examined_pairs + pairs > 8_000_000:
                raise guard.ExactGuardBudgetExceeded("declared run004 exact acceptance budget exhausted")
            self.points += 1
            self.quotients += roots
            self.examined_pairs += pairs

    exact_budget = ContractExactGuardBudget()
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    (output / "checkpoints").mkdir()
    started_at = time.monotonic()
    caps = contract["resources"]
    limits = {"H": 257, "CI": 257, "gradient": 65, "optimizer": 1,
              "validation_Dirac": 7, "initial_hS": 7, "CI_internal_hS": 1799,
              "geometry_eigvalsh": 36, "scipy_eigh": 1306, "tracking_resolution": 321,
              "tracking_SVD": 1605, "tracking_overlap": 322,
              "polar_retraction": 1792, "polar_numpy_eigh": 1792,
              "unfiltered_target_density": 325}
    attempted = dict.fromkeys(limits, 0)
    completed = dict(attempted)
    checks, trajectory, raw_hS = [], [], []
    previous = None
    captured = {}

    def flush(phase):
        write(output / "progress.json", {"phase": phase, "attempted": attempted,
              "completed": completed, "controls": checks, "trajectory": trajectory,
              "exact_guard_budget": exact_budget.ledger(),
              "elapsed_uncontrolled_s": time.monotonic() - started_at})

    def check(name, error, limit=0.):
        passed = bool(np.isfinite(error) and error <= limit)
        checks.append({"name": name, "error": float(error), "limit": limit, "passed": passed})
        if not passed:
            flush(name)
            raise QualificationStop("Scientific gate failed: " + name)

    def wrap(name, function, capture=False):
        def call(*args, **kwargs):
            if attempted[name] >= limits[name]:
                flush("before_resource_stop_" + name)
                raise ResourceStop("Producer call cap: " + name)
            attempted[name] += 1
            flush("before_" + name)
            result = function(*args, **kwargs)
            completed[name] += 1
            if capture:
                raw_hS.append((name, int(args[1]), result[0].copy(), result[1].copy()))
                np.savez_compressed(output / f"native_raw_hS_{len(raw_hS)-1:03d}.npz",
                                    kappa=np.array(int(args[1])), raw_h=result[0], S=result[1])
            flush("after_" + name)
            return result
        return call

    original_hS = dirac.build_dirac_matrices
    variational.build_dirac_matrices = wrap("initial_hS", original_hS, True)
    matrix.build_dirac_matrices = wrap("CI_internal_hS", original_hS, True)
    matrix.build_ci_matrix = wrap("H", matrix.build_ci_matrix)
    variational.ci_solve = wrap("CI", variational.ci_solve)
    dirac.solve_dirac = wrap("validation_Dirac", dirac.solve_dirac)
    variational.CSFOrbitalFunctional.gradient_from_weights = wrap(
        "gradient", variational.CSFOrbitalFunctional.gradient_from_weights)
    CIBasisTracking.resolve = wrap("tracking_resolution", CIBasisTracking.resolve)
    tracking_module.ci_subspace_overlaps = wrap("tracking_overlap", tracking_module.ci_subspace_overlaps)
    np.linalg.svd = wrap("tracking_SVD", np.linalg.svd)
    np.linalg.eigvalsh = wrap("geometry_eigvalsh", np.linalg.eigvalsh)
    scipy_linalg.eigh = wrap("scipy_eigh", scipy_linalg.eigh)
    optimization._polar_retraction = wrap("polar_retraction", optimization._polar_retraction)
    np.linalg.eigh = wrap("polar_numpy_eigh", np.linalg.eigh)
    optimization.optimize_orbitals = wrap("optimizer", optimization.optimize_orbitals)

    def path(role):
        return REPO / contract["input_roles"][role]["path"]

    def archive(role):
        with np.load(path(role), allow_pickle=False) as values:
            return {k: values[k].copy() for k in values.files}

    def waves_from(values, keyname):
        return {tuple(map(int, key)): (p.copy(), q.copy()) for key, p, q in
                zip(values[keyname], values["P"], values["Q"], strict=True)}

    radial, saved_g, saved_waves = archive("geometry_workspace"), archive("saved_gradient"), archive("saved_waves")
    geometry = archive("screened_geometry")
    data = {k[6:]: v.item() if v.ndim == 0 else v.copy() for k, v in radial.items() if k.startswith("basis_")}
    models = load(path("screened_models"))["models"]
    old531 = [c for row in models["baseline"]["configs"] for c in enumerate_jj_csfs(row["config"])]
    parent = [c for row in models["balanced_virtual"]["configs"] for c in enumerate_jj_csfs(row["config"])]
    sectors = {(j, p) for j, p, _ in contract["fixed"]["target_blocks"]}
    csfs = [c for c in parent if (c.total_J2, c.parity) in sectors]
    check("full531_reference", len(old531) != 531)
    check("full2163_parent", len(parent) != 2163)
    check("complete1046_target_space", len(csfs) != 1046)
    check("same69electrons", {c.n_electrons for c in csfs} != {69})
    for j, p, count in contract["fixed"]["target_blocks"]:
        check(f"complete_sector_{j}_{p}", sum((c.total_J2, c.parity) == (j, p) for c in csfs) != count)
    signatures = [json.dumps(plain((s.subshells, s.coupling, c.total_J2, c.parity)), separators=(",", ":"))
                  for c in csfs for s in [csf_signature(c)]]
    check("same_ordered_signed1046", signatures != load(path("screened_basis"))["balanced_virtual"]["retained_signatures"])
    targets = load(path("tracking_binding"))["targets"]
    average = CIStateAverage(tuple((t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.) for t in targets))
    nuclear = make_nuclear_potential(70, "fermi", A=175)
    reference = archive("tracking_ci")
    tracker = CIBasisTracking(average, old531, CIResult(reference["energies_Ha"], reference["mixing"]),
                             waves_from(radial, "orbital_keys"),
                             {k: geometry[f"s_{k}"] for k in set(key[1] for key in map(tuple, saved_waves["keys"]))},
                             max_determinant_pairs=1_000_000)
    functional = variational.CSFOrbitalFunctional(csfs, data, 70, nuclear,
        state_average=average, level_tracking=tracker, retain_hamiltonian=True, max_determinants=2000)
    functional.radial_density.contract_bilinear = wrap(
        "unfiltered_target_density", functional.radial_density.contract_bilinear)
    check("ordered29keys", tuple(functional.keys) != tuple(map(tuple, contract["fixed"]["keys29"])))
    initial_raw = {k: (h, s) for _, k, h, s in raw_hS}
    for k, (h, s) in initial_raw.items():
        logical = np.tril(h) + np.tril(h, -1).T
        check(f"logical_h_{k}", not np.array_equal(logical, functional.one_body[k]))
        check(f"original_logical_h_{k}", not np.array_equal(logical, geometry[f"h_{k}"]))
        check(f"original_S_{k}", not np.array_equal(s, geometry[f"s_{k}"]))
    raw_checked = 0

    def raw_identity():
        nonlocal raw_checked
        for n in range(raw_checked, len(raw_hS)):
            kind, k, h, s = raw_hS[n]
            old_h, old_s = initial_raw[k]
            check(f"captured_raw_h_{kind}_{n}", h.tobytes() != old_h.tobytes())
            check(f"captured_S_{kind}_{n}", s.tobytes() != old_s.tobytes())
        raw_checked = len(raw_hS)

    layout_rows = load(path("saved_layout"))
    layout = [(tuple(map(tuple, r["keys"])), saved_g[f"projector_{i}"], tuple(r["shape"]), slice(*r["slice"]))
              for i, r in enumerate(layout_rows)]
    identity_payload = {"method": "Atomx public native mixed-CI MCDHF", "objective": "ordinary reported EOL",
        "ordered_signed_CSFs": signatures, "targets": [{"label": t["label"], "reference_UID":
        [t["twoJ"], t["parity"], t["models"]["DC"]["local_root"]], "physical_weight": .2} for t in targets],
        "operator": "same scalar DC; no Breit/QED/recoil", "nucleus": "Fermi Z70 A175",
        "radial_basis_sha256": contract["input_roles"]["geometry_workspace"]["sha256"],
        "frozen_tracker_inputs": {k: contract["input_roles"][k] for k in
                                  ("tracking_ci", "tracking_binding", "tracking_orbitals")},
        "source_lineage": contract["source"], "new3451_metric": contract["metric"]}
    objective_identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True,
                                                  separators=(",", ":")).encode()).hexdigest()
    previous_identity = load(REPO / contract["pending_prerequisites"]["prior_objective_identity"]["path"])
    check("same_original_physical_objective_payload", identity_payload != previous_identity["payload"])
    check("same_original_physical_objective_SHA", objective_identity != previous_identity["sha256"])
    write(output / "objective_identity.json", {"payload": identity_payload, "sha256": objective_identity,
                                               "caller_declaration_not_certification": True})
    initial = optimization.OrbitalOptimizationState.from_arrays(functional, layout=layout,
        coordinates=saved_g["x"], inverse_diagonal=saved_g["inverse_diagonal"],
        inverse_blocks=[saved_g[f"inverse_blocks_{i}"] for i in range(7)],
        projector_source="nuclear", occupation_floor=.05, objective_identity=objective_identity)
    check("same_residual_metric", initial.residual_metric_sha256 != contract["metric"]["residual_metric_sha256"])
    check("same_physical_geometry", initial.geometry_sha256 != contract["metric"]["geometry_sha256"])
    check("same_saved_x_bytes", initial.coordinates.tobytes() != saved_g["x"].tobytes())

    def unpack(state):
        waves = {}
        for keys, projector, shape, where in state.layout:
            c = projector @ state.coordinates[where].reshape(shape)
            for j, key in enumerate(keys):
                waves[key] = c[:123, j].copy(), c[123:, j].copy()
        return waves

    def tangent(state, vector):
        result = vector.copy()
        for _, _, shape, where in state.layout:
            y, g = state.coordinates[where].reshape(shape), vector[where].reshape(shape)
            a = y.T @ g
            result[where] = (g - .5 * y @ (a + a.T)).ravel()
        return result

    waves = unpack(initial)
    old_waves = waves_from(saved_waves, "keys")
    for key in functional.keys:
        check("same_direct_saved_wave_" + str(key), any(a.tobytes() != b.tobytes()
              for a, b in zip(waves[key], old_waves[key], strict=True)))
    original_evaluate = functional.evaluate

    def evaluate(orbitals, gradient=True, ci_result=None, **kwargs):
        if gradient and ci_result is not None:
            # The public line search already accepted this CI. Preserve its
            # forensic physical point even if the subsequent gradient fails.
            candidate_h = ci_result.hamiltonian_for(csfs, orbitals, functional.keys, data, 70, 123,
                nuclear_potential=nuclear, max_determinants=2000)
            candidate_path = output / f"accepted_trial_before_gradient_{completed['gradient']:04d}.npz"
            np.savez_compressed(candidate_path, H=candidate_h, energies=ci_result.state_energies,
                mixing=ci_result.mixing_coefficients, keys=np.array(functional.keys),
                P=np.array([orbitals[k][0] for k in functional.keys]),
                Q=np.array([orbitals[k][1] for k in functional.keys]))
            write(candidate_path.with_suffix(".json"), {"checkpoint_qualified": False,
                "array_sha256": sha(candidate_path), "gradient_completed": False,
                "previous_qualified_checkpoint": previous["manifest_path"] if previous else None,
                "owner": plain(ci_result.hamiltonian_snapshot.identity_payload())})
        result = original_evaluate(orbitals, gradient=gradient, ci_result=ci_result, **kwargs)
        if gradient:
            captured.update(radial_gradient={k: v.copy() for k, v in result[1].items()},
                            ci=result[-1], energy=float(result[0]))
        return result

    functional.evaluate = evaluate
    nominal = {row["label"]: row["NIST_nominal_cm1"] for row in load(path("nominal_NIST_rows"))["DC_rows"]}
    continuation_baseline = contract["input_roles"]["continuation_baseline_arrays"]
    with np.load(REPO / continuation_baseline["path"], allow_pickle=False) as values:
        continuation_baseline_arrays = {key: values[key].copy() for key in values.files}

    def checkpoint(iteration, state, orbitals, ci, diagnostics):
        nonlocal previous
        if iteration > 64 or len(trajectory) >= 65:
            raise ResourceStop("Accepted checkpoint cap")
        point_path = output / "checkpoints" / f"accepted_{iteration:04d}"
        h = ci.hamiltonian_for(csfs, orbitals, functional.keys, data, 70, 123,
                              nuclear_potential=nuclear, max_determinants=2000)
        if captured.get("ci") is not ci:
            raise QualificationStop("Checkpoint has no same-CI original radial gradient")
        raw_g = np.zeros(3451)
        for keys, projector, _, where in state.layout:
            raw_g[where] = (projector.T @ np.column_stack([captured["radial_gradient"][k] for k in keys])).ravel()
        g = tangent(state, raw_g)
        inverse_g = np.zeros_like(g)
        for (_, _, shape, where), blocks in zip(state.layout, state.inverse_blocks, strict=True):
            a = g[where].reshape(shape)
            inverse_g[where] = np.column_stack([block @ a[:, j] for j, block in enumerate(blocks)]).ravel()
        pg = tangent(state, inverse_g)
        arrays = {"x": state.coordinates, "H": h, "energies": ci.state_energies,
                  "mixing": ci.mixing_coefficients, "keys": np.array(functional.keys),
                  "P": np.array([orbitals[k][0] for k in functional.keys]),
                  "Q": np.array([orbitals[k][1] for k in functional.keys]),
                  "raw_coordinate_gradient": raw_g, "tangent_gradient": g,
                  "preconditioned_gradient": pg,
                  "radial_gradient": np.array([captured["radial_gradient"][k] for k in functional.keys]),
                  "inverse_diagonal": state.inverse_diagonal,
                  **{f"projector_{i}": row[1] for i, row in enumerate(state.layout)},
                  **{f"inverse_blocks_{i}": np.asarray(v) for i, v in enumerate(state.inverse_blocks)}}
        np.savez_compressed(point_path.with_suffix(".npz"), **arrays)
        manifest = {"iteration": iteration, "global_accepted_step": 16 + iteration, "checkpoint_qualified": False,
            "array_sha256": sha(point_path.with_suffix(".npz")), "diagnostics": diagnostics,
            "state_sha256": state.state_sha256, "metric_sha256": state.residual_metric_sha256,
            "geometry_sha256": state.geometry_sha256, "objective_sha256": objective_identity,
            "owner": plain(ci.hamiltonian_snapshot.identity_payload()),
            "owner_input_sha256": ci.hamiltonian_snapshot.input_sha256,
            "owner_H_sha256": ci.hamiltonian_snapshot.matrix_sha256,
            "previous_qualified_checkpoint": previous["manifest_path"] if previous else None,
            "source_map_sha256": contract["source"]["candidate356_map_sha256"],
            "contract_sha256": sha(HERE / "contract.json"), "driver_sha256": sha(__file__)}
        write(point_path.with_suffix(".json"), manifest)
        flush(f"checkpoint_{iteration}_forensic_saved_before_gates")
        check(f"point{iteration}_same_metric", state.residual_metric_sha256 != initial.residual_metric_sha256)
        check(f"point{iteration}_same_geometry", state.geometry_sha256 != initial.geometry_sha256)
        check(f"point{iteration}_same_objective", state.objective_identity != objective_identity)
        check(f"point{iteration}_same_inverse_diagonal", state.inverse_diagonal.tobytes() != initial.inverse_diagonal.tobytes())
        for i, ((keys, projector, _, _), blocks) in enumerate(zip(state.layout, state.inverse_blocks, strict=True)):
            check(f"point{iteration}_P_bytes_{i}", projector.tobytes() != initial.layout[i][1].tobytes())
            check(f"point{iteration}_inverse_bytes_{i}", np.asarray(blocks).tobytes() != np.asarray(initial.inverse_blocks[i]).tobytes())
            c = np.column_stack([np.concatenate(orbitals[k]) for k in keys])
            s = functional.metric[keys[0][1]]
            check(f"point{iteration}_radial_Gram_{i}", np.max(abs(c.T @ s @ c - np.eye(len(keys)))), 1e-10)
            outside = c - projector @ (projector.T @ s @ c)
            check(f"point{iteration}_P_leakage_{i}", np.max(np.sum(outside * (s @ outside), axis=0)), 1e-12)
        check(f"point{iteration}_Hermiticity", np.max(abs(h - h.T)), 1e-8)
        _, blocks = ci_level_blocks(csfs, ci, require_full=True)
        for sector, cols in blocks.items():
            rows = np.array([i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector])
            row_set = set(rows)
            others = np.array([i for i in range(1046) if i not in row_set])
            check(f"point{iteration}_exact_H_cross_{sector}", np.max(abs(h[np.ix_(rows, others)])))
            check(f"point{iteration}_exact_C_support_{sector}", np.max(abs(ci.mixing_coefficients[np.ix_(others, cols)])))
            c = ci.mixing_coefficients[np.ix_(rows, cols)]
            hh = h[np.ix_(rows, rows)]
            check(f"point{iteration}_complete_CI_residual_{sector}", np.max(np.linalg.norm(
                hh @ c - c * ci.state_energies[cols], axis=0)), 1e-8)
            check(f"point{iteration}_complete_CI_Gram_{sector}", np.max(abs(c.T @ c - np.eye(len(cols)))), 1e-10)
        weights = functional.last_state_weights.copy()
        tracking = functional.last_tracking_diagnostics
        check(f"point{iteration}_equal5", not (np.count_nonzero(weights) == 5
              and np.array_equal(weights[weights > 0], np.repeat(.2, 5))))
        matched = target_bindings(targets, tracking)
        rows, columns, ids = [], [], []
        for target in targets:
            uid = target["twoJ"], target["parity"], target["models"]["DC"]["local_root"]
            binding = matched[uid]
            j, p, root = binding["selected_levels"][0]
            column = int(blocks[j, p][root])
            check(f"point{iteration}_UID_weight_{uid}", weights[column] != .2)
            columns.append(column)
            ids.append(json.dumps([target["label"], *uid], separators=(",", ":")))
            rows.append({"label": target["label"], "UID": list(uid), "column": column,
                         "selected_level": [j, p, root], "energy_Ha": float(ci.state_energies[column]),
                         "tracking": binding})
        check(f"point{iteration}_reported_EO", float(weights @ ci.state_energies) != diagnostics["energy_ha"])
        check(f"point{iteration}_captured_gradient_EO", captured["energy"] != diagnostics["energy_ha"])
        check(f"point{iteration}_original_residual_replay", abs(float(np.linalg.norm(pg)) - diagnostics["orbital_residual"]), 1e-12)
        check(f"point{iteration}_original_indicator_replay", abs(.5 * abs(float(g @ pg)) - diagnostics["estimated_energy_error"]), 1e-14)
        if iteration == 0:
            binding = load(path("baseline_source_binding"))
            check("baseline_corrected_owner_input", ci.hamiltonian_snapshot.input_sha256 != binding["owner_input_sha256"])
            check("baseline_corrected_owner_matrix", ci.hamiltonian_snapshot.matrix_sha256 != binding["owner_H_sha256"])
            check("baseline_line_metadata_none", diagnostics["accepted_line_search"] is not None)
            for key in ("H", "energies", "mixing", "P", "Q"):
                check("baseline_corrected_source_bitwise_" + key, arrays[key].tobytes() != continuation_baseline_arrays[key].tobytes())
            check("baseline_raw_coordinate_g_whole", np.max(abs(raw_g - saved_g["raw_coordinate_gradient"])), 1e-9)
            check("baseline_radial_g_whole", np.max(abs(arrays["radial_gradient"] - saved_g["radial_gradient"])), 1e-9)
            check("baseline_original_pg_whole", np.max(abs(pg - saved_g["preconditioned_gradient"])), 1e-10)
            check("baseline_corrected_owner_canonical", manifest["owner"] != plain(load(path("saved_owner"))["owner"]))
        full_density = {}
        for column in columns:
            for key, value in functional.radial_density.contract_bilinear(
                    ci.mixing_coefficients[:, column], ci.mixing_coefficients[:, column]).items():
                full_density[key] = math.fsum((full_density.get(key, 0.), weights[column] * value))
        write(point_path.with_name(point_path.name + "_signed_density.json"), {
            "unfiltered": [[plain(k), float(v)] for k, v in full_density.items()],
            "actual_gradient_retained": [[plain(k), float(v)] for k, v in functional.last_weights.items()],
            "original_gradient_coefficient_cutoff": 1e-15, "additional_H_CI_gradient_builders": 0,
            "saved_amplitude_angular_density_recontractions": 5})
        cert = guard.exact_reported_point(h, ci.mixing_coefficients[:, columns],
            ci.state_energies[columns], weights[columns], float(diagnostics["energy_ha"]),
            target_ids=ids, budget=exact_budget)
        decision = None if iteration == 0 else guard.reported_acceptance_decision(
            previous["certificate"], cert, diagnostics["accepted_line_search"])
        manifest.update(exact_point=cert.ledger(), exact_acceptance=decision,
                        exact_guard_budget=exact_budget.ledger())
        write(point_path.with_suffix(".json"), manifest)
        if decision is not None and decision["qualified"] is not True:
            raise QualificationStop("Unresolved accepted-point finite stored-H Armijo margin")
        by_label = physical_rows_by_label(rows)
        lower = by_label["6s_1/2"]["energy_Ha"]
        old_rows = previous["rows"] if previous else None
        old_by_label = physical_rows_by_label(old_rows) if old_rows else None
        for row in rows:
            row.update(excitation_cm1=(row["energy_Ha"] - lower) * HARTREE_CM1,
                       NIST_nominal_cm1=nominal[row["label"]], NIST_uncertainty_cm1=None)
            row["nominal_gap_cm1"] = row["excitation_cm1"] - row["NIST_nominal_cm1"]
            if old_rows:
                old = old_by_label[row["label"]]
                du, dl = row["energy_Ha"] - old["energy_Ha"], lower - old_by_label["6s_1/2"]["energy_Ha"]
                row.update(delta_upper_Ha=du, delta_lower_Ha=dl, transition_shift_cm1=(du - dl) * HARTREE_CM1)
        multiplets = []
        for prefix, a, b, w in (("5d", "5d_3/2", "5d_5/2", (4, 6)), ("6p", "6p_1/2", "6p_3/2", (2, 4))):
            multiplets.append({"label": prefix, "degeneracy_weights": w,
                "centroid_cm1": (w[0] * by_label[a]["excitation_cm1"] + w[1] * by_label[b]["excitation_cm1"]) / sum(w),
                "splitting_cm1": by_label[b]["excitation_cm1"] - by_label[a]["excitation_cm1"]})
        manifest.update(checkpoint_qualified=True, targets=rows, multiplets=multiplets,
            nominal_four_gap_RMS_cm1=nominal_four_gap_rms(rows),
            stationarity_passed=(diagnostics["orbital_residual"] < 1e-7
                                 and diagnostics["estimated_energy_error"] < 1e-8),
            experimental_certification=False, physical_completeness_certification=False)
        write(point_path.with_suffix(".json"), manifest)
        trajectory.append({"iteration": iteration, "global_accepted_step": 16 + iteration, "manifest_path": str(point_path.with_suffix(".json").relative_to(REPO)),
                           "manifest_sha256": sha(point_path.with_suffix(".json")),
                           "array_sha256": manifest["array_sha256"], "diagnostics": diagnostics,
                           "stationarity_passed": manifest["stationarity_passed"],
                           "nominal_four_gap_RMS_cm1": manifest["nominal_four_gap_RMS_cm1"]})
        previous = {"certificate": cert, "rows": rows, "manifest_path": trajectory[-1]["manifest_path"]}
        raw_identity()
        flush(f"checkpoint_{iteration}_qualified")

    status, error, diagnostics = "not_started", None, None
    try:
        _, _, diagnostics = optimization.optimize_orbitals(functional, waves, resume_state=initial,
            objective_identity=objective_identity, allow_objective_change=False, max_iter=64,
            tolerance=1e-7, energy_tolerance=1e-8, projector_source="nuclear", occupation_floor=.05,
            orbital_preconditioner="spectral", history_size=10, energy_difference=None,
            checkpoint_callback=checkpoint, max_wall_time=caps["optimizer_softwall_seconds"], return_state=False)
        status = "bounded_public_optimizer_returned"
    except (ResourceStop, QualificationStop, guard.ExactGuardBudgetExceeded) as failure:
        status = "resource_limited_last_qualified_point" if isinstance(failure, (ResourceStop, guard.ExactGuardBudgetExceeded)) else "first_unqualified_candidate_preserved"
        error = {"type": type(failure).__name__, "message": str(failure), "traceback": traceback.format_exc()}
    except Exception as failure:
        status = "unexpected_failure_last_qualified_point_preserved"
        error = {"type": type(failure).__name__, "message": str(failure), "traceback": traceback.format_exc()}
    finally:
        signal.alarm(0)
        result = {"status": status, "error": error, "controls": checks, "trajectory": trajectory,
            "attempted_calls": attempted, "completed_calls": completed,
            "last_qualified_checkpoint": previous["manifest_path"] if previous else None,
            "optimizer_final_diagnostics": diagnostics, "exact_guard_budget": exact_budget.ledger(),
            "source_map_sha256": contract["source"]["candidate356_map_sha256"],
            "driver_sha256": sha(__file__), "contract_sha256": sha(HERE / "contract.json"),
            "execution_sha256": execution["execution_file_sha256"], "resource_admission": admission,
            "original_reported_derivative_failure_preserved": True,
            "ordinary_reported_objective_changed": False, "energy_callback_used": False,
            "physical_completeness_certified": False, "experimental_certification": False,
            "stationarity_passed": bool(trajectory and trajectory[-1]["stationarity_passed"]),
            "initial_global_accepted_step": 16, "total_accepted_steps": 16 + max(0, len(trajectory) - 1),
            "terminal_point_selected_without_gap_optimization": True,
            "elapsed_uncontrolled_s": time.monotonic() - started_at,
            "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
        write(output / "result.json", result)
    if error is not None:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-contract", type=Path)
    parser.add_argument("--metadata", action="store_true")
    args = parser.parse_args()
    contract = load(HERE / "contract.json")
    if args.metadata:
        metadata(contract)
        print(json.dumps({"metadata_admitted": True, "science_calls": 0}))
        return
    if args.execution_contract is None:
        raise ValueError("Explicit exact root execution contract required")
    execution = load(args.execution_contract)
    execution["execution_file_sha256"] = sha(args.execution_contract)
    execute(contract, execution)


if __name__ == "__main__":
    main()
