"""Prospective native531 retained-H qualification; root execution seal required."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import resource
import signal
import sys
import time
import traceback
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TOPIC = ROOT / "docs/yb_mcdhf_rci_2026_10_11"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def json_numeric_scalar(value):
    """Only convert NumPy numerical scalar metadata, never arrays/physical objects."""
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        native = value.item()
        if type(native) in (bool, int, float):
            return native
    raise TypeError("Unsupported JSON metadata type: " + type(value).__name__)


def atomic_json(path, value):
    temporary = Path(str(path) + ".part")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=json_numeric_scalar)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def host_resources(caps):
    cpus = sorted(os.sched_getaffinity(0))
    available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:")) / 2**20
    loads = os.getloadavg()
    return {"available_CPUs": cpus, "chosen_CPU": cpus[-3] if len(cpus) >= 3 else None,
            "load_average": list(loads), "load1_per_available_CPU": loads[0] / len(cpus),
            "host_available_GiB": available,
            "admitted": len(cpus) >= 3 and available >= caps["minimum_host_available_GiB"]
            and loads[0] / len(cpus) <= caps["max_load1_per_available_CPU"],
            "controlled_performance_claim": False}


def source_and_input_checks(contract):
    checks = []
    for name, expected in contract["inputs_sha256"].items():
        checks.append({"name": "input:" + name, "passed": sha(ROOT / name) == expected})
    identity = load(ROOT / contract["candidate_source_identity_path"])
    source_path = ROOT / contract["candidate_frozen_source_path"]
    actual = {"src/" + str(path.relative_to(source_path)): sha(path)
              for path in sorted((source_path / "atomx").rglob("*.py"))}
    checks.extend([
        {"name": "complete354_source_files", "passed": len(actual) == 354},
        {"name": "actual_frozen_source_map", "passed": actual == identity["all_atomx_python_sha256"]},
        {"name": "canonical_candidate_source_map", "passed": canonical_sha(actual)
         == contract["candidate_source_map_required_sha256"] == identity["source_map_sha256"]},
    ])
    parent = load(TOPIC / "public_mixed_resume_run003/source_identity.json")
    old = parent["all_atomx_python_sha256"]
    changes = sorted(name for name in set(old) | set(actual) if old.get(name) != actual.get(name))
    checks.append({"name": "only_declared_numerical_retention_changes",
                   "passed": changes == sorted(contract["candidate_source_changes_allowed"])})
    return checks, actual


def preflight():
    """Hash/JSON/AST/zip admission only; never import the physical package."""
    contract = load(HERE / "contract.json")
    checks, _ = source_and_input_checks(contract)
    path = ROOT / contract["candidate_frozen_source_path"]
    owners = ast.parse((path / "atomx/physics/csf/hamiltonian_snapshot.py").read_text())
    ci = ast.parse((path / "atomx/methods/ci.py").read_text())
    functional = ast.parse((path / "atomx/methods/csf/variational.py").read_text())
    checks.extend([
        {"name": "typed_public_snapshot", "passed": any(isinstance(node, ast.ClassDef)
         and node.name == "CIHamiltonianSnapshot" for node in owners.body)},
        {"name": "public_physical_snapshot_producer", "passed": any(isinstance(node, ast.FunctionDef)
         and node.name == "build_ci_hamiltonian_snapshot" for node in owners.body)},
        {"name": "public_CI_retention_option", "passed": any(isinstance(node, ast.FunctionDef)
         and node.name == "ci_solve" and "retain_hamiltonian" in [arg.arg for arg in node.args.kwonlyargs]
         for node in ci.body)},
        {"name": "public_functional_retention_option", "passed": any(isinstance(node, ast.FunctionDef)
         and node.name == "__init__" and "retain_hamiltonian" in [arg.arg for arg in node.args.kwonlyargs]
         for node in ast.walk(functional))},
    ])
    run = TOPIC / "public_mixed_resume_run003"
    checkpoint = load(run / "checkpoints/accepted_0160.json")
    proof = load(run / "independent_endpoint_verification.json")
    checks.extend([
        {"name": "accepted320_exact_sourcebound_complete_point",
         "passed": checkpoint["accepted_complete_point"] and checkpoint["iteration"] == 160
         and checkpoint["parent_accepted_iteration"] == 160
         and checkpoint["array_sha256"] == contract["fixed_conditions"]["point_arrays_sha256"]},
        {"name": "independent1011_complete_endpoint_proof",
         "passed": proof["checks_count"] == 1011 and proof["all_identity_replay_checks_passed"]},
        {"name": "original_stationarity_failure_retained",
         "passed": proof["actual_original_metric_orbital_residual"] > 1e-7
         and proof["original_stationarity_gates_passed"] is False},
    ])
    with zipfile.ZipFile(run / "checkpoints/accepted_0160.npz") as stream:
        names = set(stream.namelist())
    checks.append({"name": "complete_point_gradient_geometry_CI_fields",
                   "passed": {name + ".npy" for name in (
                       "x", "P", "Q", "orbital_keys", "matrix_Ha", "energies_Ha", "mixing",
                       "original_projected_gradient", "original_preconditioned_gradient",
                       "inverse_diagonal")} <= names})
    host = host_resources(contract["resources"])
    checks.append({"name": "CPU1_3.95GiB_300s_feasibility", "passed": host["admitted"]})
    output = {"all_passed": all(item["passed"] for item in checks), "checks_count": len(checks),
              "checks": checks, "host": host, "contract_sha256": sha(HERE / "contract.json"),
              "driver_sha256": sha(Path(__file__)), "science_calls": 0,
              "actual_science_source_seal_pending_root_review": True}
    atomic_json(HERE / "preflight.json", output)
    if not output["all_passed"]:
        raise ValueError("Read-only preparation admission failed")
    print(json.dumps({"preflight_passed": True, "checks": len(checks), "host": host}, indent=2))


def validate_execution(approval, contract):
    """Reject missing/wrong authorization before imports or physical work."""
    required = {
        "execution_authorized_by_root": True,
        "qualification_authorized": True,
        "driver_sha256": sha(Path(__file__)),
        "preparation_contract_sha256": sha(HERE / "contract.json"),
        "candidate_source_identity_sha256": sha(ROOT / contract["candidate_source_identity_path"]),
        "candidate_source_map_sha256": contract["candidate_source_map_required_sha256"],
        "point_arrays_sha256": contract["fixed_conditions"]["point_arrays_sha256"],
        "max_full_gradients": 2,
        "max_CI_solves": 2,
        "max_public_H_builds": 3,
        "max_optimizer_calls": 0,
    }
    failed = [name for name, expected in required.items() if approval.get(name) != expected]
    if failed:
        raise ValueError("Missing exact root qualification authorization: " + ", ".join(failed))


def prepare_point(contract):
    """Build the exact public context only; no H build, CI solve, gradient or optimizer.

    The caller must first seal/import the declared candidate source epoch. The
    helper does not write qualification artifacts and is reusable by a separate
    source-bound performance consumer. Point construction is outside its timed
    energy-trial/gradient/archive chain unless the consumer declares otherwise.
    """
    import numpy as np

    from atomx.angular.csf_hamiltonian import csf_signature
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.ci import CIResult
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.physics.csf.tracking import CIBasisTracking
    from atomx.potentials.nuclear import make_nuclear_potential

    for name, module in sys.modules.items():
        if ((name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None)
                and not Path(module.__file__).resolve().is_relative_to(
                    (ROOT / contract["candidate_frozen_source_path"]).resolve())):
            raise ValueError("Point helper imported an undeclared source epoch")
    gates, fixed = contract["acceptance"], contract["fixed_conditions"]
    checks = []

    def check(name, passed, **detail):
        checks.append({"name": name, "passed": bool(passed), **detail})
        if not passed:
            raise ValueError("Public point setup failed: " + name)

    def close(name, error, tolerance):
        check(name, np.isfinite(error) and error <= tolerance,
              error=float(error), limit=tolerance)

    native = TOPIC / "native_run/nuclear_eol005_projected_increment"
    replay = TOPIC / "balanced_cv_replay"
    parent = TOPIC / "public_mixed_resume_run003"
    checkpoint = load(parent / "checkpoints/accepted_0160.json")
    endpoint_proof = load(parent / "independent_endpoint_verification.json")
    with np.load(parent / "checkpoints/accepted_0160.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    with np.load(native / "orbitals_workspace.npz", allow_pickle=False) as radial:
        data = {name[6:]: radial[name].item() if radial[name].ndim == 0 else radial[name].copy()
                for name in radial.files if name.startswith("basis_")}
        original_waves = {tuple(key): (p.copy(), q.copy()) for key, p, q in
                          zip(radial["orbital_keys"], radial["P"], radial["Q"], strict=True)}
    waves = {tuple(key): (p.copy(), q.copy()) for key, p, q in
             zip(arrays["orbital_keys"], arrays["P"], arrays["Q"], strict=True)}
    enumeration = load(replay / "basis_identity.json")
    csfs, records = [], []
    for config_index, row in enumerate(enumeration["configs"]):
        for csf in enumerate_jj_csfs(row["config"]):
            signature = csf_signature(csf)
            records.append({"index": len(csfs), "config_index": config_index, "family": row["family"],
                            "label": csf.label, "twoJ": csf.total_J2, "parity": csf.parity,
                            "subshells": [list(s) for s in signature.subshells],
                            "coupling": [list(c) for c in signature.coupling],
                            "canonical_pattern": [list(p) for p in signature.pattern]})
            csfs.append(csf)
    check("complete531_signed_CSF_order", records == enumeration["CSFs"]
          and len(csfs) == 531 and {csf.n_electrons for csf in csfs} == {69})
    old_objective = load(parent / "objective_identity.json")
    check("exact_old_actual_model_records", records == old_objective["new"]["CSFs"])
    targets = load(replay / "stage_4f5p5s/target_binding.json")["targets"]
    target_identity = [{"label": t["label"], "twoJ": t["twoJ"], "parity": t["parity"],
                        "root": t["models"]["DC"]["local_root"], "weight": .2} for t in targets]
    check("exact_original_targets_operator_nucleus", target_identity == old_objective["new"]["target_identity"]
          and old_objective["new"]["operator"] == "full declared no-pair Dirac-Coulomb"
          and old_objective["new"]["nucleus"] == "native005 Fermi A175")
    actual_model_payload = {**old_objective["new"], "source_map_sha256": contract["candidate_source_map_required_sha256"]}
    epoch_identity = {"physical_payload_except_source_unchanged": True,
                      "original_actual_objective_sha256": old_objective["new_sha256"],
                      "candidate_actual_objective_payload": actual_model_payload,
                      "candidate_actual_objective_sha256": canonical_sha(actual_model_payload)}
    average = CIStateAverage(tuple((t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.)
                                   for t in targets))
    nuclear = make_nuclear_potential(70, "fermi", A=175)
    with np.load(replay / "stage_4f5p5s/DC_arrays.npz", allow_pickle=False) as archive:
        baseline = CIResult(archive["energies_Ha"].copy(), archive["mixing"].copy())

    def functional(retain):
        temporary = CSFOrbitalFunctional(csfs, data, 70, nuclear, state_average=average,
                                         max_determinants=2000, retain_hamiltonian=retain)
        tracker = CIBasisTracking(average, csfs, baseline, original_waves, temporary.metric,
                                 max_determinant_pairs=1_000_000)
        return CSFOrbitalFunctional(csfs, data, 70, nuclear, state_average=average,
                                    level_tracking=tracker, cache=temporary.cache,
                                    max_determinants=2000, retain_hamiltonian=retain)

    ordinary, retained = functional(False), functional(True)
    layout = load(TOPIC / "native_run/nuclear_eol004_stable_delta_trace/failed_coordinate_layout.json")
    with np.load(TOPIC / "native_run/nuclear_eol004_stable_delta_trace/failed_coordinates.npz",
                 allow_pickle=False) as archive:
        old_metric = {name: archive[name].copy() for name in archive.files
                      if name == "inverse_diagonal" or name.startswith(("projector_", "inverse_blocks_"))}
    digest = hashlib.sha256()
    for i, row in enumerate(layout):
        keys, shape, where = [tuple(k) for k in row["keys"]], tuple(row["shape"]), slice(*row["slice"])
        projector = old_metric[f"projector_{i}"]
        digest.update(repr((keys, shape)).encode())
        digest.update(np.ascontiguousarray(projector).tobytes())
        check(f"original_P_identity:{i}", np.array_equal(projector, arrays[f"projector_{i}"]))
        c = projector @ arrays["x"][where].reshape(shape)
        actual = np.column_stack([np.concatenate(waves[key]) for key in keys])
        check(f"exact_originalP_times_archived_x:{i}", np.array_equal(c, actual))
        metric = ordinary.metric[keys[0][1]]
        close(f"original_radial_Gram:{i}", np.max(abs(c.T @ metric @ c - np.eye(len(keys)))),
              gates["radial_Gram_error"])
        outside = c - projector @ (projector.T @ metric @ c)
        close(f"original_no_pair_span:{i}", np.sqrt(max(0., np.max(np.sum(outside * (metric @ outside),
              axis=0)))), gates["projector_leakage_norm"])
    digest.update(old_metric["inverse_diagonal"].tobytes())
    for i in range(len(layout)):
        check(f"original_inverse_blocks_identity:{i}",
              np.array_equal(old_metric[f"inverse_blocks_{i}"], arrays[f"inverse_blocks_{i}"]))
        for block in old_metric[f"inverse_blocks_{i}"]:
            digest.update(np.ascontiguousarray(block).tobytes())
    check("original_inverse_diagonal_identity", np.array_equal(old_metric["inverse_diagonal"],
                                                               arrays["inverse_diagonal"]))
    check("complete_original_metric_byte_digest", digest.hexdigest() == fixed["original_metric_sha256"])

    def original_gradients(gradient):
        g, pg = np.zeros(2856), np.zeros(2856)
        for i, row in enumerate(layout):
            keys, shape, where = [tuple(k) for k in row["keys"]], tuple(row["shape"]), slice(*row["slice"])
            y = arrays["x"][where].reshape(shape)
            block = old_metric[f"projector_{i}"].T @ np.column_stack([gradient[key] for key in keys])
            multiplier = y.T @ block
            g[where] = (block - y @ (.5 * (multiplier + multiplier.T))).ravel()
            b = g[where].reshape(shape)
            z = np.column_stack([inverse @ b[:, col]
                                 for col, inverse in enumerate(old_metric[f"inverse_blocks_{i}"])])
            multiplier = y.T @ z
            pg[where] = (z - y @ (.5 * (multiplier + multiplier.T))).ravel()
        return g, pg

    return {"csfs": csfs, "waves": waves, "data": data, "nuclear": nuclear,
            "ordinary": ordinary, "retained": retained, "make_functional": functional,
            "archive": arrays, "metric": old_metric, "layout": layout,
            "checkpoint": checkpoint, "endpoint_proof": endpoint_proof,
            "original_gradients": original_gradients, "metric_sha256": digest.hexdigest(),
            "epoch_identity": epoch_identity, "setup_checks": checks}


def qualify(execution_contract):
    contract = load(HERE / "contract.json")
    approval = load(execution_contract)
    validate_execution(approval, contract)
    preliminary, actual_source = source_and_input_checks(contract)
    if not all(item["passed"] for item in preliminary):
        raise ValueError("Scientific source/input admission failed before physical imports")
    caps, gates, fixed = contract["resources"], contract["acceptance"], contract["fixed_conditions"]
    host = host_resources(caps)
    if not host["admitted"]:
        raise RuntimeError("Fresh host resource admission failed")
    if approval.get("chosen_CPU") != host["chosen_CPU"]:
        raise ValueError("Root chosen CPU does not match prospective available-affinity[-3]")
    atomic_json(HERE / "execution_resource_admission.json", host)
    os.sched_setaffinity(0, {host["chosen_CPU"]})
    limit = int(caps["address_space_cap_GiB"] * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    cache = HERE / "numba_cache"
    cache.mkdir(exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache)

    def timeout(*_):
        raise TimeoutError("Native531 retained-H qualification exceeded300s cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(caps["wall_cap_s"])
    start = time.monotonic()
    source_path = ROOT / contract["candidate_frozen_source_path"]
    sys.path.insert(0, str(source_path))
    import numpy as np

    import atomx.methods.csf.variational as functional_module
    import atomx.physics.csf.matrix as matrix_module
    from atomx.methods.ci import CIResult
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.physics.csf.hamiltonian_snapshot import CIHamiltonianSnapshot
    from atomx.physics.csf.levels import ci_level_blocks
    from atomx.potentials.nuclear import make_nuclear_potential

    checks = preliminary
    counters = {"started": {"H_build": 0, "CI_solve": 0, "gradient": 0},
                "completed": {"H_build": 0, "CI_solve": 0, "gradient": 0}}
    phases = []
    saved_arrays = {}

    def persist_controls(event, stage):
        atomic_json(HERE / "controls.json", {"checks": checks, "counters": counters,
                    "phase_records": phases, "event": event, "event_stage": stage,
                    "driver_sha256": sha(Path(__file__)), "contract_sha256": sha(HERE / "contract.json"),
                    "execution_contract_sha256": sha(execution_contract),
                    "qualification_completed": (HERE / "result.json").exists()})

    def save_arrays(event):
        temporary = HERE / "arrays.npz.part"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **saved_arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(HERE / "arrays.npz")
        persist_controls(event, "arrays_saved")

    persist_controls("scientific_imports_completed", "before_point_setup")

    def check(name, passed, **detail):
        persist_controls(name, "before_gate")
        checks.append({"name": name, "passed": bool(passed), **detail})
        persist_controls(name, "after_gate")
        if not passed:
            atomic_json(HERE / "failed_controls.json", {"checks": checks, "counters": counters,
                                                        "phase_records": phases})
            raise ValueError("Native531 qualification failed: " + name)

    def close(name, error, tolerance):
        check(name, np.isfinite(error) and error <= tolerance,
              error=float(error), limit=tolerance)

    def numerical_ci_checks(name, h, ci_result):
        shape_ok = (h.shape == (531, 531) and ci_result.state_energies.shape == (531,)
                    and ci_result.mixing_coefficients.shape == (531, 531)
                    and all(a.dtype == np.float64 and np.isfinite(a).all()
                            for a in (h, ci_result.state_energies, ci_result.mixing_coefficients)))
        check(name + "/complete_finite_arrays", shape_ok)
        close(name + "/all531_eigenpair_residual_Ha", np.max(np.linalg.norm(
            h @ ci_result.mixing_coefficients - ci_result.mixing_coefficients * ci_result.state_energies,
            axis=0)), gates["CI_residual_Ha"])
        close(name + "/all531_signed_Gram", np.max(abs(ci_result.mixing_coefficients.T
            @ ci_result.mixing_coefficients - np.eye(531))), gates["CI_orthogonality"])
        close(name + "/matrix_hermiticity", np.max(abs(h - h.T)), gates["matrix_hermiticity_Ha"])

    point = prepare_point(contract)
    checks.extend(point["setup_checks"])
    persist_controls("point_setup", "completed")
    csfs, waves, data, nuclear = (point[name] for name in ("csfs", "waves", "data", "nuclear"))
    ordinary, retained, arrays = (point[name] for name in ("ordinary", "retained", "archive"))
    checkpoint, endpoint_proof = point["checkpoint"], point["endpoint_proof"]
    original_gradients = point["original_gradients"]
    metric_sha256 = point["metric_sha256"]
    atomic_json(HERE / "objective_epoch_identity.json", point["epoch_identity"])

    raw_build = matrix_module.build_ci_matrix
    raw_ci = functional_module.ci_solve
    raw_gradient = CSFOrbitalFunctional.gradient_from_weights

    def counted(kind, original):
        def forward(*args, **kwargs):
            counters["started"][kind] += 1
            persist_controls(kind, "call_started")
            ceiling = {"H_build": 3, "CI_solve": 2, "gradient": 2}[kind]
            if counters["started"][kind] > ceiling:
                raise RuntimeError("Predeclared call cap exceeded: " + kind)
            result = original(*args, **kwargs)
            counters["completed"][kind] += 1
            persist_controls(kind, "call_completed")
            return result
        return forward

    matrix_module.build_ci_matrix = counted("H_build", raw_build)
    functional_module.ci_solve = counted("CI_solve", raw_ci)
    CSFOrbitalFunctional.gradient_from_weights = counted("gradient", raw_gradient)

    def phase(name):
        persist_controls(name, "before_phase_record")
        phases.append({"name": name, "counters": {stage: dict(values) for stage, values in counters.items()}})
        persist_controls(name, "after_phase_record")

    try:
        phase("before_ordinary_fresh_reference")
        ordinary_energy, ordinary_gradient, ordinary_ci = ordinary.evaluate(waves)
        check("ordinary_retention_false_has_no_snapshot", ordinary_ci.hamiltonian_snapshot is None)
        h_ordinary = matrix_module.build_ci_matrix(csfs, waves, ordinary.keys, data, 70, 123,
            nuclear_potential=nuclear, convol_cache=ordinary.cache, max_determinants=2000)
        phase("after_ordinary_reference_plus_independent_publicH_comparator")
        saved_arrays = {**arrays, "ordinary_Ha": h_ordinary,
                        "ordinary_energies_Ha": ordinary_ci.state_energies,
                        "ordinary_mixing": ordinary_ci.mixing_coefficients}
        save_arrays("ordinary_reference")
        check("ordinary_H_bitwise_equal_frozen320", np.array_equal(h_ordinary, arrays["matrix_Ha"]))
        numerical_ci_checks("ordinary", h_ordinary, ordinary_ci)
        retained_energy, trial = retained.evaluate(waves, gradient=False)
        phase("after_retained_energy_trial")
        check("public_typed_physical_owner_present", isinstance(trial.hamiltonian_snapshot, CIHamiltonianSnapshot))
        snapshot = trial.hamiltonian_snapshot
        kwargs = {"nuclear_potential": nuclear, "max_determinants": 2000}
        getter_args = (csfs, waves, retained.keys, data, 70, 123)
        before = dict(counters["completed"])
        h_retained = trial.hamiltonian_for(*getter_args, **kwargs)
        saved_arrays.update(retained_Ha=h_retained, retained_energies_Ha=trial.state_energies,
                            retained_mixing=trial.mixing_coefficients)
        save_arrays("retained_trial")
        atomic_json(HERE / "owner_header.json", {"identity_payload": snapshot.identity_payload(),
                    "input_sha256": snapshot.input_sha256, "matrix_sha256": snapshot.matrix_sha256,
                    "saved_array_sha256": sha(HERE / "arrays.npz"),
                    "scope": "Header-only archive; never unpickle/inject arbitrary H as a physical snapshot"})
        check("getter_returns_actual_owned_full_matrix", h_retained is snapshot.matrix_Ha)
        check("samepoint_getter_no_extra_H_or_CI", counters["completed"] == before)
        check("retained_H_bitwise_equal_ordinary_and_frozen320", np.array_equal(h_retained, h_ordinary)
              and np.array_equal(h_retained, arrays["matrix_Ha"]))
        numerical_ci_checks("retained_trial_before_gradient_reuse", h_retained, trial)
        close("all531_retained_vs_ordinary_energies", np.max(abs(trial.state_energies
              - ordinary_ci.state_energies)), gates["all_root_energy_retained_vs_ordinary_max_Ha"])
        close("all531_energy_replay_vs_frozen320", np.max(abs(trial.state_energies
              - arrays["energies_Ha"])), gates["all_root_energy_vs_frozen320_max_Ha"])
        phase("before_samepoint_CI_backed_gradient_completion")
        completed_energy, retained_gradient, completed_ci = retained.evaluate(waves, ci_result=trial)
        phase("after_samepoint_CI_backed_gradient_completion")
        check("completion_reuses_the_same_CI_owner", completed_ci is trial)
        check("trial_completion_adds_only_one_gradient",
              counters["completed"] == {"H_build": 3, "CI_solve": 2, "gradient": 2})
        close("same_EO_energy_ordinary_retained_completion", max(abs(ordinary_energy - retained_energy),
              abs(retained_energy - completed_energy)), 0.)
        check("complete_public_signed_RDM_key_order_exact",
              list(ordinary.last_weights) == list(retained.last_weights))
        close("all_public_signed_RDM_values_exact", max(abs(value - retained.last_weights[key])
              for key, value in ordinary.last_weights.items()), gates["complete_public_signed_RDM_retained_vs_ordinary_max"])
        rdm = [{"key": key, "value": float(value)} for key, value in ordinary.last_weights.items()]
        atomic_json(HERE / "signed_public_RDM.json", {"terms": rdm, "count": len(rdm),
                    "negative_terms": int(sum(row["value"] < 0 for row in rdm)),
                    "ordinary_retained_values_and_order_exact": True,
                    "scope": "All terms returned by existing public target-ensemble contraction, including signed interference. Existing angular coefficient retention convention unchanged; not a new unscreened-RDM implementation."})
        keys = ordinary.keys
        radial_g_ordinary = np.array([ordinary_gradient[key] for key in keys])
        radial_g_retained = np.array([retained_gradient[key] for key in keys])
        close("full_radial_gradient_retained_vs_ordinary", np.max(abs(radial_g_ordinary - radial_g_retained)),
              gates["full_radial_gradient_retained_vs_ordinary_max"])
        g_ordinary, pg_ordinary = original_gradients(ordinary_gradient)
        g_retained, pg_retained = original_gradients(retained_gradient)
        saved_arrays.update(radial_gradient_keys=np.array(keys), ordinary_radial_gradient=radial_g_ordinary,
                            retained_radial_gradient=radial_g_retained,
                            ordinary_projected_g=g_ordinary, ordinary_preconditioned_pg=pg_ordinary,
                            retained_projected_g=g_retained, retained_preconditioned_pg=pg_retained,
                            ordinary_state_weights=ordinary.last_state_weights,
                            retained_state_weights=retained.last_state_weights)
        for i, row in enumerate(point["layout"]):
            saved_arrays[f"actual_S_{i}"] = ordinary.metric[row["keys"][0][1]]
        save_arrays("complete_gradients_before_remaining_gates")
        for name, g, pg in (("ordinary", g_ordinary, pg_ordinary), ("retained", g_retained, pg_retained)):
            close(name + "/whole2856_g_vs_frozen320", np.max(abs(g - arrays["original_projected_gradient"])),
                  gates["complete_projected_coordinate_g_replay_max"])
            close(name + "/whole2856_pg_vs_frozen320", np.max(abs(pg - arrays["original_preconditioned_gradient"])),
                  gates["complete_preconditioned_pg_replay_max"])
        residual, indicator = float(np.linalg.norm(pg_retained)), .5 * abs(float(g_retained @ pg_retained))
        close("original_residual_matches1011proof", abs(residual
              - endpoint_proof["actual_original_metric_orbital_residual"]), 1e-10 + 1e-6 * residual)
        close("original_indicator_matches1011proof", abs(indicator
              - endpoint_proof["actual_original_metric_energy_indicator_Ha"]), 1e-12 + 1e-6 * indicator)
        check("original_stationarity_FAIL_preserved", residual > 1e-7 and indicator < 1e-8
              and endpoint_proof["original_stationarity_gates_passed"] is False)

        symmetry = [(c.total_J2, c.parity) for c in csfs]
        cross = np.array([[left != right for right in symmetry] for left in symmetry])
        close("complete_cross_symmetry_H_zero", np.max(abs(h_retained[cross])), gates["cross_symmetry_zero_Ha"])
        _, ordinary_columns = ci_level_blocks(csfs, ordinary_ci, require_full=True)
        _, retained_columns = ci_level_blocks(csfs, trial, require_full=True)
        _, saved_columns = ci_level_blocks(csfs, CIResult(arrays["energies_Ha"], arrays["mixing"]), require_full=True)
        cluster_records = []
        for key, columns in ordinary_columns.items():
            start_root = 0
            while start_root < len(columns):
                end_root = start_root + 1
                while end_root < len(columns) and (ordinary_ci.state_energies[columns[end_root]]
                        - ordinary_ci.state_energies[columns[end_root - 1]]
                        <= gates["numerical_cluster_span_gap_Ha"]):
                    end_root += 1
                old = ordinary_ci.mixing_coefficients[:, columns[start_root:end_root]]
                candidate = trial.mixing_coefficients[:, retained_columns[key][start_root:end_root]]
                saved = arrays["mixing"][:, saved_columns[key][start_root:end_root]]
                error = max(float(np.max(abs(old @ old.T - candidate @ candidate.T))),
                            float(np.max(abs(old @ old.T - saved @ saved.T))))
                close(f"same_signed_symmetry_root_span:{key}:{start_root}:{end_root}", error,
                      gates["same_signed_CI_span_projector_max"])
                cluster_records.append({"twoJ": key[0], "parity": key[1], "roots": [start_root, end_root],
                                        "span_max_error": error})
                start_root = end_root
        identities = retained.last_tracking_diagnostics
        def canonical_groups(groups):
            return sorted((tuple(tuple(level) for level in group["reference_levels"]),
                tuple(tuple(level) for level in group["selected_levels"]), group["per_level_weight"])
                for group in groups)
        check("same_original_frozen_physical_target_groups", canonical_groups(identities)
              == canonical_groups(checkpoint["physical_tracking"])
              == canonical_groups(ordinary.last_tracking_diagnostics))
        check("exact_five_equal_actual_state_weights", np.count_nonzero(retained.last_state_weights) == 5
              and np.array_equal(retained.last_state_weights, ordinary.last_state_weights)
              and np.array_equal(retained.last_state_weights[retained.last_state_weights > 0], np.repeat(.2, 5)))
        check("original_reference_majority_tracking_retained",
              all(min(group["principal_overlap_squared"]) > .5 for group in identities))

        # Controlled stale-input checks: each must reject before another physical call.
        stale_checks = []

        def reject(name, callback):
            count_before = {stage: dict(values) for stage, values in counters.items()}
            try:
                callback()
            except (ValueError, TypeError) as error:
                check("stale_reject:" + name, counters == count_before,
                      error_type=type(error).__name__, message=str(error))
                stale_checks.append({"name": name, "rejected": True, "message": str(error),
                                     "extra_physical_calls": 0})
            else:
                check("stale_reject:" + name, False)

        changed_waves = {key: (p.copy(), q.copy()) for key, (p, q) in waves.items()}
        changed_waves[keys[0]][0][1] = np.nextafter(changed_waves[keys[0]][0][1], np.inf)
        reject("one_bit_radial_point_getter", lambda: trial.hamiltonian_for(csfs, changed_waves,
               retained.keys, data, 70, 123, **kwargs))
        reject("one_bit_radial_point_CI_gradient_completion", lambda: retained.evaluate(
               changed_waves, ci_result=trial))
        changed_data = dict(data)
        changed_data["gr"] = data["gr"].copy()
        changed_data["gr"].flat[0] = np.nextafter(changed_data["gr"].flat[0], np.inf)
        reject("one_bit_workspace", lambda: trial.hamiltonian_for(csfs, waves, retained.keys,
               changed_data, 70, 123, **kwargs))
        reject("nucleus", lambda: trial.hamiltonian_for(*getter_args,
               nuclear_potential=make_nuclear_potential(70, "fermi", A=176), max_determinants=2000))
        reject("ordered_signed_CSF_basis", lambda: trial.hamiltonian_for(list(reversed(csfs)), waves,
               retained.keys, data, 70, 123, **kwargs))
        reject("ordered_orbital_basis", lambda: trial.hamiltonian_for(csfs, waves,
               tuple(reversed(retained.keys)), data, 70, 123, **kwargs))
        reject("determinant_cap", lambda: trial.hamiltonian_for(*getter_args,
               nuclear_potential=nuclear, max_determinants=1999))
        reject("Z", lambda: trial.hamiltonian_for(csfs, waves, retained.keys, data, 71, 123, **kwargs))
        reject("ns", lambda: trial.hamiltonian_for(csfs, waves, retained.keys, data, 70, 124, **kwargs))
        reject("Breit_operator", lambda: trial.hamiltonian_for(*getter_args, **kwargs, include_breit=True))
        reject("effective_shift_operator", lambda: trial.hamiltonian_for(*getter_args,
               **kwargs, one_body_shifts={}))
        reject("arbitrary_matrix_constructor", lambda: CIHamiltonianSnapshot(h_ordinary))
        reject("ordinary_result_no_owner_no_fallback", lambda: ordinary_ci.hamiltonian_for(*getter_args, **kwargs))
        immutable_before = sha_bytes = hashlib.sha256(h_retained.tobytes()).hexdigest()

        def make_writable():
            h_retained.setflags(write=True)

        reject("matrix_cannot_be_made_writable", make_writable)
        check("owned_matrix_bytes_preserved", hashlib.sha256(h_retained.tobytes()).hexdigest()
              == immutable_before == sha_bytes and not h_retained.flags.writeable)
        check("normalized_default_determinant_cap_matches", trial.hamiltonian_for(*getter_args,
              nuclear_potential=nuclear, max_determinants=None) is h_retained)
        header = snapshot.identity_payload()
        detached = snapshot.identity_payload()
        detached["max_determinants"] = 1
        check("detached_header_cannot_change_owner", snapshot.identity_payload() == header)
        changed_mixing = trial.mixing_coefficients.copy()
        changed_mixing[0, 0] += .01
        bad_residual = float(np.max(np.linalg.norm(h_retained @ changed_mixing
            - changed_mixing * trial.state_energies, axis=0)))
        bad_gram = float(np.max(abs(changed_mixing.T @ changed_mixing - np.eye(531))))
        check("separate_eigenpair_validator_rejects_mutable_CI_defect",
              bad_residual > gates["CI_residual_Ha"] or bad_gram > gates["CI_orthogonality"],
              saved_bad_residual_Ha=bad_residual, saved_bad_Gram=bad_gram,
              invalid_CI_passed_into_functional=False)
        phase("after_staleness_and_ownership_controls")
        check("actual_exact_call_budget", counters["started"] == counters["completed"]
              == {"H_build": 3, "CI_solve": 2, "gradient": 2})
        saved_arrays = {**arrays, "ordinary_Ha": h_ordinary, "retained_Ha": h_retained,
                        "ordinary_energies_Ha": ordinary_ci.state_energies,
                        "ordinary_mixing": ordinary_ci.mixing_coefficients,
                        "retained_energies_Ha": trial.state_energies, "retained_mixing": trial.mixing_coefficients,
                        "radial_gradient_keys": np.array(keys), "ordinary_radial_gradient": radial_g_ordinary,
                        "retained_radial_gradient": radial_g_retained, "ordinary_projected_g": g_ordinary,
                        "ordinary_preconditioned_pg": pg_ordinary, "retained_projected_g": g_retained,
                        "retained_preconditioned_pg": pg_retained, "ordinary_state_weights": ordinary.last_state_weights,
                        "retained_state_weights": retained.last_state_weights}
        for i, row in enumerate(point["layout"]):
            kappa = row["keys"][0][1]
            check(f"ordinary_retained_actual_S_identity:{i}",
                  np.array_equal(ordinary.metric[kappa], retained.metric[kappa]))
            saved_arrays[f"actual_S_{i}"] = ordinary.metric[kappa]
        destination = HERE / "arrays.npz"
        temporary = HERE / "arrays.npz.part"
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **saved_arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        atomic_json(HERE / "owner_header.json", {"identity_payload": header, "input_sha256": snapshot.input_sha256,
                    "matrix_sha256": snapshot.matrix_sha256, "saved_array_sha256": sha(destination),
                    "scope": "Header-only archive; never unpickle/inject arbitrary H as a physical snapshot"})
        rdm = [{"key": key, "value": float(value)} for key, value in ordinary.last_weights.items()]
        atomic_json(HERE / "signed_public_RDM.json", {"terms": rdm, "count": len(rdm),
                    "negative_terms": int(sum(row["value"] < 0 for row in rdm)),
                    "ordinary_retained_values_and_order_exact": True,
                    "scope": "All terms returned by existing public target-ensemble contraction, including signed interference. Existing angular coefficient retention convention unchanged; not a new unscreened-RDM implementation."})
        for name, expected in contract["inputs_sha256"].items():
            check("unchanged_after:" + name, sha(ROOT / name) == expected)
        for name, module in sorted(sys.modules.items()):
            if (name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None):
                check("actual_imported_epoch:" + name, Path(module.__file__).resolve().is_relative_to(source_path.resolve()))
        elapsed = time.monotonic() - start
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        check("actual_runtime_caps", elapsed < 300 and peak < 3.95 * 1024
              and sorted(os.sched_getaffinity(0)) == [host["chosen_CPU"]])
        result = {"qualification_numeric_passed": True, "checks_count": len(checks), "checks": checks,
                  "candidate_source_map_sha256": canonical_sha(actual_source),
                  "candidate_source_identity_sha256": sha(ROOT / contract["candidate_source_identity_path"]),
                  "contract_sha256": sha(HERE / "contract.json"),
                  "execution_contract_sha256": sha(execution_contract), "driver_sha256": sha(Path(__file__)),
                  "arrays_sha256": sha(destination), "owner_header_sha256": sha(HERE / "owner_header.json"),
                  "signed_RDM_sha256": sha(HERE / "signed_public_RDM.json"),
                  "phase_counters": phases, "actual_calls": counters,
                  "owner_input_sha256": snapshot.input_sha256, "owner_matrix_sha256": snapshot.matrix_sha256,
                  "original_metric_sha256": metric_sha256,
                  "point_arrays_sha256": fixed["point_arrays_sha256"],
                  "actual_original_orbital_residual": residual, "actual_original_energy_indicator_Ha": indicator,
                  "original_stationarity_passed": False, "production_certified": False,
                  "experimental_gate_passed": False, "highest_level_reached": False,
                  "optimizer_calls": 0, "new_spectrum_or_orbital_point": False,
                  "signed_cluster_span_controls": cluster_records, "physical_tracking": identities,
                  "stale_owner_controls": stale_checks,
                  "scope": "Actual native complete531 samepoint correctness qualification of opt-in H retention; no performance certificate or stationarity claim",
                  "resource_admission": host, "seconds_uncontrolled": elapsed, "peak_RSS_MiB": peak,
                  "controlled_performance_claim": False}
        atomic_json(HERE / "result.json", result)
        persist_controls("result_archived", "completed")
        print(json.dumps({"qualification_numeric_passed": True, "checks_count": len(checks),
                          "actual_calls": counters, "residual": residual, "indicator": indicator,
                          "seconds_uncontrolled": elapsed, "peak_RSS_MiB": peak}, indent=2))
    finally:
        matrix_module.build_ci_matrix = raw_build
        functional_module.ci_solve = raw_ci
        CSFOrbitalFunctional.gradient_from_weights = raw_gradient
        signal.alarm(0)
        persist_controls("scientific_wrapper_exit", "completed_or_failed_see_result_or_failure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "qualify"))
    parser.add_argument("--execution-contract", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
    else:
        if args.execution_contract is None:
            raise ValueError("Qualification requires an exact root execution contract")
        qualify(args.execution_contract)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        atomic_json(HERE / "failure.json", {"error": str(error), "traceback": traceback.format_exc(),
                    "driver_sha256": sha(Path(__file__)), "no_gate_or_model_repair_attempted": True})
        raise
