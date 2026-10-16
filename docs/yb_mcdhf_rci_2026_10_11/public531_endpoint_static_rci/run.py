"""One complete static-Breit RCI spectrum on the exact nonstationary public531 point.

Default preflight never imports Atomx. Actual execution requires root's exact
driver/contract/source/input authorization. No orbital gradient or optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as stream:
        stream.write(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def validate_identities(contract):
    for name, expected in contract["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    fixed = contract["fixed_conditions"]
    source = load(ROOT / fixed["source_identity"])
    source_root = ROOT / fixed["source_root"]
    actual = {"src/" + str(path.relative_to(source_root)): sha(path)
              for path in sorted((source_root / "atomx").rglob("*.py"))}
    if (actual != source["all_atomx_python_sha256"]
            or actual != contract["frozen_source_files_sha256"]
            or canonical_sha(actual) != fixed["source_map_sha256"]):
        raise ValueError("Complete source epoch changed")
    checkpoint, proof = load(ROOT / fixed["endpoint_checkpoint"]), load(ROOT / fixed["endpoint_verification"])
    objective = load(ROOT / fixed["endpoint_objective"])
    run_contract = load(ROOT / fixed["endpoint_run_contract"])
    expected = {
        "source_identity_sha256": sha(ROOT / fixed["source_identity"]),
        "preparation_contract_sha256": sha(ROOT / fixed["endpoint_run_contract"]),
        "execution_contract_sha256": sha(ROOT / fixed["endpoint_execution_contract"]),
    }
    if (checkpoint["accepted_complete_point"] is not True
            or checkpoint["array_sha256"] != sha(ROOT / fixed["endpoint_arrays"])
            or any(checkpoint.get(name) != value for name, value in expected.items())
            or checkpoint["frozen_scientific_inputs_sha256"] != run_contract["inputs_sha256"]
            or proof["all_identity_replay_checks_passed"] is not True
            or proof["accepted_checkpoint_sha256"] != sha(ROOT / fixed["endpoint_checkpoint"])
            or proof["accepted_arrays_sha256"] != sha(ROOT / fixed["endpoint_arrays"])
            or proof["objective_identity_file_sha256"] != sha(ROOT / fixed["endpoint_objective"])
            or proof["new_source_identity_sha256"] != expected["source_identity_sha256"]
            or proof["execution_contract_sha256"] != expected["execution_contract_sha256"]
            or proof["preparation_contract_sha256"] != expected["preparation_contract_sha256"]
            or canonical_sha(objective["new"]) != objective["new_sha256"]
            or objective["new_sha256"] != proof["independently_rebuilt_objective_sha256"]
            or objective["new_sha256"] != checkpoint["objective_sha256"]
            or checkpoint["original_metric_sha256"] != fixed["original_metric_sha256"]):
        raise ValueError("Endpoint is not the exact independently verified accepted point")
    inherited = contract["inherited_DC_stationarity_evidence"]
    if (inherited["independent_verification_sha256"] != sha(ROOT / fixed["endpoint_verification"])
            or inherited["orbital_residual"] != proof["actual_original_metric_orbital_residual"]
            or inherited["estimated_energy_error_Ha"] != proof["actual_original_metric_energy_indicator_Ha"]
            or inherited["stationarity_passed"] != proof["original_stationarity_gates_passed"]
            or proof["original_stationarity_gates_passed"] is not False):
        raise ValueError("Declared nonstationary fixed-point evidence changed")
    return checkpoint, proof


def host_resources(caps):
    affinity = sorted(os.sched_getaffinity(0))
    available = len(affinity)
    load_average = os.getloadavg()
    memory = {line.split(":")[0]: int(line.split()[1])
              for line in Path("/proc/meminfo").read_text().splitlines()}
    host_available = memory["MemAvailable"] * 1024
    cgroup = Path("/sys/fs/cgroup")
    if (cgroup / "memory.max").exists():
        maximum = (cgroup / "memory.max").read_text().strip()
        if maximum != "max":
            host_available = min(host_available, int(maximum) - int((cgroup / "memory.current").read_text()))
    enough_cpus = available >= caps["minimum_available_affinity_CPUs"]
    return {"original_accounting_affinity": affinity, "original_accounting_CPUs": available,
            "selected_CPU": affinity[-3] if enough_cpus else None,
            "load_average": load_average, "load1_per_available_CPU": load_average[0] / max(available, 1),
            "host_available_GiB": host_available / 1024**3,
            "admitted": enough_cpus and host_available / 1024**3 >= caps["minimum_host_available_GiB"]
            and load_average[0] / max(available, 1) <= caps["max_load1_per_available_CPU"],
            "purpose": "Resource feasibility; not a controlled performance comparison"}


def execution_authorized(approval, contract_path, driver_path, contract):
    return (approval.get("execution_authorized_by_root") is True
            and approval.get("driver_sha256") == sha(driver_path)
            and approval.get("preparation_contract_sha256") == sha(contract_path)
            and approval.get("source_identity_sha256") == sha(ROOT / contract["fixed_conditions"]["source_identity"])
            and approval.get("scientific_inputs_sha256") == contract["inputs_sha256"]
            and approval.get("source_map_sha256") == contract["fixed_conditions"]["source_map_sha256"]
            and approval.get("max_new_full_H_builds") == 1
            and approval.get("max_new_CI_spectrum_solves") == 1
            and approval.get("max_new_gradients") == 0
            and approval.get("max_optimizer_calls") == 0
            and approval.get("wall_cap_s") == 180
            and approval.get("address_space_cap_GiB") == 4
            and approval.get("threads") == 1
            and approval.get("affinity_selection") == "original_available_affinity[-3]")


def preflight(contract):
    checkpoint, proof = validate_identities(contract)
    basis = load(ROOT / contract["fixed_conditions"]["full_basis"])
    blocks = {}
    for row in basis["CSFs"]:
        key = row["twoJ"], row["parity"]
        blocks.setdefault(key, []).append(row["index"])
    if len(basis["CSFs"]) != 531 or len(blocks) != 16:
        raise ValueError("Complete declared model changed")
    previous = load(ROOT / "docs/yb_mcdhf_rci_2026_10_11/balanced_cv_replay/preflight.json")["stages"]["stage_4f5p5s"]
    if {tuple((row["twoJ"], row["parity"])): row["CSFs"] for row in previous["blocks"]} != {
            key: len(indices) for key, indices in blocks.items()}:
        raise ValueError("Complete J/parity block resource identity changed")
    maximum = max(row["raw_M_equals_J"] for row in previous["blocks"])
    if maximum > contract["resources"]["max_raw_determinants_per_block"]:
        raise MemoryError("Fixed complete model exceeds raw determinant cap")
    result = {"preparation_only": True, "contract_sha256": sha(HERE / "contract.json"),
              "driver_sha256": sha(Path(__file__)), "inputs_sha256": contract["inputs_sha256"],
              "source_files": len(contract["frozen_source_files_sha256"]),
              "source_map_sha256": contract["fixed_conditions"]["source_map_sha256"],
              "CSFs": 531, "blocks": [{"twoJ": key[0], "parity": key[1], "CSFs": len(indices)}
                                         for key, indices in sorted(blocks.items())],
              "max_raw_M_equals_J": maximum, "dense_full_matrix_bytes": 8 * 531**2,
              "endpoint_iteration": checkpoint["iteration"],
              "inherited_nonstationary_orbital_residual": proof["actual_original_metric_orbital_residual"],
              "inherited_DC_stationarity_passed": False,
              "resources": host_resources(contract["resources"]),
              "Atomx_imports": 0, "H_builds": 0, "CI_solves": 0, "gradients": 0, "optimizer_calls": 0}
    atomic_json(HERE / "preflight.json", result)
    return result


def execute(execution_contract):
    started = time.monotonic()
    contract = load(HERE / "contract.json")
    if execution_contract is None or not execution_contract.is_file():
        raise ValueError("An exact root execution contract is required before any Atomx import")
    approval = load(execution_contract)
    if not execution_authorized(approval, HERE / "contract.json", Path(__file__), contract):
        raise ValueError("Exact root static-RCI authorization failed before any Atomx import")
    if (HERE / "result.json").exists() or (HERE / "execution_source.py").exists():
        raise FileExistsError("Completed or failed evidence already exists; use a declared successor")
    checkpoint, endpoint_proof = validate_identities(contract)
    admission = host_resources(contract["resources"])
    atomic_json(HERE / "execution_resource_admission.json", admission)
    if not admission["admitted"]:
        raise RuntimeError("Resource admission failed before static-RCI")
    (HERE / "execution_source.py").write_bytes(Path(__file__).read_bytes())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.sched_setaffinity(0, {admission["selected_CPU"]})
    cap = int(contract["resources"]["address_space_cap_GiB"] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    def interrupted(*_):
        raise TimeoutError("Complete fixed-point static-RCI exceeded180s; preserve evidence")

    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(contract["resources"]["wall_cap_s"])
    fixed, gates = contract["fixed_conditions"], contract["acceptance"]
    sys.path.insert(0, str(ROOT / fixed["source_root"]))
    import numpy as np
    from scipy.linalg import eigh

    from atomx.angular.csf_hamiltonian import csf_signature
    from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
    from atomx.constants import HARTREE_CM1
    from atomx.methods.ci import CIResult
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.physics.csf.levels import ci_level_blocks
    from atomx.physics.csf.matrix import build_ci_matrix
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.physics.csf.tracking import CIBasisTracking
    from atomx.potentials.nuclear import make_nuclear_potential

    checks = []

    def check(name, error, limit):
        error = float(error)
        row = {"name": name, "error": error, "limit": limit,
               "passed": bool(np.isfinite(error) and error <= limit)}
        checks.append(row)
        if not row["passed"]:
            atomic_json(HERE / "failed_numeric_gate.json", {"checks": checks})
            raise ValueError("Fixed-point numeric/identity gate failed: " + name)

    basis = load(ROOT / fixed["full_basis"])
    csfs, records = [], []
    for config_index, config in enumerate(basis["configs"]):
        for csf in enumerate_jj_csfs(config["config"]):
            signature = csf_signature(csf)
            records.append({"index": len(csfs), "config_index": config_index, "family": config["family"],
                            "label": csf.label, "twoJ": csf.total_J2, "parity": csf.parity,
                            "subshells": plain(signature.subshells), "coupling": plain(signature.coupling),
                            "canonical_pattern": plain(signature.pattern)})
            csfs.append(csf)
    check("complete_signed531_order_and69e", len(csfs) != 531 or records != basis["CSFs"]
          or {c.n_electrons for c in csfs} != {69}, 0)
    saved = np.load(ROOT / fixed["endpoint_arrays"])
    radial = np.load(ROOT / fixed["original_tracker_waves"])
    workspace = {name[6:]: radial[name].item() if radial[name].ndim == 0 else radial[name]
                 for name in radial.files if name.startswith("basis_")}
    waves = {tuple(key): (p.copy(), q.copy()) for key, p, q in
             zip(saved["orbital_keys"], saved["P"], saved["Q"], strict=True)}
    original_waves = {tuple(key): (p.copy(), q.copy()) for key, p, q in
                      zip(radial["orbital_keys"], radial["P"], radial["Q"], strict=True)}
    check("complete24radial_pool_ns123", len(waves) != 24 or workspace["ns"] != 123
          or set(waves) != set(get_shared_orbital_basis(csfs)), 0)
    targets = load(ROOT / fixed["original_tracker_targets"])["targets"]
    selectors = tuple((t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.) for t in targets)
    average = CIStateAverage(selectors)
    nuclear = make_nuclear_potential(70, "fermi", A=175)
    # Geometry and public physical tracker only: no evaluate / gradient call.
    provisional = CSFOrbitalFunctional(csfs, workspace, 70, nuclear, state_average=average,
                                      max_determinants=2000)
    old = np.load(ROOT / fixed["original_tracker_CI"])
    tracking = CIBasisTracking(average, csfs, CIResult(old["energies_Ha"], old["mixing"]),
                              original_waves, provisional.metric, max_determinant_pairs=1_000_000)
    metric = np.load(ROOT / fixed["original_metric_arrays"])
    layout = load(ROOT / fixed["original_layout"])
    metric_digest = hashlib.sha256()
    for i, group in enumerate(layout):
        keys, shape, where = [tuple(k) for k in group["keys"]], tuple(group["shape"]), slice(*group["slice"])
        projector = metric[f"projector_{i}"]
        metric_digest.update(repr((keys, shape)).encode())
        metric_digest.update(np.ascontiguousarray(projector).tobytes())
        check(f"original_projector:{i}", np.max(abs(projector - saved[f"projector_{i}"])), 0)
        c = np.column_stack([np.concatenate(waves[key]) for key in keys])
        check(f"exact_authoritative_x_unpack:{i}", np.max(abs(c - projector @ saved["x"][where].reshape(shape))), 0)
        s = provisional.metric[keys[0][1]]
        check(f"radial_Gram:{i}", np.max(abs(c.T @ s @ c - np.eye(len(keys)))), gates["radial_Gram_error"])
        outside = c - projector @ (projector.T @ s @ c)
        check(f"original_no_pair:{i}", np.sqrt(max(0., np.max(np.sum(outside * (s @ outside), axis=0)))),
              gates["projector_leakage_norm"])
    metric_digest.update(metric["inverse_diagonal"].tobytes())
    check("original_inverse_diagonal", np.max(abs(saved["inverse_diagonal"] - metric["inverse_diagonal"])), 0)
    for i in range(len(layout)):
        check(f"original_inverse_blocks:{i}", np.max(abs(saved[f"inverse_blocks_{i}"] - metric[f"inverse_blocks_{i}"])), 0)
        for block in metric[f"inverse_blocks_{i}"]:
            metric_digest.update(np.ascontiguousarray(block).tobytes())
    check("complete_original_metric_SHA", metric_digest.hexdigest() != fixed["original_metric_sha256"], 0)
    dc = CIResult(saved["energies_Ha"], saved["mixing"])

    def spectrum_checks(label, matrix, ci):
        check(label + "/finite_complete_H_CI", matrix.shape != (531, 531)
              or ci.state_energies.shape != (531,) or ci.mixing_coefficients.shape != (531, 531)
              or any(not np.isfinite(a).all() for a in (matrix, ci.state_energies, ci.mixing_coefficients)), 0)
        check(label + "/residual_Ha", np.max(np.linalg.norm(matrix @ ci.mixing_coefficients
              - ci.mixing_coefficients * ci.state_energies, axis=0)), gates["CI_residual_Ha"])
        check(label + "/full_CI_Gram", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients - np.eye(531))),
              gates["CI_orthogonality"])
        check(label + "/Hermiticity", np.max(abs(matrix - matrix.T)), gates["matrix_hermiticity_Ha"])
        symmetry = [(c.total_J2, c.parity) for c in csfs]
        cross = np.array([[a != b for b in symmetry] for a in symmetry])
        check(label + "/cross_symmetry_zero", np.max(abs(matrix[cross])), gates["cross_symmetry_zero_Ha"])

    spectrum_checks("verified_saved_samepoint_DC", saved["matrix_Ha"], dc)
    dc_weights, dc_identity = tracking.resolve(csfs, dc, waves, provisional.metric)
    check("DC_five_equal_physical_targets", np.count_nonzero(dc_weights) != 5
          or not np.array_equal(dc_weights[dc_weights > 0], np.repeat(.2, 5)), 0)
    # Exactly one new public physical H build and one complete spectrum solve.
    h_started = time.monotonic()
    h = build_ci_matrix(csfs, waves, get_shared_orbital_basis(csfs), workspace, 70, 123,
                        nuclear_potential=nuclear, include_breit=True, breit_orbital_energies=None,
                        max_determinants=2000)
    h_seconds = time.monotonic() - h_started
    groups = {}
    for index, csf in enumerate(csfs):
        groups.setdefault((csf.total_J2, csf.parity), []).append(index)
    check("complete16symmetryblocks", len(groups) != 16, 0)
    values, columns = [], []
    solve_started = time.monotonic()
    for _, indices in sorted(groups.items(), key=lambda item: (item[0][0], -item[0][1])):
        e, v = eigh(h[np.ix_(indices, indices)], driver="evd", check_finite=True)
        full = np.zeros((531, len(indices)))
        full[indices] = v
        values.extend(e)
        columns.append(full)
    order = np.argsort(values, kind="stable")
    static = CIResult(np.asarray(values)[order], np.column_stack(columns)[:, order],
                      provenance={"physical_operator": "Atomx public complete static Breit Gaunt+zero-frequency retardation",
                                  "numerical_engine": "consumer scipy.linalg.eigh evd exact J/parity blocks",
                                  "n_blocks": len(groups), "n_csf": 531, "n_states": 531})
    solve_seconds = time.monotonic() - solve_started
    spectrum_checks("static_Breit", h, static)
    static_weights, static_identity = tracking.resolve(csfs, static, waves, provisional.metric)
    check("static_five_equal_physical_targets", np.count_nonzero(static_weights) != 5
          or not np.array_equal(static_weights[static_weights > 0], np.repeat(.2, 5)), 0)
    nominal = {row["label"]: row for row in load(ROOT / fixed["experimental_reference"])["levels"]}
    dc_verified_rows = {row["label"]: row for row in endpoint_proof["DC_lower_upper_rows"]}

    def model_rows(ci, identity):
        _, blocks = ci_level_blocks(csfs, ci, require_full=True)
        selected = {(j, parity): (root, int(blocks[j, parity][root]))
                    for group in identity for j, parity, root in group["selected_levels"]}
        lower_col = selected[(1, 1)][1]
        result = []
        for target in targets:
            root, col = selected[target["twoJ"], target["parity"]]
            energy = float(ci.state_energies[col])
            transition = float((energy - ci.state_energies[lower_col]) * HARTREE_CM1)
            ref = nominal[target["label"]]
            check(target["label"] + "/NIST_target_Jparity", (ref["two_J"], ref["parity"])
                  != (target["twoJ"], target["parity"]), 0)
            result.append({"label": target["label"], "twoJ": target["twoJ"], "parity": target["parity"],
                           "column": col, "local_root": root, "energy_Ha": energy, "excitation_cm1": transition,
                           "excitation_from_actual_finite_ground_cm1": float((energy - ci.state_energies[0]) * HARTREE_CM1),
                           "NIST_nominal_cm1": ref["excitation_cm1"], "NIST_uncertainty_cm1": ref["uncertainty_cm1"],
                           "nominal_gap_cm1": transition - ref["excitation_cm1"],
                           "nominal_relative_gap": (transition / ref["excitation_cm1"] - 1) if ref["excitation_cm1"] else None})
        return result, lower_col == 0

    dc_rows, dc_ground = model_rows(dc, dc_identity)
    static_rows, static_ground = model_rows(static, static_identity)
    dc_map, static_map = {r["label"]: r for r in dc_rows}, {r["label"]: r for r in static_rows}
    delta_lower = static_map["6s_1/2"]["energy_Ha"] - dc_map["6s_1/2"]["energy_Ha"]
    for row in static_rows:
        before = dc_map[row["label"]]
        check(row["label"] + "/DC_samepoint_verified_energy", abs(before["energy_Ha"]
              - dc_verified_rows[row["label"]]["energy_Ha"]), 5e-9)
        upper_delta = row["energy_Ha"] - before["energy_Ha"]
        transition_delta = row["excitation_cm1"] - before["excitation_cm1"]
        check(row["label"] + "/same_lower_upper_delta", abs(transition_delta
              - (upper_delta - delta_lower) * HARTREE_CM1), gates["lower_upper_delta_identity_cm1"])
        row.update(absolute_upper_static_minus_DC_Ha=upper_delta, same_lower_static_minus_DC_Ha=delta_lower,
                   static_minus_DC_transition_cm1=transition_delta,
                   nominal_1percent_pass=(abs(row["nominal_relative_gap"]) < .01
                                          if row["nominal_relative_gap"] is not None else None))
    multiplets = []
    for label, low, high in (("5d", "5d_3/2", "5d_5/2"), ("6p", "6p_1/2", "6p_3/2")):
        weights = [nominal[name]["two_J"] + 1 for name in (low, high)]
        entry = {"label": label, "degeneracy_weights": weights}
        for model, rows in (("DC", dc_map), ("static_Breit", static_map), ("NIST_nominal", nominal)):
            energies = [rows[name]["excitation_cm1"] for name in (low, high)]
            entry[model] = {"centroid_cm1": sum(w * e for w, e in zip(weights, energies, strict=True)) / sum(weights),
                            "splitting_cm1": energies[1] - energies[0]}
        for field in ("centroid_cm1", "splitting_cm1"):
            entry["static_minus_DC_" + field] = entry["static_Breit"][field] - entry["DC"][field]
            entry["DC_gap_" + field] = entry["DC"][field] - entry["NIST_nominal"][field]
            entry["static_gap_" + field] = entry["static_Breit"][field] - entry["NIST_nominal"][field]
        multiplets.append(entry)
    # Bind actually imported native modules and inputs before publication.
    source_root = (ROOT / fixed["source_root"]).resolve()
    loaded_modules = {}
    for name, module in tuple(sys.modules.items()):
        if name == "atomx" or name.startswith("atomx."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(source_root):
                raise ValueError("Loaded native module is outside the frozen source epoch")
            loaded_modules[name] = {"file": str(path), "sha256": sha(path)}
    validate_identities(contract)
    signed = dc.mixing_coefficients.T @ static.mixing_coefficients
    key_order = sorted(waves)
    arrays = {"static_matrix_Ha": h, "static_energies_Ha": static.state_energies,
              "static_mixing": static.mixing_coefficients, "DC_matrix_Ha": saved["matrix_Ha"],
              "DC_energies_Ha": dc.state_energies, "DC_mixing": dc.mixing_coefficients,
              "DC_to_static_signed_common_CSF_coefficient_overlap": signed,
              "x": saved["x"], "orbital_keys": np.array(key_order),
              "P": np.array([waves[k][0] for k in key_order]), "Q": np.array([waves[k][1] for k in key_order]),
              "DC_state_weights": dc_weights, "static_state_weights": static_weights}
    temporary = HERE / "arrays.npz.part"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(HERE / "arrays.npz")
    result = {"status": "nonstationary_fixed_point_complete531_static_RCI_diagnostic",
              "static_RCI_numeric_integrity_passed": all(row["passed"] for row in checks),
              "DC_original_stationarity_passed": False,
              "inherited_DC_stationarity_evidence": contract["inherited_DC_stationarity_evidence"],
              "checks_count": len(checks), "checks": checks, "DC_rows": dc_rows, "static_rows": static_rows,
              "multiplets": multiplets, "DC_tracking": dc_identity, "static_tracking": static_identity,
              "tracked6s_is_actual_finite_ground": {"DC": dc_ground, "static_Breit": static_ground},
              "static_minus_DC_same_lower_Ha": delta_lower, "CI_provenance": static.provenance,
              "nominal_four_NIST_gap_goal_passed": all(row["nominal_1percent_pass"] is not False for row in static_rows),
              "stationarity_certified": False, "production_certified": False, "experimental_gate_passed": False,
              "highest_level_reached": False, "frequency_dependent_Breit_included": False,
              "QED_recoil_included": False, "full_correlation_complete": False,
              "calls": {"full_H_builds": 1, "full_CI_spectrum_solves": 1, "new_DC_H_builds": 0,
                        "new_DC_CI_solves": 0, "gradients": 0, "optimizer_calls": 0},
              "timings_uncontrolled_s": {"static_H": h_seconds, "complete_blocked_CI": solve_seconds,
                                         "total": time.monotonic() - started},
              "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "resource_admission": admission, "actual_CPU_affinity": sorted(os.sched_getaffinity(0)),
              "contract_sha256": sha(HERE / "contract.json"), "driver_sha256": sha(Path(__file__)),
              "execution_contract_sha256": sha(execution_contract), "arrays_sha256": sha(HERE / "arrays.npz"),
              "source_map_sha256": fixed["source_map_sha256"], "inputs_sha256": contract["inputs_sha256"],
              "loaded_native_modules": loaded_modules, "frozen_inputs_verified_after_execution": True,
              "independent_saved_array_verification_pending": True, "performance_comparison_valid": False}
    atomic_json(HERE / "result.json", result)
    atomic_json(HERE / "card_evidence.json", {"method": "Nonstationary531 DC endpoint -> fixed-orbital static Breit RCI",
                "contract_sha256": sha(HERE / "contract.json"), "result_sha256": sha(HERE / "result.json"),
                "stationarity": contract["inherited_DC_stationarity_evidence"], "rows": static_rows,
                "multiplets": multiplets, "production_certified": False, "experimental_gate_passed": False,
                "highest_level_reached": False, "source_map_sha256": fixed["source_map_sha256"]})
    signal.alarm(0)
    print(json.dumps({"static_numeric_integrity": result["static_RCI_numeric_integrity_passed"],
                      "DC_stationarity_passed": False, "seconds_uncontrolled": result["timings_uncontrolled_s"]["total"],
                      "static_same_lower_delta_Ha": delta_lower}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "solve"), default="preflight", nargs="?")
    parser.add_argument("--execution-contract", type=Path)
    args = parser.parse_args()
    if args.mode == "preflight":
        answer = preflight(load(HERE / "contract.json"))
        print(json.dumps({k: answer[k] for k in ("preparation_only", "CSFs", "max_raw_M_equals_J", "resources")}))
    else:
        try:
            execute(args.execution_contract)
        except BaseException as error:
            atomic_json(HERE / "execution_failure.json", {"error": str(error), "traceback": traceback.format_exc(),
                        "stationarity_certified": False, "production_certified": False,
                        "experimental_gate_passed": False})
            raise


if __name__ == "__main__":
    main()
