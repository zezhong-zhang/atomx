"""Prepared original rotation / NEW normalized Rayleigh qualification, never an optimizer.

Admission requires a separate exact root execution seal before numerical
imports. The immutable preparation itself provides no execution authority.
"""
import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


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


def target_bindings(targets, tracking):
    """Bind physical state UIDs, independently of symmetry insertion order."""
    matched = {}
    for binding in tracking:
        if len(binding["reference_levels"]) != 1 or len(binding["selected_levels"]) != 1:
            raise ValueError("Expected singleton physical target")
        uid = tuple(binding["reference_levels"][0])
        if uid in matched or tuple(binding["selected_levels"][0][:2]) != uid[:2]:
            raise ValueError("Duplicate or changed physical target sector")
        matched[uid] = binding
    expected = {(t["twoJ"], t["parity"], t["models"]["DC"]["local_root"]) for t in targets}
    if len(expected) != 5 or set(matched) != expected:
        raise ValueError("Incomplete original five-target UID mapping")
    return matched


def bind_inputs(contract, execution):
    """Pure standard-library metadata admission; no numerical imports."""
    if contract["status"] != "prepared_requires_exact_root_execution":
        raise ValueError("Unprepared source/input admission")
    preparation_path = HERE / "preparation_verification.json"
    required = {"execution_authorized_by_root": True,
                "driver_sha256": sha(__file__), "contract_sha256": sha(HERE / "contract.json"),
                "preparation_sha256": sha(preparation_path),
                "source_map_sha256": contract["source"]["candidate356_map_sha256"],
                "new_scientific_target": "EOL_Rayleigh", "CPU": 31,
                "max_H": 3, "max_CI": 3, "max_full_gradient": 0, "max_optimizer": 0,
                "max_validation_Dirac": 7, "max_total_hS": 42,
                "wall_seconds": 120, "address_space_GiB": 3.95}
    if any(execution.get(k) != v for k, v in required.items()):
        raise ValueError("Wrong exact root execution binding")
    preparation = json.loads(preparation_path.read_text())
    if (preparation.get("all_passed") is not True
            or preparation.get("runner_sha256") != required["driver_sha256"]
            or preparation.get("contract_sha256") != required["contract_sha256"]):
        raise ValueError("Unqualified immutable preparation")
    for prerequisite in contract["pending_prerequisites"].values():
        if sha(REPO / prerequisite["path"]) != prerequisite["sha256"]:
            raise ValueError("Changed prerequisite proof")
    for role in contract["input_roles"].values():
        if sha(REPO / role["path"]) != role["sha256"]:
            raise ValueError("Changed scientific input")
    audit = json.loads((REPO / contract["input_roles"]["saved_gradient_audit"]["path"]).read_text())
    if (audit.get("all_saved_array_checks_passed") is not True
            or audit.get("original_probe_qualified") is not False
            or audit.get("whole_public_samepoint_gradient_replay_executed") is not False
            or audit.get("new_metric_sha256") != contract["metric"]["residual_metric_sha256"]):
        raise ValueError("Saved gradient/metric audit missing or wrong scope")
    for key, expected in (("screened002_independent_saved_proof", "all_independent_saved_array_checks_passed"),
                          ("guard003_actual_qualification", "guard_qualification_passed"),
                          ("guard003_independent_source_review", "source_review_all_passed")):
        proof = json.loads((REPO / contract["pending_prerequisites"][key]["path"]).read_text())
        if proof.get(expected) is not True:
            raise ValueError("Prerequisite gate failed")
    identity = json.loads((REPO / contract["input_roles"]["candidate_source_identity"]["path"]).read_text())
    if identity["source_map_sha256"] != contract["source"]["candidate356_map_sha256"]:
        raise ValueError("Changed full candidate source epoch")
    root = REPO / contract["source"]["candidate_archive"]
    for name, expected in identity["all_atomx_python_sha256"].items():
        if sha(root / name.removeprefix("src/")) != expected:
            raise ValueError("Frozen source byte mismatch")
    return root


def admit(contract, execution):
    root = bind_inputs(contract, execution)
    if any(os.environ.get(k) != "1" for k in (
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")):
        raise ValueError("Explicit single-thread process required")
    accounting = sorted(os.sched_getaffinity(0))
    if 31 not in accounting or os.getloadavg()[0] > execution["maximum_system_load1"]:
        raise ValueError("Host resource admission failed")
    resource.setrlimit(resource.RLIMIT_AS, (int(3.95 * 2**30), int(3.95 * 2**30)))
    os.sched_setaffinity(0, {31})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("120s qualification cap")))
    signal.alarm(120)
    if any(n == "atomx" or n.startswith("atomx.") for n in sys.modules):
        raise ValueError("No preimported physical source is permitted")
    sys.path.insert(0, str(root))
    return root, accounting


def probe(contract, execution):
    """Complete consumer; admission refuses without the exact root seal."""
    root, accounting = admit(contract, execution)
    import numpy as np
    from scipy import linalg as scipy_linalg

    import atomx.methods.csf.variational as variational
    import atomx.numerics.bspline.dirac_assembly as dirac
    import atomx.physics.csf.matrix as matrix
    import atomx.physics.csf.radial_increment as radial_increment
    import atomx.physics.csf.tracking as tracking_module
    from atomx.angular.csf_hamiltonian import csf_signature
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.ci import CIResult
    from atomx.numerics.bspline.orbital_optimization import (
        OrbitalOptimizationState,
        _polar_retraction,
    )
    from atomx.physics.csf.levels import ci_level_blocks
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.physics.csf.tracking import CIBasisTracking
    from atomx.potentials.nuclear import make_nuclear_potential

    for name, module in tuple(sys.modules.items()):
        if ((name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None)
                and not Path(module.__file__).resolve().is_relative_to(root.resolve())):
            raise ValueError("Undeclared physical source import")
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    limits = {"H": 3, "CI": 3, "gradient": 0, "validation_Dirac": 7,
              "initial_hS": 7, "CI_internal_hS": 21, "mixed_hS": 14,
              "tracking_resolution": 10, "tracking_SVD": 50,
              "tracking_overlap": 11, "geometry_eigvalsh": 36,
              "scipy_eigh": 36, "polar_retraction": 14,
              "polar_numpy_eigh": 14}
    started = dict.fromkeys(limits, 0)
    completed = dict(started)
    controls, records, raw_hS = [], [], []
    begin = time.monotonic()

    def flush(phase):
        write(output / "progress.json", {"phase": phase, "started": started, "completed": completed,
              "controls": controls, "records": records, "elapsed_uncontrolled_s": time.monotonic() - begin})

    def check(name, error, limit=0.):
        passed = bool(np.isfinite(error) and error <= limit)
        controls.append({"name": name, "error": float(error), "limit": limit, "passed": passed})
        flush(name)
        if not passed:
            raise ValueError("Scientific gate failed: " + name)

    def wrap(name, fn, capture=False):
        def call(*args, **kwargs):
            started[name] += 1
            flush("before_" + name)
            if started[name] > limits[name]:
                raise RuntimeError("Attempted call cap: " + name)
            result = fn(*args, **kwargs)
            completed[name] += 1
            if capture:
                raw_hS.append((name, int(args[1]), result[0].copy(), result[1].copy()))
                np.savez_compressed(output / f"native_raw_hS_{len(raw_hS)-1:03d}.npz",
                                    kappa=np.array(int(args[1])), raw_h=result[0], S=result[1])
            flush("after_" + name)
            return result
        return call

    original_builder = dirac.build_dirac_matrices
    variational.build_dirac_matrices = wrap("initial_hS", original_builder, True)
    matrix.build_dirac_matrices = wrap("CI_internal_hS", original_builder, True)
    radial_increment.build_dirac_matrices = wrap("mixed_hS", original_builder, True)
    dirac.solve_dirac = wrap("validation_Dirac", dirac.solve_dirac)
    matrix.build_ci_matrix = wrap("H", matrix.build_ci_matrix)
    variational.ci_solve = wrap("CI", variational.ci_solve)
    variational.CSFOrbitalFunctional.gradient_from_weights = wrap(
        "gradient", variational.CSFOrbitalFunctional.gradient_from_weights)
    CIBasisTracking.resolve = wrap("tracking_resolution", CIBasisTracking.resolve)
    tracking_module.ci_subspace_overlaps = wrap("tracking_overlap", tracking_module.ci_subspace_overlaps)
    np.linalg.svd = wrap("tracking_SVD", np.linalg.svd)
    np.linalg.eigvalsh = wrap("geometry_eigvalsh", np.linalg.eigvalsh)
    scipy_linalg.eigh = wrap("scipy_eigh", scipy_linalg.eigh)
    np.linalg.eigh = wrap("polar_numpy_eigh", np.linalg.eigh)
    _polar_retraction = wrap("polar_retraction", _polar_retraction)

    def path(role):
        return REPO / contract["input_roles"][role]["path"]

    def load(role):
        return json.loads(path(role).read_text())

    def archive(role):
        with np.load(path(role), allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}

    def waves_from(values, keyname):
        return {tuple(map(int, key)): (p.copy(), q.copy()) for key, p, q in
                zip(values[keyname], values["P"], values["Q"], strict=True)}

    saved, geometry, saved_point = archive("checkpoint_arrays"), archive("screened_geometry"), archive("saved_point")
    saved_gradient = archive("saved_gradient")
    radial = archive("geometry_workspace")
    data = {k[6:]: v.item() if v.ndim == 0 else v.copy() for k, v in radial.items() if k.startswith("basis_")}
    saved_waves = waves_from(saved_point, "keys")
    old_keys = set(map(tuple, saved["orbital_keys"]))
    new_keys = set(saved_waves) - old_keys
    models = load("screened_models")["models"]
    full_old = [csf for row in models["baseline"]["configs"] for csf in enumerate_jj_csfs(row["config"])]
    full_new = [csf for row in models["balanced_virtual"]["configs"] for csf in enumerate_jj_csfs(row["config"])]
    sectors = {(j, p) for j, p, _ in contract["fixed"]["target_blocks"]}
    selected = [c for c in full_new if (c.total_J2, c.parity) in sectors]
    check("full_old531", abs(len(full_old) - 531))
    check("full_parent2163", abs(len(full_new) - 2163))
    check("selected_complete1046", abs(len(selected) - 1046))
    check("same69electrons", {c.n_electrons for c in selected} != {69})
    for j, p, count in contract["fixed"]["target_blocks"]:
        check(f"complete_sector_{j}_{p}", abs(sum((c.total_J2, c.parity) == (j, p) for c in selected) - count))
    signatures = [json.dumps(plain((s.subshells, s.coupling, c.total_J2, c.parity)), separators=(",", ":"))
                  for c in selected for s in [csf_signature(c)]]
    check("same1046signed_order", signatures != load("screened_basis")["balanced_virtual"]["retained_signatures"])
    targets = load("tracking_binding")["targets"]
    average = CIStateAverage(tuple((t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.) for t in targets))
    nuclear = make_nuclear_potential(70, "fermi", A=175)
    reference = archive("tracking_ci")
    original_metric = {k: geometry[f"s_{k}"] for k in set(key[1] for key in saved_waves)}
    tracker = CIBasisTracking(average, full_old, CIResult(reference["energies_Ha"], reference["mixing"]),
                             waves_from(radial, "orbital_keys"), original_metric,
                             max_determinant_pairs=1_000_000)
    functional = variational.CSFOrbitalFunctional(selected, data, 70, nuclear,
                                                 state_average=average, level_tracking=tracker,
                                                 retain_hamiltonian=True, max_determinants=2000)
    for kind, k, h, metric in raw_hS:
        if kind == "initial_hS":
            logical = np.tril(h) + np.tril(h, -1).T
            check(f"logical_h_{k}", not np.array_equal(logical, functional.one_body[k]))
            check(f"original_logical_h_{k}", not np.array_equal(logical, geometry[f"h_{k}"]))
            check(f"original_S_{k}", not np.array_equal(metric, geometry[f"s_{k}"]))

    raw_reference = {k: (h, s) for kind, k, h, s in raw_hS if kind == "initial_hS"}

    raw_verified = 0

    def raw_identity():
        nonlocal raw_verified
        for index in range(raw_verified, len(raw_hS)):
            kind, k, h, metric = raw_hS[index]
            reference_h, reference_s = raw_reference[k]
            check(f"captured_raw_h_identity_{kind}_{index}", not (np.array_equal(h, reference_h)
                  and h.tobytes() == reference_h.tobytes()))
            check(f"captured_S_identity_{kind}_{index}", not (np.array_equal(metric, reference_s)
                  and metric.tobytes() == reference_s.tobytes()))
        raw_verified = len(raw_hS)

    original_projectors = {tuple(row["keys"][0])[1]: saved[f"projector_{i}"]
                           for i, row in enumerate(load("original_layout"))}

    def complete_point(name, waves):
        result = functional.evaluate(waves, gradient=False)
        energy, ci = (result[0], result[-1])
        h = ci.hamiltonian_for(selected, waves, functional.keys, data, 70, 123,
                              nuclear_potential=nuclear, max_determinants=2000)
        np.savez_compressed(output / (name + ".npz"), H=h, energies=ci.state_energies,
                            mixing=ci.mixing_coefficients, keys=np.array(functional.keys),
                            P=np.array([waves[k][0] for k in functional.keys]),
                            Q=np.array([waves[k][1] for k in functional.keys]))
        write(output / (name + "_owner.json"), ci.hamiltonian_snapshot.identity_payload())
        raw_identity()
        check(name + "_whole_H_exact_symmetry", not np.array_equal(h, h.T))
        for kappa, metric in functional.metric.items():
            keys = [k for k in functional.keys if k[1] == kappa]
            c = np.column_stack([np.concatenate(waves[k]) for k in keys])
            check(f"{name}_radial_Gram_{kappa}", np.max(abs(c.T @ metric @ c - np.eye(len(keys)))), 1e-10)
            projector = original_projectors[kappa]
            outside = c - projector @ (projector.T @ metric @ c)
            check(f"{name}_P_leakage_{kappa}", np.max(np.sum(outside * (metric @ outside), axis=0)), 1e-12)
        check(name + "_all_CI_residual", np.max(np.linalg.norm(h @ ci.mixing_coefficients -
              ci.mixing_coefficients * ci.state_energies, axis=0)), 1e-8)
        check(name + "_all_CI_Gram", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients - np.eye(1046))), 1e-10)
        weights, tracking = tracker.resolve(selected, ci, waves, functional.metric)
        check(name + "_equal5", not (np.count_nonzero(weights) == 5 and np.array_equal(weights[weights > 0], np.repeat(.2, 5))))
        _, blocks = ci_level_blocks(selected, ci, require_full=True)
        rows = []
        matched = target_bindings(targets, tracking)
        check(name + "_complete_unique_original_physical_UIDs", False)
        for target in targets:
            uid = target["twoJ"], target["parity"], target["models"]["DC"]["local_root"]
            binding = matched[uid]
            j, p, local = binding["selected_levels"][0]
            check(name + "_selected_physical_sector_" + target["label"], (j, p) != uid[:2])
            column = int(blocks[j, p][local])
            rows.append({"label": target["label"], "UID": binding["reference_levels"][0],
                         "column": column, "reported_energy_Ha": float(ci.state_energies[column]),
                         "tracking": binding})
            c = ci.mixing_coefficients[:, column].astype(np.longdouble)
            lam = np.longdouble(ci.state_energies[column])
            centered = h.astype(np.longdouble)
            centered.flat[::len(h) + 1] -= lam
            residual_vector = centered @ c
            norm = c @ c
            correction = c @ residual_vector / norm
            rows[-1].update(CI_norm=str(norm), quotient_minus_reported_Ha=float(correction),
                            normalized_Rayleigh_Ha=str(lam + correction),
                            normalized_reported_residual_Ha=float(np.sqrt(residual_vector @ residual_vector / norm)))
        energies = {row["label"]: row["reported_energy_Ha"] for row in rows}
        rayleigh_energies = {row["label"]: np.longdouble(row["normalized_Rayleigh_Ha"]) for row in rows}
        ground = energies["6s_1/2"]
        from atomx.constants import HARTREE_CM1
        for row in rows:
            row["reported_transition_cm1"] = (row["reported_energy_Ha"] - ground) * HARTREE_CM1
            row["normalized_Rayleigh_transition_cm1"] = float((rayleigh_energies[row["label"]] -
                rayleigh_energies["6s_1/2"]) * np.longdouble(HARTREE_CM1))
        for prefix, weights2 in (("5d", (4, 6)), ("6p", (2, 4))):
            low, high = energies[prefix + "_" + ("3/2" if prefix == "5d" else "1/2")], energies[prefix + "_" + ("5/2" if prefix == "5d" else "3/2")]
            rayleigh_low = rayleigh_energies[prefix + "_" + ("3/2" if prefix == "5d" else "1/2")]
            rayleigh_high = rayleigh_energies[prefix + "_" + ("5/2" if prefix == "5d" else "3/2")]
            rows.append({"multiplet": prefix, "reported_centroid_cm1":
                         ((weights2[0] * low + weights2[1] * high) / sum(weights2) - ground) * HARTREE_CM1,
                         "reported_splitting_cm1": (high - low) * HARTREE_CM1,
                         "normalized_Rayleigh_centroid_cm1": float(((weights2[0] * rayleigh_low +
                           weights2[1] * rayleigh_high) / sum(weights2) - rayleigh_energies["6s_1/2"]) *
                           np.longdouble(HARTREE_CM1)),
                         "normalized_Rayleigh_splitting_cm1": float((rayleigh_high - rayleigh_low) *
                           np.longdouble(HARTREE_CM1))})
        records.append({"name": name, "reported_EO_Ha": energy,
                        "normalized_Rayleigh_EO_Ha": str(sum((np.longdouble(r["normalized_Rayleigh_Ha"]) / 5
                              for r in rows if "label" in r), np.longdouble(0))),
                        "observable": "EOL_Rayleigh", "targets": rows})
        flush(name + "_complete")
        return energy, ci, h

    # Consume the already audited exact new layout and point; no canonical seed replay.
    layout_rows = load("saved_layout")
    layout = [(tuple(map(tuple, row["keys"])), saved_gradient[f"projector_{i}"],
               tuple(row["shape"]), slice(*row["slice"]))
              for i, row in enumerate(layout_rows)]
    check("original_exact_new_keys", new_keys != set(map(tuple, contract["direction"]["new_keys"])))
    check("saved3451coordinates", abs(saved_gradient["x"].size - 3451))
    new_objective = "EOL_Rayleigh:contract_sha256=" + sha(HERE / "contract.json")
    state = OrbitalOptimizationState.from_arrays(functional, layout=layout,
        coordinates=saved_gradient["x"], inverse_diagonal=saved_gradient["inverse_diagonal"],
        inverse_blocks=[saved_gradient[f"inverse_blocks_{i}"] for i in range(7)],
        projector_source="nuclear", objective_identity=new_objective, occupation_floor=.05)
    check("same3451_original_metric", state.residual_metric_sha256 != contract["metric"]["residual_metric_sha256"])
    check("same_original_geometry", state.geometry_sha256 != contract["metric"]["geometry_sha256"])
    check("saved_x_bytes", state.coordinates.tobytes() != saved_gradient["x"].tobytes())
    check("saved_inverse_diagonal_bytes", state.inverse_diagonal.tobytes() != saved_gradient["inverse_diagonal"].tobytes())
    for i, (_, projector, _, _) in enumerate(state.layout):
        check(f"state_P_bytes_{i}", projector.tobytes() != saved_gradient[f"projector_{i}"].tobytes())
        check(f"state_inverse_blocks_bytes_{i}", np.asarray(state.inverse_blocks[i]).tobytes() !=
              saved_gradient[f"inverse_blocks_{i}"].tobytes())
        check(f"old7_P_bytes_{i}", projector.tobytes() != saved[f"projector_{i}"].tobytes())

    def unpack(coordinates):
        waves = {}
        for keys, projector, shape, where in state.layout:
            c = projector @ coordinates[where].reshape(shape)
            for j, key in enumerate(keys):
                waves[key] = c[:123, j].copy(), c[123:, j].copy()
        return waves

    def tangent(coordinates, vector):
        result = vector.copy()
        for _, _, shape, where in state.layout:
            y, g0 = coordinates[where].reshape(shape), vector[where].reshape(shape)
            a = y.T @ g0
            result[where] = (g0 - .5 * y @ (a + a.T)).ravel()
        return result

    x, waves = state.coordinates, unpack(state.coordinates)
    for key in functional.keys:
        check("exact_saved_recovered_waves_" + str(key), not all(
            a.tobytes() == b.tobytes() for a, b in zip(waves[key], saved_waves[key], strict=True)))
    np.savez_compressed(output / "same_point_coordinates_metric.npz", x=x,
                        inverse_diagonal=state.inverse_diagonal,
                        keys=np.array(functional.keys), tangent_gradient=saved_gradient["tangent_gradient"],
                        preconditioned_gradient=saved_gradient["preconditioned_gradient"],
                        **{f"projector_{i}": row[1] for i, row in enumerate(state.layout)},
                        **{f"inverse_blocks_{i}": np.asarray(v) for i, v in enumerate(state.inverse_blocks)})
    write(output / "point_identity.json", {"new_scientific_target": "EOL_Rayleigh",
          "new_objective_declaration": state.objective_identity, "state_sha256": state.state_sha256,
          "old_reported_objective_state_sha256": contract["metric"]["old_point_state_sha256"],
          "new_metric_sha256": state.residual_metric_sha256, "geometry_sha256": state.geometry_sha256,
          "same_saved_coordinates_and_metric_bytes": True, "baseline_polar_repair": False,
          "fresh_native_baseline_required": True, "objective_identity_is_caller_declaration": True})
    base = complete_point("fresh_recovered_coordinate_point", waves)
    for name, value in (("H", base[2]), ("energies", base[1].state_energies),
                        ("mixing", base[1].mixing_coefficients)):
        check("fresh_baseline_bitwise_" + name,
              not (np.array_equal(value, saved_point[name]) and value.tobytes() == saved_point[name].tobytes()))
    check("fresh_baseline_real_owner_identity", base[1].hamiltonian_snapshot.identity_payload() != load("saved_point_owner"))
    g, pg = saved_gradient["tangent_gradient"], saved_gradient["preconditioned_gradient"]
    residual, indicator = float(np.linalg.norm(pg)), .5 * abs(float(g @ pg))
    write(output / "saved_gradient_scope.json", {"gradient_new_calls": 0,
          "saved_gradient_audit_sha256": contract["input_roles"]["saved_gradient_audit"]["sha256"],
          "conditional_on_saved_public_radial_gradient": True, "whole_public_gradient_replayed": False,
          "new3451_orbital_residual": residual, "energy_indicator_Ha": indicator,
          "stationarity_gates_passed": residual < 1e-7 and indicator < 1e-8,
          "orbital_tolerance": 1e-7, "energy_tolerance_Ha": 1e-8})

    # Exactly the unexecuted original direction1 generator, never chosen from g.
    rng = np.random.Generator(np.random.PCG64(20261012))
    direction = np.zeros(3451)
    for keys, _, shape, where in state.layout:
        y = x[where].reshape(shape)
        omega = np.zeros((shape[1], shape[1]))
        for i, a in enumerate(keys):
            for j in range(i + 1, len(keys)):
                if (a in new_keys) != (keys[j] in new_keys):
                    omega[i, j] = rng.standard_normal()
                    omega[j, i] = -omega[i, j]
        direction[where] = (y @ omega).ravel()
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-14:
        raise ValueError("Inconclusive original rotation; no replacement direction or h scan")
    direction /= norm
    np.savez_compressed(output / "direction1_coordinates.npz", direction=direction)
    check("direction1_shared_tangent", np.max(abs(tangent(x, direction) - direction)), 1e-12)
    analytical = float(g @ direction)
    trials, diagnostics = [], []
    for sign in (1., -1.):
        candidate = x + sign * 1e-6 * direction
        for _, _, shape, where in state.layout:
            candidate[where] = _polar_retraction(candidate[where].reshape(shape)).ravel()
        trial_waves = unpack(candidate)
        increment = unpack(candidate - x)
        name = "direction1_" + ("plus" if sign > 0 else "minus")
        np.savez_compressed(output / (name + "_coordinates.npz"), candidate_x=candidate,
                            explicit_increment_P=np.array([increment[k][0] for k in functional.keys]),
                            explicit_increment_Q=np.array([increment[k][1] for k in functional.keys]))
        trial = complete_point(name, trial_waves)
        diagnostic = functional.mixed_ci_energy_difference(waves, trial_waves, base[1], trial[1],
                                                            increments=increment)
        raw_identity()
        check(name + "_diagnostic_not_optimizer_certificate", not (
              diagnostic["diagnostic_only"] is True and diagnostic["certified_error_bound"] is False
              and diagnostic["optimizer_adapter"] is False))
        diagnostics.append(diagnostic)
        trials.append(trial[0])
        write(output / "direction1_diagnostics.json", diagnostics)
    ordinary = (trials[0] - trials[1]) / 2e-6
    reported = (diagnostics[0]["reported_change"] - diagnostics[1]["reported_change"]) / 2e-6
    rayleigh = (diagnostics[0]["rayleigh_change"] - diagnostics[1]["rayleigh_change"]) / 2e-6
    direct_quotient = (diagnostics[0]["direct_retained_quotient_difference"] -
                       diagnostics[1]["direct_retained_quotient_difference"]) / 2e-6
    ledger = {"new_scientific_target": "EOL_Rayleigh", "analytic_saved_g": analytical,
              "ordinary_reported_total_central": ordinary, "diagnostic_reported_central": reported,
              "Rayleigh_central": rayleigh, "direct_retained_quotient_central": direct_quotient,
              "reported_minus_Rayleigh_central": reported - rayleigh,
              "Rayleigh_minus_direct_quotient_central": rayleigh - direct_quotient,
              "ordinary_one_ULP_derivative_scale": float(np.spacing(abs(base[0])) / 1e-6),
              "gate": max(1e-8, 1e-5 * abs(analytical)), "h": 1e-6, "PCG64_seed": 20261012,
              "reported_derivative_is_acceptance_gate": False,
              "finite_response_norm_residuals_retained": True,
              "uniform_roundoff_error_bound_certified": False, "optimizer_authorized": False}
    write(output / "direction1_central.json", ledger)
    check("direction1_NEW_Rayleigh_derivative", abs(rayleigh - analytical), ledger["gate"])
    for name, expected in (("H", 3), ("CI", 3), ("gradient", 0), ("validation_Dirac", 7),
                           ("initial_hS", 7), ("mixed_hS", 14), ("tracking_resolution", 10),
                           ("tracking_SVD", 50), ("tracking_overlap", 11),
                           ("geometry_eigvalsh", 36), ("polar_retraction", 14)):
        check("complete_call_count_" + name, abs(completed[name] - expected))
    check("CI_internal_hS_within21", max(0, completed["CI_internal_hS"] - 21))
    check("complete_spectral_bound36", max(0, completed["scipy_eigh"] - 36))
    write(output / "result.json", {"new_rotation_Rayleigh_qualification_passed": True,
          "new_scientific_target": "EOL_Rayleigh", "controls": controls, "records": records,
          "derivative_ledger": ledger, "started": started, "completed": completed,
          "accounting_CPUs": accounting, "source_map_sha256": contract["source"]["candidate356_map_sha256"],
          "new_metric_sha256": state.residual_metric_sha256, "geometry_sha256": state.geometry_sha256,
          "objective_identity_declaration": state.objective_identity,
          "original_P_preserved": True, "original_tracker_rebased": False,
          "new3451_orbital_residual": residual, "energy_indicator_Ha": indicator,
          "stationarity_gates_passed": residual < 1e-7 and indicator < 1e-8,
          "saved_g_only_no_whole_public_gradient_replay": True,
          "original_reported_root_probe_passed": False, "original_failure_unchanged": True,
          "optimizer_calls": 0, "optimizer_authorized": False,
          "uniform_error_bound_or_callback_certified": False,
          "full_MCDHF_optimization_completed": False, "experimental_certification": False,
          "elapsed_uncontrolled_s": time.monotonic() - begin})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-contract", required=True)
    parser.add_argument("--authorize", action="store_true", help="Pure metadata validation, zero numerical imports")
    args = parser.parse_args()
    contract = json.loads((HERE / "contract.json").read_text())
    execution = json.loads(Path(args.execution_contract).read_text())
    if args.authorize:
        bind_inputs(contract, execution)
        print(json.dumps({"metadata_root_admission_passed": True, "physical_calls": 0,
                          "numeric_imports": 0, "new_scientific_target": "EOL_Rayleigh"}))
        return
    try:
        probe(contract, execution)
    except BaseException as error:
        output = HERE / "actual"
        if output.is_dir():
            import traceback
            write(output / "failure.json", {"scientific_success_claim": False,
                  "error": str(error), "traceback": traceback.format_exc(),
                  "progress_path": "progress.json", "driver_sha256": sha(__file__),
                  "contract_sha256": sha(HERE / "contract.json"), "original_failure_unchanged": True})
        raise


if __name__ == "__main__":
    main()
