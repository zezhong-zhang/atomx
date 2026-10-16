"""Prepared native mixed29 derivative probe, never an optimizer.

Admission requires a separate exact root execution seal before numerical
imports. The immutable preparation itself provides no execution authority.
"""
import argparse
import hashlib
import json
import math
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


def admit(contract, execution):
    if contract["status"] != "prepared_requires_exact_root_execution":
        raise ValueError("Unprepared source/input/proof admission")
    preparation_path = HERE / "preparation_verification.json"
    required = {"execution_authorized_by_root": True,
                "driver_sha256": sha(__file__), "contract_sha256": sha(HERE / "contract.json"),
                "preparation_sha256": sha(preparation_path),
                "source_map_sha256": contract["source"]["candidate356_map_sha256"],
                "max_H": 6, "max_CI": 6, "max_full_gradient": 1, "max_optimizer": 0,
                "max_validation_Dirac": 7, "max_total_hS": 77}
    if any(execution.get(k) != v for k, v in required.items()):
        raise ValueError("Wrong exact root execution contract")
    preparation = json.loads(preparation_path.read_text())
    if (preparation.get("all_passed") is not True
            or preparation.get("runner_sha256") != required["driver_sha256"]
            or preparation.get("contract_sha256") != required["contract_sha256"]):
        raise ValueError("Unqualified immutable preparation")
    for prerequisite in contract["pending_prerequisites"].values():
        if not isinstance(prerequisite, dict) or sha(REPO / prerequisite["path"]) != prerequisite["sha256"]:
            raise ValueError("Pending or changed prerequisite proof")
    prerequisites = contract["pending_prerequisites"]
    screened = json.loads((REPO / prerequisites["screened002_independent_saved_proof"]["path"]).read_text())
    guard = json.loads((REPO / prerequisites["guard003_actual_qualification"]["path"]).read_text())
    review = json.loads((REPO / prerequisites["guard003_independent_source_review"]["path"]).read_text())
    if (screened.get("all_independent_saved_array_checks_passed") is not True
            or guard.get("guard_qualification_passed") is not True
            or review.get("source_review_all_passed") is not True):
        raise ValueError("Prerequisite scientific or source review failed")
    for role in contract["input_roles"].values():
        if sha(REPO / role["path"]) != role["sha256"]:
            raise ValueError("Changed scientific input")
    identity = json.loads((REPO / contract["input_roles"]["candidate_source_identity"]["path"]).read_text())
    if identity["source_map_sha256"] != contract["source"]["candidate356_map_sha256"]:
        raise ValueError("Changed full candidate source epoch")
    root = REPO / contract["source"]["candidate_archive"]
    for name, expected in identity["all_atomx_python_sha256"].items():
        if sha(root / name.removeprefix("src/")) != expected:
            raise ValueError("Frozen source byte mismatch")
    if any(os.environ.get(k) != "1" for k in (
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")):
        raise ValueError("Explicit single-thread process required")
    accounting = sorted(os.sched_getaffinity(0))
    selected = execution.get("CPU")
    if selected not in accounting or os.getloadavg()[0] > execution["maximum_system_load1"]:
        raise ValueError("Host resource admission failed")
    resource.setrlimit(resource.RLIMIT_AS, (int(3.95 * 2**30), int(3.95 * 2**30)))
    os.sched_setaffinity(0, {selected})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("300s probe cap")))
    signal.alarm(300)
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
    limits = {"H": 6, "CI": 6, "gradient": 1, "validation_Dirac": 7,
              "initial_hS": 7, "CI_internal_hS": 42, "mixed_hS": 28,
              "tracking_resolution": 21, "tracking_SVD": 105,
              "tracking_overlap": 22, "geometry_eigvalsh": 36,
              "scipy_eigh": 37, "polar_retraction": 28,
              "polar_numpy_eigh": 28}
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

    saved, geometry, canonical = archive("checkpoint_arrays"), archive("screened_geometry"), archive("canonical29")
    radial = archive("geometry_workspace")
    data = {k[6:]: v.item() if v.ndim == 0 else v.copy() for k, v in radial.items() if k.startswith("basis_")}
    canonical_waves = waves_from(canonical, "radial_keys")
    old_keys = set(map(tuple, saved["orbital_keys"]))
    new_keys = set(canonical_waves) - old_keys
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
    original_metric = {k: geometry[f"s_{k}"] for k in set(key[1] for key in canonical_waves)}
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

    def complete_point(name, waves, gradient=False):
        result = functional.evaluate(waves, gradient=gradient)
        energy, ci = (result[0], result[-1])
        h = ci.hamiltonian_for(selected, waves, functional.keys, data, 70, 123,
                              nuclear_potential=nuclear, max_determinants=2000)
        np.savez_compressed(output / (name + ".npz"), H=h, energies=ci.state_energies,
                            mixing=ci.mixing_coefficients, keys=np.array(functional.keys),
                            P=np.array([waves[k][0] for k in functional.keys]),
                            Q=np.array([waves[k][1] for k in functional.keys]))
        if gradient:
            np.savez_compressed(output / (name + "_raw_radial_gradient.npz"),
                                keys=np.array(functional.keys),
                                radial_gradient=np.array([result[1][k] for k in functional.keys]))
        write(output / (name + "_owner.json"), ci.hamiltonian_snapshot.identity_payload())
        raw_identity()
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
                         "column": column, "energy_Ha": float(ci.state_energies[column]),
                         "tracking": binding})
        energies = {row["label"]: row["energy_Ha"] for row in rows}
        ground = energies["6s_1/2"]
        from atomx.constants import HARTREE_CM1
        for row in rows:
            row["transition_cm1"] = (row["energy_Ha"] - ground) * HARTREE_CM1
        for prefix, weights2 in (("5d", (4, 6)), ("6p", (2, 4))):
            low, high = energies[prefix + "_" + ("3/2" if prefix == "5d" else "1/2")], energies[prefix + "_" + ("5/2" if prefix == "5d" else "3/2")]
            rows.append({"multiplet": prefix, "centroid_cm1":
                         ((weights2[0] * low + weights2[1] * high) / sum(weights2) - ground) * HARTREE_CM1,
                         "splitting_cm1": (high - low) * HARTREE_CM1})
        records.append({"name": name, "EOL_Ha": energy, "targets": rows})
        flush(name + "_complete")
        return energy, ci, h, result[1] if gradient else None

    # First H/CI uses the exact saved physical seed. It is not called recovered-x.
    anchor = complete_point("canonical_anchor", canonical_waves)
    old_h, old_ci = archive("screened_H"), archive("screened_CI")
    check("canonical_H_bitwise", not np.array_equal(anchor[2], old_h["matrix_Ha"]))
    check("canonical_E_bitwise", not np.array_equal(anchor[1].state_energies, old_ci["energies_Ha"]))
    check("canonical_C_bitwise", not np.array_equal(anchor[1].mixing_coefficients, old_ci["mixing"]))

    layout, diagonal, inverse, coordinates = [], [], [], []
    cursor = 0
    copy_sources = {tuple(map(int, key.split(","))): tuple(value) for key, value in
                    contract["metric"]["new5_same_kappa_SPD_block_copy_sources"].items()}
    for i, old in enumerate(load("original_layout")):
        old_group, old_shape = tuple(map(tuple, old["keys"])), tuple(old["shape"])
        kappa = old_group[0][1]
        group = tuple(k for k in functional.keys if k[1] == kappa)
        projector = saved[f"projector_{i}"]
        old_y = saved["x"][slice(*old["slice"])].reshape(old_shape)
        old_d = saved["inverse_diagonal"][slice(*old["slice"])].reshape(old_shape)
        y, d, blocks = [], [], []
        for key in group:
            if key in old_group:
                column = old_group.index(key)
                value = old_y[:, column].copy()
            else:
                column = old_group.index(copy_sources[key])
                value = projector.T @ functional.metric[kappa] @ np.concatenate(canonical_waves[key])
            y.append(value)
            d.append(old_d[:, column].copy())
            blocks.append(saved[f"inverse_blocks_{i}"][column].copy())
        y, d = np.column_stack(y), np.column_stack(d)
        shape, extent = y.shape, y.size
        layout.append((group, projector, shape, slice(cursor, cursor + extent)))
        coordinates.append(y.ravel())
        diagonal.append(d.ravel())
        inverse.append(blocks)
        cursor += extent
    check("3451coordinate_count", abs(cursor - 3451))
    identity = "correlated29-complete1046-original5-DC-new3451-metric-no-history"
    state = OrbitalOptimizationState.from_arrays(functional, layout=layout,
        coordinates=np.concatenate(coordinates), inverse_diagonal=np.concatenate(diagonal),
        inverse_blocks=inverse, projector_source="nuclear", objective_identity=identity,
        occupation_floor=.05)
    check("new_metric_differs_from_old24_metric", state.residual_metric_sha256 ==
          contract["metric"]["legacy_original_metric_sha256"])
    for i, (_, projector, _, _) in enumerate(state.layout):
        check(f"state_P_bytes_{i}", not (np.array_equal(projector, saved[f"projector_{i}"])
              and projector.tobytes() == saved[f"projector_{i}"].tobytes()))
        group, _, shape, where = state.layout[i]
        old = load("original_layout")[i]
        old_group = tuple(map(tuple, old["keys"]))
        old_d = saved["inverse_diagonal"][slice(*old["slice"])].reshape(tuple(old["shape"]))
        actual_d = state.inverse_diagonal[where].reshape(shape)
        for j, key in enumerate(group):
            source_key = key if key in old_group else copy_sources[key]
            source_column = old_group.index(source_key)
            check(f"inverse_block_bytes_{key}", state.inverse_blocks[i][j].tobytes() !=
                  saved[f"inverse_blocks_{i}"][source_column].tobytes())
            check(f"inverse_diagonal_column_bytes_{key}", actual_d[:, j].tobytes() !=
                  old_d[:, source_column].tobytes())

    def unpack(x):
        waves = {}
        for keys, projector, shape, where in state.layout:
            c = projector @ x[where].reshape(shape)
            for j, key in enumerate(keys):
                waves[key] = c[:123, j].copy(), c[123:, j].copy()
        return waves

    def tangent(x, vector):
        result = vector.copy()
        for _, _, shape, where in state.layout:
            y, g = x[where].reshape(shape), vector[where].reshape(shape)
            a = y.T @ g
            result[where] = (g - .5 * y @ (a + a.T)).ravel()
        return result

    x, waves = state.coordinates, unpack(state.coordinates)
    for key in old_keys:
        check("old24_exact_" + str(key), not all(np.array_equal(a, b) for a, b in
              zip(waves[key], canonical_waves[key], strict=True)))
    drift = {str(k): max(float(np.max(abs(a - b))) for a, b in zip(waves[k], canonical_waves[k], strict=True))
             for k in new_keys}
    for key, error in drift.items():
        check("new5_coordinate_recovery_drift_" + key, error, 1e-12)
    np.savez_compressed(output / "full_point_identity.npz", x=x,
                        canonical_P=canonical["P"], canonical_Q=canonical["Q"],
                        recovered_P=np.array([waves[k][0] for k in functional.keys]),
                        recovered_Q=np.array([waves[k][1] for k in functional.keys]),
                        keys=np.array(functional.keys), old_x=saved["x"])
    write(output / "point_identity.json", {"canonical_is_not_recovered_point": True, "new5_wave_drift": drift,
          "old24_seed_bitwise": True, "new_metric_sha256": state.residual_metric_sha256,
          "geometry_sha256": state.geometry_sha256, "state_sha256": state.state_sha256,
          "objective_identity_declaration": state.objective_identity,
          "objective_identity_is_caller_declaration_not_certification": True,
          "projector_source": state.projector_source, "occupation_floor": state.occupation_floor})
    write(output / "new_coordinate_layout.json", [{"keys": row[0], "shape": row[2],
          "slice": [row[3].start, row[3].stop]} for row in state.layout])
    base = complete_point("recovered_coordinate_point", waves, gradient=True)
    raw_g = np.zeros(3451)
    for keys, projector, _shape, where in state.layout:
        raw_g[where] = (projector.T @ np.column_stack([base[3][k] for k in keys])).ravel()
    g = tangent(x, raw_g)
    inverse_g = g.copy()
    for (_, _, shape, where), blocks in zip(state.layout, state.inverse_blocks, strict=True):
        a = g[where].reshape(shape)
        inverse_g[where] = np.column_stack([block @ a[:, j] for j, block in enumerate(blocks)]).ravel()
    pg = tangent(x, inverse_g)
    np.savez_compressed(output / "full_gradient_metric.npz", x=x, raw_coordinate_gradient=raw_g,
                        tangent_gradient=g, preconditioned_gradient=pg,
                        radial_gradient=np.array([base[3][k] for k in functional.keys]),
                        inverse_diagonal=state.inverse_diagonal,
                        **{f"projector_{i}": row[1] for i, row in enumerate(state.layout)},
                        **{f"inverse_blocks_{i}": np.array(v) for i, v in enumerate(state.inverse_blocks)})
    weights, _ = tracker.resolve(selected, base[1], waves, functional.metric)
    full_density = {}
    for column in np.flatnonzero(weights):
        for key, value in functional.radial_density.contract_bilinear(
                base[1].mixing_coefficients[:, column], base[1].mixing_coefficients[:, column]).items():
            full_density[key] = math.fsum((full_density.get(key, 0.), weights[column] * value))
    write(output / "signed_density.json", {"unfiltered": [[key, value] for key, value in full_density.items()],
          "actual_gradient_retained": [[key, value] for key, value in functional.last_weights.items()],
          "retained_contract_cutoff": 1e-15, "CI_column_norms":
          [float(base[1].mixing_coefficients[:, j] @ base[1].mixing_coefficients[:, j]) for j in np.flatnonzero(weights)],
          "unfiltered_vs_retained_max_coefficient_difference": max(abs(full_density.get(key, 0.) -
           functional.last_weights.get(key, 0.)) for key in set(full_density) | set(functional.last_weights))})
    residual, indicator = float(np.linalg.norm(pg)), .5 * abs(float(g @ pg))
    external_gradient, rotational_gradient = np.zeros(3451), np.zeros(3451)
    for keys, _, shape, where in state.layout:
        y, raw = x[where].reshape(shape), raw_g[where].reshape(shape)
        external = raw - y @ (y.T @ raw)
        for j, key in enumerate(keys):
            if key not in new_keys:
                external[:, j] = 0.
        antisymmetric = .5 * (y.T @ raw - raw.T @ y)
        for i, a in enumerate(keys):
            for j, b in enumerate(keys):
                if (a in new_keys) == (b in new_keys):
                    antisymmetric[i, j] = 0.
        external_gradient[where] = external.ravel()
        rotational_gradient[where] = (y @ antisymmetric).ravel()
    euler_parts = {"new5_external_coordinate_gradient_norm": float(np.linalg.norm(external_gradient)),
                   "old_new_rotation_coordinate_gradient_norm": float(np.linalg.norm(rotational_gradient)),
                   "not_a_missing_physics_error_bound": True}
    write(output / "correlated_Euler_parts.json", euler_parts)
    np.savez_compressed(output / "correlated_Euler_parts.npz", new5_external=external_gradient,
                        old_new_rotation=rotational_gradient)

    for number, spec in enumerate(contract["directions"]):
        rng = np.random.Generator(np.random.PCG64(spec["PCG64_seed"]))
        direction = np.zeros(3451)
        for keys, _, shape, where in state.layout:
            y = x[where].reshape(shape)
            if number == 0:
                r = np.zeros(shape)
                for j, key in enumerate(keys):
                    if key in new_keys:
                        r[:, j] = rng.standard_normal(shape[0])
                direction[where] = (r - y @ (y.T @ r)).ravel()
            else:
                omega = np.zeros((shape[1], shape[1]))
                for i, a in enumerate(keys):
                    for j in range(i + 1, len(keys)):
                        if (a in new_keys) != (keys[j] in new_keys):
                            omega[i, j] = rng.standard_normal()
                            omega[j, i] = -omega[i, j]
                direction[where] = (y @ omega).ravel()
        norm = float(np.linalg.norm(direction))
        if norm <= contract["finite_step"]["zero_direction_norm_threshold"]:
            raise ValueError("Inconclusive direction; no replacement or step scan")
        direction /= norm
        check(f"direction{number}_shared_tangent", np.max(abs(tangent(x, direction) - direction)), 1e-12)
        analytical = float(g @ direction)
        trials, diagnostics = [], []
        for sign in (1., -1.):
            candidate = x + sign * 1e-6 * direction
            for _, _, shape, where in state.layout:
                candidate[where] = _polar_retraction(candidate[where].reshape(shape)).ravel()
            trial_waves = unpack(candidate)
            trial = complete_point(f"direction{number}_{'plus' if sign > 0 else 'minus'}", trial_waves)
            increment = unpack(candidate - x)
            diagnostic = functional.mixed_ci_energy_difference(waves, trial_waves, base[1], trial[1],
                                                               increments=increment)
            raw_identity()
            diagnostics.append(diagnostic)
            trials.append(trial[0])
            write(output / f"direction{number}_diagnostics.json", diagnostics)
            np.savez_compressed(output / f"direction{number}_{'plus' if sign > 0 else 'minus'}_coordinates.npz",
                                candidate_x=candidate, explicit_increment_P=np.array([increment[k][0] for k in functional.keys]),
                                explicit_increment_Q=np.array([increment[k][1] for k in functional.keys]))
        ordinary = (trials[0] - trials[1]) / 2e-6
        reported = (diagnostics[0]["reported_change"] - diagnostics[1]["reported_change"]) / 2e-6
        rayleigh = (diagnostics[0]["rayleigh_change"] - diagnostics[1]["rayleigh_change"]) / 2e-6
        ledger = {"analytic": analytical, "ordinary_central": ordinary, "reported_central": reported,
                  "Rayleigh_central": rayleigh, "ordinary_one_ULP_derivative_scale":
                  float(np.spacing(abs(base[0])) / 1e-6), "gate": max(1e-8, 1e-5 * abs(analytical))}
        write(output / f"direction{number}_central.json", ledger)
        np.savez_compressed(output / f"direction{number}_coordinates.npz", direction=direction)
        check(f"direction{number}_reported_derivative", abs(reported - analytical), ledger["gate"])
        check(f"direction{number}_Rayleigh_derivative", abs(rayleigh - analytical), ledger["gate"])
    for name, expected in (("H", 6), ("CI", 6), ("gradient", 1), ("validation_Dirac", 7),
                           ("initial_hS", 7), ("mixed_hS", 28),
                           ("tracking_resolution", 21), ("tracking_SVD", 105),
                           ("tracking_overlap", 22), ("geometry_eigvalsh", 36),
                           ("scipy_eigh", 37), ("polar_retraction", 28)):
        check("complete_call_count_" + name, abs(completed[name] - expected))
    delta_rows = []
    canonical_energies = {r["label"]: r["energy_Ha"] for r in records[0]["targets"] if "label" in r}
    recovered_energies = {r["label"]: r["energy_Ha"] for r in records[1]["targets"] if "label" in r}
    from atomx.constants import HARTREE_CM1
    lower_change = recovered_energies["6s_1/2"] - canonical_energies["6s_1/2"]
    for label, value in recovered_energies.items():
        upper_change = value - canonical_energies[label]
        delta_rows.append({"label": label, "upper_change_Ha": upper_change,
                           "lower_change_Ha": lower_change,
                           "transition_change_cm1": (upper_change - lower_change) * HARTREE_CM1})
    write(output / "canonical_to_recovered_energy_changes.json", delta_rows)
    write(output / "result.json", {"derivative_diagnostic_qualified": True, "controls": controls,
          "records": records, "started": started, "completed": completed, "accounting_CPUs": accounting,
          "original_P_preserved": True, "original_tracker_rebased": False, "optimizer_calls": 0,
          "objective_identity_declaration": state.objective_identity,
          "new_metric_sha256": state.residual_metric_sha256,
          "geometry_sha256": state.geometry_sha256, "state_sha256": state.state_sha256,
          "new_metric_orbital_residual": residual, "new_metric_energy_indicator_Ha": indicator,
          "stationarity_gates_passed": residual < 1e-7 and indicator < 1e-8,
          "same_point_whole_gradient_replay_executed": False,
          "same_point_raw_g_1e_minus9_pg_1e_minus10_replay_gates_passed": None,
          "whole_gradient_replay_scope": "One public full gradient budget; independent same-point rawg/pg replay requires a separately authorized successor.",
          "full_MCDHF_optimization_completed": False, "experimental_certification": False,
          "correlated_Euler_parts": euler_parts, "elapsed_uncontrolled_s": time.monotonic() - begin})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-contract", required=True)
    args = parser.parse_args()
    contract = json.loads((HERE / "contract.json").read_text())
    execution = json.loads(Path(args.execution_contract).read_text())
    try:
        probe(contract, execution)
    except BaseException as error:
        output = HERE / "actual"
        if output.is_dir():
            import traceback
            write(output / "failure.json", {"scientific_success_claim": False,
                  "error": str(error), "traceback": traceback.format_exc(),
                  "progress_path": "progress.json", "driver_sha256": sha(__file__),
                  "contract_sha256": sha(HERE / "contract.json")})
        raise


if __name__ == "__main__":
    main()
