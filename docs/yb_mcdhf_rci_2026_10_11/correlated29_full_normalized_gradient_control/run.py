"""One frozen native baseline, two complete signed gradient contractions.

This is a diagnostic consumer, never an optimizer or production energy adapter.
"""
import argparse
import ast
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def key(value):
    return tuple(key(v) for v in value) if isinstance(value, list) else value


def target_bindings(targets, tracking):
    """Use original physical UIDs, never symmetry insertion/column order."""
    matched = {}
    for row in tracking:
        if len(row["reference_levels"]) != 1 or len(row["selected_levels"]) != 1:
            raise ValueError("Only complete isolated original targets supported")
        uid = tuple(row["reference_levels"][0])
        if uid in matched or tuple(row["selected_levels"][0][:2]) != uid[:2]:
            raise ValueError("Changed or duplicate physical target")
        matched[uid] = row
    expected = {(t["twoJ"], t["parity"], t["models"]["DC"]["local_root"]) for t in targets}
    if len(expected) != 5 or set(matched) != expected:
        raise ValueError("Incomplete five original UIDs")
    return matched


def metadata(contract):
    """Read-only admission; no numerical imports or source execution."""
    for row in contract["input_roles"].values():
        if sha(REPO / row["path"]) != row["sha256"]:
            raise ValueError("Changed bound input " + row["path"])
    identity = json.loads((REPO / contract["input_roles"]["candidate_source_identity"]["path"]).read_text())
    if identity["source_map_sha256"] != contract["source"]["map_sha256"]:
        raise ValueError("Wrong frozen source epoch")
    root = REPO / contract["source"]["archive"]
    for name, expected in identity["all_atomx_python_sha256"].items():
        if sha(root / name.removeprefix("src/")) != expected:
            raise ValueError("Frozen source drift " + name)
    for name in ("centering_result", "centering0_result"):
        centered = json.loads((REPO / contract["input_roles"][name]["path"]).read_text())
        if not (centered.get("saved_only_diagnosis_completed") is True
                and centered.get("same_saved_point_conditional_gate_passed") is True
                and centered.get("original_rotation002_passed") is False
                and centered.get("optimizer_authorized") is False):
            raise ValueError("Centering prerequisite absent or changed in scope")
    return root


def execute(contract, authority):
    root = metadata(contract)
    required = {"execution_authorized_by_root": True,
                "contract_sha256": sha(HERE / "contract.json"), "driver_sha256": sha(__file__),
                "preparation_sha256": sha(HERE / "preparation_verification.json"),
                "source_map_sha256": contract["source"]["map_sha256"],
                "max_H": 1, "max_CI": 1, "max_gradient_from_weights": 2,
                "max_optimizer": 0, "max_Dirac": 0, "CPU": 31,
                "wall_seconds": 300, "address_space_GiB": 3.95}
    if any(authority.get(k) != v for k, v in required.items()):
        raise ValueError("Exact root execution contract required")
    preparation = json.loads((HERE / "preparation_verification.json").read_text())
    if not (preparation["all_passed"] and preparation["runner_sha256"] == sha(__file__)
            and preparation["contract_sha256"] == sha(HERE / "contract.json")):
        raise ValueError("Preparation is stale")
    if any(os.environ.get(k) != "1" for k in (
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")):
        raise ValueError("Single-thread environment required")
    accounting = sorted(os.sched_getaffinity(0))
    if 31 not in accounting or os.getloadavg()[0] > authority["maximum_system_load1"]:
        raise ValueError("Resource admission failed")
    resource.setrlimit(resource.RLIMIT_AS, (int(3.95 * 2**30), int(3.95 * 2**30)))
    os.sched_setaffinity(0, {31})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("300s hard cap")))
    signal.alarm(300)
    if any(n == "atomx" or n.startswith("atomx.") for n in sys.modules):
        raise ValueError("No preimported physical source")
    sys.path.insert(0, str(root))
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    begin = time.monotonic()
    started = dict.fromkeys(contract["calls"], 0)
    completed = dict(started)
    controls, rows, raw_operators = [], [], {}

    def flush(phase):
        write(output / "progress.json", {"phase": phase, "started": started,
              "completed": completed, "controls": controls, "targets": rows,
              "elapsed_uncontrolled_s": time.monotonic() - begin})

    try:
        import numpy as np
        from scipy import linalg

        import atomx.methods.csf.variational as variational
        import atomx.numerics.bspline.dirac_assembly as dirac
        import atomx.physics.csf.matrix as matrix
        import atomx.physics.csf.tracking as tracking_module
        from atomx.angular.csf_hamiltonian import csf_signature
        from atomx.config.csf import enumerate_jj_csfs
        from atomx.methods.ci import CIResult
        from atomx.numerics.bspline.orbital_optimization import _residual_metric_sha256
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.physics.csf.state_average import CIStateAverage
        from atomx.physics.csf.tracking import CIBasisTracking
        from atomx.potentials.nuclear import make_nuclear_potential

        def check(name, error, tolerance=0.):
            passed = bool(np.isfinite(error) and error <= tolerance)
            controls.append(dict(name=name, error=float(error), tolerance=tolerance, passed=passed))
            flush(name)
            if not passed:
                raise ValueError("Gate failed " + name)

        def wrapper(name, fn, capture=False):
            def call(*args, **kwargs):
                started[name] += 1
                flush("before_" + name)
                if started[name] > contract["calls"][name]:
                    raise RuntimeError("Attempted call cap " + name)
                value = fn(*args, **kwargs)
                completed[name] += 1
                if capture:
                    kappa = int(args[1])
                    if kappa in raw_operators:
                        h, s = raw_operators[kappa]
                        if not (np.array_equal(h, value[0]) and np.array_equal(s, value[1])):
                            raise ValueError("Changed raw h/S within same point")
                    else:
                        raw_operators[kappa] = (value[0].copy(), value[1].copy())
                    np.savez_compressed(output / f"captured_{name}_{completed[name]:02d}.npz",
                                        kappa=np.array(kappa), raw_h=value[0], S=value[1])
                flush("after_" + name)
                return value
            return call

        variational.build_dirac_matrices = wrapper("initial_hS", dirac.build_dirac_matrices, True)
        matrix.build_dirac_matrices = wrapper("CI_hS", dirac.build_dirac_matrices, True)
        matrix.build_ci_matrix = wrapper("H", matrix.build_ci_matrix)
        variational.ci_solve = wrapper("CI", variational.ci_solve)
        variational.CSFOrbitalFunctional.gradient_from_weights = wrapper(
            "gradient_from_weights", variational.CSFOrbitalFunctional.gradient_from_weights)
        variational.CSFOrbitalFunctional._pair_potential = wrapper(
            "gradient_pair_potential", variational.CSFOrbitalFunctional._pair_potential)
        CIBasisTracking.resolve = wrapper("tracker_resolve", CIBasisTracking.resolve)
        tracking_module.ci_subspace_overlaps = wrapper("tracker_overlap", tracking_module.ci_subspace_overlaps)
        np.linalg.svd = wrapper("tracker_SVD", np.linalg.svd)
        linalg.eigh = wrapper("CI_eigh", linalg.eigh)
        dirac.solve_dirac = wrapper("Dirac", dirac.solve_dirac)

        def path(name):
            return REPO / contract["input_roles"][name]["path"]

        def load(name):
            return json.loads(path(name).read_text())

        def archive(name):
            with np.load(path(name), allow_pickle=False) as a:
                return {k: a[k].copy() for k in a.files}

        def waves_from(a, keyname):
            return {tuple(map(int, k)): (p.copy(), q.copy()) for k, p, q in
                    zip(a[keyname], a["P"], a["Q"], strict=True)}

        saved, saved_gradient = archive("saved_point"), archive("saved_gradient")
        geometry, radial = archive("screened_geometry"), archive("geometry_workspace")
        waves = waves_from(saved, "keys")
        data = {k[6:]: v.item() if v.ndim == 0 else v.copy()
                for k, v in radial.items() if k.startswith("basis_")}
        models = load("screened_models")["models"]
        old_csfs = [c for row in models["baseline"]["configs"] for c in enumerate_jj_csfs(row["config"])]
        full = [c for row in models["balanced_virtual"]["configs"] for c in enumerate_jj_csfs(row["config"])]
        sectors = {(j, p) for j, p, _ in contract["fixed"]["target_blocks"]}
        csfs = [c for c in full if (c.total_J2, c.parity) in sectors]
        check("complete531_reference", abs(len(old_csfs) - 531))
        check("complete1046_target_CSFS", abs(len(csfs) - 1046))
        for j, p, count in contract["fixed"]["target_blocks"]:
            check(f"complete_target_sector_{j}_{p}", abs(sum((c.total_J2, c.parity) == (j, p) for c in csfs) - count))
        signatures = [json.dumps(plain((s.subshells, s.coupling, c.total_J2, c.parity)), separators=(",", ":"))
                      for c in csfs for s in [csf_signature(c)]]
        check("exact_signed_basis_order", signatures != load("screened_basis")["balanced_virtual"]["retained_signatures"])
        check("charge69", {c.n_electrons for c in csfs} != {69})
        targets = load("tracking_binding")["targets"]
        average = CIStateAverage(tuple((t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.) for t in targets))
        reference = archive("tracking_ci")
        metrics = {k[1]: geometry[f"s_{k[1]}"] for k in waves}
        tracker = CIBasisTracking(average, old_csfs, CIResult(reference["energies_Ha"], reference["mixing"]),
                                 waves_from(radial, "orbital_keys"), metrics, max_determinant_pairs=1_000_000)
        functional = variational.CSFOrbitalFunctional(csfs, data, 70, make_nuclear_potential(70, "fermi", A=175),
                                                     state_average=average, level_tracking=tracker,
                                                     retain_hamiltonian=True, max_determinants=2000)
        check("exact29keys", tuple(functional.keys) != tuple(map(tuple, contract["fixed"]["keys29"])))
        for k, (h, s) in raw_operators.items():
            logical = np.tril(h) + np.tril(h, -1).T
            check(f"raw_logical_h_{k}", not np.array_equal(logical, functional.one_body[k]))
            check(f"original_logical_h_{k}", not np.array_equal(logical, geometry[f"h_{k}"]))
            check(f"original_S_{k}", not np.array_equal(s, geometry[f"s_{k}"]))

        layout = [(tuple(map(tuple, r["keys"])), saved_gradient[f"projector_{i}"],
                   tuple(r["shape"]), slice(*r["slice"])) for i, r in enumerate(load("saved_layout"))]
        inverse = [saved_gradient[f"inverse_blocks_{i}"] for i in range(7)]
        fingerprint = _residual_metric_sha256(layout, saved_gradient["inverse_diagonal"], inverse)
        check("unchanged_original_metric", fingerprint != contract["metric_sha256"])
        check("complete3451", saved_gradient["x"].shape != (3451,))
        for keys, projector, shape, where in layout:
            c = projector @ saved_gradient["x"][where].reshape(shape)
            s = functional.metric[keys[0][1]]
            for j, k in enumerate(keys):
                check("actual_P_times_x_" + str(k), not (np.array_equal(c[:123, j], waves[k][0])
                      and np.array_equal(c[123:, j], waves[k][1])
                      and c[:123, j].tobytes() == waves[k][0].tobytes()
                      and c[123:, j].tobytes() == waves[k][1].tobytes()))
            check("actual_radial_Gram_" + str(keys[0][1]), np.max(abs(c.T @ s @ c - np.eye(len(keys)))), 1e-10)
            outside = c - projector @ (projector.T @ s @ c)
            check("original_P_leak_" + str(keys[0][1]), np.max(np.sum(outside * (s @ outside), axis=0)), 1e-12)

        energy, ci = functional.evaluate(waves, gradient=False)
        h = ci.hamiltonian_for(csfs, waves, functional.keys, data, 70, 123,
                              nuclear_potential=functional.nuclear_potential, max_determinants=2000)
        np.savez_compressed(output / "fresh_baseline.npz", H=h, energies=ci.state_energies,
                            mixing=ci.mixing_coefficients, keys=np.array(functional.keys),
                            P=np.array([waves[k][0] for k in functional.keys]), Q=np.array([waves[k][1] for k in functional.keys]))
        write(output / "owner.json", ci.hamiltonian_snapshot.identity_payload())
        for name, a in (("H", h), ("energies", ci.state_energies), ("mixing", ci.mixing_coefficients)):
            check("fresh_bitwise_" + name, not (np.array_equal(a, saved[name]) and a.tobytes() == saved[name].tobytes()))
        check("fresh_owner_payload", plain(ci.hamiltonian_snapshot.identity_payload()) != load("saved_point_owner"))
        check("full_H_symmetric", not np.array_equal(h, h.T))
        check("full_CI_Gram", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients - np.eye(1046))), 1e-10)
        check("full_CI_residual", np.max(np.linalg.norm(h @ ci.mixing_coefficients - ci.mixing_coefficients * ci.state_energies, axis=0)), 1e-8)
        weights, tracking = tracker.resolve(csfs, ci, waves, functional.metric)
        check("five_original_equalweights", not (np.count_nonzero(weights) == 5 and np.array_equal(weights[weights > 0], np.repeat(.2, 5))))
        _, blocks = ci_level_blocks(csfs, ci, require_full=True)
        matched = target_bindings(targets, tracking)
        check("reported_EO_bitwise", energy.hex() != float(weights @ saved["energies"]).hex())
        functional.radial_density.contract = wrapper("retained_RDM", functional.radial_density.contract)
        functional.radial_density.contract_bilinear = wrapper("unfiltered_target_RDM", functional.radial_density.contract_bilinear)
        retained = functional.radial_density.contract(ci, weights)
        captured = {key(k): float(v) for k, v in load("saved_density")["actual_gradient_retained"]}
        check("same_retained_signed_table", retained != captured)
        contributions, target_tables = {}, []
        for t in targets:
            uid = t["twoJ"], t["parity"], t["models"]["DC"]["local_root"]
            j, p, rank = matched[uid]["selected_levels"][0]
            column = int(blocks[j, p][rank])
            c = ci.mixing_coefficients[:, column]
            n = sum((Fraction.from_float(float(v)) ** 2 for v in c), Fraction(0))
            if n <= 0:
                raise ValueError("Nonpositive exact CI norm")
            w = Fraction.from_float(float(weights[column]))
            table = functional.radial_density.contract_bilinear(c, c)
            for k, value in table.items():
                contributions.setdefault(k, []).append(w * Fraction.from_float(float(value)) / n)
            target_tables.append(dict(UID=uid, column=column,
                                      unfiltered=[[k, value] for k, value in table.items()]))
            rows.append(dict(label=t["label"], UID=uid, selected=(j, p, rank), column=column,
                             weight=float(w), exact_CI_norm=(str(n.numerator), str(n.denominator)),
                             norm_defect=float(n - 1), signed_nonzero_terms=len(table), tracking=matched[uid]))
        exact_full = {k: sum(v, Fraction(0)) for k, v in contributions.items()}
        exact_full = {k: v for k, v in exact_full.items() if v != 0}

        def represent(v):
            result = float(v)
            if result == 0 and v != 0:
                raise ValueError("Nonzero coefficient cannot be represented; no silent omission")
            if not math.isfinite(result):
                raise ValueError("Nonfinite coefficient")
            return result

        full_weights = {k: represent(v) for k, v in exact_full.items()}
        ordered_union = list(full_weights) + [k for k in retained if k not in full_weights]
        delta_exact = {k: Fraction.from_float(full_weights.get(k, 0.)) - Fraction.from_float(retained.get(k, 0.))
                       for k in ordered_union}
        delta_weights = {k: represent(v) for k, v in delta_exact.items() if v != 0}
        write(output / "signed_tables.json", {"normalization":"exact binary64 CI norms and targetwise rational aggregate, then explicit float64 coefficient rounding",
              "targets": rows, "target_tables": target_tables,
              "retained": [[k, v] for k, v in retained.items()],
              "full_normalized": [[k, v] for k, v in full_weights.items()],
              "full_minus_retained": [[k, v] for k, v in delta_weights.items()],
              "delta_order": "full table original order then retained-only original order",
              "coefficient_counts_are_not_bounds": True, "no_small_coefficient_filter": True})
        flush("all_signed_tables_archived")
        delta_g = functional.gradient_from_weights(waves, delta_weights, record_diagnostics=False)
        np.savez_compressed(output / "delta_gradient.npz", keys=np.array(functional.keys),
                            radial_gradient=np.array([delta_g[k] for k in functional.keys]))
        full_g = functional.gradient_from_weights(waves, full_weights, record_diagnostics=False)
        raw = saved_gradient["radial_gradient"]
        delta_array = np.array([delta_g[k] for k in functional.keys])
        full_array = np.array([full_g[k] for k in functional.keys])
        reconstructed = raw + delta_array
        np.savez_compressed(output / "logical_gradients.npz", keys=np.array(functional.keys), saved_raw=raw,
                            delta=delta_array, saved_plus_delta=reconstructed, direct_full=full_array)
        check("whole_vector_signed_linearity", np.max(abs(full_array - reconstructed)), 1e-9)

        # Pure samepoint raw nonsymmetric-h/ket-norm ambient correction.
        wide = np.longdouble
        corrected = {k: full_g[k].astype(wide) for k in functional.keys}
        v = {k: np.concatenate(waves[k]).astype(wide) for k in functional.keys}
        for term, coefficient in full_weights.items():
            if term[0] != "h":
                continue
            _, na, kappa, nb = term
            a, b = (na, kappa), (nb, kappa)
            raw_h, s = (m.astype(wide) for m in raw_operators[kappa])
            logical = functional.one_body[kappa].astype(wide)
            sb, hb, hta = s @ v[b], raw_h @ v[b], raw_h.T @ v[a]
            n, numerator = v[b] @ sb, v[a] @ hb
            if n <= 0:
                raise ValueError("Nonpositive radial ket norm")
            corrected[a] += coefficient * (hb / n - logical @ v[b])
            corrected[b] += coefficient * (hta / n - logical @ v[a] - 2 * numerator * sb / n**2)
        corrected_array = np.array([corrected[k] for k in functional.keys])

        def coordinate(radial_gradient):
            result = np.zeros(3451, dtype=radial_gradient.dtype)
            lookup = {k: radial_gradient[i] for i, k in enumerate(functional.keys)}
            for keys, projector, _, where in layout:
                result[where] = (projector.astype(radial_gradient.dtype).T @ np.column_stack([lookup[k] for k in keys])).ravel()
            return result

        def tangent(g):
            result = g.copy()
            for _, _, shape, where in layout:
                y, z = saved_gradient["x"][where].reshape(shape).astype(g.dtype), g[where].reshape(shape)
                yz = y.T @ z
                result[where] = (z - .5 * y @ (yz + yz.T)).ravel()
            return result

        def metric_action(g):
            result = g.copy()
            for (_, _, shape, where), inv in zip(layout, inverse, strict=True):
                z = g[where].reshape(shape)
                result[where] = np.column_stack([inv[j].astype(g.dtype) @ z[:, j] for j in range(shape[1])]).ravel()
            return tangent(result)

        all_vectors, vector_metrics = {}, {}
        for name, a in (("saved", raw), ("delta", delta_array), ("full_logical", full_array),
                        ("saved_plus_delta", reconstructed), ("full_raw_quotient", corrected_array)):
            cg, tg = coordinate(a), tangent(coordinate(a))
            pg = metric_action(tg)
            all_vectors.update({name + "_radial": a, name + "_raw_coordinate": cg,
                                name + "_tangent": tg, name + "_preconditioned": pg})
            vector_metrics[name] = dict(radial_norm=float(np.linalg.norm(a)), tangent_norm=float(np.linalg.norm(tg)),
                                         original_metric_residual=float(np.linalg.norm(pg)),
                                         original_metric_energy_indicator=float(.5 * abs(tg @ pg)))
        check("saved_raw_coordinate_replay", np.max(abs(all_vectors["saved_raw_coordinate"] - saved_gradient["raw_coordinate_gradient"])), 1e-9)
        check("saved_pg_replay", np.max(abs(all_vectors["saved_preconditioned"] - saved_gradient["preconditioned_gradient"])), 1e-10)
        check("fullvector_pg_linearity", np.max(abs(all_vectors["full_logical_preconditioned"] - all_vectors["saved_plus_delta_preconditioned"])), 1e-10)
        np.savez_compressed(output / "all_gradients_metric.npz", keys=np.array(functional.keys), x=saved_gradient["x"],
                            inverse_diagonal=saved_gradient["inverse_diagonal"], **all_vectors,
                            **{f"projector_{i}": row[1] for i, row in enumerate(layout)},
                            **{f"inverse_blocks_{i}": a for i, a in enumerate(inverse)})
        differences = {}
        for name in ("full_logical", "full_raw_quotient"):
            differences[name] = {kind: float(np.linalg.norm(all_vectors[name + "_" + kind] - all_vectors["saved_" + kind]))
                                 for kind in ("radial", "raw_coordinate", "tangent", "preconditioned")}
        direction_records = []
        for index in (0, 1):
            direction = archive(f"direction{index}")["direction"].astype(wide)
            central = load(f"direction{index}_central")
            check(f"d{index}_original_tangent", np.max(abs(tangent(direction) - direction)), 1e-12)
            actions = {name: float(all_vectors[name + "_tangent"].astype(wide) @ direction) for name in vector_metrics}
            anchor = central["Rayleigh_central"]
            centered = load("centering_result" if index == 1 else "centering0_result")
            # The archived extension keeps its original fixed multiplier.
            # Apply its derivative to the fresh unprojected ambient action.
            ambient_action = float(all_vectors["full_raw_quotient_raw_coordinate"] @ direction)
            centered_action = ambient_action - centered["ideal_normal_derivative"]
            check(f"d{index}_NEW_full_partial_centered_control",
                  abs(centered_action - centered["centered_Rayleigh_central"]), central["gate"])
            direction_records.append(dict(direction=index, actions=actions, old_complete_mixed_Rayleigh_central=anchor,
                full_partial_minus_old_radial_anchor=actions["full_raw_quotient"] - anchor,
                old_analytic=central["analytic_saved_g"] if index == 1 else central["analytic"],
                old_original_gate=central["gate"], old_gate_status_upgraded=False,
                finite_CI_response_not_in_partial_gradient=True,
                fresh_full_ambient_ideal_action=ambient_action,
                archived_fixed_multiplier_normal_derivative=centered["ideal_normal_derivative"],
                fresh_full_partial_centered_action=centered_action,
                conditional_centered_reference=centered["centered_Rayleigh_central"],
                fresh_full_partial_centered_error=centered_action - centered["centered_Rayleigh_central"],
                scope="new samepoint normalized-table HF partial compared to existing fixed-multiplier radial finite anchor; old failed FD not rerun"))
        for name, expected in (("H", 1), ("CI", 1), ("gradient_from_weights", 2),
                               ("initial_hS", 7), ("retained_RDM", 1), ("unfiltered_target_RDM", 5),
                               ("tracker_resolve", 2), ("tracker_SVD", 10), ("tracker_overlap", 3), ("Dirac", 0)):
            check("completed_calls_" + name, abs(completed[name] - expected))
        for name, module in tuple(sys.modules.items()):
            if ((name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None)
                    and not Path(module.__file__).resolve().is_relative_to(root.resolve())):
                raise ValueError("Physical source import drift")
        metadata(contract)
        write(output / "result.json", {"samepoint_full_normalized_gradient_control_completed": True,
              "all_numeric_controls_passed": all(c["passed"] for c in controls), "controls": controls,
              "source_map_sha256": contract["source"]["map_sha256"], "original_metric_sha256": fingerprint,
              "reported_EO_Ha": energy, "targets": rows, "gradient_metrics": vector_metrics,
              "whole_vector_differences": differences, "samepoint_direction_diagnostics": direction_records,
              "started": started, "completed": completed, "scope":"full normalized unfiltered public angular-table HF partial plus exact-known raw-h/ket-norm ambient correction",
              "partial_gradient_contains_finite_CI_eigenresponse": False,
              "table_angular_and_aggregation_roundoff_error_bound": False,
              "historical_derivative_failures_unchanged": True, "optimizer_authorized": False,
              "production_callback_or_uniform_error_bound": False, "full_MCDHF_or_experimental_certified": False,
              "elapsed_uncontrolled_s": time.monotonic() - begin,
              "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "CPU":31, "accounting_CPUs":accounting,
              "driver_sha256":sha(__file__), "contract_sha256":sha(HERE / "contract.json")})
        flush("closed_complete")
    except BaseException as error:
        import traceback
        flush("failed")
        write(output / "failure.json", {"error":str(error), "traceback":traceback.format_exc(),
              "started":started, "completed":completed, "scientific_success_claim":False,
              "historical_failures_unchanged":True, "optimizer_authorized":False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-contract")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    contract = json.loads((HERE / "contract.json").read_text())
    if args.metadata_only:
        metadata(contract)
        ast.parse(Path(__file__).read_text())
        print(json.dumps({"metadata_passed":True, "numeric_imports":0, "physical_calls":0}))
        return
    if not args.execution_contract:
        raise ValueError("Root execution contract required; no numerical imports performed")
    execute(contract, json.loads(Path(args.execution_contract).read_text()))


if __name__ == "__main__":
    main()
