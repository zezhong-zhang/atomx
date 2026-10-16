"""Corrected-source full baseline and two fixed constrained derivative controls.

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
    if contract["source"]["parent_map_sha256"] != identity["parent_source_map_sha256"]:
        raise ValueError("Wrong corrected source lineage")
    if identity["changed_paths"] != ["src/atomx/numerics/bspline/convol.py"]:
        raise ValueError("Unqualified additional implementation change")
    scale = json.loads((REPO / contract["input_roles"]["scale_result"]["path"]).read_text())
    receipt = json.loads((REPO / contract["input_roles"]["scale_root_execution"]["path"]).read_text())
    if not (scale["positive_inner_order_discriminator_completed"] is True
            and scale["all_declared_discriminator_controls_passed"] is True
            and scale["fixed_CI_expectation_only"] is True
            and scale["new_spectrum_or_experiment_qualification"] is False
            and scale["source_maps"] == {"ordinary": contract["source"]["parent_map_sha256"],
                                         "candidate": contract["source"]["map_sha256"]}
            and receipt["exit_code"] == 0):
        raise ValueError("Unclosed corrected scalar/real-point discriminator prerequisite")
    old = json.loads((REPO / contract["input_roles"]["old_fullnorm_result"]["path"]).read_text())
    if not (old["all_numeric_controls_passed"] is True and old["source_map_sha256"] == contract["source"]["parent_map_sha256"]):
        raise ValueError("Original500ac matched-point reference is unqualified")
    return root


def execute(contract, authority):
    root = metadata(contract)
    if contract["pending_prerequisites"]:
        raise ValueError("Corrected-source analytic and real-point discriminator inputs pending")
    if authority.get("execution_authorized_by_root") is not True:
        raise ValueError("Exact root execution contract required")
    required = {"execution_authorized_by_root": True,
                "contract_sha256": sha(HERE / "contract.json"), "driver_sha256": sha(__file__),
                "preparation_sha256": sha(HERE / "preparation_verification.json"),
                "source_map_sha256": contract["source"]["map_sha256"],
                "max_H": 5, "max_CI": 5, "max_gradient_from_weights": 3,
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
    if any(n == "atomx" or n.startswith("atomx.") or n == "numpy" or n.startswith("numpy.") for n in sys.modules):
        raise ValueError("No preimported physical source")
    cache_path = Path(os.environ["NUMBA_CACHE_DIR"]).resolve()
    if not cache_path.is_relative_to(HERE.resolve()) or any(cache_path.iterdir()):
        raise ValueError("Fresh isolated corrected-source NumBa cache required")
    sys.path.insert(0, str(root))
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    begin = time.monotonic()
    started = dict.fromkeys(contract["calls"], 0)
    completed = dict(started)
    controls, rows, raw_operators = [], [], {}
    tracking_record = None

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
        import atomx.physics.csf.radial_increment as radial_increment
        import atomx.physics.csf.tracking as tracking_module
        from atomx.angular.csf_hamiltonian import csf_signature
        from atomx.config.csf import enumerate_jj_csfs
        from atomx.methods.ci import CIResult
        from atomx.numerics.bspline.orbital_optimization import (
            _polar_retraction,
            _residual_metric_sha256,
        )
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
                nonlocal tracking_record
                started[name] += 1
                flush("before_" + name)
                if started[name] > contract["calls"][name]:
                    raise RuntimeError("Attempted call cap " + name)
                value = fn(*args, **kwargs)
                completed[name] += 1
                if name == "tracker_resolve":
                    tracking_record = value
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
        radial_increment.build_dirac_matrices = wrapper("mixed_hS", dirac.build_dirac_matrices, True)
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
        _polar_retraction = wrapper("polar_retraction", _polar_retraction)
        np.linalg.eigh = wrapper("polar_numpy_eigh", np.linalg.eigh)

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

        functional.radial_density.contract = wrapper("retained_RDM", functional.radial_density.contract)
        functional.radial_density.contract_bilinear = wrapper("unfiltered_target_RDM", functional.radial_density.contract_bilinear)
        energy, production_gradient, ci = functional.evaluate(waves, gradient=True)
        retained = dict(functional.last_weights)
        h = ci.hamiltonian_for(csfs, waves, functional.keys, data, 70, 123,
                              nuclear_potential=functional.nuclear_potential, max_determinants=2000)
        np.savez_compressed(output / "fresh_baseline.npz", H=h, energies=ci.state_energies,
                            mixing=ci.mixing_coefficients, keys=np.array(functional.keys),
                            P=np.array([waves[k][0] for k in functional.keys]), Q=np.array([waves[k][1] for k in functional.keys]))
        write(output / "owner.json", ci.hamiltonian_snapshot.identity_payload())
        np.savez_compressed(output / "fresh_production_gradient.npz", keys=np.array(functional.keys),
                            radial_gradient=np.array([production_gradient[k] for k in functional.keys]))
        source_differences = {}
        for name, a in (("H", h), ("energies", ci.state_energies), ("mixing", ci.mixing_coefficients)):
            source_differences[name] = dict(bitwise_equal_old500ac=bool(np.array_equal(a, saved[name])
                and a.tobytes() == saved[name].tobytes()), max_abs_difference=float(np.max(abs(a - saved[name]))))
        # Physical input ownership is source-independent; H source identity is separately sealed.
        check("fresh_owner_physical_payload", plain(ci.hamiltonian_snapshot.identity_payload()) != load("saved_point_owner"))
        check("full_H_symmetric", not np.array_equal(h, h.T))
        check("full_CI_Gram", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients - np.eye(1046))), 1e-10)
        check("full_CI_residual", np.max(np.linalg.norm(h @ ci.mixing_coefficients - ci.mixing_coefficients * ci.state_energies, axis=0)), 1e-8)
        weights, tracking = tracking_record
        check("five_original_equalweights", not (np.count_nonzero(weights) == 5 and np.array_equal(weights[weights > 0], np.repeat(.2, 5))))
        _, blocks = ci_level_blocks(csfs, ci, require_full=True)
        matched = target_bindings(targets, tracking)
        check("fresh_reported_EO_own_CI", energy.hex() != float(weights @ ci.state_energies).hex())
        old_targets = load("old_fullnorm_result")["targets"]
        old_rows = {row["label"]: row for row in old_targets}
        for t in targets:
            uid = (t["twoJ"], t["parity"], t["models"]["DC"]["local_root"])
            check("old500ac_physical_UID_" + t["label"], tuple(old_rows[t["label"]]["UID"]) != uid)
            old_rows[t["label"]]["energy_Ha"] = float(saved["energies"][old_rows[t["label"]]["column"]])
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
                             energy_Ha=float(ci.state_energies[column]), old500ac_energy_Ha=float(old_rows[t["label"]]["energy_Ha"]),
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
        raw = np.array([production_gradient[k] for k in functional.keys])
        delta_array = np.array([delta_g[k] for k in functional.keys])
        full_array = np.array([full_g[k] for k in functional.keys])
        reconstructed = raw + delta_array
        np.savez_compressed(output / "logical_gradients.npz", keys=np.array(functional.keys), production_raw=raw,
                            delta=delta_array, production_plus_delta=reconstructed, direct_full=full_array)
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
        for name, a in (("production", raw), ("old500ac", saved_gradient["radial_gradient"]), ("delta", delta_array), ("full_logical", full_array),
                        ("production_plus_delta", reconstructed), ("full_raw_quotient", corrected_array)):
            cg, tg = coordinate(a), tangent(coordinate(a))
            pg = metric_action(tg)
            all_vectors.update({name + "_radial": a, name + "_raw_coordinate": cg,
                                name + "_tangent": tg, name + "_preconditioned": pg})
            vector_metrics[name] = dict(radial_norm=float(np.linalg.norm(a)), tangent_norm=float(np.linalg.norm(tg)),
                                         original_metric_residual=float(np.linalg.norm(pg)),
                                         original_metric_energy_indicator=float(.5 * abs(tg @ pg)))
        check("old500ac_raw_coordinate_saved_replay", np.max(abs(all_vectors["old500ac_raw_coordinate"] - saved_gradient["raw_coordinate_gradient"])), 1e-9)
        check("old500ac_pg_saved_replay", np.max(abs(all_vectors["old500ac_preconditioned"] - saved_gradient["preconditioned_gradient"])), 1e-10)
        check("fullvector_pg_linearity", np.max(abs(all_vectors["full_logical_preconditioned"] - all_vectors["production_plus_delta_preconditioned"])), 1e-10)
        np.savez_compressed(output / "all_gradients_metric.npz", keys=np.array(functional.keys), x=saved_gradient["x"],
                            inverse_diagonal=saved_gradient["inverse_diagonal"], **all_vectors,
                            **{f"projector_{i}": row[1] for i, row in enumerate(layout)},
                            **{f"inverse_blocks_{i}": a for i, a in enumerate(inverse)})
        differences = {"corrected_production_minus_old500ac": {
            kind: float(np.linalg.norm(all_vectors["production_" + kind] - all_vectors["old500ac_" + kind]))
            for kind in ("radial", "raw_coordinate", "tangent", "preconditioned")}}
        for name in ("full_logical", "full_raw_quotient"):
            differences[name] = {kind: float(np.linalg.norm(all_vectors[name + "_" + kind] - all_vectors["production_" + kind]))
                                 for kind in ("radial", "raw_coordinate", "tangent", "preconditioned")}
        from constraint_bookkeeping import ConstraintBookkeeping

        from atomx.constants import HARTREE_CM1

        def spectrum(point_rows):
            values = {r["label"]: r["energy_Ha"] for r in point_rows}
            if len(values) != 5 or set(values) != set(old_rows):
                raise ValueError("Missing physical spectrum targets")
            ground = values["6s_1/2"]
            result = {label: (v - ground) * HARTREE_CM1 for label, v in values.items()}
            for prefix, low, high, wl, wh in (("5d", "5d_3/2", "5d_5/2", 4, 6),
                                             ("6p", "6p_1/2", "6p_3/2", 2, 4)):
                result[prefix + "_centroid"] = ((wl * values[low] + wh * values[high]) / (wl + wh) - ground) * HARTREE_CM1
                result[prefix + "_splitting"] = (values[high] - values[low]) * HARTREE_CM1
            return result

        old_spectrum, new_spectrum = spectrum(list(old_rows.values())), spectrum(rows)
        for row in rows:
            row["source_delta_energy_Ha"] = row["energy_Ha"] - row["old500ac_energy_Ha"]
            row["transition_cm1"] = new_spectrum[row["label"]]
            row["source_delta_transition_cm1"] = new_spectrum[row["label"]] - old_spectrum[row["label"]]
        write(output / "baseline_source_comparison.json", {"old_source_map": contract["source"]["parent_map_sha256"],
              "corrected_source_map": contract["source"]["map_sha256"], "arrays": source_differences,
              "targets": rows, "old_spectrum_cm1": old_spectrum, "corrected_spectrum_cm1": new_spectrum,
              "original_metric": fingerprint, "corrected_owner_input_sha256": ci.hamiltonian_snapshot.input_sha256,
              "corrected_owner_H_sha256": ci.hamiltonian_snapshot.matrix_sha256,
              "source_implementation_is_separate_from_owner_physical_input_identity": True,
              "owner_header_sha256": sha(output / "owner.json"), "baseline_npz_sha256": sha(output / "fresh_baseline.npz"),
              "no_corrected_value_equals_oldsource_gate": True})
        write(output / "baseline_gradient_qualification.json", {"metrics": vector_metrics,
              "normalization_wholevector_controls_passed": True, "old500ac_gradient_is_comparison_only": True,
              "orbital_tolerance": 1e-7, "energy_indicator_tolerance_Ha": 1e-8,
              "corrected_production_stationary": vector_metrics["production"]["original_metric_residual"] < 1e-7
                  and vector_metrics["production"]["original_metric_energy_indicator"] < 1e-8,
              "new_derivative_qualification_not_yet_executed": True})
        flush("complete_baseline_qualified_before_any_finite_trials")
        centering = ConstraintBookkeeping(functional.keys, waves, full_g, full_weights, raw_operators,
                                         functional.one_body, layout,
                                         max_products=contract["resources"]["max_exact_scalar_products"])

        def unpack(coordinates):
            result = {}
            for keys, projector, shape, where in layout:
                c = projector @ coordinates[where].reshape(shape)
                for j, k in enumerate(keys):
                    result[k] = c[:123, j].copy(), c[123:, j].copy()
            return result

        def complete_trial(name, trial_waves):
            trial_energy, trial_ci = functional.evaluate(trial_waves, gradient=False)
            trial_h = trial_ci.hamiltonian_for(csfs, trial_waves, functional.keys, data, 70, 123,
                    nuclear_potential=functional.nuclear_potential, max_determinants=2000)
            np.savez_compressed(output / (name + ".npz"), H=trial_h, energies=trial_ci.state_energies,
                                mixing=trial_ci.mixing_coefficients, keys=np.array(functional.keys),
                                P=np.array([trial_waves[k][0] for k in functional.keys]),
                                Q=np.array([trial_waves[k][1] for k in functional.keys]))
            write(output / (name + "_owner.json"), trial_ci.hamiltonian_snapshot.identity_payload())
            check(name + "_whole_H_symmetry", not np.array_equal(trial_h, trial_h.T))
            check(name + "_all1046_CI_Gram", np.max(abs(trial_ci.mixing_coefficients.T @ trial_ci.mixing_coefficients - np.eye(1046))), 1e-10)
            check(name + "_all1046_CI_residual", np.max(np.linalg.norm(trial_h @ trial_ci.mixing_coefficients - trial_ci.mixing_coefficients * trial_ci.state_energies, axis=0)), 1e-8)
            for keys, projector, _, _ in layout:
                metric = functional.metric[keys[0][1]]
                c = np.column_stack([np.concatenate(trial_waves[k]) for k in keys])
                check(name + "_radial_Gram_" + str(keys[0][1]), np.max(abs(c.T @ metric @ c - np.eye(len(keys)))), 1e-10)
                outside = c - projector @ (projector.T @ metric @ c)
                check(name + "_originalP_leak_" + str(keys[0][1]), np.max(np.sum(outside * (metric @ outside), axis=0)), 1e-12)
            trial_weights, bindings = tracking_record
            check(name + "_equal5_originalweights", not (np.count_nonzero(trial_weights) == 5 and np.array_equal(trial_weights[trial_weights > 0], np.repeat(.2, 5))))
            bindings = target_bindings(targets, bindings)
            _, trial_blocks = ci_level_blocks(csfs, trial_ci, require_full=True)
            trial_rows = []
            for t in targets:
                uid = (t["twoJ"], t["parity"], t["models"]["DC"]["local_root"])
                j, p, rank = bindings[uid]["selected_levels"][0]
                column = int(trial_blocks[j, p][rank])
                trial_rows.append(dict(label=t["label"], UID=uid, column=column,
                                       energy_Ha=float(trial_ci.state_energies[column]), tracking=bindings[uid]))
            write(output / (name + "_targets.json"), {"targets": trial_rows, "spectrum_cm1": spectrum(trial_rows),
                  "owner_input_sha256": trial_ci.hamiltonian_snapshot.input_sha256,
                  "owner_H_sha256": trial_ci.hamiltonian_snapshot.matrix_sha256})
            return trial_energy, trial_ci

        direction_records = []
        for index in (0, 1):
            direction = archive(f"direction{index}")["direction"]
            check(f"d{index}_original_tangent", np.max(abs(tangent(direction) - direction)), 1e-12)
            check(f"d{index}_original_unit_norm", abs(np.linalg.norm(direction) - 1), 1e-12)
            ideal = centering.ideal(direction)
            diagnostics, normals, trial_energies = [], [], []
            for sign in (1., -1.):
                name = f"direction{index}_" + ("plus" if sign > 0 else "minus")
                candidate = saved_gradient["x"] + sign * 1e-6 * direction
                for _, _, shape, where in layout:
                    candidate[where] = _polar_retraction(candidate[where].reshape(shape)).ravel()
                trial_waves, increment = unpack(candidate), unpack(candidate - saved_gradient["x"])
                np.savez_compressed(output / (name + "_coordinates.npz"), candidate_x=candidate,
                                    explicit_increment_P=np.array([increment[k][0] for k in functional.keys]),
                                    explicit_increment_Q=np.array([increment[k][1] for k in functional.keys]))
                trial_energy, trial_ci = complete_trial(name, trial_waves)
                diagnostic = functional.mixed_ci_energy_difference(waves, trial_waves, ci, trial_ci, increments=increment)
                check(name + "_mixed_diagnostic_scope", not (diagnostic["diagnostic_only"] is True
                       and diagnostic["certified_error_bound"] is False and diagnostic["optimizer_adapter"] is False))
                normal = centering.trial(trial_waves, increment)
                diagnostics.append(diagnostic)
                normals.append(normal)
                trial_energies.append(trial_energy)
                write(output / (name + "_finite_diagnostic.json"), {"mixed": diagnostic,
                      "fixed_fullnormalized_Lambda_normal_Ha": float(normal["normal"]),
                      "fixed_explicit_normal_Ha": float(normal["normal_explicit"]),
                      "full_signed_Gram_blocks": normal["blocks"], "exact_products": centering.products})
            def finite(field, diagnostics=diagnostics):
                return float((Fraction.from_float(float(diagnostics[0][field])) -
                    Fraction.from_float(float(diagnostics[1][field]))) / Fraction.from_float(2e-6))
            centered = float((Fraction.from_float(float(diagnostics[0]["rayleigh_change"])) - normals[0]["normal"]
                            - Fraction.from_float(float(diagnostics[1]["rayleigh_change"])) + normals[1]["normal"])
                            / Fraction.from_float(2e-6))
            analytic = float(ideal["centered"])
            gate = max(1e-8, 1e-5 * abs(analytic))
            ledger = {"direction": index, "archived_seed": 20261011 + index, "h": 1e-6,
                      "fresh_fullnormalized_ambient_ideal": float(ideal["ambient"]),
                      "fresh_fixed_Lambda_ideal_normal": float(ideal["normal"]),
                      "new_centered_analytic": analytic, "new_complete_centered_Rayleigh_central": centered,
                      "new_centered_error": centered - analytic, "original_abs_rel_gate": gate,
                      "mixed_reported_central": finite("reported_change"), "mixed_Rayleigh_central": finite("rayleigh_change"),
                      "direct_stored_quotient_central_forensic": finite("direct_retained_quotient_difference"),
                      "ordinary_reported_central_resolution_only": (trial_energies[0] - trial_energies[1]) / 2e-6,
                      "oneULP_over_h": float(np.spacing(abs(energy)) / 1e-6),
                      "finite_CI_residual_norm_phase_response_retained": True,
                      "fixed_Lambda_finite_remainder_has_no_uniform_bound": True,
                      "production_objective_remains_reported_default": True,
                      "historical_failed_gates_unchanged": True, "optimizer_authorized": False}
            direction_records.append(ledger)
            write(output / f"direction{index}_central.json", ledger)
            check(f"d{index}_new_corrected_full_centered_gate", abs(centered - analytic), gate)
        for name, expected in (("H", 5), ("CI", 5), ("gradient_from_weights", 3),
                               ("initial_hS", 7), ("mixed_hS", 28), ("retained_RDM", 1),
                               ("unfiltered_target_RDM", 25), ("tracker_resolve", 13),
                               ("tracker_SVD", 65), ("tracker_overlap", 14), ("CI_eigh", 25),
                               ("polar_retraction", 28), ("Dirac", 0)):
            check("completed_calls_" + name, abs(completed[name] - expected))
        for name, module in tuple(sys.modules.items()):
            if ((name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None)
                    and not Path(module.__file__).resolve().is_relative_to(root.resolve())):
                raise ValueError("Physical source import drift")
        metadata(contract)
        write(output / "result.json", {"corrected_samepoint_full_gradient_and_centered_directions_completed": True,
              "all_numeric_controls_passed": all(c["passed"] for c in controls), "controls": controls,
              "source_map_sha256": contract["source"]["map_sha256"], "original_metric_sha256": fingerprint,
              "reported_EO_Ha": energy, "targets": rows, "gradient_metrics": vector_metrics,
              "whole_vector_differences": differences, "samepoint_direction_diagnostics": direction_records,
              "started": started, "completed": completed, "source_differences": source_differences, "exact_constraint_products": centering.products,
              "scope":"corrected-source normalized full public HF partial and two fixed Q-Lagrange finite controls; ordinary reported production untouched",
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
