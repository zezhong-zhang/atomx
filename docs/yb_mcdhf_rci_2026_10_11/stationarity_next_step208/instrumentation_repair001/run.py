"""One prospective literal208 original-metric directional control.

Preparation reads source/JSON only. Numerical imports, archives and public
producers require a separate exact root execution seal. No optimizer is called.
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
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "pyproject.toml").is_file())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    # NumPy longdouble.item()/tolist() can return another longdouble scalar.
    # Handle native numerical scalars before recursive array conversion. JSON
    # diagnostics use Python scalar rounding; numerical arrays stay in NPZ.
    if hasattr(value, "dtype") and getattr(value, "ndim", None) == 0:
        kind = value.dtype.kind
        if kind == "f":
            return float(value)
        if kind in "iu":
            return int(value)
        if kind == "b":
            return bool(value)
        if kind == "c":
            raise TypeError("Complex diagnostic scalar is unsupported")
        return plain(value.item())
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


def metadata(contract, *, archives=False):
    """Do not read even NPZ bytes during source-only preparation."""
    roles = contract["input_roles"]
    for row in roles.values():
        path = REPO / row["path"]
        if path.suffix == ".npz" and not archives:
            if not path.is_file():
                raise ValueError("Missing declared future archive")
            continue
        if sha(path) != row["sha256"]:
            raise ValueError("Changed input: " + row["path"])
    identity = load(REPO / roles["candidate_source_identity"]["path"])
    if identity["source_map_sha256"] != contract["source"]["map_sha256"]:
        raise ValueError("Changed frozen physical source")
    source = REPO / contract["source"]["archive"]
    for name, expected in identity["all_atomx_python_sha256"].items():
        if sha(source / name.removeprefix("src/")) != expected:
            raise ValueError("Changed archived source: " + name)
    point = load(REPO / roles["literal208_manifest"]["path"])
    prior = load(REPO / roles["campaign_result"]["path"])
    expected = contract["metric"]
    if not (point["checkpoint_qualified"] is True and point["global_accepted_step"] == 208
            and point["stationarity_passed"] is False
            and point["array_sha256"] == roles["literal208_arrays"]["sha256"]
            and point["source_map_sha256"] == contract["source"]["map_sha256"]
            and point["state_sha256"] == expected["state_sha256"]
            and point["metric_sha256"] == expected["residual_metric_sha256"]
            and point["geometry_sha256"] == expected["geometry_sha256"]
            and point["objective_sha256"] == expected["objective_sha256"]
            and point["owner_input_sha256"] == contract["point"]["owner_input_sha256"]
            and point["owner_H_sha256"] == contract["point"]["owner_H_sha256"]
            and prior["status"] == "bounded_public_optimizer_returned"
            and prior["error"] is None and prior["total_accepted_steps"] == 208):
        raise ValueError("Wrong closed literal208 point")
    objective = load(REPO / roles["campaign_objective"]["path"])
    payload_hash = hashlib.sha256(json.dumps(objective["payload"], sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    if not (objective["sha256"] == expected["objective_sha256"] == payload_hash):
        raise ValueError("Changed declared objective")
    if any(name == "numpy" or name.startswith("numpy.") or name == "atomx"
           or name.startswith("atomx.") for name in sys.modules):
        raise ValueError("Numerical imports precede admission")
    return source


def target_bindings(targets, bindings):
    """Original reference UID, never global CI column or insertion order."""
    mapped = {}
    for row in bindings:
        if len(row["reference_levels"]) != 1 or len(row["selected_levels"]) != 1:
            raise ValueError("Unresolved physical target span; rank-one consumer unsupported")
        uid = tuple(row["reference_levels"][0])
        if uid in mapped or tuple(row["selected_levels"][0][:2]) != uid[:2]:
            raise ValueError("Duplicate or changed original UID")
        mapped[uid] = row
    expected = {(t["twoJ"], t["parity"], t["models"]["DC"]["local_root"]) for t in targets}
    if len(expected) != 5 or set(mapped) != expected:
        raise ValueError("Incomplete original five targets")
    return mapped


def install_resolution_capture(tracker_type, target_tracker, capture):
    """Observe only the target; delegate to the existing class counter wrapper.

    Frozen tracker instances remain untouched. All other instances execute
    their original resolution and return values without capture side effects.
    """
    previous = tracker_type.resolve

    def resolve(self, *args, **kwargs):
        result = previous(self, *args, **kwargs)
        if self is target_tracker:
            capture["weights"], capture["bindings"] = result
        return result

    tracker_type.resolve = resolve
    return previous


def execute(contract, seal):
    source = metadata(contract, archives=True)
    caps = contract["resources"]
    required = {"execution_authorized_by_root": True, "driver_sha256": sha(__file__),
                "contract_sha256": sha(HERE / "contract.json"),
                "preparation_sha256": sha(HERE / "preparation_verification.json"),
                "source_map_sha256": contract["source"]["map_sha256"],
                "CPU": caps["CPU"], "wall_seconds": caps["wall_seconds"],
                "address_space_GiB": caps["address_space_GiB"],
                "max_H": 5, "max_CI": 5, "max_gradient_from_weights": 2,
                "max_optimizer": 0, "max_Dirac": 0}
    if any(seal.get(k) != value for k, value in required.items()):
        raise ValueError("Exact root execution seal required")
    proof = load(HERE / "preparation_verification.json")
    if not (proof["all_source_and_JSON_checks_passed"] is True
            and proof["driver_sha256"] == required["driver_sha256"]
            and proof["contract_sha256"] == required["contract_sha256"]):
        raise ValueError("Wrong prepared consumer")
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError("Single-thread admission required")
    available, host_load = sorted(os.sched_getaffinity(0)), os.getloadavg()
    if caps["CPU"] not in available or host_load[0] > seal["maximum_system_load1"]:
        raise ValueError("Host admission failed")
    cache = Path(os.environ["NUMBA_CACHE_DIR"])
    if not cache.is_relative_to(HERE) or not cache.is_dir() or any(cache.iterdir()):
        raise ValueError("New empty owned cache required")
    # A sole attempt marker survives failures; no automatic rerun.
    with (HERE / "once_marker.json").open("x") as marker:
        json.dump({"seal_sha256": sha(HERE / "execution_contract.json"),
                   "driver_sha256": required["driver_sha256"]}, marker)
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS, (int(caps["address_space_GiB"] * 2**30),) * 2)
    os.sched_setaffinity(0, {caps["CPU"]})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(RuntimeError("300s wall cap")))
    signal.alarm(caps["wall_seconds"])
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))

    import numpy as np
    from scipy import linalg

    import atomx.methods.csf.variational as variational
    import atomx.numerics.bspline.dirac_assembly as dirac
    import atomx.numerics.bspline.orbital_optimization as optimizer
    import atomx.physics.csf.matrix as matrix
    import atomx.physics.csf.radial_increment as increment_module
    import atomx.physics.csf.tracking as tracking_module
    from atomx.angular.csf_hamiltonian import csf_signature
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.ci import CIResult
    from atomx.physics.csf.levels import ci_level_blocks
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.physics.csf.tracking import CIBasisTracking
    from atomx.potentials.nuclear import make_nuclear_potential

    for name, module in tuple(sys.modules.items()):
        if ((name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None)
                and not Path(module.__file__).resolve().is_relative_to(source.resolve())):
            raise ValueError("Mixed physical source epoch")
    limits = {"H": 5, "CI": 5, "gradient": 2, "initial_hS": 7, "CI_internal_hS": 35,
              "mixed_hS": 28, "scipy_eigh": 50, "polar": 28, "polar_eigh": 28,
              "tracking_resolve": 13, "tracking_SVD": 65, "tracking_overlap": 14,
              "bilinear_RDM": 25, "retained_H_getter_requests": 21,
              "Dirac": 0, "optimizer": 0, "eigvalsh": 0}
    attempted, completed = dict.fromkeys(limits, 0), dict.fromkeys(limits, 0)
    checks, raw_operators, records, certificates = [], {}, [], {}
    local_target_levels = {}
    start, phase = time.monotonic(), "admitted"
    roles = contract["input_roles"]

    def flush():
        write(output / "progress.json", {"phase": phase, "attempted": attempted,
              "completed": completed, "checks": checks, "finite_records": records,
              "elapsed_uncontrolled_s": time.monotonic() - start})

    def check(name, error, gate=0.):
        passed = bool(np.isfinite(error) and error <= gate)
        checks.append(dict(name=name, error=float(error), gate=gate, passed=passed))
        if not passed:
            flush()
            raise ValueError("Scientific gate failed: " + name)

    def wrap(name, function, *, raw=False):
        def call(*args, **kwargs):
            if attempted[name] >= limits[name]:
                raise RuntimeError("Producer cap before " + name)
            attempted[name] += 1
            result = function(*args, **kwargs)
            completed[name] += 1
            if raw:
                k = int(args[1])
                if k not in raw_operators:
                    raw_operators[k] = tuple(v.copy() for v in result)
                else:
                    check(name + "_rawh_" + str(k), not np.array_equal(result[0], raw_operators[k][0]))
                    check(name + "_S_" + str(k), not np.array_equal(result[1], raw_operators[k][1]))
            return result
        return call

    builder = dirac.build_dirac_matrices
    variational.build_dirac_matrices = wrap("initial_hS", builder, raw=True)
    matrix.build_dirac_matrices = wrap("CI_internal_hS", builder, raw=True)
    increment_module.build_dirac_matrices = wrap("mixed_hS", builder, raw=True)
    matrix.build_ci_matrix = wrap("H", matrix.build_ci_matrix)
    variational.ci_solve = wrap("CI", variational.ci_solve)
    CIResult.hamiltonian_for = wrap("retained_H_getter_requests", CIResult.hamiltonian_for)
    variational.CSFOrbitalFunctional.gradient_from_weights = wrap(
        "gradient", variational.CSFOrbitalFunctional.gradient_from_weights)
    dirac.solve_dirac = wrap("Dirac", dirac.solve_dirac)
    optimizer.optimize_orbitals = wrap("optimizer", optimizer.optimize_orbitals)
    optimizer._polar_retraction = wrap("polar", optimizer._polar_retraction)
    np.linalg.eigh = wrap("polar_eigh", np.linalg.eigh)
    np.linalg.eigvalsh = wrap("eigvalsh", np.linalg.eigvalsh)
    linalg.eigh = wrap("scipy_eigh", linalg.eigh)
    CIBasisTracking.resolve = wrap("tracking_resolve", CIBasisTracking.resolve)
    tracking_module.ci_subspace_overlaps = wrap("tracking_overlap", tracking_module.ci_subspace_overlaps)
    np.linalg.svd = wrap("tracking_SVD", np.linalg.svd)

    def role(name):
        return REPO / roles[name]["path"]

    def archive(name):
        with np.load(role(name), allow_pickle=False) as saved:
            return {k: saved[k].copy() for k in saved.files}

    def import_helper(name, name_role):
        spec = importlib.util.spec_from_file_location(name, role(name_role))
        helper = importlib.util.module_from_spec(spec)
        sys.modules[name] = helper
        spec.loader.exec_module(helper)
        return helper

    error = None
    try:
        radial, saved, geometry = archive("geometry_workspace"), archive("literal208_arrays"), archive("screened_geometry")
        data = {k[6:]: v.item() if v.ndim == 0 else v.copy() for k, v in radial.items() if k.startswith("basis_")}
        models = load(role("screened_models"))["models"]
        old531 = [c for row in models["baseline"]["configs"] for c in enumerate_jj_csfs(row["config"])]
        parent = [c for row in models["balanced_virtual"]["configs"] for c in enumerate_jj_csfs(row["config"])]
        sectors = {(j, p) for j, p, _ in contract["fixed"]["target_blocks"]}
        csfs = [c for c in parent if (c.total_J2, c.parity) in sectors]
        check("complete69e_531_2163_1046", not (len(old531) == 531 and len(parent) == 2163
              and len(csfs) == 1046 and {c.n_electrons for c in csfs} == {69}))
        signatures = [json.dumps(plain((s.subshells, s.coupling, c.total_J2, c.parity)), separators=(",", ":"))
                      for c in csfs for s in [csf_signature(c)]]
        check("signed_ordered_CSF_identity", signatures != load(role("screened_basis"))["balanced_virtual"]["retained_signatures"])
        targets = load(role("tracking_binding"))["targets"]
        average = CIStateAverage(tuple((t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.) for t in targets))
        reference = archive("tracking_ci")
        reference_waves = {tuple(map(int, key)): (p.copy(), q.copy()) for key, p, q in
                           zip(radial["orbital_keys"], radial["P"], radial["Q"], strict=True)}
        tracker = CIBasisTracking(average, old531, CIResult(reference["energies_Ha"], reference["mixing"]),
                                 reference_waves, {k: geometry[f"s_{k}"] for k in {v[1] for v in saved["keys"]}},
                                 max_determinant_pairs=1_000_000)
        nuclear = make_nuclear_potential(70, "fermi", A=175)
        functional = variational.CSFOrbitalFunctional(csfs, data, 70, nuclear,
            state_average=average, level_tracking=tracker, retain_hamiltonian=True, max_determinants=2000)
        functional.radial_density.contract_bilinear = wrap("bilinear_RDM", functional.radial_density.contract_bilinear)
        check("whole29_order", tuple(functional.keys) != tuple(map(tuple, contract["fixed"]["keys29"])))
        for k, (raw_h, s) in raw_operators.items():
            logical = np.tril(raw_h) + np.tril(raw_h, -1).T
            check("logical_L_rawh_" + str(k), not np.array_equal(logical, functional.one_body[k]))
            check("logical_saved_h_" + str(k), not np.array_equal(logical, geometry[f"h_{k}"]))
            check("saved_S_" + str(k), not np.array_equal(s, geometry[f"s_{k}"]))
        np.savez_compressed(output / "raw_hS_connection.npz",
            **{f"raw_h_{k}": pair[0] for k, pair in raw_operators.items()},
            **{f"S_{k}": pair[1] for k, pair in raw_operators.items()},
            **{f"logical_h_{k}": functional.one_body[k] for k in raw_operators})
        layout = [(tuple(map(tuple, row["keys"])), saved[f"projector_{i}"], tuple(row["shape"]), slice(*row["slice"]))
                  for i, row in enumerate(load(role("saved_layout")))]
        x = saved["x"].copy()
        check("full_coordinate_dimension", not (x.dtype == np.float64 and x.shape == (3451,)))
        inverse = tuple(tuple(m for m in saved[f"inverse_blocks_{i}"]) for i in range(7))
        metric_hash = optimizer._residual_metric_sha256(layout, saved["inverse_diagonal"], inverse)
        check("actual_original_metric_SHA", metric_hash != contract["metric"]["residual_metric_sha256"])
        physical_hash = optimizer._physical_geometry_sha256(functional)
        geometry_hash = hashlib.sha256((physical_hash + metric_hash).encode())
        operators = tuple((keys[0][1], functional.one_body[keys[0][1]]) for keys, _, _, _ in layout)
        metrics = tuple((keys[0][1], functional.metric[keys[0][1]]) for keys, _, _, _ in layout)
        optimizer._hash_value(geometry_hash, (operators, metrics, "nuclear", .05))
        check("actual_same_physical_geometry_SHA", geometry_hash.hexdigest() != contract["metric"]["geometry_sha256"])
        state_hash = hashlib.sha256(geometry_hash.hexdigest().encode())
        optimizer._hash_value(state_hash, (x, contract["metric"]["objective_sha256"]))
        check("actual_same_literal208_state_SHA", state_hash.hexdigest() != contract["metric"]["state_sha256"])

        def unpack(coordinates):
            waves = {}
            for keys, projector, shape, where in layout:
                c = projector @ coordinates[where].reshape(shape)
                for j, key in enumerate(keys):
                    waves[key] = c[:123, j].copy(), c[123:, j].copy()
            return waves

        def coordinate(gradient):
            v = np.zeros(3451, dtype=np.asarray(next(iter(gradient.values()))).dtype)
            for keys, p, _, where in layout:
                v[where] = (p.astype(v.dtype).T @ np.column_stack([gradient[k] for k in keys])).ravel()
            return v

        def tangent(vector):
            result = vector.copy()
            for _, _, shape, where in layout:
                y, a = x[where].reshape(shape).astype(result.dtype), result[where].reshape(shape)
                z = y.T @ a
                result[where] = (a - .5 * y @ (z + z.T)).ravel()
            return result

        def original_pg(g):
            v = g.copy()
            for i, (_, _, shape, where) in enumerate(layout):
                a = g[where].reshape(shape)
                v[where] = np.column_stack([m.astype(v.dtype) @ a[:, j]
                        for j, m in enumerate(saved[f"inverse_blocks_{i}"])]).ravel()
            return tangent(v)

        waves = unpack(x)
        check("saved_all29_keys", not np.array_equal(saved["keys"], functional.keys))
        for j, key in enumerate(functional.keys):
            check("literal_Px_" + str(key), not (np.array_equal(waves[key][0], saved["P"][j])
                  and np.array_equal(waves[key][1], saved["Q"][j])))
        capture = {}
        install_resolution_capture(CIBasisTracking, tracker, capture)
        guard = import_helper("stored_H_guard208", "exact_guard_helper")
        class PointBudget(guard.ExactGuardBudget):
            def reserve(self, roots, pairs):
                if roots < 0 or pairs < 0 or self.points + 1 > 5 or self.quotients + roots > 25 or self.examined_pairs + pairs > 1500000:
                    raise RuntimeError("Stored-H Fraction certificate cap")
                self.points += 1
                self.quotients += roots
                self.examined_pairs += pairs

        point_budget = PointBudget()

        def full_point(name, point_x, point_waves, ci, energy):
            h = ci.hamiltonian_for(csfs, point_waves, functional.keys, data, 70, 123,
                                  nuclear_potential=nuclear, max_determinants=2000)
            np.savez_compressed(output / (name + ".npz"), x=point_x, H=h, energies=ci.state_energies,
                                mixing=ci.mixing_coefficients, keys=np.array(functional.keys),
                                P=np.array([point_waves[k][0] for k in functional.keys]),
                                Q=np.array([point_waves[k][1] for k in functional.keys]))
            write(output / (name + "_owner.json"), ci.hamiltonian_snapshot.identity_payload())
            check(name + "_whole_H_symmetry", not np.array_equal(h, h.T))
            check(name + "_full_CI_residual", np.max(np.linalg.norm(h @ ci.mixing_coefficients
                  - ci.mixing_coefficients * ci.state_energies, axis=0)), 1e-8)
            check(name + "_full_CI_Gram", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients - np.eye(1046))), 1e-10)
            _, blocks = ci_level_blocks(csfs, ci, require_full=True)
            matched = target_bindings(targets, capture["bindings"])
            weights = capture["weights"].copy()
            check(name + "_original_equal_five", not (np.count_nonzero(weights) == 5
                  and np.array_equal(weights[weights > 0], np.repeat(.2, 5))))
            rows, columns, identities = [], [], []
            for t in targets:
                uid = (t["twoJ"], t["parity"], t["models"]["DC"]["local_root"])
                j, p, rank = matched[uid]["selected_levels"][0]
                if name == "baseline":
                    local_target_levels[uid] = (j, p, rank)
                else:
                    check(name + "_same_local_isolated_branch_" + t["label"],
                          (j, p, rank) != local_target_levels[uid])
                column = int(blocks[j, p][rank])
                check(name + "_weight_" + t["label"], weights[column] != .2)
                neighbors = [other for other in blocks[j, p] if other != column]
                gap = min(abs(ci.state_energies[column] - ci.state_energies[other]) for other in neighbors)
                check(name + "_rank_one_isolated_target_scope_" + t["label"], not (gap > 1e-7))
                columns.append(column)
                identities.append(json.dumps([t["label"], *uid], separators=(",", ":")))
                rows.append(dict(label=t["label"], UID=uid, selected=[j, p, rank], column=column,
                                 energy_Ha=float(ci.state_energies[column]), tracking=matched[uid]))
            check(name + "_own_reported_EO", float(weights @ ci.state_energies) != float(energy))
            replay_values, replay_vectors = [], []
            for sector in sorted(blocks, key=lambda value: (value[0], -value[1])):
                indices = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector]
                other = sorted(set(range(1046)) - set(indices))
                check(name + "_exact_H_support_" + str(sector), np.max(abs(h[np.ix_(indices, other)]), initial=0.))
                check(name + "_exact_C_support_" + str(sector), np.max(abs(ci.mixing_coefficients[np.ix_(other, blocks[sector])]), initial=0.))
                # Same f043 old LAPACK prescription; no new centered source.
                e, c = linalg.eigh(h[np.ix_(indices, indices)], subset_by_index=None, driver="evd")
                full = np.zeros((1046, len(e)))
                full[indices] = c
                replay_values.extend(e)
                replay_vectors.append(full)
            order = np.argsort(replay_values, kind="stable")
            replay_e = np.asarray(replay_values)[order]
            replay_c = np.column_stack(replay_vectors)[:, order]
            check(name + "_same_H_replay_allroots", np.max(abs(replay_e - ci.state_energies)), 1e-8)
            check(name + "_replay_full_residual", np.max(np.linalg.norm(h @ replay_c - replay_c * replay_e, axis=0)), 1e-8)
            check(name + "_replay_full_Gram", np.max(abs(replay_c.T @ replay_c - np.eye(1046))), 1e-10)
            for sector, cols in blocks.items():
                indices = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector]
                begin = 0
                for end in range(1, len(cols) + 1):
                    if (end == len(cols) or min(abs(ci.state_energies[cols[end]] - ci.state_energies[cols[end-1]]),
                            abs(replay_e[cols[end]] - replay_e[cols[end-1]])) > 1e-7):
                        cluster = cols[begin:end]
                        a, b = ci.mixing_coefficients[np.ix_(indices, cluster)], replay_c[np.ix_(indices, cluster)]
                        check(name + "_all_signed_cluster_" + str(list(map(int, cluster))), np.max(abs(a @ a.T - b @ b.T)), 1e-8)
                        begin = end
            for keys, p, _, _ in layout:
                s = functional.metric[keys[0][1]]
                c = np.column_stack([np.concatenate(point_waves[k]) for k in keys])
                check(name + "_radial_Gram_" + str(keys[0][1]), np.max(abs(c.T @ s @ c - np.eye(len(keys)))), 1e-10)
                outside = c - p @ (p.T @ s @ c)
                check(name + "_sameP_leakage_" + str(keys[0][1]), np.max(np.sum(outside * (s @ outside), axis=0)), 1e-12)
            certificate = guard.exact_reported_point(h, ci.mixing_coefficients[:, columns],
                ci.state_energies[columns], weights[columns], float(energy), target_ids=identities, budget=point_budget)
            certificates[name] = certificate
            write(output / (name + "_targets.json"), {"targets": rows, "exact_stored_H_point": certificate.ledger(),
                  "owner_input_sha256": ci.hamiltonian_snapshot.input_sha256,
                  "owner_H_sha256": ci.hamiltonian_snapshot.matrix_sha256,
                  "array_sha256": sha(output / (name + ".npz"))})
            return h, weights

        phase = "baseline"
        energy, production, ci = functional.evaluate(waves, gradient=True)
        h, weights = full_point("baseline", x, waves, ci, energy)
        check("same_literal208_H_bytes", h.tobytes() != saved["H"].tobytes())
        check("same_literal208_energies_bytes", ci.state_energies.tobytes() != saved["energies"].tobytes())
        check("same_owner_input", ci.hamiltonian_snapshot.input_sha256 != contract["point"]["owner_input_sha256"])
        check("same_owner_H", ci.hamiltonian_snapshot.matrix_sha256 != contract["point"]["owner_H_sha256"])
        check("baseline_saved_full_radial_g", np.max(abs(np.array([production[k] for k in functional.keys]) - saved["radial_gradient"])), 1e-9)
        raw, g = coordinate(production), tangent(coordinate(production))
        pg = original_pg(g)
        check("baseline_saved_raw_g", np.max(abs(raw - saved["raw_coordinate_gradient"])), 1e-9)
        check("baseline_saved_tangent_g", np.max(abs(g - saved["tangent_gradient"])), 1e-9)
        check("baseline_saved_original_pg", np.max(abs(pg - saved["preconditioned_gradient"])), 1e-10)
        check("baseline_saved_residual", abs(np.linalg.norm(pg) - contract["point"]["orbital_residual"]), 1e-12)
        check("baseline_saved_indicator", abs(.5 * abs(g @ pg) - contract["point"]["estimated_energy_error_Ha"]), 1e-14)
        # No comparison of individual columns inside an unresolved old span.
        _, old_blocks = ci_level_blocks(csfs, CIResult(saved["energies"], saved["mixing"]), require_full=True)
        for sector, cols in old_blocks.items():
            indices = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector]
            begin = 0
            for end in range(1, len(cols) + 1):
                if end == len(cols) or abs(saved["energies"][cols[end]] - saved["energies"][cols[end-1]]) > 1e-7:
                    subset = cols[begin:end]
                    a, b = saved["mixing"][np.ix_(indices, subset)], ci.mixing_coefficients[np.ix_(indices, subset)]
                    check("baseline_saved_signed_cluster_" + str(list(map(int, subset))), np.max(abs(a @ a.T - b @ b.T)), 1e-8)
                    begin = end
        complete, complete_tables = {}, []
        for column in np.flatnonzero(weights):
            c = ci.mixing_coefficients[:, column]
            norm = sum((Fraction.from_float(float(v))**2 for v in c), Fraction(0))
            if norm <= 0:
                raise ValueError("Nonpositive exact CI norm")
            table = functional.radial_density.contract_bilinear(c, c)
            complete_tables.append({"column": int(column), "weight": float(weights[column]),
                "exact_CI_norm": {"numerator": str(norm.numerator), "denominator": str(norm.denominator)},
                "unfiltered_signed_RDM": [[k, float(v)] for k, v in table.items()]})
            for term, value in table.items():
                complete[term] = complete.get(term, Fraction(0)) + Fraction.from_float(float(weights[column])) * Fraction.from_float(float(value)) / norm
        full_weights = {k: float(v) for k, v in complete.items() if v != 0}
        if any(v == 0 or not math.isfinite(v) for v in full_weights.values()):
            raise ValueError("Signed full coefficient cannot be represented")
        full_g = functional.gradient_from_weights(waves, full_weights, record_diagnostics=False)
        full_tangent = tangent(coordinate(full_g))
        check("fullnorm_signed_tangent_whole", np.max(abs(full_tangent - g)), 1e-9)
        check("fullnorm_signed_original_pg_whole", np.max(abs(original_pg(full_tangent) - pg)), 1e-10)
        centering_module = import_helper("constraint_connection208", "constraint_bookkeeping")
        connection = centering_module.ConstraintBookkeeping(functional.keys, waves, full_g, full_weights,
            raw_operators, functional.one_body, layout, max_products=8000000)
        corrected = {k: np.array([float(v) for v in connection.g[k]], dtype=np.longdouble) for k in functional.keys}
        corrected_tangent = tangent(coordinate(corrected))
        check("raw_quotient_connection_full_tangent", np.max(abs(corrected_tangent - g)), 1e-9)
        check("raw_quotient_connection_original_pg", np.max(abs(original_pg(corrected_tangent) - pg)), 1e-10)
        direction = -saved["preconditioned_gradient"] / np.linalg.norm(saved["preconditioned_gradient"])
        check("original_whole_direction_tangent", np.max(abs(tangent(direction) - direction)), 1e-12)
        check("original_whole_direction_unit", abs(np.linalg.norm(direction) - 1), 1e-12)
        check("original_TMT_signed_slope_identity", abs(g @ direction + (g @ pg) / np.linalg.norm(pg)), 1e-12)
        ideal = connection.ideal(direction)
        analytic = float(ideal["centered"])
        derivative_gate = max(1e-8, 1e-5 * abs(analytic))
        check("full_connection_vs_production_analytic", abs(analytic - g @ direction), derivative_gate)
        check("original_metric_direction_descent", not (analytic < 0 and g @ direction < 0))
        np.savez_compressed(output / "baseline_gradient_direction.npz", x=x, direction=direction,
            raw_g=raw, g=g, pg=pg, radial_gradient=np.array([production[k] for k in functional.keys]),
            full_signed_radial_gradient=np.array([full_g[k] for k in functional.keys]),
            full_quotient_tangent=corrected_tangent,
            inverse_diagonal=saved["inverse_diagonal"],
            **{f"projector_{i}": row[1] for i, row in enumerate(layout)},
            **{f"inverse_blocks_{i}": saved[f"inverse_blocks_{i}"] for i in range(7)})
        write(output / "baseline_signed_workspace.json", {"full_normalized_signed_weights": [[k, v] for k, v in full_weights.items()],
              "individual_unfiltered_normalized_tables": complete_tables,
              "analytic_centered": analytic, "analytic_ambient": float(ideal["ambient"]),
              "analytic_signed_normal": float(ideal["normal"]), "gate": derivative_gate,
              "source_map_sha256": contract["source"]["map_sha256"], "metric": contract["metric"],
              "stationarity_passed": False, "no_explicit_dC_firstorder_missingphysics_claim": True})
        flush()
        for index, step in enumerate(contract["direction"]["fixed_h"]):
            finite, normals = [], []
            for sign in (1., -1.):
                name = ("h" if index == 0 else "half_h") + ("_plus" if sign > 0 else "_minus")
                phase = name
                candidate = x + sign * step * direction
                for _, _, shape, where in layout:
                    candidate[where] = optimizer._polar_retraction(candidate[where].reshape(shape)).ravel()
                point_waves, delta = unpack(candidate), unpack(candidate - x)
                point_energy, point_ci = functional.evaluate(point_waves, gradient=False)
                full_point(name, candidate, point_waves, point_ci, point_energy)
                diagnostic = functional.mixed_ci_energy_difference(waves, point_waves, ci, point_ci, increments=delta)
                normal = connection.trial(point_waves, delta)
                finite.append(diagnostic)
                normals.append(normal)
                exact_H_delta = certificates[name].weighted_quotient - certificates["baseline"].weighted_quotient
                write(output / (name + "_finite.json"), {"mixed": diagnostic, "normal_Ha": float(normal["normal"]),
                      "normal_explicit_Ha": float(normal["normal_explicit"]), "signed_full_Gram": normal["blocks"],
                      "exact_constraint_products": connection.products,
                      "exact_stored_H_quotient_delta": {"numerator": str(exact_H_delta.numerator), "denominator": str(exact_H_delta.denominator)},
                      "mixed_Rayleigh_minus_exact_stored_H_delta_Ha": float(Fraction.from_float(float(diagnostic["rayleigh_change"])) - exact_H_delta),
                      "connection_comparison_is_not_uniform_radial_error_certificate": True})
                flush()
            values = {}
            for field in ("explicit_radial", "reconstruction_radial", "reported_response", "rayleigh_response", "reported_change",
                          "rayleigh_change", "direct_reported_root_difference", "direct_retained_quotient_difference"):
                values[field + "_central"] = float((Fraction.from_float(float(finite[0][field]))
                      - Fraction.from_float(float(finite[1][field]))) / Fraction.from_float(2 * step))
            plus = Fraction.from_float(float(finite[0]["rayleigh_change"])) - normals[0]["normal"]
            minus = Fraction.from_float(float(finite[1]["rayleigh_change"])) - normals[1]["normal"]
            centered = float((plus - minus) / Fraction.from_float(2 * step))
            row = {"h": step, "analytic": analytic, "centered_Rayleigh_central": centered,
                   "error": centered - analytic, "original_derivative_gate": derivative_gate,
                   "even_centered_curvature_observation": float((plus + minus) / Fraction.from_float(step)**2),
                   "fixed_connection_finite_remainder_not_uniformly_bounded": True, **values}
            records.append(row)
            flush()
            check(("h" if index == 0 else "half_h") + "_same208_centered_derivative", abs(centered - analytic), derivative_gate)
        check("exact_five_H", completed["H"] != 5)
        check("exact_five_CI", completed["CI"] != 5)
        check("exact_two_gradients", completed["gradient"] != 2)
        check("exact50_CI_replay_eigh", completed["scipy_eigh"] != 50)
        # h/S equality is deliberately not inferred from CI request count.
        check("new_hS_producer_total_cap", sum(completed[k] for k in ("initial_hS", "CI_internal_hS", "mixed_hS")) > 70)
        for role_row in roles.values():
            check("frozen_input_at_exit_" + role_row["path"], sha(REPO / role_row["path"]) != role_row["sha256"])
        identity = load(role("candidate_source_identity"))
        for path, digest in identity["all_atomx_python_sha256"].items():
            check("frozen_source_at_exit_" + path, sha(source / path.removeprefix("src/")) != digest)
        phase = "bounded_directional_control_completed"
    except BaseException as failure:
        error = f"{type(failure).__name__}: {failure}"
        write(output / "failure.json", {"error": error, "phase": phase, "traceback": traceback.format_exc(),
              "attempted": attempted, "completed": completed, "controls": checks,
              "later_phases_not_executed": True})
    finally:
        flush()
        write(output / "result.json", {"status": phase, "error": error,
              "all_declared_directional_controls_passed": error is None,
              "controls": checks, "attempted": attempted, "completed": completed, "directions": records,
              "source_map_sha256": contract["source"]["map_sha256"], "driver_sha256": sha(__file__),
              "contract_sha256": sha(HERE / "contract.json"), "execution_sha256": sha(HERE / "execution_contract.json"),
              "metric": contract["metric"], "stationarity_passed": False,
              "production_reported_objective_changed": False, "optimizer_called": False,
              "GPU_called": False, "physical_completeness_or_experiment_certified": False,
              "isolated_direction_only": True, "host_admission": {"available_CPUs": available, "CPU": caps["CPU"], "load": host_load},
              "elapsed_uncontrolled_s": time.monotonic() - start, "peak_RSS_KiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    return 0 if error is None else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    contract = load(HERE / "contract.json")
    if args.prepare == args.execute:
        parser.error("Choose exactly one mode")
    if args.prepare:
        metadata(contract, archives=False)
        print("source/JSON admission only; no NPZ bytes, arrays or scientific calls")
        return 0
    return execute(contract, load(HERE / "execution_contract.json"))


if __name__ == "__main__":
    raise SystemExit(main())
