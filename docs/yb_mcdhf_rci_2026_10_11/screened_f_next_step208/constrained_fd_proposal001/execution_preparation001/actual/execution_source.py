"""One native31 combined4f/5f constrained FD; no numerical default-mode imports."""
import argparse
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time
import traceback
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    # Wide NumPy scalar tolist()/item() can return itself. JSON floats are
    # display-only; exact arithmetic below uses original as_integer_ratio().
    if (hasattr(value, "dtype") and getattr(value, "ndim", None) == 0
            and hasattr(value, "as_integer_ratio")):
        return float(value)
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if hasattr(value, "item"):
        return plain(value.item())
    return value


def write(path, value):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def metadata(*, arrays=False):
    """Source/JSON admission only unless a separate root seal permits NPZ reads."""
    contract = load(HERE / "contract.json")
    roles = {name: ROOT / row["path"] for name, row in contract["input_roles"].items()}
    deferred = []
    for name, row in contract["input_roles"].items():
        if roles[name].suffix == ".npz" and not arrays:
            deferred.append(dict(role=name, **row))
        elif sha(roles[name]) != row["sha256"]:
            raise ValueError("Bound input drift: " + name)
    identity = load(roles["source_identity"])
    if (identity["source_map_sha256"] != contract["fixed"]["source_map_sha256"]
            or len(identity["all_atomx_python_sha256"]) != 356):
        raise ValueError("Wrong frozen f043 source identity")
    source = ROOT / contract["source_archive"]
    for name, expected in identity["all_atomx_python_sha256"].items():
        if sha(source / name.removeprefix("src/")) != expected:
            raise ValueError("Frozen source drift: " + name)
    result, receipt, completion = (load(roles[n]) for n in
        ("gradient_qualification_result", "gradient_qualification_root_receipt",
         "gradient_qualification_completion"))
    if (result["all_declared_controls_passed"] is not True or len(result["checks"]) != 438
            or any(not row["passed"] for row in result["checks"])
            or receipt["status"] != "foreground_closed" or receipt["exit_code"] != 0
            or completion["all_completion_metadata_checks_passed"] is not True
            or completion["actual_live_controls_count"] != 438
            or completion["independent_numerical_array_replay_performed"] is not False
            or result["expanded_MCDHF_optimized"] is not False
            or result["finite_CI_response_qualified"] is not False
            or result["source_map_sha256"] != contract["fixed"]["source_map_sha256"]):
        raise ValueError("Closed31 prerequisite identity or scope changed")
    for field in ("new_metric_sha256", "new_geometry_sha256", "new_state_sha256", "objective_identity"):
        if result[field] != contract["fixed"][field]:
            raise ValueError("Wrong original recovered31 point: " + field)
    delivery = load(roles["hour27_delivery"])
    if (delivery["passed"] is not True or delivery["actual_send"]["passed"] is not True
            or delivery["cycle"] != "2026-10-11T19" or delivery["position"] != 1621
            or delivery["service_iteration"]["automatic_jobs_in_top_level"] != 0):
        raise ValueError("Separate hour27 root iteration trigger changed")
    return contract, roles, source, deferred


def authorize(binding_path):
    # Refuse before NPZ hashing or physical imports unless exact root approval
    # matches the frozen consumer and every input identity.
    contract, roles, source, _ = metadata(arrays=False)
    authority = load(binding_path)
    expected = dict(execution_authorized_by_root=True,
                    preparation_contract_sha256=sha(HERE / "contract.json"),
                    driver_sha256=sha(__file__),
                    preparation_sha256=sha(HERE / "preparation.json"),
                    source_map_sha256=contract["fixed"]["source_map_sha256"],
                    scientific_inputs_sha256=contract["input_roles"],
                    resources=contract["resources"])
    if any(authority.get(k) != v for k, v in expected.items()):
        raise ValueError("Exact separate root authorization is required; no arrays/science")
    prep = load(HERE / "preparation.json")
    if (prep["all_source_JSON_preparation_checks_passed"] is not True
            or prep["driver_sha256"] != sha(__file__)
            or prep["contract_sha256"] != sha(HERE / "contract.json")):
        raise ValueError("Preparation identity is stale")
    metadata(arrays=True)
    return authority, contract, roles, source


def execute(binding_path):
    authority, contract, roles, source = authorize(binding_path)
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    (output / "execution_source.py").write_bytes(Path(__file__).read_bytes())
    caps, gates, fixed = contract["resources"], contract["gates"], contract["fixed"]
    limits = caps["calls"]
    attempted, completed = dict.fromkeys(limits, 0), dict.fromkeys(limits, 0)
    checks, stages, points, restored = [], [], {}, []
    raw_operators, coverage = {}, {}
    diagnostics, derivative_domains = {}, {}
    exact_products = [0]
    current_CI = [None]
    hS_requests = dict(attempted=0, completed=0, hits=0, misses=0,
                       by_CI={}, carry_events=[])
    started = time.monotonic()

    def progress(phase):
        write(output / "progress.json", dict(phase=phase, attempted=attempted,
              completed=completed, checks_count=len(checks),
              last_check=checks[-1] if checks else None, archives_count=len(stages),
              last_archive=stages[-1] if stages else None, completed_point_names=list(points),
              hS_requests=hS_requests, exact_products=exact_products[0], elapsed_uncontrolled_s=time.monotonic() - started))

    def timeout(*_):
        raise TimeoutError("Declared whole-run cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(caps["inner_seconds"])
    try:
        if sys.executable != caps["interpreter_literal"]:
            raise ValueError("Use the declared literal .venv interpreter path")
        thread_names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
        if any(os.environ.get(n) != "1" for n in thread_names):
            raise ValueError("All declared thread variables must be1 before import")
        available = sorted(os.sched_getaffinity(0))
        mem = next(int(r.split()[1]) for r in Path("/proc/meminfo").read_text().splitlines()
                   if r.startswith("MemAvailable:")) / 1024**2
        admission = dict(original_affinity=available, CPU=caps["CPU"],
                         load_average=os.getloadavg(), host_available_GiB=mem,
                         timing_claim_valid=False)
        write(output / "resource_admission.json", admission)
        if (caps["CPU"] not in available or admission["load_average"][0] > caps["maximum_system_load1"]
                or mem < caps["minimum_host_available_GiB"]):
            raise RuntimeError("Host admission rejected before numerical import")
        os.sched_setaffinity(0, {caps["CPU"]})
        maximum = int(caps["AS_GiB"] * 2**30)
        resource.setrlimit(resource.RLIMIT_AS, (maximum, maximum))
        if any(n == "atomx" or n.startswith("atomx.") for n in sys.modules):
            raise ValueError("Fresh immutable-source process required")
        cache_dir = output / "numba_cache"
        cache_dir.mkdir()
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
        sys.path.insert(0, str(source))
        import numpy as np
        from scipy import linalg

        import atomx.methods.csf.variational as variational
        import atomx.numerics.bspline.dirac_assembly as dirac
        import atomx.numerics.bspline.orbital_optimization as optimization
        import atomx.physics.csf.matrix as matrix
        import atomx.physics.csf.radial_increment as radial_increment
        import atomx.physics.csf.tracking as tracking_module
        from atomx.angular.csf_hamiltonian import csf_signature
        from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
        from atomx.constants import HARTREE_CM1
        from atomx.methods import ci as ci_module
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.physics.csf.radial_density import CIRadialDensity
        from atomx.physics.csf.state_average import CIStateAverage
        from atomx.potentials.nuclear import make_nuclear_potential

        def check(name, error, limit=0.):
            row = dict(name=name, error=float(error), limit=limit,
                       passed=bool(np.isfinite(error) and error <= limit))
            checks.append(row)
            progress(name)
            if not row["passed"]:
                raise ValueError("Unchanged scientific gate failed: " + name)

        def save(name, **arrays):
            path = output / name
            if path.exists():
                raise FileExistsError("Keep completed arrays immutable: " + name)
            temporary = path.with_name(path.name + ".part")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            stages.append(dict(path=str(path.relative_to(HERE)), sha256=sha(path)))
            progress("saved_" + name)

        def counted(name, function, capture=False):
            def call(*args, **kwargs):
                attempted[name] += 1
                if name in ("initial_hS", "CI_internal_hS", "mixed_radial_hS"):
                    attempted["hS_total"] += 1
                    if attempted["hS_total"] > limits["hS_total"]:
                        raise RuntimeError("Whole h/S production cap before builder")
                if attempted[name] > limits[name]:
                    progress("overbudget_" + name)
                    raise RuntimeError("Prospective call cap: " + name)
                quiet = name in ("finite_component_cache_miss", "finite_potential_cache_miss",
                                 "finite_density_cache_miss")
                if not quiet:
                    progress("before_" + name)
                value = function(*args, **kwargs)
                completed[name] += 1
                if name in ("initial_hS", "CI_internal_hS", "mixed_radial_hS"):
                    completed["hS_total"] += 1
                if capture:
                    kappa = int(args[1])
                    h, s = value
                    if kappa in raw_operators:
                        a, b = raw_operators[kappa]
                        if not (np.array_equal(a, h) and np.array_equal(b, s)):
                            save(f"different_raw_hS_{name}_{completed[name]}.npz", raw_h=h, S=s)
                            raise ValueError("Samepoint native raw h/S changed")
                    else:
                        raw_operators[kappa] = h.copy(), s.copy()
                        save(f"raw_hS_{kappa}.npz", kappa=np.array(kappa), raw_h=h, S=s)
                if not quiet or completed[name] % 128 == 0:
                    progress("after_" + name)
                return value
            return call

        def patch(module, name, counter, capture=False):
            original = getattr(module, name)
            restored.append((module, name, original))
            setattr(module, name, counted(counter, original, capture))

        patch(variational, "build_dirac_matrices", "initial_hS", True)
        patch(matrix, "build_dirac_matrices", "CI_internal_hS", True)
        patch(radial_increment, "build_dirac_matrices", "mixed_radial_hS", True)
        patch(matrix, "build_ci_matrix", "H")
        patch(ci_module, "ci_solve", "CI")
        restored.append((variational, "ci_solve", variational.ci_solve))
        variational.ci_solve = ci_module.ci_solve
        patch(dirac, "solve_dirac", "validation_Dirac")
        patch(optimization, "optimize_orbitals", "optimizer")
        patch(optimization, "_polar_retraction", "polar_retraction")
        patch(variational.CSFOrbitalFunctional, "evaluate", "public_reported_energy_evaluate")
        patch(variational.CSFOrbitalFunctional, "gradient_from_weights", "gradient_from_weights")
        patch(variational.CSFOrbitalFunctional, "_pair_potential", "gradient_pair_potential")
        patch(variational.CSFOrbitalFunctional, "mixed_ci_energy_difference", "mixed_ci_finite_increment")
        patch(CIRadialDensity, "contract", "retained_RDM")
        patch(CIRadialDensity, "contract_bilinear", "unfiltered_target_RDM")
        patch(tracking_module.CIBasisTracking, "resolve", "tracker_resolve")
        patch(tracking_module, "ci_subspace_overlaps", "tracker_overlap")
        patch(np.linalg, "svd", "tracking_SVD")
        patch(np.linalg, "eigvalsh", "geometry_SPD_eigvalsh")
        patch(np.linalg, "eigh", "polar_numpy_eigh_fallback")
        patch(linalg, "eigh", "complete_SciPy_eigh")
        for method, counter in (("components", "finite_component_cache_miss"),
                                ("_potential", "finite_potential_cache_miss"),
                                ("_density", "finite_density_cache_miss")):
            patch(radial_increment._DCRadialIncrement, method, counter)

        original_blocks = matrix._dirac_blocks
        original_reset = matrix._reset_memo_for_ci_build
        restored.extend(((matrix, "_dirac_blocks", original_blocks),
                         (matrix, "_reset_memo_for_ci_build", original_reset)))

        def blocks(ns, kappa, Z, workspace):
            hS_requests["attempted"] += 1
            if hS_requests["attempted"] > caps["maximum_hS_getter_requests"]:
                raise RuntimeError("h/S getter observation cap before forwarding")
            hit = (("h0", ns, kappa, Z) in matrix._SLATER_MEMO
                   and matrix._MEMO_CONTEXT is workspace)
            value = original_blocks(ns, kappa, Z, workspace)
            hS_requests["completed"] += 1
            hS_requests["hits" if hit else "misses"] += 1
            if current_CI[0] is not None:
                coverage[current_CI[0]].add(int(kappa))
            return value

        def reset(workspace, nucleus):
            previous = {k: v for k, v in matrix._SLATER_MEMO.items()
                        if isinstance(k, tuple) and k and k[0] == "h0"}
            same_workspace = matrix._MEMO_CONTEXT is workspace
            same_nucleus = matrix._SLATER_MEMO.get("_nuclear_potential") is nucleus
            value = original_reset(workspace, nucleus)
            carried = {k: v for k, v in matrix._SLATER_MEMO.items()
                       if isinstance(k, tuple) and k and k[0] == "h0"}
            hS_requests["carry_events"].append(dict(CI=current_CI[0],
                same_workspace=same_workspace, same_nucleus=same_nucleus,
                previous=len(previous), carried=len(carried),
                identical_tuples=all(k in previous and v is previous[k] for k, v in carried.items())))
            return value

        matrix._dirac_blocks = blocks
        matrix._reset_memo_for_ci_build = reset

        def archive(role):
            with np.load(roles[role], allow_pickle=False) as a:
                return {k: a[k].copy() for k in a.files}

        def signature(c):
            s = csf_signature(c)
            return json.dumps(plain((s.subshells, s.coupling, c.total_J2, c.parity)),
                              separators=(",", ":"))

        saved_g, saved_point, saved_CI, radial, original_hS = (archive(r) for r in
            ("qualified_full31_gradient", "recovered_coordinates_and_waves", "recovered_full_CI",
             "geometry_workspace", "screened_geometry"))
        data = {k[6:]: v.item() if v.ndim == 0 else v.copy()
                for k, v in radial.items() if k.startswith("basis_")}
        models = load(roles["models"])
        sectors = {tuple(s) for s in models["selected_sectors"]}
        full = [c for row in models["models"]["complete5f"]["configs"]
                for c in enumerate_jj_csfs(row["config"])]
        csfs = [c for c in full if (c.total_J2, c.parity) in sectors]
        basis = load(roles["canonical_basis"])["complete5f"]["retained_signatures"]
        check("complete_parent3789", len(full) != 3789)
        check("complete_target1595", len(csfs) != 1595)
        check("five_complete_target_blocks", Counter((c.total_J2, c.parity) for c in csfs)
              != {(r["twoJ"], r["parity"]): r["CSFs"] for r in fixed["complete_target_blocks"]})
        check("full_signed_CSF_order", [signature(c) for c in csfs] != basis)
        check("same_charge69", {c.n_electrons for c in full} != {69})
        write(output / "basis_identity.json", dict(signatures=basis, parent3789_enumerated=True,
              complete_target1595=True, global3789_spectrum_certified=False))
        keys31 = tuple(tuple(map(int, k)) for k in saved_point["keys"])
        check("qualified31_keys", keys31 != tuple(sorted(get_shared_orbital_basis(csfs))))
        check("fullgradient_saved31_keys", not np.array_equal(saved_g["keys"], saved_point["keys"]))
        reference_csfs = [c for row in load(roles["reference_models"])["models"]["baseline"]["configs"]
                          for c in enumerate_jj_csfs(row["config"])]
        check("original531_tracker_reference", len(reference_csfs) != 531)
        reference_ci = archive("tracking_ci")
        reference_waves = {tuple(map(int, k)): (p.copy(), q.copy()) for k, p, q in
                           zip(radial["orbital_keys"], radial["P"], radial["Q"], strict=True)}
        targets = load(roles["tracking_binding"])["targets"]
        expected_UIDs = {tuple(uid): label for label, uid in fixed["original_UIDs"]}
        check("original_five_UIDs", {(t["twoJ"], t["parity"], t["models"]["DC"]["local_root"])
              for t in targets} != set(expected_UIDs))
        average = CIStateAverage(tuple((t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.)
                                      for t in targets))
        tracker = tracking_module.CIBasisTracking(average, reference_csfs,
            ci_module.CIResult(reference_ci["energies_Ha"], reference_ci["mixing"]), reference_waves,
            {k: original_hS[f"s_{k}"] for k in fixed["kappas"]},
            max_determinant_pairs=caps["maximum_signed_overlap_pairs"])
        nucleus = make_nuclear_potential(70, "fermi", A=175)
        functional = variational.CSFOrbitalFunctional(csfs, data, 70, nucleus,
            state_average=average, level_tracking=tracker, retain_hamiltonian=True, max_determinants=2000)
        check("functional_exact31_order", functional.keys != keys31)
        for kappa in fixed["kappas"]:
            h, s = raw_operators[kappa]
            check("actual_L_raw_h_" + str(kappa),
                  not np.array_equal(functional.one_body[kappa], np.tril(h) + np.tril(h, -1).T))
            check("saved_logical_h_" + str(kappa),
                  not np.array_equal(functional.one_body[kappa], original_hS[f"h_{kappa}"]))
            check("saved_S_" + str(kappa), not np.array_equal(s, original_hS[f"s_{kappa}"]))
        term_capacity = len({key for structure, _, _ in functional.radial_density.blocks
                             for key in structure.keys})
        check("complete_angular_capacity_not_pruned", term_capacity,
              caps["signed_term_work_cap"]["complete_angular_term_capacity_per_trial"])
        write(output / "complete_angular_capacity.json", dict(terms=term_capacity,
              exact_zero_only=True, no_small_term_filter_added=True, entire_model_rejected_if_overbudget=True))

        layout = tuple((tuple(map(tuple, row["keys"])), saved_point[f"projector_{i}"],
                        tuple(row["shape"]), slice(*row["slice"]))
                       for i, row in enumerate(load(roles["new_layout"])))
        inverses = tuple(tuple(b for b in saved_point[f"inverse_blocks_{i}"])
                         for i in range(len(layout)))
        for name in ("x", "inverse_diagonal", *[f"projector_{i}" for i in range(7)],
                     *[f"inverse_blocks_{i}" for i in range(7)]):
            check("same_qualified_array_bytes_" + name, saved_point[name].tobytes() != saved_g[name].tobytes())
        state = optimization.OrbitalOptimizationState.from_arrays(functional, layout=layout,
            coordinates=saved_point["x"], inverse_diagonal=saved_point["inverse_diagonal"],
            inverse_blocks=inverses, projector_source="nuclear",
            objective_identity=fixed["objective_identity"], occupation_floor=.05)
        state.validate(functional)
        check("same3689_dimension", state.coordinates.shape != (3689,))
        for field, actual in (("new_metric_sha256", state.residual_metric_sha256),
                              ("new_geometry_sha256", state.geometry_sha256),
                              ("new_state_sha256", state.state_sha256)):
            check("unchanged_" + field, actual != fixed[field])
        check("authoritative_x_bytes", state.coordinates.tobytes() != saved_point["x"].tobytes())
        write(output / "state_identity.json", dict(metric_sha256=state.residual_metric_sha256,
              geometry_sha256=state.geometry_sha256, state_sha256=state.state_sha256,
              objective_identity=state.objective_identity, layout=load(roles["new_layout"]),
              authoritative_point="direct original nuclear P times own recovered3689 x",
              initial_polar_repair=False, metric_refit=False))

        def unpack(coordinates):
            result = {}
            for keys, projector, shape, where in state.layout:
                c = projector @ coordinates[where].reshape(shape)
                for j, k in enumerate(keys):
                    result[k] = c[:123, j].copy(), c[123:, j].copy()
            return result

        def point_constraints(name, coordinates, waves):
            arrays = {}
            for keys, projector, shape, where in state.layout:
                kappa = keys[0][1]
                y = coordinates[where].reshape(shape)
                c = np.column_stack([np.concatenate(waves[k]) for k in keys])
                s = functional.metric[kappa]
                arrays[f"coordinate_Gram_{kappa}"] = y.T @ y
                arrays[f"physical_Gram_{kappa}"] = c.T @ s @ c
                check(name + "/coordinate_Gram_" + str(kappa), np.max(abs(y.T @ y - np.eye(len(keys)))),
                      gates["full_same_kappa_radial_Gram"])
                check(name + "/physical_Gram_" + str(kappa), np.max(abs(c.T @ s @ c - np.eye(len(keys)))),
                      gates["full_same_kappa_radial_Gram"])
                for j, key in enumerate(keys):
                    outside = c[:, j] - projector @ (projector.T @ s @ c[:, j])
                    check(name + "/originalP_leak_" + str(key), abs(float(outside @ s @ outside)),
                          gates["S_squared_originalP_leak"])
            save(name + "_point.npz", x=coordinates, keys=np.array(functional.keys),
                 P=np.array([waves[k][0] for k in functional.keys]),
                 Q=np.array([waves[k][1] for k in functional.keys]), **arrays)

        base_waves = unpack(state.coordinates)
        for i, key in enumerate(functional.keys):
            for side, part in enumerate(("P", "Q")):
                check("same_originalP_x_radial_bytes_" + str(key) + part,
                      base_waves[key][side].tobytes() != saved_point[part][i].tobytes())
                check("same_own_recovered_CI_point_" + str(key) + part,
                      base_waves[key][side].tobytes() != saved_CI[part][i].tobytes())
        point_constraints("baseline", state.coordinates, base_waves)
        save("immutable_original_geometry_metric.npz", x=state.coordinates,
             inverse_diagonal=state.inverse_diagonal,
             **{f"projector_{i}": r[1] for i, r in enumerate(state.layout)},
             **{f"inverse_blocks_{i}": np.stack(b) for i, b in enumerate(state.inverse_blocks)})

        def matching(name, weights, tracked):
            check(name + "/five_equal_weights", not (np.count_nonzero(weights) == 5
                  and np.array_equal(weights[weights > 0], np.repeat(.2, 5))))
            matched = {}
            for row in tracked:
                if len(row["reference_levels"]) != 1 or len(row["selected_levels"]) != 1:
                    raise ValueError("Only original rank-one physical targets admitted")
                uid, selected = tuple(row["reference_levels"][0]), tuple(row["selected_levels"][0])
                if uid in matched or uid not in expected_UIDs or uid[:2] != selected[:2]:
                    raise ValueError("Duplicate/changed physical UID")
                matched[uid] = row
            check(name + "/complete_UID_map", set(matched) != set(expected_UIDs))
            return matched

        def fresh(name, waves):
            current_CI[0] = name
            coverage[name] = set()
            before = {k: hS_requests[k] for k in ("attempted", "completed", "hits", "misses")}
            ci = ci_module.ci_solve(csfs, waves, functional.keys, data, 70, 123,
                nuclear_potential=nucleus, convol_cache=functional.cache, solver="dense",
                n_states=None, include_breit=False, max_determinants=2000, retain_hamiltonian=True)
            hS_requests["by_CI"][name] = {k: hS_requests[k] - v for k, v in before.items()}
            check(name + "/all7_CI_hS_coverage", coverage[name] != set(fixed["kappas"]))
            h = ci.hamiltonian_for(csfs, waves, functional.keys, data, 70, 123,
                nuclear_potential=nucleus, max_determinants=2000)
            energy, returned = functional.evaluate(waves, gradient=False, ci_result=ci)
            check(name + "/same_real_owner_reused", returned is not ci)
            weights, tracked = tracker.resolve(csfs, ci, waves, functional.metric)
            check(name + "/reported_EO_not_quotient", energy.hex() != float(weights @ ci.state_energies).hex())
            matched = matching(name, weights, tracked)
            _, blocks = ci_level_blocks(csfs, ci, require_full=True)
            rows = []
            for label, uid_list in fixed["original_UIDs"]:
                uid = tuple(uid_list)
                j, parity, rank = matched[uid]["selected_levels"][0]
                column = int(blocks[j, parity][rank])
                check(name + "/selected_weight_" + label, weights[column] != .2)
                rows.append(dict(label=label, UID=uid, selected_level=(j, parity, rank), column=column,
                    weight=float(weights[column]), energy_Ha=float(ci.state_energies[column]), tracker=matched[uid]))
            by_label = {r["label"]: r for r in rows}
            lower = by_label["6s_1/2"]
            check(name + "/lower_UID", tuple(lower["UID"]) != (1, 1, 0))
            for r in rows:
                r["excitation_cm1"] = (r["energy_Ha"] - lower["energy_Ha"]) * HARTREE_CM1
                if "baseline" in points:
                    old = {v["label"]: v for v in points["baseline"]["rows"]}
                    check(name + "/same_original_UID_" + r["label"], r["UID"] != old[r["label"]]["UID"])
                    r["delta_lower_Ha"] = lower["energy_Ha"] - old["6s_1/2"]["energy_Ha"]
                    r["delta_upper_Ha"] = r["energy_Ha"] - old[r["label"]]["energy_Ha"]
                    r["delta_transition_cm1"] = r["excitation_cm1"] - old[r["label"]]["excitation_cm1"]
            multiplets = {}
            for term, labels in (("5d", ("5d_3/2", "5d_5/2")), ("6p", ("6p_1/2", "6p_3/2"))):
                lo, hi = (by_label[label] for label in labels)
                degeneracies = (lo["UID"][0] + 1, hi["UID"][0] + 1)
                multiplets[term] = dict(centroid_cm1=(degeneracies[0] * lo["excitation_cm1"]
                    + degeneracies[1] * hi["excitation_cm1"]) / sum(degeneracies),
                    splitting_cm1=hi["excitation_cm1"] - lo["excitation_cm1"])
            points[name] = dict(rows=rows, EOL_reported_Ha=energy, multiplets=multiplets,
                physical_point=name, operator="dirac-coulomb", original_tracker_not_rebased=True,
                actual_EO_ULP_Ha=math.ulp(energy), no_stationarity_claim=True)
            save(name + "_whole_CI.npz", H=h, energies=ci.state_energies, mixing=ci.mixing_coefficients,
                 keys=np.array(functional.keys), P=np.array([waves[k][0] for k in functional.keys]),
                 Q=np.array([waves[k][1] for k in functional.keys]))
            write(output / (name + "_owner.json"), ci.hamiltonian_snapshot.identity_payload())
            write(output / (name + "_tracking.json"), dict(weights=weights, tracked=tracked))
            write(output / "five_point_reported_states.json", points)
            check(name + "/complete_H_symmetric", not np.array_equal(h, h.T))
            check(name + "/all1595_CI_Gram", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients
                  - np.eye(1595))), gates["all_complete_CI_Gram"])
            check(name + "/all1595_CI_residual_Ha", np.max(np.linalg.norm(h @ ci.mixing_coefficients
                  - ci.mixing_coefficients * ci.state_energies, axis=0)), gates["all_complete_CI_residual_Ha"])
            for sector in sectors:
                inside = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector]
                outside = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) != sector]
                check(name + "/cross_sector_" + str(sector), np.max(abs(h[np.ix_(inside, outside)])),
                      gates["cross_symmetry_zero_Ha"])
            return ci, h

        base_ci, base_h = fresh("baseline", base_waves)
        for name, value in (("H", base_h), ("energies", base_ci.state_energies),
                            ("mixing", base_ci.mixing_coefficients)):
            check("own_recovered31_baseline_bitwise_" + name,
                  not (np.array_equal(value, saved_CI[name]) and value.tobytes() == saved_CI[name].tobytes()))
        check("same_own_baseline_owner", plain(base_ci.hamiltonian_snapshot.identity_payload())
              != load(roles["recovered_owner"]))
        check("same_own_baseline_tracking", plain(load(output / "baseline_tracking.json"))
              != load(roles["recovered_tracking"]))
        qualified = load(roles["gradient_qualification_result"])
        check("same_own_baseline_EO", points["baseline"]["EOL_reported_Ha"].hex()
              != qualified["points"]["recovered"]["EOL_reported_Ha"].hex())

        # One predetermined complete pair direction, never a fitted pair/step scan.
        eta0 = np.zeros(3689, dtype=np.float64)
        saved_direction_rows = []
        for keys, _, shape, where in state.layout:
            kappa = keys[0][1]
            y = state.coordinates[where].reshape(shape)
            if kappa not in (-4, 3):
                continue
            gg = saved_g["full_normalized_raw_coordinate"][where].reshape(shape)
            omega = .5 * (y.T @ gg - gg.T @ y)
            selection = np.zeros_like(omega)
            old, new = keys.index((4, kappa)), keys.index((5, kappa))
            selection[old, new], selection[new, old] = omega[old, new], omega[new, old]
            independent = -y @ selection
            recorded = saved_g[f"full_normalized_old4f_newf_antisymmetric_{kappa}"]
            check("whole_saved_direction_reconstruction_" + str(kappa),
                  np.max(abs(independent - recorded)), gates["whole_direction_source_reconstruction"])
            eta0[where] = recorded.ravel()
            saved_direction_rows.append(dict(kappa=kappa, source_dtype=str(recorded.dtype),
                cast_to_native_float64_max=float(np.max(abs(recorded - eta0[where].reshape(shape))))))
        norm = float(np.linalg.norm(eta0))
        if not math.isfinite(norm) or norm < contract["direction"]["minimum_norm_for_admission"]:
            raise ValueError("Direction is zero/nonfinite; no substitute direction admitted")
        eta = eta0 / norm
        for keys, _, shape, where in state.layout:
            y, ee = state.coordinates[where].reshape(shape), eta[where].reshape(shape)
            absolute = float(np.max(abs(y.T @ ee + ee.T @ y)))
            input_norm = float(np.linalg.norm(ee))
            check("combined_scaled_tangent_" + str(keys[0][1]), absolute / max(1., input_norm),
                  gates["scaled_tangent_identity"])
            saved_direction_rows.append(dict(kappa=keys[0][1], absolute_tangent_defect=absolute,
                input_G_norm=input_norm, scaled_defect=absolute / max(1., input_norm)))
        save("predetermined_combined_direction.npz", direction=eta, unnormalized=eta0,
             norm=np.array(norm), keys=np.array(functional.keys), x=state.coordinates)
        write(output / "direction_identity.json", dict(rows=saved_direction_rows,
              full3689_norm=norm, no_per_kappa_normalization=True, no_gap_selection=True,
              qualified_saved_gradient_not_a_finite_response_certificate=True))

        # Consumer-only exact represented-number algebra. No physical kernels.
        fraction_cache = {}

        def f(value):
            pair = value.as_integer_ratio()
            if pair not in fraction_cache:
                fraction_cache[pair] = Fraction(*pair)
            return fraction_cache[pair]

        def mul(a, b):
            exact_products[0] += 1
            if exact_products[0] > caps["exact_Gram_Lambda_products"]:
                raise RuntimeError("Complete signed exact Gram/Lambda product cap")
            return a * b

        def dot(a, b):
            return sum((mul(x, y) for x, y in zip(a, b, strict=True)), Fraction(0))

        def vector(values):
            return [f(v) for v in values]

        def sparse(a):
            return [[(int(j), f(row[j])) for j in np.flatnonzero(row)] for row in a]

        def action(rows, v):
            return [sum((mul(value, v[j]) for j, value in row), Fraction(0)) for row in rows]

        def difference(a, b):
            return [x - y for x, y in zip(a, b, strict=True)]

        def add(a, b):
            return [x + y for x, y in zip(a, b, strict=True)]

        def rational(v):
            return dict(numerator=str(v.numerator), denominator=str(v.denominator), display=float(v))

        def matrix_display(a):
            return [[float(v) for v in row] for row in a]

        base_vectors = {key: vector(np.concatenate(base_waves[key])) for key in functional.keys}
        corrected = saved_g["raw_h_quotient_corrected_radial"]
        check("whole_corrected_radial_shape", corrected.shape != (31, 2, 123))
        ambient = {key: vector(np.concatenate(corrected[i])) for i, key in enumerate(functional.keys)}
        groups, ideal = [], {}
        lambdas, base_grams, base_Sc = {}, {}, {}
        for keys, projector, shape, where in state.layout:
            kappa = keys[0][1]
            rows = sparse(functional.metric[kappa])
            ideal_rows = sparse(projector)
            for j, key in enumerate(keys):
                ideal[key] = action(ideal_rows, vector(eta[where].reshape(shape)[:, j]))
                base_Sc[key] = action(rows, base_vectors[key])
            lambdas[kappa] = [[(dot(base_vectors[a], ambient[b]) + dot(base_vectors[b], ambient[a])) / 4
                              for b in keys] for a in keys]
            base_grams[kappa] = [[dot(base_vectors[a], base_Sc[b]) for b in keys] for a in keys]
            groups.append((keys, rows))
        ideal_S = {key: action(rows, ideal[key]) for keys, rows in groups for key in keys}
        ideal_Gram_action = Fraction(0)
        ideal_Gram_ledgers = {}
        for keys, _ in groups:
            kappa = keys[0][1]
            dm = [[dot(base_vectors[a], ideal_S[b]) + dot(ideal[a], base_Sc[b]) for b in keys] for a in keys]
            ideal_Gram_ledgers[kappa] = dm
            ideal_Gram_action += sum((mul(lambdas[kappa][i][j], dm[i][j])
                                     for i in range(len(keys)) for j in range(len(keys))), Fraction(0))
        retained_analytic = dot(vector(saved_g["retained_raw_coordinate"]), vector(eta))
        normalized_analytic = dot(vector(saved_g["full_normalized_raw_coordinate"]), vector(eta))
        ambient_action = sum((dot(ambient[key], ideal[key]) for key in functional.keys), Fraction(0))
        centered_analytic = ambient_action - ideal_Gram_action
        diagnostics["analytic"] = dict(public_retained=rational(retained_analytic),
            full_normalized=rational(normalized_analytic), ambient_action=rational(ambient_action),
            fixedLambda_ideal_Gram_action=rational(ideal_Gram_action),
            centered_analytic=rational(centered_analytic),
            finite_CI_response_not_included_in_HF_partial=True)
        write(output / "baseline_full_signed_Lambda.json", dict(
            same_kappa_Lambda={str(k): [[rational(v) for v in row] for row in a] for k, a in lambdas.items()},
            baseline_Grams={str(k): matrix_display(a) for k, a in base_grams.items()},
            ideal_Gram_derivatives={str(k): matrix_display(a) for k, a in ideal_Gram_ledgers.items()},
            analytic=diagnostics["analytic"], exact_all_signed_offdiagonal=True,
            input_gradient_dtype=str(corrected.dtype), no_global_bound=True))

        centered_values = {}
        trial_names = {}
        for index, h in enumerate(fixed["h"]):
            for sign in (-1, 1):
                name = f"h{index}_{'minus' if sign < 0 else 'plus'}"
                trial_names[index, sign] = name
                candidate = state.coordinates.copy()
                for _, _, shape, where in state.layout:
                    y = state.coordinates[where].reshape(shape)
                    candidate[where] = optimization._polar_retraction(
                        y + sign * h * eta[where].reshape(shape)).ravel()
                waves = unpack(candidate)
                increments = unpack(candidate - state.coordinates)
                point_constraints(name, candidate, waves)
                save(name + "_increments.npz", keys=np.array(functional.keys),
                     coordinate_delta=candidate - state.coordinates,
                     P=np.array([increments[key][0] for key in functional.keys]),
                     Q=np.array([increments[key][1] for key in functional.keys]))
                trial_ci, trial_h = fresh(name, waves)
                mixed = functional.mixed_ci_energy_difference(base_waves, waves, base_ci, trial_ci,
                                                               increments=increments)
                check(name + "/mixed_diagnostic_only", mixed["diagnostic_only"] is not True
                      or mixed["certified_error_bound"] is not False or mixed["optimizer_adapter"] is not False)
                check(name + "/full_signed_density_capacity", mixed["signed_density_terms"],
                      caps["signed_term_work_cap"]["complete_angular_term_capacity_per_trial"])
                check(name + "/actual_owner_H_identity", list(mixed["endpoint_H_sha256"])
                      != [base_ci.hamiltonian_snapshot.matrix_sha256, trial_ci.hamiltonian_snapshot.matrix_sha256])
                check(name + "/mixed_five_UIDs", {tuple(t["reference_level"]) for t in mixed["targets"]}
                      != set(expected_UIDs))
                check(name + "/mixed_original_weights", any(t["weight"] != .2 for t in mixed["targets"]))
                write(output / (name + "_mixed_finite_change.json"), mixed)
                del trial_h, trial_ci
                total = {key: difference(vector(np.concatenate(waves[key])), base_vectors[key])
                         for key in functional.keys}
                explicit = {key: vector(np.concatenate(increments[key])) for key in functional.keys}
                reconstruction = {key: difference(total[key], explicit[key]) for key in functional.keys}
                sd = {key: action(rows, total[key]) for keys, rows in groups for key in keys}
                se = {key: action(rows, explicit[key]) for keys, rows in groups for key in keys}
                sr = {key: action(rows, reconstruction[key]) for keys, rows in groups for key in keys}
                normal = Fraction(0)
                ledgers, gram_arrays = {}, {}
                for keys, _ in groups:
                    kappa = keys[0][1]
                    n = len(keys)
                    dm = [[Fraction(0)] * n for _ in range(n)]
                    expm, recm, linear, quadratic = ([r.copy() for r in dm] for _ in range(4))
                    trial_gram = [[Fraction(0)] * n for _ in range(n)]
                    components = dict(diagonal=Fraction(0), offdiagonal=Fraction(0),
                                      linear=Fraction(0), quadratic=Fraction(0),
                                      explicit=Fraction(0), reconstruction=Fraction(0))
                    for i, a in enumerate(keys):
                        for j, b in enumerate(keys):
                            linear[i][j] = dot(base_vectors[a], sd[b]) + dot(total[a], base_Sc[b])
                            quadratic[i][j] = dot(total[a], sd[b])
                            dm[i][j] = linear[i][j] + quadratic[i][j]
                            expm[i][j] = (dot(base_vectors[a], se[b]) + dot(explicit[a], base_Sc[b])
                                         + dot(explicit[a], se[b]))
                            recm[i][j] = (dot(base_vectors[a], sr[b]) + dot(reconstruction[a], base_Sc[b])
                                + dot(explicit[a], sr[b]) + dot(reconstruction[a], se[b])
                                + dot(reconstruction[a], sr[b]))
                            trial_gram[i][j] = dot(add(base_vectors[a], total[a]), add(base_Sc[b], sd[b]))
                            if dm[i][j] != trial_gram[i][j] - base_grams[kappa][i][j]:
                                raise ValueError("Full nonsym S actual trialGram-baseGram exact identity failed")
                            if dm[i][j] != expm[i][j] + recm[i][j]:
                                raise ValueError("Full explicit/reconstruction Gram exact identity failed")
                            lam = lambdas[kappa][i][j]
                            components["diagonal" if i == j else "offdiagonal"] += mul(lam, dm[i][j])
                            for field, table in (("linear", linear), ("quadratic", quadratic),
                                                 ("explicit", expm), ("reconstruction", recm)):
                                components[field] += mul(lam, table[i][j])
                    normal += components["diagonal"] + components["offdiagonal"]
                    check(name + "/exact_actual_Gram_identity_" + str(kappa), False)
                    check(name + "/exact_reconstruction_Gram_identity_" + str(kappa), False)
                    gram_arrays.update({f"{prefix}_{kappa}": np.array(matrix_display(table)) for prefix, table in
                        (("DeltaGram", dm), ("explicit_DeltaGram", expm), ("reconstruction_DeltaGram", recm),
                         ("linear_DeltaGram", linear), ("quadratic_DeltaGram", quadratic))})
                    ledgers[str(kappa)] = dict(keys=keys, exact_signed_Lambda_DeltaGram={k: rational(v)
                                              for k, v in components.items()},
                                              all_Gram_identities_exact=True, no_offdiagonal_or_quadratic_omission=True)
                centered = f(mixed["rayleigh_change"]) - normal
                centered_values[index, sign] = centered
                diagnostics[name] = dict(mixed_change=mixed, fixedLambda_DeltaGram=rational(normal),
                    centered_rayleigh_change=rational(centered), samepoint_fixedLambda=ledgers,
                    production_reported_objective_unchanged=True, direct_storedH_forensic_only=True)
                save(name + "_exact_Gram_displays.npz", **gram_arrays)
                write(output / (name + "_centered_diagnostic.json"), diagnostics[name])
                progress("completed_" + name)

        # Domain gates are evaluated only after both h ledgers exist. An
        # ordinary FAIL remains a FAIL even when the independent diagnostic passes.
        for index, h in enumerate(fixed["h"]):
            minus, plus = (points[trial_names[index, sign]]["EOL_reported_Ha"] for sign in (-1, 1))
            ordinary = (plus - minus) / (2 * h)
            centered = (centered_values[index, 1] - centered_values[index, -1]) / (2 * f(h))
            analytic = float(retained_analytic)
            rule = max(gates["FD_absolute"], gates["FD_relative"] * abs(analytic))
            ordinary_row = dict(h=h, derivative=ordinary, analytic=analytic,
                error=ordinary - analytic, threshold=rule, passed=abs(ordinary - analytic) <= rule,
                endpoint_ULP_over_2h=(math.ulp(plus) + math.ulp(minus)) / (2 * h),
                difference_against_full_normalized=ordinary - float(normalized_analytic),
                reported_target_and_EO_offsets_in_mixed_ledgers=True)
            centered_rows = []
            for label, reference in (("fixedLambda_ideal", centered_analytic), ("public_retained", retained_analytic)):
                error = centered - reference
                threshold = max(gates["FD_absolute"], gates["FD_relative"] * abs(float(reference)))
                centered_rows.append(dict(reference=label, derivative=rational(centered), analytic=rational(reference),
                    error=rational(error), threshold=threshold, passed=abs(error) <= f(threshold)))
            derivative_domains[str(h)] = dict(ordinary_reported=ordinary_row,
                independent_centered=centered_rows, production_callback=False, no_uniform_error_bound=True)
            checks.append(dict(name=f"ordinary_reported_FD_h{index}", error=abs(ordinary - analytic),
                               limit=rule, passed=ordinary_row["passed"], domain="ordinary_reported"))
            for row in centered_rows:
                checks.append(dict(name=f"centered_{row['reference']}_FD_h{index}",
                    error=abs(row["error"]["display"]), limit=row["threshold"], passed=row["passed"],
                    domain="independent_centered_diagnostic"))
        write(output / "derivative_domains.json", derivative_domains)
        for name, exact in (("H", 5), ("CI", 5), ("public_reported_energy_evaluate", 5),
                            ("initial_hS", 7), ("mixed_radial_hS", 28), ("validation_Dirac", 7),
                            ("geometry_SPD_eigvalsh", 38), ("polar_retraction", 28),
                            ("tracker_resolve", 18), ("tracking_SVD", 90), ("tracker_overlap", 19),
                            ("mixed_ci_finite_increment", 4), ("unfiltered_target_RDM", 20),
                            ("retained_RDM", 0), ("gradient_from_weights", 0), ("optimizer", 0), ("GPU", 0)):
            check("exact_source_calls_" + name, completed[name] != exact)
        check("actual_hS_miss_producers", hS_requests["misses"] != completed["CI_internal_hS"])
        check("original_hS_cache_misses", hS_requests["misses"] != 7)
        check("all_five_carry_events", len(hS_requests["carry_events"]) != 5)
        for row in hS_requests["carry_events"]:
            check("unaltered_native_hS_carry_" + row["CI"], not row["identical_tuples"])
        for name, module in tuple(sys.modules.items()):
            if (name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None):
                check("immutable_import_" + name, not Path(module.__file__).resolve().is_relative_to(source.resolve()))
        all_passed = all(row["passed"] for row in checks)
        final = dict(status="native31_single_combined_direction_two_h_FD_completed",
            all_declared_controls_passed=all_passed, checks=checks, attempted=attempted, completed=completed,
            archives=stages, points=points, hS_requests=hS_requests, diagnostics=diagnostics,
            derivative_domains=derivative_domains, exact_products=exact_products[0],
            angular_capacity=term_capacity, new_metric_sha256=state.residual_metric_sha256,
            new_geometry_sha256=state.geometry_sha256, new_state_sha256=state.state_sha256,
            objective_identity=state.objective_identity, source_map_sha256=fixed["source_map_sha256"],
            contract_sha256=sha(HERE / "contract.json"), driver_sha256=sha(__file__),
            exact_execution_binding_sha256=sha(binding_path),
            single_direction_finite_measurement_only=True, full31_space_certificate=False,
            original_stationarity_passed=False, expanded_MCDHF_optimized=False,
            no_energy_callback=True, ordinary_reported_objective_unchanged=True,
            old_failed_scientific_gates_unchanged=True, experimental_gate_passed=False,
            elapsed_uncontrolled_s=time.monotonic() - started,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            timing_claim_valid=False, actual_affinity=sorted(os.sched_getaffinity(0)))
        write(output / "result.json", final)
        progress("completed_all_domains" if all_passed else "completed_with_failed_FD_domains")
        print(json.dumps(dict(status=final["status"], all_passed=all_passed,
                              controls=len(checks), completed=completed)), flush=True)
        if not all_passed:
            raise ValueError("Original finite derivative gates failed; all four finite ledgers preserved")
    except BaseException as error:
        write(output / "failure.json", dict(status="failed_preserved_no_retry",
            exception=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            attempted=attempted, completed=completed, checks=checks, archives=stages, points=points,
            diagnostics=diagnostics, derivative_domains=derivative_domains, exact_products=exact_products[0],
            hS_requests=hS_requests, elapsed_uncontrolled_s=time.monotonic() - started))
        progress("failed")
        raise
    finally:
        for module, name, original in reversed(restored):
            setattr(module, name, original)
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-binding")
    args = parser.parse_args()
    if args.execution_binding:
        execute(Path(args.execution_binding))
    else:
        contract, _, _, deferred = metadata(arrays=False)
        print(json.dumps(dict(mode="SOURCE_JSON_metadata_only", source_map=contract["fixed"]["source_map_sha256"],
              NPZ_reads=0, deferred_arrays=len(deferred), physical_calls=0)))


if __name__ == "__main__":
    main()
