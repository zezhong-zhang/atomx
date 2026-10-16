"""Private native31 gradient discriminator; default mode is source/JSON only.

Two complete CI owners bind the canonical saved31 and the separately recovered
3689-coordinate point. This consumer neither optimizes nor changes reported EOL.
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
    """No NumPy/Atomx imports. Preparation must not open NPZ bytes."""
    contract = load(HERE / "contract.json")
    roles = {name: ROOT / row["path"] for name, row in contract["input_roles"].items()}
    deferred = []
    for name, row in contract["input_roles"].items():
        path = roles[name]
        if path.suffix == ".npz" and not arrays:
            deferred.append(dict(role=name, **row))
        elif sha(path) != row["sha256"]:
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
                                   ("canonical_result", "canonical_root_receipt", "canonical_completion"))
    if (result["all_original_numeric_identity_gates_passed"] is not True
            or len(result["checks"]) != 941 or any(not r["passed"] for r in result["checks"])
            or receipt["status"] != "foreground_closed" or receipt["exit_code"] != 0
            or completion["all_completion_metadata_checks_passed"] is not True
            or completion["all_actual_live_controls_passed"] is not True
            or completion["total_actual_live_controls"] != 941
            or result["source_map_sha256"] != contract["fixed"]["source_map_sha256"]
            or result["expanded_MCDHF_executed"] is not False):
        raise ValueError("Completed fixed-RCI prerequisite or its scope changed")
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
    raw_operators, coverage = {}, {"canonical": set(), "recovered": set()}
    current_CI = [None]
    hS_requests = dict(attempted=0, completed=0, hits=0, misses=0,
                       by_CI={}, carry_events=[])
    started = time.monotonic()

    def progress(phase):
        write(output / "progress.json", dict(phase=phase, attempted=attempted,
              completed=completed, checks_count=len(checks),
              last_check=checks[-1] if checks else None, archives_count=len(stages),
              last_archive=stages[-1] if stages else None, completed_point_names=list(points),
              hS_requests=hS_requests, elapsed_uncontrolled_s=time.monotonic() - started))

    def timeout(*_):
        raise TimeoutError("Declared300s whole-run cap")

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
                if attempted[name] > limits[name]:
                    progress("overbudget_" + name)
                    raise RuntimeError("Prospective call cap: " + name)
                quiet = name == "gradient_pair_potential"
                if not quiet:
                    progress("before_" + name)
                value = function(*args, **kwargs)
                completed[name] += 1
                if capture:
                    kappa = int(args[1])
                    h, s = value
                    if kappa in raw_operators:
                        a, b = raw_operators[kappa]
                        if not (np.array_equal(a, h) and np.array_equal(b, s)):
                            raise ValueError("Raw h/S changed under fixed geometry")
                    else:
                        raw_operators[kappa] = h.copy(), s.copy()
                    save(f"captured_{name}_{completed[name]:02d}.npz",
                         kappa=np.array(kappa), raw_h=h, S=s)
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
        patch(matrix, "build_ci_matrix", "H")
        patch(ci_module, "ci_solve", "CI")
        restored.append((variational, "ci_solve", variational.ci_solve))
        variational.ci_solve = ci_module.ci_solve
        patch(dirac, "solve_dirac", "validation_Dirac")
        patch(optimization, "optimize_orbitals", "optimizer")
        patch(variational.CSFOrbitalFunctional, "gradient_from_weights", "gradient_from_weights")
        patch(variational.CSFOrbitalFunctional, "_pair_potential", "gradient_pair_potential")
        patch(CIRadialDensity, "contract", "retained_RDM")
        patch(CIRadialDensity, "contract_bilinear", "unfiltered_target_RDM")
        patch(tracking_module.CIBasisTracking, "resolve", "tracker_resolve")
        patch(tracking_module, "ci_subspace_overlaps", "tracker_overlap")
        patch(np.linalg, "svd", "tracking_SVD")
        patch(np.linalg, "eigvalsh", "geometry_SPD")
        patch(linalg, "eigh", "complete_eigh")

        original_blocks = matrix._dirac_blocks
        original_reset = matrix._reset_memo_for_ci_build
        restored.extend(((matrix, "_dirac_blocks", original_blocks),
                         (matrix, "_reset_memo_for_ci_build", original_reset)))

        def blocks(ns, kappa, Z, workspace):
            hS_requests["attempted"] += 1
            if hS_requests["attempted"] > caps["maximum_hS_getter_requests"]:
                progress("overbudget_hS_getter")
                raise RuntimeError("h/S getter request cap")
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

        canonical, oldpoint, radial, original_hS = (archive(r) for r in
                ("canonical31_arrays", "literal208_arrays", "geometry_workspace", "screened_geometry"))
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
              != {(r["twoJ"], r["parity"]): r["CSFs"] for r in fixed["target_blocks"]})
        check("full_signed_CSF_order", [signature(c) for c in csfs] != basis)
        check("same_charge69", {c.n_electrons for c in full} != {69})
        write(output / "basis_identity.json", dict(signatures=basis, full_parent3789_enumerated=True,
              complete_target1595_diagonalized=True, global_spectrum_certified=False))
        canonical_keys = tuple(tuple(map(int, k)) for k in canonical["keys"])
        check("canonical31_keys", canonical_keys != tuple(map(tuple, fixed["keys31"])))
        check("public31_basis", tuple(sorted(get_shared_orbital_basis(csfs))) != canonical_keys)
        waves = {k: (p.copy(), q.copy()) for k, p, q in
                 zip(canonical_keys, canonical["P"], canonical["Q"], strict=True)}
        for k, p, q in zip(oldpoint["keys"], oldpoint["P"], oldpoint["Q"], strict=True):
            check("canonical_old29_bytes_" + str(tuple(k)),
                  not (waves[tuple(k)][0].tobytes() == p.tobytes()
                       and waves[tuple(k)][1].tobytes() == q.tobytes()))
        reference_models = load(roles["reference_models"])["models"]
        reference_csfs = [c for row in reference_models["baseline"]["configs"]
                          for c in enumerate_jj_csfs(row["config"])]
        check("unchanged_reference531", len(reference_csfs) != 531)
        reference_ci = archive("tracking_ci")
        reference_waves = {tuple(map(int, k)): (p.copy(), q.copy()) for k, p, q in
                           zip(radial["orbital_keys"], radial["P"], radial["Q"], strict=True)}
        targets = load(roles["tracking_binding"])["targets"]
        expected_UIDs = {tuple(uid): label for label, uid in fixed["target_UIDs"]}
        check("original_UIDs", {(t["twoJ"], t["parity"], t["models"]["DC"]["local_root"])
              for t in targets} != set(expected_UIDs))
        average = CIStateAverage(tuple((t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.)
                                      for t in targets))
        metrics = {k: original_hS[f"s_{k}"] for k in fixed["kappas"]}
        tracker = tracking_module.CIBasisTracking(average, reference_csfs,
            ci_module.CIResult(reference_ci["energies_Ha"], reference_ci["mixing"]),
            reference_waves, metrics, max_determinant_pairs=caps["maximum_signed_overlap_pairs"])
        nucleus = make_nuclear_potential(70, "fermi", A=175)
        functional = variational.CSFOrbitalFunctional(csfs, data, 70, nucleus,
            state_average=average, level_tracking=tracker, retain_hamiltonian=True,
            max_determinants=2000)
        check("functional_exact31_order", functional.keys != canonical_keys)
        for kappa in fixed["kappas"]:
            h, s = raw_operators[kappa]
            check("actual_L_raw_h_" + str(kappa),
                  not np.array_equal(functional.one_body[kappa], np.tril(h) + np.tril(h, -1).T))
            check("saved_logical_h_" + str(kappa),
                  not np.array_equal(functional.one_body[kappa], original_hS[f"h_{kappa}"]))
            check("saved_S_" + str(kappa), not np.array_equal(s, original_hS[f"s_{kappa}"]))

        def matching(weights, tracked):
            check("equal_original_five_weights", not (np.count_nonzero(weights) == 5
                  and np.array_equal(weights[weights > 0], np.repeat(.2, 5))))
            matched = {}
            for r in tracked:
                if len(r["reference_levels"]) != 1 or len(r["selected_levels"]) != 1:
                    raise ValueError("Ambiguous/degenerate target not supported by this contract")
                uid, selected = tuple(r["reference_levels"][0]), tuple(r["selected_levels"][0])
                if uid in matched or uid not in expected_UIDs or uid[:2] != selected[:2]:
                    raise ValueError("Changed/duplicate physical UID")
                matched[uid] = r
            check("complete_original_UID_map", set(matched) != set(expected_UIDs))
            return matched

        def describe(name, current, ci, h, weights, tracked):
            matched = matching(weights, tracked)
            _, level_blocks = ci_level_blocks(csfs, ci, require_full=True)
            rows = []
            for label, uid in fixed["target_UIDs"]:
                uid = tuple(uid)
                j, p, rank = matched[uid]["selected_levels"][0]
                column = int(level_blocks[j, p][rank])
                check(name + "/sector_weight_" + label, weights[column] != .2)
                rows.append(dict(label=label, UID=uid, selected_level=(j, p, rank),
                    column=column, weight=float(weights[column]), energy_Ha=float(ci.state_energies[column]),
                    tracker=matched[uid]))
            lower = next(r for r in rows if r["label"] == "6s_1/2")
            check(name + "/lower_UID", tuple(lower["UID"]) != (1, 1, 0))
            for r in rows:
                r["excitation_cm1"] = (r["energy_Ha"] - lower["energy_Ha"]) * HARTREE_CM1
            points[name] = dict(rows=rows, EOL_reported_Ha=float(weights @ ci.state_energies),
                                physical_point=name, operator="dirac-coulomb",
                                original_tracker_not_rebased=True)
            write(output / (name + "_tracking.json"), dict(weights=weights, tracked=tracked))
            check(name + "/all_H_symmetric", not np.array_equal(h, h.T))
            check(name + "/all1595_CI_Gram", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients
                  - np.eye(1595))), gates["CI_Gram"])
            check(name + "/all1595_CI_residual_Ha", np.max(np.linalg.norm(h @ ci.mixing_coefficients
                  - ci.mixing_coefficients * ci.state_energies, axis=0)), gates["CI_residual_Ha"])
            for sector in sectors:
                inside = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector]
                outside = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) != sector]
                check(name + "/cross_sector_" + str(sector), np.max(abs(h[np.ix_(inside, outside)])),
                      gates["cross_symmetry_zero_Ha"])
            save(name + "_whole_CI.npz", H=h, energies=ci.state_energies, mixing=ci.mixing_coefficients,
                 keys=np.array(functional.keys), P=np.array([current[k][0] for k in functional.keys]),
                 Q=np.array([current[k][1] for k in functional.keys]))
            write(output / (name + "_owner.json"), ci.hamiltonian_snapshot.identity_payload())

        def fresh(name, current):
            current_CI[0] = name
            before = {k: hS_requests[k] for k in ("attempted", "completed", "hits", "misses")}
            ci = ci_module.ci_solve(csfs, current, functional.keys, data, 70, 123,
                nuclear_potential=nucleus, convol_cache=functional.cache, solver="dense",
                n_states=None, include_breit=False, max_determinants=2000, retain_hamiltonian=True)
            hS_requests["by_CI"][name] = {k: hS_requests[k] - v for k, v in before.items()}
            check(name + "/all_seven_CI_hS_kappas", coverage[name] != set(fixed["kappas"]))
            h = ci.hamiltonian_for(csfs, current, functional.keys, data, 70, 123,
                                  nuclear_potential=nucleus, max_determinants=2000)
            return ci, h

        canonical_ci, canonical_h = fresh("canonical", waves)
        canonical_weights, canonical_tracking = tracker.resolve(csfs, canonical_ci, waves, functional.metric)
        describe("canonical", waves, canonical_ci, canonical_h, canonical_weights, canonical_tracking)
        for name, value in (("H", canonical_h), ("energies", canonical_ci.state_energies),
                            ("mixing", canonical_ci.mixing_coefficients)):
            check("canonical_saved_bitwise_" + name,
                  not (np.array_equal(value, canonical[name]) and value.tobytes() == canonical[name].tobytes()))
        check("canonical_saved_owner", plain(canonical_ci.hamiltonian_snapshot.identity_payload())
              != load(roles["canonical_owner"]))
        old_assignment = {tuple(r["reference_levels"][0]): tuple(r["selected_levels"][0])
                          for r in load(roles["canonical_tracking"])["tracked"]}
        check("canonical_saved_UID_assignment", {tuple(r["reference_levels"][0]): tuple(r["selected_levels"][0])
              for r in canonical_tracking} != old_assignment)
        check("canonical_saved_EO_bitwise", float(canonical_weights @ canonical_ci.state_energies).hex()
              != float(load(roles["canonical_evidence"])["EOL_reported_Ha"]).hex())

        old_layout = [(tuple(map(tuple, r["keys"])), oldpoint[f"projector_{i}"],
                      tuple(r["shape"]), slice(*r["slice"]))
                      for i, r in enumerate(load(roles["saved_layout"]))]
        old_inverse = [oldpoint[f"inverse_blocks_{i}"] for i in range(7)]
        check("original3451_metric_SHA", optimization._residual_metric_sha256(
            old_layout, oldpoint["inverse_diagonal"], old_inverse) != fixed["original_metric_sha256"])
        new_layout, inverses, coordinates, diagonals, layout_rows = [], [], [], [], []
        cursor = 0
        for i, (old_keys, projector, shape, where) in enumerate(old_layout):
            kappa = old_keys[0][1]
            keys = tuple(k for k in functional.keys if k[1] == kappa)
            old_y = oldpoint["x"][where].reshape(shape)
            old_d = oldpoint["inverse_diagonal"][where].reshape(shape)
            mapping = {k: j for j, k in enumerate(old_keys)}
            yy, dd, bb = [], [], []
            for k in keys:
                if k in mapping:
                    j = mapping[k]
                    yy.append(old_y[:, j].copy())
                    dd.append(old_d[:, j].copy())
                    bb.append(old_inverse[i][j].copy())
                else:
                    check("only_declared_new_f", k not in ((5, 3), (5, -4)))
                    j = mapping[4, kappa]
                    yy.append(projector.T @ functional.metric[kappa] @ np.concatenate(waves[k]))
                    dd.append(old_d[:, j].copy())
                    bb.append(old_inverse[i][j].copy())
            y, diagonal = np.column_stack(yy), np.column_stack(dd)
            new_shape = y.shape
            stop = cursor + y.size
            new_layout.append((keys, projector, new_shape, slice(cursor, stop)))
            inverses.append(tuple(bb))
            coordinates.append(y.ravel())
            diagonals.append(diagonal.ravel())
            layout_rows.append(dict(keys=keys, shape=new_shape, slice=(cursor, stop),
                                    new_f_inverse_source="original same-kappa4f SPD only" if kappa in (-4, 3) else None))
            for k, j in mapping.items():
                jj = keys.index(k)
                check("old29_coordinate_bytes_" + str(k), y[:, jj].tobytes() != old_y[:, j].tobytes())
                check("old29_inverse_bytes_" + str(k), bb[jj].tobytes() != old_inverse[i][j].tobytes())
                check("old29_inverse_diagonal_bytes_" + str(k), diagonal[:, jj].tobytes() != old_d[:, j].tobytes())
            check("same_originalP_bytes_" + str(kappa), projector.tobytes() != oldpoint[f"projector_{i}"].tobytes())
            cursor = stop
        x, diagonal = np.concatenate(coordinates), np.concatenate(diagonals)
        check("new3689_dimension", x.shape != (3689,))
        objective_payload = dict(source_map_sha256=fixed["source_map_sha256"],
            full_signed_CSF_order_sha256=hashlib.sha256(json.dumps(basis, separators=(",", ":")).encode()).hexdigest(),
            original_UIDs=fixed["target_UIDs"], weights=[.2]*5, operator="dirac-coulomb",
            reference_CI_sha256=contract["input_roles"]["tracking_ci"]["sha256"],
            reference_radials_sha256=contract["input_roles"]["geometry_workspace"]["sha256"],
            nucleus="FermiA175", orbital_keys=functional.keys, target="ordinary reported EOL",
            point="original29Y columns plus2f recovered; authoritative originalP@Y")
        objective = hashlib.sha256(json.dumps(plain(objective_payload), sort_keys=True,
                                   separators=(",", ":")).encode()).hexdigest()
        write(output / "new_layout.json", layout_rows)
        write(output / "new_objective_identity.json", dict(identity=objective, payload=objective_payload))
        state = optimization.OrbitalOptimizationState.from_arrays(functional,
            layout=new_layout, coordinates=x, inverse_diagonal=diagonal, inverse_blocks=inverses,
            projector_source="nuclear", objective_identity=objective, occupation_floor=.05)
        state.validate(functional)
        recovered, drift = {}, []
        actual_Grams = {}
        for keys, projector, shape, where in state.layout:
            y = state.coordinates[where].reshape(shape)
            c = projector @ y
            s = functional.metric[keys[0][1]]
            check("new_Y_Gram_" + str(keys[0][1]), np.max(abs(y.T @ y - np.eye(len(keys)))), gates["radial_Gram"])
            check("new_actual_SGram_" + str(keys[0][1]), np.max(abs(c.T @ s @ c - np.eye(len(keys)))), gates["radial_Gram"])
            actual_Grams["coordinate_Gram_" + str(keys[0][1])] = y.T @ y
            actual_Grams["physical_Gram_" + str(keys[0][1])] = c.T @ s @ c
            canonical_group = np.column_stack([np.concatenate(waves[k]) for k in keys])
            actual_Grams["canonical_physical_Gram_" + str(keys[0][1])] = canonical_group.T @ s @ canonical_group
            check("canonical_full_samek_Gram_" + str(keys[0][1]),
                  np.max(abs(actual_Grams["canonical_physical_Gram_" + str(keys[0][1])]
                             - np.eye(len(keys)))), gates["radial_Gram"])
            for j, k in enumerate(keys):
                recovered[k] = c[:123, j].copy(), c[123:, j].copy()
                delta = c[:, j] - np.concatenate(waves[k])
                drift.append(dict(key=k, canonical_vs_public_max=float(np.max(abs(delta))),
                                  bitwise_equal=c[:, j].tobytes() == np.concatenate(waves[k]).tobytes(),
                                  original29_coordinate_preserved=k in set(map(tuple, oldpoint["keys"]))))
                check("declared_recovery_max_" + str(k), np.max(abs(delta)), gates["recovery_max"])
                outside = c[:, j] - projector @ (projector.T @ s @ c[:, j])
                check("originalP_leak_" + str(k), abs(float(outside @ s @ outside)), gates["P_leak_S_squared"])
        write(output / "point_recovery.json", dict(rows=drift, canonical_and_recovered_samepoint_claim=False,
              original3451_metric_SHA=fixed["original_metric_sha256"],
              new3689_metric_SHA=state.residual_metric_sha256, new_geometry_SHA=state.geometry_sha256,
              new_state_SHA=state.state_sha256, new_objective_identity=objective))
        save("recovered3689_point.npz", x=state.coordinates, keys=np.array(functional.keys),
             P=np.array([recovered[k][0] for k in functional.keys]),
             Q=np.array([recovered[k][1] for k in functional.keys]),
             inverse_diagonal=state.inverse_diagonal,
             **{f"projector_{i}": r[1] for i, r in enumerate(state.layout)},
             **{f"inverse_blocks_{i}": np.stack(b) for i, b in enumerate(state.inverse_blocks)})
        save("complete_samek_Grams.npz", **actual_Grams)

        recovered_ci, recovered_h = fresh("recovered", recovered)
        # Gradient reuses this actual point's owner. evaluate() does not solve CI again.
        energy, retained_g, reused_ci = functional.evaluate(recovered, gradient=True, ci_result=recovered_ci)
        check("real_recovered_owner_reused", reused_ci is not recovered_ci)
        weights, tracked = functional.last_state_weights, functional.last_tracking_diagnostics
        describe("recovered", recovered, recovered_ci, recovered_h, weights, tracked)
        check("reported_EO_not_quotient", energy.hex() != float(weights @ recovered_ci.state_energies).hex())
        canonical_rows = {r["label"]: r for r in points["canonical"]["rows"]}
        for row in points["recovered"]["rows"]:
            previous = canonical_rows[row["label"]]
            check("recovered_same_UID_" + row["label"], row["UID"] != previous["UID"])
            row["delta_lower_Ha"] = (next(r["energy_Ha"] for r in points["recovered"]["rows"]
                if r["label"] == "6s_1/2") - canonical_rows["6s_1/2"]["energy_Ha"])
            row["delta_upper_Ha"] = row["energy_Ha"] - previous["energy_Ha"]
            row["delta_transition_cm1"] = row["excitation_cm1"] - previous["excitation_cm1"]
        write(output / "canonical_vs_recovered_five_states.json", points)
        retained = functional.last_weights
        matched = matching(weights, tracked)
        _, level_blocks = ci_level_blocks(csfs, recovered_ci, require_full=True)
        contributions, target_tables = {}, []
        for label, uid in fixed["target_UIDs"]:
            uid = tuple(uid)
            j, p, rank = matched[uid]["selected_levels"][0]
            column = int(level_blocks[j, p][rank])
            c = recovered_ci.mixing_coefficients[:, column]
            norm = sum((Fraction.from_float(float(v))**2 for v in c), Fraction(0))
            if norm <= 0:
                raise ValueError("Nonpositive target CI norm")
            table = functional.radial_density.contract_bilinear(c, c)
            for k, value in table.items():
                contributions.setdefault(k, []).append(Fraction.from_float(float(weights[column]))
                    * Fraction.from_float(float(value)) / norm)
            target_tables.append(dict(label=label, UID=uid, column=column,
                exact_CI_norm=(str(norm.numerator), str(norm.denominator)),
                norm_defect=float(norm - 1), unfiltered=[[k, v] for k, v in table.items()]))

        def represent(value):
            result = float(value)
            if not math.isfinite(result) or (value != 0 and result == 0):
                raise ValueError("Unrepresentable nonzero angular coefficient; no silent omission")
            return result

        exact = {k: sum(v, Fraction(0)) for k, v in contributions.items()}
        full_weights = {k: represent(v) for k, v in exact.items() if v != 0}
        ordered = list(full_weights) + [k for k in retained if k not in full_weights]
        delta_weights = {}
        for k in ordered:
            value = Fraction.from_float(full_weights.get(k, 0.)) - Fraction.from_float(retained.get(k, 0.))
            if value != 0:
                delta_weights[k] = represent(value)
        write(output / "signed_tables.json", dict(targets=target_tables,
            retained=[[k, v] for k, v in retained.items()],
            full_normalized=[[k, v] for k, v in full_weights.items()],
            full_minus_retained=[[k, v] for k, v in delta_weights.items()],
            normalization="exact binary64 CI norms and .2 rational aggregate then explicitfloat64 coefficient rounding",
            delta_order="full order then retained-only original order",
            exact_zero_only=True, coefficient_counts_are_not_bounds=True))
        delta_g = functional.gradient_from_weights(recovered, delta_weights, record_diagnostics=False)
        full_g = functional.gradient_from_weights(recovered, full_weights, record_diagnostics=False)
        retained_array = np.array([retained_g[k] for k in functional.keys])
        delta_array = np.array([delta_g[k] for k in functional.keys])
        full_array = np.array([full_g[k] for k in functional.keys])
        plus_delta = retained_array + delta_array
        save("all_logical_gradients.npz", keys=np.array(functional.keys), retained=retained_array,
             full_minus_retained=delta_array, retained_plus_delta=plus_delta, direct_full=full_array)
        check("whole31_signed_linearity", np.max(abs(plus_delta - full_array)), gates["whole_gradient_linearity"])

        wide = np.longdouble
        corrected = {k: full_g[k].astype(wide) for k in functional.keys}
        vectors = {k: np.concatenate(recovered[k]).astype(wide) for k in functional.keys}
        for term, coefficient in full_weights.items():
            if term[0] != "h":
                continue
            _, na, kappa, nb = term
            a, b = (na, kappa), (nb, kappa)
            h, s = (v.astype(wide) for v in raw_operators[kappa])
            logical = functional.one_body[kappa].astype(wide)
            va, vb = vectors[a], vectors[b]
            sb, hb, hta = s @ vb, h @ vb, h.T @ va
            norm, numerator = vb @ sb, va @ hb
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("Nonpositive physical radial ket norm")
            corrected[a] += coefficient * (hb / norm - logical @ vb)
            corrected[b] += coefficient * (hta / norm - logical @ va - 2 * numerator * sb / norm**2)
        corrected_array = np.array([corrected[k] for k in functional.keys])

        def coordinate(radial_gradient):
            result = np.zeros(3689, dtype=radial_gradient.dtype)
            lookup = {k: radial_gradient[i] for i, k in enumerate(functional.keys)}
            for keys, projector, _, where in state.layout:
                result[where] = (projector.astype(result.dtype).T
                                @ np.column_stack([lookup[k] for k in keys])).ravel()
            return result

        def tangent(vector):
            result = vector.copy()
            for _, _, shape, where in state.layout:
                y, g = state.coordinates[where].reshape(shape).astype(vector.dtype), vector[where].reshape(shape)
                multiplier = y.T @ g
                result[where] = (g - .5 * y @ (multiplier + multiplier.T)).ravel()
            return result

        def metric_action(vector):
            result = vector.copy()
            for (_, _, shape, where), inverse in zip(state.layout, state.inverse_blocks, strict=True):
                g = vector[where].reshape(shape)
                result[where] = np.column_stack([b.astype(vector.dtype) @ g[:, j]
                                                for j, b in enumerate(inverse)]).ravel()
            return tangent(result)

        all_vectors, action_metrics, columns, accessibility, constraints = {}, {}, [], [], []
        actions = dict(retained=retained_array, retained_plus_delta=plus_delta,
                       full_normalized=full_array, raw_h_quotient_corrected=corrected_array)
        for name, radial_gradient in actions.items():
            raw = coordinate(radial_gradient)
            tg = tangent(raw)
            pg = metric_action(tg)
            all_vectors[name + "_raw_coordinate"] = raw
            all_vectors[name + "_tangent"] = tg
            all_vectors[name + "_preconditioned_tangent"] = pg
            action_metrics[name] = dict(raw_radial_norm=float(np.linalg.norm(radial_gradient)),
                tangent_norm=float(np.linalg.norm(tg)), metric_residual=float(np.linalg.norm(pg)),
                indicator_Ha=.5 * abs(float(tg @ pg)),
                measured_new3689_metric=True, strict_thresholds=(1e-7, 1e-8),
                energy_or_spectrum_error_bound=False)
            for keys, _, shape, where in state.layout:
                y, gg = state.coordinates[where].reshape(shape).astype(raw.dtype), raw[where].reshape(shape)
                tt, pp = tg[where].reshape(shape), pg[where].reshape(shape)
                for j, k in enumerate(keys):
                    columns.append(dict(action=name, key=k, raw_coordinate_norm=float(np.linalg.norm(gg[:, j])),
                        tangent_norm=float(np.linalg.norm(tt[:, j])), pg_norm=float(np.linalg.norm(pp[:, j])),
                        pg_norm_squared=float(pp[:, j] @ pp[:, j]), signed_g_dot_pg=float(tt[:, j] @ pp[:, j]),
                        original_or_new="newf" if k in ((5, 3), (5, -4)) else "original29"))
                defect = float(np.max(abs(y.T @ tt + tt.T @ y)))
                input_norm = float(np.linalg.norm(gg))
                constraints.append(dict(action=name, component="whole_samek_tangent",
                    kappa=keys[0][1], absolute_defect=defect, input_G_norm=input_norm,
                    scaled_defect=defect / max(1., input_norm),
                    no_stationarity_or_error_bound=True))
                check(name + "/scaled_samek_tangent_" + str(keys[0][1]),
                      defect / max(1., input_norm), gates["tangent_identity_scaled"])
                if keys[0][1] not in (-4, 3):
                    continue
                new = keys.index((5, keys[0][1]))
                old = keys.index((4, keys[0][1]))
                ext = np.zeros_like(gg)
                ext[:, new] = -(gg[:, new] - y @ (y.T @ gg[:, new]))
                omega = .5 * (y.T @ gg - gg.T @ y)
                restricted = np.zeros_like(omega)
                restricted[old, new], restricted[new, old] = omega[old, new], omega[new, old]
                rotation = -y @ restricted
                for component, direction in (("newf_external", ext), ("old4f_newf_antisymmetric", rotation)):
                    defect = float(np.max(abs(y.T @ direction + direction.T @ y)))
                    constraints.append(dict(action=name, component=component,
                        kappa=keys[0][1], absolute_defect=defect, input_G_norm=input_norm,
                        scaled_defect=defect / max(1., input_norm),
                        no_stationarity_or_error_bound=True))
                    check(name + "/" + component + "_scaled_tangent_" + str(keys[0][1]),
                          defect / max(1., input_norm), gates["tangent_identity_scaled"])
                    all_vectors[name + "_" + component + "_" + str(keys[0][1])] = direction
                    accessibility.append(dict(action=name, component=component, kappa=keys[0][1],
                        Frobenius_norm=float(np.linalg.norm(direction)),
                        signed_HF_partial_action=float(np.sum(gg * direction)),
                        no_auxiliary_Fock_Euler_claim=True, finite_CI_response_not_included=True))
        check("whole3689_tangent_linearity", np.max(abs(all_vectors["retained_plus_delta_tangent"]
              - all_vectors["full_normalized_tangent"])), gates["projected_metric_linearity"])
        check("whole3689_metric_linearity", np.max(abs(all_vectors["retained_plus_delta_preconditioned_tangent"]
              - all_vectors["full_normalized_preconditioned_tangent"])), gates["projected_metric_linearity"])
        save("full31_gradient_metric.npz", keys=np.array(functional.keys), x=state.coordinates,
             retained_radial=retained_array, full_minus_retained_radial=delta_array,
             full_normalized_radial=full_array, raw_h_quotient_corrected_radial=corrected_array,
             inverse_diagonal=state.inverse_diagonal, **all_vectors,
             **{f"projector_{i}": r[1] for i, r in enumerate(state.layout)},
             **{f"inverse_blocks_{i}": np.stack(b) for i, b in enumerate(state.inverse_blocks)})
        write(output / "all31_column_actions.json", columns)
        write(output / "newf_accessible_actions.json", accessibility)
        write(output / "complete_tangent_constraints.json", constraints)
        write(output / "action_metrics.json", dict(metrics=action_metrics,
            full_vs_retained={name: float(np.linalg.norm(all_vectors["full_normalized_" + name]
                - all_vectors["retained_" + name])) for name in
                ("raw_coordinate", "tangent", "preconditioned_tangent")},
            corrected_vs_full={name: float(np.linalg.norm(all_vectors["raw_h_quotient_corrected_" + name]
                - all_vectors["full_normalized_" + name])) for name in
                ("raw_coordinate", "tangent", "preconditioned_tangent")},
            declared_diagnostic_metric=state.residual_metric_sha256,
            HF_partial_not_finiteCIresponse=True, no_error_bound=True))
        for name, exact in (("H", 2), ("CI", 2), ("initial_hS", 7),
                            ("validation_Dirac", 7), ("geometry_SPD", 38),
                            ("gradient_from_weights", 3), ("unfiltered_target_RDM", 5),
                            ("retained_RDM", 1), ("tracker_resolve", 2),
                            ("tracking_SVD", 10), ("tracker_overlap", 3), ("optimizer", 0)):
            check("exact_completed_source_calls_" + name, completed[name] != exact)
        check("total_hS_producer_cap", completed["initial_hS"] + completed["CI_internal_hS"], 21)
        for name, module in tuple(sys.modules.items()):
            if (name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None):
                check("immutable_import_" + name, not Path(module.__file__).resolve().is_relative_to(source.resolve()))
        write(output / "result.json", dict(status="native31_full_signed_gradient_discriminator_completed",
            all_declared_controls_passed=True, checks=checks, attempted=attempted, completed=completed,
            archives=stages, points=points, hS_requests=hS_requests, recovery=drift,
            new_metric_sha256=state.residual_metric_sha256, new_geometry_sha256=state.geometry_sha256,
            new_state_sha256=state.state_sha256, objective_identity=objective,
            original3451_metric_sha256=fixed["original_metric_sha256"],
            source_map_sha256=fixed["source_map_sha256"], contract_sha256=sha(HERE / "contract.json"),
            driver_sha256=sha(__file__), exact_execution_binding_sha256=sha(binding_path),
            action_metrics=action_metrics, accessible_actions=accessibility,
            original_stationarity_passed=False, expanded_MCDHF_optimized=False,
            full_gradient_is_HF_partial=True, finite_CI_response_qualified=False,
            stationarity_not_inferred_from_auxiliary_Fock=True, experimental_gate_passed=False,
            no_old3451_residual_improvement_claim=True, no_energy_callback=True,
            elapsed_uncontrolled_s=time.monotonic() - started,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            timing_claim_valid=False, actual_affinity=sorted(os.sched_getaffinity(0))))
        progress("completed")
        print(json.dumps(dict(status="completed", controls=len(checks), completed=completed)), flush=True)
    except BaseException as error:
        write(output / "failure.json", dict(status="failed_preserved_no_retry",
            exception=type(error).__name__, message=str(error), traceback=traceback.format_exc(),
            attempted=attempted, completed=completed, checks=checks, archives=stages, points=points,
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
