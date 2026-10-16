"""Bounded private semiseparable-H qualification; requires exact root seal.

Preparation imports only stdlib. Numerical workers run in separate processes;
anchors precede two genuine-public retained-H/complete-gradient chains. Stage
profiles are uncontrolled observations and certify no optimizer/GPU speedup.
"""
import argparse
import cProfile
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
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
    with temporary.open("w") as handle:
        handle.write(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def admit(execution):
    contract = json.loads((HERE / "contract.json").read_text())
    prep = json.loads((HERE / "preparation_verification.json").read_text())
    required = {"execution_authorized_by_root": True, "driver_sha256": sha(__file__),
                "contract_sha256": sha(HERE / "contract.json"),
                "preparation_sha256": sha(HERE / "preparation_verification.json"),
                "ordinary_source_map_sha256": contract["ordinary"]["source_map_sha256"],
                "candidate_source_map_sha256": contract["candidate"]["source_map_sha256"],
                "CPU": 30, "max_H": 2, "max_CI": 2, "max_gradient": 2,
                "max_scipy_eigh": 17,
                "wall_seconds": 180, "address_space_GiB": 3.95,
                "purpose": "new_centered_epoch_complete29_equivalence_uncontrolled_profile"}
    if any(execution.get(k) != v for k, v in required.items()):
        raise ValueError("Missing or changed exact root execution seal")
    if (prep.get("all_passed") is not True or prep.get("driver_sha256") != required["driver_sha256"]
            or prep.get("contract_sha256") != required["contract_sha256"]):
        raise ValueError("Preparation proof mismatch")
    for row in contract["inputs"].values():
        if sha(REPO / row["path"]) != row["sha256"]:
            raise ValueError("Immutable input mismatch: " + row["path"])
    baseline = json.loads((REPO / contract["inputs"]["corrected_baseline_result"]["path"]).read_text())
    closure = json.loads((REPO / contract["inputs"]["corrected_baseline_completion"]["path"]).read_text())
    if not (baseline.get("all_numeric_controls_passed") is True
            and baseline.get("corrected_samepoint_full_gradient_and_centered_directions_completed") is True
            and closure.get("all_completion_metadata_checks_passed") is True
            and baseline.get("source_map_sha256") == contract["lineage"]["actual_baseline_source_map"]):
        raise ValueError("Corrected samepoint baseline prerequisite is not closed and passed")
    for variant in ("ordinary", "candidate"):
        row = contract[variant]
        identity = json.loads((REPO / row["identity"]).read_text())
        if identity["source_map_sha256"] != row["source_map_sha256"]:
            raise ValueError("Source identity mismatch")
        for name, expected in identity["all_atomx_python_sha256"].items():
            if sha(REPO / row["archive"] / name.removeprefix("src/")) != expected:
                raise ValueError("Frozen source mismatch: " + name)
    if any(os.environ.get(k) != "1" for k in contract["resources"]["thread_variables"]):
        raise ValueError("Explicit one-thread environment required")
    if 30 not in os.sched_getaffinity(0):
        raise ValueError("CPU30 unavailable")
    # Correctness may run at high load; these timings are never controlled.
    return contract


def worker(phase, execution):
    contract = admit(execution)
    resource.setrlimit(resource.RLIMIT_AS, (int(3.95 * 2**30), int(3.95 * 2**30)))
    os.sched_setaffinity(0, {30})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Shared180s cap")))
    signal.alarm(max(1, int(execution["remaining_seconds"])))
    variant = "candidate" if phase == "anchors" else phase
    source = REPO / contract[variant]["archive"]
    if any(name == "atomx" or name.startswith("atomx.") for name in sys.modules):
        raise ValueError("Physical module imported before admission")
    sys.path.insert(0, str(source))
    import numba
    import numpy as np
    import scipy

    import atomx.numerics.bspline.convol as convol
    import atomx.physics.csf.matrix as matrix
    output = HERE / "actual" / phase
    output.mkdir(exist_ok=False)
    stages, checks, started, completed, seconds = {}, [], {}, {}, {}
    clock = time.monotonic()

    def flush(status):
        write(output / "progress.json", {"phase": status, "started": started,
              "completed": completed, "call_wall_seconds_uncontrolled": seconds,
              "stages_uncontrolled_seconds": stages, "checks": checks,
              "elapsed_uncontrolled_seconds": time.monotonic() - clock})

    def gate(name, condition, error=None, limit=None):
        passed = bool(condition)
        row = {"name": name, "passed": passed}
        if error is not None:
            row.update(error=float(error), limit=float(limit))
        checks.append(row)
        if not passed:
            flush(name)
            raise ValueError("Qualification gate failed: " + name)

    def close(name, a, b, atol=0., rtol=0.):
        delta = abs(np.asarray(a) - np.asarray(b))
        bounds = atol + rtol * abs(np.asarray(b))
        gate(name, np.all(np.isfinite(delta)) and np.all(delta <= bounds),
             np.max(delta, initial=0.), np.max(bounds, initial=0.))

    def wrap(name, function, cap=None, capture=None):
        def call(*args, **kwargs):
            started[name] = started.get(name, 0) + 1
            if cap is not None and started[name] > cap:
                flush("cap_" + name)
                raise RuntimeError("Call cap exceeded: " + name)
            before = time.monotonic()
            result = function(*args, **kwargs)
            seconds[name] = seconds.get(name, 0.) + time.monotonic() - before
            completed[name] = completed.get(name, 0) + 1
            if capture is not None:
                capture(args, result)
            return result
        return call

    # Forward-only instrumentation preserves the actual numerical functions.
    density = convol._density_sym
    dense = convol._convol_case1_t
    factored = convol._convol_case1_semiseparable_t
    summation = convol._sum_amb_banded
    for module in (convol, matrix):
        module._density_sym = wrap("density_band", density)
        module._convol_case1_t = wrap("direct_full_bank", dense)
        module._convol_case1_semiseparable_t = wrap("direct_semiseparable", factored)
        module._sum_amb_banded = wrap("band_scalar", summation)
    matrix._bspline_pair_potential = wrap("CI_source_pair", matrix._bspline_pair_potential)
    convol.ConvolCache.get_rkb_t = wrap("bank_get", convol.ConvolCache.get_rkb_t)
    convol.ConvolCache.get_coulomb_factors = wrap("factor_get", convol.ConvolCache.get_coulomb_factors)
    profiler = cProfile.Profile()
    try:
        profiler.enable()
        if phase == "anchors":
            from scipy import linalg

            import atomx.methods.ci as ci_module
            from atomx.numerics.bspline import basis
            from atomx.numerics.bspline.knots import make_knots
            linalg.eigh = wrap("scipy_eigh", linalg.eigh, 2)
            ci_module._centered_dense_eigh = wrap("centered_dense_helper", ci_module._centered_dense_eigh, 2)
            # Independent signed2x2 spectral projectors use the closed angle
            # formula, not a second LAPACK solver or the centering algorithm.
            for sign in (-1., 1.):
                offset, diagonal, coupling = -float(2**14), 2.**-13, sign * 2.**-17
                h = np.array([[offset - diagonal, coupling], [coupling, offset + diagonal]])
                original_bytes = h.tobytes()
                values, vectors, origin = ci_module._centered_dense_eigh(h, driver="evd")
                radius = math.hypot(diagonal, coupling)
                angle = .5 * math.atan2(coupling, diagonal)
                expected = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
                close(f"toy_alpha_{sign}", origin, offset)
                close(f"toy_energy_{sign}", values, offset + np.array([-radius, radius]), 1e-8)
                close(f"toy_Gram_{sign}", vectors.T @ vectors, np.eye(2), 1e-10)
                close(f"toy_original_H_residual_{sign}", np.linalg.norm(h @ vectors - vectors * values, axis=0), 0., 1e-8)
                for column in (0, 1):
                    close(f"toy_signed_projector_{sign}_{column}", np.outer(vectors[:, column], vectors[:, column]),
                          np.outer(expected[:, column], expected[:, column]), 1e-12)
                gate(f"toy_original_H_bytes_{sign}", h.tobytes() == original_bytes)
            basis.prepare_bspline_data = wrap("basis_setup", basis.prepare_bspline_data, 7)
            # NumPy leggauss uses one small companion-matrix eigvalsh per
            # geometry; this is declared quadrature setup, not a Dirac solve.
            np.linalg.eigvalsh = wrap("Gauss_setup_eigvalsh", np.linalg.eigvalsh, 7)
            for index, case in enumerate(contract["anchor_setup"]["cases"]):
                ns, ks, balance = case["ns"], case["ks"], case["kinetic_balance"]
                analytic = case.get("monomial_degrees")
                knots = (np.r_[np.zeros(ks), np.full(ks, .7)] if analytic is not None
                         else make_knots(ns, ks, .0125, 50.))
                data = basis.prepare_bspline_data(knots, ns, ks,
                    n_gauss=16 if analytic is not None else None, kinetic_balance=balance)
                cache = convol.ConvolCache(data)
                cache.release_shared_banks()
                orders = data.get("ksp", ks), data.get("ksq", ks)
                if analytic is not None:
                    from math import comb
                    coeff = []
                    for order, degree in zip(orders, analytic, strict=True):
                        active = np.array([0. if i < degree else .7**degree *
                            comb(i, degree) / comb(order - 1, degree) for i in range(order)])
                        coeff.append(np.r_[active, np.zeros(ns - order)])
                    waves = tuple((scale * coeff[0], scale * coeff[1]) for scale in (1., -2., 3., 4.))
                    if balance:
                        gate("exact_RKB_unused_P_tail_zero", np.all(coeff[0][orders[0]:] == 0.))
                else:
                    rng = np.random.Generator(np.random.PCG64(19011 + index))
                    waves = tuple((rng.normal(size=ns), rng.normal(size=ns)) for _ in range(4))
                np.savez_compressed(output / f"seed_{index}.npz", **data,
                    **{f"wave_{i}_{j}": v for i, pair in enumerate(waves) for j, v in enumerate(pair)})
                target = tuple(convol._density_sym(waves[0][j], waves[1][j], ns, orders[j]) for j in range(2))
                source_density = tuple(convol._density_sym(waves[2][j], waves[3][j], ns, orders[j]) for j in range(2))
                for rank in case["ranks"]:
                    reference_parts = [np.zeros((ns, order)) for order in orders]
                    analytic_parts = [0., 0.]
                    for left in range(2):
                        for right in range(2):
                            labels = ("p", "q")
                            component = labels[left] + labels[right] + labels[left] + labels[right]
                            factors = cache.get_coulomb_factors(rank, component)
                            bank, extent, _ = cache.get_rkb_t(rank, component)
                            near = factors[0]
                            ip, jp, i, offset = np.indices(near.shape)
                            first_j = i + ip - orders[left] + 1
                            j = first_j + offset
                            valid = (j >= 0) & (j < ns) & (j <= i + orders[right] - jp - 1)
                            close(f"near_{index}_{rank}_{component}", near[valid],
                                  bank[ip[valid], jp[valid], i[valid], j[valid]], 1e-14, 2e-12)
                            a = convol._convol_case1_t(bank, source_density[right], extent, orders[left], orders[right])
                            b = convol._convol_case1_semiseparable_t(*factors, source_density[right], ns,
                                                                    orders[left], orders[right])
                            close(f"potential_{index}_{rank}_{component}", b, a, 1e-12, 2e-12)
                            va = convol._sum_amb_banded(target[left], a, ns, orders[left])
                            vb = convol._sum_amb_banded(target[left], b, ns, orders[left])
                            close(f"signed_{index}_{rank}_{component}", vb, va, 1e-10, 2e-12)
                            if analytic is not None:
                                degree_left, degree_right = 2 * analytic[left], 2 * analytic[right]
                                expected = (-24 * .7**(degree_left + degree_right + 1)
                                    / (degree_left + degree_right + 1)
                                    * (1 / (degree_left + rank + 1) + 1 / (degree_right + rank + 1)))
                                close(f"analytic_dense_{index}_{rank}_{component}", va, expected, 0., 2e-13)
                                close(f"analytic_factored_{index}_{rank}_{component}", vb, expected, 0., 2e-13)
                                analytic_parts[left] += expected
                            reference_parts[left] += a
                    counts = {k: started.get(k, 0) for k in ("direct_full_bank", "direct_semiseparable")}
                    routed = matrix._bspline_pair_potential(cache, ns, orders, rank, source_density)
                    branch = "direct_semiseparable" if ns >= 96 else "direct_full_bank"
                    gate(f"route_{index}_{rank}", started.get(branch, 0) - counts[branch] == 4)
                    for left in range(2):
                        close(f"routed_{index}_{rank}_{left}", routed[left], reference_parts[left], 2e-12, 2e-12)
                        scalar = convol._sum_amb_banded(target[left], routed[left], ns, orders[left])
                        if analytic is not None:
                            close(f"analytic_routed_{index}_{rank}_{left}", scalar,
                                  analytic_parts[left], 0., 2e-13)
                cache.release_shared_banks()
                flush("geometry_" + str(index))
            for name, expected in contract["anchor_calls"].items():
                gate("exact_calls_" + name, started.get(name, 0) == expected and completed.get(name, 0) == expected)
            stages["anchors_total"] = time.monotonic() - clock
            arrays = None
            owner = None
        else:
            arrays, owner = chain(phase, contract, output, np, wrap, gate, close, stages, flush)
        profiler.disable()
        profiler.dump_stats(output / "profile.pstats")
        import pstats
        with (output / "profile.txt").open("w") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(60)
        for name, module in tuple(sys.modules.items()):
            if ((name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None)):
                gate("source_import_" + name, Path(module.__file__).resolve().is_relative_to(source.resolve()))
        # Same immutable sources/inputs must still match after the physical chain.
        admit(execution)
        result = {"all_passed": True, "phase": phase, "calls_started": started,
                  "calls_completed": completed, "call_wall_seconds_uncontrolled": seconds,
                  "stages_uncontrolled_seconds": stages, "checks": checks,
                  "versions": {"Python": sys.version, "NumPy": np.__version__,
                               "SciPy": scipy.__version__, "Numba": numba.__version__},
                  "physical_stationarity_qualification": False, "GPU_called": False,
                  "controlled_speedup_certified": False,
                  "host_load": list(os.getloadavg()), "source_map_sha256": contract[variant]["source_map_sha256"],
                  "arrays_sha256": None if arrays is None else sha(arrays),
                  "owner_sha256": None if owner is None else sha(owner)}
        write(output / "result.json", result)
        flush("completed")
    except BaseException as exc:
        profiler.disable()
        write(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
              "started": started, "completed": completed, "checks": checks,
              "elapsed_uncontrolled_seconds": time.monotonic() - clock})
        flush("failed")
        raise


def chain(phase, contract, output, np, wrap, gate, close, stages, flush):
    from scipy import linalg

    import atomx.methods.ci as ci_module
    import atomx.methods.csf.variational as variational
    import atomx.numerics.bspline.dirac_assembly as dirac
    import atomx.physics.csf.matrix as matrix
    import atomx.physics.csf.tracking as tracking
    from atomx.angular.csf_hamiltonian import csf_signature
    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.ci import CIResult
    from atomx.numerics.bspline.orbital_optimization import _residual_metric_sha256
    from atomx.physics.csf.levels import ci_level_blocks
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.potentials.nuclear import make_nuclear_potential

    raw_hs = []
    def capture(args, result):
        raw_hs.append((int(args[1]), result[0].copy(), result[1].copy()))
    original = dirac.build_dirac_matrices
    variational.build_dirac_matrices = wrap("initial_hS", original, 7, capture)
    matrix.build_dirac_matrices = wrap("CI_internal_hS", original, 7, capture)
    dirac.solve_dirac = wrap("Dirac_eigensolve", dirac.solve_dirac, 0)
    matrix.build_ci_matrix = wrap("H", matrix.build_ci_matrix, 1)
    variational.ci_solve = wrap("CI", variational.ci_solve, 1)
    variational.CSFOrbitalFunctional.gradient_from_weights = wrap(
        "gradient", variational.CSFOrbitalFunctional.gradient_from_weights, 1)
    variational.CSFOrbitalFunctional._pair_potential = wrap(
        "gradient_source_pair", variational.CSFOrbitalFunctional._pair_potential)
    tracking.CIBasisTracking.resolve = wrap("tracking_resolution", tracking.CIBasisTracking.resolve, 2)
    tracking.ci_subspace_overlaps = wrap("tracking_overlap", tracking.ci_subspace_overlaps, 3)
    np.linalg.svd = wrap("tracking_SVD", np.linalg.svd, 10)
    np.linalg.eigvalsh = wrap("eigvalsh", np.linalg.eigvalsh, 0)
    np.linalg.eigh = wrap("numpy_eigh", np.linalg.eigh, 0)
    eigen_cap = 10 if phase == "ordinary" else 5
    linalg.eigh = wrap("scipy_eigh", linalg.eigh, eigen_cap)
    centered_calls = []
    def capture_centered(args, result):
        centered_calls.append({"size": len(args[0]), "origin_Ha": float(result[2]),
                               "first_real_diagonal_Ha": float(args[0][0, 0].real)})
    ci_module._centered_dense_eigh = wrap("centered_dense_helper", ci_module._centered_dense_eigh,
                                        eigen_cap, capture_centered)
    # Gradient aliases count actual kernels as well as the CI matrix aliases.
    import atomx.numerics.bspline.convol as convol
    for name in ("_density_sym", "_convol_case1_t", "_convol_case1_semiseparable_t"):
        setattr(variational, name, getattr(convol, name))
    variational._banded_to_full_add = wrap("gradient_band_expand", variational._banded_to_full_add)

    def load(role):
        path = REPO / contract["inputs"][role]["path"]
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as saved:
                return {k: saved[k].copy() for k in saved.files}
        return json.loads(path.read_text())

    start = time.monotonic()
    saved, gradient, geometry, radial = (load(k) for k in ("saved_point", "saved_gradient", "screened_geometry", "geometry_workspace"))
    # The new reference is the genuine corrected-f043 baseline. Its historical
    # original500ac fields are comparison data, never an equivalence oracle.
    for public, stored in (("radial_gradient", "production_radial"),
                           ("raw_coordinate_gradient", "production_raw_coordinate"),
                           ("tangent_gradient", "production_tangent"),
                           ("preconditioned_gradient", "production_preconditioned")):
        gradient[public] = gradient[stored]
    waves = {tuple(map(int, key)): (p.copy(), q.copy()) for key, p, q in
             zip(saved["keys"], saved["P"], saved["Q"], strict=True)}
    data = {k[6:]: v.item() if v.ndim == 0 else v.copy() for k, v in radial.items() if k.startswith("basis_")}
    models = load("screened_models")["models"]
    old = [c for row in models["baseline"]["configs"] for c in enumerate_jj_csfs(row["config"])]
    full = [c for row in models["balanced_virtual"]["configs"] for c in enumerate_jj_csfs(row["config"])]
    sectors = {(j, p) for j, p, _ in contract["fixed"]["target_blocks"]}
    csfs = [c for c in full if (c.total_J2, c.parity) in sectors]
    gate("complete_dimensions_531_2163_1046_29", (len(old), len(full), len(csfs), len(waves)) == (531, 2163, 1046, 29))
    gate("complete69electrons", {c.n_electrons for c in csfs} == {69})
    for j, p, count in contract["fixed"]["target_blocks"]:
        gate(f"complete_sector_{j}_{p}", sum((c.total_J2, c.parity) == (j, p) for c in csfs) == count)
    signatures = [json.dumps(plain((s.subshells, s.coupling, c.total_J2, c.parity)), separators=(",", ":"))
                  for c in csfs for s in [csf_signature(c)]]
    gate("complete_signed_order", signatures == load("screened_basis")["balanced_virtual"]["retained_signatures"])
    targets = load("tracking_binding")["targets"]
    average = CIStateAverage(tuple((t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.) for t in targets))
    metric = {k: geometry[f"s_{k}"] for k in {key[1] for key in waves}}
    reference = load("tracking_ci")
    old_waves = {tuple(map(int, k)): (p.copy(), q.copy()) for k, p, q in
                 zip(radial["orbital_keys"], radial["P"], radial["Q"], strict=True)}
    tracker = tracking.CIBasisTracking(average, old, CIResult(reference["energies_Ha"], reference["mixing"]),
                                      old_waves, metric, max_determinant_pairs=1_000_000)
    nuclear = make_nuclear_potential(70, "fermi", A=175)
    functional = variational.CSFOrbitalFunctional(csfs, data, 70, nuclear, state_average=average,
        level_tracking=tracker, retain_hamiltonian=True, max_determinants=2000)
    gate("exact29radialkey_order", functional.keys == tuple(map(tuple, saved["keys"])))
    originals = {}
    for k, h, s in raw_hs:
        close(f"original_Lh_{k}", functional.one_body[k], np.tril(h) + np.tril(h, -1).T)
        close(f"saved_Lh_{k}", functional.one_body[k], geometry[f"h_{k}"])
        gate(f"saved_Lh_bytes_{k}", functional.one_body[k].tobytes() == geometry[f"h_{k}"].tobytes())
        close(f"saved_metric_{k}", s, metric[k])
        gate(f"saved_metric_bytes_{k}", s.dtype == metric[k].dtype and s.tobytes() == metric[k].tobytes())
        originals[k] = (h, s)
    rows = load("saved_layout")
    layout = [(tuple(map(tuple, row["keys"])), gradient[f"projector_{i}"], tuple(row["shape"]),
               slice(*row["slice"])) for i, row in enumerate(rows)]
    inverse = [gradient[f"inverse_blocks_{i}"] for i in range(7)]
    gate("original3451_metric_bytes", _residual_metric_sha256(layout, gradient["inverse_diagonal"], inverse)
         == contract["fixed"]["residual_metric_sha256"])
    close("exact3451coordinate_shape", gradient["x"].size, 3451)
    for i, (keys, projector, shape, where) in enumerate(layout):
        y = gradient["x"][where].reshape(shape)
        coefficients = projector @ y
        for col, key in enumerate(keys):
            close(f"original_coordinate_wave_{key}", np.concatenate(waves[key]), coefficients[:, col])
        close(f"coordinate_Gram_{i}", y.T @ y, np.eye(shape[1]), 1e-10)
        c = np.column_stack([np.concatenate(waves[key]) for key in keys])
        s = metric[keys[0][1]]
        close(f"radial_Gram_{i}", c.T @ s @ c, np.eye(shape[1]), 1e-10)
        outside = c - projector @ (projector.T @ s @ c)
        close(f"no_pair_leakage_{i}", np.sum(outside * (s @ outside), axis=0), 0., 1e-12)
    stages["source_point_angular_setup"] = time.monotonic() - start
    flush("setup_complete")

    start = time.monotonic()
    energy, ci = functional.evaluate(waves, gradient=False)
    h = ci.hamiltonian_for(csfs, waves, functional.keys, data, 70, 123,
                          nuclear_potential=nuclear, max_determinants=2000)
    groups = ci_module._group_csfs_by_J_parity(csfs)
    gate("complete_five_centered_public_blocks", len(groups) == len(centered_calls) == 5)
    public_origins = [{"sector": sector, **call} for (sector, _), call in zip(groups, centered_calls, strict=True)]
    close("actual_provenance_centered_origins", [call["origin_Ha"] for call in centered_calls],
          ci.provenance["dense_centering_shifts_Ha"])
    write(output / "centered_origins.json", public_origins)
    stages["public_energy_trial_including_tracking_owner"] = time.monotonic() - start
    owner = output / "owner.json"
    write(owner, ci.hamiltonian_snapshot.identity_payload())
    start = time.monotonic()
    close("whole_H_exact_symmetric", h, h.T)
    close("full_CI_column_norm_residual", np.max(np.linalg.norm(h @ ci.mixing_coefficients
          - ci.mixing_coefficients * ci.state_energies, axis=0)), 0., 1e-8)
    close("full_CI_Gram", ci.mixing_coefficients.T @ ci.mixing_coefficients, np.eye(1046), 1e-10)
    gate("read_only_owned_H", not h.flags.writeable)
    mutated = {k: (p.copy(), q.copy()) for k, (p, q) in waves.items()}
    mutated[functional.keys[0]][0][0] = np.nextafter(mutated[functional.keys[0]][0][0], np.inf)
    try:
        ci.hamiltonian_for(csfs, mutated, functional.keys, data, 70, 123,
                           nuclear_potential=nuclear, max_determinants=2000)
    except ValueError:
        pass
    else:
        raise ValueError("Stale physical point accepted")
    gate("stale_owner_rejected", True)
    stages["complete_trial_eigenpair_Gram_owner_validation"] = time.monotonic() - start
    start = time.monotonic()
    energy2, radial_g, returned = functional.evaluate(waves, ci_result=ci, gradient=True)
    gate("same_trial_CI_object", returned is ci)
    close("same_reported_objective", energy2, energy)
    stages["public_accepted_gradient_including_tracking"] = time.monotonic() - start
    raw = np.zeros(3451)
    for keys, projector, _shape, where in layout:
        raw[where] = (projector.T @ np.column_stack([radial_g[k] for k in keys])).ravel()

    def tangent(vector):
        answer = vector.copy()
        for _, _, shape, where in layout:
            y, g = gradient["x"][where].reshape(shape), vector[where].reshape(shape)
            a = y.T @ g
            answer[where] = (g - .5 * y @ (a + a.T)).ravel()
        return answer

    g = tangent(raw)
    inverse_g = g.copy()
    for (_, _, shape, where), blocks in zip(layout, inverse, strict=True):
        a = g[where].reshape(shape)
        inverse_g[where] = np.column_stack([block @ a[:, j] for j, block in enumerate(blocks)]).ravel()
    pg = tangent(inverse_g)
    close("saved_complete_radial_gradient", np.array([radial_g[k] for k in functional.keys]), gradient["radial_gradient"], 1e-9)
    close("saved_original_raw_g", raw, gradient["raw_coordinate_gradient"], 1e-9)
    close("saved_original_tangent_g", g, gradient["tangent_gradient"], 1e-9)
    close("saved_original_metric_pg", pg, gradient["preconditioned_gradient"], 1e-10)
    weights = functional.last_state_weights
    gate("original_equal5weights", np.count_nonzero(weights) == 5 and np.array_equal(weights[weights > 0], np.repeat(.2, 5)))
    _, blocks = ci_level_blocks(csfs, ci, require_full=True)
    write(output / "sector_columns.json", [{"sector": sector, "columns": columns,
          "CSF_rows": [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector]}
          for sector, columns in blocks.items()])
    write(output / "signed_density.json", [[key, float(value)] for key, value in functional.last_weights.items()])
    bindings = {}
    for row in functional.last_tracking_diagnostics:
        gate("rank_one_binding", len(row["reference_levels"]) == len(row["selected_levels"]) == 1)
        uid = tuple(row["reference_levels"][0])
        gate("unique_UID_" + str(uid), uid not in bindings and tuple(row["selected_levels"][0][:2]) == uid[:2])
        bindings[uid] = row
    reference_targets = load("corrected_baseline_result")["targets"]
    target_rows = []
    for target in targets:
        uid = target["twoJ"], target["parity"], target["models"]["DC"]["local_root"]
        selected = bindings[uid]["selected_levels"][0]
        column = int(blocks[tuple(selected[:2])][selected[2]])
        old_target = next(row for row in reference_targets if row.get("label") == target["label"])
        gate("same_original_target_" + target["label"], column == old_target["column"])
        target_rows.append({"label": target["label"], "UID": uid, "column": column,
                            "energy_Ha": float(ci.state_energies[column]), "tracking": bindings[uid]})
    gate("exact_five_UID_support", len(bindings) == len(target_rows) == 5)
    for k, raw_h, s in raw_hs[7:]:
        close(f"same_raw_h_{k}", raw_h, originals[k][0])
        close(f"same_S_{k}", s, originals[k][1])
    h_limit = 0. if phase == "ordinary" else contract["gates"]["candidate_H_entry_Ha"]
    close("saved_full_H", h, saved["H"], h_limit)
    close("historical_unshifted_full_roots_source_evolution", ci.state_energies, saved["energies"],
          contract["gates"]["candidate_all_roots_Ha"])
    replay_arrays = {}
    if phase == "ordinary":
        gate("saved_H_bytes", h.dtype == saved["H"].dtype and h.tobytes() == saved["H"].tobytes())
        replay_start = time.monotonic()
        replay_values, replay_vectors = [], []
        for _sector, indices in groups:
            values, vectors, _origin = ci_module._centered_dense_eigh(h[np.ix_(indices, indices)],
                                                                    subset_by_index=None, driver="evd")
            full_vectors = np.zeros((1046, len(values)))
            full_vectors[indices] = vectors
            replay_values.extend(values)
            replay_vectors.append(full_vectors)
        order = np.argsort(replay_values, kind="stable")
        replay_energies = np.asarray(replay_values)[order]
        replay_mixing = np.column_stack(replay_vectors)[:, order]
        gate("new_centered_epoch_all_roots_bitwise_replay", ci.state_energies.dtype == replay_energies.dtype
             and ci.state_energies.tobytes() == replay_energies.tobytes())
        close("new_centered_replay_original_H_residual", np.max(np.linalg.norm(h @ replay_mixing
              - replay_mixing * replay_energies, axis=0)), 0., 1e-8)
        close("new_centered_replay_complete_Gram", replay_mixing.T @ replay_mixing, np.eye(1046), 1e-10)
        close("new_centered_replay_same_origins", [call["origin_Ha"] for call in centered_calls[5:]],
              [call["origin_Ha"] for call in centered_calls[:5]])
        for sector, columns in blocks.items():
            csf_rows = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector]
            start = 0
            for stop in range(1, len(columns) + 1):
                if stop == len(columns) or abs(ci.state_energies[columns[stop]]
                                               - ci.state_energies[columns[stop - 1]]) > contract["gates"]["cluster_gap_Ha"]:
                    cluster = columns[start:stop]
                    a = ci.mixing_coefficients[np.ix_(csf_rows, cluster)]
                    b = replay_mixing[np.ix_(csf_rows, cluster)]
                    close("new_centered_replay_signed_cluster_" + str(cluster), a @ a.T, b @ b.T, 1e-8)
                    start = stop
        replay_arrays = {"replay_energies": replay_energies, "replay_mixing": replay_mixing}
        stages["ordinary_complete_savedH_centered_replay"] = time.monotonic() - replay_start
    write(output / "targets_metric.json", {"targets": target_rows,
          "residual_metric_sha256": contract["fixed"]["residual_metric_sha256"],
          "residual": float(np.linalg.norm(pg)), "indicator_Ha": float(.5 * abs(g @ pg)),
          "stationarity_passed": bool(np.linalg.norm(pg) <= 1e-7),
          "indicator_passed": bool(.5 * abs(g @ pg) <= 1e-8),
          "original_physical_failures_retained": True, "reported_EO_Ha": energy})
    start = time.monotonic()
    arrays = output / "arrays.npz"
    values = {"H": h, "energies": ci.state_energies, "mixing": ci.mixing_coefficients,
              "radial_gradient": np.array([radial_g[k] for k in functional.keys]),
              "raw_g": raw, "g": g, "pg": pg, "weights": weights, "x": gradient["x"],
              "keys": np.array(functional.keys), "P": saved["P"], "Q": saved["Q"]}
    values.update(replay_arrays)
    with arrays.open("wb") as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    with np.load(arrays, allow_pickle=False) as stored:
        for name, value in values.items():
            close("archive_exact_" + name, stored[name], value)
    stages["complete_checked_compressed_archive_fsync_readback"] = time.monotonic() - start
    return arrays, owner


def compare(execution):
    """Saved-array-only full-root/cluster-span and original-gradient comparison."""
    contract = admit(execution)
    import numpy as np
    with np.load(HERE / "actual/ordinary/arrays.npz", allow_pickle=False) as ar:
        ordinary = {k: ar[k].copy() for k in ar.files}
    with np.load(HERE / "actual/candidate/arrays.npz", allow_pickle=False) as ar:
        candidate = {k: ar[k].copy() for k in ar.files}
    checks = []
    def check(name, error, limit):
        passed = bool(np.isfinite(error) and error <= limit)
        checks.append({"name": name, "error": float(error), "limit": limit, "passed": passed})
        if not passed:
            write(HERE / "actual/comparison_failed.json", {"all_passed": False, "checks": checks})
            raise ValueError("Saved parity gate failed: " + name)
    for name, limit in (("H", 1e-9), ("energies", 1e-8), ("radial_gradient", 1e-9),
                        ("raw_g", 1e-9), ("g", 1e-9), ("pg", 1e-10), ("weights", 0.),
                        ("x", 0.), ("keys", 0.), ("P", 0.), ("Q", 0.)):
        check("whole_" + name, np.max(abs(candidate[name] - ordinary[name])), limit)
    # CSF row/global energy order is identical. Group complete CI roots using
    # both endpoints' declared1e-7 clusters, never fix individual phases inside
    # unresolved degeneracies. Unlike a full-sector span alone this binds roots.
    energies0, energies1 = ordinary["energies"], candidate["energies"]
    blocks = []
    with (HERE / "actual/ordinary/targets_metric.json").open() as handle:
        targets = json.load(handle)["targets"]
    # Full roots in ci_solve are grouped by increasing sector then energy; get
    # the exact sector columns saved by each worker's public CI block consumer.
    cols = json.loads((HERE / "actual/ordinary/sector_columns.json").read_text())
    for row in cols:
        columns = row["columns"]
        csf_rows = row["CSF_rows"]
        outside = sorted(set(range(1046)) - set(csf_rows))
        for label, values in (("ordinary", ordinary), ("candidate", candidate)):
            check(label + "_complete_sector_exact_zeros_" + str(row["sector"]),
                  np.max(abs(values["mixing"][np.ix_(outside, columns)]), initial=0.), 0.)
        start = 0
        for stop in range(1, len(columns) + 1):
            if (stop == len(columns) or
                    min(abs(energies0[columns[stop]] - energies0[columns[stop-1]]),
                        abs(energies1[columns[stop]] - energies1[columns[stop-1]])) > contract["gates"]["cluster_gap_Ha"]):
                cluster = columns[start:stop]
                c0, c1 = ordinary["mixing"][np.ix_(csf_rows, cluster)], candidate["mixing"][np.ix_(csf_rows, cluster)]
                error = np.max(abs(c0 @ c0.T - c1 @ c1.T))
                check("signed_cluster_" + str(cluster), error, contract["gates"]["signed_cluster_span"])
                blocks.append(cluster)
                start = stop
    reference_partition = json.loads((REPO / contract["inputs"]["common_shift_result"]["path"]).read_text())
    check("complete_unchanged_original_cluster_partition",
          blocks != [row["columns"] for row in reference_partition["shifted_complete_cluster_spans"]], 0.)
    density0 = json.loads((HERE / "actual/ordinary/signed_density.json").read_text())
    density1 = json.loads((HERE / "actual/candidate/signed_density.json").read_text())
    terms0 = {json.dumps(row[0]): row[1] for row in density0}
    terms1 = {json.dumps(row[0]): row[1] for row in density1}
    check("complete_same_signed_RDM_support", set(terms0) != set(terms1), 0.)
    check("complete_signed_RDM_coefficients", max(abs(terms0[k] - terms1[k]) for k in terms0), 1e-9)
    check("same_complete_sector_columns", json.loads((HERE / "actual/ordinary/sector_columns.json").read_text()) !=
          json.loads((HERE / "actual/candidate/sector_columns.json").read_text()), 0.)
    check("complete_same_physical_owner", json.loads((HERE / "actual/ordinary/owner.json").read_text()) !=
          json.loads((HERE / "actual/candidate/owner.json").read_text()), 0.)
    current = json.loads((HERE / "actual/candidate/targets_metric.json").read_text())["targets"]
    check("same5targetUIDbindings",
          [(r["label"], r["UID"], r["column"]) for r in targets] !=
          [(r["label"], r["UID"], r["column"]) for r in current], 0.)
    origins0 = json.loads((HERE / "actual/ordinary/centered_origins.json").read_text())
    origins1 = json.loads((HERE / "actual/candidate/centered_origins.json").read_text())
    check("same_per_sector_nearest_integer_centering", [(r["sector"], r["origin_Ha"]) for r in origins0] !=
          [(r["sector"], r["origin_Ha"]) for r in origins1], 0.)
    for phase in ("ordinary", "candidate"):
        actual = json.loads((HERE / "actual" / phase / "result.json").read_text())
        for name, expected in contract["chain_exact_calls"][phase].items():
            check(phase + "_calls_" + name, abs(actual["calls_started"].get(name, 0) - expected), 0.)
            check(phase + "_completed_" + name, abs(actual["calls_completed"].get(name, 0) - expected), 0.)
    write(HERE / "actual/comparison.json", {"all_passed": True, "checks": checks,
          "complete_root_clusters": blocks, "physical_stationarity_qualification": False,
          "speedup_certified": False, "GPU_called": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", required=True)
    parser.add_argument("--worker", choices=("anchors", "ordinary", "candidate"))
    args = parser.parse_args()
    execution_path = Path(args.execution).absolute()
    execution = json.loads(execution_path.read_text())
    if args.worker:
        worker(args.worker, execution)
        return
    contract = admit(execution)
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    write(output / "once_marker.json", {"root_execution_sha256": sha(execution_path),
          "driver_sha256": sha(__file__), "contract_sha256": sha(HERE / "contract.json"),
          "source_maps": {variant: contract[variant]["source_map_sha256"] for variant in ("ordinary", "candidate")},
          "no_retry": True, "original_routeFAIL_retained": True})
    begin = time.monotonic()
    for phase in ("anchors", "ordinary", "candidate"):
        remaining = 180 - (time.monotonic() - begin)
        if remaining <= 0:
            raise TimeoutError("Shared180s exhausted before " + phase)
        child = dict(execution, remaining_seconds=remaining)
        childpath = output / (phase + "_execution.json")
        write(childpath, child)
        env = os.environ.copy()
        env["NUMBA_CACHE_DIR"] = str(output / (phase + "_numba_cache"))
        cmd = [contract["resources"]["python"], str(HERE / "run.py"), "--execution", str(childpath), "--worker", phase]
        with (output / (phase + ".log")).open("wb") as log:
            try:
                result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=remaining + 2, check=False)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                returncode = -999
        write(output / (phase + "_process.json"), {"command": cmd, "exit_code": returncode,
              "elapsed_total_uncontrolled_s": time.monotonic() - begin, "log_sha256": sha(output / (phase + ".log"))})
        if returncode:
            write(output / "failure.json", {"failed_worker": phase, "exit_code": returncode,
                  "no_retry": True, "remaining_phases_unexecuted": True})
            raise RuntimeError("Closed failed worker: " + phase)
    resource.setrlimit(resource.RLIMIT_AS, (int(3.95 * 2**30), int(3.95 * 2**30)))
    os.sched_setaffinity(0, {30})
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Comparison shared180s cap")))
    signal.alarm(max(1, int(180 - (time.monotonic() - begin))))
    compare(execution)
    write(output / "result.json", {"all_equivalence_gates_passed": True,
          "H": 2, "CI": 2, "gradient": 2, "optimizer": 0, "GPU": 0,
          "scipy_eigh": 17, "Gauss_setup_eigvalsh": 7,
          "original_route_qualification002": "FAIL retained; this is a new centered source epoch",
          "elapsed_uncontrolled_s": time.monotonic() - begin,
          "comparison_sha256": sha(output / "comparison.json"),
          "root_execution_sha256": sha(execution_path), "driver_sha256": sha(__file__),
          "contract_sha256": sha(HERE / "contract.json"),
          "no_speedup_claim": True, "original_physical_failures_retained": True})


if __name__ == "__main__":
    main()
