"""One separately sealed bounded native31 f043 reported-EOL optimization.

Preparation imports only stdlib. Frozen f043 is selected before all Atomx
imports; unresolved current-source regressions are not reinterpreted here.
Full signed-density checks and exact stored-H acceptance are checkpoint
qualification guards, never energy or gradient replacements.
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
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())


class QualificationStop(RuntimeError):
    """Preserve the first unqualified candidate and previous valid point."""


class ResourceStop(RuntimeError):
    """Stop before forwarding an overbudget producer."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "item"):
        return plain(value.item())
    return value


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def integer_resource_census(contract):
    """Reject inconsistent complete-support budgets before numerical work."""
    resources = contract["resources"]
    dimensions = [block["CSFs"] for block in contract["fixed"]["complete_target_blocks"]]
    if dimensions != [219, 223, 367, 370, 416]:
        raise ValueError("Changed complete1595 target sector dimensions")
    points = resources["max_new_accepted"] + 1
    pairs_per_point = sum(n * (n + 1) // 2 for n in dimensions)
    pairs = points * pairs_per_point
    quotients = points * len(dimensions)
    if (sum(dimensions) != 1595 or points != 17
            or resources["max_points"] != points
            or resources["exact_guard_points"] != points
            or resources["full_sector_support_pair_upper_per_point"] != pairs_per_point
            or resources["full17points_support_pair_upper"] != pairs
            or pairs > resources["exact_guard_examined_pairs"]
            or resources["exact_guard_examined_pairs"] != 5_000_000
            or quotients != resources["exact_guard_target_quotients"]
            or quotients != 85
            or resources["calls"]["gradient_from_weights"] != 3 * points
            or resources["calls"]["unfiltered_target_RDM"] != quotients):
        raise ValueError("Inconsistent prospective complete-support resource census")
    return dict(points=points, target_quotients=quotients,
                complete_sector_dimensions=dimensions,
                support_pair_upper_per_point=pairs_per_point,
                support_pair_upper_all_points=pairs,
                examined_pair_cap=resources["exact_guard_examined_pairs"])


def metadata(contract, *, verify_arrays=False):
    roles = contract["input_roles"]
    for row in roles.values():
        if row["path"].endswith(".npz") and not verify_arrays:
            continue
        if sha(REPO / row["path"]) != row["sha256"]:
            raise ValueError("Changed source-bound input: " + row["path"])
    identity = load(REPO / roles["source_identity"]["path"])
    if identity["source_map_sha256"] != contract["source"]["map_sha256"]:
        raise ValueError("Wrong immutable f043 source epoch")
    source = REPO / contract["source"]["archive"]
    for name, digest in identity["all_atomx_python_sha256"].items():
        if sha(source / name.removeprefix("src/")) != digest:
            raise ValueError("Immutable f043 module changed: " + name)
    g, fd = (load(REPO / roles[r]["path"]) for r in ("gradient_result", "FD_result"))
    for result in (g, fd):
        if (result["all_declared_controls_passed"] is not True
                or result["source_map_sha256"] != contract["source"]["map_sha256"]
                or result["new_metric_sha256"] != contract["fixed"]["new_metric_sha256"]
                or result["new_geometry_sha256"] != contract["fixed"]["new_geometry_sha256"]
                or result["expanded_MCDHF_optimized"] is not False
                or not all(row["passed"] for row in result["checks"])):
            raise ValueError("Original own31 qualification incomplete")
    if (len(g["checks"]) != 438 or len(fd["checks"]) != 730
            or fd["single_direction_finite_measurement_only"] is not True
            or fd["full31_space_certificate"] is not False):
        raise ValueError("Wrong original qualification measurement scope")
    for role in ("gradient_receipt", "FD_receipt"):
        receipt = load(REPO / roles[role]["path"])
        if receipt["status"] != "foreground_closed" or receipt["exit_code"] != 0:
            raise ValueError("Own qualification process not successfully closed")
    completion = load(REPO / roles["FD_completion"]["path"])
    if (completion["all_completion_metadata_checks_passed"] is not True
            or completion["ordinary_both_h_passed"] is not True
            or completion["independent_centered_both_h_both_references_passed"] is not True):
        raise ValueError("Own combined finite response not closed")
    return source, identity


def execute(contract, binding_path):
    binding = load(binding_path)
    caps, gates, fixed = (contract[k] for k in ("resources", "gates", "fixed"))
    expected = {"execution_authorized_by_root": True, "driver_sha256": sha(__file__),
                "contract_sha256": sha(HERE / "contract.json"),
                "preparation_sha256": sha(HERE / "preparation.json"),
                "source_map_sha256": contract["source"]["map_sha256"],
                "CPU": 31, "AS_GiB": 3.95, "inner_seconds": 600,
                "outer_seconds": 615, "max_H": 65, "max_CI": 65,
                "max_gradient_from_weights": 51, "max_optimizer": 1,
                "max_new_accepted": 16}
    if any(binding.get(k) != v for k, v in expected.items()):
        raise ValueError("Missing or mismatched exact root execution binding")
    census = integer_resource_census(contract)
    prep = load(HERE / "preparation.json")
    if (prep["all_source_only_checks_passed"] is not True
            or prep["driver_sha256"] != expected["driver_sha256"]
            or prep["contract_sha256"] != expected["contract_sha256"]):
        raise ValueError("Changed prepared executor")
    if any(n == "numpy" or n.startswith("numpy.") or n == "atomx"
           or n.startswith("atomx.") for n in sys.modules):
        raise ValueError("Fresh source-isolated process required")
    if any(os.environ.get(k) != "1" for k in caps["thread_environment"]):
        raise ValueError("All declared native/BLAS threads must be1")
    available = sorted(os.sched_getaffinity(0))
    load_average = os.getloadavg()
    if 31 not in available or load_average[0] > caps["maximum_system_load1"]:
        raise ValueError("Host admission rejected before producers")
    source, identity = metadata(contract, verify_arrays=True)
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    (output / "checkpoints").mkdir()
    cache = output / "numba_cache"
    cache.mkdir()
    os.environ["NUMBA_CACHE_DIR"] = str(cache)
    os.sched_setaffinity(0, {31})
    maximum = int(caps["AS_GiB"] * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (maximum, maximum))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(ResourceStop("600s cap")))
    signal.alarm(caps["inner_seconds"])
    started = time.monotonic()
    sys.path.insert(0, str(source))
    attempted = dict.fromkeys(caps["calls"], 0)
    completed = dict(attempted)
    checks, trajectory, archives, operator_events = [], [], [], []
    previous, diagnostics, error = None, None, None
    status, captured, raw_operators, loaded = "not_started", {}, {}, {}
    guard, exact_budget = None, None
    admission = dict(original_accounting_CPUs=available, affinity=[31],
        load_average=load_average, all_threads=1, inner_seconds=600,
        outer_seconds=615, AS_GiB=3.95, timing_claim_valid=False,
        beforeloop_integer_resource_census=census)

    def progress(phase):
        write(output / "progress.json", dict(phase=phase, attempted=attempted,
            completed=completed, checks_count=len(checks),
            last_check=checks[-1] if checks else None, qualified_points=len(trajectory),
            last_qualified_checkpoint=previous["manifest"] if previous else None,
            exact_budget=exact_budget.ledger() if exact_budget else None,
            elapsed_uncontrolled_s=time.monotonic() - started))

    def source_guard():
        for name, module in tuple(sys.modules.items()):
            if not (name == "atomx" or name.startswith("atomx.")):
                continue
            filename = getattr(module, "__file__", None)
            if not filename:
                continue
            path = Path(filename).resolve()
            if not path.is_relative_to(source.resolve()):
                raise QualificationStop("Mutable or external Atomx import: " + name)
            if name not in loaded:
                key = "src/" + str(path.relative_to(source.resolve()))
                if identity["all_atomx_python_sha256"].get(key) != sha(path):
                    raise QualificationStop("Unbound imported source: " + name)
                loaded[name] = dict(path=str(path), sha256=sha(path))

    try:
        import numpy as np
        from scipy import linalg

        import atomx.methods.csf.variational as variational
        import atomx.numerics.bspline.dirac_assembly as dirac
        import atomx.numerics.bspline.orbital_optimization as optimization
        import atomx.physics.csf.matrix as matrix
        import atomx.physics.csf.tracking as tracking
        from atomx.angular.csf_hamiltonian import csf_signature
        from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
        from atomx.constants import HARTREE_CM1
        from atomx.methods.ci import CIResult
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.physics.csf.state_average import CIStateAverage
        from atomx.potentials.nuclear import make_nuclear_potential

        source_guard()
        helper = REPO / contract["input_roles"]["exact_guard"]["path"]
        spec = importlib.util.spec_from_file_location("native31_exact_f043_guard", helper)
        guard = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = guard
        spec.loader.exec_module(guard)
        exact_budget = guard.ExactGuardBudget()

        def check(name, error_value, limit=0.):
            row = dict(name=name, error=float(error_value), limit=limit,
                       passed=bool(np.isfinite(error_value) and error_value <= limit))
            checks.append(row)
            if not row["passed"]:
                raise QualificationStop("Original scientific gate failed: " + name)

        def save(path, **arrays):
            temporary = path.with_suffix(path.suffix + ".part")
            if path.exists():
                raise FileExistsError("Immutable actual archive already exists")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(path)
            digest = sha(path)
            archives.append(dict(path=str(path.relative_to(REPO)), sha256=digest))
            return digest

        def counted(name, function, capture_operator=False):
            def call(*args, **kwargs):
                if attempted[name] >= caps["calls"][name]:
                    progress("resource_cap_" + name)
                    raise ResourceStop("Before producer cap: " + name)
                attempted[name] += 1
                if name != "gradient_pair_source":
                    progress("before_" + name)
                try:
                    value = function(*args, **kwargs)
                except (ValueError, FloatingPointError) as failure:
                    if name in {"H", "CI", "gradient_from_weights", "tracker_resolve",
                                "unfiltered_target_RDM", "retained_RDM", "gradient_pair_source"}:
                        raise QualificationStop("Physical producer failed: " + name) from failure
                    raise
                completed[name] += 1
                if name == "gradient_from_weights":
                    progress("after_" + name)
                if name == "H":
                    captured["last_complete_H_build"] = value
                if capture_operator:
                    kappa, h, s = int(args[1]), value[0], value[1]
                    if kappa in raw_operators:
                        check("raw_h_bytes_" + str(len(operator_events)),
                              h.tobytes() != raw_operators[kappa][0].tobytes())
                        check("raw_S_bytes_" + str(len(operator_events)),
                              s.tobytes() != raw_operators[kappa][1].tobytes())
                    else:
                        raw_operators[kappa] = h.copy(), s.copy()
                    operator_events.append(dict(producer=name, kappa=kappa,
                        H_sha256=hashlib.sha256(h.tobytes()).hexdigest(),
                        S_sha256=hashlib.sha256(s.tobytes()).hexdigest()))
                return value
            return call

        original_dirac = dirac.build_dirac_matrices
        variational.build_dirac_matrices = counted("initial_hS", original_dirac, True)
        matrix.build_dirac_matrices = counted("CI_hS", original_dirac, True)
        matrix.build_ci_matrix = counted("H", matrix.build_ci_matrix)
        dirac.solve_dirac = counted("validation_Dirac", dirac.solve_dirac)
        variational.CSFOrbitalFunctional.gradient_from_weights = counted(
            "gradient_from_weights", variational.CSFOrbitalFunctional.gradient_from_weights)
        tracking.CIBasisTracking.resolve = counted("tracker_resolve", tracking.CIBasisTracking.resolve)
        tracking.ci_subspace_overlaps = counted("tracker_overlap", tracking.ci_subspace_overlaps)
        np.linalg.svd = counted("tracking_SVD", np.linalg.svd)
        np.linalg.eigvalsh = counted("geometry_SPD", np.linalg.eigvalsh)
        linalg.eigh = counted("SciPy_eigh", linalg.eigh)
        optimization._polar_retraction = counted("polar", optimization._polar_retraction)
        np.linalg.eigh = counted("polar_numpy_eigh", np.linalg.eigh)
        optimization.optimize_orbitals = counted("optimizer", optimization.optimize_orbitals)

        def path(role):
            return REPO / contract["input_roles"][role]["path"]

        def array_archive(role):
            with np.load(path(role), allow_pickle=False) as values:
                return {k: values[k].copy() for k in values.files}

        saved, saved_g, saved_CI = (array_archive(r) for r in
            ("recovered_coordinates_and_waves", "qualified_full31_gradient", "recovered_full_CI"))
        radial, original_hS = array_archive("geometry_workspace"), array_archive("screened_geometry")
        data = {k[6:]: v.item() if v.ndim == 0 else v.copy()
                for k, v in radial.items() if k.startswith("basis_")}
        models = load(path("models"))
        sectors = {tuple(s) for s in models["selected_sectors"]}
        full = [c for row in models["models"]["complete5f"]["configs"]
                for c in enumerate_jj_csfs(row["config"])]
        csfs = [c for c in full if (c.total_J2, c.parity) in sectors]
        basis = load(path("canonical_basis"))["complete5f"]["retained_signatures"]
        signatures = []
        for c in csfs:
            s = csf_signature(c)
            signatures.append(json.dumps(plain((s.subshells, s.coupling, c.total_J2, c.parity)),
                                         separators=(",", ":")))
        check("parent3789", len(full) != 3789)
        check("complete1595", len(csfs) != 1595)
        check("charge69", {c.n_electrons for c in full} != {69})
        check("complete_five_blocks", Counter((c.total_J2, c.parity) for c in csfs)
              != {(r["twoJ"], r["parity"]): r["CSFs"] for r in fixed["complete_target_blocks"]})
        check("signed_CSF_order", signatures != basis)
        keys31 = tuple(tuple(map(int, k)) for k in saved["keys"])
        check("complete31keys", keys31 != tuple(sorted(get_shared_orbital_basis(csfs))))
        reference = [c for row in load(path("reference_models"))["models"]["baseline"]["configs"]
                     for c in enumerate_jj_csfs(row["config"])]
        check("original531reference", len(reference) != 531)
        reference_ci = array_archive("tracking_ci")
        targets = load(path("tracking_binding"))["targets"]
        expected_UIDs = {tuple(uid): label for label, uid in fixed["original_UIDs"]}
        check("original_five_UIDs", {(t["twoJ"], t["parity"], t["models"]["DC"]["local_root"])
              for t in targets} != set(expected_UIDs))
        average = CIStateAverage(tuple((t["twoJ"], t["parity"],
            t["models"]["DC"]["local_root"], 1.) for t in targets))
        reference_waves = {tuple(map(int, k)): (p.copy(), q.copy()) for k, p, q in
                          zip(radial["orbital_keys"], radial["P"], radial["Q"], strict=True)}
        tracker = tracking.CIBasisTracking(average, reference,
            CIResult(reference_ci["energies_Ha"], reference_ci["mixing"]), reference_waves,
            {k: original_hS[f"s_{k}"] for k in fixed["kappas"]},
            max_determinant_pairs=caps["maximum_signed_overlap_pairs"])
        nuclear = make_nuclear_potential(70, "fermi", A=175)
        functional = variational.CSFOrbitalFunctional(csfs, data, 70, nuclear,
            state_average=average, level_tracking=tracker, retain_hamiltonian=True, max_determinants=2000)
        functional.radial_density.contract = counted("retained_RDM", functional.radial_density.contract)
        functional.radial_density.contract_bilinear = counted(
            "unfiltered_target_RDM", functional.radial_density.contract_bilinear)
        functional._pair_potential = counted("gradient_pair_source", functional._pair_potential)
        check("functional31order", functional.keys != keys31)
        angular_keys = {k for structure, _, _ in functional.radial_density.blocks for k in structure.keys}
        check("full19589angularterms", len(angular_keys) != 19589)
        for kappa in fixed["kappas"]:
            h, s = raw_operators[kappa]
            check("logical_raw_h_" + str(kappa), not np.array_equal(
                functional.one_body[kappa], np.tril(h) + np.tril(h, -1).T))
            check("saved_S_" + str(kappa), not np.array_equal(s, original_hS[f"s_{kappa}"]))
        layout = tuple((tuple(map(tuple, r["keys"])), saved[f"projector_{i}"],
                        tuple(r["shape"]), slice(*r["slice"]))
                       for i, r in enumerate(load(path("new_layout"))))
        objective = load(path("new_objective"))
        check("objective_payload_SHA", hashlib.sha256(json.dumps(objective["payload"],
              sort_keys=True, separators=(",", ":")).encode()).hexdigest() != objective["identity"])
        check("f043_objective_source", objective["payload"]["source_map_sha256"]
              != contract["source"]["map_sha256"])
        check("objective_full_signed_CSF_order", objective["payload"]["full_signed_CSF_order_sha256"]
              != hashlib.sha256(json.dumps(basis, separators=(",", ":")).encode()).hexdigest())
        initial = optimization.OrbitalOptimizationState.from_arrays(functional, layout=layout,
            coordinates=saved["x"], inverse_diagonal=saved["inverse_diagonal"],
            inverse_blocks=tuple(tuple(v) for v in (saved[f"inverse_blocks_{i}"] for i in range(7))),
            projector_source="nuclear", objective_identity=objective["identity"], occupation_floor=.05)
        for field, value in (("new_metric_sha256", initial.residual_metric_sha256),
                              ("new_geometry_sha256", initial.geometry_sha256),
                              ("new_state_sha256", initial.state_sha256)):
            check("unchanged_" + field, value != fixed[field])
        for name in ("x", "inverse_diagonal", *[f"projector_{i}" for i in range(7)],
                     *[f"inverse_blocks_{i}" for i in range(7)]):
            check("same_initial_saved_bytes_" + name, saved[name].tobytes() != saved_g[name].tobytes())

        def unpack(state):
            waves = {}
            for keys, projector, shape, where in state.layout:
                columns = projector @ state.coordinates[where].reshape(shape)
                for j, key in enumerate(keys):
                    waves[key] = columns[:123, j].copy(), columns[123:, j].copy()
            return waves

        waves = unpack(initial)
        for i, key in enumerate(functional.keys):
            for side, part in enumerate(("P", "Q")):
                check("same_literal_recovered_point_" + str(key) + part,
                      waves[key][side].tobytes() != saved[part][i].tobytes()
                      or waves[key][side].tobytes() != saved_CI[part][i].tobytes())

        def bind(weights, tracked):
            check("original_equal5weights", not (np.count_nonzero(weights) == 5
                  and np.array_equal(weights[weights > 0], np.repeat(.2, 5))))
            matched = {}
            for row in tracked:
                if len(row["reference_levels"]) != 1 or len(row["selected_levels"]) != 1:
                    raise QualificationStop("Original rank1 physical targets required")
                uid, selected = tuple(row["reference_levels"][0]), tuple(row["selected_levels"][0])
                if uid in matched or uid not in expected_UIDs or uid[:2] != selected[:2]:
                    raise QualificationStop("Changed or duplicate physical UID")
                matched[uid] = row
            check("complete_original_UID_binding", set(matched) != set(expected_UIDs))
            return matched

        original_ci = variational.ci_solve
        last_ci_path = None

        def ci_solve(*args, **kwargs):
            nonlocal last_ci_path
            before_H = completed["H"]
            try:
                ci = counted("CI", original_ci)(*args, **kwargs)
            except BaseException:
                if completed["H"] > before_H:
                    save(output / f"failed_CI_complete_H_{attempted['CI']:04d}.npz",
                         H=captured["last_complete_H_build"], keys=np.array(functional.keys),
                         P=np.array([args[1][k][0] for k in functional.keys]),
                         Q=np.array([args[1][k][1] for k in functional.keys]))
                raise
            h = ci.hamiltonian_for(csfs, args[1], functional.keys, data, 70, 123,
                                   nuclear_potential=nuclear, max_determinants=2000)
            prefix = output / f"trial_CI_{completed['CI']:04d}"
            last_ci_path = prefix.with_suffix(".npz")
            digest = save(last_ci_path, H=h, energies=ci.state_energies,
                mixing=ci.mixing_coefficients, keys=np.array(functional.keys),
                P=np.array([args[1][k][0] for k in functional.keys]),
                Q=np.array([args[1][k][1] for k in functional.keys]))
            write(prefix.with_suffix(".json"), dict(qualified=False, array_sha256=digest,
                owner=ci.hamiltonian_snapshot.identity_payload(), source_map=contract["source"]["map_sha256"]))
            check("whole1595_H_symmetric", not np.array_equal(h, h.T))
            check("whole1595_CI_shape", ci.mixing_coefficients.shape != (1595, 1595)
                  or ci.state_energies.shape != (1595,))
            check("whole1595_CI_Gram", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients
                  - np.eye(1595))), gates["CI_Gram"])
            check("whole1595_CI_residual", np.max(np.linalg.norm(h @ ci.mixing_coefficients
                  - ci.mixing_coefficients * ci.state_energies, axis=0)), gates["CI_residual_Ha"])
            for sector in sectors:
                inside = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector]
                outside = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) != sector]
                check("cross_sector_H_" + str(sector), np.max(abs(h[np.ix_(inside, outside)])),
                      gates["cross_sector_H_Ha"])
            weights, tracked = tracker.resolve(csfs, ci, args[1], functional.metric)
            bind(weights, tracked)
            write(prefix.with_suffix(".json"), dict(qualified=True, array_sha256=digest,
                owner=ci.hamiltonian_snapshot.identity_payload(), tracking=tracked, weights=weights,
                source_map=contract["source"]["map_sha256"], original_reference_not_rebased=True))
            source_guard()
            return ci

        variational.ci_solve = ci_solve
        original_evaluate = functional.evaluate

        def evaluate(orbitals, *, gradient=True, ci_result=None, **kwargs):
            if gradient and ci_result is not None:
                write(output / "accepted_candidate_before_gradient.json", dict(qualified=False,
                    same_CI_archive=str(last_ci_path.relative_to(REPO)) if last_ci_path else None,
                    previous_qualified=previous["manifest"] if previous else None,
                    owner=ci_result.hamiltonian_snapshot.identity_payload()))
            value = original_evaluate(orbitals, gradient=gradient, ci_result=ci_result, **kwargs)
            if gradient:
                captured.update(ci=value[2], radial={k: v.copy() for k, v in value[1].items()},
                                reported_energy=float(value[0]))
            return value

        functional.evaluate = evaluate
        nominal = {r["label"]: r["NIST_nominal_cm1"] for r in load(path("nominal_NIST"))["DC_rows"]}

        def vectors(state, radial):
            raw = np.zeros(3689, dtype=radial.dtype)
            lookup = dict(zip(functional.keys, radial, strict=True))
            for keys, projector, _, where in state.layout:
                raw[where] = (projector.astype(radial.dtype).T
                    @ np.column_stack([lookup[k] for k in keys])).ravel()

            def tangent(v):
                result = v.copy()
                for _, _, shape, where in state.layout:
                    y, a = state.coordinates[where].reshape(shape).astype(v.dtype), v[where].reshape(shape)
                    multiplier = y.T @ a
                    result[where] = (a - .5 * y @ (multiplier + multiplier.T)).ravel()
                return result

            g = tangent(raw)
            inverse = np.zeros_like(g)
            for (_, _, shape, where), blocks in zip(state.layout, state.inverse_blocks, strict=True):
                a = g[where].reshape(shape)
                inverse[where] = np.column_stack([b.astype(g.dtype) @ a[:, j]
                    for j, b in enumerate(blocks)]).ravel()
            pg = tangent(inverse)
            absolute = max(np.max(abs(state.coordinates[where].reshape(shape).astype(g.dtype).T
                @ g[where].reshape(shape) + g[where].reshape(shape).T
                @ state.coordinates[where].reshape(shape).astype(g.dtype)))
                for _, _, shape, where in state.layout)
            check("full_tangent_scaled_identity", absolute / max(1., float(np.linalg.norm(raw))),
                  gates["scaled_tangent_identity"])
            return raw, g, pg, dict(raw_norm=float(np.linalg.norm(raw)),
                tangent_norm=float(np.linalg.norm(g)), residual=float(np.linalg.norm(pg)),
                indicator_Ha=.5 * abs(float(g @ pg)), absolute_tangent_defect=float(absolute),
                no_energy_or_spectrum_error_bound=True)

        def checkpoint(iteration, state, orbitals, ci, diag):
            nonlocal previous
            if iteration > 16 or len(trajectory) >= 17:
                raise ResourceStop("Checkpoint budget")
            prefix = output / "checkpoints" / f"accepted_{iteration:04d}"
            h = ci.hamiltonian_for(csfs, orbitals, functional.keys, data, 70, 123,
                                   nuclear_potential=nuclear, max_determinants=2000)
            if captured.get("ci") is not ci:
                raise QualificationStop("Checkpoint radial gradient does not own this CI")
            check("immutable3689_metric", state.residual_metric_sha256 != fixed["new_metric_sha256"])
            check("immutable_geometry", state.geometry_sha256 != fixed["new_geometry_sha256"])
            retained = np.array([captured["radial"][k] for k in functional.keys])
            check("complete31x246_gradient", retained.shape != (31, 246))
            raw, g, pg, retained_metrics = vectors(state, retained)
            arrays = dict(x=state.coordinates, H=h, energies=ci.state_energies,
                mixing=ci.mixing_coefficients, keys=np.array(functional.keys),
                P=np.array([orbitals[k][0] for k in functional.keys]),
                Q=np.array([orbitals[k][1] for k in functional.keys]),
                retained_radial=retained, raw_coordinate_gradient=raw,
                tangent_gradient=g, preconditioned_gradient=pg,
                inverse_diagonal=state.inverse_diagonal,
                **{f"projector_{i}": r[1] for i, r in enumerate(state.layout)},
                **{f"inverse_blocks_{i}": np.stack(b) for i, b in enumerate(state.inverse_blocks)})
            forensic = prefix.with_name(prefix.name + "_before_fullsigned.npz")
            forensic_sha = save(forensic, **arrays)
            manifest = dict(iteration=iteration, qualified=False, forensic_array_sha256=forensic_sha,
                previous_qualified=previous["manifest"] if previous else None,
                diagnostics=diag, owner=ci.hamiltonian_snapshot.identity_payload(),
                metric_sha256=state.residual_metric_sha256, geometry_sha256=state.geometry_sha256,
                objective_identity=state.objective_identity, source_map_sha256=contract["source"]["map_sha256"])
            write(prefix.with_suffix(".json"), manifest)
            check("actual_reported_EO_samefloat", captured["reported_energy"].hex() != float(diag["energy_ha"]).hex())
            check("original_residual_reconstruction", abs(retained_metrics["residual"] - diag["orbital_residual"]), 1e-10)
            check("original_indicator_reconstruction", abs(retained_metrics["indicator_Ha"] - diag["estimated_energy_error"]), 1e-10)
            constraints = {}
            for keys, projector, shape, where in state.layout:
                kappa, y = keys[0][1], state.coordinates[where].reshape(shape)
                c = np.column_stack([np.concatenate(orbitals[k]) for k in keys])
                s = functional.metric[kappa]
                constraints[str(kappa)] = dict(YGram=y.T @ y, actualSGram=c.T @ s @ c)
                check("coordinate_Gram_" + str(kappa), np.max(abs(y.T @ y - np.eye(len(keys)))), gates["radial_Gram"])
                check("physical_Gram_" + str(kappa), np.max(abs(c.T @ s @ c - np.eye(len(keys)))), gates["radial_Gram"])
                for j, key in enumerate(keys):
                    outside = c[:, j] - projector @ (projector.T @ s @ c[:, j])
                    check("no_pair_" + str(key), abs(float(outside @ s @ outside)), gates["P_leak_S_squared"])
            weights, tracked = functional.last_state_weights, functional.last_tracking_diagnostics
            matched = bind(weights, tracked)
            _, blocks = ci_level_blocks(csfs, ci, require_full=True)
            columns, ids, rows = [], [], []
            for label, uid_list in fixed["original_UIDs"]:
                uid = tuple(uid_list)
                j, p, rank = matched[uid]["selected_levels"][0]
                column = int(blocks[j, p][rank])
                outside = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) != (j, p)]
                check("exact_complete_sector_support_" + label, np.any(ci.mixing_coefficients[outside, column] != 0))
                columns.append(column)
                ids.append(json.dumps([label, list(uid)], separators=(",", ":")))
                rows.append(dict(label=label, UID=uid, selected_level=(j, p, rank),
                    column=column, weight=float(weights[column]), energy_Ha=float(ci.state_energies[column])))
            if iteration == 0:
                for name in ("H", "energies", "mixing", "P", "Q"):
                    check("own_f043_baseline_bitwise_" + name, arrays[name].tobytes() != saved_CI[name].tobytes())
                check("own_f043_baseline_owner", plain(ci.hamiltonian_snapshot.identity_payload()) != load(path("recovered_owner")))
                check("own_f043_baseline_tracking", plain(dict(weights=weights, tracked=tracked)) != load(path("recovered_tracking")))
                check("own_f043_baseline_retained_radialg", np.max(abs(retained - saved_g["retained_radial"])), 1e-9)
                check("own_f043_baseline_raw_coordinate_g", np.max(abs(raw - saved_g["retained_raw_coordinate"])), 1e-9)
                check("own_f043_baseline_pg", np.max(abs(pg - saved_g["retained_preconditioned_tangent"])), 1e-10)
                check("point0_no_line_search", diag["accepted_line_search"] is not None)
            contributions, tables = {}, []
            for row, column in zip(rows, columns, strict=True):
                column_vector = ci.mixing_coefficients[:, column]
                norm = sum((Fraction.from_float(float(v))**2 for v in column_vector), Fraction(0))
                if norm <= 0:
                    raise QualificationStop("Nonpositive exact CI norm")
                table = functional.radial_density.contract_bilinear(column_vector, column_vector)
                for key, value in table.items():
                    contributions.setdefault(key, []).append(Fraction.from_float(float(weights[column]))
                        * Fraction.from_float(float(value)) / norm)
                tables.append(dict(label=row["label"], UID=row["UID"], column=column,
                    exact_norm=[str(norm.numerator), str(norm.denominator)],
                    unfiltered=[[k, v] for k, v in table.items()]))

            def represented(value):
                result = float(value)
                if not math.isfinite(result) or (value != 0 and result == 0):
                    raise QualificationStop("Unrepresentable nonzero signed coefficient")
                return result

            full = {k: represented(v) for k, values in contributions.items()
                    if (v := sum(values, Fraction(0))) != 0}
            retained_weights = dict(functional.last_weights)
            ordered = list(full) + [k for k in retained_weights if k not in full]
            delta = {}
            for key in ordered:
                value = Fraction.from_float(full.get(key, 0.)) - Fraction.from_float(retained_weights.get(key, 0.))
                if value != 0:
                    delta[key] = represented(value)
            write(prefix.with_name(prefix.name + "_signed_tables.json"), dict(targets=tables,
                retained=[[k, v] for k, v in retained_weights.items()],
                full_normalized=[[k, v] for k, v in full.items()],
                full_minus_retained=[[k, v] for k, v in delta.items()], exact_zero_only=True,
                delta_order="full original order then retained-only original order",
                no_gradient_or_energy_replacement=True))
            delta_g = functional.gradient_from_weights(orbitals, delta, record_diagnostics=False)
            full_g = functional.gradient_from_weights(orbitals, full, record_diagnostics=False)
            delta_array = np.array([delta_g[k] for k in functional.keys])
            full_array = np.array([full_g[k] for k in functional.keys])
            check("whole_signed_linearity", np.max(abs(retained + delta_array - full_array)), 1e-9)
            plus = vectors(state, retained + delta_array)
            full_vectors = vectors(state, full_array)
            check("whole_tangent_linearity", np.max(abs(plus[1] - full_vectors[1])), 1e-10)
            check("whole_pg_linearity", np.max(abs(plus[2] - full_vectors[2])), 1e-10)
            wide = np.longdouble
            corrected = {k: full_g[k].astype(wide) for k in functional.keys}
            radial_vectors = {k: np.concatenate(orbitals[k]).astype(wide) for k in functional.keys}
            for term, coefficient in full.items():
                if term[0] != "h":
                    continue
                _, na, kappa, nb = term
                a, b = (na, kappa), (nb, kappa)
                raw_h, s = (v.astype(wide) for v in raw_operators[kappa])
                logical = functional.one_body[kappa].astype(wide)
                va, vb = radial_vectors[a], radial_vectors[b]
                sb, hb, hta = s @ vb, raw_h @ vb, raw_h.T @ va
                norm, numerator = vb @ sb, va @ hb
                if not np.isfinite(norm) or norm <= 0:
                    raise QualificationStop("Nonpositive radial ket norm")
                corrected[a] += coefficient * (hb / norm - logical @ vb)
                corrected[b] += coefficient * (hta / norm - logical @ va - 2 * numerator * sb / norm**2)
            corrected_array = np.array([corrected[k] for k in functional.keys])
            corrected_vectors = vectors(state, corrected_array)
            for name, vector in (("full_normalized", full_vectors), ("raw_h_quotient_corrected", corrected_vectors)):
                check(name + "_whole_tangent_difference", np.max(abs(vector[1] - g)), 1e-9)
                check(name + "_whole_pg_difference", np.max(abs(vector[2] - pg)), 1e-10)
                if iteration == 0:
                    for suffix, value in zip(("raw_coordinate", "tangent", "preconditioned_tangent"), vector[:3], strict=True):
                        check("same_own_baseline_" + name + suffix,
                              np.max(abs(value - saved_g[name + "_" + suffix])), 1e-9 if suffix != "preconditioned_tangent" else 1e-10)
            arrays.update(full_minus_retained_radial=delta_array, full_normalized_radial=full_array,
                raw_h_quotient_corrected_radial=corrected_array,
                full_normalized_raw_coordinate=full_vectors[0], full_normalized_tangent=full_vectors[1],
                full_normalized_pg=full_vectors[2], raw_h_corrected_raw_coordinate=corrected_vectors[0],
                raw_h_corrected_tangent=corrected_vectors[1], raw_h_corrected_pg=corrected_vectors[2])
            digest = save(prefix.with_suffix(".npz"), **arrays)
            cert = guard.exact_reported_point(h, ci.mixing_coefficients[:, columns],
                ci.state_energies[columns], weights[columns], float(diag["energy_ha"]),
                target_ids=tuple(ids), budget=exact_budget)
            decision = None if iteration == 0 else guard.reported_acceptance_decision(
                previous["certificate"], cert, diag["accepted_line_search"])
            manifest.update(array_sha256=digest, exact_point=cert.ledger(), exact_acceptance=decision,
                exact_budget=exact_budget.ledger(), fullsigned_metrics=dict(retained=retained_metrics,
                    full_normalized=full_vectors[3], raw_h_quotient_corrected=corrected_vectors[3]),
                full_same_kappa_constraints=constraints, HF_partial_not_finite_response=True)
            write(prefix.with_suffix(".json"), manifest)
            if decision is not None and decision["qualified"] is not True:
                raise QualificationStop("Unresolved finite storedH acceptance margin")
            by_label = {r["label"]: r for r in rows}
            lower = by_label["6s_1/2"]
            check("lower_physical_UID", tuple(lower["UID"]) != (1, 1, 0))
            for row in rows:
                row["excitation_cm1"] = (row["energy_Ha"] - lower["energy_Ha"]) * HARTREE_CM1
                row["NIST_nominal_cm1"] = nominal[row["label"]]
                row["nominal_gap_cm1"] = row["excitation_cm1"] - nominal[row["label"]]
                if previous:
                    old = previous["rows"]
                    du = row["energy_Ha"] - old[row["label"]]["energy_Ha"]
                    dl = lower["energy_Ha"] - old["6s_1/2"]["energy_Ha"]
                    row.update(delta_upper_Ha=du, delta_lower_Ha=dl,
                               transition_shift_cm1=(du - dl) * HARTREE_CM1)
            multiplets = []
            for label, a, b, w in (("5d", "5d_3/2", "5d_5/2", (4, 6)), ("6p", "6p_1/2", "6p_3/2", (2, 4))):
                multiplets.append(dict(label=label, centroid_cm1=(w[0] * by_label[a]["excitation_cm1"]
                    + w[1] * by_label[b]["excitation_cm1"]) / sum(w),
                    splitting_cm1=by_label[b]["excitation_cm1"] - by_label[a]["excitation_cm1"]))
            stationary = all(m["residual"] < 1e-7 and m["indicator_Ha"] < 1e-8
                             for m in (retained_metrics, full_vectors[3], corrected_vectors[3]))
            manifest.update(qualified=True, rows=rows, multiplets=multiplets,
                nominal_four_gap_RMS_cm1=math.sqrt(math.fsum(r["nominal_gap_cm1"]**2
                    for r in rows if r["label"] != "6s_1/2") / 4),
                original_stationarity_passed=stationary,
                no_physical_completeness_or_experimental_certificate=True)
            write(prefix.with_suffix(".json"), manifest)
            entry = dict(iteration=iteration, manifest=str(prefix.with_suffix(".json").relative_to(REPO)),
                manifest_sha256=sha(prefix.with_suffix(".json")), array_sha256=digest,
                diagnostics=diag, original_stationarity_passed=stationary,
                nominal_four_gap_RMS_cm1=manifest["nominal_four_gap_RMS_cm1"])
            trajectory.append(entry)
            previous = dict(certificate=cert, manifest=entry["manifest"], rows=by_label)
            source_guard()
            progress("accepted_checkpoint_qualified")

        _, _, diagnostics = optimization.optimize_orbitals(functional, waves,
            resume_state=initial, objective_identity=objective["identity"], allow_objective_change=False,
            max_iter=16, tolerance=1e-7, energy_tolerance=1e-8,
            projector_source="nuclear", occupation_floor=.05, orbital_preconditioner="spectral",
            history_size=10, energy_difference=None, checkpoint_callback=checkpoint,
            max_wall_time=caps["optimizer_soft_seconds"], return_state=False)
        source_guard()
        status = "bounded_f043_public_optimizer_returned"
    except BaseException as failure:
        status = "first_failure_preserved_no_retry"
        error = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        write(output / "result.json", dict(status=status, error=error, checks=checks,
            all_completed_controls_passed=all(r["passed"] for r in checks),
            attempted=attempted, completed=completed, trajectory=trajectory, archives=archives,
            operator_events=operator_events, loaded_f043_modules=loaded,
            last_qualified_checkpoint=previous["manifest"] if previous else None,
            optimizer_diagnostics=diagnostics, exact_budget=exact_budget.ledger() if exact_budget else None,
            original_stationarity_passed=bool(trajectory and trajectory[-1]["original_stationarity_passed"]),
            source_map_sha256=contract["source"]["map_sha256"], driver_sha256=sha(__file__),
            contract_sha256=sha(HERE / "contract.json"), execution_binding_sha256=sha(binding_path),
            resource_admission=admission, ordinary_reported_objective_unchanged=True,
            current_source_adopted=False, current_regression_failures_unchanged=True,
            no_energy_callback=True, no_GPU=True, no_global_derivative_error_bound=True,
            experimental_certified=False, physical_completeness_certified=False,
            elapsed_uncontrolled_s=time.monotonic() - started,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    if error:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-binding", type=Path)
    parser.add_argument("--metadata", action="store_true")
    args = parser.parse_args()
    contract = load(HERE / "contract.json")
    if args.metadata:
        metadata(contract, verify_arrays=False)
        print(json.dumps(dict(source_metadata_admitted=True, NPZ_reads_or_rehashes=0, science_calls=0)))
        return
    if args.execution_binding is None:
        raise ValueError("Separate exact root execution binding required")
    execute(contract, args.execution_binding)


if __name__ == "__main__":
    main()
