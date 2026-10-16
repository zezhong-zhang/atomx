"""Bounded genuine mixed-CI EOL gradient, derivative and one-step diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def write(name, value):
    (HERE / name).write_text(json.dumps(plain(value), indent=2, ensure_ascii=False) + "\n")


def main():
    started = time.monotonic()
    contract = json.loads((HERE / "contract.json").read_text())
    fixed, gates, caps = contract["fixed_conditions"], contract["acceptance"], contract["resources"]
    checks, stages, trials = [], {}, []
    result = {"status": "running", "scope": "bounded mixed-CI EOL stationarity/one-step diagnostic",
              "expanded_MCDHF_converged": False, "production_certified": False,
              "experimental_gate_passed": False, "highest_level_reached": False,
              "contract_sha256": sha(HERE / "contract.json")}

    def check(name, error, limit):
        error = float(error)
        row = {"name": name, "error": error, "limit": limit,
               "passed": bool(error <= limit)}
        checks.append(row)
        if not row["passed"]:
            write("checks_partial.json", checks)
            raise AssertionError(f"{name}: {error} > {limit}")

    def stage(name, call):
        before = time.monotonic()
        answer = call()
        stages[name] = time.monotonic() - before
        print(json.dumps({"stage": name, "seconds_uncontrolled": stages[name],
                          "elapsed_s": time.monotonic() - started}), flush=True)
        return answer

    done = threading.Event()

    def heartbeat():
        while not done.wait(30):
            print(json.dumps({"elapsed_s": time.monotonic() - started,
                              "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                              "status": result["status"]}), flush=True)

    threading.Thread(target=heartbeat, daemon=True).start()

    def timeout(*_):
        raise TimeoutError("prospective 890 second mixed-CI probe cap exceeded")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(caps["wall_cap_s"])
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = "1"
    byte_cap = int(caps["address_space_cap_GiB"] * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (byte_cap, byte_cap))
    try:
        for path, expected in contract["inputs_sha256"].items():
            if sha(ROOT / path) != expected:
                raise ValueError("frozen input changed: " + path)
        native = ROOT / fixed["native_run"]
        replay = ROOT / fixed["balanced_replay"]
        saved_metric = ROOT / fixed["original_metric_run"]
        frozen = ROOT / fixed["frozen_source_root"]
        source = json.loads((native / "source_identity.json").read_text())
        if source["source_map_sha256"] != fixed["source_map_sha256"]:
            raise ValueError("source map identity changed")
        for path, expected in source["all_atomx_python_sha256"].items():
            if sha(frozen / Path(path).relative_to("src")) != expected:
                raise ValueError("frozen005 source changed: " + path)
        sys.path.insert(0, str(frozen))
        import numpy as np

        from atomx.angular.csf_hamiltonian import csf_signature, pattern_m_counts
        from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
        from atomx.constants import HARTREE_CM1
        from atomx.methods.ci import CIResult
        from atomx.methods.csf.variational import CSFOrbitalFunctional
        from atomx.numerics.bspline.orbital_optimization import _polar_retraction
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.physics.csf.matrix import build_ci_matrix
        from atomx.physics.csf.state_average import CIStateAverage
        from atomx.physics.csf.tracking import CIBasisTracking
        from atomx.potentials.nuclear import make_nuclear_potential

        enumeration = json.loads((replay / "basis_identity.json").read_text())
        csfs = [csf for config in enumeration["configs"] for csf in enumerate_jj_csfs(config["config"])]
        if len(csfs) != 531 or {csf.n_electrons for csf in csfs} != {69}:
            raise ValueError("complete 531-CSF/69-electron physical model changed")
        metadata = enumeration["CSFs"]
        for index, csf in enumerate(csfs):
            sig = csf_signature(csf)
            expected = metadata[index]
            if (sig.total_J2 != expected["twoJ"] or csf.parity != expected["parity"]
                    or plain(sig.subshells) != expected["subshells"]
                    or plain(sig.coupling) != expected["coupling"]):
                raise ValueError("signed CSF identity changed")
        blocks = {}
        for index, csf in enumerate(csfs):
            blocks.setdefault((csf.total_J2, csf.parity), []).append(index)
        preflight_blocks = []
        overlap_pair_upper = 0
        target_binding = json.loads((replay / "stage_4f5p5s/target_binding.json").read_text())
        targets = target_binding["targets"]
        selectors = tuple((t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.) for t in targets)
        target_symmetries = {(row[0], row[1]) for row in selectors}
        for symmetry, indices in sorted(blocks.items()):
            patterns = {csf_signature(csfs[index]).pattern for index in indices}
            raw = sum(dict(pattern_m_counts(pattern)).get(symmetry[0], 0) for pattern in patterns)
            if raw > caps["max_raw_determinants_per_J_parity"]:
                raise MemoryError("complete physical block exceeds declared raw cap")
            if symmetry in target_symmetries:
                overlap_pair_upper += raw**2
            preflight_blocks.append({"twoJ": symmetry[0], "parity": symmetry[1],
                                     "CSFs": len(indices), "raw_M_equals_J": raw})
        if overlap_pair_upper > caps["max_determinant_overlap_pairs"]:
            raise MemoryError("complete selected-state physical overlap exceeds pair budget")
        radial = np.load(native / "orbitals_workspace.npz")
        data = {name[6:]: radial[name].item() if radial[name].ndim == 0 else radial[name]
                for name in radial.files if name.startswith("basis_")}
        orbitals = {tuple(key): (p.copy(), q.copy()) for key, p, q in
                    zip(radial["orbital_keys"], radial["P"], radial["Q"], strict=True)}
        if set(get_shared_orbital_basis(csfs)) != set(orbitals) or len(orbitals) != 24 or data["ns"] != 123:
            raise ValueError("same24 radial basis identity changed")
        metric_archive = np.load(saved_metric / "failed_coordinates.npz")
        layout = json.loads((saved_metric / "failed_coordinate_layout.json").read_text())
        metric_digest = hashlib.sha256()
        for group, row in enumerate(layout):
            keys = [tuple(key) for key in row["keys"]]
            metric_digest.update(repr((keys, tuple(row["shape"]))).encode())
            metric_digest.update(np.ascontiguousarray(metric_archive[f"projector_{group}"]).tobytes())
        metric_digest.update(metric_archive["inverse_diagonal"].tobytes())
        for group, _ in enumerate(layout):
            for block in metric_archive[f"inverse_blocks_{group}"]:
                metric_digest.update(np.ascontiguousarray(block).tobytes())
        if metric_digest.hexdigest() != fixed["original_projector_and_preconditioner_metric_sha256"]:
            raise ValueError("original full projector/preconditioner residual metric mismatch")
        preflight = {"CSFs": 531, "electrons": 69, "radial_orbitals": 24, "blocks": preflight_blocks,
                     "original_metric_sha256": metric_digest.hexdigest(),
                     "max_raw_M_equals_J": max(b["raw_M_equals_J"] for b in preflight_blocks),
                     "selected_overlap_pair_work_upper": overlap_pair_upper,
                     "resource_caps": caps, "source_files_verified": len(source["all_atomx_python_sha256"]),
                     "inputs_sha256": contract["inputs_sha256"], "load_average": os.getloadavg()}
        write("preflight.json", preflight)
        print(json.dumps({"preflight": "pass", "CSFs": 531, "overlap_pair_upper": overlap_pair_upper}), flush=True)
        state_average = CIStateAverage(selectors)
        baseline_arrays = np.load(replay / "stage_4f5p5s/DC_arrays.npz")
        reference_ci = CIResult(baseline_arrays["energies_Ha"].copy(), baseline_arrays["mixing"].copy())
        provisional = stage("build_full_mixed_functional", lambda: CSFOrbitalFunctional(
            csfs, data, 70, make_nuclear_potential(70, "fermi"), state_average=state_average, max_determinants=2000))
        reference_radial_metric = provisional.metric
        tracking = stage("bind_full_physical_CI_reference", lambda: CIBasisTracking(
            state_average, csfs, reference_ci, orbitals, reference_radial_metric,
            max_determinant_pairs=caps["max_determinant_overlap_pairs"]))
        functional = CSFOrbitalFunctional(csfs, data, 70, make_nuclear_potential(70, "fermi"),
            state_average=state_average, level_tracking=tracking, cache=provisional.cache, max_determinants=2000)
        del provisional

        def check_geometry(waves, name):
            gram_max, no_pair_max = 0., 0.
            for group, row in enumerate(layout):
                keys = [tuple(key) for key in row["keys"]]
                c = np.column_stack([np.concatenate(waves[key]) for key in keys])
                s = functional.metric[keys[0][1]]
                p = metric_archive[f"projector_{group}"]
                gram_max = max(gram_max, float(np.max(abs(c.T @ s @ c - np.eye(len(keys))))))
                outside = c - p @ (p.T @ s @ c)
                no_pair_max = max(no_pair_max, float(np.sqrt(max(0., np.max(np.einsum(
                    "ij,ij->j", outside, s @ outside))))))
            check(name + "/radial_Gram", gram_max, gates["radial_Gram_error"])
            check(name + "/same_projector_leakage_norm", no_pair_max, gates["projector_leakage_norm"])
            return {"radial_Gram_error": gram_max, "projector_leakage_norm": no_pair_max}

        def matrix(waves):
            return build_ci_matrix(csfs, waves, functional.keys, data, 70, 123,
                                   nuclear_potential=functional.nuclear_potential,
                                   convol_cache=functional.cache, max_determinants=2000)

        def check_ci(waves, ci, name, h=None):
            if h is None:
                h = matrix(waves)
            e, v = ci.state_energies, ci.mixing_coefficients
            if e.shape != (531,) or v.shape != (531, 531) or not np.isfinite(h).all():
                raise ValueError("incomplete or nonfinite real full CI")
            check(name + "/CI_residual_Ha", np.max(np.linalg.norm(h @ v - v * e, axis=0)), gates["CI_residual_Ha"])
            check(name + "/CI_orthogonality", np.max(abs(v.T @ v - np.eye(531))), gates["CI_orthogonality"])
            check(name + "/Hermiticity_Ha", np.max(abs(h - h.T)), gates["matrix_hermiticity_Ha"])
            cross = max(float(np.max(abs(h[np.ix_(indices, others)])))
                        for symmetry, indices in blocks.items()
                        for other, others in blocks.items() if symmetry != other)
            check(name + "/cross_symmetry_zero_Ha", cross, gates["cross_symmetry_zero_Ha"])
            weights, identities = tracking.resolve(csfs, ci, waves, functional.metric)
            if len(np.flatnonzero(weights)) != 5 or not np.array_equal(weights[weights > 0], np.repeat(.2, 5)):
                raise ValueError("fixed five equal physical target weights changed")
            check_geometry(waves, name)
            return h, weights, identities

        result["status"] = "baseline_actual_gradient"
        energy, gradient, ci = stage("actual_full_mixed_EO_gradient_and_CI", lambda: functional.evaluate(orbitals))
        check("same_baseline_all_CI_energies_Ha", np.max(abs(ci.state_energies - reference_ci.state_energies)),
              gates["baseline_target_energy_replay_Ha"])
        baseline_h, weights, identities = stage("baseline_full_numeric_identity", lambda: check_ci(
            orbitals, ci, "baseline"))
        check("same_baseline_matrix_Ha", np.max(abs(baseline_h - baseline_arrays["matrix_Ha"])), gates["baseline_matrix_replay_Ha"])
        x = np.zeros_like(metric_archive["x"])
        reconstruction_max = 0.
        for group, row in enumerate(layout):
            keys, shape, where = [tuple(key) for key in row["keys"]], tuple(row["shape"]), slice(*row["slice"])
            p = metric_archive[f"projector_{group}"]
            c = np.column_stack([np.concatenate(orbitals[key]) for key in keys])
            y = np.asarray(p.astype(np.longdouble).T @ functional.metric[keys[0][1]].astype(np.longdouble)
                           @ c.astype(np.longdouble), dtype=float)
            x[where] = y.ravel()
            reconstruction_max = max(reconstruction_max, float(np.max(abs(p @ y - c))))
        check("actual005_coordinate_reconstruction_max", reconstruction_max, gates["coordinate_reconstruction_max"])

        def tangent(point, vector):
            projected = vector.copy()
            for row in layout:
                shape, where = tuple(row["shape"]), slice(*row["slice"])
                y, g = point[where].reshape(shape), vector[where].reshape(shape)
                multiplier = y.T @ g
                projected[where] = (g - y @ (.5 * (multiplier + multiplier.T))).ravel()
            return projected

        def coordinate_gradient(point, radial_gradient):
            g = np.zeros_like(point)
            for group, row in enumerate(layout):
                keys, where = [tuple(key) for key in row["keys"]], slice(*row["slice"])
                p = metric_archive[f"projector_{group}"]
                g[where] = (p.T @ np.column_stack([radial_gradient[key] for key in keys])).ravel()
            return tangent(point, g)

        def precondition(point, g):
            scaled = np.zeros_like(g)
            for group, row in enumerate(layout):
                where, shape = slice(*row["slice"]), tuple(row["shape"])
                matrix_g = g[where].reshape(shape)
                scaled[where] = np.column_stack([block @ matrix_g[:, col] for col, block in
                    enumerate(metric_archive[f"inverse_blocks_{group}"])]).ravel()
            return tangent(point, scaled)

        def unpack(point):
            waves = {}
            for group, row in enumerate(layout):
                keys, shape, where = [tuple(key) for key in row["keys"]], tuple(row["shape"]), slice(*row["slice"])
                c = metric_archive[f"projector_{group}"] @ point[where].reshape(shape)
                for col, key in enumerate(keys):
                    waves[key] = c[:123, col].copy(), c[123:, col].copy()
            return waves

        def retract(point):
            trial = point.copy()
            for row in layout:
                shape, where = tuple(row["shape"]), slice(*row["slice"])
                trial[where] = _polar_retraction(point[where].reshape(shape)).ravel()
            return trial

        g = coordinate_gradient(x, gradient)
        pg = precondition(x, g)
        residual = float(np.linalg.norm(pg))
        estimated = .5 * abs(float(g @ pg))
        gradient_norm = float(np.linalg.norm(g))
        result["baseline"] = {"EOL_objective_Ha": energy, "gradient_norm": gradient_norm,
                              "original_metric_orbital_residual": residual,
                              "original_orbital_tolerance": gates["original_orbital_tolerance"],
                              "original_metric_quadratic_energy_indicator_Ha": estimated,
                              "original_energy_tolerance_Ha": gates["original_estimated_energy_tolerance_Ha"],
                              "stationarity_passed": residual < 1e-7 and estimated < 1e-8,
                              "quadratic_indicator_is_not_strict_missing_physics_bound": True,
                              "coordinate_reconstruction_max": reconstruction_max,
                              "equal_EO_state_weights": weights, "physical_identity_tracking": identities}
        np.savez_compressed(HERE / "baseline_arrays.npz", x=x, g=g, original_metric_preconditioned_gradient=pg,
                            gradient_keys=np.array(functional.keys), gradient=np.array([gradient[key] for key in functional.keys]),
                            matrix_Ha=baseline_h, energies_Ha=ci.state_energies, mixing=ci.mixing_coefficients)
        write("result_partial.json", result)
        print(json.dumps({"baseline_orbital_residual": residual, "quadratic_indicator_Ha": estimated,
                          "stationarity_passed": result["baseline"]["stationarity_passed"]}), flush=True)
        zero_waves = unpack(x)
        zero_energy, zero_ci = stage("zero_reconstructed_coordinate_control", lambda: functional.evaluate(zero_waves, gradient=False))
        check("zero_coordinate_objective_replay_Ha", abs(zero_energy - energy), gates["baseline_target_energy_replay_Ha"])
        result["baseline"]["zero_coordinate_EOL_delta_Ha"] = zero_energy - energy
        result["mixed_CI_delta_API"] = {
            "fixed_density_difference_returns_None": functional.fixed_density_energy_difference(orbitals, zero_waves, ci, zero_ci) is None,
            "constrained_difference_returns_None": functional.constrained_energy_difference(orbitals, zero_waves, ci, zero_ci) is None,
            "actual_mixed_CI_re_solved_energy_used": True}
        if not all(result["mixed_CI_delta_API"][name] for name in
                   ("fixed_density_difference_returns_None", "constrained_difference_returns_None")):
            raise ValueError("unsupported mixed/tracked CI silently used fixed-density difference")

        def evaluate_trial(point, label):
            trial_waves = unpack(point)
            value, trial_ci = functional.evaluate(trial_waves, gradient=False)
            h, tw, tracked = check_ci(trial_waves, trial_ci, label)
            filename = label.replace("/", "_") + ".npz"
            np.savez_compressed(HERE / filename, x=point, energies_Ha=trial_ci.state_energies,
                                mixing=trial_ci.mixing_coefficients, matrix_Ha=h,
                                orbital_keys=np.array(functional.keys),
                                P=np.array([trial_waves[key][0] for key in functional.keys]),
                                Q=np.array([trial_waves[key][1] for key in functional.keys]))
            row = {"name": label, "EOL_objective_Ha": value, "delta_from_actual_baseline_Ha": value - energy,
                   "physical_tracking": tracked, "state_weights": tw, "artifact": filename}
            trials.append(row)
            write("trials_partial.json", trials)
            print(json.dumps({"trial": label, "EOL_delta_Ha": value - energy,
                              "elapsed_s": time.monotonic() - started}), flush=True)
            return value, trial_ci, trial_waves, tw, row

        descent = -pg
        if np.linalg.norm(descent) == 0:
            raise ValueError("zero direction: expected mixed response needs no one-step probe")
        directions = {"normalized_original_metric_descent": descent / np.linalg.norm(descent)}
        valence = np.zeros_like(descent)
        for row in layout:
            keys, shape, where = [tuple(key) for key in row["keys"]], tuple(row["shape"]), slice(*row["slice"])
            masked = descent[where].reshape(shape).copy()
            for col, key in enumerate(keys):
                if key not in ((6, -1), (5, 2), (5, -3), (6, 1), (6, -2)):
                    masked[:, col] = 0.
            valence[where] = masked.ravel()
        valence = tangent(x, valence)
        if np.linalg.norm(valence) == 0:
            raise ValueError("declared valence-focused tangent has zero norm")
        directions["normalized_valence_focused_tangent"] = valence / np.linalg.norm(valence)
        derivative_rows = []
        result["status"] = "independent_CI_energy_derivative_controls"
        for name, direction in directions.items():
            analytic = float(g @ direction)
            slopes = []
            for step in (1e-4, 5e-5):
                plus = evaluate_trial(retract(x + step * direction), f"FD_{name}_{step}_plus")[0]
                minus = evaluate_trial(retract(x - step * direction), f"FD_{name}_{step}_minus")[0]
                central = (plus - minus) / (2 * step)
                allowance = gates["central_derivative_absolute_Ha_per_coordinate"] + gates["central_derivative_relative"] * abs(analytic)
                row = {"direction": name, "step": step, "analytic_Ha_per_coordinate": analytic,
                       "central_re_solved_CI_Ha_per_coordinate": central,
                       "absolute_error": abs(central - analytic), "predeclared_allowance": allowance}
                derivative_rows.append(row)
                check(f"mixed_HF_vs_resolved_CI_derivative/{name}/{step}", abs(central - analytic), allowance)
                slopes.append(central)
            print(json.dumps({"validated_direction": name, "analytic": analytic, "central_slopes": slopes}), flush=True)
        result["derivative_controls"] = derivative_rows
        np.savez_compressed(HERE / "derivative_directions.npz", **directions)
        write("result_partial.json", result)

        result["status"] = "bounded_one_step_line_search"
        directional_derivative = float(g @ descent)
        if directional_derivative >= 0:
            raise ValueError("original-metric direction is not descent")
        line_search, accepted = [], None
        for attempt in range(caps["line_search_CI_re_solves_max"]):
            step = 2.**(-attempt)
            candidate = retract(x + step * descent)
            trial_energy, trial_ci, trial_waves, trial_weights, row = evaluate_trial(candidate, f"line_trial_{attempt}")
            change = trial_energy - energy
            bound = gates["Armijo"] * step * directional_derivative
            accepted_here = bool(np.isfinite(change) and change <= bound)
            line_search.append({"attempt": attempt, "step": step, "EOL_delta_Ha": change,
                                "same_objective_Armijo_bound_Ha": bound, "accepted": accepted_here,
                                "actual_total_energy_comparison": True})
            if accepted_here:
                accepted = candidate, trial_energy, trial_ci, trial_waves, trial_weights, row
                break
        result["line_search"] = line_search
        result["accepted_one_step"] = accepted is not None
        if accepted is not None:
            candidate, trial_energy, trial_ci, trial_waves, trial_weights, row = accepted
            final_energy, final_gradient, final_ci = stage("accepted_mixed_EO_actual_gradient", lambda: functional.evaluate(
                trial_waves, ci_result=trial_ci))
            final_g = coordinate_gradient(candidate, final_gradient)
            final_pg = precondition(candidate, final_g)
            final_residual = float(np.linalg.norm(final_pg))
            final_estimated = .5 * abs(float(final_g @ final_pg))
            _, baseline_level_blocks = ci_level_blocks(csfs, ci, require_full=True)
            _, final_level_blocks = ci_level_blocks(csfs, final_ci, require_full=True)
            energy_rows = []
            for target in targets:
                symmetry = target["twoJ"], target["parity"]
                before = [int(col) for col in baseline_level_blocks[symmetry] if weights[col] > 0]
                after = [int(col) for col in final_level_blocks[symmetry] if trial_weights[col] > 0]
                if len(before) != 1 or len(after) != 1:
                    raise ValueError("one-step target identity lost or nonunique")
                energy_rows.append({"label": target["label"], "twoJ": symmetry[0], "parity": symmetry[1],
                                   "baseline_column": before[0], "accepted_column": after[0],
                                   "baseline_energy_Ha": float(ci.state_energies[before[0]]),
                                   "accepted_energy_Ha": float(final_ci.state_energies[after[0]]),
                                   "absolute_energy_delta_Ha": float(final_ci.state_energies[after[0]] - ci.state_energies[before[0]])})
            lower = energy_rows[0]
            for item in energy_rows:
                item["transition_delta_from_same_lower_cm1"] = (item["absolute_energy_delta_Ha"] - lower["absolute_energy_delta_Ha"]) * HARTREE_CM1
                item["baseline_transition_cm1"] = (item["baseline_energy_Ha"] - lower["baseline_energy_Ha"]) * HARTREE_CM1
                item["accepted_transition_cm1"] = (item["accepted_energy_Ha"] - lower["accepted_energy_Ha"]) * HARTREE_CM1
            result["one_step"] = {"EOL_objective_Ha": final_energy, "EOL_delta_Ha": final_energy - energy,
                                  "original_metric_orbital_residual": final_residual,
                                  "original_metric_quadratic_energy_indicator_Ha": final_estimated,
                                  "stationarity_passed": final_residual < 1e-7 and final_estimated < 1e-8,
                                  "lower_upper_difference_rows": energy_rows,
                                  "same_original_metric_SHA": metric_digest.hexdigest(),
                                  "physical_tracking": functional.last_tracking_diagnostics,
                                  "evidence_artifact": row["artifact"],
                                  "not_full_MCDHF_or_missing_physics_bound": True}
            np.savez_compressed(HERE / "accepted_gradient.npz", x=candidate, g=final_g,
                                original_metric_preconditioned_gradient=final_pg,
                                gradient_keys=np.array(functional.keys),
                                gradient=np.array([final_gradient[key] for key in functional.keys]))
        result["status"] = "bounded_probe_completed"
        result["scientific_identity_and_derivative_gates_passed"] = True
    except Exception as exc:
        result["status"] = "failed_bounded_probe"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        result["scientific_identity_and_derivative_gates_passed"] = False
        print(traceback.format_exc(), flush=True)
    finally:
        signal.alarm(0)
        done.set()
        result["checks"] = checks
        result["stages_uncontrolled_s"] = stages
        result["trials"] = trials
        result["inputs_sha256"] = contract["inputs_sha256"]
        result["source_map_sha256"] = fixed["source_map_sha256"]
        result["original_metric_sha256"] = fixed["original_projector_and_preconditioner_metric_sha256"]
        result["resource"] = {"seconds_uncontrolled": time.monotonic() - started,
                              "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                              "threads": 1, "address_space_cap_GiB": caps["address_space_cap_GiB"],
                              "wall_cap_s": caps["wall_cap_s"], "timing_is_not_controlled_benchmark": True}
        result["artifacts_sha256"] = {path.name: sha(path) for path in sorted(HERE.iterdir())
                                     if path.name not in {"result.json", "README.md", "verification.json", "run.log", "trace.md"}
                                     and path.is_file()}
        write("result.json", result)
        write("verification.json", {"scientific_identity_and_derivative_gates_passed": result["scientific_identity_and_derivative_gates_passed"],
                                    "checks": checks, "result_sha256": sha(HERE / "result.json"),
                                    "artifacts_sha256": result["artifacts_sha256"],
                                    "expanded_MCDHF_converged": False, "production_certified": False})
        print(json.dumps({"status": result["status"], "seconds_uncontrolled": result["resource"]["seconds_uncontrolled"],
                          "result_sha256": sha(HERE / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
