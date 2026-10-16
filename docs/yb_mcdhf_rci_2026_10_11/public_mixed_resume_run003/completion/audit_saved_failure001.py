"""Read-only saved-array audit: no Atomx import or physical solver invocation."""

import ast
import hashlib
import json
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = HERE.parents[3]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    start = time.monotonic()
    contract = load(HERE / "contract.json")
    seal = load(HERE / "execution_contract.json")
    if seal["audit_script_sha256"] != sha(Path(__file__)):
        raise ValueError("Audit script changed after prospective execution seal")
    if seal["contract_sha256"] != sha(HERE / "contract.json"):
        raise ValueError("Prospective contract changed")
    caps = contract["resources"]
    os.sched_setaffinity(0, set(caps["cpu_affinity"]))
    resource.setrlimit(resource.RLIMIT_AS, (caps["address_space_cap_bytes"],) * 2)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("30 s saved audit cap")))
    signal.alarm(caps["wall_cap_s"])
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    import numpy as np

    gates = contract["acceptance"]
    fixed = contract["fixed_conditions"]
    checks = []

    def check(name, passed, **detail):
        checks.append({"name": name, "passed": bool(passed), **detail})
        if not passed:
            raise ValueError("Saved audit failed: " + name)

    def close(name, error, limit):
        check(name, np.isfinite(error) and error <= limit, error=float(error), limit=limit)

    inputs = load(HERE / "input_file_sha256.json")
    check("prospective_input_manifest_identity",
          sha(HERE / "input_file_sha256.json") == contract["input_file_manifest_sha256"])
    for name, expected in inputs.items():
        check("before_hash:" + name, sha(ROOT / name) == expected)
    source = load(RUN / "source_identity.json")
    check("full353_scientific_source_map", source["source_map_sha256"] == fixed["source_map_sha256"])
    check("full353_source_count", len(source["all_atomx_python_sha256"]) == 353)
    for name, expected in source["all_atomx_python_sha256"].items():
        check("frozen_source:" + name, sha(RUN / "source" / name.removeprefix("src/")) == expected)
    objective = load(RUN / "objective_identity.json")
    check("same_real_CI_objective", objective["new_sha256"] == fixed["objective_sha256"]
          and objective["same_objective_as_parent"] is True)
    check("complete531_basis", len(objective["new"]["CSFs"]) == 531)
    check("equal_weight_five_targets", sorted(t["weight"] for t in objective["new"]["target_identity"]) == [.2] * 5)
    check("physical_scope", objective["new"]["Z"] == 70 and objective["new"]["electrons"] == 69
          and objective["new"]["nucleus"] == "native005 Fermi A175")
    target_order = ("6s_1/2", "5d_3/2", "5d_5/2", "6p_1/2", "6p_3/2")
    target_specs = {t["label"]: (t["twoJ"], t["parity"], t["root"])
                    for t in objective["new"]["target_identity"]}
    blocks = {}
    for record in objective["new"]["CSFs"]:
        blocks.setdefault((record["twoJ"], record["parity"]), []).append(record["index"])
    check("complete16_symmetry_blocks", len(blocks) == 16)
    block_keys = sorted(blocks)
    block_rows = [np.array(blocks[key]) for key in block_keys]
    parent = ROOT / "docs/yb_mcdhf_rci_2026_10_11/public_mixed_resume_run002"
    layout_path = ROOT / "docs/yb_mcdhf_rci_2026_10_11/native_run/nuclear_eol004_stable_delta_trace/failed_coordinate_layout.json"
    layout = load(layout_path)
    result = load(RUN / "result_unverified.json")
    replay = load(RUN / "independent_endpoint_verification.json")
    parent_replay = load(parent / "independent_endpoint_verification.json")
    check("root_solve_exit0", load(RUN / "root_execution.json")["exit_code"] == 0)
    check("root_independent_replay_exit0", load(RUN / "root_endpoint_execution.json")["exit_code"] == 0)
    check("exact1011_replay_proof", replay["checks_count"] == 1011
          and replay["all_identity_replay_checks_passed"] is True)
    for name, limit in (("saved_samepoint_complete_projected_gradient", 1e-9),
                        ("saved_samepoint_complete_preconditioned_gradient", 1e-10)):
        entry = next(c for c in replay["checks"] if c["name"] == name)
        check("independent_full_vector:" + name, entry["passed"] and entry["limit"] == limit
              and entry["error"] <= limit, observed_error=entry["error"], limit=limit)
    calls = [n for n in ast.walk(ast.parse((RUN / "run.py").read_text()))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "optimize_orbitals"]
    check("single_public_optimizer_call_in_frozen_driver", len(calls) == 1)
    check("one_completed_native_additional160", result["actual_full_public_optimizer_call"]
          and result["same_objective_as_parent"] and result["parent_accepted_iteration"] == 160
          and result["diagnostics"]["iterations"] == 160 and len(result["checkpoints"]) == 161)
    check("fresh_secants_same_objective", result["fresh_secant_history_declared"]
          and result["diagnostics"]["initial_history_size"] == 0
          and result["diagnostics"]["objective_changed"] is False)
    check("actual_calls_bound", result["optimizer_full_gradient_calls"] == 161
          and result["baseline_gradient_calls"] == 2 and result["total_run_full_gradient_calls"] == 163
          and result["energy_trial_calls"] == 252)
    check("original_gate_and_iteration_stop", result["diagnostics"]["orbital_tolerance"] == 1e-7
          and result["diagnostics"]["energy_tolerance"] == 1e-8
          and result["diagnostics"]["termination_reason"] == "maximum_iterations"
          and result["diagnostics"]["line_search_fallbacks"] == 0
          and result["diagnostics"]["stable_energy_comparisons"] == 0)
    check("inherited_metadata_does_not_rebuild_metric", result["diagnostics"]["restored_original_metric"]
          and result["diagnostics"]["residual_metric_scope"]
          == "restored original projector/inverse metric; no occupation or spectral rebuild")

    with np.load(parent / "checkpoints/accepted_0160.npz", allow_pickle=False) as data:
        parent_arrays = {name: data[name].copy() for name in (
            "x", "original_projected_gradient", "original_preconditioned_gradient")}
    records = []
    first_geometry = None
    previous_vectors = None
    previous_energies = None
    previous_columns = None
    first_vectors = None
    tracking_initial = None
    legacy_metric = None
    max_pg_replay = max_residual_replay = max_indicator_replay = max_wave_replay = 0.
    max_selected_eigenpair_residual = max_target_support_leakage = 0.
    all_wave_exact = True
    for iteration in range(161):
        path = RUN / f"checkpoints/accepted_{iteration:04d}.json"
        manifest = load(path)
        file = path.with_suffix(".npz")
        check(f"checkpoint{iteration}:manifest_record",
              result["checkpoints"][iteration]["path"] == str(path.relative_to(RUN))
              and result["checkpoints"][iteration]["sha256"] == sha(path))
        check(f"checkpoint{iteration}:complete_identities",
              manifest["iteration"] == iteration and manifest["accepted_complete_point"]
              and manifest["parent_accepted_iteration"] == 160
              and manifest["array_sha256"] == sha(file)
              and manifest["objective_sha256"] == fixed["objective_sha256"]
              and manifest["original_metric_sha256"] == fixed["original_metric_sha256"]
              and manifest["geometry_sha256"] == fixed["geometry_sha256"])
        check(f"checkpoint{iteration}:execution_inputs",
              manifest["source_identity_sha256"] == sha(RUN / "source_identity.json")
              and manifest["preparation_contract_sha256"] == sha(RUN / "contract.json")
              and manifest["execution_contract_sha256"] == sha(RUN / "execution_contract.json"))
        with np.load(file, allow_pickle=False) as data:
            x = data["x"]
            g = data["original_projected_gradient"]
            pg = data["original_preconditioned_gradient"]
            check(f"checkpoint{iteration}:full_gradient_point_arrays",
                  all(a.shape == (2856,) and a.dtype == np.float64 and np.isfinite(a).all()
                      for a in (x, g, pg)))
            geometry = {name: data[name] for name in data.files
                        if name == "inverse_diagonal" or name.startswith(("projector_", "inverse_blocks_"))}
            if first_geometry is None:
                first_geometry = {name: a.copy() for name, a in geometry.items()}
                digest = hashlib.sha256()
                for i, row in enumerate(layout):
                    keys = [tuple(key) for key in row["keys"]]
                    digest.update(repr((keys, tuple(row["shape"]))).encode())
                    digest.update(np.ascontiguousarray(geometry[f"projector_{i}"]).tobytes())
                digest.update(geometry["inverse_diagonal"].tobytes())
                for i in range(len(layout)):
                    for block in geometry[f"inverse_blocks_{i}"]:
                        digest.update(np.ascontiguousarray(block).tobytes())
                legacy_metric = digest.hexdigest()
                check("exact_original_legacy_metric_byte_digest", legacy_metric == fixed["original_metric_sha256"])
                for name, array in parent_arrays.items():
                    actual = {"x": x, "original_projected_gradient": g,
                              "original_preconditioned_gradient": pg}[name]
                    check("exact_parent160_start:" + name, np.array_equal(actual, array))
            check(f"checkpoint{iteration}:unchanged_full_geometry",
                  geometry.keys() == first_geometry.keys()
                  and all(np.array_equal(a, first_geometry[name]) for name, a in geometry.items()))
            pg_replay = np.empty_like(pg)
            keys_array = data["orbital_keys"]
            check(f"checkpoint{iteration}:24_radial_keys", keys_array.shape == (24, 2)
                  and data["P"].shape == (24, 123) and data["Q"].shape == (24, 123))
            key_position = {tuple(key): i for i, key in enumerate(keys_array)}
            check(f"checkpoint{iteration}:unique_radial_keys", len(key_position) == 24)
            wave_error = tangent_error = coordinate_gram = 0.
            wave_exact = True
            for i, row in enumerate(layout):
                where = slice(*row["slice"])
                y, b = x[where].reshape(row["shape"]), g[where].reshape(row["shape"])
                inverse = geometry[f"inverse_blocks_{i}"]
                # Independent saved-array inverse action, followed by actual x tangent.
                z = np.column_stack([inverse[col] @ b[:, col] for col in range(y.shape[1])])
                multiplier = y.T @ z
                pg_replay[where] = (z - y @ (.5 * (multiplier + multiplier.T))).ravel()
                tangent_error = max(tangent_error, float(np.max(abs(y.T @ b + b.T @ y))))
                coordinate_gram = max(coordinate_gram, float(np.max(abs(y.T @ y - np.eye(y.shape[1])))))
                waves = geometry[f"projector_{i}"] @ y
                for col, key in enumerate(row["keys"]):
                    j = key_position[tuple(key)]
                    actual = np.concatenate((data["P"][j], data["Q"][j]))
                    wave_error = max(wave_error, float(np.max(abs(waves[:, col] - actual))))
                    wave_exact &= np.array_equal(waves[:, col], actual)
            pg_error = float(np.max(abs(pg_replay - pg)))
            close(f"checkpoint{iteration}:full2856_inverse_tangent_replay", pg_error,
                  gates["inverse_tangent_saved_pg_max"])
            close(f"checkpoint{iteration}:direct_originalP_times_x", wave_error,
                  gates["coordinate_reconstruction_max"])
            residual = float(np.linalg.norm(pg))
            indicator = .5 * abs(float(g @ pg))
            reported = manifest["diagnostics"]
            residual_error = abs(residual - reported["orbital_residual"])
            indicator_error = abs(indicator - reported["estimated_energy_error"])
            close(f"checkpoint{iteration}:residual_from_complete_saved_pg", residual_error, 1e-15)
            close(f"checkpoint{iteration}:indicator_from_complete_saved_g_pg", indicator_error, 1e-20)
            check(f"checkpoint{iteration}:same_original_gates",
                  reported["orbital_tolerance"] == 1e-7 and reported["energy_tolerance"] == 1e-8
                  and manifest["stationarity_passed"] == (residual < 1e-7 and indicator < 1e-8))
            tracking = {(entry["selected_levels"][0][0], entry["selected_levels"][0][1],
                         entry["selected_levels"][0][2]): entry for entry in manifest["physical_tracking"]}
            check(f"checkpoint{iteration}:five_exact_tracked_target_groups", len(tracking) == 5
                  and set(tracking) == set(target_specs.values())
                  and all(entry["reference_levels"] == entry["selected_levels"]
                          and len(entry["selected_levels"]) == 1 and entry["per_level_weight"] == .2
                          and min(entry["principal_overlap_squared"]) > .5 for entry in tracking.values()))
            tracking_values = [float(tracking[target_specs[label]]["principal_overlap_squared"][0])
                               for label in target_order]
            if tracking_initial is None:
                tracking_initial = tracking_values
            energies = data["energies_Ha"]
            mixing = data["mixing"]
            matrix = data["matrix_Ha"]
            check(f"checkpoint{iteration}:complete_saved_CI_arrays", energies.shape == (531,)
                  and mixing.shape == (531, 531) and matrix.shape == (531, 531)
                  and all(a.dtype == np.float64 and np.isfinite(a).all() for a in (energies, mixing, matrix)))
            close(f"checkpoint{iteration}:saved_H_hermiticity", float(np.max(abs(matrix - matrix.T))),
                  gates["saved_matrix_hermiticity_Ha"])
            mass = np.array([np.sum(mixing[rows] ** 2, axis=0) for rows in block_rows])
            assignment = np.argmax(mass, axis=0)
            check(f"checkpoint{iteration}:all531_symmetry_block_counts",
                  all(np.count_nonzero(assignment == i) == len(rows) for i, rows in enumerate(block_rows)))
            columns = []
            for label in target_order:
                two_j, parity, local_root = target_specs[label]
                index = block_keys.index((two_j, parity))
                block_columns = np.flatnonzero(assignment == index)
                order = block_columns[np.argsort(energies[block_columns], kind="stable")]
                columns.append(int(order[local_root]))
            vectors = mixing[:, columns]
            selected_energies = energies[columns]
            leakage = 0.
            for col, label in zip(columns, target_order, strict=True):
                key = target_specs[label][:2]
                outside = [row for other, rows in blocks.items() if other != key for row in rows]
                leakage = max(leakage, float(np.linalg.norm(mixing[outside, col])))
            close(f"checkpoint{iteration}:selected_CI_exact_symmetry_support", leakage,
                  gates["saved_CI_target_support_leakage_norm"])
            ci_residual = float(np.max(np.linalg.norm(matrix @ vectors - vectors * selected_energies, axis=0)))
            close(f"checkpoint{iteration}:five_selected_saved_eigenpairs", ci_residual,
                  gates["selected_saved_CI_residual_Ha"])
            weighted_energy = float(np.dot(np.repeat(.2, 5), selected_energies))
            eol_error = abs(weighted_energy - reported["energy_ha"])
            close(f"checkpoint{iteration}:same_actual_equal_weight_CI_objective", eol_error, 5e-9)
            if first_vectors is None:
                first_vectors = vectors.copy()
            aligned_to_initial = vectors * np.where(np.sum(first_vectors * vectors, axis=0) >= 0, 1., -1.)
            from_initial_norm = np.linalg.norm(aligned_to_initial - first_vectors, axis=0)
            adjacent_norm = None
            root_column_switches = None
            state_increment = None
            if previous_vectors is not None:
                aligned = vectors * np.where(np.sum(previous_vectors * vectors, axis=0) >= 0, 1., -1.)
                adjacent_norm = np.linalg.norm(aligned - previous_vectors, axis=0).tolist()
                root_column_switches = int(sum(a != b for a, b in zip(columns, previous_columns, strict=True)))
                state_increment = (selected_energies - previous_energies).tolist()
            record = {"segment_iteration": iteration, "total_accepted_iterations": 160 + iteration,
                      "energy_Ha": reported["energy_ha"], "weighted_CI_energy_replay_error_Ha": eol_error,
                      "orbital_residual": residual, "energy_indicator_Ha": indicator,
                      "saved_projected_gradient_norm": float(np.linalg.norm(g)),
                      "saved_pg_replay_error": pg_error, "direct_Px_wave_error": wave_error,
                      "direct_Px_wave_bitwise_equal": wave_exact,
                      "coordinate_Gram_error": coordinate_gram,
                      "saved_g_tangent_symmetry_error": tangent_error,
                      "original_stationarity_both_gates": residual < 1e-7 and indicator < 1e-8,
                      "selected_columns_by_signed_symmetry_local_root": dict(zip(target_order, columns, strict=True)),
                      "target_energies_Ha": dict(zip(target_order, selected_energies.tolist(), strict=True)),
                      "adjacent_target_energy_delta_Ha": state_increment,
                      "phase_aligned_CI_coefficient_change_from_initial_norm": from_initial_norm.tolist(),
                      "phase_aligned_adjacent_CI_coefficient_change_norm": adjacent_norm,
                      "selected_global_column_changes_from_previous": root_column_switches,
                      "physical_frozen_reference_overlap_squared": dict(zip(target_order, tracking_values, strict=True)),
                      "target_saved_eigenpair_residual_Ha": ci_residual,
                      "target_saved_symmetry_support_leakage_norm": leakage}
            records.append(record)
            previous_vectors, previous_energies, previous_columns = vectors.copy(), selected_energies.copy(), columns
            max_pg_replay = max(max_pg_replay, pg_error)
            max_residual_replay = max(max_residual_replay, residual_error)
            max_indicator_replay = max(max_indicator_replay, indicator_error)
            max_wave_replay = max(max_wave_replay, wave_error)
            max_selected_eigenpair_residual = max(max_selected_eigenpair_residual, ci_residual)
            max_target_support_leakage = max(max_target_support_leakage, leakage)
            all_wave_exact &= wave_exact

    history = np.array([record["energy_Ha"] for record in records])
    check("whole161_history_matches_optimizer", np.array_equal(history, result["diagnostics"]["energy_history"]))
    differences = np.diff(history)
    spacing = np.abs(np.spacing(history[:-1]))
    decrease_ulp = -differences / spacing
    for i, record in enumerate(records[1:]):
        record.update(accepted_energy_delta_Ha=float(differences[i]),
                      preceding_float64_spacing_Ha=float(spacing[i]),
                      accepted_decrease_ULP=float(decrease_ulp[i]))

    def history_summary(start_index):
        values = differences[start_index:]
        ulps = decrease_ulp[start_index:]
        return {"accepted_steps": int(len(values)), "net_energy_change_Ha": float(np.sum(values)),
                "recorded_decrease_steps": int(np.count_nonzero(values < 0)),
                "recorded_equal_energy_steps": int(np.count_nonzero(values == 0)),
                "recorded_increase_steps": int(np.count_nonzero(values > 0)),
                "minimum_decrease_ULP": float(np.min(ulps)), "median_decrease_ULP": float(np.median(ulps)),
                "maximum_decrease_ULP": float(np.max(ulps)),
                "absolute_change_at_most128_ULP_steps": int(np.count_nonzero(np.abs(ulps) <= 128)),
                "absolute_change_at_most1_ULP_steps": int(np.count_nonzero(np.abs(ulps) <= 1))}

    close("final_saved_residual_matches_independent1011", abs(records[-1]["orbital_residual"]
          - replay["actual_original_metric_orbital_residual"]), 1e-15)
    close("final_saved_indicator_matches_independent1011", abs(records[-1]["energy_indicator_Ha"]
          - replay["actual_original_metric_energy_indicator_Ha"]), 1e-20)
    close("parent160_residual_match", abs(records[0]["orbital_residual"]
          - parent_replay["actual_original_metric_orbital_residual"]), 1e-15)
    endpoint_by_label = {row["label"]: row for row in replay["DC_lower_upper_rows"]}
    for label in target_order:
        close("final_selected_energy_matches_independent1011:" + label,
              abs(records[-1]["target_energies_Ha"][label] - endpoint_by_label[label]["energy_Ha"]), 5e-9)
    check("original_failed_stationarity_preserved", not records[-1]["original_stationarity_both_gates"]
          and records[-1]["orbital_residual"] > 1e-7 and records[-1]["energy_indicator_Ha"] < 1e-8
          and replay["original_stationarity_gates_passed"] is False)
    for name, expected in inputs.items():
        check("after_hash:" + name, sha(ROOT / name) == expected)
    check("zero_Atomx_imports", not any(name == "atomx" or name.startswith("atomx.") for name in sys.modules))
    signal.alarm(0)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    elapsed = time.monotonic() - start
    check("actual_saved_audit_caps", elapsed < caps["wall_cap_s"] and peak_rss < 1024
          and sorted(os.sched_getaffinity(0)) == caps["cpu_affinity"])
    summary = {"segment_initial_residual": records[0]["orbital_residual"],
               "segment_final_residual": records[-1]["orbital_residual"],
               "residual_reduction_factor": records[0]["orbital_residual"] / records[-1]["orbital_residual"],
               "minimum_residual": min(r["orbital_residual"] for r in records),
               "minimum_residual_at_segment_iteration": min(records, key=lambda r: r["orbital_residual"])["segment_iteration"],
               "residual_increasing_adjacent_steps": sum(b["orbital_residual"] > a["orbital_residual"]
                                                        for a, b in zip(records, records[1:], strict=False)),
               "accepted_points_passing_original_both_stationarity_gates": sum(r["original_stationarity_both_gates"]
                                                                              for r in records),
               "initial_energy_indicator_Ha": records[0]["energy_indicator_Ha"],
               "final_energy_indicator_Ha": records[-1]["energy_indicator_Ha"],
               "maximum_original_pg_replay_error": max_pg_replay,
               "maximum_saved_residual_replay_error": max_residual_replay,
               "maximum_saved_indicator_replay_error_Ha": max_indicator_replay,
               "maximum_direct_original_Px_wave_error": max_wave_replay,
               "all_saved_radial_waves_bitwise_equal_direct_Px": all_wave_exact,
               "maximum_selected_saved_CI_residual_Ha": max_selected_eigenpair_residual,
               "maximum_selected_symmetry_support_leakage_norm": max_target_support_leakage,
               "float64_energy_spacing_Ha": float(spacing[-1]),
               "whole_segment_energy": history_summary(0), "last20_energy": history_summary(140),
               "last10_energy": history_summary(150), "last_step_energy": history_summary(159),
               "final_CI_coefficient_change_from_initial_norm": dict(zip(
                   target_order, records[-1]["phase_aligned_CI_coefficient_change_from_initial_norm"], strict=True)),
               "maximum_phase_aligned_adjacent_CI_coefficient_change_norm": dict(zip(target_order,
                   np.max([r["phase_aligned_adjacent_CI_coefficient_change_norm"] for r in records[1:]],
                          axis=0).tolist(), strict=True)),
               "selected_global_column_changes": sum(r["selected_global_column_changes_from_previous"] or 0 for r in records),
               "frozen_reference_overlap_squared_initial": dict(zip(target_order, tracking_initial, strict=True)),
               "frozen_reference_overlap_squared_final": records[-1]["physical_frozen_reference_overlap_squared"],
               "CI_response_interpretation": "Signed coefficient changes in the moving orbital basis; not independent physical cross-orbital overlaps. Physical overlap probabilities are the saved original-reference tracker records, hash-bound but not recomputed here.",
               "near_roundoff_interpretation": "Accepted-only recorded energies resolve their observed float64 differences; no rejected trials, Armijo margins, exact mixed-CI energy delta oracle or failure causality is available. Maximum-iteration stop is observed, no line-search failure is established."}
    report = {"all_saved_audit_checks_passed": True, "checks_count": len(checks),
              "source_map_sha256": fixed["source_map_sha256"], "original_metric_sha256": legacy_metric,
              "objective_sha256": fixed["objective_sha256"], "accepted_points": 161,
              "additional_accepted_steps": 160, "total_accepted_steps_across_two_fresh_history_segments": 320,
              "full_gradient_calls_in_completed_segment": {"baseline": 2, "optimizer": 161, "total": 163},
              "energy_trial_calls_in_completed_segment": 252,
              "summary": summary, "records": records,
              "original_orbital_gate_passed": False, "original_energy_indicator_gate_passed": True,
              "matched_point_full_g_pg_replay_gates_passed": True,
              "termination_reason": "maximum_iterations", "line_search_fallbacks": 0,
              "stationarity_certified": False, "production_certified": False,
              "experimental_gate_passed": False, "highest_level_reached": False,
              "static320_spectrum_calculated": False,
              "consumed_1011_endpoint_proof_sha256": sha(RUN / "independent_endpoint_verification.json"),
              "contract_sha256": sha(HERE / "contract.json"),
              "execution_contract_sha256": sha(HERE / "execution_contract.json"),
              "inputs_manifest_sha256": sha(HERE / "input_file_sha256.json"),
              "audit_script_sha256": sha(Path(__file__)),
              "resources": {"actual_affinity": sorted(os.sched_getaffinity(0)),
                            "address_space_cap_bytes": caps["address_space_cap_bytes"],
                            "wall_cap_s": caps["wall_cap_s"], "seconds_uncontrolled": elapsed,
                            "peak_RSS_MiB": peak_rss, "controlled_performance_claim": False},
              "new_physical_calls": {"H_builds": 0, "CI_solves": 0, "gradients": 0, "optimizer_calls": 0}}
    save(HERE / "trajectory_report.json", report)
    verification = {"all_passed": True, "checks_count": len(checks), "checks": checks,
                    "trajectory_report_sha256": sha(HERE / "trajectory_report.json"),
                    "contract_sha256": sha(HERE / "contract.json"),
                    "execution_contract_sha256": sha(HERE / "execution_contract.json"),
                    "new_physical_calls": report["new_physical_calls"], "resources": report["resources"]}
    save(HERE / "verification.json", verification)
    print(json.dumps({"checks_count": len(checks), "summary": summary, "resources": report["resources"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        save(HERE / "failure.json", {"error": str(error), "traceback": traceback.format_exc(),
                                    "physical_calls_authorized": 0,
                                    "failed_script_sha256": sha(Path(__file__))})
        raise
