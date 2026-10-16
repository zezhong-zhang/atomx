"""Independent saved complete screened CI/H/spectrum algebra; zero new solves."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import resource
import signal
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    audit = load(HERE / "contract.json")
    binding_path = HERE / audit["required_completed_input_binding"]
    if not binding_path.exists():
        raise ValueError("Pending actual result; no saved-array numerical audit authorized")
    binding = load(binding_path)
    if (binding.get("completed_inputs_authorized_by_root") is not True
            or binding.get("pending_actual_result") is not False
            or binding.get("audit_contract_sha256") != sha(HERE / "contract.json")
            or binding.get("audit_driver_sha256") != sha(__file__)
            or binding.get("scientific_result_sha256") != sha(RUN / "result.json")):
        raise ValueError("Missing/mismatched completed actual result binding")
    for name, expected in binding["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen completed actual virtual array changed: " + name)
    for name, expected in audit["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen saved virtual evidence changed: " + name)
    if (HERE / "verification.json").exists():
        raise FileExistsError("Preserve completed independent virtual audit")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 3:
        raise RuntimeError("Require original affinity >=3")
    os.sched_setaffinity(0, {affinity[-2]})
    cap = audit["resources"]["address_space_GiB"] * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    def timeout(*_):
        raise TimeoutError("Saved virtual array audit wall cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(audit["resources"]["wall_s"])
    import numpy as np

    checks, outputs = [], []

    def check(name, error, limit=0):
        error = float(error)
        passed = bool(np.isfinite(error) and error <= limit)
        checks.append(dict(name=name, error=error, limit=limit, passed=passed))
        if not passed:
            (HERE / "failure.json").write_text(json.dumps(dict(error=name, checks=checks, new_scientific_calls=0), indent=2) + "\n")
            raise ValueError("Independent screened saved-array check failed: " + name)

    contract, result = load(RUN / "contract.json"), load(RUN / "result.json")
    gates = contract["acceptance"]
    roles = {name: ROOT / row["path"] for name, row in contract["input_roles"].items()}
    checkpoint = np.load(roles["checkpoint_arrays"], allow_pickle=False)
    old = np.load(roles["original_geometry_arrays"], allow_pickle=False)
    layout = load(roles["original_layout"])
    point = load(roles["checkpoint_manifest"])
    basis = load(roles["basis_identity"])
    specs = load(RUN / "models.json")["models"]
    saved_rows = load(RUN / "basis_rows.json")
    sectors = {tuple(row) for row in contract["fixed_conditions"]["sectors"]}
    check("source_epoch10d481", result["source_map_sha256"] != contract["fixed_conditions"]["source_map_sha256"])
    check("original160_point", result["checkpoint_manifest_sha256"] != sha(roles["checkpoint_manifest"]))
    check("exact_contract", result["contract_sha256"] != sha(RUN / "contract.json"))
    check("exact_driver", result["driver_sha256"] != sha(RUN / "run.py"))
    check("root_execution_binding", result["execution_binding_sha256"] != sha(RUN / "execution_binding.json"))
    check("runtime_all_numeric_gates_passed", result["numeric_identity_gates_passed"] is not True or not all(r["passed"] for r in result["checks"]))
    calls = result["calls"]
    two = ("attempted_complete_CI", "completed_complete_CI", "implicit_H_attempts", "completed_implicit_H",
           "explicit_H_attempts", "completed_explicit_H")
    four = ("complement_SVD_attempts", "completed_complement_SVD", "full_gvd_attempts", "completed_full_gvd",
            "complete_spectrum_refinement_attempts", "completed_complete_spectrum_refinements", "near_identity_Gram_whitenings")
    variable = {"internal_CI_onebody_attempts", "completed_internal_CI_onebody_matrices", "attempted_complete_sweeps",
                "completed_complete_sweeps", "attempted_pair_rotations", "completed_pair_rotations", "skipped_exact_zero_pairs",
                "physical_residual_evaluations"}
    for name, count in calls.items():
        if name not in variable:
            expected = 2 if name in two else 4 if name in four else 5 if name == "virtual_control_hS_consumed" else 1 if name in (
                "reused_qualified_complete_spectra", "ConvolCache_constructions") else 0
            check("actual_count/" + name, count != expected)
    check("raw_internal_CI_hS_forwarded_count", not (0 < calls["internal_CI_onebody_attempts"] <= 28
          and calls["internal_CI_onebody_attempts"] == calls["completed_internal_CI_onebody_matrices"]))
    check("complete_sweep_attempts_completed", not (4 <= calls["attempted_complete_sweeps"] <= 48
          and calls["attempted_complete_sweeps"] == calls["completed_complete_sweeps"]))
    check("pair_budget_and_exactzero_accounting", calls["attempted_pair_rotations"] > 310560 or
          calls["attempted_pair_rotations"] != calls["completed_pair_rotations"] + calls["skipped_exact_zero_pairs"])
    check("full_residual_evaluation_budget", not (calls["physical_residual_evaluations"] ==
          calls["completed_complete_sweeps"] + 4 <= 52))
    check("qualified_first_reused_scope", result["qualified_first_kappa_reused_without_refinement"] is not True or
          result["external_geometry_hS_builders"] != 0 or result["shared_source_353_epoch_unchanged"] is not True)
    check("helper_exact_result_binding", result["numerical_refinement_helper_sha256"] != sha(RUN / "refine_spectrum.py"))
    closure = load(RUN / "root_execution.json")
    check("foreground_completed_once", closure["exit_code"] != 0 or closure["outer_wall_cap_stopped"])
    for stage in result["stages"]:
        if "path" in stage and "sha256" in stage:
            check("all_stage_bytes/" + stage["path"], sha(RUN / stage["path"]) != stage["sha256"])
    for name in ("original_stationarity_passed", "expanded_stationarity_evaluated", "experimental_gate_passed",
                 "full_spectrum_or_globalground_certified", "QED_recoil_dynamic_or_static_Breit_added",
                 "performance_comparison_valid", "original_tracker_rebased"):
        check("claim_boundary/" + name, result[name] is not False)
    check("original24_bytes_claim", result["old24_wave_bytes_preserved"] is not True)
    check("old_residual_failure", result["inherited_orbital_residual"] != point["diagnostics"]["orbital_residual"]
          or result["inherited_orbital_residual"] <= 1e-7)
    def signature(row):
        return json.dumps([row["subshells"], row["coupling"], row["twoJ"], row["parity"]], separators=(",", ":"))
    check("baseline_original_signed531", saved_rows["baseline"]["full_signatures"] != [signature(r) for r in basis["CSFs"]])
    constant_tree = ast.parse((ROOT / contract["fixed_conditions"]["frozen_source_root"] / "atomx/constants.py").read_text())
    literals = {n.targets[0].id: ast.literal_eval(n.value) for n in constant_tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("CODATA", "DEFAULT_CODATA_VERSION")}
    codata = literals["CODATA"][literals["DEFAULT_CODATA_VERSION"]]
    cm1 = codata["_Ha_J"] / (100 * codata["_hplanck"] * codata["_c"])
    data, row_by_label = {}, {}
    ci_stages = [stage for stage in result["stages"] if "CI_path" in stage]
    check("exact_two_complete_CI_stage_names", [stage["name"] for stage in ci_stages] != ["baseline", "balanced_virtual"])
    for stage in ci_stages:
        name, model = stage["name"], specs[stage["name"]]
        for path, expected in ((stage["CI_path"], stage["CI_sha256"]), (stage["H_path"], stage["H_sha256"]),
                               (stage["tracking_path"], stage["tracking_sha256"])):
            check(name + "/archive_SHA/" + path, sha(RUN / path) != expected)
        ci = np.load(RUN / stage["CI_path"], allow_pickle=False)
        h = np.load(RUN / stage["H_path"], allow_pickle=False)["matrix_Ha"]
        energies, mixing = ci["energies_Ha"], ci["mixing"]
        declarations = saved_rows[name]
        full_sig, full_sector = declarations["full_signatures"], declarations["full_sectors"]
        selected = [i for i, sector in enumerate(full_sector) if tuple(sector) in sectors]
        signatures = [full_sig[i] for i in selected]
        row_sectors = [tuple(full_sector[i]) for i in selected]
        dimension = model["complete_target_CSFs"]
        check(name + "/complete_full_selected_counts", len(full_sig) != model["full_CSFs"] or len(selected) != dimension)
        check(name + "/complete_sector_subset", selected != declarations["retained_indices"]
              or signatures != declarations["retained_signatures"])
        check(name + "/selected_signature_uniqueness", len(set(signatures)) != dimension)
        check(name + "/signed_basis_SHA", hashlib.sha256(json.dumps(signatures).encode()).hexdigest() != stage["signed_basis_sha256"])
        check(name + "/selected_array_indices", not np.array_equal(ci["indices_in_own_full_basis"], selected))
        expected_keys = checkpoint["orbital_keys"] if name == "baseline" else np.array(sorted(
            list(map(tuple, checkpoint["orbital_keys"])) + [tuple(key) for key in contract["fixed_conditions"]["new_virtuals"]["new_keys"]]))
        check(name + "/complete_actual_radialkeys", not np.array_equal(ci["radial_keys"], expected_keys))
        counts = Counter(row_sectors)
        check(name + "/all_five_complete_counts", counts != {
            (r["twoJ"], r["parity"]): r["CSFs"] for r in model["complete_target_blocks"]})
        check(name + "/matrix_spectrum_shapes", h.shape != (dimension, dimension) or mixing.shape != h.shape
              or energies.shape != (dimension,))
        check(name + "/all_finite", any(not np.isfinite(a).all() for a in (h, energies, mixing)))
        check(name + "/Hermitian", np.max(abs(h-h.T)), gates["matrix_hermiticity_Ha"])
        check(name + "/all_eigenpair_residuals", np.max(np.linalg.norm(h @ mixing-mixing*energies, axis=0)), gates["CI_residual_Ha"])
        check(name + "/whole_CI_Gram", np.max(abs(mixing.T @ mixing-np.eye(dimension))), gates["CI_orthogonality"])
        rows = {sector: [i for i, s in enumerate(row_sectors) if s == sector] for sector in sectors}
        blocks = {sector: [] for sector in sectors}
        ordered_sectors = sorted(sectors)
        support = np.array([np.sum(mixing[rows[sector], :]**2, axis=0) for sector in ordered_sectors])
        assigned = np.argmax(support, axis=0)
        maximum = support[assigned, np.arange(dimension)]
        check(name + "/all_complete_column_support", np.max(abs(maximum - 1)), gates["CI_orthogonality"])
        check(name + "/all_complete_other_sector_support", np.max(np.sum(support, axis=0) - maximum), gates["CI_orthogonality"])
        for sector_index, sector in enumerate(ordered_sectors):
            blocks[sector] = list(map(int, np.flatnonzero(assigned == sector_index)))
        for sector, where in rows.items():
            blocks[sector].sort(key=lambda col: (energies[col], col))
            check(name + "/whole_block_spectrum_dimension/" + str(sector), len(blocks[sector]) != len(where))
            outside = [i for i in range(dimension) if i not in where]
            check(name + "/cross_sector_exact/" + str(sector), np.max(abs(h[np.ix_(where, outside)])),
                  gates["cross_symmetry_zero_Ha"])
        tracking = load(RUN / stage["tracking_path"])
        check(name + "/immutable_reference", tracking["original_reference_changed"] is not False)
        check(name + "/original_target_reference_levels", [r["reference_levels"] for r in tracking["physical_tracking"]]
              != [r["reference_levels"] for r in point["physical_tracking"]])
        weights, selected_columns = np.zeros(dimension), []
        by_label = {row["label"]: row for row in stage["rows"]}
        lower = by_label["6s_1/2"]["energy_Ha"]
        for row in stage["rows"]:
            sector = row["twoJ"], row["parity"]
            col = blocks[sector][row["local_root"]]
            check(name + "/physical_column/" + row["label"], col != row["column"])
            check(name + "/selected_energy/" + row["label"], abs(float(energies[col])-row["energy_Ha"]))
            weights[col] = .2
            selected_columns.append(col)
            group = next(g for g in tracking["physical_tracking"] if tuple(g["reference_levels"][0][:2]) == sector)
            check(name + "/selected_level/" + row["label"], group["selected_levels"] != [[*sector, row["local_root"]]])
            principal = group["principal_overlap_squared"]
            check(name + "/physical_majority/" + row["label"], min(principal) <= .5 or max(principal) > 1+1e-8)
            check(name + "/saved_principal_replay/" + row["label"], principal != row["original_reference_principal_squared"])
            cluster = [i for i in blocks[sector] if abs(energies[i]-energies[col]) < 1e-10]
            check(name + "/complete_equal_weight_degenerate_cluster/" + row["label"], any(weights[i] != .2 for i in cluster))
            excitation = (energies[col]-lower)*cm1
            check(name + "/upper_lower_excitation/" + row["label"], abs(excitation-row["excitation_cm1"]), 1e-7)
            check(name + "/nominal_gap/" + row["label"], abs(excitation-row["NIST_nominal_cm1"]-row["nominal_gap_cm1"]), 1e-7)
        check(name + "/five_equal_weights", not np.array_equal(weights, tracking["state_weights"]))
        check(name + "/EOL_replay", abs(float(weights @ energies)-stage["EOL_Ha"]), 5e-9)
        for multiplet, a, b, g in (("5d", "5d_3/2", "5d_5/2", (4, 6)), ("6p", "6p_1/2", "6p_3/2", (2, 4))):
            report = next(row for row in stage["multiplets"] if row["label"] == multiplet)
            centroid = (g[0]*by_label[a]["excitation_cm1"]+g[1]*by_label[b]["excitation_cm1"])/sum(g)
            splitting = by_label[b]["excitation_cm1"]-by_label[a]["excitation_cm1"]
            check(name + "/centroid/" + multiplet, abs(centroid-report["centroid_cm1"]), 1e-7)
            check(name + "/splitting/" + multiplet, abs(splitting-report["splitting_cm1"]), 1e-7)
        rms = float(np.sqrt(np.mean([r["nominal_gap_cm1"]**2 for r in stage["rows"] if r["nominal_relative_gap"] is not None])))
        check(name + "/RMS_gap", abs(rms-stage["nominal_four_gap_RMS_cm1"]), 1e-7)
        data[name], row_by_label[name] = (h, energies, mixing, signatures, blocks, rows, selected_columns), by_label
    hb, eb, cb, sb, bb, rb, _ = data["baseline"]
    hr, er, cr, sr, br, rr, _ = data["balanced_virtual"]
    mapping = [sr.index(sig) for sig in sb]
    check("signed_old261_mapping", mapping != ci_stages[1]["nested_old261_indices"])
    check("whole_signed_nested261_H", np.max(abs(hr[np.ix_(mapping, mapping)]-hb)), gates["nested_signed_old261_principal_H_error_Ha"])
    baseline_indices = saved_rows["baseline"]["retained_indices"]
    check("baseline_same160_H", np.max(abs(hb-checkpoint["matrix_Ha"][np.ix_(baseline_indices, baseline_indices)])),
          gates["baseline_saved_H_replay_Ha"])
    for sector in sectors:
        check("all_ordered_nested_Ritz_roots/" + str(sector), max(0., float(np.max(
              er[br[sector]][:len(bb[sector])]-eb[bb[sector]]))), gates["nested_ordered_Ritz_eigenvalue_increase_Ha"])
    lower_delta = row_by_label["balanced_virtual"]["6s_1/2"]["energy_Ha"]-row_by_label["baseline"]["6s_1/2"]["energy_Ha"]
    for label, row in row_by_label["balanced_virtual"].items():
        oldrow = row_by_label["baseline"][label]
        upper_delta = row["energy_Ha"]-oldrow["energy_Ha"]
        response = (upper_delta-lower_delta)*cm1
        check("individual_upper_delta/" + label, abs(upper_delta-row["absolute_delta_upper_Ha"]), 1e-12)
        check("individual_lower_delta/" + label, abs(lower_delta-row["absolute_delta_lower_Ha"]), 1e-12)
        check("transition_response_upper_minus_lower/" + label,
              abs(response-row["transition_delta_upper_minus_lower_cm1"]), 1e-7)
        left, right = cb[:, oldrow["column"]], cr[mapping, row["column"]]
        signed_overlap = float(left @ right)
        outputs.append(dict(label=label, upper_delta_Ha=upper_delta, lower_delta_Ha=lower_delta,
                            transition_delta_cm1=response, same160_signed_coefficient_overlap=signed_overlap,
                            same160_embedded_baseline_principal_squared=signed_overlap**2))
    geometry = np.load(RUN / "native_hS_original_geometry.npz", allow_pickle=False)
    canonical = np.load(RUN / "canonical29_orbitals.npz", allow_pickle=False)
    workspace = np.load(roles["geometry_workspace"], allow_pickle=False)
    old_keys = list(map(tuple, checkpoint["orbital_keys"]))
    new_keys = [tuple(key) for key in contract["fixed_conditions"]["new_virtuals"]["new_keys"]]
    keys29 = list(map(tuple, canonical["radial_keys"]))
    check("canonical_complete29keys", keys29 != sorted(old_keys + new_keys))
    for name in ("original_x", "inverse_diagonal", "old24_P", "old24_Q", "old24_keys"):
        original_name = {"original_x": "x", "old24_P": "P", "old24_Q": "Q", "old24_keys": "orbital_keys"}.get(name, name)
        check("native_geometry_byte_identity/" + name, not np.array_equal(geometry[name], checkpoint[original_name]))
    check("canonical_x_unchanged", not np.array_equal(canonical["original_x"], checkpoint["x"]))
    for i, key in enumerate(old_keys):
        position = keys29.index(key)
        check("canonical_old24_PQ_bytes/" + str(key), not (np.array_equal(canonical["P"][position], checkpoint["P"][i])
              and np.array_equal(canonical["Q"][position], checkpoint["Q"][i])))
    for i in range(len(layout)):
        for name in ("projector_", "inverse_blocks_"):
            check("archived_original_metric_bytes/" + name + str(i), not np.array_equal(geometry[name + str(i)], checkpoint[name + str(i)]))
    ns = int(workspace["basis_ns"])
    weights = workspace["basis_grw"].astype(np.longdouble).ravel()

    def local_samples(component):
        values = workspace["basis_bsp_" + component].astype(np.longdouble)
        count = int(workspace["basis_ns" + component])
        grid = np.zeros((values.shape[0], values.shape[1], ns), dtype=np.longdouble)
        for interval in range(values.shape[0]):
            width = min(values.shape[2], max(0, count - interval))
            if width:
                grid[interval, :, interval:interval + width] = values[interval, :, :width]
        return grid.reshape(-1, ns)

    bp, bq = local_samples("p"), local_samples("q")
    independent_s = np.zeros((2 * ns, 2 * ns), dtype=np.longdouble)
    independent_s[:ns, :ns] = bp.T @ (weights[:, None] * bp)
    independent_s[ns:, ns:] = bq.T @ (weights[:, None] * bq)
    for kappa in geometry["kappas"]:
        h, metric = geometry["h_" + str(kappa)], geometry["s_" + str(kappa)]
        check("all7_native_h_finite_symmetric/" + str(kappa), not np.isfinite(h).all() or not np.array_equal(h, h.T))
        check("all7_S_independent_grid_quadrature/" + str(kappa), np.max(abs(metric.astype(np.longdouble)-independent_s)), 1e-10)
    controls = load(RUN / "virtual_construction_controls.json")
    check("all5_saved_minimum_controls", len(controls) != 5 or [tuple(row["key"]) for row in controls] != new_keys)
    check("control_JSON_SHA", sha(RUN / "virtual_construction_controls.json") != result["virtual_control_archive_sha256"])
    screened_outputs = []
    completed_sweeps, attempted_pairs = 0, 0
    original_fields = contract["saved_operator_roles"]
    for j, (key, report) in enumerate(zip(new_keys, controls, strict=True)):
        archive = np.load(RUN / report["path"], allow_pickle=False)
        field = np.load(ROOT / original_fields["field_" + str(j)]["path"], allow_pickle=False)
        check("virtual_control_SHA/" + str(key), sha(RUN / report["path"]) != report["sha256"])
        h, metric, projector, frozen, complement, b = [archive[name] for name in ("h", "s", "original_P", "frozen", "complement", "basis")]
        dimension = [116, 116, 114, 113, 114][j]
        index = next(i for i, row in enumerate(layout) if row["keys"][0][1] == key[1])
        check("virtual_original_P_bytes/" + str(key), not np.array_equal(projector, checkpoint["projector_" + str(index)]))
        check("virtual_exact_saved_screened_field/" + str(key), not (np.array_equal(h, field["auxiliary_Fock_Ha"])
              and np.array_equal(metric, geometry["s_" + str(key[1])]) and
              np.array_equal(field["nuclear_h_Ha"], geometry["h_" + str(key[1])]) and
              np.array_equal(h, field["nuclear_h_Ha"] + field["JK_Ha"])))
        if j == 0:
            prior = np.load(ROOT / original_fields["qualified_first_arrays"]["path"], allow_pickle=False)
            check("first116_reused_physical_and_coeff_bytes", not (np.array_equal(prior["normalized_physical_vectors"],
                  archive["normalized_physical_vectors"]) and np.array_equal(prior["full_coefficients"], archive["full_coefficients"])))
            check("first116_reused_reporting_boundary", report["reported_energy_mode"] != "reused actual physical Rayleigh"
                  or report["refinement_sweeps"] != [])
        else:
            pre = np.load(RUN / ("virtual_" + str(j) + "_native_complement.npz"), allow_pickle=False)
            initial = np.load(RUN / ("virtual_" + str(j) + "_initial_complete_representation.npz"), allow_pickle=False)
            gvd = np.load(RUN / ("virtual_" + str(j) + "_initial_full_gvd.npz"), allow_pickle=False)
            for name in ("h", "s", "original_P", "frozen", "complement", "basis", "projected_h", "projected_s"):
                check("pre_refinement_archive_identity/" + str(key) + "/" + name, not np.array_equal(pre[name], archive[name]))
            check("complete_initial_gvd_consumed/" + str(key), not np.array_equal(initial["initial_complete_coefficients"],
                  gvd["full_coefficients"].astype(np.longdouble)))
            sweep_rows = report["refinement_sweeps"]
            count = len(sweep_rows)
            check("bounded_complete_sweeps/" + str(key), not (1 <= count <= 12))
            check("every_sweep_all_roots/" + str(key), any(row["full_root_count"] != dimension for row in sweep_rows))
            completed_sweeps += count
            attempted_pairs += count * dimension * (dimension - 1) // 2
            last = np.load(RUN / ("virtual_" + str(j) + "_complete_sweep_" + str(count).zfill(2) + ".npz"), allow_pickle=False)
            rotation = last["full_rotations"].astype(np.longdouble)
            order = np.argsort(last["actual_Rayleigh_Ha"], kind="stable")
            check("complete_sorted_final_physical/" + str(key), not np.array_equal(last["normalized_physical_vectors"][:, order],
                  archive["normalized_physical_vectors"]))
            check("complete_sorted_final_coefficients/" + str(key), not np.array_equal(last["coefficients"][:, order],
                  archive["full_coefficients"]))
            check("full_rotation_Gram/" + str(key), np.max(abs(rotation.T @ rotation - np.eye(dimension))), 1e-10)
            congruent = rotation.T @ initial["initial_reduced_operator"] @ rotation
            scale = max(1., float(np.max(np.sum(abs(congruent), axis=1))))
            check("whole_rotated_matrix_congruence/" + str(key), np.max(np.sum(abs(congruent-last["jacobi_matrix"]), axis=1))/scale, 1e-12)
            check("full_coefficient_rotation/" + str(key), np.max(abs(initial["coefficient_basis"] @ rotation-last["coefficients"])), 1e-14)
            check("Jacobi_diagonal_reporting/" + str(key), not np.array_equal(np.diag(last["jacobi_matrix"])[order],
                  archive["reported_energies_Ha"]) or report["reported_energy_mode"] != "refined Jacobi coordinate diagonal")
            initial_vectors = np.asarray(initial["basis"] @ initial["initial_complete_coefficients"], dtype=float).astype(np.longdouble)
            initial_norm = np.sum(initial_vectors * (metric.astype(np.longdouble) @ initial_vectors), axis=0)
            initial_vectors /= np.sqrt(initial_norm)
            signed_complete_span = initial_vectors.T @ metric.astype(np.longdouble) @ archive["normalized_physical_vectors"].astype(np.longdouble)
            check("all_signed_seed_span_retained/" + str(key), np.max(abs(signed_complete_span.T @ signed_complete_span - np.eye(dimension))), 1e-10)
        check("complete_complement_dimension/" + str(key), complement.shape[1] != dimension or dimension != projector.shape[1]-frozen.shape[1])
        check("complete_complement_coordinate_Gram/" + str(key), np.max(abs(complement.T @ complement-np.eye(dimension))), 1e-10)
        check("frozen_orthogonal_complete_complement/" + str(key), np.max(abs(frozen.T @ metric @ projector @ complement)), 1e-10)
        check("complete_basis_congruence/" + str(key), np.max(abs(b-projector @ complement)), 1e-12)
        check("projected_H_congruence/" + str(key), np.max(abs(archive["projected_h"]-b.T @ h @ b)), 1e-10)
        check("projected_S_congruence/" + str(key), np.max(abs(archive["projected_s"]-b.T @ metric @ b)), 1e-10)
        physical = archive["normalized_physical_vectors"]
        hld, sld, bld, vld = [array.astype(np.longdouble) for array in (h, metric, b, physical)]
        mapped = np.asarray(bld @ archive["full_coefficients"].astype(np.longdouble), dtype=float)
        mld = mapped.astype(np.longdouble)
        factors = np.sqrt(np.sum(mld * (sld @ mld), axis=0))
        reconstructed = mapped / np.asarray(factors, dtype=float)
        check("truthful_all_coefficient_normalization_factors/" + str(key), np.max(abs(factors-archive["normalization_factors"])), 1e-14)
        check("all_normalized_physical_reconstruction/" + str(key), np.max(abs(physical-reconstructed)), 1e-14)
        hv, sv = hld @ vld, sld @ vld
        norms = np.sum(vld * sv, axis=0)
        rq = np.sum(vld * hv, axis=0)/norms
        residual = bld.T @ (hv-sv*rq)
        absolute = np.sqrt(np.sum(residual**2, axis=0))
        relative = absolute / np.maximum(1., np.maximum(abs(rq), np.sqrt(np.sum((bld.T @ hv)**2, axis=0))))
        check("all_complete_screened_vector_SGram/" + str(key), np.max(abs(vld.T @ sv-np.eye(dimension))), gates["screened_full_spectrum_SGram_error"])
        check("all_screened_constrained_residual/" + str(key), np.max(relative), gates["screened_full_spectrum_constrained_residual_relative"])
        check("saved_screened_Rayleigh_replay/" + str(key), np.max(abs(rq-archive["actual_Rayleigh_Ha"])), 1e-12)
        check("saved_screened_relative_residual_replay/" + str(key), np.max(abs(relative-archive["full_constrained_residual_relative"])), 1e-14)
        check("all_complete_sorted_roots/" + str(key), np.max(np.maximum(0., -np.diff(rq))))
        raw_vector = archive["raw_original_vector"]
        phase = float(archive["canonical_phase"])
        canonical_vector = np.concatenate((canonical["P"][keys29.index(key)], canonical["Q"][keys29.index(key)]))
        check("raw_is_qualified_lowest_vector/" + str(key), not np.array_equal(raw_vector, physical[:, 0]))
        check("canonical_only_joint_sign/" + str(key), phase not in (-1., 1.) or not np.array_equal(canonical_vector, phase*raw_vector)
              or not np.array_equal(canonical_vector, archive["canonical_raw_vector"]))
        coordinate = projector.T @ metric @ raw_vector
        anchor = int(np.argmax(abs(coordinate)))
        check("originalP_phase_anchor/" + str(key), report["original_P_phase_anchor"] != anchor or
              phase * coordinate[anchor] <= 0)
        raw_ld = canonical_vector.astype(np.longdouble)
        raw_norm = raw_ld @ sld @ raw_ld
        raw_rq = raw_ld @ hld @ raw_ld / raw_norm
        overlap = raw_ld @ sld @ vld[:, 0]
        principal = overlap**2/(raw_norm*norms[0])
        gap = rq[1]-rq[0]
        check("same_original_minimum_gate/" + str(key), abs(raw_rq-rq[0]), gates["screened_complement_minimum_energy_match_Ha"])
        check("original_unique_minimum_cluster/" + str(key), gap <= gates["screened_minimum_cluster_gap_Ha"])
        check("raw_lowest_vector_identity/" + str(key), principal <= gates["screened_raw_lowest_vector_S_overlap_squared_strict_min"])
        pld = projector.astype(np.longdouble)
        missing = raw_ld-pld @ (pld.T @ sld @ raw_ld)
        check("independent_virtual_no_pair_leakage/" + str(key), missing @ sld @ missing, gates["nuclear_projector_leakage_S_norm_squared"])
        same_old = [i for i, old_key in enumerate(old_keys) if old_key[1] == key[1]]
        expected_frozen = np.column_stack([np.concatenate((checkpoint["P"][i], checkpoint["Q"][i])) for i in same_old])
        check("exact_frozen_same_kappa/" + str(key), not np.array_equal(frozen, expected_frozen))
        whole = np.column_stack((frozen, canonical_vector)).astype(np.longdouble)
        check("independent_old_new_constraint_Gram/" + str(key), np.max(abs(whole.T @ independent_s @ whole-np.eye(whole.shape[1]))), gates["actual_radial_Gram_error"])
        raw_res = pld.T @ (hld @ raw_ld-raw_rq*(sld @ raw_ld))
        normal = pld.T @ sld @ frozen.astype(np.longdouble)
        raw_res -= normal @ (normal.T @ raw_res)
        raw_relative = np.sqrt(np.sum(raw_res**2))/max(1.,abs(raw_rq),np.sqrt(np.sum((pld.T @ hld @ raw_ld)**2)))
        check("independent_original_auxiliary_residual/" + str(key), raw_relative, gates["new_spinor_projected_auxiliary_residual_relative"])
        check("saved_minimum_gap_replay/" + str(key), abs(float(gap)-report["minimum_gap_actual_Rayleigh_Ha"]), 1e-12)
        check("saved_raw_minimum_replay/" + str(key), abs(float(raw_rq-rq[0])-report["raw_minus_actual_minimum_Ha"]), 1e-12)
        check("saved_raw_overlap_replay/" + str(key), abs(float(principal)-report["raw_lowest_vector_S_overlap_squared"]), 1e-12)
        screened_outputs.append(dict(key=list(key), complete_roots=dimension, lowest_Rayleigh_Ha=float(raw_rq),
             actual_minimum_gap_Ha=float(gap), raw_lowest_signed_overlap=float(overlap),
             raw_lowest_principal_squared=float(principal), whole_spectrum_relative_residual_max=float(max(relative)),
             lowest_absolute_residual_Ha=float(absolute[0]), reused_first_kappa=(j == 0)))
    check("exact_complete_sweeps_accounted", completed_sweeps != calls["completed_complete_sweeps"])
    check("all_attempted_complete_pairs_accounted", attempted_pairs != calls["attempted_pair_rotations"])
    diagonal = np.load(RUN.parent / "rank_vs_virtual_discriminator/minimum_root_diagnostic/operators.npz", allow_pickle=False)
    check("kappa_minus3_independent_frozen_diagnostic_hS", not (np.array_equal(diagonal["h"], geometry["h_-3"])
          and np.array_equal(diagonal["s"], geometry["s_-3"])))
    fingerprint = hashlib.sha256()
    for i, row in enumerate(layout):
        keys, shape = [tuple(k) for k in row["keys"]], tuple(row["shape"])
        p = checkpoint[f"projector_{i}"]
        check("original_P_bytes/" + str(i), not np.array_equal(p, old[f"projector_{i}"]))
        check("original_inverse_blocks/" + str(i), not np.array_equal(checkpoint[f"inverse_blocks_{i}"], old[f"inverse_blocks_{i}"]))
        fingerprint.update(repr((keys, shape)).encode())
        fingerprint.update(np.ascontiguousarray(p).tobytes())
        wave = p @ checkpoint["x"][slice(*row["slice"])].reshape(shape)
        for j, key in enumerate(keys):
            index = list(map(tuple, checkpoint["orbital_keys"])).index(key)
            check("original_point_wave/" + str(key), not np.array_equal(wave[:, j],
                  np.concatenate((checkpoint["P"][index], checkpoint["Q"][index]))))
    fingerprint.update(checkpoint["inverse_diagonal"].tobytes())
    for i in range(len(layout)):
        for block in checkpoint[f"inverse_blocks_{i}"]:
            fingerprint.update(np.ascontiguousarray(block).tobytes())
    check("original_full_metric_SHA", fingerprint.hexdigest() != result["original_metric_sha256"])
    check("no_Atomx_imported", any(name == "atomx" or name.startswith("atomx.") for name in sys.modules))
    signal.alarm(0)
    proof = dict(all_independent_saved_array_checks_passed=True, checks_count=len(checks), checks=checks,
                 transition_controls=outputs, screened_controls=screened_outputs, bound_input_sha256=audit["inputs_sha256"],
                 completed_input_binding_sha256=sha(binding_path), completed_array_inputs_sha256=binding["inputs_sha256"],
                 audit_contract_sha256=sha(HERE / "contract.json"), audit_driver_sha256=sha(__file__),
                 new_H_CI_Dirac_virtual_gradient_optimizer_eigensolves=0, Atomx_imports=0,
                 original_stationarity_passed=False, expanded_stationarity_evaluated=False,
                 experimental_gate_passed=False, all_spectrum_globalground_certified=False,
                 limits="Archived native complete angular signatures are hash/sector/signed-nesting audited, not independently recontracted. Original physical minor-overlap probabilities are recorded public outputs replayed, not newly recontracted. Same-point signed coefficient overlaps are independently computed. Saved h/S and unchanged no-pair P independently verified; screened spectra/vectors and complete rotations are saved outputs audited algebraically, not recomputed. No new one-body builder/spectral solve.",
                 scope="Same accepted160 old24 finite DC balanced virtual response; lower/upper, centroids and splitting separately. No optimized model, physics completeness, or additive correction claim.")
    (HERE / "verification.json").write_text(json.dumps(proof, indent=2)+"\n")
    print(json.dumps(dict(checks=len(checks), all_passed=True, transitions=outputs)))


if __name__ == "__main__":
    main()
