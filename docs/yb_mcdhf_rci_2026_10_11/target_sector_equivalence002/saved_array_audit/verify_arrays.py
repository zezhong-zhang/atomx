"""Independent saved-array algebra; no Hamiltonian construction or CI solve."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
REPO = RUN.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    audit_contract = load(HERE / "contract.json")
    binding = load(RUN / "execution_binding.json")
    contract = load(RUN / "contract.json")
    result = load(RUN / "result.json")
    if (sha(RUN / "execution_binding.json") != audit_contract["fixed_execution_binding_sha256"]
            or sha(RUN / "contract.json") != audit_contract["preparation_contract_sha256"]
            or sha(RUN / "run.py") != audit_contract["execution_driver_sha256"]
            or result.get("equivalence_gates_passed") is not True):
        raise ValueError("Completed exact root-bound equivalence result missing; no array audit")
    if (HERE / "verification.json").exists():
        raise ValueError("Independent audit already frozen; refuse overwrite")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 3:
        raise ValueError("Three original affinity CPUs required; no silent fallback")
    os.sched_setaffinity(0, {affinity[-2]})
    byte_cap = audit_contract["resources"]["address_space_cap_GiB"] * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (byte_cap, byte_cap))

    def timeout(*_):
        raise TimeoutError("Saved-array-only audit30s resource cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(audit_contract["resources"]["wall_cap_s"])
    import numpy as np

    checks = []

    def check(name, error, limit=0):
        value = float(error)
        passed = bool(np.isfinite(value) and value <= limit)
        checks.append({"name": name, "error": value, "limit": limit, "passed": passed})
        if not passed:
            raise ValueError("Independent saved-array check failed: " + name)

    gates = contract["acceptance"]
    check("result_exact_binding", result["execution_binding_sha256"] != sha(RUN / "execution_binding.json"))
    check("result_exact_contract", result["contract_sha256"] != sha(RUN / "contract.json"))
    check("result_source_epoch", result["source_map_sha256"] != binding["expected_source_map_sha256"])
    check("unchanged_failed_stationarity_scope", result["original_stationarity_already_passed"] is not False)
    check("no_experimental_or_timing_certificate",
          result["experimental_gate_passed"] is not False or result["parallel_or_end_to_end_speedup_claim"] is not False)
    for count in ("attempted_CI_calls", "completed_CI_calls", "attempted_gradient_calls",
                  "completed_gradient_calls", "explicit_H_builds"):
        check("runtime_exact_two_count:" + count, result[count] != 2)
    check("runtime_no_optimizer", result["optimizer_calls"])
    check("runtime_all_gates_pass", not all(row["passed"] for row in result["checks"]))
    for name, expected in contract["predecessor_failed_invocation_sha256"].items():
        check("previous_failure_retained:" + name, sha(REPO / name) != expected)
    files = {role: REPO / entry["path"] for role, entry in binding["files"].items()}
    for role, entry in binding["files"].items():
        check("frozen_input:" + role, sha(files[role]) != entry["sha256"])
    checkpoint = load(files["checkpoint_manifest"])
    proof = load(files["checkpoint_verification"])
    basis = load(files["basis_identity"])
    keys = tuple(tuple(k) for k in contract["fixed_conditions"]["radial_keys"])
    full_rows = basis["CSFs"]
    sectors = {(row["twoJ"], row["parity"])
               for row in contract["fixed_conditions"]["sector_selection"]["sector_counts"]}
    kept = [i for i, row in enumerate(full_rows) if (row["twoJ"], row["parity"]) in sectors]
    check("complete_signed_row_subset", kept != contract["fixed_conditions"]["sector_selection"]["indices_full531"])
    check("five_whole_sectors261", len(kept) != 261 or len(sectors) != 5)
    check("all24_radial_keys_retained", sorted({tuple(s[:2]) for i in kept for s in full_rows[i]["subshells"]}) != list(keys))
    arrays = np.load(RUN / "equivalence_arrays.npz", allow_pickle=False)
    check("result_saved_arrays_hash", sha(RUN / "equivalence_arrays.npz") != result["arrays_sha256"])
    check("saved_complete_CSF_indices", not np.array_equal(arrays["retained_CSF_indices"], kept))
    saved_checkpoint = np.load(files["checkpoint_arrays"], allow_pickle=False)
    old_geometry = np.load(files["original_geometry_arrays"], allow_pickle=False)
    layout = load(files["original_layout"])
    models = []
    for label, rows in (("full531", full_rows), ("five_sectors261", [full_rows[i] for i in kept])):
        evidence = next(item for item in result["completed_model_evidence"] if item["label"] == label)
        for file_field, hash_field in (("CI_gradient_path", "CI_gradient_sha256"),
                                      ("target_diagnostics_path", "target_diagnostics_sha256"),
                                      ("H_path", "H_sha256")):
            check(label + "/completed_evidence:" + file_field,
                  sha(RUN / evidence[file_field]) != evidence[hash_field])
        model_arrays = np.load(RUN / evidence["CI_gradient_path"], allow_pickle=False)
        model_h = np.load(RUN / evidence["H_path"], allow_pickle=False)["matrix_Ha"]
        diagnostics = load(RUN / evidence["target_diagnostics_path"])
        prefix = "full" if label == "full531" else "sector"
        h, e, c = arrays[prefix + "_matrix_Ha"], arrays[prefix + "_energies_Ha"], arrays[prefix + "_mixing"]
        g = arrays[prefix + "_radial_gradient"]
        weights = model_arrays["state_weights"]
        check(label + "/all_finite", any(not np.isfinite(value).all() for value in (h, e, c, g, weights)))
        check(label + "/shapes", h.shape != (len(rows), len(rows)) or c.shape != h.shape
              or e.shape != (len(rows),) or g.shape != (24, 246) or weights.shape != e.shape)
        for field, value in (("energies_Ha", e), ("mixing", c), ("radial_gradient", g)):
            check(label + "/independent_per_model_archive:" + field, not np.array_equal(model_arrays[field], value))
        check(label + "/independent_per_model_H", not np.array_equal(model_h, h))
        check(label + "/full24_radial_key_order", not np.array_equal(model_arrays["radial_keys"], keys))
        check(label + "/exact_point", not np.array_equal(model_arrays["original_x"], saved_checkpoint["x"]))
        check(label + "/Hermiticity", np.max(abs(h - h.T)), gates["matrix_hermiticity_Ha"])
        check(label + "/CI_residual_all_saved_eigenpairs", np.max(np.linalg.norm(h @ c - c * e, axis=0)), gates["CI_residual_Ha"])
        check(label + "/CI_Gram_all_saved_eigenvectors", np.max(abs(c.T @ c - np.eye(len(rows)))), gates["CI_orthogonality"])
        row_blocks = {}
        for i, row in enumerate(rows):
            row_blocks.setdefault((row["twoJ"], row["parity"]), []).append(i)
        column_blocks = {}
        for col in range(len(rows)):
            masses = {sector: float(c[indices, col] @ c[indices, col]) for sector, indices in row_blocks.items()}
            chosen = max(masses, key=masses.get)
            check(label + "/column_single_sector_support:" + str(col), abs(masses[chosen] - 1.), gates["CI_orthogonality"])
            check(label + "/column_unreachable_support:" + str(col), sum(value for sector, value in masses.items() if sector != chosen), gates["CI_orthogonality"])
            column_blocks.setdefault(chosen, []).append(col)
        for sector, indices in row_blocks.items():
            column_blocks[sector] = sorted(column_blocks[sector], key=lambda col: (e[col], col))
            check(label + "/complete_block_dimension:" + str(sector), len(column_blocks[sector]) != len(indices))
            outside = sorted(set(range(len(rows))) - set(indices))
            if outside:
                check(label + "/complete_block_zero_coupling:" + str(sector), np.max(abs(h[np.ix_(indices, outside)])), gates["cross_symmetry_zero_Ha"])
        selected_columns = []
        for group in checkpoint["physical_tracking"]:
            check(label + "/single_original_physical_target_group", len(group["selected_levels"]) != 1)
            j, p, root = group["selected_levels"][0]
            selected_columns.append(column_blocks[j, p][root])
        expected_weights = np.zeros_like(weights)
        expected_weights[selected_columns] = .2
        check(label + "/exact_five_original_weights", not np.array_equal(weights, expected_weights))
        check(label + "/weights_saved_JSON", not np.array_equal(weights, diagnostics["state_weights"]))
        assignments = [row["selected_levels"] for row in diagnostics["physical_tracking"]]
        check(label + "/physical_target_values_order", assignments != [row["selected_levels"] for row in checkpoint["physical_tracking"]])
        energy = float(weights @ e)
        check(label + "/replayed_EOL", abs(energy - diagnostics["EOL_Ha"]), gates["EOL_equivalence_Ha"])
        models.append((h, e, c, g, weights, column_blocks, selected_columns, energy, diagnostics))
    hf, ef, cf, gf, wf, bf, _, eolf, df = models[0]
    hs, es, cs, gs, ws, bs, _, eols, ds = models[1]
    check("same_saved_checkpoint_H", np.max(abs(hf - saved_checkpoint["matrix_Ha"])), gates["saved_full_matrix_replay_Ha"])
    check("whole_sector_principal_H", np.max(abs(hs - hf[np.ix_(kept, kept)])), gates["matrix_full_subset_equivalence_Ha"])
    check("EOL_equivalence", abs(eolf - eols), gates["EOL_equivalence_Ha"])
    targets = []
    for row in result["target_signed_spans"]:
        sector = (row["twoJ"], row["parity"])
        roots = row["local_roots_in_complete_comparison_span"]
        fcols, scols = bf[sector], bs[sector]
        check("complete_sector_spectrum:" + str(sector), np.max(abs(ef[fcols] - es[scols])), gates["target_energy_equivalence_Ha"])
        check("cluster_roots_valid:" + str(sector), not roots or sorted(set(roots)) != roots
              or min(roots) < 0 or max(roots) >= len(fcols))
        for root in roots:
            for other in range(len(fcols)):
                gap = min(abs(ef[fcols[root]] - ef[fcols[other]]), abs(es[scols[root]] - es[scols[other]]))
                if gap <= gates["near_degenerate_cluster_gap_Ha"]:
                    check("complete_near_degenerate_span:" + str((sector, other)), other not in roots)
        left, right = cf[np.ix_(kept, [fcols[r] for r in roots])], cs[:, [scols[r] for r in roots]]
        signed = left.T @ right
        principal = np.linalg.svd(signed, compute_uv=False)**2
        check("signed_coefficient_span_principal:" + str(sector), np.max(abs(principal - 1)), gates["squared_principal_overlap_error"])
        check("complete_span_projector:" + str(sector), np.max(abs(left @ left.T - right @ right.T)), gates["complete_degenerate_projector_error"])
        physical_principal = np.linalg.svd(np.asarray(row["signed_overlap"]), compute_uv=False)**2
        check("recorded_physical_span_singular_replay:" + str(sector), np.max(abs(physical_principal - row["squared_principal_overlaps"])), gates["squared_principal_overlap_error"])
        check("recorded_physical_span_pass:" + str(sector), np.max(abs(physical_principal - 1)), gates["squared_principal_overlap_error"])
        if len(roots) == 1:
            sign = 1. if signed[0, 0] >= 0 else -1.
            check("signed_vector_internal_phases:" + str(sector), np.max(abs(left[:, 0] - sign * right[:, 0])), gates["signed_non_degenerate_vector_max"])
        targets.append({"sector": sector, "roots": roots, "coefficient_signed_overlap": signed.tolist(),
                        "coefficient_principal_squared": principal.tolist(),
                        "physical_principal_squared_saved_replay": physical_principal.tolist()})
    density_f = (cf * wf) @ cf.T
    density_s = (cs * ws) @ cs.T
    check("independent_coherent_CI_density_full_principal", np.max(abs(density_f[np.ix_(kept, kept)] - density_s)), gates["RDM_key_union_max_absolute"])
    omitted = sorted(set(range(531)) - set(kept))
    check("unreachable_CI_density_zero", np.max(abs(density_f[omitted])), gates["RDM_key_union_max_absolute"])
    frdm = {json.dumps(row["term"]): row["weight"] for row in df["coherent_RDM"]}
    srdm = {json.dumps(row["term"]): row["weight"] for row in ds["coherent_RDM"]}
    check("signed_radial_RDM_saved_union", max(abs(frdm.get(term, 0) - srdm.get(term, 0)) for term in set(frdm) | set(srdm)), gates["RDM_key_union_max_absolute"])
    check("full24_gradient_absolute", np.max(abs(gf - gs)), gates["full_radial_gradient_max_absolute"])
    check("full24_gradient_relative", np.max(abs(gf - gs)) / max(float(np.max(abs(gf))), 1e-30), gates["full_radial_gradient_max_relative"])
    x = arrays["original_x"]
    check("authoritative_x_exact", not np.array_equal(x, saved_checkpoint["x"]))
    metric_digest = hashlib.sha256()
    for i, row in enumerate(layout):
        row_keys = [tuple(key) for key in row["keys"]]
        shape = tuple(row["shape"])
        projector = saved_checkpoint[f"projector_{i}"]
        check("original_projector_bitwise:" + str(i), not np.array_equal(projector, old_geometry[f"projector_{i}"]))
        check("original_inverse_blocks_bitwise:" + str(i), not np.array_equal(saved_checkpoint[f"inverse_blocks_{i}"], old_geometry[f"inverse_blocks_{i}"]))
        metric_digest.update(repr((row_keys, shape)).encode())
        metric_digest.update(np.ascontiguousarray(projector).tobytes())
    check("original_inverse_diagonal_bitwise", not np.array_equal(saved_checkpoint["inverse_diagonal"], old_geometry["inverse_diagonal"]))
    metric_digest.update(saved_checkpoint["inverse_diagonal"].tobytes())
    for i in range(len(layout)):
        for block in saved_checkpoint[f"inverse_blocks_{i}"]:
            metric_digest.update(np.ascontiguousarray(block).tobytes())
    check("original_complete_metric_fingerprint", metric_digest.hexdigest() != result["original_metric_sha256"]
          or metric_digest.hexdigest() != contract["fixed_conditions"]["original_metric_sha256"])

    def original_metric(gradient):
        tangent, direction = np.zeros_like(x), np.zeros_like(x)
        for i, row in enumerate(layout):
            where, shape = slice(*row["slice"]), tuple(row["shape"])
            y = x[where].reshape(shape)
            derivative = np.column_stack([gradient[keys.index(tuple(key))] for key in row["keys"]])
            block = saved_checkpoint[f"projector_{i}"].T @ derivative
            tangent[where] = (block - y @ (.5 * (y.T @ block + block.T @ y))).ravel()
            t = tangent[where].reshape(shape)
            z = np.column_stack([b @ t[:, j] for j, b in enumerate(saved_checkpoint[f"inverse_blocks_{i}"])])
            direction[where] = (z - y @ (.5 * (y.T @ z + z.T @ y))).ravel()
        return tangent, direction, float(np.linalg.norm(direction)), .5 * abs(float(tangent @ direction))

    metric_results = []
    for name, gradient in (("full", gf), ("sector", gs)):
        tangent, direction, residual, indicator = original_metric(gradient)
        check(name + "/manual_tangent_replay", np.max(abs(tangent - arrays["original_" + name + "_tangent"])), gates["original_metric_tangent_gradient_max_absolute"])
        check(name + "/manual_preconditioned_replay", np.max(abs(direction - arrays["original_" + name + "_preconditioned"])), gates["original_metric_preconditioned_gradient_max_absolute"])
        check(name + "/manual_residual_replay", abs(residual - result["original_" + name + "_residual"]), gates["original_metric_residual_absolute"])
        check(name + "/manual_indicator_replay", abs(indicator - result["original_" + name + "_energy_indicator_Ha"]), gates["original_metric_energy_indicator_absolute_Ha"])
        check(name + "/stationarity_remains_failed", residual <= 1e-7)
        metric_results.append({"variant": name, "residual": residual, "energy_indicator_Ha": indicator,
                               "residual_gate_passed": residual < 1e-7, "indicator_gate_passed": indicator < 1e-8})
    check("full_sector_original_residual_equivalence", abs(metric_results[0]["residual"] - metric_results[1]["residual"]), gates["original_metric_residual_absolute"])
    check("full_checkpoint_original_residual", abs(metric_results[0]["residual"] - proof["actual_original_metric_orbital_residual"]), gates["original_metric_residual_absolute"])
    check("no_Atomx_source_consumer_imported", any(name == "atomx" or name.startswith("atomx.") for name in sys.modules))
    transitions = []
    lower_f, lower_s = bf[1, 1][0], bs[1, 1][0]
    for j, p in sorted(sectors - {(1, 1)}):
        upper_f, upper_s = bf[j, p][0], bs[j, p][0]
        shift = float((es[upper_s] - es[lower_s]) - (ef[upper_f] - ef[lower_f]))
        check("same_model_upper_lower_line_equivalence:" + str((j, p)), abs(shift), 2 * gates["target_energy_equivalence_Ha"])
        transitions.append({"upper_sector": [j, p], "full_transition_Ha": float(ef[upper_f] - ef[lower_f]),
                            "sector_transition_Ha": float(es[upper_s] - es[lower_s]), "sector_minus_full_transition_Ha": shift})
    signal.alarm(0)
    record = {"all_independent_saved_array_checks_passed": all(row["passed"] for row in checks),
              "checks_count": len(checks), "checks": checks, "targets": targets,
              "original_metric_results": metric_results, "same_model_transition_comparison": transitions,
              "input_result_sha256": sha(RUN / "result.json"), "input_arrays_sha256": sha(RUN / "equivalence_arrays.npz"),
              "execution_binding_sha256": sha(RUN / "execution_binding.json"), "audit_contract_sha256": sha(HERE / "contract.json"),
              "audit_driver_sha256": sha(__file__), "source_map_sha256": result["source_map_sha256"],
              "new_H_builds": 0, "new_CI_diagonalizations": 0, "new_gradient_calls": 0, "optimizer_calls": 0,
              "Atomx_imports": 0, "all_spectrum_certificate": False, "experimental_gate_passed": False,
              "stationarity_gate_passed": False, "expanded_MCDHF_certified": False,
              "physical_overlap_limit": "Independent signed coefficient-span checks use identical frozen radial/CSF basis; public physical minor-overlap result is replayed from saved values, not re-contracted.",
              "S_metric_limit": "Radial S-Gram/positive projector coverage inherited from bound988-gate endpoint and runtimeactualSGram checks; no new Dirac/S matrix construction in this array-only audit.",
              "scope": "Complete target-sector/EOL representation equivalence at one frozen finite-model point only. No new physics, gap improvement, all-spectrum/global-ground or controlled speedup claim."}
    (HERE / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"checks_passed": record["checks_count"], "all_passed": record["all_independent_saved_array_checks_passed"],
                      "new_H_CI_gradient_calls": 0, "metric_results": metric_results}))


if __name__ == "__main__":
    main()
