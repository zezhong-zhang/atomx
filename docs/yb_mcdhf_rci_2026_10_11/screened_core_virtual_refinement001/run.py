"""Prospective complete116 saved-operator Jacobi refinement; no physical builders."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def metadata():
    contract = load(HERE / "contract.json")
    binding = load(HERE / "execution_binding.json")
    expected = dict(contract_sha256=sha(HERE / "contract.json"), driver_sha256=sha(__file__),
                    source_map_sha256=contract["fixed_conditions"]["source_map_sha256"],
                    original_metric_sha256=contract["fixed_conditions"]["original_metric_sha256"],
                    checkpoint_manifest_sha256=contract["input_roles"]["checkpoint_manifest"]["sha256"],
                    saved_audit_sha256=contract["input_roles"]["saved_audit"]["sha256"],
                    iteration_trigger_sha256=contract["input_roles"]["iteration_trigger"]["sha256"])
    for name, value in expected.items():
        if binding.get(name) != value:
            raise ValueError("Exact root execution binding mismatch: " + name)
    if binding.get("root_authorized") is not True:
        raise ValueError("Root has not authorized this exact bounded numerical invocation")
    for name, row in contract["input_roles"].items():
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("Frozen saved input changed: " + name)
    source = load(ROOT / contract["input_roles"]["source_identity"]["path"])
    original = ROOT / contract["fixed_conditions"]["frozen_source_root"]
    if len(source["all_atomx_python_sha256"]) != 353:
        raise ValueError("Require exact original353 source map")
    for name, expected in source["all_atomx_python_sha256"].items():
        if sha(original / "atomx" / name.removeprefix("src/atomx/")) != expected:
            raise ValueError("Original frozen source changed: " + name)
    return contract, binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize", action="store_true", help="Check metadata only; no numerical imports")
    args = parser.parse_args()
    contract, binding = metadata()
    if args.authorize:
        print(json.dumps(dict(exact_root_binding=True, Atomx_imports=0, numerical_imports=0,
                              refinement_calls=0, input_roles=len(contract["input_roles"]))))
        return
    if any((HERE / name).exists() for name in ("result.json", "failure.json", "resource_admission.json")):
        raise FileExistsError("Preserve a single admitted refinement invocation")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 3:
        raise RuntimeError("Require at least3 original affinity CPUs")
    os.sched_setaffinity(0, {affinity[-2]})
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    write(HERE / "resource_admission.json", dict(original_affinity=affinity, selected_cpu=affinity[-2],
          address_space_GiB=4, wall_s=120, before_numerical_import=True,
          root_binding_sha256=sha(HERE / "execution_binding.json")))
    start = time.monotonic()

    def timeout(*_):
        raise TimeoutError("Complete-spectrum refinement120s cap exhausted")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    import numpy as np

    checks, stages, sweep_rows = [], [], []
    calls = dict(complete_spectrum_refinement_attempts=0, completed_complete_spectrum_refinements=0,
                 attempted_complete_sweeps=0, completed_complete_sweeps=0,
                 attempted_pair_rotations=0, completed_pair_rotations=0, skipped_exact_zero_pairs=0,
                 physical_residual_evaluations=0, near_identity_Gram_whitenings=0,
                 new_LAPACK_eigensolves=0, new_SVD=0, new_matrix_linear_solves=0,
                 new_Atomx_Fock_Slater_Dirac_SCF_hS_H_CI_gradient_optimizer_calls=0)
    paths = {name: ROOT / row["path"] for name, row in contract["input_roles"].items()}
    gates, limits = contract["acceptance"], contract["call_budgets"]

    def check(name, error, limit=0):
        error = float(error)
        passed = bool(np.isfinite(error) and error <= limit)
        checks.append(dict(name=name, error=error, limit=limit, passed=passed))
        if not passed:
            raise ValueError("Complete-spectrum refinement gate: " + name)

    def save(name, **arrays):
        path = HERE / (name + ".npz")
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        stages.append(dict(name=name, path=path.name, sha256=sha(path)))

    try:
        check("longdouble_has_more_precision_than_float64", np.finfo(np.longdouble).eps >= np.finfo(float).eps)
        old_failure, audit = load(paths["failed_result"]), load(paths["saved_audit"])
        check("original_failure_and_audit_remain_failed", old_failure["scientific_success_claim"] is not False
              or audit["original_scientific_invocation_passed"] is not False
              or not audit["all_independent_saved_evidence_checks_passed"])
        point = load(paths["checkpoint_manifest"])
        check("original_stationarity_not_passed", point["diagnostics"]["orbital_residual"] <= 1e-7)
        a = np.load(paths["original_control"], allow_pickle=False)
        pre, raw = np.load(paths["complement"], allow_pickle=False), np.load(paths["raw_gvd"], allow_pickle=False)
        field, geometry = np.load(paths["field"], allow_pickle=False), np.load(paths["geometry"], allow_pickle=False)
        checkpoint = np.load(paths["checkpoint_arrays"], allow_pickle=False)
        layout = load(paths["original_layout"])
        for name in ("h", "s", "original_P", "frozen", "complement", "basis", "projected_h", "projected_s"):
            check("original_saved_complement_bytes/" + name, not np.array_equal(a[name], pre[name]))
        check("native_auxiliary_operator_bytes", not np.array_equal(a["h"], field["auxiliary_Fock_Ha"]))
        check("native_S_bytes", not np.array_equal(a["s"], geometry["s_-3"]))
        check("complete_initial116_coefficients_bytes", not np.array_equal(a["full_coefficients"], raw["full_coefficients"]))
        for name, original in (("old24_keys", "orbital_keys"), ("old24_P", "P"), ("old24_Q", "Q"),
                               ("original_x", "x"), ("inverse_diagonal", "inverse_diagonal")):
            check("original_point_bytes/" + name, not np.array_equal(geometry[name], checkpoint[original]))
        fingerprint = hashlib.sha256()
        for i, row in enumerate(layout):
            keys, shape = tuple(tuple(key) for key in row["keys"]), tuple(row["shape"])
            p, blocks = checkpoint["projector_" + str(i)], checkpoint["inverse_blocks_" + str(i)]
            check("original_P_inverse_bytes/" + str(i), not np.array_equal(p, geometry["projector_" + str(i)])
                  or not np.array_equal(blocks, geometry["inverse_blocks_" + str(i)]))
            fingerprint.update(repr((list(keys), shape)).encode())
            fingerprint.update(np.ascontiguousarray(p).tobytes())
        fingerprint.update(checkpoint["inverse_diagonal"].tobytes())
        for i in range(len(layout)):
            for block in checkpoint["inverse_blocks_" + str(i)]:
                fingerprint.update(np.ascontiguousarray(block).tobytes())
        check("original_complete_metric_SHA", fingerprint.hexdigest() != contract["fixed_conditions"]["original_metric_sha256"])
        h, s, b, p, frozen = [a[name].astype(np.longdouble) for name in ("h", "s", "basis", "original_P", "frozen")]
        n = contract["fixed_conditions"]["full_spectrum_size"]
        check("complete116_shape", b.shape != (246, n) or a["full_coefficients"].shape != (n, n))
        check("original_h_exact_Hermitian", np.max(abs(h - h.T)))
        check("original_s_exact_Hermitian", np.max(abs(s - s.T)))
        check("original_P_Gram", np.max(abs(p.T @ s @ p - np.eye(p.shape[1]))), gates["saved_P_metric_Gram_error"])
        check("original_B_Gram", np.max(abs(b.T @ s @ b - np.eye(n))), gates["actual_radial_Gram_error"])
        check("old_frozen_span_orthogonal_to_B", np.max(abs(frozen.T @ s @ b)), gates["actual_radial_Gram_error"])
        projected_h, projected_s = b.T @ h @ b, b.T @ s @ b
        ah, ms = (projected_h + projected_h.T) / 2, (projected_s + projected_s.T) / 2
        initial = a["full_coefficients"].astype(np.longdouble)
        gram = initial.T @ ms @ initial
        defect = (gram + gram.T) / 2 - np.eye(n, dtype=np.longdouble)
        check("complete_basis_near_identity_Gram_inf", np.max(np.sum(abs(defect), axis=1)), 1e-10)
        calls["near_identity_Gram_whitenings"] += 1
        d2, d3 = defect @ defect, defect @ defect @ defect
        d4 = d2 @ d2
        whitening = np.eye(n, dtype=np.longdouble) - defect / 2 + 3 * d2 / 8 - 5 * d3 / 16 + 35 * d4 / 128
        coefficient_basis = initial @ whitening
        check("whitened_actual_Gram", np.max(abs(coefficient_basis.T @ projected_s @ coefficient_basis - np.eye(n))),
              gates["screened_full_spectrum_SGram_error"])
        reduced = coefficient_basis.T @ ah @ coefficient_basis
        reduced = (reduced + reduced.T) / 2
        original_reduced = reduced.copy()
        rotations = np.eye(n, dtype=np.longdouble)
        save("initial_complete_representation", projected_h_longdouble=projected_h, projected_s_longdouble=projected_s,
             symmetric_projected_h_longdouble=ah, symmetric_projected_s_longdouble=ms,
             initial_complete_coefficients=initial, initial_Gram=gram, whitening=whitening,
             coefficient_basis=coefficient_basis, initial_reduced_operator=original_reduced,
             original_h=a["h"], original_s=a["s"], original_P=a["original_P"], basis=a["basis"], frozen=a["frozen"])

        def physical_state():
            calls["physical_residual_evaluations"] += 1
            if calls["physical_residual_evaluations"] > limits["maximum_saved_residual_evaluations"]:
                raise RuntimeError("Predeclared full residual evaluation budget exhausted")
            coeff = coefficient_basis @ rotations
            vectors = np.asarray(b @ coeff, dtype=float)
            vv = vectors.astype(np.longdouble)
            norms = np.sum(vv * (s @ vv), axis=0)
            check("complete_physical_norms_positive/" + str(calls["physical_residual_evaluations"]),
                  np.any(norms <= 0) or np.any(~np.isfinite(norms)))
            vectors /= np.asarray(np.sqrt(norms), dtype=float)
            vv = vectors.astype(np.longdouble)
            hv, sv = h @ vv, s @ vv
            norms_after = np.sum(vv * sv, axis=0)
            rayleigh = np.sum(vv * hv, axis=0) / norms_after
            constrained = b.T @ (hv - sv * rayleigh)
            absolute = np.sqrt(np.sum(constrained**2, axis=0))
            relative = absolute / np.maximum(1., np.maximum(abs(rayleigh), np.sqrt(np.sum((b.T @ hv)**2, axis=0))))
            gram_after = vv.T @ sv
            off = reduced - np.diag(np.diag(reduced))
            off_relative = np.sqrt(np.sum(off**2, axis=0)) / np.maximum(1., abs(np.diag(reduced)))
            return dict(coefficients=coeff, normalized_physical_vectors=vectors, norm_after=norms_after,
                        actual_Rayleigh_Ha=rayleigh, actual_full_SGram=gram_after,
                        full_constrained_residual=constrained, full_constrained_residual_norm_Ha=absolute,
                        full_constrained_residual_relative=relative, reduced_offdiagonal_relative=off_relative)

        before = physical_state()
        save("before_refinement", **before)
        calls["complete_spectrum_refinement_attempts"] += 1
        converged = False
        final = None
        for sweep in range(1, limits["maximum_complete_sweeps"] + 1):
            calls["attempted_complete_sweeps"] += 1
            for i in range(n - 1):
                for j in range(i + 1, n):
                    calls["attempted_pair_rotations"] += 1
                    if calls["attempted_pair_rotations"] > limits["maximum_attempted_pair_rotations"]:
                        raise RuntimeError("Predeclared complete Jacobi pair budget exhausted")
                    cross = reduced[i, j]
                    if cross == 0:
                        calls["skipped_exact_zero_pairs"] += 1
                        continue
                    difference = (reduced[j, j] - reduced[i, i]) / 2
                    t = (np.copysign(np.longdouble(1), cross) if difference == 0 else
                         np.copysign(np.longdouble(1), difference) * cross /
                         (abs(difference) + np.hypot(difference, cross)))
                    cosine = 1 / np.sqrt(1 + t * t)
                    sine = t * cosine
                    if not np.isfinite(t):
                        raise ValueError("Nonfinite Jacobi rotation")
                    left, right = reduced[:, i].copy(), reduced[:, j].copy()
                    ii, jj = reduced[i, i], reduced[j, j]
                    rotated_left, rotated_right = cosine * left - sine * right, sine * left + cosine * right
                    reduced[:, i], reduced[i, :] = rotated_left, rotated_left
                    reduced[:, j], reduced[j, :] = rotated_right, rotated_right
                    reduced[i, i], reduced[j, j] = ii - t * cross, jj + t * cross
                    reduced[i, j], reduced[j, i] = 0, 0
                    left, right = rotations[:, i].copy(), rotations[:, j].copy()
                    rotations[:, i], rotations[:, j] = cosine * left - sine * right, sine * left + cosine * right
                    calls["completed_pair_rotations"] += 1
            calls["completed_complete_sweeps"] += 1
            final = physical_state()
            row = dict(sweep=sweep, full_root_count=n,
                       original_full_physical_residual_relative_max=float(np.max(final["full_constrained_residual_relative"])),
                       full_Gram_error=float(np.max(abs(final["actual_full_SGram"] - np.eye(n)))),
                       complete_reduced_offdiagonal_relative_max=float(np.max(final["reduced_offdiagonal_relative"])))
            sweep_rows.append(row)
            save("complete_sweep_" + str(sweep).zfill(2), **final, jacobi_matrix=reduced, full_rotations=rotations)
            write(HERE / "sweep_history.json", sweep_rows)
            if (row["original_full_physical_residual_relative_max"] <= gates["screened_full_spectrum_constrained_residual_relative"]
                    and row["full_Gram_error"] <= gates["screened_full_spectrum_SGram_error"]
                    and row["complete_reduced_offdiagonal_relative_max"] <= contract["bounds"]["Jacobi_reduced_offdiagonal_relative_norm"]):
                converged = True
                break
        check("complete_spectrum_refinement_converged_within_budget", not converged)
        calls["completed_complete_spectrum_refinements"] += 1
        order = np.argsort(final["actual_Rayleigh_Ha"], kind="stable")
        vectors = final["normalized_physical_vectors"][:, order]
        rayleigh = final["actual_Rayleigh_Ha"][order]
        absolute = final["full_constrained_residual_norm_Ha"][order]
        relative = final["full_constrained_residual_relative"][order]
        vv = vectors.astype(np.longdouble)
        raw_lowest = vectors[:, 0].copy()
        coordinate = p.T @ s @ raw_lowest.astype(np.longdouble)
        anchor = int(np.argmax(abs(coordinate)))
        phase = 1. if coordinate[anchor] >= 0 else -1.
        canonical = phase * raw_lowest
        cv = canonical.astype(np.longdouble)
        raw_energy = (cv @ h @ cv) / (cv @ s @ cv)
        missing = cv - p @ (p.T @ s @ cv)
        low_original_overlap = a["normalized_physical_vectors"][:, 0].astype(np.longdouble) @ s @ cv
        overlap = a["normalized_physical_vectors"].astype(np.longdouble).T @ s @ vv
        whole = np.column_stack((frozen, cv))
        full_p_missing = vv - p @ (p.T @ s @ vv)
        projected_residual = p.T @ (h @ cv - raw_energy * (s @ cv))
        normal = p.T @ s @ frozen
        projected_residual -= normal @ (normal.T @ projected_residual)
        raw_relative = np.sqrt(np.sum(projected_residual**2)) / max(1., abs(raw_energy), np.sqrt(np.sum((p.T @ h @ cv)**2)))
        save("final_complete_spectrum", full_sorted_indices=order, full_coefficients=final["coefficients"][:, order],
             normalized_physical_vectors=vectors, actual_Rayleigh_Ha=rayleigh,
             actual_full_SGram=vv.T @ s @ vv, full_constrained_residual_norm_Ha=absolute,
             full_constrained_residual_relative=relative, original_to_refined_signed_overlap=overlap,
             raw_original_vector=raw_lowest, canonical_raw_vector=canonical, phase=np.array(phase),
             original_P_phase_anchor=np.array(anchor), raw_Rayleigh_Ha=np.array(raw_energy),
             original_P=a["original_P"], basis=a["basis"], frozen=a["frozen"], h=a["h"], s=a["s"],
             full_final_rotations=rotations, final_reduced_operator=reduced)
        check("full116_SGram_original_gate", np.max(abs(vv.T @ s @ vv - np.eye(n))), gates["screened_full_spectrum_SGram_error"])
        check("full116_constrained_residual_original_gate", np.max(relative), gates["screened_full_spectrum_constrained_residual_relative"])
        check("full116_original_P_leakage", np.max(np.sum(full_p_missing * (s @ full_p_missing), axis=0)), gates["nuclear_projector_leakage_S_norm_squared"])
        check("minimum_cluster_resolved_original_gate", rayleigh[1] - rayleigh[0] <= gates["screened_minimum_cluster_gap_Ha"])
        check("raw_minimum_energy_original_gate", abs(raw_energy - rayleigh[0]), gates["screened_complement_minimum_energy_match_Ha"])
        check("raw_original_root_identity", low_original_overlap**2 <= gates["screened_raw_lowest_vector_S_overlap_squared_strict_min"])
        check("new_raw_original_P_leakage", missing @ s @ missing, gates["nuclear_projector_leakage_S_norm_squared"])
        check("new_raw_projected_residual_original_gate", raw_relative, gates["new_spinor_projected_auxiliary_residual_relative"])
        check("new_raw_frozen_whole_Gram_original_gate", np.max(abs(whole.T @ s @ whole - np.eye(whole.shape[1]))), gates["actual_radial_Gram_error"])
        check("lowest_phase_anchor_nonzero", coordinate[anchor] == 0)
        check("no_Atomx_import", any(name == "atomx" or name.startswith("atomx.") for name in sys.modules))
        check("no_scipy_import", any(name == "scipy" or name.startswith("scipy.") for name in sys.modules))
        result = dict(numerical_first_kappa_qualification_passed=True, contract_sha256=sha(HERE / "contract.json"),
              driver_sha256=sha(__file__), execution_binding_sha256=sha(HERE / "execution_binding.json"),
              input_roles=contract["input_roles"], source_map_sha256=contract["fixed_conditions"]["source_map_sha256"],
              original_metric_sha256=contract["fixed_conditions"]["original_metric_sha256"],
              calls=calls, checks=checks, completed_evidence=stages, sweeps=sweep_rows,
              first_kappa_full_root_count=n, original_failure_preserved=True,
              original49_failed_roots_not_selected_or_pruned=True, numerical_method=contract["numerical_method"],
              initial_full_residual_max=float(np.max(before["full_constrained_residual_relative"])),
              final_full_residual_max=float(np.max(relative)), lowest_actual_Rayleigh_Ha=float(rayleigh[0]),
              first_actual_gap_Ha=float(rayleigh[1] - rayleigh[0]), raw_actual_Rayleigh_Ha=float(raw_energy),
              raw_old_root_overlap_squared=float(low_original_overlap**2), raw_projected_residual_relative=float(raw_relative),
              actual_wall_s=time.monotonic() - start, Atomx_imports=0, new_physical_builders=0,
              screened_CI_response_available=False, other_four_kappas_qualified=False,
              original_stationarity_passed=False, experimental_validation_passed=False,
              scope=contract["scope"])
        write(HERE / "result.json", result)
        print(json.dumps(dict(numerical_first_kappa_qualification_passed=True, full_roots=n, calls=calls,
                             original_residual_max=result["initial_full_residual_max"], refined_residual_max=result["final_full_residual_max"],
                             lowest_Ha=result["lowest_actual_Rayleigh_Ha"])))
    except Exception as error:
        if "reduced" in locals() and "rotations" in locals():
            save("failed_partial_iteration", jacobi_matrix=reduced, full_rotations=rotations,
                 coefficient_basis=coefficient_basis)
        write(HERE / "failure.json", dict(error=repr(error), contract_sha256=sha(HERE / "contract.json"),
              driver_sha256=sha(__file__), execution_binding_sha256=sha(HERE / "execution_binding.json"),
              calls=calls, checks=checks, completed_evidence=stages, sweeps=sweep_rows,
              original_failure_preserved=True, new_physical_builders=0, numerical_success_claim=False))
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
