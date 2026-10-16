"""Independent saved complete116 refinement arrays; zero new scientific solves."""
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
ROOT = RUN.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    c = load(HERE / "contract.json")
    for name, expected in c["bound_inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Saved refinement input changed: " + name)
    if (HERE / "verification.json").exists():
        raise FileExistsError("Preserve completed firstκ independent audit")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 3:
        raise RuntimeError("Require at least3 affinity CPUs")
    os.sched_setaffinity(0, {affinity[-2]})
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))

    def timeout(*_):
        raise TimeoutError("Saved116-root audit90s cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(90)
    import numpy as np

    checks = []

    def check(name, error, limit=0):
        error = float(error)
        passed = bool(np.isfinite(error) and error <= limit)
        checks.append(dict(name=name, error=error, limit=limit, passed=passed))
        if not passed:
            raise ValueError("Independent saved116 gate: " + name)

    try:
        r, original = load(RUN / "result.json"), load(RUN / "contract.json")
        closure = load(RUN / "root_execution.json")
        check("actual_foreground_closed_success", closure["exit_code"] != 0 or closure["outer_wall_cap_stopped"])
        check("runtime_source_driver_bound", r["driver_sha256"] != sha(RUN / "run.py")
              or r["contract_sha256"] != sha(RUN / "contract.json")
              or r["execution_binding_sha256"] != sha(RUN / "execution_binding.json"))
        check("firstκ_only_scope", r["first_kappa_full_root_count"] != 116 or r["other_four_kappas_qualified"]
              or r["screened_CI_response_available"] or r["original_stationarity_passed"] or r["experimental_validation_passed"])
        expected = dict(complete_spectrum_refinement_attempts=1, completed_complete_spectrum_refinements=1,
              attempted_complete_sweeps=1, completed_complete_sweeps=1, attempted_pair_rotations=6670,
              completed_pair_rotations=6670, physical_residual_evaluations=2, near_identity_Gram_whitenings=1)
        for name, value in r["calls"].items():
            check("actual_counter/" + name, value != expected.get(name, 0))
        for row in r["completed_evidence"]:
            check("stage_bytes/" + row["path"], sha(RUN / row["path"]) != row["sha256"])
        paths = {name: ROOT / row["path"] for name, row in original["input_roles"].items()}
        old_failure = load(paths["failed_result"])
        check("original_failure_preserved", old_failure["scientific_success_claim"] is not False)
        init = np.load(RUN / "initial_complete_representation.npz", allow_pickle=False)
        final = np.load(RUN / "final_complete_spectrum.npz", allow_pickle=False)
        sweep = np.load(RUN / "complete_sweep_01.npz", allow_pickle=False)
        old = np.load(paths["original_control"], allow_pickle=False)
        for name in ("h", "s", "basis", "original_P", "frozen"):
            check("unchanged_original_arrays/" + name, not np.array_equal(final[name], old[name]))
        check("whole116_sort_permutation", not np.array_equal(np.sort(final["full_sorted_indices"]), np.arange(116)))
        h, s, b, p, frozen, v = [final[name].astype(np.longdouble) for name in
                                  ("h", "s", "basis", "original_P", "frozen", "normalized_physical_vectors")]
        check("all116_physical_columns", v.shape != (246, 116))
        hv, sv = h @ v, s @ v
        norms = np.sum(v * sv, axis=0)
        check("all116_norms_positive", np.any(norms <= 0) or np.any(~np.isfinite(norms)))
        rq = np.sum(v * hv, axis=0) / norms
        residual = b.T @ (hv - sv * rq)
        absolute = np.sqrt(np.sum(residual**2, axis=0))
        relative = absolute / np.maximum(1., np.maximum(abs(rq), np.sqrt(np.sum((b.T @ hv)**2, axis=0))))
        gram = v.T @ sv
        check("full116_original_h_S_residual", np.max(relative), 1e-8)
        check("full116_original_SGram", np.max(abs(gram - np.eye(116))), 1e-10)
        check("saved_actual_Rayleigh_replay", np.max(abs(rq - final["actual_Rayleigh_Ha"])), 1e-12)
        check("saved_absolute_residual_replay", np.max(abs(absolute - final["full_constrained_residual_norm_Ha"])), 1e-12)
        check("saved_relative_residual_replay", np.max(abs(relative - final["full_constrained_residual_relative"])), 1e-14)
        check("all116_sorted_Rayleigh", np.max(np.maximum(0., -np.diff(rq))))
        check("all116_frozen_orthogonality", np.max(abs(frozen.T @ sv)), 1e-10)
        missing = v - p @ (p.T @ sv)
        leakage = np.sum(missing * (s @ missing), axis=0)
        check("all116_originalP_leakage", np.max(leakage), 1e-12)
        overlap = old["normalized_physical_vectors"].astype(np.longdouble).T @ sv
        check("full_signed_overlap_replay", np.max(abs(overlap - final["original_to_refined_signed_overlap"])), 1e-12)
        check("complete_signed_span_preserved", np.max(abs(overlap.T @ overlap - np.eye(116))), 1e-10)
        rotation = final["full_final_rotations"]
        check("full_rotation_orthogonality", np.max(abs(rotation.T @ rotation - np.eye(116))), 1e-10)
        rotated = rotation.T @ init["initial_reduced_operator"] @ rotation
        matrix_difference = rotated - final["final_reduced_operator"]
        scale = max(1., float(np.max(np.sum(abs(rotated), axis=1))))
        check("full_rotation_congruence_relative", np.max(np.sum(abs(matrix_difference), axis=1)) / scale, 1e-12)
        check("coefficient_whitening_and_rotation", np.max(abs(init["coefficient_basis"] @ rotation
              - final["full_coefficients"][:, np.argsort(final["full_sorted_indices"])])), 1e-14)
        order = final["full_sorted_indices"]
        check("final_sweep_sorted_physical_bytes", not np.array_equal(sweep["normalized_physical_vectors"][:, order], final["normalized_physical_vectors"]))
        raw, canonical = final["raw_original_vector"], final["canonical_raw_vector"]
        check("new_raw_actual_lowest_bytes", not np.array_equal(raw, final["normalized_physical_vectors"][:, 0]))
        check("joint_phase_raw_bytes", not np.array_equal(canonical, raw * float(final["phase"])))
        cv = canonical.astype(np.longdouble)
        coordinate = p.T @ s @ raw.astype(np.longdouble)
        anchor = int(np.argmax(abs(coordinate)))
        check("originalP_phase_anchor", anchor != int(final["original_P_phase_anchor"]))
        check("canonical_phase_is_fixed_anchor", coordinate[anchor] * float(final["phase"]) < 0 or coordinate[anchor] == 0)
        energy = (cv @ h @ cv) / (cv @ s @ cv)
        check("minimum_physical_energy_original_gate", abs(energy - rq[0]), 1e-8)
        check("minimum_isolated_cluster_original_gate", rq[1] - rq[0] <= 1e-10)
        old_overlap = old["normalized_physical_vectors"][:, 0].astype(np.longdouble) @ s @ cv
        check("lowest_original_signed_root_identity", old_overlap**2 <= .5)
        whole = np.column_stack((frozen, cv))
        check("lowest_and_frozen_Gram", np.max(abs(whole.T @ s @ whole - np.eye(whole.shape[1]))), 1e-10)
        normal = p.T @ s @ frozen
        raw_residual = p.T @ (h @ cv - energy * (s @ cv))
        raw_residual -= normal @ (normal.T @ raw_residual)
        raw_relative = np.sqrt(np.sum(raw_residual**2)) / max(1., abs(energy), np.sqrt(np.sum((p.T @ h @ cv)**2)))
        check("raw_projected_auxiliary_residual_original_gate", raw_relative, 1e-8)
        check("runtime_summary_allroot_replay", abs(float(np.max(relative)) - r["final_full_residual_max"]), 1e-14)
        check("no_Atomx_or_scipy_import", any(n == "atomx" or n.startswith("atomx.") or n == "scipy" or n.startswith("scipy.") for n in sys.modules))
        proof = dict(all_independent_saved_array_checks_passed=True, checks_count=len(checks), checks=checks,
              contract_sha256=sha(HERE / "contract.json"), audit_driver_sha256=sha(__file__),
              bound_inputs_sha256=c["bound_inputs_sha256"], first_kappa_complete_roots=116,
              original_full_residual_max=r["initial_full_residual_max"], independently_replayed_full_residual_max=float(np.max(relative)),
              independently_replayed_Gram_error=float(np.max(abs(gram - np.eye(116)))),
              lowest_Rayleigh_Ha=float(energy), first_Rayleigh_gap_Ha=float(rq[1] - rq[0]),
              old_lowest_signed_overlap_squared=float(old_overlap**2), full_rotation_congruence_absolute_max_Ha=float(np.max(abs(matrix_difference))),
              full_rotation_congruence_relative=float(np.max(np.sum(abs(matrix_difference), axis=1)) / scale),
              original_failed_invocation_remains_failed=True, other_four_kappas_qualified=False,
              screened_CI_response_available=False, original_stationarity_passed=False, experimental_gate_passed=False,
              new_Atomx_H_CI_Dirac_virtual_gradient_optimizer_eigensolves=0, Atomx_imports=0,
              scope="Firstκ−3 complete116 saved-spectrum qualification; all original gates replayed. No physical builder, eigen solve, screenedCI response, otherκ, or experimental claim.")
        (HERE / "verification.json").write_text(json.dumps(proof, indent=2) + "\n")
        print(json.dumps(dict(checks=len(checks), full_roots=116, residual=float(np.max(relative)),
                             Gram=float(np.max(abs(gram - np.eye(116)))), new_scientific_calls=0)))
    except Exception as error:
        (HERE / "failure.json").write_text(json.dumps(dict(error=repr(error), checks=checks,
              audit_driver_sha256=sha(__file__), contract_sha256=sha(HERE / "contract.json"),
              new_scientific_calls=0), indent=2) + "\n")
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
