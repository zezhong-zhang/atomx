"""Replay saved signed operators/method differences; no amplitude or FCI solve."""

import ast
import hashlib
import json
import os
import resource
import sys
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np  # noqa: E402
import probe  # noqa: E402
import scipy  # noqa: E402
from scipy.linalg import expm  # noqa: E402

OWNER, RUN = probe.OWNER, probe.RUN


def maximum(array):
    return float(np.max(abs(array), initial=0))


def scientific_driver_ast(path):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.FunctionDef) and node.name == "dump")
                 and not (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "RUN" for target in node.targets))]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def main():
    resource.setrlimit(resource.RLIMIT_AS, (probe.CAP, probe.CAP))
    reader, oracle = probe.helpers()
    from atomx.constants import HARTREE_CM1
    from atomx.methods.fscc import load_integrals
    saved = np.load(RUN / "fci_arrays.npz")
    a = load_integrals(RUN / "fixed_inputs")
    mc, ma = saved["core_masks"].tolist(), saved["attachment_masks"].tolist()
    check_mc, hc = oracle.determinant_matrix(a, 4)
    check_ma, ha = oracle.determinant_matrix(a, 5)
    ri, pi = mc.index(15), saved["model_indices"]
    checks = dict(fixed_core_masks=check_mc == mc, fixed_attachment_masks=check_ma == ma,
                  fixed_core_signed_H_max_Ha=maximum(hc - saved["full_core_hamiltonian_Ha"]),
                  fixed_attachment_signed_H_max_Ha=maximum(ha - saved["full_attachment_hamiltonian_Ha"]),
                  FCI_core_all_vectors_eigen_residual_max_Ha=maximum(hc @ saved["exact_core_vectors"] - saved["exact_core_vectors"] * saved["exact_core_roots_Ha"]),
                  FCI_attachment_all_vectors_eigen_residual_max_Ha=maximum(ha @ saved["exact_attachment_vectors"] - saved["exact_attachment_vectors"] * saved["exact_attachment_roots_Ha"]))
    historical = np.load(probe.REPO / "docs/atomic_external_benchmark_2026_10_11/fscc/run003/accuracy_four_core_corrected/full_signed_audit.npz")
    checks.update(historical_core_same_H_max_Ha=maximum(hc - historical["full_core_hamiltonian_Ha"]),
                  historical_attachment_same_H_max_Ha=maximum(ha - historical["full_attachment_hamiltonian_Ha"]))
    results = {name: json.loads((RUN / (name + "_result.json")).read_text())
               for name in ("fci", "lccsd", "ccsd", "ccsdt")}
    core_fci = saved["exact_core_roots_Ha"][0]
    method_checks = {}
    for name in ("lccsd", "ccsd", "ccsdt"):
        arrays = np.load(RUN / (name + "_arrays.npz"))
        result = results[name]
        check = dict(arrays_hash=probe.sha(RUN / (name + "_arrays.npz")) == result["arrays_sha256"])
        if name == "lccsd":
            t = oracle.excitation_matrix(mc, [
                (tuple((p, True) for p in range(4, 12) if mask & (1 << p)) +
                 tuple((i, False) for i in reversed(range(4)) if not mask & (1 << i)), amplitude /
                 oracle.apply(15, tuple((p, True) for p in range(4, 12) if mask & (1 << p)) +
                              tuple((i, False) for i in reversed(range(4)) if not mask & (1 << i)))[1])
                for mask, amplitude in zip((mc[index] for index in arrays["sd_indices"][1:]), arrays["amplitudes"], strict=True)])
            transformed = hc + hc @ t - t @ hc
            residual = transformed[:, ri].copy()
            energy = transformed[ri, ri]
            residual[ri] -= energy
            retained = (arrays["core_ranks"] > 0) & (arrays["core_ranks"] <= 2)
            check.update(independent_cluster_max=maximum(t - arrays["core_cluster"]),
                         independent_linear_operator_max_Ha=maximum(transformed - arrays["full_linearized_H_Ha"]),
                         independent_full_signed_residual_max_Ha=maximum(residual - arrays["core_residual_Ha"]),
                         core_retained_residual_max_Ha=maximum(residual[retained]),
                         core_signed_method_error_difference_Ha=abs(float((energy - core_fci).real) - result["core_fci_signed_method_error_Ha"]))
        else:
            rank = 2 if name == "ccsd" else 3
            tensors = {f"{stem}{level}c": reader.decode_diagram(RUN / name / "scratch" / f"{stem}{level}c.dg", 12)[0]
                       for stem in ("t", "s") for level in range(1, rank + 1)}
            terms = reader.cluster_terms([tensors[f"t{level}c"] for level in range(1, rank + 1)])
            tc, ta = oracle.excitation_matrix(mc, terms), oracle.excitation_matrix(ma, terms)
            bc, ba = expm(-tc) @ hc @ expm(tc), expm(-ta) @ ha @ expm(ta)
            energy = bc[ri, ri]
            rc = bc[:, ri].copy()
            rc[ri] -= energy
            sa = oracle.excitation_matrix(ma, reader.cluster_terms([tensors[f"s{level}c"] for level in range(1, rank + 1)], attachment=True))
            omega = (np.eye(len(ma)) + sa)[:, pi]
            matrix = np.load(RUN / name / "model_space_matrix.npz")["matrix_Ha"]
            total = matrix + energy * np.eye(8)
            ra = ba @ omega - omega @ total
            ec = np.sort(np.linalg.eigvals(total).real)
            errors = ec - saved["exact_attachment_roots_Ha"][:8]
            core_error = float((energy - core_fci).real)
            check.update(decoded_tensors_max=max(maximum(tensors[key] - arrays[key]) for key in tensors),
                         independent_core_cluster_max=maximum(tc - arrays["core_cluster"]),
                         independent_attachment_cluster_max=maximum(ta - arrays["attachment_cluster"]),
                         independent_core_hbar_max_Ha=maximum(bc - arrays["core_hbar_Ha"]),
                         independent_attachment_hbar_max_Ha=maximum(ba - arrays["attachment_hbar_Ha"]),
                         independent_omega_max=maximum(omega - arrays["omega"]),
                         independent_full_core_residual_max_Ha=maximum(rc - arrays["core_residual_Ha"]),
                         independent_full_attachment_residual_max_Ha=maximum(ra - arrays["attachment_residual_Ha"]),
                         core_retained_residual_max_Ha=maximum(rc[(arrays["core_ranks"] > 0) & (arrays["core_ranks"] <= rank)]),
                         attachment_retained_residual_max_Ha=maximum(ra[arrays["attachment_ranks"] < rank]),
                         core_signed_method_error_difference_Ha=abs(core_error - result["core_fci_signed_method_error_Ha"]),
                         attachment_signed_errors_difference_Ha=maximum(errors - result["attachment_fci_signed_method_errors_Ha"]),
                         attachment_minus_own_core_errors_difference_Ha=maximum(errors - core_error - result["attachment_minus_own_core_fci_signed_method_errors_Ha"]))
        method_checks[name] = check
    original = OWNER / "run001/executed_probe.py"
    identity = dict(original_executed_probe_sha256=probe.sha(original), current_probe_sha256=probe.sha(OWNER / "probe.py"),
                    original_scientific_ast_sha256=scientific_driver_ast(original), current_scientific_ast_sha256=scientific_driver_ast(OWNER / "probe.py"),
                    excluded_changes="private JSON dump function and RUN directory literal only; all scientific functions, resource gates, stopping, tolerances, model/root/H contracts unchanged")
    identity["scientific_AST_identical"] = identity["original_scientific_ast_sha256"] == identity["current_scientific_ast_sha256"]
    old_same = {}
    for name in ("lccsd", "ccsd"):
        before = np.load(OWNER / "run001" / (name + "_arrays.npz"))
        after = np.load(RUN / (name + "_arrays.npz"))
        old_same[name] = dict(keys_identical=before.files == after.files,
                              complete_arrays_max_difference=max(maximum(before[key] - after[key]) for key in before.files))
    unchanged = probe.unchanged()
    gates = dict(source_inputs_unchanged=all(unchanged.values()),
                 saved_receipts_pass=all(all(row["gates"].values()) for row in results.values()),
                 same_frozen_H=checks["fixed_core_signed_H_max_Ha"] == checks["fixed_attachment_signed_H_max_Ha"] == checks["historical_core_same_H_max_Ha"] == checks["historical_attachment_same_H_max_Ha"] == 0,
                 all_FCI_vectors_residual=max(checks["FCI_core_all_vectors_eigen_residual_max_Ha"], checks["FCI_attachment_all_vectors_eigen_residual_max_Ha"]) <= probe.GATE,
                 all_complete_operator_replays=max(value for check in method_checks.values() for key, value in check.items() if key.endswith("difference_Ha") or key.startswith("independent") or key == "decoded_tensors_max") <= 1e-12,
                 all_retained_equations=max(value for check in method_checks.values() for key, value in check.items() if "retained_residual" in key) <= probe.GATE,
                 arrays_hashes=all(check["arrays_hash"] for check in method_checks.values()),
                 serializer_only_scientific_AST=identity["scientific_AST_identical"],
                 failed_receipt_attempt_same_science=all(row["keys_identical"] and row["complete_arrays_max_difference"] == 0 for row in old_same.values()))
    summary = dict(status="fixed_H_method_rank_discriminator_pass" if all(gates.values()) else "replay_failure_preserved", gates=gates,
                   contract_sha256=probe.sha(RUN / "contract.json"), executed_probe_sha256=probe.sha(OWNER / "probe.py"),
                   replay_source_sha256=probe.sha(Path(__file__)), source_archive_sha256=probe.sha(RUN / "source_inputs.tar.gz"),
                   results={name: dict(path=str((RUN / (name + "_result.json")).relative_to(probe.REPO)), sha256=probe.sha(RUN / (name + "_result.json"))) for name in results},
                   source_and_inputs_unchanged=unchanged, frozen_H_and_FCI_replay=checks, method_operator_replay=method_checks,
                   persistence_driver_identity_chain=identity, original_failed_receipt_science_comparison=old_same,
                   runtime_versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__),
                   unit_conversion_Hartree_to_cm1=float(HARTREE_CM1),
                   core_signed_FCI_method_errors_cm1={name: result["core_fci_signed_method_error_Ha"] * HARTREE_CM1 for name, result in results.items() if name != "fci"},
                   attachment_total_root_abs_errors_cm1={name: results[name]["attachment_fci_abs_method_error_max_Ha"] * HARTREE_CM1 for name in ("ccsd", "ccsdt")},
                   attachment_minus_own_core_abs_errors_cm1={name: results[name]["attachment_minus_own_core_fci_abs_method_error_max_Ha"] * HARTREE_CM1 for name in ("ccsd", "ccsdt")},
                   inference="This one frozen finite Z=4 reference shows real method truncation errors at unchanged orbital space, even when each method's retained equations converge. Actual nonlinear SD and self-consistent triples response can matter much more than their numerical residual. It does not identify the dominant Yb defect or bound omitted orbital/correlation/nuclear/QED error.",
                   limits=["Complete LCCSD core was solved by dense public finite correctness anchor, not native pCI HPC CoreSD. No matched-H LCCSD attachment solve was run.",
                           "CCSDT has omitted connected T4/S4; no arbitrary excitation rank, full atom basis, experimental or production Yb certificate.",
                           "Differences are self-consistent whole-method responses; lower-rank amplitudes change with rank. Do not add separate nominal nonlinear/triples/Breit/QED corrections.",
                           "Real finite-frequency transverse exchange and point nucleus are frozen in every method; no dynamic Breit increment is isolated here.",
                           "Operational wall time and RUSAGE_SELF/CHILDREN maxima are resource receipts, not paired performance benchmarks or parallel efficiencies; RSS maxima are not simultaneous process-tree sums."],
                   performance_claim=False)
    probe.dump(RUN / "replay.json", summary)
    print(summary["status"], gates)
    if not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
