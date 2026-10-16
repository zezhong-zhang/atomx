"""Independent archived-operator/vector algebra; no Atomx or eigensolve."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    args = parser.parse_args()
    prospective = load(HERE / "contract.json")
    binding = load(args.binding)
    if (binding.get("completed_input_hashes_bound") is not True
            or binding.get("audit_contract_sha256") != sha(HERE / "contract.json")
            or binding.get("audit_driver_sha256") != sha(__file__)):
        raise ValueError("Completed diagnostic exact audit binding required")
    for name, expected in binding["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Completed evidence changed: " + name)
    if (HERE / "verification.json").exists():
        raise FileExistsError("Preserve independent diagnostic audit")
    result = load(RUN / "result.json")
    if (sha(RUN / "run.py") != prospective["diagnostic_driver_sha256"]
            or sha(RUN / "contract.json") != prospective["diagnostic_contract_sha256"]):
        raise ValueError("Prospective diagnostic epoch changed")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 3:
        raise RuntimeError("Three original affinity cores required")
    os.sched_setaffinity(0, {available[-2]})
    cap = prospective["resources"]["address_space_GiB"] * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    def timeout(*_):
        raise TimeoutError("Saved-vector-only diagnostic audit wall cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(prospective["resources"]["wall_s"])
    import numpy as np

    checks, controls = [], []

    def check(name, error, limit=0):
        error = float(error)
        passed = bool(np.isfinite(error) and error <= limit)
        checks.append(dict(name=name, error=error, limit=limit, passed=passed))
        if not passed:
            raise ValueError("Independent diagnostic audit failed: " + name)

    check("result_exact_contract", result["contract_sha256"] != sha(RUN / "contract.json"))
    check("result_exact_driver", result["driver_sha256"] != sha(RUN / "run.py"))
    check("exact_source_epoch", result["source_map_sha256"] != prospective["source_map_sha256"])
    check("original_gate_unchanged", result["original_scientific_gate_Ha"] != 1e-8)
    check("raw_unchanged_not_repair_claim", result["raw_vector_unchanged"] is not True or
          result["no_repaired_generator_or_threepoint_claim"] is not True)
    check("no_experimental_or_stationary_claim", result["experimental_gate_passed"] is not False or
          result["original_stationarity_passed"] is not False or result["new_state_or_boundary"] is not False)
    expected = dict(one_body_attempts=1, one_body_completed=1, public_Dirac_attempts=1,
                    public_Dirac_completed=1, complement_SVD_attempts=2, complement_SVD_completed=2,
                    condition_SVD_attempts=2, condition_SVD_completed=2,
                    subset_values_attempts=2, subset_values_completed=2, subset_vectors_attempts=2,
                    subset_vectors_completed=2, full_vectors_attempts=2, full_vectors_completed=2,
                    bordered_Newton_attempts=2, bordered_Newton_completed=2,
                    CI_solves=0, many_electron_H=0, virtual_generator_calls=0, gradient_calls=0,
                    optimizer_calls=0)
    for name, count in expected.items():
        check("actual_public_diagnostic_count/" + name, result["calls"][name] != count)
    check("operators_archive_SHA", sha(RUN / result["operators"]["path"]) != result["operators"]["sha256"])
    ops = np.load(RUN / result["operators"]["path"], allow_pickle=False)
    h, s, raw, frozen = ops["h"], ops["s"], ops["raw_vector"], ops["frozen"]
    hld, sld = h.astype(np.longdouble), s.astype(np.longdouble)
    check("operators_finite", any(not np.isfinite(x).all() for x in (h, s, raw, frozen)))
    check("canonical_H_symmetric", np.max(abs(h-h.T)))
    check("canonical_S_symmetric", np.max(abs(s-s.T)))
    check("raw_S_norm", abs(float(raw @ s @ raw)-1), 1e-10)
    check("raw_frozen_orthogonal", np.max(abs(frozen.T @ s @ raw)), 1e-10)

    def rq(vector, high=False):
        if high:
            v = vector.astype(np.longdouble)
            return float(v @ hld @ v / (v @ sld @ v))
        return float(vector @ h @ vector / (vector @ s @ vector))

    def residual(basis, vector, energy):
        r = basis.astype(np.longdouble).T @ (hld @ vector.astype(np.longdouble)
              - np.longdouble(energy)*(sld @ vector.astype(np.longdouble)))
        return float(np.linalg.norm(r)), float(np.max(abs(r)))

    for report in result["complements"]:
        name = report["name"]
        archive = report["arrays"]
        check(name + "/archive_SHA", sha(RUN / archive["path"]) != archive["sha256"])
        a = np.load(RUN / archive["path"], allow_pickle=False)
        basis, projected_h, projected_s = a["basis"], a["projected_h"], a["projected_s"]
        check(name + "/finite_all_arrays", any(not np.isfinite(a[key]).all() for key in a.files))
        check(name + "/complete_dimension", basis.shape[1] != report["complement_dimension"])
        check(name + "/original_complete_congruence_H", not np.array_equal(projected_h, basis.T @ h @ basis))
        check(name + "/original_complete_congruence_S", not np.array_equal(projected_s, basis.T @ s @ basis))
        check(name + "/frozen_complement_orthogonal", np.max(abs(frozen.T @ s @ basis)), 1e-10)
        check(name + "/complement_SGram", np.max(abs(projected_s-np.eye(len(projected_s)))), 1e-10)
        sv = a["projected_s_singular_values"]
        check(name + "/S_condition_replay", abs(float(sv.max()/sv.min())-report["projected_S_condition"]), 1e-14)
        check(name + "/H_norm_replay", abs(float(np.linalg.norm(projected_h, np.inf))-report["projected_H_inf_norm_Ha"]))
        check(name + "/raw_Rayleigh_longdouble_replay", abs(rq(raw, True)-report["raw_vector"]["longdouble_Rayleigh_Ha"]), 1e-12)
        check(name + "/raw_reported_minimum_replay", abs((rq(raw)-float(a["subset_values"][0]))
              -report["raw_vector"]["raw_minus_reported_minimum_Ha"]), 1e-12)
        check(name + "/old_minimum_gate_boolean_replay", report["original_reported_minimum_gate_passed"]
              != (abs(rq(raw)-float(a["subset_values"][0])) <= 1e-8))
        rnorm, rmax = residual(basis, raw, rq(raw, True))
        check(name + "/raw_constrained_residual_replay", abs(rnorm-report["raw_vector"]["complement_residual_norm_Ha"]), 1e-12)
        for method, energies, vectors in (
                ("gvx_vectors", a["subset_vector_energies"], a["subset_vectors"]),
                ("full_gvd", a["full_energies"][:2], a["full_vectors"][:, :2])):
            check(name + "/" + method + "/SGram", np.max(abs(vectors.T @ projected_s @ vectors-np.eye(2))), 1e-10)
            for root in range(2):
                vector = basis @ vectors[:, root]
                record = next(row for row in report["eigenvector_controls"] if row["method"] == method and row["root"] == root)
                rayleigh, energy = rq(vector, True), float(energies[root])
                rn, rm = residual(basis, vector, rayleigh)
                for key, expected_value in dict(reported_energy_Ha=energy, longdouble_Rayleigh_Ha=rayleigh,
                        reported_minus_longdouble_Ha=energy-rayleigh,
                        raw_minus_vector_Rayleigh_Ha=rq(raw, True)-rayleigh,
                        complement_residual_norm_Ha=rn, complement_residual_max_Ha=rm).items():
                    check(name + "/" + method + "/" + str(root) + "/" + key, abs(record[key]-expected_value), 1e-12)
        controls.append(dict(name=name, original_reported_minimum_gate_passed=report["original_reported_minimum_gate_passed"],
            raw_minus_reported_minimum_Ha=rq(raw)-float(a["subset_values"][0]),
            raw_minus_gvx_vector_Rayleigh_Ha=rq(raw, True)-rq(basis @ a["subset_vectors"][:, 0], True),
            raw_minus_full_gvd_vector_Rayleigh_Ha=rq(raw, True)-rq(basis @ a["full_vectors"][:, 0], True),
            projected_H_inf_norm_Ha=report["projected_H_inf_norm_Ha"], projected_S_condition=report["projected_S_condition"],
            raw_constrained_residual_norm_Ha=rnorm))
        for record in report["Newton_controls"]:
            iteration = record["iteration"]
            newton = np.load(RUN / ("Newton" + str(iteration) + ".npz"), allow_pickle=False)
            vector = newton["vector"]
            check("Newton" + str(iteration) + "/same_basis_exact", not np.array_equal(vector, basis @ newton["coordinates"]))
            check("Newton" + str(iteration) + "/normalized", abs(float(vector @ s @ vector)-1), 1e-10)
            check("Newton" + str(iteration) + "/frozen_constraint", np.max(abs(frozen.T @ s @ vector)), 1e-10)
            rn, _ = residual(basis, vector, rq(vector, True))
            check("Newton" + str(iteration) + "/energy_replay", abs(rq(vector, True)-record["longdouble_Rayleigh_Ha"]), 1e-12)
            check("Newton" + str(iteration) + "/residual_replay", abs(rn-record["complement_residual_norm_Ha"]), 1e-12)
    check("no_Atomx_consumer_imported", any(name == "atomx" or name.startswith("atomx.") for name in sys.modules))
    signal.alarm(0)
    proof = dict(all_independent_saved_array_checks_passed=True, checks_count=len(checks), checks=checks,
                 numerical_minimum_controls=controls, bound_input_sha256=binding["inputs_sha256"],
                 audit_contract_sha256=sha(HERE / "contract.json"), audit_driver_sha256=sha(__file__),
                 new_H_CI_Dirac_virtual_gradient_optimizer_calls=0, new_eigensolves=0, Atomx_imports=0,
                 original_failed_virtual_gate_retained=True, experimental_gate_passed=False,
                 original_stationarity_passed=False, generator_repair_certified=False,
                 limit="Saved algebra diagnoses reported eigenvalue/vector Rayleigh consistency; no production fix or three-point physics response.")
    (HERE / "verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(dict(checks=len(checks), all_passed=True, controls=controls)))


if __name__ == "__main__":
    main()
