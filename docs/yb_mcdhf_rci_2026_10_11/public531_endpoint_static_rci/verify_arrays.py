"""Independent saved-array replay; no Atomx imports or physical evaluations."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import resource
import signal
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def source_hartree_cm1(path):
    """Read the frozen CODATA literals; never import or execute Atomx."""
    literals = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("CODATA", "DEFAULT_CODATA_VERSION"):
                    literals[target.id] = ast.literal_eval(node.value)
    active = literals["CODATA"][literals["DEFAULT_CODATA_VERSION"]]
    return active["_Ha_J"] / (100.0 * active["_hplanck"] * active["_c"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-contract", required=True, type=Path)
    args = parser.parse_args()
    approval = load(args.execution_contract)
    contract, result = load(HERE / "contract.json"), load(HERE / "result.json")
    if (approval.get("execution_authorized_by_root") is not True
            or approval.get("independent_saved_array_verification_authorized") is not True
            or approval.get("driver_sha256") != sha(HERE / "run.py")
            or approval.get("verifier_sha256") != sha(Path(__file__))
            or approval.get("preparation_contract_sha256") != sha(HERE / "contract.json")
            or approval.get("scientific_inputs_sha256") != contract["inputs_sha256"]
            or result["contract_sha256"] != sha(HERE / "contract.json")
            or result["execution_contract_sha256"] != sha(args.execution_contract)
            or result["driver_sha256"] != sha(HERE / "run.py")
            or result["arrays_sha256"] != sha(HERE / "arrays.npz")):
        raise ValueError("Saved-array replay lacks exact root/source/point binding")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 3:
        raise RuntimeError("Need three CPU identities for deterministic third-last core")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    os.sched_setaffinity(0, {affinity[-3]})
    cap = 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Saved replay30s cap")))
    signal.alarm(30)
    import numpy as np

    fixed, gates, checks = contract["fixed_conditions"], contract["acceptance"], []

    def check(name, value, limit):
        value = float(value)
        checks.append({"name": name, "error": value, "limit": limit,
                       "passed": bool(np.isfinite(value) and value <= limit)})

    for name, expected in contract["inputs_sha256"].items():
        check("immutable_input:" + name, sha(ROOT / name) != expected, 0)
    for name, expected in contract["frozen_source_files_sha256"].items():
        check("frozen_source:" + name, sha(ROOT / fixed["source_root"] / Path(name).relative_to("src")) != expected, 0)
    archive = np.load(HERE / "arrays.npz")
    endpoint = np.load(ROOT / fixed["endpoint_arrays"])
    for name, target in (("DC_matrix_Ha", "matrix_Ha"), ("DC_energies_Ha", "energies_Ha"), ("DC_mixing", "mixing"), ("x", "x")):
        check("exact_saved_endpoint:" + name, not np.array_equal(archive[name], endpoint[target]), 0)
    waves = {tuple(key): (p, q) for key, p, q in zip(archive["orbital_keys"], archive["P"], archive["Q"], strict=True)}
    original = {tuple(key): (p, q) for key, p, q in zip(endpoint["orbital_keys"], endpoint["P"], endpoint["Q"], strict=True)}
    check("same_complete24_keys", len(waves) != 24 or set(waves) != set(original), 0)
    for key in original:
        check("same_point_radial:" + str(key), any(not np.array_equal(a, b)
              for a, b in zip(waves[key], original[key], strict=True)), 0)
    records = load(ROOT / fixed["full_basis"])["CSFs"]
    groups = {}
    for row in records:
        groups.setdefault((row["twoJ"], row["parity"]), []).append(row["index"])
    check("complete531_all16spaces", len(records) != 531 or len(groups) != 16, 0)
    for model in ("DC", "static"):
        h, e, v = archive[model + "_matrix_Ha"], archive[model + "_energies_Ha"], archive[model + "_mixing"]
        check(model + "/full_finite_dimensions", h.shape != (531, 531) or e.shape != (531,)
              or v.shape != (531, 531) or any(a.dtype != np.float64 or not np.isfinite(a).all() for a in (h, e, v)), 0)
        # Rootwise Rayleigh/residual anchors differ from producer's batched
        # residual; arbitrary phases/complete degenerate rotations are valid.
        for col in range(531):
            vector = v[:, col]
            check(f"{model}/norm:{col}", abs(float(vector @ vector) - 1), gates["CI_orthogonality"])
            check(f"{model}/Rayleigh:{col}", abs(float(vector @ h @ vector) - float(e[col])), gates["CI_residual_Ha"])
            check(f"{model}/residual:{col}", np.linalg.norm(h @ vector - e[col] * vector), gates["CI_residual_Ha"])
        check(model + "/complete_CI_Gram", np.max(abs(v.T @ v - np.eye(531))), gates["CI_orthogonality"])
        check(model + "/H_Hermitian", np.max(abs(h - h.T)), gates["matrix_hermiticity_Ha"])
        for symmetry, indices in groups.items():
            outside = [i for i in range(531) if i not in indices]
            check(model + "/cross_symmetry:" + str(symmetry), np.max(abs(h[np.ix_(indices, outside)])),
                  gates["cross_symmetry_zero_Ha"])
        weights = archive[model + "_state_weights"]
        check(model + "/five_equal_targets", weights.shape != (531,) or np.count_nonzero(weights) != 5
              or not np.array_equal(weights[weights > 0], np.repeat(.2, 5)), 0)
    signed = archive["DC_mixing"].T @ archive["static_mixing"]
    check("signed_common_CSF_overlap_replay", np.max(abs(signed
          - archive["DC_to_static_signed_common_CSF_coefficient_overlap"])), 0)
    nominal = {r["label"]: r for r in load(ROOT / fixed["experimental_reference"])["levels"]}
    dc, static = ({r["label"]: r for r in result[key]} for key in ("DC_rows", "static_rows"))
    factor = source_hartree_cm1(ROOT / fixed["source_root"] / "atomx/constants.py")
    check("five_target_labelset", set(dc) != set(nominal) or set(static) != set(nominal), 0)
    lower_dc, lower_static = dc["6s_1/2"]["energy_Ha"], static["6s_1/2"]["energy_Ha"]
    for label in nominal:
        a, b = dc[label], static[label]
        for model, row in (("DC", a), ("static", b)):
            symmetry = (nominal[label]["two_J"], nominal[label]["parity"])
            indices = groups[symmetry]
            support = np.sum(archive[model + "_mixing"][indices]**2, axis=0)
            columns = np.flatnonzero(support > 1 - 1e-10).tolist()
            check(model + "/target_exact_symmetry:" + label, support[row["column"]] < 1 - 1e-10
                  or (row["twoJ"], row["parity"]) != symmetry, 0)
            check(model + "/target_complete_local_rank:" + label, len(columns) != len(indices)
                  or row["column"] not in columns or columns.index(row["column"]) != row["local_root"], 0)
            check(model + "/target_equal_selected_weight:" + label,
                  archive[model + "_state_weights"][row["column"]] != .2, 0)
            check(model + "/selected_target_energy:" + label, abs(row["energy_Ha"]
                  - archive[model + "_energies_Ha"][row["column"]]), 0)
            lower = lower_dc if model == "DC" else lower_static
            check(model + "/same_lower_transition:" + label, abs(row["excitation_cm1"]
                  - (row["energy_Ha"] - lower) * factor), 1e-7)
            check(model + "/nominal_gap:" + label, abs(row["nominal_gap_cm1"]
                  - (row["excitation_cm1"] - nominal[label]["excitation_cm1"])), 1e-7)
        shift = ((b["energy_Ha"] - a["energy_Ha"]) - (lower_static - lower_dc)) * factor
        check("upper_lower_static_delta:" + label, abs(b["static_minus_DC_transition_cm1"] - shift), 1e-7)
    for entry in result["multiplets"]:
        labels = ("5d_3/2", "5d_5/2") if entry["label"] == "5d" else ("6p_1/2", "6p_3/2")
        weight = [nominal[label]["two_J"] + 1 for label in labels]
        check(entry["label"] + "/statistical_weights", entry["degeneracy_weights"] != weight, 0)
        for model, rows in (("DC", dc), ("static_Breit", static), ("NIST_nominal", nominal)):
            es = [rows[label]["excitation_cm1"] for label in labels]
            center = (weight[0] * es[0] + weight[1] * es[1]) / sum(weight)
            check(entry["label"] + "/" + model + "/centroid", abs(entry[model]["centroid_cm1"] - center), 1e-7)
            check(entry["label"] + "/" + model + "/splitting", abs(entry[model]["splitting_cm1"] - (es[1] - es[0])), 1e-7)
    check("original_failed_stationarity_visible", result["DC_original_stationarity_passed"] is not False
          or result["inherited_DC_stationarity_evidence"] != contract["inherited_DC_stationarity_evidence"], 0)
    check("no_science_certificate_inherited", any(result.get(name) is not False for name in
          ("stationarity_certified", "production_certified", "experimental_gate_passed", "highest_level_reached")), 0)
    check("declared_single_static_spectrum_no_gradient_optimizer", result["calls"] != {
        "full_H_builds": 1, "full_CI_spectrum_solves": 1, "new_DC_H_builds": 0,
        "new_DC_CI_solves": 0, "gradients": 0, "optimizer_calls": 0}, 0)
    check("no_Atomx_import", any(n == "atomx" or n.startswith("atomx.") for n in sys.modules), 0)
    proof = {"all_passed": all(row["passed"] for row in checks), "checks_count": len(checks), "checks": checks,
             "scope": "Saved complete531 arrays/source/input identity and independent rootwise Rayleigh/residual only",
             "result_sha256": sha(HERE / "result.json"), "arrays_sha256": sha(HERE / "arrays.npz"),
             "contract_sha256": sha(HERE / "contract.json"), "verifier_sha256": sha(Path(__file__)),
             "execution_contract_sha256": sha(args.execution_contract), "Atomx_imports": 0,
             "new_H_builds": 0, "new_CI_solves": 0, "gradients": 0, "optimizer_calls": 0,
             "DC_stationarity_passed": False, "production_certified": False, "experimental_gate_passed": False,
             "actual_CPU_affinity": sorted(os.sched_getaffinity(0)), "address_space_cap_bytes": cap}
    (HERE / "saved_array_verification.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
    signal.alarm(0)
    print(json.dumps({"all_passed": proof["all_passed"], "checks_count": proof["checks_count"]}))
    if not proof["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
