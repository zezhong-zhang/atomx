"""Independent saved-spinor quadrature algebra; no Atomx or H/CI construction."""
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
    contract = load(HERE / "contract.json")
    for name, expected in contract["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Saved failure evidence changed: " + name)
    if (HERE / "verification.json").exists():
        raise FileExistsError("Preserve completed independent failure audit")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 3:
        raise RuntimeError("Original affinity >=3 required")
    os.sched_setaffinity(0, {available[-2]})
    memory_cap = contract["resources"]["address_space_GiB"] * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (memory_cap, memory_cap))

    def timeout(*_):
        raise TimeoutError("Saved-array-only audit wall cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(contract["resources"]["wall_s"])
    import numpy as np

    fixed = load(RUN / "contract.json")
    roles = {name: ROOT / row["path"] for name, row in fixed["input_roles"].items()}
    old = np.load(roles["checkpoint_arrays"], allow_pickle=False)
    original = np.load(roles["original_geometry_arrays"], allow_pickle=False)
    workspace = np.load(roles["geometry_workspace"], allow_pickle=False)
    raw = np.load(RUN / "raw_native_virtuals.npz", allow_pickle=False)
    layout = load(roles["original_layout"])
    failure = load(RUN / "failure.json")
    checks, rows = [], []

    def check(name, error, limit=0):
        error = float(error)
        passed = bool(np.isfinite(error) and error <= limit)
        checks.append(dict(name=name, error=error, limit=limit, passed=passed))
        if not passed:
            raise ValueError("Independent saved-array identity failed: " + name)

    check("failed_invocation_not_three_point_result", (RUN / "result.json").exists())
    check("correct_measured_failure", failure["message"] !=
          "Scientific/numeric gate failed: virtual_original_P_minimum/(6, -3)")
    gates = fixed["acceptance"]
    measured = next(r for r in failure["checks"] if not r["passed"])
    check("unaltered_minimum_gate", measured["limit"] != 1e-8)
    check("minimum_actually_failed", measured["error"] <= measured["limit"])
    for name in ("attempted_complete_CI", "completed_complete_CI", "implicit_H_attempts",
                 "completed_implicit_H", "explicit_H_attempts", "completed_explicit_H", "gradients",
                 "optimizer_calls"):
        check("zero_actual_manyelectron_calls/" + name, failure["calls"][name])
    for name, count in (("completed_native_nuclear_Dirac", 5),
                        ("completed_native_complement_lowest_root", 5),
                        ("completed_initial_one_body_matrices", 7),
                        ("completed_original_P_gap_spectral", 1)):
        check("actual_preflight_calls/" + name, failure["calls"][name] != count)
    check("raw_all5_saved", not np.array_equal(raw["radial_keys"], fixed["fixed_conditions"]["virtual_generator"]["new_keys"]))
    check("raw_vectors_finite", not np.isfinite(raw["P"]).all() or not np.isfinite(raw["Q"]).all())
    ns = int(workspace["basis_ns"])
    weights = workspace["basis_grw"].astype(np.longdouble).ravel()

    def samples(component):
        values = workspace["basis_bsp_" + component].astype(np.longdouble)
        count = int(workspace["basis_ns" + component])
        grid = np.zeros((values.shape[0], values.shape[1], ns), dtype=np.longdouble)
        for interval in range(values.shape[0]):
            width = min(values.shape[2], max(0, count - interval))
            if width:
                grid[interval, :, interval:interval + width] = values[interval, :, :width]
        return grid.reshape(-1, ns)

    bp, bq = samples("p"), samples("q")
    s = np.zeros((2 * ns, 2 * ns), dtype=np.longdouble)
    s[:ns, :ns] = bp.T @ (weights[:, None] * bp)
    s[ns:, ns:] = bq.T @ (weights[:, None] * bq)
    check("positive_quadrature_weights", float(np.min(weights)) <= 0)
    check("quadrature_S_symmetric", np.max(abs(s - s.T)), 1e-18)
    projectors, digest = {}, hashlib.sha256()
    for i, block in enumerate(layout):
        keys = [tuple(key) for key in block["keys"]]
        shape = tuple(block["shape"])
        p = old[f"projector_{i}"]
        check("original_P_bytes/" + str(i), not np.array_equal(p, original[f"projector_{i}"]))
        check("original_inverse_blocks_bytes/" + str(i), not np.array_equal(
              old[f"inverse_blocks_{i}"], original[f"inverse_blocks_{i}"]))
        digest.update(repr((keys, shape)).encode())
        digest.update(np.ascontiguousarray(p).tobytes())
        y = old["x"][slice(*block["slice"])].reshape(shape)
        coefficients = p @ y
        for col, key in enumerate(keys):
            index = [tuple(k) for k in old["orbital_keys"]].index(key)
            check("old_point_exact/" + str(key), not np.array_equal(
                  coefficients[:, col], np.concatenate((old["P"][index], old["Q"][index]))))
        p = p.astype(np.longdouble)
        check("independent_P_SGram/" + str(keys[0][1]),
              np.max(abs(p.T @ s @ p - np.eye(p.shape[1]))), gates["saved_P_metric_Gram_error"])
        projectors[keys[0][1]] = p
    digest.update(old["inverse_diagonal"].tobytes())
    for i in range(len(layout)):
        for block in old[f"inverse_blocks_{i}"]:
            digest.update(np.ascontiguousarray(block).tobytes())
    check("original_inverse_diagonal_bytes", not np.array_equal(
          old["inverse_diagonal"], original["inverse_diagonal"]))
    check("complete_original_metric_SHA", digest.hexdigest() != fixed["fixed_conditions"]["original_metric_sha256"])
    old_keys = [tuple(k) for k in old["orbital_keys"]]
    for i, key in enumerate(tuple(map(int, row)) for row in raw["radial_keys"]):
        v = np.concatenate((raw["P"][i], raw["Q"][i])).astype(np.longdouble)
        p = projectors[key[1]]
        positions = [j for j, old_key in enumerate(old_keys) if old_key[1] == key[1]]
        frozen = np.column_stack([np.concatenate((old["P"][j], old["Q"][j])) for j in positions]).astype(np.longdouble)
        coordinates = p.T @ s @ v
        outside = v - p @ coordinates
        leakage = float(outside @ s @ outside)
        norm = float(v @ s @ v)
        orthogonality = float(np.max(abs(frozen.T @ s @ v)))
        whole = np.column_stack((frozen, v))
        gram_error = float(np.max(abs(whole.T @ s @ whole - np.eye(whole.shape[1]))))
        check("raw_original_P_leakage/" + str(key), leakage, gates["nuclear_projector_leakage_S_norm_squared"])
        check("raw_original_frozen_Gram/" + str(key), gram_error, gates["actual_radial_Gram_error"])
        anchor = int(np.argmax(abs(coordinates)))
        check("nonzero_saved_phase_anchor/" + str(key), coordinates[anchor] == 0)
        rows.append(dict(key=key, quadrature_norm=norm, whole_SGram_error=gram_error,
                         old_frozen_overlap_max=orthogonality, original_P_leakage_S_norm_squared=leakage,
                         original_P_leakage_coefficient_max=float(np.max(abs(outside))),
                         original_P_coordinate_anchor=anchor, raw_phase=float(np.sign(coordinates[anchor])),
                         raw_coefficient_max=float(np.max(abs(v))),
                         original_P_minimum_energy_independently_replayed=False))
    check("no_Atomx_imported", any(name == "atomx" or name.startswith("atomx.") for name in sys.modules))
    signal.alarm(0)
    record = dict(all_independent_saved_array_checks_passed=True, checks=checks, checks_count=len(checks),
                  raw_five_virtuals=rows, failed_original_minimum_gate=measured,
                  original_metric_sha256=digest.hexdigest(), input_sha256=contract["inputs_sha256"],
                  audit_contract_sha256=sha(HERE / "contract.json"), audit_driver_sha256=sha(__file__),
                  new_H_builds=0, CI_solves=0, gradient_calls=0, optimizer_calls=0,
                  new_Dirac_solves=0, new_virtual_constructions=0, Atomx_imports=0,
                  only_new_algebra="Independent longdouble saved local B-grid quadrature S and vector products",
                  cause_not_yet_identified=True, minimum_H_replay_unavailable=True,
                  original_stationarity_passed=False, experimental_gate_passed=False,
                  scope="Saved raw vectors lie in original P and are orthogonal, but minimum mismatch remains unresolved; no repair or projection.")
    (HERE / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(checks=len(checks), all_passed=True, new_H_CI_Dirac_calls=0, vectors=rows)))


if __name__ == "__main__":
    main()
