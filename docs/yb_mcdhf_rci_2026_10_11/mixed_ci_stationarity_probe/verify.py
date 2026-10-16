"""Independent saved-array checks, without a third gradient or extra CI solve."""

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = "1"
    contract = json.loads((HERE / "contract.json").read_text())
    result = json.loads((HERE / "result.json").read_text())
    fixed = contract["fixed_conditions"]
    native, metric_root = ROOT / fixed["native_run"], ROOT / fixed["original_metric_run"]
    frozen = ROOT / fixed["frozen_source_root"]
    sys.path.insert(0, str(frozen))
    import numpy as np

    from atomx.constants import HARTREE_CM1
    from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices
    from atomx.potentials.nuclear import make_nuclear_potential

    checks = []

    def check(name, error, limit):
        error = float(error)
        passed = bool(np.isfinite(error) and error <= limit)
        checks.append({"name": name, "error": error, "limit": limit, "passed": passed})
        if not passed:
            raise AssertionError(name)

    def flag(name, passed):
        check(name, 0. if passed else 1., 0.)

    flag("core_probe_68_gates", len(result["checks"]) == 68 and all(c["passed"] for c in result["checks"]))
    for path, expected in result["artifacts_sha256"].items():
        flag("artifact_SHA/" + path, sha(HERE / path) == expected)
    for path, expected in contract["inputs_sha256"].items():
        flag("input_SHA/" + path, sha(ROOT / path) == expected)
    source = json.loads((native / "source_identity.json").read_text())
    for path, expected in source["all_atomx_python_sha256"].items():
        flag("frozen_source_SHA/" + path, sha(frozen / Path(path).relative_to("src")) == expected)
    layout = json.loads((metric_root / "failed_coordinate_layout.json").read_text())
    metric = np.load(metric_root / "failed_coordinates.npz")
    digest = hashlib.sha256()
    for group, row in enumerate(layout):
        digest.update(repr(([tuple(key) for key in row["keys"]], tuple(row["shape"]))).encode())
        digest.update(np.ascontiguousarray(metric[f"projector_{group}"]).tobytes())
    digest.update(metric["inverse_diagonal"].tobytes())
    for group, _ in enumerate(layout):
        for block in metric[f"inverse_blocks_{group}"]:
            digest.update(np.ascontiguousarray(block).tobytes())
    flag("original_complete_metric_SHA", digest.hexdigest() == fixed["original_projector_and_preconditioner_metric_sha256"])
    baseline = np.load(HERE / "baseline_arrays.npz")
    accepted = np.load(HERE / result["one_step"]["evidence_artifact"])
    final_gradient = np.load(HERE / "accepted_gradient.npz")
    directions = np.load(HERE / "derivative_directions.npz")
    rows = result["one_step"]["lower_upper_difference_rows"]
    for name, arrays, field in (("baseline", baseline, "baseline_column"), ("accepted", accepted, "accepted_column")):
        h, e, v = arrays["matrix_Ha"], arrays["energies_Ha"], arrays["mixing"]
        check(name + "/full_CI_residual_Ha", np.max(np.linalg.norm(h @ v - v * e, axis=0)), 1e-8)
        check(name + "/full_signed_CI_Gram", np.max(abs(v.T @ v - np.eye(531))), 1e-10)
        check(name + "/Hermiticity_Ha", np.max(abs(h - h.T)), 1e-10)
        objective = sum(float(e[row[field]]) for row in rows) / 5
        record = result["baseline" if name == "baseline" else "one_step"]
        check(name + "/matched_equal_EO_objective_Ha", abs(objective - record["EOL_objective_Ha"]), 1e-10)
    flag("actual_Armijo_energy_decrease", result["line_search"][0]["accepted"] and
         result["line_search"][0]["EOL_delta_Ha"] <= result["line_search"][0]["same_objective_Armijo_bound_Ha"])
    for name, arrays in (("baseline", baseline), ("accepted", final_gradient)):
        x, g = arrays["x"], arrays["g"]
        projected = np.zeros_like(g)
        for group, row in enumerate(layout):
            where, shape = slice(*row["slice"]), tuple(row["shape"])
            y, gg = x[where].reshape(shape), g[where].reshape(shape)
            action = np.column_stack([block @ gg[:, col] for col, block in enumerate(metric[f"inverse_blocks_{group}"])])
            multiplier = y.T @ action
            projected[where] = (action - y @ (.5 * (multiplier + multiplier.T))).ravel()
        check(name + "/original_metric_action", np.max(abs(projected - arrays["original_metric_preconditioned_gradient"])), 1e-13)
        record = result["baseline" if name == "baseline" else "one_step"]
        check(name + "/original_residual", abs(np.linalg.norm(projected) - record["original_metric_orbital_residual"]), 1e-13)
        check(name + "/quadratic_indicator_Ha", abs(.5 * abs(g @ projected) -
              record["original_metric_quadratic_energy_indicator_Ha"]), 1e-13)
        flag(name + "/truthful_stationarity_failure", not record["stationarity_passed"] and
             record["original_metric_orbital_residual"] > 1e-7 and record["original_metric_quadratic_energy_indicator_Ha"] > 1e-8)
    for row in result["derivative_controls"]:
        direction, step = directions[row["direction"]], row["step"]
        values = []
        for sign in ("plus", "minus"):
            name = f"FD_{row['direction']}_{step}_{sign}"
            trial = next(t for t in result["trials"] if t["name"] == name)
            arrays = np.load(HERE / trial["artifact"])
            value = float(np.asarray(trial["state_weights"]) @ arrays["energies_Ha"])
            check(name + "/actual_re_solved_CI_objective_Ha", abs(value - trial["EOL_objective_Ha"]), 1e-10)
            values.append(value)
        analytic, central = float(baseline["g"] @ direction), (values[0] - values[1]) / (2 * step)
        check("FD_vs_analytic/" + row["direction"] + "/" + str(step),
              abs(analytic - central), row["predeclared_allowance"])
    radial = np.load(native / "orbitals_workspace.npz")
    data = {name[6:]: radial[name].item() if radial[name].ndim == 0 else radial[name]
            for name in radial.files if name.startswith("basis_")}
    waves = {tuple(key): np.concatenate((p, q)) for key, p, q in
             zip(accepted["orbital_keys"], accepted["P"], accepted["Q"], strict=True)}
    for group, row in enumerate(layout):
        keys = [tuple(key) for key in row["keys"]]
        c = np.column_stack([waves[key] for key in keys])
        s = build_dirac_matrices(123, keys[0][1], 70, data, make_nuclear_potential(70, "fermi"))[1]
        check(f"accepted_original_metric_Gram/{group}", np.max(abs(c.T @ s @ c - np.eye(len(keys)))), 1e-10)
    lower = rows[0]
    for row in rows:
        delta = float(accepted["energies_Ha"][row["accepted_column"]] - baseline["energies_Ha"][row["baseline_column"]])
        check("actual_state_delta/" + row["label"], abs(delta - row["absolute_energy_delta_Ha"]), 1e-12)
        transition = (delta - lower["absolute_energy_delta_Ha"]) * HARTREE_CM1
        check("matched_lower_upper_delta_cm1/" + row["label"], abs(transition - row["transition_delta_from_same_lower_cm1"]), 1e-7)
    minima = [min(g["principal_overlap_squared"]) for trial in result["trials"] for g in trial["physical_tracking"]]
    flag("all_perturbed_physical_overlaps_majority", min(minima) > .5)
    flag("no_full_MCDHF_or_experimental_certification", not result["production_certified"] and
         not result["expanded_MCDHF_converged"] and not result["experimental_gate_passed"])
    output = {"all_passed": all(c["passed"] for c in checks), "checks_count": len(checks), "checks": checks,
              "result_sha256": sha(HERE / "result.json"), "source_map_sha256": fixed["source_map_sha256"],
              "original_metric_sha256": digest.hexdigest(), "minimum_trial_physical_overlap_squared": min(minima),
              "scope": "saved-array and public geometry replay; no third gradient or extra CI solve",
              "production_certified": False}
    (HERE / "independent_verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("all_passed", "checks_count", "minimum_trial_physical_overlap_squared", "result_sha256")}))


if __name__ == "__main__":
    main()
