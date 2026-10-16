"""Compare frozen polar implementations at an unchanged failed Yb endpoint."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="stable_polar_endpoint")
    args = parser.parse_args()
    owner = Path(__file__).resolve().parent
    repo = owner.parents[2]
    retained = owner / "nuclear_eol004_stable_delta_trace"
    out = owner / args.label
    out.mkdir(exist_ok=False)
    sources = {}
    for source in sorted((repo / "src/atomx").rglob("*.py")):
        target = out / "source" / source.relative_to(repo / "src")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[str(source.relative_to(repo))] = sha(source)
    bindings = ("contract.json", "source_identity.json", "failed_coordinates.npz",
                "failed_coordinate_layout.json", "failed_line_search.json",
                "orbitals_workspace.npz")
    contract = {
        "hypothesis": "Nearly unit Gram eigendecomposition introduces roundoff-sized coordinate changes that alter tiny retained objective decreases; evaluate the same polar map with the new wide Gram correction, without restarting or certifying the failed solve.",
        "fixed_conditions": "Identical accepted endpoint, original nuclear projector and spectral inverse blocks, original gradient, five one-CSF equal-weight states, 24 common radial orbitals, identical objective and unchanged numerical gates.",
        "retained_source_inputs": {name: sha(retained / name) for name in bindings},
        "source_sha256": sources,
        "source_map_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "retraction_source_contract": {"path": "docs/yb_mcdhf_rci_2026_10_11/stable_polar/contract.json",
                                       "sha256": sha(owner.parent / "stable_polar/contract.json")},
        "steps": [0., 1., .5, .1, .01, -.01, -.1, 2. ** -10, 2. ** -20, 2. ** -27],
        "acceptance": {"original_analytic_gradient_replay_max_error_lt": 1e-10,
                       "original_metric_residual_relative_error_lt": 1e-12,
                       "polar_orthogonality_error_lt": 1e-12},
        "stopping": "Diagnostic comparison only; preserve failed solve and do not change original stationarity tolerances or certify an endpoint restart.",
        "additional_angular_audit": "Record prefilter objective coefficients, including any nonzero value below or equal to 1e-15; fixed retained RDM alone does not bound the complete angular filter."}
    write(out / "contract.json", contract)
    shutil.copyfile(Path(__file__), out / "diagnostic_source.py")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = "1"
    sys.path.insert(0, str(out / "source"))
    import numpy as np

    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.numerics.bspline.orbital_optimization import _polar_retraction
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.potentials.nuclear import make_nuclear_potential

    started = time.monotonic()
    model = json.loads((retained / "contract.json").read_text())["fixed_conditions"]
    csfs = [csf for config in model["configs"] for csf in enumerate_jj_csfs(config)]
    workspace = np.load(retained / "orbitals_workspace.npz")
    data = {name[6:]: workspace[name].item() if workspace[name].ndim == 0 else workspace[name]
            for name in workspace.files if name.startswith("basis_")}
    functional = CSFOrbitalFunctional(csfs, data, 70, make_nuclear_potential(70, "fermi"),
        state_average=CIStateAverage(tuple(tuple(row) for row in model["selected_levels"])))
    saved = np.load(retained / "failed_coordinates.npz")
    layout = json.loads((retained / "failed_coordinate_layout.json").read_text())
    x, g, direction = (saved[name] for name in ("x", "g", "direction"))

    def tangent(point, vector):
        result = vector.copy()
        for group in layout:
            where = slice(*group["slice"])
            y = point[where].reshape(group["shape"])
            derivative = vector[where].reshape(group["shape"])
            multiplier = y.T @ derivative
            result[where] = (derivative - y @ (.5 * (multiplier + multiplier.T))).ravel()
        return result

    def old_polar(y):
        values, vectors = np.linalg.eigh(y.T @ y)
        return y @ ((vectors / np.sqrt(values)) @ vectors.T)

    def retract(point, polar):
        result = point.copy()
        for group in layout:
            where = slice(*group["slice"])
            result[where] = polar(point[where].reshape(group["shape"])).ravel()
        return result

    def unpack(point):
        orbitals = {}
        for index, group in enumerate(layout):
            coefficients = saved[f"projector_{index}"] @ point[slice(*group["slice"])].reshape(group["shape"])
            for column, key in enumerate(group["keys"]):
                orbitals[tuple(key)] = (coefficients[:functional.ns, column].copy(),
                                       coefficients[functional.ns:, column].copy())
        return orbitals

    def projected_gradient(point, gradient):
        result = np.zeros_like(point)
        for index, group in enumerate(layout):
            radial = np.column_stack([gradient[tuple(key)] for key in group["keys"]])
            result[slice(*group["slice"])] = (saved[f"projector_{index}"].T @ radial).ravel()
        return tangent(point, result)

    inverse = np.zeros_like(g)
    for index, group in enumerate(layout):
        where = slice(*group["slice"])
        derivative = g[where].reshape(group["shape"])
        inverse[where] = np.column_stack([block @ derivative[:, column]
            for column, block in enumerate(saved[f"inverse_blocks_{index}"])]).ravel()
    residual = float(np.linalg.norm(tangent(x, inverse)))
    reference = unpack(x)
    energy, gradient, ci = functional.evaluate(reference)
    base_gradient = projected_gradient(x, gradient)
    gradient_error = float(np.max(np.abs(base_gradient - g)))
    state_weights = functional.state_average.weights(csfs, ci)
    raw_nonzero = []
    tiny = []
    angular_blocks = []
    for structure, projection, columns in functional.radial_density.blocks:
        selected = state_weights > 0
        amplitudes = projection @ ci.mixing_coefficients[columns][:, selected]
        products = amplitudes[structure.pairs[:, 0]] * amplitudes[structure.pairs[:, 1]]
        coefficients = np.asarray(structure.table.T @ (products @ state_weights[selected])).ravel()
        nonzero = coefficients[coefficients != 0]
        small = coefficients[(coefficients != 0) & (np.abs(coefficients) <= 1e-15)]
        raw_nonzero.extend(nonzero.tolist())
        tiny.extend(small.tolist())
        angular_blocks.append({"columns": columns, "coefficient_count": len(coefficients),
            "nonzero_count": len(nonzero), "nonzero_le_1e15_count": len(small),
            "max_small_abs": float(np.max(np.abs(small))) if len(small) else 0.})
    rows = []
    for step in contract["steps"]:
        points = {}
        for name, polar in (("eigh_original", old_polar), ("wide_gram_correction", _polar_retraction)):
            candidate = retract(x + step * direction, polar)
            points[name] = candidate
            trial = unpack(candidate)
            trial_energy, trial_gradient, trial_ci = functional.evaluate(trial)
            actual_gradient = projected_gradient(candidate, trial_gradient)
            delta = candidate - x
            exact = functional.fixed_density_energy_difference(reference, trial, ci, trial_ci)
            reverse = functional.fixed_density_energy_difference(trial, reference, trial_ci, ci)
            orthogonality = max(float(np.max(np.abs(
                candidate[slice(*group["slice"])].reshape(group["shape"]).T @
                candidate[slice(*group["slice"])].reshape(group["shape"]) -
                np.eye(group["shape"][1])))) for group in layout)
            rows.append({"step": step, "retraction": name,
                "direct_difference_Ha": trial_energy - energy,
                "exact_difference_Ha": exact, "antisymmetry_error_Ha": abs(exact + reverse),
                "linear_prediction_Ha": float(step * g @ direction),
                "gradient_trapezoid_diagnostic_Ha": float(.5 * (g + actual_gradient) @ delta),
                "gradient_norm": float(np.linalg.norm(actual_gradient)),
                "orthogonality_error": orthogonality,
                "coordinate_displacement_norm": float(np.linalg.norm(delta)),
                "coordinate_displacement_max": float(np.max(np.abs(delta)))})
        rows[-1]["candidate_difference_between_retractions_norm"] = float(
            np.linalg.norm(points["eigh_original"] - points["wide_gram_correction"]))
    retained_rows = json.loads((retained / "failed_line_search.json").read_text())
    final_iteration = max(row["iteration"] for row in retained_rows)
    retained_residual = next(row["base_orbital_residual"] for row in retained_rows
                            if row["iteration"] == final_iteration)
    checks = {"original_analytic_gradient_replay_max_error": gradient_error,
        "original_metric_residual": residual, "retained_metric_residual": retained_residual,
        "metric_residual_relative_error": abs(residual - retained_residual) / retained_residual,
        "all_trial_orthogonality_passed": all(row["orthogonality_error"] < 1e-12 for row in rows)}
    result = {"checks": checks, "base_energy_Ha": energy,
        "directional_derivative_Ha": float(g @ direction), "rows": rows,
        "angular_audit": {"blocks": angular_blocks, "raw_nonzero_count": len(raw_nonzero),
            "nonzero_le_1e15_count": len(tiny),
            "max_dropped_abs": max(map(abs, tiny), default=0.),
            "min_nonzero_abs": min(map(abs, raw_nonzero), default=0.)},
        "seconds_uncontrolled": time.monotonic() - started,
        "diagnostic_only": True, "certifies_original_solve": False}
    write(out / "result.json", result)
    if (gradient_error > 1e-10 or checks["metric_residual_relative_error"] > 1e-12
            or not checks["all_trial_orthogonality_passed"]):
        raise ValueError(checks)
    write(out / "verification.json", {"all_source_sha256_verified": len(sources),
        "gradient_and_original_metric_replay_passed": True,
        "replay_script_sha256": sha(Path(__file__)), "result_sha256": sha(out / "result.json"),
        "source_epoch_bound": True, "certifies_original_solve": False})
    print(json.dumps({"checks": checks, "angular_audit": result["angular_audit"],
                      "seconds_uncontrolled": result["seconds_uncontrolled"]}))


if __name__ == "__main__":
    main()
