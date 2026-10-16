"""Replay a retained failed MCDHF line search in its original coordinate metric."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    root = Path(args.input).resolve()
    output = root / "line_search_diagnosis"
    output.mkdir(exist_ok=False)
    names = ("contract.json", "source_identity.json", "failed_line_search.json",
             "failed_coordinates.npz", "failed_coordinate_layout.json", "orbitals_workspace.npz")
    contract = {"hypothesis": "Saved Armijo failure is caused by total-energy cancellation or an inconsistent derivative; replay the exact retained coordinate metric and evaluate symmetric displacements without changing optimizer acceptance.",
        "fixed_conditions": "Same accepted x, signed gradient, inverse blocks, nuclear projector, CSFs and equal EOL weights from the failed solve; no endpoint restart or changed residual metric",
        "inputs_sha256": {name: sha(root / name) for name in names},
        "direction": "the actual final fallback direction, plus unit-normalized version for finite-difference diagnosis",
        "steps": [1.0, 0.5, 0.1, 0.01, -0.01, -0.1, -0.5, -1.0],
        "normalized_direction_steps": [1e-3, -1e-3, 1e-4, -1e-4],
        "acceptance": {"saved_metric_residual_replay_relative_error_lt": 1e-12,
            "saved_analytic_gradient_replay_max_error_lt": 1e-10},
        "stopping": "diagnostics only, never certify original failed solve or widen its numerical gates",
        "diagnostic_script_sha256": sha(Path(__file__)),
        "polar_rule": "Use the retained source epoch's explicit polar helper when present; earlier archived source epochs implement the same eig polar inline."}
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    shutil.copyfile(Path(__file__), output / "diagnostic_source.py")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = "1"
    sys.path.insert(0, str(root / "source"))
    import numpy as np

    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.numerics.bspline import orbital_optimization as radial_optimizer
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.potentials.nuclear import make_nuclear_potential

    started = time.monotonic()
    original = json.loads((root / "contract.json").read_text())
    model = original["fixed_conditions"]
    csfs = [csf for config in model["configs"] for csf in enumerate_jj_csfs(config)]
    workspace = np.load(root / "orbitals_workspace.npz")
    data = {name[6:]: workspace[name].item() if workspace[name].ndim == 0 else workspace[name]
            for name in workspace.files if name.startswith("basis_")}
    functional = CSFOrbitalFunctional(csfs, data, 70, make_nuclear_potential(70, "fermi"),
        state_average=CIStateAverage(tuple(tuple(row) for row in model["selected_levels"])))
    saved = np.load(root / "failed_coordinates.npz")
    layout = json.loads((root / "failed_coordinate_layout.json").read_text())
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
    def retract(point):
        result = point.copy()
        for group in layout:
            where = slice(*group["slice"])
            y = point[where].reshape(group["shape"])
            if "_polar_retraction" in vars(radial_optimizer):
                result[where] = radial_optimizer._polar_retraction(y).ravel()
            else:
                values, vectors = np.linalg.eigh(y.T @ y)
                result[where] = (y @ ((vectors / np.sqrt(values)) @ vectors.T)).ravel()
        return result
    def unpack(point):
        orbitals = {}
        for index, group in enumerate(layout):
            where = slice(*group["slice"])
            coefficients = saved[f"projector_{index}"] @ point[where].reshape(group["shape"])
            for column, key in enumerate(group["keys"]):
                orbitals[tuple(key)] = (coefficients[: functional.ns, column].copy(),
                    coefficients[functional.ns :, column].copy())
        return orbitals
    def gradient_at(point, gradient):
        result = np.zeros_like(point)
        for index, group in enumerate(layout):
            derivatives = np.column_stack([gradient[tuple(key)] for key in group["keys"]])
            result[slice(*group["slice"])] = (saved[f"projector_{index}"].T @ derivatives).ravel()
        return tangent(point, result)
    inverse = np.zeros_like(g)
    for index, group in enumerate(layout):
        where = slice(*group["slice"])
        derivative = g[where].reshape(group["shape"])
        blocks = saved[f"inverse_blocks_{index}"]
        inverse[where] = np.column_stack([block @ derivative[:, column]
            for column, block in enumerate(blocks)]).ravel()
    preconditioned = tangent(x, inverse)
    reference_orbitals = unpack(x)
    base_energy, base_gradient, base_ci = functional.evaluate(reference_orbitals)
    analytic = gradient_at(x, base_gradient)
    samples = []
    for direction_name, vector, steps in (
        ("actual_fallback", direction, contract["steps"]),
        ("unit_fallback", direction / np.linalg.norm(direction), contract["normalized_direction_steps"]),
    ):
        derivative = float(g @ vector)
        for step in steps:
            candidate = retract(x + step * vector)
            trial_orbitals = unpack(candidate)
            energy, radial_gradient, trial_ci = functional.evaluate(trial_orbitals)
            candidate_gradient = gradient_at(candidate, radial_gradient)
            delta = candidate - x
            trapezoid_integral = float(.5 * (g + candidate_gradient) @ delta)
            complete_change = (functional.fixed_density_energy_difference(
                reference_orbitals, trial_orbitals, base_ci, trial_ci)
                if "fixed_density_energy_difference" in vars(CSFOrbitalFunctional) else None)
            constrained_change = (functional.constrained_energy_difference(
                reference_orbitals, trial_orbitals, base_ci, trial_ci,
                increments=unpack(candidate - x))
                if "constrained_energy_difference" in vars(CSFOrbitalFunctional) else None)
            samples.append({"direction": direction_name, "step": step,
                "energy_Ha": energy, "energy_minus_base_Ha": energy - base_energy,
                "complete_polynomial_difference_Ha": complete_change,
                "constrained_polynomial_difference_Ha": constrained_change,
                "directional_derivative_Ha": derivative,
                "linear_prediction_Ha": step * derivative,
                "gradient_trapezoid_energy_diff_Ha": trapezoid_integral,
                "candidate_gradient_norm": float(np.linalg.norm(candidate_gradient)),
                "candidate_vs_base_coordinate_change": float(np.linalg.norm(delta))})
    rows = json.loads((root / "failed_line_search.json").read_text())
    failed_iteration = max(row["iteration"] for row in rows)
    final_trials = [row for row in rows if row["iteration"] == failed_iteration]
    retained_residual = final_trials[0]["base_orbital_residual"]
    replay_residual = float(np.linalg.norm(preconditioned))
    result = {"same_retained_metric_replay_residual": replay_residual,
        "retained_original_residual": retained_residual,
        "residual_relative_error": abs(replay_residual-retained_residual)/retained_residual,
        "analytic_gradient_replay_max_error": float(np.max(abs(analytic-g))),
        "analytic_gradient_norm": float(np.linalg.norm(g)),
        "final_fallback_direction_norm": float(np.linalg.norm(direction)),
        "fallback_direction_dot_gradient": float(g @ direction),
        "base_energy_Ha": base_energy, "total_energy_float64_spacing_Ha": float(abs(np.spacing(base_energy))),
        "failed_iteration": failed_iteration, "final_trial_count": len(final_trials),
        "final_trials": final_trials, "samples": samples,
        "seconds_uncontrolled": time.monotonic() - started,
        "certifies_original_solve": False, "source_epoch": sha(root / "source_identity.json")}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("same_retained_metric_replay_residual",
        "analytic_gradient_replay_max_error", "fallback_direction_dot_gradient",
        "total_energy_float64_spacing_Ha", "final_trial_count", "seconds_uncontrolled")}))


if __name__ == "__main__":
    main()
