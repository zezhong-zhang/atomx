"""Replay the retained endpoint with its frozen stable-difference source epoch."""

import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    owner = Path(__file__).resolve().parent
    old = owner / "nuclear_eol002_trace"
    out = owner / "stable_delta_endpoint"
    contract = json.loads((out / "contract.json").read_text())
    for path, digest in contract["source_sha256"].items():
        source = out / "source" / Path(path).relative_to("src")
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"source identity mismatch: {path}")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0, str(out / "source"))
    import numpy as np

    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.potentials.nuclear import make_nuclear_potential

    model = json.loads((old / "contract.json").read_text())["fixed_conditions"]
    csfs = [csf for config in model["configs"] for csf in enumerate_jj_csfs(config)]
    workspace = np.load(old / "orbitals_workspace.npz")
    data = {name[6:]: workspace[name].item() if workspace[name].ndim == 0 else workspace[name]
            for name in workspace.files if name.startswith("basis_")}
    functional = CSFOrbitalFunctional(csfs, data, 70, make_nuclear_potential(70, "fermi"),
        state_average=CIStateAverage(tuple(tuple(row) for row in model["selected_levels"])))
    coordinates = np.load(old / "failed_coordinates.npz")
    layout = json.loads((old / "failed_coordinate_layout.json").read_text())
    x, g, direction = (coordinates[name] for name in ("x", "g", "direction"))
    def unpack(point):
        orbitals = {}
        for index, group in enumerate(layout):
            coefficients = coordinates[f"projector_{index}"] @ point[slice(*group["slice"])].reshape(group["shape"])
            for column, key in enumerate(group["keys"]):
                orbitals[tuple(key)] = (coefficients[:functional.ns, column],
                                       coefficients[functional.ns:, column])
        return orbitals
    def retract(point):
        result = point.copy()
        for group in layout:
            where = slice(*group["slice"])
            y = point[where].reshape(group["shape"])
            values, vectors = np.linalg.eigh(y.T @ y)
            result[where] = (y @ ((vectors / np.sqrt(values)) @ vectors.T)).ravel()
        return result
    reference = unpack(x)
    energy, gradient, ci = functional.evaluate(reference)
    replay_gradient = np.zeros_like(g)
    for index, group in enumerate(layout):
        where = slice(*group["slice"])
        projection = coordinates[f"projector_{index}"].T @ np.column_stack(
            [gradient[tuple(key)] for key in group["keys"]])
        projection = projection.reshape(group["shape"])
        y = x[where].reshape(group["shape"])
        multiplier = y.T @ projection
        replay_gradient[where] = (projection - y @ (.5 * (multiplier + multiplier.T))).ravel()
    rows = []
    for step in contract["steps"]:
        trial = unpack(retract(x + step * direction))
        trial_energy, trial_ci = functional.evaluate(trial, gradient=False)
        forward = functional.fixed_density_energy_difference(reference, trial, ci, trial_ci)
        reverse = functional.fixed_density_energy_difference(trial, reference, trial_ci, ci)
        rows.append({"step": step, "direct_difference_Ha": trial_energy - energy,
            "exact_difference_Ha": forward, "reverse_exact_difference_Ha": reverse,
            "antisymmetry_error_Ha": abs(forward + reverse),
            "linear_prediction_Ha": float(step * g @ direction)})
    retained = json.loads((out / "result.json").read_text())
    checks = {"all_source_sha256_verified": len(contract["source_sha256"]),
        "gradient_replay_max_error": float(np.max(abs(replay_gradient - g))),
        "retained_rows_reproduced_exactly": rows == retained["rows"],
        "replay_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_epoch_bound": True, "certifies_original_failed_solve": False}
    if checks["gradient_replay_max_error"] > 1e-10 or not checks["retained_rows_reproduced_exactly"]:
        raise ValueError(checks)
    (out / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks))


if __name__ == "__main__":
    main()
