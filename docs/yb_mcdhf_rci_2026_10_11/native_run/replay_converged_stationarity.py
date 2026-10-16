"""Independently replay a completed run in its hash-identical original metric."""

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
    parser.add_argument("--label", default="stationarity_replay")
    args = parser.parse_args()
    owner = Path(__file__).resolve().parent
    run = owner / "nuclear_eol005_projected_increment"
    metric_source = owner / "nuclear_eol004_stable_delta_trace"
    out = run / args.label
    out.mkdir(exist_ok=False)
    contract = {
        "hypothesis": "The converged public native solve must independently pass its original spectral metric and analytic gradient, rather than a restarted endpoint's changed preconditioner.",
        "fixed_conditions": "Same final actual 005 radial spinors, states, cell integrals and frozen source; original projectors, inverse diagonal and inverse blocks retained in 004 are admissible only if their complete metric hash matches the 005 diagnostic.",
        "source_identity_sha256": sha(run / "source_identity.json"),
        "inputs_sha256": {"result": sha(run / "result.json"),
            "final_spinors": sha(run / "orbitals_workspace.npz"),
            "final_independent_gradient": sha(run / "ci_arrays.npz"),
            "original_metric": sha(metric_source / "failed_coordinates.npz"),
            "original_layout": sha(metric_source / "failed_coordinate_layout.json")},
        "coordinate_recovery": "Recover coordinates with original P.T S C in wide precision; evaluate the gradient on actual final stored C, not reconstructed spinors. Report forward reconstruction error explicitly because the exact successful optimizer x was not separately archived.",
        "acceptance": {"original_metric_hash_equal": True,
            "actual_radial_gradient_replay_max_error_lt": 1e-10,
            "coordinate_forward_reconstruction_max_error_lt": 1e-12,
            "original_metric_orbital_residual_lt": 1e-7,
            "estimated_energy_error_Ha_lt": 1e-8},
        "stopping": "One final state evaluation only; no endpoint solve, changed metric, physical model or gate."}
    write(out / "contract.json", contract)
    shutil.copyfile(Path(__file__), out / "diagnostic_source.py")
    source = json.loads((run / "source_identity.json").read_text())
    for path, digest in source["all_atomx_python_sha256"].items():
        assert sha(run / "source" / Path(path).relative_to("src")) == digest
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = "1"
    sys.path.insert(0, str(run / "source"))
    import numpy as np

    from atomx.config.csf import enumerate_jj_csfs
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.potentials.nuclear import make_nuclear_potential

    started = time.monotonic()
    final = json.loads((run / "result.json").read_text())
    model = json.loads((run / "contract.json").read_text())["fixed_conditions"]
    csfs = [csf for config in model["configs"] for csf in enumerate_jj_csfs(config)]
    workspace = np.load(run / "orbitals_workspace.npz")
    data = {name[6:]: workspace[name].item() if workspace[name].ndim == 0 else workspace[name]
            for name in workspace.files if name.startswith("basis_")}
    functional = CSFOrbitalFunctional(csfs, data, 70, make_nuclear_potential(70, "fermi"),
        state_average=CIStateAverage(tuple(tuple(row) for row in model["selected_levels"])))
    orbitals = {tuple(key): (p.copy(), q.copy()) for key, p, q in
                zip(workspace["orbital_keys"], workspace["P"], workspace["Q"], strict=True)}
    saved = np.load(metric_source / "failed_coordinates.npz")
    layout = json.loads((metric_source / "failed_coordinate_layout.json").read_text())
    digest = hashlib.sha256()
    for index, group in enumerate(layout):
        keys, shape = [tuple(key) for key in group["keys"]], tuple(group["shape"])
        digest.update(repr((keys, shape)).encode())
        digest.update(np.ascontiguousarray(saved[f"projector_{index}"]).tobytes())
    digest.update(saved["inverse_diagonal"].tobytes())
    for index, _ in enumerate(layout):
        for block in saved[f"inverse_blocks_{index}"]:
            digest.update(np.ascontiguousarray(block).tobytes())
    actual_metric_sha = digest.hexdigest()
    if actual_metric_sha != final["orbital_optimization"]["residual_metric_sha256"]:
        raise ValueError("retained original metric does not match final optimizer metric")
    energy, gradient, _ = functional.evaluate(orbitals)
    previous = np.load(run / "ci_arrays.npz")
    radial_gradient_error = max(float(np.max(np.abs(gradient[tuple(key)] - expected)))
        for key, expected in zip(previous["gradient_keys"], previous["gradient"], strict=True))
    x, g = np.zeros_like(saved["x"]), np.zeros_like(saved["g"])
    reconstruction_error = 0.
    for index, group in enumerate(layout):
        where = slice(*group["slice"])
        keys = [tuple(key) for key in group["keys"]]
        p = saved[f"projector_{index}"]
        coefficients = np.column_stack([np.concatenate(orbitals[key]) for key in keys])
        y = np.asarray(p.astype(np.longdouble).T @ functional.metric[keys[0][1]].astype(np.longdouble)
                       @ coefficients.astype(np.longdouble), dtype=float)
        x[where] = y.ravel()
        reconstruction_error = max(reconstruction_error, float(np.max(np.abs(p @ y - coefficients))))
        derivative = p.T @ np.column_stack([gradient[key] for key in keys])
        multiplier = y.T @ derivative
        g[where] = (derivative - y @ (.5 * (multiplier + multiplier.T))).ravel()
    inverse = np.zeros_like(g)
    for index, group in enumerate(layout):
        where = slice(*group["slice"])
        derivative = g[where].reshape(group["shape"])
        inverse[where] = np.column_stack([block @ derivative[:, column]
            for column, block in enumerate(saved[f"inverse_blocks_{index}"])]).ravel()
        y = x[where].reshape(group["shape"])
        matrix = inverse[where].reshape(group["shape"])
        multiplier = y.T @ matrix
        inverse[where] = (matrix - y @ (.5 * (multiplier + multiplier.T))).ravel()
    residual = float(np.linalg.norm(inverse))
    energy_error = float(.5 * g @ inverse)
    np.savez_compressed(out / "recovered_coordinates.npz", x=x, g=g, preconditioned_gradient=inverse)
    result = {"original_metric_sha256": actual_metric_sha,
        "same_original_metric_hash": True, "source_files_verified": len(source["all_atomx_python_sha256"]),
        "actual_radial_gradient_replay_max_error": radial_gradient_error,
        "coordinate_forward_reconstruction_max_error": reconstruction_error,
        "objective_replay_Ha": energy, "objective_replay_error_Ha": energy - final["objective_replay_Ha"],
        "original_metric_residual_after_coordinate_recovery": residual,
        "reported_exact_coordinate_residual": final["orbital_optimization"]["orbital_residual"],
        "residual_recovery_absolute_difference": residual - final["orbital_optimization"]["orbital_residual"],
        "estimated_energy_error_Ha": energy_error,
        "coordinate_recovery_is_not_bitwise_original_x": True,
        "stationarity_gate_replayed": residual < 1e-7 and energy_error < 1e-8,
        "seconds_uncontrolled": time.monotonic() - started,
        "experimental_certification": False}
    write(out / "result.json", result)
    if not (radial_gradient_error < 1e-10 and reconstruction_error < 1e-12
            and result["stationarity_gate_replayed"]):
        raise ValueError("independent original-metric stationarity replay failed")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
