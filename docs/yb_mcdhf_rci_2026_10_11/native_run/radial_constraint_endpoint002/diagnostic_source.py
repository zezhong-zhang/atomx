"""Measure radial unpack defects and angular-filter increments at a saved endpoint."""

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
    parser.add_argument("--label", default="radial_constraint_endpoint")
    args = parser.parse_args()
    owner = Path(__file__).resolve().parent
    retained = owner / "nuclear_eol004_stable_delta_trace"
    epoch = owner / "stable_polar_endpoint002"
    out = owner / args.label
    out.mkdir(exist_ok=False)
    contract = {
        "hypothesis": "Float64 projector-to-radial multiplication leaves order-1e-15 metric defects whose normal Lagrange contribution can exceed the final tangent descent; angular coefficient filtering is an independent possible discrepancy and needs a paired exact increment control.",
        "fixed_conditions": "Saved final 004 endpoint, original coordinate gradient/projectors/inverse metric, identical five-state EOL and same complete cell integrals. No solver restart, modified objective acceptance or tolerance change.",
        "inputs_sha256": {name: sha(retained / name) for name in
            ("contract.json", "failed_coordinates.npz", "failed_coordinate_layout.json", "orbitals_workspace.npz", "failed_line_search.json")},
        "source_epoch_contract_sha256": sha(epoch / "contract.json"),
        "steps": [0., .1, .01, 2. ** -27],
        "controls": ["Actual old eig polar and new wide Gram polar",
            "Actual radial float64 coefficients and longdouble S Gram using the same stored metric",
            "Raw Euclidean gradient and normalized one-body quotient derivative for off-manifold diagnostic",
            "Same candidate complete exact increment with retained angular RDM and unfiltered raw objective coefficients"],
        "acceptance": "Original saved analytic gradient replay error below 1e-10; trial P/Q retained. Normal linear contribution is diagnostic and never replaces actual objective or stationarity tests.",
        "stopping": "One unchanged endpoint, eight actual candidates, no iterative optimization, no certification of the failed solve."}
    write(out / "contract.json", contract)
    shutil.copyfile(Path(__file__), out / "diagnostic_source.py")
    source_contract = json.loads((epoch / "contract.json").read_text())
    for path, digest in source_contract["source_sha256"].items():
        assert sha(epoch / "source" / Path(path).relative_to("src")) == digest
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[variable] = "1"
    sys.path.insert(0, str(epoch / "source"))
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

    def unpack(point):
        orbitals = {}
        for index, group in enumerate(layout):
            coefficients = saved[f"projector_{index}"] @ point[slice(*group["slice"])].reshape(group["shape"])
            for column, key in enumerate(group["keys"]):
                orbitals[tuple(key)] = (coefficients[:functional.ns, column].copy(),
                                       coefficients[functional.ns:, column].copy())
        return orbitals

    def retract(point, polar):
        result = point.copy()
        for group in layout:
            where = slice(*group["slice"])
            result[where] = polar(point[where].reshape(group["shape"])).ravel()
        return result

    def old_polar(y):
        values, vectors = np.linalg.eigh(y.T @ y)
        return y @ ((vectors / np.sqrt(values)) @ vectors.T)

    def raw_contract(ci, state_weights):
        weights = {}
        selected = state_weights > 0
        for structure, projection, columns in functional.radial_density.blocks:
            amplitudes = projection @ ci.mixing_coefficients[columns][:, selected]
            products = amplitudes[structure.pairs[:, 0]] * amplitudes[structure.pairs[:, 1]]
            coefficients = np.asarray(structure.table.T @ (products @ state_weights[selected])).ravel()
            for key, coefficient in zip(structure.keys, coefficients, strict=True):
                if coefficient != 0:
                    weights[key] = weights.get(key, 0.) + coefficient
        return weights

    reference = unpack(x)
    energy, gradient, ci = functional.evaluate(reference)
    weights = functional.radial_density.contract(ci, functional.state_average.weights(csfs, ci))
    quotient_gradient = {key: value.copy() for key, value in gradient.items()}
    # CI uses h_ab / (b.T S b), so only the off-manifold diagnostic
    # derivative differs; all retained tangent derivatives keep the old gate.
    for term, coefficient in weights.items():
        if term[0] != "h":
            continue
        _, na, kappa, nb = term
        a, b = (na, kappa), (nb, kappa)
        va, vb = np.concatenate(reference[a]), np.concatenate(reference[b])
        h, metric = functional.one_body[kappa], functional.metric[kappa]
        norm = float(vb @ metric @ vb)
        h_value = float(va @ h @ vb)
        quotient_gradient[a] += coefficient * (1. / norm - 1.) * (h @ vb)
        quotient_gradient[b] += coefficient * ((1. / norm - 1.) * (h @ va)
            - 2. * h_value / norm ** 2 * (metric @ vb))
    projected = np.zeros_like(g)
    for index, group in enumerate(layout):
        where = slice(*group["slice"])
        y = x[where].reshape(group["shape"])
        raw = saved[f"projector_{index}"].T @ np.column_stack([gradient[tuple(key)] for key in group["keys"]])
        multiplier = y.T @ raw
        projected[where] = (raw - y @ (.5 * (multiplier + multiplier.T))).ravel()
    gradient_error = float(np.max(np.abs(projected - g)))
    radial_arrays = {}
    for key, values in reference.items():
        radial_arrays[f"base_{key[0]}_{key[1]}"] = np.concatenate(values)
    rows = []
    for step_index, step in enumerate(contract["steps"]):
        for name, polar in (("eigh_original", old_polar), ("wide_gram_correction", _polar_retraction)):
            candidate = retract(x + step * direction, polar)
            trial = unpack(candidate)
            trial_energy, trial_ci = functional.evaluate(trial, gradient=False)
            exact = functional.fixed_density_energy_difference(reference, trial, ci, trial_ci)
            retained_contract = functional.radial_density.contract
            functional.radial_density.contract = raw_contract
            try:
                unfiltered = functional.fixed_density_energy_difference(reference, trial, ci, trial_ci)
            finally:
                functional.radial_density.contract = retained_contract
            normal_raw, normal_quotient = np.longdouble(0), np.longdouble(0)
            defect_raw, defect_quotient = np.longdouble(0), np.longdouble(0)
            full_raw, full_quotient = np.longdouble(0), np.longdouble(0)
            groups = []
            for group in layout:
                keys = [tuple(key) for key in group["keys"]]
                kappa = keys[0][1]
                base = np.column_stack([np.concatenate(reference[key]) for key in keys]).astype(np.longdouble)
                new = np.column_stack([np.concatenate(trial[key]) for key in keys]).astype(np.longdouble)
                delta = new - base
                metric = functional.metric[kappa].astype(np.longdouble)
                base_gram, trial_gram = base.T @ metric @ base, new.T @ metric @ new
                cross = base.T @ metric @ delta
                normal = .5 * (cross + cross.T)
                # The quadratic curvature belongs to the physical polar
                # constraint. Only the remaining actual Gram increment is
                # a defect; never count the required curvature as noise.
                defect_change = cross + cross.T + delta.T @ metric @ delta
                raw = np.column_stack([gradient[key] for key in keys]).astype(np.longdouble)
                quotient = np.column_stack([quotient_gradient[key] for key in keys]).astype(np.longdouble)
                raw_lagrange, quotient_lagrange = base.T @ raw, base.T @ quotient
                normal_raw += np.sum(raw_lagrange * normal)
                normal_quotient += np.sum(quotient_lagrange * normal)
                defect_raw += .5 * np.sum(raw_lagrange * defect_change)
                defect_quotient += .5 * np.sum(quotient_lagrange * defect_change)
                full_raw += np.sum(raw * delta)
                full_quotient += np.sum(quotient * delta)
                groups.append({"kappa": kappa,
                    "base_metric_defect_max_wide": float(np.max(np.abs(base_gram - np.eye(len(keys))))),
                    "trial_metric_defect_max_wide": float(np.max(np.abs(trial_gram - np.eye(len(keys))))),
                    "metric_gram_change_max_wide": float(np.max(np.abs(trial_gram - base_gram))),
                    "normal_increment_max": float(np.max(np.abs(normal)))})
                for key in keys:
                    radial_arrays[f"delta_{step_index}_{name}_{key[0]}_{key[1]}"] = (
                        np.concatenate(trial[key]) - np.concatenate(reference[key]))
            rows.append({"step": step, "retraction": name, "groups": groups,
                "retained_exact_difference_Ha": exact,
                "unfiltered_exact_difference_Ha": unfiltered,
                "angular_filter_increment_difference_Ha": unfiltered - exact,
                "raw_euclidean_linear_Ha": float(full_raw),
                "one_body_quotient_euclidean_linear_Ha": float(full_quotient),
                "raw_normal_lagrange_linear_Ha": float(normal_raw),
                "quotient_normal_lagrange_linear_Ha": float(normal_quotient),
                "raw_normal_metric_defect_Ha": float(defect_raw),
                "quotient_normal_metric_defect_Ha": float(defect_quotient),
                "exact_minus_quotient_metric_defect_Ha": exact - float(defect_quotient),
                "exact_minus_quotient_normal_Ha": exact - float(normal_quotient),
                "coordinate_tangent_linear_Ha": float(g @ (candidate - x)),
                "coordinate_linear_descent_Ha": float(step * g @ direction),
                "direct_total_energy_difference_Ha": trial_energy - energy})
    np.savez_compressed(out / "actual_radial_increments.npz", **radial_arrays)
    result = {"gradient_replay_max_error": gradient_error, "base_energy_Ha": energy,
        "rows": rows, "longdouble_bits": np.finfo(np.longdouble).nmant,
        "seconds_uncontrolled": time.monotonic() - started,
        "actual_radial_increments_sha256": sha(out / "actual_radial_increments.npz"),
        "source_files_verified": len(source_contract["source_sha256"]),
        "all_measurements_diagnostic": True, "solver_acceptance_unchanged": True,
        "production_certified": False, "certifies_original_solve": False}
    write(out / "result.json", result)
    if gradient_error > 1e-10:
        raise ValueError("original coordinate gradient replay failed")
    print(json.dumps({"gradient_replay_max_error": gradient_error,
                      "seconds_uncontrolled": result["seconds_uncontrolled"]}))


if __name__ == "__main__":
    main()
