"""Evaluate explicit projector increments without subtracting rounded spinors."""

import hashlib
import json
import math
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
    owner = Path(__file__).resolve().parent
    retained = owner / "nuclear_eol004_stable_delta_trace"
    epoch = owner / "stable_polar_endpoint002"
    out = owner / "linear_map_increment_endpoint"
    out.mkdir(exist_ok=False)
    contract = {
        "hypothesis": "Computing trial-base from separately rounded projector products introduces positive-space-external radial increments. Direct projector@(trial_y-base_y) preserves the same linear map and permits a more accurate finite-polynomial constrained difference without altering the stored CI trials or original residual metric.",
        "fixed_conditions": "Original accepted 004 coordinates, exact same projectors/gradients, five fixed one-CSF RDMs, same complete cell integral kernels and actual old/new polar candidates. No optimizer acceptance change or endpoint certification.",
        "source_epoch_contract_sha256": sha(epoch / "contract.json"),
        "inputs_sha256": {name: sha(retained / name) for name in
            ("contract.json", "failed_coordinates.npz", "failed_coordinate_layout.json", "orbitals_workspace.npz")},
        "steps": [0., .1, .01, 2. ** -27],
        "controls": ["Current API with rounded trial-base increments",
            "Independent complete polynomial with those same rounded increments",
            "Independent complete polynomial with explicit projector@coordinate_increment",
            "Actual h/n quotient normal multipliers and full quadratic delta S Gram",
            "Exact identity candidate with zero increments"],
        "acceptance": {"rounded_increment_polynomial_vs_API_absolute_error_lt_Ha": 1e-19,
                       "exact_zero_increment_difference_Ha": 0.},
        "stopping": "One endpoint diagnostic only; no reoptimization, experimental fitting or numerical gate change."}
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

    def old_polar(y):
        values, vectors = np.linalg.eigh(y.T @ y)
        return y @ ((vectors / np.sqrt(values)) @ vectors.T)

    def retract(point, polar):
        result = point.copy()
        for group in layout:
            where = slice(*group["slice"])
            result[where] = polar(point[where].reshape(group["shape"])).ravel()
        return result

    reference = unpack(x)
    _, gradient, ci = functional.evaluate(reference)
    weights = functional.radial_density.contract(ci, functional.state_average.weights(csfs, ci))
    quotient_gradient = {key: value.copy() for key, value in gradient.items()}
    for term, coefficient in weights.items():
        if term[0] != "h":
            continue
        _, na, kappa, nb = term
        a, b = (na, kappa), (nb, kappa)
        va, vb = np.concatenate(reference[a]), np.concatenate(reference[b])
        h, metric = functional.one_body[kappa], functional.metric[kappa]
        norm, h_value = float(vb @ metric @ vb), float(va @ h @ vb)
        quotient_gradient[a] += coefficient * (1. / norm - 1.) * (h @ vb)
        quotient_gradient[b] += coefficient * ((1. / norm - 1.) * (h @ va)
            - 2. * h_value / norm ** 2 * (metric @ vb))

    def complete_explicit_difference(increments):
        changes, sources = [], {}
        for term, coefficient in weights.items():
            if term[0] != "h":
                _, rank, a, b, c, d = term
                sources.setdefault((rank, *sorted((c, d))), []).append((a, b, coefficient))
                continue
            _, na, kappa, nb = term
            a, b = (na, kappa), (nb, kappa)
            va, vb = np.concatenate(reference[a]), np.concatenate(reference[b])
            da, db = np.concatenate(increments[a]), np.concatenate(increments[b])
            h, metric = functional.one_body[kappa], functional.metric[kappa]
            hb, hdb = h @ vb, h @ db
            dn = math.fsum((float(db @ metric @ vb), float(vb @ metric @ db), float(db @ metric @ db)))
            norm = float(vb @ metric @ vb)
            dh = math.fsum((float(da @ hb), float(va @ hdb), float(da @ hdb)))
            changes.append(coefficient * (dh - float(va @ hb) / norm * dn) / (norm + dn))
        for (rank, c, d), terms in sources.items():
            original = functional._pair_potential(rank, reference[c], reference[d])
            difference = functional._pair_potential(rank, reference[c], reference[d],
                                                    change=(increments[c], increments[d]))
            for a, b, coefficient in terms:
                components = []
                for side in (0, 1):
                    va, vb = reference[a][side], reference[b][side]
                    da, db = increments[a][side], increments[b][side]
                    v, dv = original[side], difference[side]
                    components.extend((float(da @ v @ vb), float(va @ v @ db), float(da @ v @ db),
                        float(va @ dv @ vb), float(da @ dv @ vb), float(va @ dv @ db), float(da @ dv @ db)))
                changes.append(coefficient * math.fsum(components))
        return math.fsum(changes)

    def constraint_defect_energy(increments):
        total = np.longdouble(0)
        for group in layout:
            keys = [tuple(key) for key in group["keys"]]
            base = np.column_stack([np.concatenate(reference[key]) for key in keys]).astype(np.longdouble)
            delta = np.column_stack([np.concatenate(increments[key]) for key in keys]).astype(np.longdouble)
            metric = functional.metric[keys[0][1]].astype(np.longdouble)
            quotient = np.column_stack([quotient_gradient[key] for key in keys]).astype(np.longdouble)
            cross = base.T @ metric @ delta
            delta_gram = cross + cross.T + delta.T @ metric @ delta
            total += .5 * np.sum((base.T @ quotient) * delta_gram)
        return float(total)

    arrays = {}
    rows = []
    candidates = [("identity_control", 0., x.copy())]
    for step in contract["steps"]:
        candidates.extend((name, step, retract(x + step * direction, polar)) for name, polar in
                          (("eigh_original", old_polar), ("wide_gram_correction", _polar_retraction)))
    for index, (name, step, candidate) in enumerate(candidates):
        trial = unpack(candidate)
        _, trial_ci = functional.evaluate(trial, gradient=False)
        rounded = {key: tuple(new - old for new, old in zip(trial[key], reference[key], strict=True))
                   for key in functional.keys}
        linear_map = unpack(candidate - x)
        api = functional.fixed_density_energy_difference(reference, trial, ci, trial_ci)
        explicit_rounded = complete_explicit_difference(rounded)
        explicit_linear = complete_explicit_difference(linear_map)
        rounded_defect = constraint_defect_energy(rounded)
        linear_defect = constraint_defect_energy(linear_map)
        rows.append({"step": step, "retraction": name, "API_full_difference_Ha": api,
            "explicit_rounded_full_difference_Ha": explicit_rounded,
            "explicit_rounded_vs_API_error_Ha": explicit_rounded - api,
            "explicit_linear_map_full_difference_Ha": explicit_linear,
            "rounded_constraint_defect_Ha": rounded_defect,
            "linear_map_constraint_defect_Ha": linear_defect,
            "rounded_centered_difference_Ha": explicit_rounded - rounded_defect,
            "linear_map_centered_difference_Ha": explicit_linear - linear_defect,
            "coordinate_tangent_linear_Ha": float(g @ (candidate - x)),
            "coordinate_direction_linear_Ha": float(step * g @ direction),
            "actual_radial_increment_representation_error_norm": float(np.sqrt(sum(
                sum(np.sum((a - b) ** 2) for a, b in zip(rounded[key], linear_map[key], strict=True))
                for key in functional.keys)))})
        arrays[f"coordinate_delta_{index}"] = candidate - x
        for key in functional.keys:
            arrays[f"map_delta_{index}_{key[0]}_{key[1]}"] = np.concatenate(linear_map[key])
            arrays[f"rounded_delta_{index}_{key[0]}_{key[1]}"] = np.concatenate(rounded[key])
    np.savez_compressed(out / "actual_increments.npz", **arrays)
    result = {"rows": rows, "actual_increments_sha256": sha(out / "actual_increments.npz"),
        "source_files_verified": len(source_contract["source_sha256"]),
        "seconds_uncontrolled": time.monotonic() - started,
        "diagnostic_only": True, "solver_acceptance_unchanged": True,
        "certifies_original_solve": False}
    write(out / "result.json", result)
    if (max(abs(row["explicit_rounded_vs_API_error_Ha"]) for row in rows) > 1e-19
            or rows[0]["explicit_linear_map_full_difference_Ha"] != 0.):
        raise ValueError("explicit polynomial failed independent same-increment control")
    write(out / "verification.json", {"same_rounded_increment_complete_polynomial_vs_API_passed": True,
        "identity_control_exactly_zero": True, "script_sha256": sha(Path(__file__)),
        "result_sha256": sha(out / "result.json"), "certifies_original_solve": False})
    print(json.dumps({"seconds_uncontrolled": result["seconds_uncontrolled"],
                      "identity_control_exactly_zero": True}))


if __name__ == "__main__":
    main()
