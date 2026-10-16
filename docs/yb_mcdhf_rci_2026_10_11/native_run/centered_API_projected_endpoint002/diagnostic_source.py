"""Bind shared constrained-difference APIs to an independent real Yb endpoint."""

import argparse
import hashlib
import inspect
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
    parser.add_argument("--label", required=True)
    parser.add_argument("--explicit-increments-keyword")
    args = parser.parse_args()
    owner = Path(__file__).resolve().parent
    repo = owner.parents[2]
    retained = owner / "nuclear_eol004_stable_delta_trace"
    control = owner / "linear_map_increment_endpoint"
    out = owner / args.label
    out.mkdir(exist_ok=False)
    sources = {}
    for source in sorted((repo / "src/atomx").rglob("*.py")):
        target = out / "source" / source.relative_to(repo / "src")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[str(source.relative_to(repo))] = sha(target)
    contract = {
        "hypothesis": "The new constrained-difference API must reproduce the independent complete-polynomial and actual quotient-normal defect formula at an unchanged real failed Yb endpoint before a new initialization solve.",
        "fixed_conditions": "Original accepted 004 endpoint, same physical states and 24 orbitals, original CI and native residual metric, same actual old/new polar candidates. This is not an endpoint optimization or certification.",
        "source_sha256": sources,
        "source_map_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "independent_control_sha256": sha(control / "result.json"),
        "input_sha256": {name: sha(retained / name) for name in
            ("contract.json", "failed_coordinates.npz", "failed_coordinate_layout.json", "orbitals_workspace.npz")},
        "explicit_increment_keyword": args.explicit_increments_keyword,
        "acceptance": {"API_vs_independent_centered_formula_absolute_error_lt_Ha": 1e-19,
                       "complete_difference_API_retained_semantics_absolute_error_lt_Ha": 1e-19,
                       "identity_exactly_zero": True},
        "stopping": "Nine actual candidates only, preserve any mismatch, do not change original scientific gates."}
    write(out / "contract.json", contract)
    shutil.copyfile(Path(__file__), out / "diagnostic_source.py")
    for path, digest in sources.items():
        assert sha(repo / path) == digest, "shared source changed during snapshot"
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
    x, direction = (saved[name] for name in ("x", "direction"))

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
    _, ci = functional.evaluate(reference, gradient=False)
    retained_control = json.loads((control / "result.json").read_text())
    rows = []
    for expected in retained_control["rows"]:
        name, step = expected["retraction"], expected["step"]
        candidate = x.copy() if name == "identity_control" else retract(x + step * direction,
            old_polar if name == "eigh_original" else _polar_retraction)
        trial = unpack(candidate)
        _, trial_ci = functional.evaluate(trial, gradient=False)
        complete = functional.fixed_density_energy_difference(reference, trial, ci, trial_ci)
        keyword_arguments = {}
        if args.explicit_increments_keyword:
            keyword_arguments[args.explicit_increments_keyword] = unpack(candidate - x)
        centered = functional.constrained_energy_difference(reference, trial, ci, trial_ci, **keyword_arguments)
        target = expected["linear_map_centered_difference_Ha" if args.explicit_increments_keyword else
                          "rounded_centered_difference_Ha"]
        rows.append({"step": step, "retraction": name,
            "complete_API_Ha": complete, "complete_reference_Ha": expected["API_full_difference_Ha"],
            "complete_difference_error_Ha": complete - expected["API_full_difference_Ha"],
            "constrained_API_Ha": centered, "independent_centered_reference_Ha": target,
            "centered_difference_error_Ha": None if centered is None else centered - target})
    result = {"rows": rows, "API_signature": str(inspect.signature(functional.constrained_energy_difference)),
        "source_files_verified": len(sources), "seconds_uncontrolled": time.monotonic() - started,
        "certifies_original_solve": False}
    write(out / "result.json", result)
    if (any(row["centered_difference_error_Ha"] is None for row in rows)
            or max(abs(row["centered_difference_error_Ha"]) for row in rows) > 1e-19
            or max(abs(row["complete_difference_error_Ha"]) for row in rows) > 1e-19
            or rows[0]["constrained_API_Ha"] != 0.):
        raise ValueError("current API did not reproduce independent real-endpoint formula")
    write(out / "verification.json", {"API_vs_independent_formula_passed": True,
        "complete_API_semantics_preserved": True, "zero_control_exactly_zero": True,
        "script_sha256": sha(Path(__file__)), "result_sha256": sha(out / "result.json"),
        "source_files_verified": len(sources), "certifies_original_solve": False})
    print(json.dumps({"checks_passed": True, "seconds_uncontrolled": result["seconds_uncontrolled"]}))


if __name__ == "__main__":
    main()
