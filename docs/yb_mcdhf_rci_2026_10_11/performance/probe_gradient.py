"""Exact MCDHF gradient action reuse, evaluated on a native coherent CI model."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import time
from pathlib import Path

import numpy as np

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.physics.csf.state_average import CIStateAverage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def action_reuse(functional, orbitals, weights, *, gradient_keys=None):
    """Candidate only: preserve each original sum while caching exact actions."""
    selected = functional.keys if gradient_keys is None else tuple(gradient_keys)
    if any(key not in functional.keys for key in selected):
        raise ValueError("gradient requested for an orbital outside the CSF basis")
    gradient = {key: np.zeros(2 * functional.ns) for key in selected}
    vectors = {key: np.concatenate(orbitals[key]) for key in functional.keys}
    one_actions = {}
    gradient_terms = {}
    for key, weight in weights.items():
        if key[0] == "h":
            _, na, kappa, nb = key
            a, b = (na, kappa), (nb, kappa)
            for target, source in ((a, b), (b, a)):
                if target in gradient:
                    if source not in one_actions:
                        one_actions[source] = functional.one_body[kappa] @ vectors[source]
                    gradient[target] += weight * one_actions[source]
        else:
            _, rank, a, b, c, d = key
            for left, right, source_a, source_b in ((a, b, c, d), (c, d, a, b)):
                if left not in gradient and right not in gradient:
                    continue
                pair = (rank, *sorted((source_a, source_b)))
                gradient_terms.setdefault(pair, []).append((left, right, weight))
    counts = {"source_pairs": len(gradient_terms), "one_body_actions": len(one_actions),
              "original_two_component_actions": 0, "reused_two_component_actions": 0}
    for (rank, source_a, source_b), terms in gradient_terms.items():
        pp, qq = functional._pair_potential(rank, orbitals[source_a], orbitals[source_b])
        actions = {}
        for left, right, weight in terms:
            for target, source in ((left, right), (right, left)):
                if target not in gradient:
                    continue
                counts["original_two_component_actions"] += 2
                if source not in actions:
                    p, q = orbitals[source]
                    actions[source] = (pp @ p, qq @ q)
                    counts["reused_two_component_actions"] += 2
                gp, gq = actions[source]
                gradient[target][:functional.ns] += weight * gp
                gradient[target][functional.ns:] += weight * gq
    return gradient, counts


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "gradient_probe001")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    contract = json.loads((HERE / "contract.json").read_text())
    model, acceptance = contract["fixed_model"], contract["acceptance"]
    ns, ks, Z = model["ns"], model["ks"], model["Z"]
    data = prepare_bspline_data(make_knots(ns, ks, model["r0"], model["rmax"]), ns, ks)
    csfs = [csf for cfg in model["configs"] for csf in enumerate_jj_csfs(cfg)
            if [csf.total_J2, csf.parity] == model["symmetry"]]
    target = CIStateAverage(tuple(map(tuple, model["ensemble"])))
    functional = CSFOrbitalFunctional(csfs, data, Z, state_average=target)
    orbitals = {}
    for kappa in functional.metric:
        spectrum = solve_dirac(functional.one_body[kappa], functional.metric[kappa], ns, kappa)
        ell = kappa if kappa > 0 else -kappa - 1
        for n, k in functional.keys:
            if k == kappa:
                p, q, _ = spectrum.get_orbital(n - ell - 1 + int(kappa > 0))
                orbitals[n, k] = p, q
    energy, original, ci = functional.evaluate(orbitals)
    weights = functional.last_weights
    candidate, counts = action_reuse(functional, orbitals, weights)
    checks = []
    differences = np.array([np.max(np.abs(original[key] - candidate[key])) for key in functional.keys])
    checks.append({"name": "all_signed_radial_gradients", "max_absolute_ha": float(differences.max()),
                   "passed": bool(differences.max() <= acceptance["gradient_max_absolute_ha"])})
    # Signed transition densities provide independent positive/negative interference.
    transition_weights = functional.radial_density.contract_bilinear(
        ci.mixing_coefficients[:, 0], ci.mixing_coefficients[:, 1])
    original_transition = functional.gradient_from_weights(orbitals, transition_weights,
                                                           record_diagnostics=False)
    candidate_transition, _ = action_reuse(functional, orbitals, transition_weights)
    error = max(np.max(np.abs(original_transition[k] - candidate_transition[k])) for k in functional.keys)
    checks.append({"name": "signed_transition_radial_gradients", "max_absolute_ha": float(error),
                   "passed": bool(error <= acceptance["gradient_max_absolute_ha"])})
    selected = functional.keys[::2]
    partial, _ = action_reuse(functional, orbitals, weights, gradient_keys=selected)
    partial_error = max(np.max(np.abs(partial[k] - original[k])) for k in selected)
    checks.append({"name": "selected_radial_gradients", "max_absolute_ha": float(partial_error),
                   "passed": bool(partial_error <= acceptance["gradient_max_absolute_ha"])})
    rng = np.random.default_rng(81217)
    direction = {}
    for kappa in functional.metric:
        keys = [k for k in functional.keys if k[1] == kappa]
        x = np.column_stack([np.concatenate(orbitals[k]) for k in keys])
        metric = functional.metric[kappa]
        d = rng.normal(size=x.shape)
        cross = x.T @ metric @ d
        d -= x @ (0.5 * (cross + cross.T))
        d /= np.sqrt(np.trace(d.T @ metric @ d))
        for i, key in enumerate(keys):
            direction[key] = d[:, i]
    analytical = sum(candidate[k] @ direction[k] for k in functional.keys)
    energies = []
    step = acceptance["finite_difference_step"]
    for signed_step in (step, -step):
        trial = {}
        for kappa in functional.metric:
            keys = [k for k in functional.keys if k[1] == kappa]
            x = np.column_stack([np.concatenate(orbitals[k]) + signed_step * direction[k] for k in keys])
            values, vectors = np.linalg.eigh(x.T @ functional.metric[kappa] @ x)
            x = x @ ((vectors / np.sqrt(values)) @ vectors.T)
            for i, key in enumerate(keys):
                trial[key] = x[:ns, i].copy(), x[ns:, i].copy()
        energies.append(functional.evaluate(trial, gradient=False)[0])
    numerical = (energies[0] - energies[1]) / (2 * step)
    difference = abs(numerical - analytical)
    fd_limit = acceptance["finite_difference_absolute_ha"] + acceptance["finite_difference_relative"] * abs(analytical)
    checks.append({"name": "independent_rediagonalized_tangent_derivative", "analytical_ha": float(analytical),
                   "numerical_ha": float(numerical), "difference_ha": float(difference),
                   "limit_ha": float(fd_limit), "passed": bool(difference <= fd_limit)})
    samples = []
    load_start = os.getloadavg()
    for i in range(acceptance["timing_pairs"]):
        row = {}
        methods = (("original", lambda: functional.gradient_from_weights(orbitals, weights,
                                                                          record_diagnostics=False)),
                   ("candidate", lambda: action_reuse(functional, orbitals, weights)))
        for name, call in methods[::1 if i % 2 == 0 else -1]:
            start = time.perf_counter()
            call()
            row[name + "_seconds"] = time.perf_counter() - start
        samples.append(row)
    profile = cProfile.Profile()
    profile.runcall(functional.evaluate, orbitals)
    profile.dump_stats(str(args.output / "functional.profile"))
    with (args.output / "profile.txt").open("w") as stream:
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(35)
    load_end = os.getloadavg()
    # Correctness work is deliberately allowed under load; this load guard blocks speedup claims.
    controlled = load_start[0] <= acceptance["timing_load1_maximum"] and load_end[0] <= acceptance["timing_load1_maximum"]
    record = {"contract_sha256": hash_file(HERE / "contract.json"), "model": model,
              "n_csfs": len(csfs), "radial_keys": functional.keys, "weight_keys": len(weights),
              "energy_ha": energy, "counts": counts, "checks": checks,
              "passed": all(row["passed"] for row in checks),
              "timing": {"scope": "gradient_from_weights only, warmed pair integral factors; per-call exact source actions",
                         "samples": samples, "load_start": load_start, "load_end": load_end,
                         "controlled_load_guard_passed": controlled,
                         "speedup_claim": None,
                         "median_original_seconds": float(np.median([s["original_seconds"] for s in samples])),
                         "median_candidate_seconds": float(np.median([s["candidate_seconds"] for s in samples]))},
              "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
              "source_files": {str(p): hash_file(ROOT / p) for p in (
                  "src/atomx/methods/csf/variational.py", "src/atomx/physics/csf/radial_density.py",
                  "src/atomx/numerics/bspline/convol.py", "src/atomx/methods/ci.py")},
              "script_sha256": hash_file(__file__)}
    write_json(args.output / "result.json", record)
    np.savez(args.output / "arrays.npz", **{f"original_{i}": original[k] for i, k in enumerate(functional.keys)},
             **{f"candidate_{i}": candidate[k] for i, k in enumerate(functional.keys)},
             ci_energies=ci.state_energies, ci_mixing=ci.mixing_coefficients)
    print(json.dumps({"passed": record["passed"], "checks": checks, "counts": counts,
                      "timing": record["timing"], "n_csfs": len(csfs)}))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
