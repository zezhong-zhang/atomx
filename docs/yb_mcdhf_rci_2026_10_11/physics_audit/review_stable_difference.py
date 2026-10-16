"""Independent full-energy oracle for the unchanged-RDM finite difference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.physics.csf.state_average import CIStateAverage

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[2]
LD = np.longdouble


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_density(a, b, ns, order):
    """Direct full pair product in the exact cell bank's symmetric band."""
    a, b = np.asarray(a, dtype=LD), np.asarray(b, dtype=LD)
    output = np.zeros((ns, order), dtype=LD)
    output[:, 0] = a * b
    for offset in range(1, order):
        output[:-offset, offset] = a[:-offset] * b[offset:] + a[offset:] * b[:-offset]
    return output


def full_energy_oracle(functional, orbitals, weights):
    """Full totals in wider precision, independent of increment code/gradient."""
    total = LD(0)
    ns, data = functional.ns, functional.data
    orders = data.get("ksp", data["ks"]), data.get("ksq", data["ks"])
    for term, coefficient in weights.items():
        if term[0] == "h":
            _, na, kappa, nb = term
            bra = np.concatenate(orbitals[na, kappa]).astype(LD)
            ket = np.concatenate(orbitals[nb, kappa]).astype(LD)
            value = (bra @ functional.one_body[kappa].astype(LD) @ ket
                     / (ket @ functional.metric[kappa].astype(LD) @ ket))
        else:
            _, rank, a, b, c, d = term
            value = LD(0)
            for left, label_left in enumerate(("p", "q")):
                rho_left = independent_density(*[orbitals[key][left] for key in (a, b)],
                                               ns, orders[left])
                for right, label_right in enumerate(("p", "q")):
                    rho_right = independent_density(*[orbitals[key][right] for key in (c, d)],
                                                    ns, orders[right])
                    components = label_left + label_right + label_left + label_right
                    bank, extent, _ = functional.cache.get_rkb_t(rank, components)
                    assert extent == ns
                    value += np.einsum("ui,vj,uvij->", rho_left.T, rho_right.T,
                                       bank.astype(LD), dtype=LD, optimize=False)
        total += LD(coefficient) * value
    return total


def raw_density(functional, result):
    """Same CI/angular identity, without the production 1e-15 coefficient filter."""
    state_weights = functional.state_average.weights(functional.csfs, result)
    chosen = state_weights > 0
    raw = {}
    for structure, projection, columns in functional.blocks:
        amplitudes = projection @ result.mixing_coefficients[columns][:, chosen]
        products = amplitudes[structure.pairs[:, 0]] * amplitudes[structure.pairs[:, 1]]
        values = np.asarray(structure.table.T @ (products @ state_weights[chosen])).ravel()
        for key, value in zip(structure.keys, values, strict=True):
            raw[key] = raw.get(key, 0.) + float(value)
    return raw


def main():
    contract_path = OWNER / "stable_difference_contract.json"
    source_paths = (
        "src/atomx/methods/csf/variational.py",
        "src/atomx/numerics/bspline/orbital_optimization.py",
        "src/atomx/physics/csf/radial_density.py",
        "src/atomx/physics/csf/matrix.py",
        "src/atomx/numerics/bspline/convol.py",
    )
    before = {path: sha(ROOT / path) for path in source_paths}
    ns = 35
    data = prepare_bspline_data(make_knots(ns, 9, .005, 30), ns, 9)
    csfs = enumerate_jj_csfs("1s1 2s1")
    functional = CSFOrbitalFunctional(csfs, data, 4, state_average=CIStateAverage())
    eigen = solve_dirac(functional.one_body[-1], functional.metric[-1], ns, -1)
    seed = eigen.coefficients[:, :len(functional.keys)]
    reference = {key: (seed[:ns, i].copy(), seed[ns:, i].copy())
                 for i, key in enumerate(functional.keys)}
    energy, ci = functional.evaluate(reference, gradient=False)
    weights = functional.radial_density.contract(ci, functional.state_average.weights(csfs, ci))
    raw = raw_density(functional, ci)
    dropped = {key: value for key, value in raw.items() if value != 0 and key not in weights}
    base = full_energy_oracle(functional, reference, weights)
    raw_base = full_energy_oracle(functional, reference, raw)

    large = seed @ np.array([[np.cos(.08), -np.sin(.08)],
                            [np.sin(.08), np.cos(.08)]])
    large = large * np.cos(.03) + eigen.coefficients[:, 2:4] * np.sin(.03)
    controls = [("finite_quartic", large)]
    for angle in (1e-8, 1e-12):
        trial = seed.copy()
        trial[:, 0] = seed[:, 0] * np.cos(angle) + eigen.coefficients[:, 4] * np.sin(angle)
        controls.append((f"admixture_{angle:.0e}", trial))
    normalized_trial = seed.copy()
    normalized_trial[:, 0] *= 1 - 2e-8
    controls.append(("ket_normalization_control", normalized_trial))

    rows = []
    for name, trial in controls:
        waves = {key: (trial[:ns, i], trial[ns:, i])
                 for i, key in enumerate(functional.keys)}
        updated, other = functional.evaluate(waves, gradient=False)
        delta = functional.fixed_density_energy_difference(reference, waves, ci, other)
        reverse = functional.fixed_density_energy_difference(waves, reference, other, ci)
        oracle_delta = full_energy_oracle(functional, waves, weights) - base
        raw_delta = full_energy_oracle(functional, waves, raw) - raw_base
        error = float(abs(LD(delta) - oracle_delta))
        tolerance = 5e-17 if name == "admixture_1e-12" else 3e-12
        row = {
            "control": name,
            "full_CI_total_difference_Ha": float(updated - energy),
            "stable_difference_Ha": delta,
            "long_double_full_difference_Ha": str(oracle_delta),
            "stable_minus_oracle_abs_Ha": error,
            "reverse_sum_abs_Ha": abs(delta + reverse),
            "discarded_density_difference_Ha": str(raw_delta - oracle_delta),
            "metric_max_abs_error": float(np.max(abs(
                trial.T @ functional.metric[-1] @ trial - np.eye(len(functional.keys))))),
        }
        row["passed"] = (abs(delta - (updated - energy)) < 3e-11
                         and error < tolerance and abs(delta + reverse) < 3e-13)
        if name == "admixture_1e-12":
            row["passed"] = bool(row["passed"] and delta != 0 and oracle_delta != 0
                                 and np.sign(delta) == np.sign(oracle_delta))
        rows.append(row)
    after = {path: sha(ROOT / path) for path in source_paths}
    report = {
        "contract_sha256": sha(contract_path),
        "source_hashes": before,
        "source_unchanged_during_run": before == after,
        "script_sha256": sha(Path(__file__)),
        "long_double_precision_bits": int(np.finfo(LD).nmant),
        "ensemble": "equal J=0/J=1 one-CSF blocks",
        "raw_density_omitted_nonzero_terms": [[repr(key), value] for key, value in dropped.items()],
        "rows": rows,
        "passed": before == after and all(row["passed"] for row in rows),
        "limits": "Finite-cell independent arithmetic anchor; no new Yb roots and no CI mixing-response claim.",
    }
    (OWNER / "stable_difference_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
