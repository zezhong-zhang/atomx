"""Independent saved-matrix replay after the complete-Fock projector repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from atomx.config.csf import enumerate_jj_csfs
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.numerics.bspline.orbital_optimization import optimize_orbitals

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    old = json.loads((OWNER / "fock_projector_result.json").read_text())
    assert sha(OWNER / "fock_projector_arrays.npz") == old["arrays_sha256"]
    rows = []
    with np.load(OWNER / "fock_projector_arrays.npz") as inputs:
        for previous in old["groups"]:
            kappa = previous["kappa"]
            # Relabel the finite semantic exercise as N=2, with two equally
            # occupied orbitals. Repeating the saved complete average as the
            # Fock input isolates no-pair handling from original CSF ordering.
            keys = tuple(tuple(key) for key in previous["keys"])
            ns = inputs[f"k{kappa}_metric"].shape[0] // 2
            metric = inputs[f"k{kappa}_metric"]
            nuclear = inputs[f"k{kappa}_nuclear"]
            expected = inputs[f"k{kappa}_complete_average"]
            functional = SimpleNamespace(
                keys=keys, csfs=enumerate_jj_csfs("1s2"), metric={kappa: metric},
                one_body={kappa: nuclear}, ns=ns, data={"ks": 9},
                last_occupations={key: 1. for key in keys},
                target_description="saved fixed-Fock matrix semantic replay",
            )

            def evaluate(waves, ns=ns, **kwargs):
                return 0., {key: np.zeros(2 * ns) for key in waves}, SimpleNamespace()

            functional.evaluate = evaluate
            spectral = solve_dirac(nuclear, metric, ns, kappa)
            first = previous["first_positive"]
            waves = {key: (spectral.coefficients[:ns, first + i].copy(),
                           spectral.coefficients[ns:, first + i].copy())
                     for i, key in enumerate(keys)}
            seen = []

            def intercept(h, s, n, k, seen=seen, **kwargs):
                seen.append(h.copy())
                return solve_dirac(h, s, n, k, **kwargs)

            with patch("atomx.numerics.bspline.dirac_assembly.solve_dirac", side_effect=intercept):
                current, _, _ = optimize_orbitals(
                    functional, waves, max_iter=0,
                    preconditioner_matrices={key: expected for key in keys},
                    projector_source="fock",
                )
            actual = seen[1]
            reference = solve_dirac(expected, metric, ns, kappa)
            actual_spectrum = solve_dirac(actual, metric, ns, kappa)
            reference_positive = reference.coefficients[:, first:]
            vectors = np.column_stack([np.concatenate(current[key]) for key in keys])
            row = {
                "kappa": kappa,
                "max_abs_projector_input_difference_Ha": float(np.max(abs(actual - expected))),
                "max_abs_spectral_difference_Ha": float(np.max(abs(
                    actual_spectrum.eigenvalues - reference.eigenvalues))),
                "max_abs_spinor_projector_leakage": float(np.max(abs(
                    vectors - reference_positive @ (reference_positive.T @ metric @ vectors)))),
                "max_abs_gram_defect": float(np.max(abs(vectors.T @ metric @ vectors - np.eye(len(keys))))),
                "old_excess_minus_nuclear_max_Ha": float(np.max(abs(
                    inputs[f"k{kappa}_actual_projector_input"] - expected - nuclear))),
            }
            rows.append(row)
    report = {
        "original_arrays_sha256": old["arrays_sha256"],
        "current_optimizer_sha256": sha(ROOT / "src/atomx/numerics/bspline/orbital_optimization.py"),
        "rows": rows,
        "passed": all(row["max_abs_projector_input_difference_Ha"] < 1e-10
                      and row["max_abs_spinor_projector_leakage"] < 1e-9
                      and row["max_abs_gram_defect"] < 1e-10 for row in rows),
        "scope": "Independent saved public full-Fock matrix semantic and positive-branch replay; no new Yb roots",
    }
    (OWNER / "fock_projector_replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
