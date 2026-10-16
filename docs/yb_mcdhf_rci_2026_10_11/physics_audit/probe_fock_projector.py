"""Intercept the public MCDHF initializer; audit its fixed Fock projector input."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from atomx import Atom, Basis, CIStateAverage, Correlation
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.numerics.bspline.orbital_optimization import optimize_orbitals
from atomx.numerics.bspline.orbital_subspace import radial_projector_start

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[2]


class Captured(Exception):
    """The requested public initialization finished; no physical steps needed."""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = json.loads((OWNER / "source_manifest.json").read_text())
    for item in source["files"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Source epoch changed before capture: {item['path']}")
    arrays = {}
    report = {"contract_sha256": sha(OWNER / "contract.json"), "source_checks": len(source["files"])}

    def capture(functional, orbitals, **kwargs):
        complete = kwargs["preconditioner_matrices"]
        grouped = {}
        for key in functional.keys:
            grouped.setdefault(key[1], []).append(key)
        occupations = {(s.n, s.kappa): max(float(s.occupation), 1.0)
                       for s in functional.csfs[0].subshells}
        report["groups"] = []
        for kappa, keys in sorted(grouped.items()):
            metric = functional.metric[kappa]
            nuclear = functional.one_body[kappa]
            average = sum(occupations.get(key, 1.) * complete[key] for key in keys)
            average /= sum(occupations.get(key, 1.) for key in keys)
            captured_inputs = []

            def intercept(h, s, ns, k, captured_inputs=captured_inputs, **options):
                captured_inputs.append(np.array(h, copy=True))
                return solve_dirac(h, s, ns, k, **options)

            # Execute the same frozen numerical code up to max_iter=0,
            # but without expensive CI/density work; the source audit above
            # binds this exact function and the genuine public matrices.
            from types import SimpleNamespace
            keys_here = tuple(keys)
            local = SimpleNamespace(
                keys=keys_here, csfs=functional.csfs, metric={kappa: metric},
                one_body={kappa: nuclear}, ns=functional.ns, data=functional.data,
                last_occupations={key: occupations.get(key, 1.) for key in keys_here},
                target_description="fixed input semantic anchor",
            )

            def zero_gradient(waves, **options):
                return 0., {key: np.zeros(2 * functional.ns) for key in waves}, SimpleNamespace()

            local.evaluate = zero_gradient
            local_orbitals = {key: orbitals[key] for key in keys_here}
            with patch("atomx.numerics.bspline.dirac_assembly.solve_dirac", side_effect=intercept):
                optimize_orbitals(local, local_orbitals, max_iter=0,
                                  preconditioner_matrices={key: complete[key] for key in keys_here},
                                  projector_source="fock")
            actual = captured_inputs[1]
            first = radial_projector_start(kappa, min(key[0] for key in keys), functional.data)
            options = {"nsp": functional.data.get("nsp"), "nsq": functional.data.get("nsq")}
            correct_spectrum = solve_dirac(average, metric, functional.ns, kappa, **options)
            actual_spectrum = solve_dirac(actual, metric, functional.ns, kappa, **options)
            correct_positive = correct_spectrum.coefficients[:, first:]
            actual_positive = actual_spectrum.coefficients[:, first:]
            principal_cosines = np.linalg.svd(correct_positive.T @ metric @ actual_positive,
                                             compute_uv=False)
            row = {
                "kappa": kappa, "keys": [list(key) for key in keys], "first_positive": first,
                "max_abs_actual_minus_full_average_Ha": float(np.max(abs(actual-average))),
                "max_abs_excess_minus_nuclear_Ha": float(np.max(abs((actual-average)-nuclear))),
                "max_abs_positive_eigenvalue_difference_Ha": float(np.max(abs(
                    actual_spectrum.eigenvalues[first:]-correct_spectrum.eigenvalues[first:]))),
                "min_positive_subspace_principal_cosine": float(principal_cosines.min()),
            }
            for label, value in (("nuclear", nuclear), ("metric", metric), ("complete_average", average),
                                 ("actual_projector_input", actual),
                                 ("correct_eigenvalues", correct_spectrum.eigenvalues),
                                 ("actual_eigenvalues", actual_spectrum.eigenvalues)):
                arrays[f"k{kappa}_{label}"] = value
            for key in keys:
                arrays[f"n{key[0]}_k{key[1]}_full_fock"] = complete[key]
            report["groups"].append(row)
        raise Captured

    atom = Atom(10, config=["1s2", "1s1 2p1", "2s2"],
                basis=Basis(ns=40, ks=9), corrections="none",
                correlation=Correlation(state_average=CIStateAverage()))
    try:
        with patch("atomx.numerics.bspline.orbital_optimization.optimize_orbitals", side_effect=capture):
            atom.solve(tol=1e-6, max_iter=3)
    except Captured:
        report["public_initializer_captured"] = True
    else:
        raise RuntimeError("Public solver did not consume the intended orbital optimizer")
    report["confirmed_one_body_double_addition"] = all(
        row["max_abs_excess_minus_nuclear_Ha"] < 1e-10 for row in report["groups"])
    report["threads"] = {key: os.environ.get(key) for key in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    np.savez_compressed(OWNER / "fock_projector_arrays.npz", **arrays)
    report["arrays_sha256"] = sha(OWNER / "fock_projector_arrays.npz")
    report["scope"] = "Interface semantic and one-body spectral anchor; no Yb energies or physical-error estimate"
    (OWNER / "fock_projector_result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
