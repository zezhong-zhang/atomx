#!/usr/bin/env python3
"""Shooting-grid convergence scan for DW electron-impact ionisation."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from atomx import Atom  # noqa: E402
from atomx.constants import HARTREE_EV  # noqa: E402
from atomx.scattering import ContinuumModel  # noqa: E402
from atomx.scattering.electron_ionization import eii_cross_section  # noqa: E402


OUT = ROOT / "benchmarks" / "results_eii_shooting_convergence.csv"


def _threshold_eV(atom: Atom) -> float:
    eps = atom._result.orbital_energies["1s1/2"]
    return float(-eps * HARTREE_EV)


def _sigma_fe_helike(atom: Atom, factor: float, n_grid: int, r_max: float, kappa_max: int,
                     n_outgoing_energies: int) -> tuple[float, float]:
    threshold = _threshold_eV(atom)
    energy = factor * threshold
    model = ContinuumModel.shooting(n_grid=n_grid, r_max=r_max)
    out = eii_cross_section(
        atom._result,
        np.array([energy]),
        method="DW",
        kappa_max=kappa_max,
        n_outgoing_energies=n_outgoing_energies,
        continuum_model=model,
    )
    return threshold, float(out.sigma_total_cm2[0])


def _with_relative_changes(rows: list[dict[str, object]], group: str) -> None:
    previous = None
    for row in rows:
        if row["scan"] != group:
            continue
        sigma = float(row["sigma_cm2"])
        if previous is None:
            row["rel_change_from_previous"] = ""
        else:
            row["rel_change_from_previous"] = abs(sigma - previous) / abs(previous)
        previous = sigma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--factor", type=float, default=2.0)
    parser.add_argument("--n-grids", default="8000,12000,16000,20000,24000")
    parser.add_argument("--r-maxes", default="40,45,50")
    parser.add_argument("--reference-r-max", type=float, default=50.0)
    parser.add_argument("--reference-n-grid", type=int, default=20000)
    parser.add_argument("--kappa-max", type=int, default=10)
    parser.add_argument("--n-outgoing-energies", type=int, default=25)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    n_grids = [int(x) for x in args.n_grids.split(",") if x.strip()]
    r_maxes = [float(x) for x in args.r_maxes.split(",") if x.strip()]
    atom = Atom(Z=26, config="1s2", charge=24, numerics="bspline", breit=False, qed=False)
    atom.solve()

    for n_grid in n_grids:
        threshold, sigma = _sigma_fe_helike(
            atom,
            args.factor,
            n_grid,
            args.reference_r_max,
            args.kappa_max,
            args.n_outgoing_energies,
        )
        rows.append({
            "scan": "n_grid",
            "Z": 26,
            "sequence": "he-like",
            "shell": "K",
            "E_over_threshold": args.factor,
            "threshold_eV": threshold,
            "n_grid": n_grid,
            "r_max_a0": args.reference_r_max,
            "sigma_cm2": sigma,
            "rel_change_from_previous": "",
        })

    for r_max in r_maxes:
        threshold, sigma = _sigma_fe_helike(
            atom,
            args.factor,
            args.reference_n_grid,
            r_max,
            args.kappa_max,
            args.n_outgoing_energies,
        )
        rows.append({
            "scan": "r_max",
            "Z": 26,
            "sequence": "he-like",
            "shell": "K",
            "E_over_threshold": args.factor,
            "threshold_eV": threshold,
            "n_grid": args.reference_n_grid,
            "r_max_a0": r_max,
            "sigma_cm2": sigma,
            "rel_change_from_previous": "",
        })

    _with_relative_changes(rows, "n_grid")
    _with_relative_changes(rows, "r_max")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        fieldnames = [
            "scan",
            "Z",
            "sequence",
            "shell",
            "E_over_threshold",
            "threshold_eV",
            "n_grid",
            "r_max_a0",
            "sigma_cm2",
            "rel_change_from_previous",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
