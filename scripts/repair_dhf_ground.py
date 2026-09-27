#!/usr/bin/env python3
"""Recalculate one failed DHF ground state with a selected SCF damping."""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_full_dhf_nist import (  # noqa: E402
    OUTPUT, Atom, DiracFock, basis_for_job, orbital_order_valid, update_lines,
)


def repair_ground(path: Path, z: int, charge: int, damping: float,
                  max_iter: int = 220) -> dict:
    if not 1 <= z <= 118 or not 0 <= charge < z:
        raise ValueError("invalid ion stage")
    if not 0 <= damping < 1 or max_iter < 1:
        raise ValueError("invalid SCF parameters")
    with sqlite3.connect(path, timeout=60) as db:
        db.execute("PRAGMA busy_timeout=60000")
        db.execute("PRAGMA journal_mode=WAL")
        jobs = db.execute("""SELECT configuration,status,nuclear_mass_me
            FROM config_jobs WHERE z=? AND charge=? AND is_ground=1""",
            (z, charge)).fetchall()
        if len(jobs) != 1:
            raise RuntimeError(f"expected one ground job, found {len(jobs)}")
        configuration, status, mass = jobs[0]
        if status != "failed":
            raise RuntimeError(f"ground job is {status}, not failed")
        basis, radius, hmax = basis_for_job(z, charge, configuration)
        started = time.monotonic()
        result = Atom(Z=z, config=configuration, engine="bspline",
                      basis=basis).solve(
            DiracFock(breit=True, qed=True, nuclear_mass_me=mass,
                      starting_potential=None, scf_damping=damping),
            max_iter=max_iter)
        if (not result.converged or not math.isfinite(result.total_energy)
                or not orbital_order_valid(result)):
            raise RuntimeError(f"untrusted SCF result: converged={result.converged}, "
                               f"residual={result.residual}")
        seconds = time.monotonic() - started
        starting = ("tfda" if z >= 20 else "hydrogenic") + (
            f"+damping={damping:g}")
        db.execute("BEGIN IMMEDIATE")
        current = db.execute("""SELECT status FROM config_jobs WHERE z=?
            AND charge=? AND configuration=?""",
            (z, charge, configuration)).fetchone()
        if current is None or current[0] != "failed":
            db.rollback()
            raise RuntimeError("ground job changed while recalculation ran")
        db.execute("""UPDATE config_jobs SET status='done',energy_ha=?,
            nms_ha=?,converged=1,elapsed_seconds=?,error=NULL,
            r_max_bohr=?,grid_hmax_bohr=?,starting_potential=?
            WHERE z=? AND charge=? AND configuration=?""",
            (float(result.total_energy),
             float(result.energy_components.get("nms", 0.0)),
             seconds, radius, hmax, starting, z, charge, configuration))
        update_lines(db, z, charge)
        db.commit()
        return {"z": z, "charge": charge, "configuration": configuration,
                "energy_ha": float(result.total_energy),
                "normal_mass_ha": float(result.energy_components.get("nms", 0.0)),
                "iterations": result.iterations, "residual": result.residual,
                "starting_potential": starting, "elapsed_seconds": seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=OUTPUT)
    parser.add_argument("--z", type=int, required=True)
    parser.add_argument("--charge", type=int, required=True)
    parser.add_argument("--damping", type=float, required=True)
    parser.add_argument("--max-iter", type=int, default=220)
    args = parser.parse_args()
    print(repair_ground(args.ledger, args.z, args.charge,
                        args.damping, args.max_iter), flush=True)


if __name__ == "__main__":
    main()
