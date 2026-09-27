#!/usr/bin/env python3
"""Accept a converged, same-grid DHF damping probe into the live ledger."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_full_dhf_nist import OUTPUT, basis_for_job, update_lines


def apply(report_path: Path, ledger: Path = OUTPUT) -> dict:
    report = json.loads(report_path.read_text())
    z, charge, config = (report["z"], report["charge"],
                         report["configuration"])
    if not (report.get("breit") is True and report.get("qed") is True
            and report.get("converged") is True
            and report.get("orbital_order_valid") is True
            and report.get("orbital_node_valid") is True):
        raise ValueError("probe lacks converged DHF+B+QED and orbital checks")
    if (not math.isfinite(report["energy_ha"])
            or not math.isfinite(report["normal_mass_shift_ha"])
            or not 0 <= report["residual"] < 1e-5):
        raise ValueError("probe energy, mass correction, or residual invalid")
    if (report["starting_potential"] != "tfda"
            or not 0 < report["scf_damping"] < 1
            or not 0 < report["scf_damping_release_residual"] < 1):
        raise ValueError("probe SCF parameters invalid")
    _, radius, hmax = basis_for_job(z, charge, config)
    if (not math.isclose(report["r_max_bohr"], radius, abs_tol=1e-12)
            or not math.isclose(report["grid_hmax_bohr"], hmax,
                                 abs_tol=1e-12)):
        raise ValueError("probe radial basis differs from production")
    with sqlite3.connect(ledger, timeout=60) as db:
        db.execute("PRAGMA busy_timeout=60000")
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("""SELECT status,energy_ha,nuclear_mass_me
            FROM config_jobs WHERE z=? AND charge=? AND configuration=?""",
                         (z, charge, config)).fetchone()
        if row is None or row[0] != "failed" or row[1] is not None:
            raise ValueError("ledger job is absent or no longer failed")
        if not math.isclose(row[2], report["nuclear_mass_me"],
                            rel_tol=0, abs_tol=1e-9):
            raise ValueError("probe isotope mass differs from ledger")
        start = (f"tfda+damping={report['scf_damping']:g}"
                 f"+release={report['scf_damping_release_residual']:g}")
        db.execute("""UPDATE config_jobs SET status='done',energy_ha=?,
            nms_ha=?,converged=1,elapsed_seconds=?,error=NULL,
            r_max_bohr=?,grid_hmax_bohr=?,starting_potential=?,
            worker_pid=NULL,worker_start_ticks=NULL,claimed_from_status=NULL
            WHERE z=? AND charge=? AND configuration=? AND status='failed'""",
                   (report["energy_ha"], report["normal_mass_shift_ha"],
                    report["elapsed_seconds"], radius, hmax, start,
                    z, charge, config))
        update_lines(db, z, charge)
    return {"z": z, "charge": charge, "configuration": config,
            "energy_ha": report["energy_ha"], "starting_potential": start}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--ledger", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(apply(args.report, args.ledger)))


if __name__ == "__main__":
    main()
