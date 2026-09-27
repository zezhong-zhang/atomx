#!/usr/bin/env python3
"""Replace the reproduced Cu II 3d9 4d wrong-root DHF calculation."""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomx import Atom, Basis  # noqa: E402
from atomx.methods import DiracFock  # noqa: E402
from scripts.generate_full_dhf_nist import (  # noqa: E402
    OUTPUT, basis_for_job, orbital_order_valid, update_lines,
)

Z, CHARGE = 29, 1
CONFIG = "1s2 2s2 2p6 3s2 3p6 3d9 4d1"
REPORT = ROOT / "output/nist_spectra/cu_ii_4d_branch_repair.json"
BOUND_AUDIT = ROOT / "output/nist_spectra/dhf_bound_state_audit.json"


def upper_d_nodes(result) -> dict[str, int]:
    profiles = {}
    for shell in result.subshells:
        if shell.n != 4 or shell.l != 2:
            continue
        wave = result.waves.orbitals.get(shell.label)
        if wave is None:
            raise ValueError(f"missing {shell.label} radial orbital")
        values = np.asarray(wave.P).ravel()
        if not values.size or not np.all(np.isfinite(values)):
            raise ValueError(f"invalid {shell.label} radial orbital")
        peak = float(np.max(np.abs(values)))
        if peak <= 0:
            raise ValueError(f"empty {shell.label} radial orbital")
        significant = values[np.abs(values) > 0.01 * peak]
        profiles[shell.label] = int(np.count_nonzero(
            np.signbit(significant[:-1]) != np.signbit(significant[1:])))
    if len(profiles) != 2:
        raise ValueError(f"expected both 4d jj orbitals, found {profiles}")
    return profiles


def solve(mass: float, basis: Basis, start: str):
    began = time.monotonic()
    result = Atom(Z=Z, config=CONFIG, engine="bspline", basis=basis).solve(
        DiracFock(breit=True, qed=True, nuclear_mass_me=mass,
                  starting_potential=start), max_iter=220)
    return result, time.monotonic() - began


def repair(ledger: Path = OUTPUT, report_path: Path = REPORT) -> dict:
    flags = json.loads(BOUND_AUDIT.read_text())["flagged"]
    flag = next((item for item in flags if
                 (item["z"], item["charge"], item["configuration"])
                 == (Z, CHARGE, CONFIG)), None)
    if flag is None or flag["theory_excess_ha"] <= 0.1:
        raise ValueError("Cu II 4d is not a large bound-state audit failure")
    with sqlite3.connect(ledger, timeout=60) as db:
        old = db.execute("""SELECT status,energy_ha,nuclear_mass_me,
            starting_potential FROM config_jobs WHERE z=? AND charge=?
            AND configuration=?""", (Z, CHARGE, CONFIG)).fetchone()
        ionic = db.execute("""SELECT energy_ha FROM config_jobs WHERE z=?
            AND charge=? AND is_ground=1 AND status='done'""",
                           (Z, CHARGE + 1)).fetchone()
    if old is None or old[0] != "done" or old[3] != "gsz" or ionic is None:
        raise ValueError("stored GSZ branch or next-ion ground has changed")
    old_energy, mass, ionic_energy = old[1], old[2], ionic[0]
    basis, radius, hmax = basis_for_job(Z, CHARGE, CONFIG)
    source, source_seconds = solve(mass, basis, "gsz")
    source_nodes = upper_d_nodes(source)
    if (not source.converged or abs(source.total_energy - old_energy) > 1e-4
            or all(value == 1 for value in source_nodes.values())):
        raise RuntimeError("stored wrong-root GSZ branch did not reproduce")
    corrected, corrected_seconds = solve(mass, basis, "tfda")
    corrected_nodes = upper_d_nodes(corrected)
    if (not corrected.converged or not orbital_order_valid(corrected)
            or not all(value == 1 for value in corrected_nodes.values())
            or not math.isfinite(corrected.total_energy)
            or corrected.total_energy >= ionic_energy - 0.05
            or corrected.total_energy >= old_energy - 0.1):
        raise RuntimeError("TFDA branch is not a verified bound 4d solution")
    refined = Basis(r_max=radius, grid_hmax=0.9 * hmax)
    crosscheck, crosscheck_seconds = solve(mass, refined, "tfda")
    crosscheck_nodes = upper_d_nodes(crosscheck)
    grid_spread = abs(crosscheck.total_energy - corrected.total_energy)
    if (not crosscheck.converged or
            not all(value == 1 for value in crosscheck_nodes.values())
            or grid_spread > 1e-4):
        raise RuntimeError("refined-grid branch does not reproduce")
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "z": Z, "charge": CHARGE, "configuration": CONFIG,
        "breit": True, "qed": True, "nuclear_mass_me": mass,
        "old_energy_ha": old_energy, "old_start": "gsz",
        "source_repeat_energy_ha": float(source.total_energy),
        "source_4d_nodes": source_nodes, "source_seconds": source_seconds,
        "corrected_energy_ha": float(corrected.total_energy),
        "corrected_nms_ha": float(corrected.energy_components.get("nms", 0.0)),
        "corrected_4d_nodes": corrected_nodes,
        "corrected_residual": float(corrected.residual),
        "corrected_seconds": corrected_seconds,
        "crosscheck_energy_ha": float(crosscheck.total_energy),
        "crosscheck_4d_nodes": crosscheck_nodes,
        "crosscheck_seconds": crosscheck_seconds,
        "grid_spread_ha": grid_spread,
        "r_max_bohr": radius, "grid_hmax_bohr": hmax,
        "ledger_updated": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    with sqlite3.connect(ledger, timeout=60) as db:
        db.execute("PRAGMA busy_timeout=60000")
        db.execute("BEGIN IMMEDIATE")
        current = db.execute("""SELECT status,energy_ha,starting_potential
            FROM config_jobs WHERE z=? AND charge=? AND configuration=?""",
                             (Z, CHARGE, CONFIG)).fetchone()
        if current != ("done", old_energy, "gsz"):
            raise RuntimeError("stored branch changed while crosscheck ran")
        db.execute("""UPDATE config_jobs SET energy_ha=?,nms_ha=?,
            converged=1,elapsed_seconds=?,error=NULL,starting_potential='tfda',
            r_max_bohr=?,grid_hmax_bohr=?
            WHERE z=? AND charge=? AND configuration=?""",
                   (float(corrected.total_energy), report["corrected_nms_ha"],
                    corrected_seconds, radius, hmax, Z, CHARGE, CONFIG))
        update_lines(db, Z, CHARGE)
    report["ledger_updated"] = True
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = repair()
    print(json.dumps({key: result[key] for key in
                      ("old_energy_ha", "corrected_energy_ha",
                       "grid_spread_ha", "ledger_updated")}))
