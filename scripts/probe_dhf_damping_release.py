#!/usr/bin/env python3
"""Probe a difficult stored DHF configuration with a controlled SCF mix."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_full_dhf_nist import (  # noqa: E402
    Atom, DiracFock, OUTPUT, basis_for_job, orbital_node_valid,
    orbital_order_valid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=OUTPUT)
    parser.add_argument("--z", type=int, required=True)
    parser.add_argument("--charge", type=int, required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--starting", default="tfda")
    parser.add_argument("--damping", type=float, default=0.7)
    parser.add_argument("--release", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    with sqlite3.connect(args.ledger) as db:
        source = db.execute("""SELECT status,energy_ha,nuclear_mass_me
            FROM config_jobs WHERE z=? AND charge=? AND configuration=?""",
            (args.z, args.charge, args.configuration)).fetchone()
    if source is None:
        parser.error("configuration absent from ledger")
    basis, radius, hmax = basis_for_job(args.z, args.charge,
                                         args.configuration)
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "z": args.z, "charge": args.charge,
              "configuration": args.configuration,
              "stored_status": source[0], "stored_energy_ha": source[1],
              "nuclear_mass_me": source[2], "breit": True, "qed": True,
              "r_max_bohr": radius, "grid_hmax_bohr": hmax,
              "starting_potential": args.starting, "scf_damping": args.damping,
              "scf_damping_release_residual": args.release,
              "max_iter": args.max_iter}
    started = time.monotonic()
    try:
        result = Atom(Z=args.z, config=args.configuration, engine="bspline",
                      basis=basis).solve(
            DiracFock(breit=True, qed=True, nuclear_mass_me=source[2],
                      starting_potential=args.starting,
                      scf_damping=args.damping,
                      scf_damping_release_residual=args.release),
            max_iter=args.max_iter)
        report.update(converged=bool(result.converged),
                      energy_ha=float(result.total_energy),
                      normal_mass_shift_ha=float(result.energy_components.get(
                          "nms", 0.0)),
                      residual=float(result.residual),
                      iterations=getattr(result, "iterations", None),
                      orbital_order_valid=bool(orbital_order_valid(result)),
                      orbital_node_valid=bool(orbital_node_valid(
                          result, args.z, args.configuration)),
                      orbital_energies={key: float(value) for key, value in
                                        result.orbital_energies.items()})
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = time.monotonic() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report.get(key) for key in
                      ("z", "charge", "configuration", "converged",
                       "energy_ha", "residual", "iterations",
                       "orbital_order_valid", "error", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
