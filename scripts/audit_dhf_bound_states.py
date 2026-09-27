#!/usr/bin/env python3
"""Flag DHF configurations above an ionization limit for a bound NIST manifold.

This is a numerical branch audit, not an accuracy cutoff. It uses only NIST
manifolds with complete evaluated levels below an evaluated principal limit.
An excited-core level above that limit is outside this particular check.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.nist_asd import HA_CM1
from scripts.generate_full_dhf_nist import OUTPUT

COMPARISON = ROOT / "output/nist_spectra/theory_comparison"
REPORT = ROOT / "output/nist_spectra/dhf_bound_state_audit.json"


def audit(ledger: Path = OUTPUT, comparison: Path = COMPARISON,
          allowance_ha: float = 0.05) -> dict:
    with sqlite3.connect(ledger) as db:
        completed = {(z, charge, config): (energy, start)
                     for z, charge, config, energy, start in db.execute(
                         """SELECT z,charge,configuration,energy_ha,starting_potential
                            FROM config_jobs WHERE status='done'""")}
        ionic_grounds = {(z, charge): energy
                         for z, charge, energy in db.execute(
                             """SELECT z,charge,energy_ha FROM config_jobs
                                WHERE status='done' AND is_ground=1""")}
    with (comparison / "ionization_dhf_nist.csv").open() as stream:
        limits = {(int(row["z"]), int(row["charge"])): row
                  for row in csv.DictReader(stream)
                  if row["reference_kind"] == "evaluated"
                  and row["nist_ionization_cm1"]}
    checked = 0
    flagged = []
    with (comparison / "configuration_centroids_dhf_nist.csv").open() as stream:
        for row in csv.DictReader(stream):
            z, charge = int(row["z"]), int(row["charge"])
            key = z, charge, row["configuration"]
            result = completed.get(key)
            ionic_ground = ionic_grounds.get((z, charge + 1))
            limit = limits.get((z, charge))
            if (result is None or ionic_ground is None or limit is None
                    or row["nist_manifold_status"] != "evaluated_complete"
                    or not row["nist_centroid_cm1"]):
                continue
            nist_centroid = float(row["nist_centroid_cm1"])
            nist_limit = float(limit["nist_ionization_cm1"])
            if nist_centroid >= nist_limit:
                continue
            checked += 1
            energy, start = result
            theoretical_binding = ionic_ground - energy
            if theoretical_binding < -allowance_ha:
                flagged.append({
                    "z": z, "charge": charge, "spectrum": row["spectrum"],
                    "configuration": row["configuration"],
                    "starting_potential": start,
                    "nist_binding_cm1": nist_limit - nist_centroid,
                    "theory_binding_cm1": theoretical_binding * HA_CM1,
                    "theory_excess_ha": -theoretical_binding,
                    "configuration_energy_ha": energy,
                    "ionic_ground_energy_ha": ionic_ground,
                })
    flagged.sort(key=lambda item: item["theory_excess_ha"], reverse=True)
    return {"checked_utc": datetime.now(timezone.utc).isoformat(),
            "allowance_ha": allowance_ha,
            "evaluated_bound_manifolds_checked": checked,
            "flagged_count": len(flagged), "flagged": flagged}


if __name__ == "__main__":
    report = audit()
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
