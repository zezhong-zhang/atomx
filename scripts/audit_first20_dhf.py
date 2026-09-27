#!/usr/bin/env python3
"""Audit a DHF job interval against the retained NIST ASD archive."""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "output/nist_spectra/spectra.sqlite"
LEDGER = ROOT / "output/nist_spectra/full_dhf.sqlite"
REPORT = ROOT / "output/nist_spectra/first20_dhf_audit.json"


def audit(archive: Path = ARCHIVE, ledger: Path = LEDGER, *,
          min_z: int = 1, max_z: int = 20) -> dict:
    if not 1 <= min_z <= max_z <= 118:
        raise ValueError("expected 1 <= min_z <= max_z <= 118")
    with sqlite3.connect(archive) as source, sqlite3.connect(ledger) as jobs:
        observed = {
            (z, charge, row): (wavelength, medium)
            for z, charge, row, wavelength, medium in source.execute(
                """SELECT z,charge,row_index,observed_nm,medium FROM lines
                   WHERE z BETWEEN ? AND ? AND observed_nm IS NOT NULL""",
                (min_z, max_z))
        }
        recorded = {
            (z, charge, row): (wavelength, medium)
            for z, charge, row, wavelength, medium in jobs.execute(
                """SELECT z,charge,row_index,observed_nm,medium
                   FROM line_jobs WHERE z BETWEEN ? AND ?""",
                (min_z, max_z))
        }
        source_stages = {
            (z, charge) for z, charge in source.execute(
                "SELECT z,charge FROM species WHERE z BETWEEN ? AND ?",
                (min_z, max_z))
        }
        ground_rows = list(jobs.execute(
            """SELECT z,charge,status,energy_ha,nms_ha,converged
               FROM config_jobs WHERE z BETWEEN ? AND ? AND is_ground=1""",
            (min_z, max_z)))
        grounds = {(z, charge): (status, energy, nms, converged)
                   for z, charge, status, energy, nms, converged in ground_rows}
        configurations = list(jobs.execute(
            """SELECT z,charge,configuration,status,energy_ha,nms_ha,converged
               FROM config_jobs WHERE z BETWEEN ? AND ?""",
            (min_z, max_z)))
        line_statuses = dict(jobs.execute(
            """SELECT status,COUNT(*) FROM line_jobs WHERE z BETWEEN ? AND ?
               GROUP BY status""", (min_z, max_z)))

    absent = sorted(observed.keys() - recorded.keys())
    extra = sorted(recorded.keys() - observed.keys())
    changed = sorted(key for key in observed.keys() & recorded.keys()
                     if observed[key] != recorded[key])
    bad_done = [(z, charge, cfg) for z, charge, cfg, status, energy, nms,
                converged in configurations if status == "done" and
                (converged != 1 or energy is None or not math.isfinite(energy)
                 or nms is None or not math.isfinite(nms))]
    missing_grounds = sorted(source_stages - grounds.keys())
    extra_grounds = sorted(grounds.keys() - source_stages)
    incomplete_grounds = sorted((z, charge, status) for (z, charge),
                                (status, *_rest) in grounds.items()
                                if status != "done")
    ground_counts = Counter((z, charge) for z, charge, *_rest in ground_rows)
    duplicate_grounds = sorted((z, charge, count)
                               for (z, charge), count in ground_counts.items()
                               if count != 1)
    nonpositive_ionizations = []
    for (z, charge), (status, energy, *_rest) in grounds.items():
        upper = grounds.get((z, charge + 1))
        if (status == "done" and upper is not None and upper[0] == "done"
                and energy is not None and upper[1] is not None
                and upper[1] <= energy):
            nonpositive_ionizations.append((z, charge, upper[1] - energy))
    nonpositive_ionizations.sort()
    statuses = dict(sorted(Counter(row[3] for row in configurations).items()))
    rydberg_series = defaultdict(list)
    for z, charge, configuration, status, energy, nms, _converged in configurations:
        if (status != "done" or energy is None or nms is None
                or not math.isfinite(energy) or not math.isfinite(nms)):
            continue
        outer = re.search(r"(\d+)([spdfghik])1$", configuration)
        if outer is not None:
            key = (z, charge, configuration[:outer.start()], outer[2])
            # energy_ha already includes the normal mass correction. Adding
            # nms_ha again would double-count it in this ordering check.
            rydberg_series[key].append((int(outer[1]), energy))
    order_warnings = []
    checked_series = 0
    for (z, charge, core, ell), rows in rydberg_series.items():
        if len(rows) < 3:
            continue
        checked_series += 1
        for (n0, e0), (n1, e1) in zip(sorted(rows), sorted(rows)[1:]):
            if e1 <= e0 + 1e-8:
                order_warnings.append({"z": z, "charge": charge,
                                       "core": core, "orbital": ell,
                                       "n_lower": n0, "n_upper": n1,
                                       "energy_lower_ha": e0,
                                       "energy_upper_ha": e1})
    complete = (not absent and not extra and not changed and not bad_done
                and not order_warnings
                and not missing_grounds and not extra_grounds
                and not duplicate_grounds and not nonpositive_ionizations
                and not incomplete_grounds and statuses == {"done": len(configurations)})
    return {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "scope": f"Z={min_z}–{max_z}",
        "nist_observed_line_rows": len(observed),
        "dhf_line_rows": len(recorded),
        "missing_line_keys": absent,
        "extra_line_keys": extra,
        "changed_line_values": changed,
        "line_statuses": line_statuses,
        "nist_ion_stages": len(source_stages),
        "ground_jobs": len(grounds),
        "ground_job_rows": len(ground_rows),
        "missing_ground_stages": missing_grounds,
        "extra_ground_stages": extra_grounds,
        "duplicate_ground_stages": duplicate_grounds,
        "incomplete_ground_stages": incomplete_grounds,
        "nonpositive_ground_ionizations": nonpositive_ionizations,
        "configuration_jobs": len(configurations),
        "configuration_statuses": statuses,
        "invalid_done_jobs": bad_done,
        "rydberg_series_checked": checked_series,
        "rydberg_energy_order_warnings": order_warnings,
        "source_identity_verified": not (absent or extra or changed),
        "all_configurations_complete": complete,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--min-z", type=int, default=1)
    parser.add_argument("--max-z", type=int, default=20)
    parser.add_argument("--report", type=Path,
                        help="default: first20 or full DHF audit JSON")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit(args.archive, args.ledger,
                   min_z=args.min_z, max_z=args.max_z)
    report = args.report or (REPORT if args.min_z == 1 and args.max_z == 20
                             else REPORT.with_name("full_dhf_audit.json"))
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key in ("nist_observed_line_rows", "dhf_line_rows",
                                 "nist_ion_stages", "ground_jobs",
                                 "configuration_statuses",
                                 "rydberg_series_checked",
                                 "source_identity_verified",
                                 "all_configurations_complete")}, indent=2))
    valid = (result["source_identity_verified"] and
             not result["missing_ground_stages"] and
             not result["extra_ground_stages"] and
             not result["duplicate_ground_stages"] and
             not result["incomplete_ground_stages"] and
             not result["nonpositive_ground_ionizations"] and
             not result["invalid_done_jobs"] and
             not result["rydberg_energy_order_warnings"])
    return 0 if valid and (not args.require_complete or
                           result["all_configurations_complete"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
