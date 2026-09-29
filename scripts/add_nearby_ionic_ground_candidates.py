#!/usr/bin/env python3
"""Queue the second-nearest NIST ionic ground shell as a DHF candidate."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_full_dhf_nist import ARCHIVE, OUTPUT, ground_configuration_for_species
from atomx.reference_data.matching import expand_nist_config

REPORT = ROOT / "output/nist_spectra/dhf_nearby_ionic_candidates.json"


def add_candidates(archive: Path = ARCHIVE, ledger: Path = OUTPUT,
                   *, dry_run: bool = False) -> dict:
    rows = []
    with sqlite3.connect(archive) as source, sqlite3.connect(
            ledger, timeout=60) as jobs:
        jobs.execute("PRAGMA busy_timeout=60000")
        stages = jobs.execute("""SELECT z,charge,configuration,isotope_a,
            nuclear_mass_me FROM config_jobs WHERE is_ground=1
            ORDER BY z,charge""").fetchall()
        for z, charge, selected, isotope_a, mass in stages:
            _, selection = ground_configuration_for_species(
                source, z, charge, has_ionization=True)
            if not selection.startswith("nist_isoelectronic_z"):
                continue
            electrons = z - charge
            seen = set()
            variants = []
            for reference_z, shells in source.execute("""SELECT z,ground_shells
                    FROM nist_ionization WHERE z<? AND z-charge=?
                      AND charge>0 AND ground_shells IS NOT NULL
                    ORDER BY z DESC LIMIT 2""", (z, electrons)):
                config = expand_nist_config(shells, electrons)
                if config is None or config in seen:
                    continue
                seen.add(config)
                variants.append((reference_z, shells, config))
                if len(variants) == 2:
                    break
            if len(variants) < 2:
                continue
            reference_z, shells, alternate = variants[1]
            existing = jobs.execute("""SELECT status FROM config_jobs
                WHERE z=? AND charge=? AND configuration=?""",
                (z, charge, alternate)).fetchone()
            if existing is not None:
                continue
            rows.append(dict(z=z, charge=charge, configuration=alternate,
                             first_reference_z=variants[0][0],
                             second_reference_z=reference_z,
                             second_reference_ground_shells=shells,
                             isotope_a=isotope_a, nuclear_mass_me=mass,
                             selected_before=selected))
        if not dry_run and rows:
            jobs.execute("BEGIN IMMEDIATE")
            jobs.executemany("""INSERT OR IGNORE INTO config_jobs
                (z,charge,configuration,status,isotope_a,nuclear_mass_me)
                VALUES (?,?,?,'candidate_pending',?,?)""",
                [(row["z"], row["charge"], row["configuration"],
                  row["isotope_a"], row["nuclear_mass_me"]) for row in rows])
            jobs.commit()
    return dict(checked_utc=datetime.now(timezone.utc).isoformat(),
                dry_run=dry_run, new_candidates=len(rows), candidates=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--ledger", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = add_candidates(args.archive, args.ledger,
                            dry_run=args.dry_run)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    part = args.report.with_suffix(args.report.suffix + ".part")
    part.write_text(json.dumps(result, indent=2) + "\n")
    part.replace(args.report)
    print(json.dumps({key: result[key] for key in ("dry_run", "new_candidates")}))


if __name__ == "__main__":
    main()
