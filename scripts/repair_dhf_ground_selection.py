#!/usr/bin/env python3
"""Repair DHF ground jobs from direct NIST or traced ionic analogues."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_full_dhf_nist import (  # noqa: E402
    ARCHIVE, OUTPUT, expand_nist_config,
    ground_configuration_with_dhf_candidates)

REPORT = ROOT / "output/nist_spectra/dhf_ground_selection_repair.json"


def repair(archive: Path = ARCHIVE, ledger: Path = OUTPUT,
           *, dry_run: bool = False) -> dict:
    changed = []
    conflicts = []
    source_counts = Counter()
    with sqlite3.connect(archive) as source, sqlite3.connect(ledger,
                                                               timeout=60) as jobs:
        jobs.execute("PRAGMA busy_timeout=60000")
        jobs.execute("PRAGMA journal_mode=WAL")
        has_ionization = source.execute("""SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='nist_ionization'""").fetchone() is not None
        has_gsie_configuration = (has_ionization and
            "ground_configuration" in {row[1] for row in source.execute(
                "PRAGMA table_info(nist_ionization)")})
        ground_rows = jobs.execute("""SELECT z,charge,configuration,status,
            isotope_a,nuclear_mass_me FROM config_jobs WHERE is_ground=1""").fetchall()
        candidates = defaultdict(dict)
        for z, charge, config, status, energy in jobs.execute(
                """SELECT z,charge,configuration,status,energy_ha
                   FROM config_jobs"""):
            candidates[z, charge][config] = (status, energy)
        old_grounds = {(z, charge): (configuration, status, isotope_a, mass)
            for z, charge, configuration, status, isotope_a, mass in ground_rows}
        stages = source.execute(
            "SELECT z,charge FROM species ORDER BY z,charge").fetchall()
        if len(ground_rows) != len(stages) or len(old_grounds) != len(stages):
            raise RuntimeError("expected exactly one current ground job per ion stage")
        for z, charge in stages:
            desired, selection = ground_configuration_with_dhf_candidates(
                source, z, charge, has_ionization=has_ionization,
                candidates=candidates[z, charge])
            source_counts[selection] += 1
            old = old_grounds.get((z, charge))
            if old is None:
                raise RuntimeError(f"missing current ground job Z={z}, q={charge}")
            old_config, old_status, isotope_a, mass = old
            if has_ionization:
                ionization = source.execute(
                    "SELECT ground_shells" + (
                        ",ground_configuration" if has_gsie_configuration else "")
                    + " FROM nist_ionization WHERE z=? AND charge=?",
                    (z, charge)).fetchone()
                from_ie = (expand_nist_config(ionization[0] or "", z-charge)
                           if ionization else None)
                if (selection == "nist_levels" and from_ie is not None
                        and from_ie != desired):
                    conflicts.append(dict(z=z, charge=charge,
                        selected_level_configuration=desired,
                        ionization_ground_shells=ionization[0],
                        ionization_ground_configuration=(
                            ionization[1] if has_gsie_configuration else None),
                        ionization_configuration=from_ie))
            if desired != old_config:
                existing = jobs.execute("""SELECT status FROM config_jobs
                    WHERE z=? AND charge=? AND configuration=?""",
                    (z, charge, desired)).fetchone()
                changed.append(dict(z=z, charge=charge, old=old_config,
                    old_status=old_status, new=desired,
                    new_status=existing[0] if existing else "new_pending",
                    source=selection, isotope_a=isotope_a,
                    nuclear_mass_me=mass))
        if not dry_run and changed:
            jobs.execute("BEGIN IMMEDIATE")
            for row in changed:
                z, charge = row["z"], row["charge"]
                current = jobs.execute("""SELECT configuration FROM config_jobs
                    WHERE z=? AND charge=? AND is_ground=1""",
                    (z, charge)).fetchall()
                if current != [(row["old"],)]:
                    raise RuntimeError(f"ground selection changed during repair: Z={z}, q={charge}")
                jobs.execute("""INSERT OR IGNORE INTO config_jobs
                    (z,charge,configuration,status,isotope_a,nuclear_mass_me)
                    VALUES (?,?,?,'pending',?,?)""",
                    (z, charge, row["new"], row["isotope_a"],
                     row["nuclear_mass_me"]))
                jobs.execute("""UPDATE config_jobs SET is_ground=0
                    WHERE z=? AND charge=? AND configuration=?""",
                    (z, charge, row["old"]))
                jobs.execute("""UPDATE config_jobs SET is_ground=1
                    WHERE z=? AND charge=? AND configuration=?""",
                    (z, charge, row["new"]))
            jobs.execute("""INSERT OR REPLACE INTO metadata VALUES
                ('ground_selection_version','nist_levels_gsie_isoelectronic_v2')""")
            jobs.commit()
    return dict(audited_utc=datetime.now(timezone.utc).isoformat(),
                dry_run=dry_run, stages=len(stages),
                selection_sources=dict(source_counts),
                repaired_count=len(changed), repaired=changed,
                nist_source_conflict_count=len(conflicts),
                nist_source_conflicts=conflicts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--ledger", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = repair(args.archive, args.ledger, dry_run=args.dry_run)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    part = args.report.with_suffix(args.report.suffix + ".part")
    part.write_text(json.dumps(result, indent=2) + "\n")
    part.replace(args.report)
    print(json.dumps({key: result[key] for key in (
        "dry_run", "stages", "selection_sources", "repaired_count",
        "nist_source_conflict_count")}, indent=2))


if __name__ == "__main__":
    main()
