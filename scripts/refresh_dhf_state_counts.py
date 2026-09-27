#!/usr/bin/env python3
"""Reconcile DHF state-count gates after NIST level-source supplements."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from generate_full_dhf_nist import (ARCHIVE, OUTPUT, expand_nist_config,
                                     update_lines)


def refresh(archive: Path = ARCHIVE, ledger: Path = OUTPUT) -> list[dict]:
    with sqlite3.connect(archive) as source, sqlite3.connect(ledger,
                                                              timeout=120) as jobs:
        jobs.execute("PRAGMA busy_timeout=120000")
        columns = {row[1] for row in source.execute("PRAGMA table_info(levels)")}
        if "source_variant" not in columns:
            return []
        affected = list(source.execute("""SELECT DISTINCT z,charge FROM levels
            WHERE source_variant IN ('term_order','line_endpoint')
            ORDER BY z,charge"""))
        report = []
        for z, charge in affected:
            state_sets: dict[str, set[tuple[str, str, float]]] = {}
            for configuration, term, j, energy in source.execute("""SELECT
                    configuration,term,j_text,energy_cm1 FROM levels
                    WHERE z=? AND charge=? AND energy_cm1 IS NOT NULL""",
                    (z, charge)):
                full = expand_nist_config(configuration or "", z - charge)
                if full:
                    state_sets.setdefault(full, set()).add(
                        (term or "", j or "", energy))
            before = jobs.execute("""SELECT COUNT(*) FROM config_state_counts
                WHERE z=? AND charge=?""", (z, charge)).fetchone()[0]
            with jobs:
                jobs.execute("""DELETE FROM config_state_counts
                    WHERE z=? AND charge=?""", (z, charge))
                jobs.executemany("""INSERT INTO config_state_counts
                    VALUES (?,?,?,?)""",
                    [(z, charge, cfg, len(states))
                     for cfg, states in state_sets.items()])
                update_lines(jobs, z, charge)
            report.append({"z": z, "charge": charge,
                           "configuration_counts_before": before,
                           "configuration_counts_after": len(state_sets)})
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--ledger", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(refresh(args.archive, args.ledger), indent=2))


if __name__ == "__main__":
    main()
