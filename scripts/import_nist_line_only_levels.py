#!/usr/bin/env python3
"""Preserve NIST line endpoint Level IDs absent from its level exports.

These are source-backed placeholders. A missing energy, term or J stays null;
no wavelength or line energy is used to infer an unreported level energy.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output/nist_spectra"
SOURCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS nist_line_only_level_sources (
 z INTEGER NOT NULL, charge INTEGER NOT NULL, level_id TEXT NOT NULL,
 line_row_index INTEGER NOT NULL, endpoint TEXT NOT NULL,
 source_row INTEGER, configuration TEXT, line_reference TEXT,
 PRIMARY KEY(z,charge,level_id));
"""


def line_only_endpoints(db: sqlite3.Connection) -> dict[tuple, list[dict]]:
    db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS levels_unique_level_id
        ON levels(z,charge,level_id) WHERE level_id IS NOT NULL""")
    records = defaultdict(list)
    for endpoint in ("lower", "upper"):
        query = f"""SELECT l.z,l.charge,l.row_index,l.source_row,
            l.{endpoint}_level_id,l.{endpoint}_configuration,
            l.{endpoint}_term,l.{endpoint}_j,l.{endpoint}_energy_cm1,
            l.line_reference FROM lines l
            WHERE l.{endpoint}_level_id IS NOT NULL
              AND l.{endpoint}_level_id!=''
              AND NOT EXISTS (SELECT 1 FROM levels e WHERE e.z=l.z
                  AND e.charge=l.charge AND e.level_id=l.{endpoint}_level_id)"""
        for (z, charge, row_index, source_row, level_id, configuration,
             term, j, energy, line_reference) in db.execute(query):
            records[(z, charge, level_id)].append({
                "line_row_index": row_index, "source_row": source_row,
                "endpoint": endpoint, "configuration": configuration,
                "term": term, "j": j, "energy": energy,
                "line_reference": line_reference})
    return records


def import_missing(db: sqlite3.Connection) -> dict:
    db.executescript(SOURCE_SCHEMA)
    columns = {row[1] for row in db.execute("PRAGMA table_info(levels)")}
    if "source_variant" not in columns:
        db.execute("""ALTER TABLE levels ADD COLUMN source_variant TEXT
            NOT NULL DEFAULT 'energy_order'""")
    records = line_only_endpoints(db)
    changed = set()
    for (z, charge, level_id), sources in sorted(records.items()):
        for field in ("configuration", "term", "j", "energy"):
            values = {source[field] for source in sources
                      if source[field] is not None and source[field] != ""}
            if len(values) > 1:
                raise ValueError(f"{z}:{charge} {level_id} conflicts in {field}")
        representative = min(sources, key=lambda row: row["line_row_index"])
        index = db.execute("""SELECT COALESCE(MAX(row_index),-1)+1 FROM levels
            WHERE z=? AND charge=?""", (z, charge)).fetchone()[0]
        configuration = representative["configuration"] or None
        term = representative["term"] or None
        j = representative["j"] or None
        energy = representative["energy"]
        db.execute("""INSERT INTO levels (
            z,charge,row_index,configuration,configuration_nist,term,j_text,
            energy_cm1,uncertainty_cm1,reference,flags,bracketed,source_row,
            energy_raw,energy_kind,level_id,source_variant)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (z, charge, index, configuration, configuration, term, j,
             energy, None, None, json.dumps(["line_endpoint_only"]), 0,
             representative["source_row"], None,
             "missing" if energy is None else "unclassified",
             level_id, "line_endpoint"))
        db.execute("""INSERT INTO nist_line_only_level_sources
            VALUES (?,?,?,?,?,?,?,?)""",
            (z, charge, level_id, representative["line_row_index"],
             representative["endpoint"], representative["source_row"],
             configuration, representative["line_reference"]))
        changed.add((z, charge))
    for z, charge in changed:
        db.execute("""UPDATE species SET level_rows=(SELECT COUNT(*) FROM levels
            WHERE z=? AND charge=?) WHERE z=? AND charge=?""",
            (z, charge, z, charge))
    return {"added": len(records), "ions": sorted(changed),
            "source_records": sum(map(len, records.values()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    with sqlite3.connect(out / "spectra.sqlite", timeout=120) as db:
        db.execute("PRAGMA busy_timeout=120000")
        result = import_missing(db)
        db.commit()
        coverage_path = out / "coverage.json"
        coverage = json.loads(coverage_path.read_text())
        coverage["levels"] = db.execute("SELECT COUNT(*) FROM levels").fetchone()[0]
        coverage["line_endpoint_only_level_rows"] = db.execute(
            "SELECT COUNT(*) FROM levels WHERE source_variant='line_endpoint'").fetchone()[0]
        coverage["official_level_holdings_gap_after_reconciliation"] = (
            coverage["official_levels_all_holdings"] - coverage["levels"]
            - coverage["isotope_levels_rows"])
        coverage["official_level_holdings_verified"] = (
            coverage["official_level_holdings_gap_after_reconciliation"] == 0)
        coverage_path.write_text(json.dumps(coverage, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
