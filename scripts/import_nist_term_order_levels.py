#!/usr/bin/env python3
"""Recover NIST levels omitted by its energy-ordered export.

For four ions NIST reports more levels than its energy-ordered table returns.
The term-ordered table supplies the missing Level IDs. Common IDs must agree
before new rows are appended; the original row indices remain stable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sqlite3
import urllib.parse
import urllib.request
from pathlib import Path

from build_nist_spectra import ion_label
from import_nist_level_details import (BASE_URL, PARAMETERS,
                                       _energy as parse_energy)
from atomx.reference_data.nist import _to_float, normalise_configuration

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output/nist_spectra"
TARGETS = ((35, 3), (60, 1), (79, 0), (79, 1))
SOURCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS nist_term_order_level_sources (
 z INTEGER NOT NULL, charge INTEGER NOT NULL, url TEXT NOT NULL,
 sha256 TEXT NOT NULL, energy_rows INTEGER NOT NULL,
 term_rows INTEGER NOT NULL, union_rows INTEGER NOT NULL,
 newly_imported_rows INTEGER NOT NULL, declared_rows INTEGER NOT NULL,
 status TEXT NOT NULL, PRIMARY KEY(z,charge));
"""
LEVEL_COLUMNS = (
    "z", "charge", "row_index", "configuration", "configuration_nist",
    "term", "j_text", "energy_cm1", "uncertainty_cm1", "reference",
    "flags", "bracketed", "source_row", "energy_raw", "energy_kind",
    "lande_g_raw", "leading_percentages", "statistical_weight_raw",
    "splitting_raw", "level_id", "energy_prefix_raw", "energy_suffix_raw",
    "source_variant",
)
IDENTITY_FIELDS = ("Configuration", "Term", "J", "g", "Prefix",
                   "Level (cm-1)", "Suffix", "Uncertainty (cm-1)",
                   "Reference")


def term_url(z: int, charge: int) -> str:
    return BASE_URL + "?" + urllib.parse.urlencode({
        **PARAMETERS, "spectrum": ion_label(z, charge),
        "multiplet_ordered": "1"})


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows or any(not row.get("Level ID") for row in rows):
        raise ValueError(f"missing NIST Level IDs in {path}")
    if len({row["Level ID"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate NIST Level IDs in {path}")
    return rows


def ensure_term_source(out: Path, z: int, charge: int, *, offline: bool,
                       refresh: bool) -> None:
    path = out / "official_level_term_order" / f"asd.{z:03d}{charge:03d}.tsv"
    if path.exists() and not refresh:
        return
    if offline:
        raise FileNotFoundError(path)
    request = urllib.request.Request(term_url(z, charge), headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                      "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"})
    with urllib.request.urlopen(request, timeout=90) as response:
        body = response.read()
    rows = list(csv.DictReader(io.StringIO(body.decode("utf-8", "replace")),
                               delimiter="\t"))
    if not rows or "Level ID" not in rows[0]:
        raise ValueError(f"NIST term-ordered response lacks levels: {term_url(z, charge)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_bytes(body)
    temporary.replace(path)


def labelled_rows(rows: list[dict[str, str]]):
    """Carry a label only within a displayed term group."""
    configuration = term = ""
    for source_row, row in enumerate(rows, start=2):
        raw_configuration = row["Configuration"].strip()
        raw_term = row["Term"].strip()
        if raw_configuration:
            configuration = raw_configuration
            term = raw_term
        elif raw_term:
            configuration = ""
            term = raw_term
        yield source_row, row, (raw_configuration or configuration), (raw_term or term)


def supplemental_values(z: int, charge: int, row_index: int,
                        source_row: int, row: dict[str, str],
                        configuration: str, term: str) -> tuple:
    prefix = row.get("Prefix", "").strip()
    value = row["Level (cm-1)"].strip()
    suffix = row.get("Suffix", "").strip()
    energy = parse_energy(value)
    if energy is None:
        raise ValueError(f"supplemental level {row['Level ID']} has no energy")
    bracketed = prefix.startswith("[") or value.startswith("[")
    kind = ("semiempirical" if bracketed else
            "theoretical" if prefix.startswith("(") or value.startswith("(")
            else "evaluated")
    flags = (["bracketed"] if bracketed else
             ["theoretical"] if kind == "theoretical" else [])
    if suffix and suffix not in ("]", ")"):
        flags.append(suffix)
    return (
        z, charge, row_index, normalise_configuration(configuration),
        configuration, term, row["J"].strip(), energy,
        _to_float(row.get("Uncertainty (cm-1)", "").strip()),
        row.get("Reference", "").strip() or None, json.dumps(flags),
        int(bracketed), source_row, prefix + value + suffix, kind,
        row.get("Lande", "").strip() or None,
        row.get("Leading percentages", "").strip() or None,
        row.get("g", "").strip() or None,
        row.get("Splitting", "").strip() or None,
        row["Level ID"].strip(), prefix or None, suffix or None,
        "term_order",
    )


def import_one(db: sqlite3.Connection, out: Path, z: int, charge: int,
               *, dry_run: bool = False) -> dict:
    name = f"asd.{z:03d}{charge:03d}.tsv"
    energy_path = out / "official_level_details" / name
    term_path = out / "official_level_term_order" / name
    energy_rows = read_rows(energy_path)
    term_rows = read_rows(term_path)
    energy_by_id = {row["Level ID"]: row for row in energy_rows}
    term_by_id = {row["Level ID"]: row for row in term_rows}
    existing_ids = {row[0] for row in db.execute(
        "SELECT level_id FROM levels WHERE z=? AND charge=?", (z, charge))}
    if not existing_ids.issuperset(energy_by_id):
        raise ValueError(f"{z}:{charge} archive lacks energy-ordered Level IDs")
    for level_id in energy_by_id.keys() & term_by_id.keys():
        left, right = energy_by_id[level_id], term_by_id[level_id]
        for field in IDENTITY_FIELDS:
            if left.get(field) and right.get(field) and left[field] != right[field]:
                raise ValueError(f"{z}:{charge} {level_id} differs in {field}")
    declared = db.execute("""SELECT declared_rows FROM source_count_discrepancies
        WHERE kind='levels' AND z=? AND charge=?""", (z, charge)).fetchone()
    if declared is None or len(energy_by_id.keys() | term_by_id.keys()) != declared[0]:
        raise ValueError(f"{z}:{charge} union does not match NIST declared count")
    additional = term_by_id.keys() - existing_ids
    start_index = db.execute("""SELECT COALESCE(MAX(row_index),-1)+1
        FROM levels WHERE z=? AND charge=?""", (z, charge)).fetchone()[0]
    records = [supplemental_values(z, charge, start_index + offset,
                                  source_row, row, configuration, term)
               for offset, (source_row, row, configuration, term) in enumerate(
                   item for item in labelled_rows(term_rows)
                   if item[1]["Level ID"] in additional)]
    if not dry_run:
        columns = ",".join(LEVEL_COLUMNS)
        placeholders = ",".join("?" for _ in LEVEL_COLUMNS)
        db.executemany(f"INSERT INTO levels ({columns}) VALUES ({placeholders})",
                       records)
        db.execute("""UPDATE species SET level_rows=(SELECT COUNT(*) FROM levels
            WHERE z=? AND charge=?) WHERE z=? AND charge=?""",
            (z, charge, z, charge))
        db.execute("""INSERT OR REPLACE INTO nist_term_order_level_sources
            VALUES (?,?,?,?,?,?,?,?,?,'imported')""",
            (z, charge, term_url(z, charge),
             hashlib.sha256(term_path.read_bytes()).hexdigest(),
             len(energy_rows), len(term_rows),
             len(energy_by_id.keys() | term_by_id.keys()),
             len(term_by_id.keys() - energy_by_id.keys()), declared[0]))
    return {"ion": ion_label(z, charge), "energy_rows": len(energy_rows),
            "term_rows": len(term_rows), "union_rows": declared[0],
            "new_rows": len(records)}


def update_coverage(db: sqlite3.Connection, out: Path) -> None:
    path = out / "coverage.json"
    coverage = json.loads(path.read_text())
    coverage["levels"] = db.execute("SELECT COUNT(*) FROM levels").fetchone()[0]
    coverage["term_order_reconciled_spectra"] = db.execute(
        "SELECT COUNT(*) FROM nist_term_order_level_sources WHERE status='imported'").fetchone()[0]
    coverage["term_order_added_level_rows"] = db.execute(
        "SELECT COUNT(*) FROM levels WHERE source_variant='term_order'").fetchone()[0]
    coverage["reconciled_source_count_discrepancies"] = (
        coverage["term_order_reconciled_spectra"])
    coverage["unreconciled_source_count_discrepancies"] = max(
        0, coverage["source_count_discrepancies"]
        - coverage["reconciled_source_count_discrepancies"])
    coverage["official_level_holdings_gap_after_reconciliation"] = (
        coverage["official_levels_all_holdings"] - coverage["levels"]
        - coverage["isotope_levels_rows"])
    path.write_text(json.dumps(coverage, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline", action="store_true",
                        help="use only cached direct NIST term-order exports")
    parser.add_argument("--refresh", action="store_true",
                        help="refresh the direct NIST term-order exports")
    args = parser.parse_args()
    out = args.out.resolve()
    for z, charge in TARGETS:
        ensure_term_source(out, z, charge, offline=args.offline,
                           refresh=args.refresh)
    with sqlite3.connect(out / "spectra.sqlite", timeout=120) as db:
        db.execute("PRAGMA busy_timeout=120000")
        if not args.dry_run:
            db.execute(SOURCE_SCHEMA)
            columns = {row[1] for row in db.execute("PRAGMA table_info(levels)")}
            if "source_variant" not in columns:
                db.execute("""ALTER TABLE levels ADD COLUMN source_variant TEXT
                    NOT NULL DEFAULT 'energy_order'""")
        report = [import_one(db, out, z, charge, dry_run=args.dry_run)
                  for z, charge in TARGETS]
        if not args.dry_run:
            db.commit()
            update_coverage(db, out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
