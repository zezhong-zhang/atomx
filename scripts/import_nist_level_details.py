#!/usr/bin/env python3
"""Import level IDs, statistical weights, and splitting from direct NIST ASD.

Every response is cached and matched to the archived level sequence before
supplementary fields are attached. A mismatch leaves archived rows untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))
from atomx.reference_data.nist import _to_float, normalise_configuration  # noqa: E402
from build_nist_spectra import ion_label  # noqa: E402
from import_nist_line_details import ENERGY_PREFIX  # noqa: E402

DEFAULT_OUT = ROOT / "output/nist_spectra"
BASE_URL = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"
PARAMETERS = {
    "units": "0", "format": "3", "output": "0", "multiplet_ordered": "0",
    "conf_out": "on", "term_out": "on", "level_out": "on",
    "unc_out": "1", "j_out": "on", "g_out": "on", "ids_out": "1",
    "lande_out": "on", "perc_out": "on", "biblio": "on",
    "splitting": "1", "submit": "Retrieve Data", "de": "0",
}
EXTRA_COLUMNS = {
    "statistical_weight_raw": "g",
    "splitting_raw": "Splitting",
    "level_id": "Level ID",
    "energy_prefix_raw": "Prefix",
    "energy_suffix_raw": "Suffix",
}
SOURCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS nist_level_detail_sources (
 z INTEGER NOT NULL, charge INTEGER NOT NULL, url TEXT NOT NULL,
 sha256 TEXT, source_rows INTEGER, matched_rows INTEGER,
 status TEXT NOT NULL, error TEXT,
 PRIMARY KEY(z,charge));
"""


def query_url(z: int, charge: int) -> str:
    return BASE_URL + "?" + urllib.parse.urlencode({
        "spectrum": ion_label(z, charge), **PARAMETERS})


def parse_response(body: bytes) -> list[dict[str, str]]:
    content = body.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(content), delimiter="\t")
    required = {"Configuration", "Level (cm-1)"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("NIST response lacks a level table")
    rows = list(reader)
    if not rows:
        raise ValueError("NIST response has no level rows")
    return rows


def fetch_one(out: Path, z: int, charge: int, offline: bool,
              refresh: bool) -> tuple[int, int, str, bytes]:
    path = out / "official_level_details" / f"asd.{z:03d}{charge:03d}.tsv"
    url = query_url(z, charge)
    if path.exists() and (offline or not refresh):
        return z, charge, url, path.read_bytes()
    if offline:
        raise FileNotFoundError(str(path))
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read()
            break
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    part.write_bytes(body)
    part.replace(path)
    return z, charge, url, body


def _energy(text: str) -> float | None:
    value = _to_float(text.strip())
    if value is not None:
        return value
    match = ENERGY_PREFIX.match(text.strip())
    return float(match.group(1)) if match else None


def matches_archive(db: sqlite3.Connection, z: int, charge: int,
                    rows: list[dict[str, str]]) -> tuple[bool, str]:
    has_variant = "source_variant" in {row[1] for row in db.execute(
        "PRAGMA table_info(levels)")}
    original_only = " AND source_variant='energy_order'" if has_variant else ""
    archived = db.execute("""SELECT row_index,configuration_nist,j_text,
        energy_cm1 FROM levels WHERE z=? AND charge=?""" + original_only +
        " ORDER BY row_index", (z, charge)).fetchall()
    if len(rows) != len(archived):
        return False, f"row count: official {len(rows)}, archive {len(archived)}"
    for index, (stored, source) in enumerate(zip(archived, rows)):
        row_index, configuration, j_text, expected = stored
        if row_index != index:
            return False, f"archive row index {row_index} at position {index}"
        actual = _energy(source["Level (cm-1)"])
        if (actual is None) != (expected is None):
            return False, f"row {index} energy: {actual} versus {expected}"
        if actual is not None and abs(actual - expected) > max(
                1e-8, 1e-7 * abs(expected)):
            return False, f"row {index} energy: {actual} versus {expected}"
        official_config = normalise_configuration(source["Configuration"].strip())
        archived_config = normalise_configuration(configuration or "")
        if official_config and archived_config and official_config != archived_config:
            return False, (f"row {index} configuration: {official_config!r} "
                           f"versus {archived_config!r}")
        official_j = (source.get("J") or "").strip()
        if official_j and j_text and official_j != j_text:
            return False, f"row {index} J: {official_j!r} versus {j_text!r}"
    return True, ""


def prepare_database(db: sqlite3.Connection) -> None:
    db.executescript(SOURCE_SCHEMA)
    existing = {row[1] for row in db.execute("PRAGMA table_info(levels)")}
    for column in EXTRA_COLUMNS:
        if column not in existing:
            db.execute(f"ALTER TABLE levels ADD COLUMN {column} TEXT")
    db.commit()


def import_one(db: sqlite3.Connection, z: int, charge: int, url: str,
               body: bytes) -> tuple[bool, str]:
    digest = hashlib.sha256(body).hexdigest()
    try:
        rows = parse_response(body)
        matched, error = matches_archive(db, z, charge, rows)
        if not matched:
            raise ValueError(error)
        assignments = ",".join(f"{column}=?" for column in EXTRA_COLUMNS)
        db.executemany(f"""UPDATE levels SET {assignments}
            WHERE z=? AND charge=? AND row_index=?""", [
            tuple((source.get(name) or "").strip() or None
                  for name in EXTRA_COLUMNS.values()) + (z, charge, index)
            for index, source in enumerate(rows)])
        db.execute("""INSERT OR REPLACE INTO nist_level_detail_sources
            VALUES (?,?,?,?,?,?,'imported',NULL)""",
            (z, charge, url, digest, len(rows), len(rows)))
        db.commit()
        return True, f"{len(rows)} rows"
    except Exception as exc:
        db.execute("""INSERT OR REPLACE INTO nist_level_detail_sources
            VALUES (?,?,?,?,?,?,'mismatch',?)""",
            (z, charge, url, digest,
             len(rows) if 'rows' in locals() else None,
             0, f"{type(exc).__name__}: {exc}"[:1000]))
        db.commit()
        return False, f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-z", type=int, default=1)
    parser.add_argument("--max-z", type=int, default=118)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.min_z <= args.max_z <= 118 or args.workers < 1:
        parser.error("require 1 <= min-z <= max-z <= 118 and workers >= 1")
    out = args.out.resolve()
    db = sqlite3.connect(out / "spectra.sqlite", timeout=120)
    db.execute("PRAGMA busy_timeout=120000")
    db.execute("PRAGMA journal_mode=WAL")
    prepare_database(db)
    species = db.execute("""SELECT z,charge FROM species WHERE level_rows>0
        AND z BETWEEN ? AND ? ORDER BY z,charge""",
        (args.min_z, args.max_z)).fetchall()
    if args.limit is not None:
        species = species[:args.limit]
    previous = {(z, q): status for z, q, status in db.execute(
        "SELECT z,charge,status FROM nist_level_detail_sources")}
    requested = len(species)
    if not args.refresh:
        species = [(z, q) for z, q in species
                   if previous.get((z, q)) != "imported"]
    succeeded = failed = missing = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(fetch_one, out, z, q, args.offline,
                                  args.refresh): (z, q) for z, q in species}
        for future in as_completed(future_map):
            z, charge = future_map[future]
            try:
                _, _, url, body = future.result()
                okay, message = import_one(db, z, charge, url, body)
                succeeded += int(okay)
                failed += int(not okay)
            except FileNotFoundError as exc:
                missing += 1
                message = f"not cached: {exc}"
            except Exception as exc:
                failed += 1
                message = f"{type(exc).__name__}: {exc}"
            if not args.quiet or not message.endswith(" rows"):
                print(f"{ion_label(z, charge)}: {message}", flush=True)
    print(f"imported={succeeded} failed={failed} uncached={missing} "
          f"already_imported={requested - len(species)}", flush=True)
    imported = db.execute("""SELECT COUNT(*) FROM nist_level_detail_sources
        WHERE status='imported'""").fetchone()[0]
    mismatched = db.execute("""SELECT COUNT(*) FROM nist_level_detail_sources
        WHERE status='mismatch'""").fetchone()[0]
    total_spectra = db.execute(
        "SELECT COUNT(*) FROM species WHERE level_rows>0").fetchone()[0]
    coverage_path = out / "coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())
        coverage.update({
            "level_details_imported_ions": imported,
            "level_details_mismatched_ions": mismatched,
            "level_details_coverage_complete": (
                imported == total_spectra and mismatched == 0),
        })
        part = coverage_path.with_suffix(".json.part")
        part.write_text(json.dumps(coverage, indent=2) + "\n")
        part.replace(coverage_path)
    db.close()
    return 1 if failed or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
