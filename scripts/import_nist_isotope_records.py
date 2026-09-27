#!/usr/bin/env python3
"""Import every NIST isotope-specific spectrum listed by its holdings pages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sqlite3
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_nist_isotope_holdings import DEFAULT_OUT  # noqa: E402
from build_nist_spectra import roman_to_int  # noqa: E402
from import_nist_level_details import (BASE_URL as LEVEL_URL,
                                       PARAMETERS as LEVEL_PARAMS)  # noqa: E402
from import_nist_line_details import (BASE_URL as LINE_URL,
                                      PARAMETERS as LINE_PARAMS)  # noqa: E402

SCHEMA = """
CREATE TABLE IF NOT EXISTS nist_isotope_sources (
 kind TEXT NOT NULL, spectrum TEXT NOT NULL, z INTEGER NOT NULL,
 mass_number INTEGER NOT NULL, charge INTEGER NOT NULL,
 url TEXT NOT NULL, sha256 TEXT, holdings_rows INTEGER NOT NULL,
 parsed_rows INTEGER, status TEXT NOT NULL, error TEXT,
 PRIMARY KEY(kind,spectrum));
CREATE TABLE IF NOT EXISTS nist_isotope_records (
 kind TEXT NOT NULL, spectrum TEXT NOT NULL, row_index INTEGER NOT NULL,
 record_json TEXT NOT NULL, PRIMARY KEY(kind,spectrum,row_index));
CREATE INDEX IF NOT EXISTS nist_isotope_by_element
 ON nist_isotope_sources(z,charge,mass_number,kind);
"""
LABEL = re.compile(r"^(?:(\d+)([A-Z][a-z]?)|(D|T))\s+([IVXLCDM]+)$")


def isotope_identity(label: str, z_by_symbol: dict[str, int]) -> tuple[int, int, int]:
    match = LABEL.fullmatch(label)
    if match is None:
        raise ValueError(f"not an isotope spectrum: {label}")
    if match.group(3):
        z, mass = 1, 2 if match.group(3) == "D" else 3
    else:
        mass = int(match.group(1))
        z = z_by_symbol[match.group(2)]
    charge = roman_to_int(match.group(4)) - 1
    if not 0 <= charge < z:
        raise ValueError(f"invalid ionization stage: {label}")
    return z, mass, charge


def query_url(kind: str, label: str) -> str:
    base, params, key = ((LINE_URL, LINE_PARAMS, "spectra") if kind == "lines"
                         else (LEVEL_URL, LEVEL_PARAMS, "spectrum"))
    return base + "?" + urllib.parse.urlencode({key: label, **params})


def fetch_one(out: Path, kind: str, label: str, offline: bool,
              refresh: bool) -> tuple[str, str, bytes]:
    url = query_url(kind, label)
    path = out / "official_isotope_records" / kind / (label.replace(" ", "_") + ".tsv")
    if path.exists() and (offline or not refresh):
        return label, url, path.read_bytes()
    if offline:
        raise FileNotFoundError(path)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".part").write_bytes(body)
    path.with_suffix(".part").replace(path)
    return label, url, body


def parse_records(body: bytes, kind: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(body.decode("utf-8", errors="replace")),
                            delimiter="\t")
    columns = set(reader.fieldnames or [])
    expected = ({"obs_wl_vac(nm)", "ritz_wl_vac(nm)"}
                if kind == "lines" else {"Configuration", "Level (cm-1)"})
    if not expected <= columns:
        raise ValueError(f"NIST {kind} export lacks {sorted(expected - columns)}")
    result = list(reader)
    if any(None in row for row in result):
        raise ValueError("NIST export has more cells than its header")
    return result


def import_one(db: sqlite3.Connection, kind: str, label: str,
               z: int, mass: int, charge: int, expected: int,
               url: str, body: bytes) -> tuple[bool, str]:
    digest = hashlib.sha256(body).hexdigest()
    try:
        records = parse_records(body, kind)
        if len(records) != expected:
            raise ValueError(f"holdings={expected}, export={len(records)}")
    except Exception as exc:
        with db:
            db.execute("""INSERT OR REPLACE INTO nist_isotope_sources
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (kind, label, z, mass, charge, url, digest, expected,
                 None, "mismatch", f"{type(exc).__name__}: {exc}"))
        return False, f"mismatch: {exc}"
    with db:
        db.execute("DELETE FROM nist_isotope_records WHERE kind=? AND spectrum=?",
                   (kind, label))
        db.executemany("""INSERT INTO nist_isotope_records VALUES (?,?,?,?)""",
                       ((kind, label, index, json.dumps(row, ensure_ascii=False))
                        for index, row in enumerate(records)))
        db.execute("""INSERT OR REPLACE INTO nist_isotope_sources
            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (kind, label, z, mass, charge, url, digest, expected,
             len(records), "imported", None))
    return True, f"{len(records)} rows"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--kind", choices=("lines", "levels", "both"), default="both")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    kinds = ("lines", "levels") if args.kind == "both" else (args.kind,)
    db = sqlite3.connect(args.output / "spectra.sqlite", timeout=120)
    db.execute("PRAGMA busy_timeout=120000")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    z_by_symbol = dict(db.execute("SELECT DISTINCT symbol,z FROM species"))
    failures = 0
    for kind in kinds:
        holdings = json.loads((args.output / f"official_isotope_{kind}_holdings.json").read_text())
        if holdings["elements"] != 118:
            raise ValueError(f"{kind} isotope holdings do not cover 118 elements")
        source = {(row[0], row[1]) for row in db.execute(
            "SELECT kind,spectrum FROM nist_isotope_sources WHERE status='imported'")}
        requested = [(row["spectrum"], row["count"],
                      isotope_identity(row["spectrum"], z_by_symbol))
                     for row in holdings["isotope_rows"]
                     if args.refresh or (kind, row["spectrum"]) not in source]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_one, args.output, kind, label,
                                   args.offline, args.refresh):
                       (label, expected, identity)
                       for label, expected, identity in requested}
            for future in as_completed(futures):
                label, expected, (z, mass, charge) = futures[future]
                try:
                    _, url, body = future.result()
                    ok, message = import_one(db, kind, label, z, mass, charge,
                                             expected, url, body)
                    failures += int(not ok)
                except Exception as exc:
                    failures += 1
                    message = f"{type(exc).__name__}: {exc}"
                print(f"{kind} {label}: {message}", flush=True)
    db.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
