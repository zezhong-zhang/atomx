#!/usr/bin/env python3
"""Import optional line columns from direct NIST ASD tabular exports.

The existing archive supplies the row inventory. Each extended NIST response
must match its archived observed/Ritz wavelengths, A values, and level energies
in order before any optional columns are attached. Raw responses are cached so
the supplementary import can be rebuilt offline.
"""

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
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))
from nist_asd import _to_float  # noqa: E402
from build_nist_spectra import ion_label  # noqa: E402

DEFAULT_OUT = ROOT / "output/nist_spectra"
BASE_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
PARAMETERS = {
    "limits_type": "0", "unit": "1", "submit": "Retrieve Data",
    "de": "0", "format": "3", "line_out": "0", "remove_js": "on",
    "en_unit": "0", "output": "0", "bibrefs": "1", "show_obs_wl": "1",
    "show_calc_wl": "1", "unc_out": "1", "show_av": "3", "A_out": "0",
    "allowed_out": "1", "forbid_out": "1", "conf_out": "on",
    "term_out": "on", "enrg_out": "on", "J_out": "on",
    "intens_out": "on", "f_out": "on", "S_out": "on",
    "loggf_out": "on", "g_out": "on", "ids_out": "1",
    "show_diff_obs_calc": "1", "show_wn": "1",
}
EXTRA_COLUMNS = {
    "observed_minus_ritz_raw": "obs-ritz",
    "wavenumber_raw": "wn(cm-1)",
    "relative_intensity_raw": "intens",
    "oscillator_strength_raw": "fik",
    "line_strength_raw": "S(a.u.)",
    "log_gf_raw": "log_gf",
    "lower_g_raw": "g_i",
    "upper_g_raw": "g_k",
    "lower_level_id": "ID_i",
    "upper_level_id": "ID_k",
}
VERIFY_COLUMNS = {
    "observed_nm": "obs_wl_vac(nm)",
    "ritz_nm": "ritz_wl_vac(nm)",
    "aki_s1": "Aki(s^-1)",
    "lower_energy_cm1": "Ei(cm-1)",
    "upper_energy_cm1": "Ek(cm-1)",
}
ENERGY_PREFIX = re.compile(r"^[\[(]?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)")
SOURCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS nist_line_detail_sources (
 z INTEGER NOT NULL, charge INTEGER NOT NULL, url TEXT NOT NULL,
 sha256 TEXT, source_rows INTEGER, matched_rows INTEGER,
 status TEXT NOT NULL, error TEXT,
 PRIMARY KEY(z,charge));
"""


def query_url(z: int, charge: int) -> str:
    return BASE_URL + "?" + urllib.parse.urlencode({
        "spectra": ion_label(z, charge), **PARAMETERS})


def parse_response(body: bytes) -> list[dict[str, str]]:
    content = body.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(content), delimiter="\t")
    if reader.fieldnames is None:
        raise ValueError("NIST response has no header")
    # ASD omits optional columns entirely for spectra without those values.
    required = set(VERIFY_COLUMNS.values())
    missing = required - set(reader.fieldnames)
    if missing:
        raise ValueError(f"NIST response lacks columns {sorted(missing)}")
    rows = list(reader)
    if not rows:
        raise ValueError("NIST response has no line rows")
    return rows


def fetch_one(out: Path, z: int, charge: int, offline: bool,
              refresh: bool) -> tuple[int, int, str, bytes]:
    path = out / "official_line_details" / f"asd.{z:03d}{charge:03d}.tsv"
    url = query_url(z, charge)
    if path.exists() and (offline or not refresh):
        return z, charge, url, path.read_bytes()
    if offline:
        raise FileNotFoundError(str(path))
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    part.write_bytes(body)
    part.replace(path)
    return z, charge, url, body


def matches_archive(db: sqlite3.Connection, z: int, charge: int,
                    rows: list[dict[str, str]]) -> tuple[bool, str]:
    archived = db.execute("""SELECT row_index,observed_nm,ritz_nm,aki_s1,
        lower_energy_cm1,upper_energy_cm1 FROM lines WHERE z=? AND charge=?
        ORDER BY row_index""", (z, charge)).fetchall()
    if len(rows) != len(archived):
        return False, f"row count: official {len(rows)}, archive {len(archived)}"
    for index, (stored, source) in enumerate(zip(archived, rows)):
        if stored[0] != index:
            return False, f"archive row index {stored[0]} at position {index}"
        for offset, (field, name) in enumerate(VERIFY_COLUMNS.items(), 1):
            actual = _to_float(source[name].strip())
            if actual is None and field.endswith("energy_cm1"):
                # NIST writes some uncertain common offsets as 50300.9+x.
                # The original HTML archive also retains the numeric base.
                match = ENERGY_PREFIX.match(source[name].strip())
                if match:
                    actual = float(match.group(1))
            expected = stored[offset]
            if (actual is None) != (expected is None):
                return False, f"row {index} {field}: {actual} versus {expected}"
            if actual is not None and abs(actual - expected) > max(
                    1e-8, 1e-5 * abs(expected)):
                return False, f"row {index} {field}: {actual} versus {expected}"
    return True, ""


def prepare_database(db: sqlite3.Connection) -> None:
    db.executescript(SOURCE_SCHEMA)
    existing = {row[1] for row in db.execute("PRAGMA table_info(lines)")}
    for column in EXTRA_COLUMNS:
        if column not in existing:
            db.execute(f"ALTER TABLE lines ADD COLUMN {column} TEXT")
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
        db.executemany(f"""UPDATE lines SET {assignments}
            WHERE z=? AND charge=? AND row_index=?""", [
            tuple((source.get(name) or "").strip() or None
                  for name in EXTRA_COLUMNS.values())
            + (z, charge, index)
            for index, source in enumerate(rows)])
        db.execute("""INSERT OR REPLACE INTO nist_line_detail_sources
            VALUES (?,?,?,?,?,?,'imported',NULL)""",
            (z, charge, url, digest, len(rows), len(rows)))
        db.commit()
        return True, f"{len(rows)} rows"
    except Exception as exc:
        db.execute("""INSERT OR REPLACE INTO nist_line_detail_sources
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
    parser.add_argument("--limit", type=int, help="limit spectra in this batch")
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
    species = db.execute("""SELECT z,charge FROM species WHERE line_rows>0
        AND z BETWEEN ? AND ? ORDER BY z,charge""",
        (args.min_z, args.max_z)).fetchall()
    if args.limit is not None:
        species = species[:args.limit]
    previous = {(z, q): status for z, q, status in db.execute(
        "SELECT z,charge,status FROM nist_line_detail_sources")}
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
    imported = db.execute("""SELECT COUNT(*) FROM nist_line_detail_sources
        WHERE status='imported'""").fetchone()[0]
    mismatched = db.execute("""SELECT COUNT(*) FROM nist_line_detail_sources
        WHERE status='mismatch'""").fetchone()[0]
    total_spectra = db.execute(
        "SELECT COUNT(*) FROM species WHERE line_rows>0").fetchone()[0]
    coverage_path = out / "coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())
        coverage.update({
            "line_details_imported_ions": imported,
            "line_details_mismatched_ions": mismatched,
            "line_details_coverage_complete": (
                imported == total_spectra and mismatched == 0),
        })
        part = coverage_path.with_suffix(".json.part")
        part.write_text(json.dumps(coverage, indent=2) + "\n")
        part.replace(coverage_path)
    db.close()
    return 1 if failed or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
