#!/usr/bin/env python
"""Import NIST ASD's official ground-state and ionization-energy table.

NIST's bulk ionization form currently covers H through Ds (Z=110). The
unmodified response is saved, and missing values remain NULL in SQLite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output/nist_spectra"
PARAMETERS = {
    "spectra": "H-Og", "units": "1", "format": "1",
    "submit": "Retrieve Data", "at_num_out": "on", "sp_name_out": "on",
    "ion_charge_out": "on", "el_name_out": "on", "seq_out": "on",
    "shells_out": "on", "conf_out": "on", "level_out": "on",
    "ion_conf_out": "on", "e_out": "0", "unc_out": "on", "biblio": "on",
}
URL = "https://physics.nist.gov/cgi-bin/ASD/ie.pl?" + urllib.parse.urlencode(PARAMETERS)
SCHEMA = """
CREATE TABLE IF NOT EXISTS nist_ionization (
 z INTEGER NOT NULL, charge INTEGER NOT NULL, spectrum TEXT NOT NULL,
 element_name TEXT, isoelectronic_sequence TEXT,
 ground_shells TEXT, ground_configuration TEXT, ground_level TEXT,
 ionized_level TEXT, ionization_eV REAL, ionization_raw TEXT,
 uncertainty_eV REAL, uncertainty_raw TEXT, reference TEXT,
 source_row INTEGER NOT NULL,
 ionization_kind TEXT, ionization_cm1 REAL, ionization_cm1_raw TEXT,
 uncertainty_cm1 REAL, binding_cm1 REAL, binding_raw TEXT,
 binding_uncertainty_cm1 REAL, binding_kind TEXT, binding_reference TEXT,
 binding_source_row INTEGER,
 PRIMARY KEY(z,charge));
"""
SUPPLEMENT_COLUMNS = {
    "ionization_kind": "TEXT", "ionization_cm1": "REAL",
    "ionization_cm1_raw": "TEXT", "uncertainty_cm1": "REAL",
    "binding_cm1": "REAL", "binding_raw": "TEXT",
    "binding_uncertainty_cm1": "REAL", "binding_kind": "TEXT",
    "binding_reference": "TEXT", "binding_source_row": "INTEGER",
}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, value: str) -> None:
        self.parts.append(value)


def _number(value: str) -> float | None:
    clean = value.strip().strip("()[]").strip()
    return float(clean) if clean else None


def energy_kind(raw: str | None) -> str:
    value = (raw or "").strip()
    return ("missing" if not value else "semiempirical" if value.startswith("[")
            else "theoretical" if value.startswith("(") else "evaluated")


def parse(content: bytes, *, min_rows: int = 6000,
          energy_header: str = "Ionization Energy") -> list[tuple]:
    text = content.decode("utf-8", errors="replace")
    match = re.search(r"<pre>(.*?)</pre>", text, re.S | re.I)
    if not match or energy_header not in match.group(1):
        raise ValueError(f"NIST response has no {energy_header} data table")
    result = []
    seen = set()
    for source_row, line in enumerate(match.group(1).splitlines(), start=1):
        if not re.match(r"^\s*\d+\s*\|", line):
            continue
        parser = _Text()
        parser.feed(line)
        fields = [field.strip() for field in "".join(parser.parts).split("|")]
        if len(fields) != 12:
            raise ValueError(f"ionization source row {source_row} has {len(fields)} fields")
        z, spectrum, charge, element, sequence, shells, config, level, ionized, \
            energy, uncertainty, reference = fields
        key = (int(z), int(charge))
        if key in seen or not (1 <= key[0] <= 118 and 0 <= key[1] < key[0]):
            raise ValueError(f"duplicate or invalid NIST ionization key: {key}")
        seen.add(key)
        result.append((key[0], key[1], spectrum, element, sequence, shells,
                       config, level, ionized, _number(energy), energy,
                       _number(uncertainty), uncertainty, reference, source_row))
    if len(result) < min_rows:
        raise ValueError(f"NIST returned only {len(result)} ionization records")
    return result


def _supplement_url(*, binding: bool) -> str:
    params = dict(PARAMETERS, units="0", e_out="1" if binding else "0")
    return "https://physics.nist.gov/cgi-bin/ASD/ie.pl?" + urllib.parse.urlencode(params)


def _read_supplement(out: Path, name: str, *, offline: bool,
                     binding: bool) -> tuple[bytes, str] | None:
    path = out / name
    url = _supplement_url(binding=binding)
    if not offline:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
        path.write_bytes(body)
    elif path.exists():
        body = path.read_bytes()
    else:
        return None
    return body, url


def import_to_database(out: Path, *, offline: bool = False) -> dict:
    out = out.resolve()
    raw_path = out / "official_ionization_energies.html"
    if not offline:
        request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            body = response.read()
        raw_path.write_bytes(body)
    elif raw_path.exists():
        body = raw_path.read_bytes()
    else:
        raise FileNotFoundError(f"missing {raw_path}; run without --offline first")
    records = parse(body)
    database = out / "spectra.sqlite"
    with sqlite3.connect(database) as db:
        known = {(z, q) for z, q in db.execute("SELECT z,charge FROM species")}
        unexpected = {(row[0], row[1]) for row in records} - known
        if unexpected:
            raise ValueError(f"ionization records absent from species inventory: {unexpected}")
        db.executescript(SCHEMA)
        existing = {row[1] for row in db.execute("PRAGMA table_info(nist_ionization)")}
        for column, sql_type in SUPPLEMENT_COLUMNS.items():
            if column not in existing:
                db.execute(f"ALTER TABLE nist_ionization ADD COLUMN {column} {sql_type}")
        db.execute("DELETE FROM nist_ionization")
        db.executemany("""INSERT INTO nist_ionization
            (z,charge,spectrum,element_name,isoelectronic_sequence,
             ground_shells,ground_configuration,ground_level,ionized_level,
             ionization_eV,ionization_raw,uncertainty_eV,uncertainty_raw,
             reference,source_row) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                       records)
        db.executemany("""UPDATE nist_ionization SET ionization_kind=?
            WHERE z=? AND charge=?""",
            ((energy_kind(row[10]), row[0], row[1]) for row in records))
        cm_source = _read_supplement(
            out, "official_ionization_cm1.html", offline=offline,
            binding=False)
        cm_records = []
        if cm_source is not None:
            cm_body, cm_url = cm_source
            cm_records = parse(cm_body)
            if {(r[0], r[1]) for r in cm_records} != {(r[0], r[1]) for r in records}:
                raise ValueError("NIST cm-1 ionization keys differ from eV keys")
            db.executemany("""UPDATE nist_ionization SET ionization_cm1=?,
                ionization_cm1_raw=?,uncertainty_cm1=? WHERE z=? AND charge=?""",
                ((r[9], r[10], r[11], r[0], r[1]) for r in cm_records))
            db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                       ("official_ionization_cm1_url", cm_url))
            db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                       ("official_ionization_cm1_sha256",
                        hashlib.sha256(cm_body).hexdigest()))
        binding_source = _read_supplement(
            out, "official_binding_cm1.html", offline=offline, binding=True)
        binding_records = []
        if binding_source is not None:
            binding_body, binding_url = binding_source
            binding_records = parse(binding_body, min_rows=5000,
                                    energy_header="Binding Energy")
            if {(r[0], r[1]) for r in binding_records} - {(r[0], r[1]) for r in records}:
                raise ValueError("NIST binding keys absent from ionization table")
            db.executemany("""UPDATE nist_ionization SET binding_cm1=?,
                binding_raw=?,binding_uncertainty_cm1=?,binding_kind=?,
                binding_reference=?,binding_source_row=? WHERE z=? AND charge=?""",
                ((r[9], r[10], r[11], energy_kind(r[10]), r[13], r[14],
                  r[0], r[1]) for r in binding_records))
            db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                       ("official_binding_cm1_url", binding_url))
            db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                       ("official_binding_cm1_sha256",
                        hashlib.sha256(binding_body).hexdigest()))
        db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                   ("official_ionization_url", URL))
        db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                   ("official_ionization_sha256", hashlib.sha256(body).hexdigest()))
    report = {
        "official_ionization_url": URL,
        "official_ionization_checked_utc": datetime.now(timezone.utc).isoformat(),
        "official_ionization_sha256": hashlib.sha256(body).hexdigest(),
        "official_ionization_rows": len(records),
        "official_ionization_with_energy": sum(row[9] is not None for row in records),
        "official_ionization_max_z": max(row[0] for row in records),
        "official_ionization_cm1_rows": len(cm_records),
        "official_binding_cm1_rows": len(binding_records),
    }
    coverage_path = out / "coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())
        coverage.update(report)
        coverage_path.write_text(json.dumps(coverage, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(import_to_database(args.out, offline=args.offline), indent=2))


if __name__ == "__main__":
    main()
