#!/usr/bin/env python3
"""Build an auditable NIST ASD v5.12 line catalog for every available ion.

The Cloudy/Nublado archive contains the unmodified NIST ASD ASCII/HTML
responses.  Its directory indexes are the inventory; an absent file is not
interpreted as a spectrum with zero lines.  The existing atomx NIST panel is
reused when possible.  Run with ``--offline`` to rebuild from cached data.

    .venv/bin/python scripts/build_nist_spectra.py
    .venv/bin/python scripts/build_nist_spectra.py --offline

Outputs are under output/nist_spectra/: source files, SQLite catalog, coverage
JSON/CSV, and an offline interactive HTML viewer.  A nonzero exit status means
the indexed archive was not completely downloaded and parsed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "src"))
from atomx.reference_data import nist as nist_asd  # noqa: E402

from atomx.constants import HARTREE_CM1, HARTREE_EV, atomic_numbers, element_symbol  # noqa: E402

BASE = "https://data.nublado.org/nist/asd_v5.12"
OFFICIAL_HOLDINGS_URL = "https://physics.nist.gov/cgi-bin/ASD/lines_pt.pl"
DEFAULT_OUT = ROOT / "output" / "nist_spectra"
PANEL = ROOT / "benchmarks" / "nist_asd"
NAME_RE = re.compile(r"^asd\.(\d{3})(\d{3})$")
LINE_TITLE_RE = re.compile(r"<u>\s*([A-Za-z]{1,2}\s+[IVXLCDM]+)\s*</u>")
LEVEL_TITLE_RE = re.compile(r"<sup>\s*</sup>\s*([A-Za-z]{1,2}\s+[IVXLCDM]+)\s*&nbsp;")
SEQ_RE = re.compile(r"<i>Z</i>\s*=\s*(\d+),\s*([A-Za-z]{1,2})\s+isoelectronic sequence")
DECLARED_RE = {
    "lines": re.compile(r"([\d,]+) Lines? of Data Found"),
    "levels": re.compile(r"([\d,]+) Levels? Found"),
}
HOLDING_RE = re.compile(
    r'title="(?P<symbol>[A-Za-z]{1,2})\s+(?P<stage>[IVXLCDM]+):\s*'
    r'(?P<count>[\d,]+) lines"')
HEADERS = {
    "lines": re.compile(r"Observed[^\n]*\|[^\n]*Aki"),
    "levels": re.compile(r"Configuration\s*\|\s*Term\s*\|\s*J\s*\|\s*Level \(cm-1\)"),
}


class IndexParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.names: set[str] = set()

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if NAME_RE.fullmatch(href):
                self.names.add(href)


def roman(n: int) -> str:
    out = ""
    for value, symbol in ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                          (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        count, n = divmod(n, value)
        out += symbol * count
    return out


def ion_label(z: int, charge: int) -> str:
    return f"{element_symbol(z)} {roman(charge + 1)}"


def _request(url: str, retries: int = 3, *, user_agent: str = "atomx-nist-spectra/1.0") -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {url}")
                return response.read()
        except (OSError, urllib.error.URLError, RuntimeError):
            if attempt == retries - 1:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def roman_to_int(text: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, char in enumerate(text):
        value = values[char]
        total += -value if i + 1 < len(text) and value < values[text[i + 1]] else value
    if roman(total) != text:
        raise ValueError(f"invalid spectroscopic Roman numeral {text!r}")
    return total


def parse_official_holdings(
        content: str, *, min_cells: int = 6105
) -> tuple[dict[tuple[int, int], int], int, int]:
    """Read NIST's per-ion line holdings and its separate headline totals."""
    cells: dict[tuple[int, int], int] = {}
    for match in HOLDING_RE.finditer(content):
        symbol = match.group("symbol")
        z = atomic_numbers.get(symbol)
        if z is None or not 1 <= z <= 118:
            raise ValueError(f"unknown element in NIST holdings: {symbol}")
        charge = roman_to_int(match.group("stage")) - 1
        if not 0 <= charge < z:
            raise ValueError(f"invalid charge in NIST holdings: {symbol} {charge}")
        key = (z, charge)
        if key in cells:
            raise ValueError(f"duplicate NIST holdings cell for {symbol} {charge}")
        cells[key] = int(match.group("count").replace(",", ""))
    if len(cells) < min_cells:
        raise ValueError(f"NIST holdings page has only {len(cells)} ion cells")
    lines = re.search(r"number of lines:\s*([\d,]+)", content)
    aki = re.search(r"number of lines with transition probabilities:\s*([\d,]+)", content)
    if not lines or not aki:
        raise ValueError("NIST holdings page lacks headline totals")
    return (cells, int(lines.group(1).replace(",", "")),
            int(aki.group(1).replace(",", "")))


def verify_official_holdings(out: Path, summary: dict, *, offline: bool) -> dict:
    """Audit every mirror ion against the live NIST per-ion holdings grid."""
    target = out / "official_lines_holdings.html"
    if not offline:
        body = _request(
            OFFICIAL_HOLDINGS_URL,
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36")
        target.write_bytes(body)
    elif not target.exists():
        raise FileNotFoundError(f"missing {target}; run without --offline first")
    body = target.read_bytes()
    cells, headline_lines, headline_aki = parse_official_holdings(
        body.decode("utf-8", errors="replace"))
    with sqlite3.connect(out / "spectra.sqlite") as db:
        local = {(z, q): count for z, q, count in db.execute(
            "SELECT z,charge,line_rows FROM species")}
    mismatches = [
        {"label": ion_label(z, q), "z": z, "charge": q,
         "nist_lines": n, "catalog_lines": local[(z, q)]}
        for (z, q), n in sorted(cells.items()) if local[(z, q)] != n
    ]
    missing_cells = [
        ion_label(z, q) for (z, q), count in sorted(local.items())
        if count > 0 and (z, q) not in cells
    ]
    with (out / "official_holdings_comparison.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("z", "charge", "label", "nist_lines", "catalog_lines", "difference"))
        for (z, q), count in sorted(cells.items()):
            writer.writerow((z, q, ion_label(z, q), count, local[(z, q)],
                             count - local[(z, q)]))
    summary.update({
        "official_holdings_url": OFFICIAL_HOLDINGS_URL,
        "official_holdings_sha256": hashlib.sha256(body).hexdigest(),
        "official_holdings_checked_utc": datetime.now(timezone.utc).isoformat(),
        "official_ion_cells": len(cells),
        "official_ion_cell_sum": sum(cells.values()),
        "official_ion_mismatches": mismatches,
        "official_missing_catalog_cells": missing_cells,
        "official_headline_lines": headline_lines,
        "official_headline_aki_rows": headline_aki,
        "official_headline_minus_cell_sum": headline_lines - sum(cells.values()),
        "official_headline_minus_catalog_aki_rows": headline_aki - summary["aki_rows"],
    })
    summary["line_holdings_verified"] = (
        summary["line_archive_complete"] and not mismatches
        and not missing_cells and summary["official_ion_cell_sum"] == summary["lines"])
    (out / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def inventory(out: Path, offline: bool) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for kind in ("lines", "levels"):
        target = out / f"{kind}_index.html"
        if not offline:
            body = _request(f"{BASE}/{kind}/")
            target.write_bytes(body)
        elif not target.exists():
            raise FileNotFoundError(f"missing {target}; run without --offline first")
        parser = IndexParser()
        parser.feed(target.read_text(errors="replace"))
        if not parser.names:
            raise ValueError(f"no ASD entries found in {target}")
        found[kind] = parser.names
    return found


def _valid(body: bytes, kind: str, z: int, charge: int) -> str:
    if len(body) < 4096:
        raise ValueError("short NIST response")
    content = body.decode("utf-8", errors="replace")
    if "</html>" not in content or not HEADERS[kind].search(content):
        raise ValueError("missing NIST table header or closing HTML tag")
    match = (LINE_TITLE_RE if kind == "lines" else LEVEL_TITLE_RE).search(content)
    expected = ion_label(z, charge)
    if not match or " ".join(match.group(1).split()).upper() != expected.upper():
        raise ValueError(f"ion title mismatch: expected {expected}")
    seq = SEQ_RE.search(content)
    if not seq or int(seq.group(1)) != z or seq.group(2) != element_symbol(z - charge):
        raise ValueError(f"isoelectronic sequence mismatch for {expected}")
    return content


def _one_file(out: Path, kind: str, name: str, offline: bool) -> dict:
    z, charge = map(int, NAME_RE.fullmatch(name).groups())
    target = out / "raw" / kind / name
    panel_path = PANEL / kind / name
    url = f"{BASE}/{kind}/{name}"
    try:
        if target.exists():
            body = target.read_bytes()
            origin = "cache"
        elif panel_path.exists():
            body = panel_path.read_bytes()
            origin = "committed panel"
        elif offline:
            raise FileNotFoundError("not cached")
        else:
            body = _request(url)
            origin = "download"
        content = _valid(body, kind, z, charge)
        declared = DECLARED_RE[kind].search(content)
        if origin == "download":
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            part.write_bytes(body)
            part.replace(target)
        return {"kind": kind, "name": name, "z": z, "charge": charge,
                "url": url, "path": str(panel_path if origin == "committed panel" else target),
                "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                "declared_rows": int(declared.group(1).replace(",", ""))
                if declared else None,
                "origin": origin, "content": content}
    except Exception as exc:
        return {"kind": kind, "name": name, "z": z, "charge": charge,
                "url": url, "error": f"{type(exc).__name__}: {exc}"}


SCHEMA = """
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE source_files (
 kind TEXT NOT NULL, z INTEGER NOT NULL, charge INTEGER NOT NULL,
 url TEXT NOT NULL, bytes INTEGER, sha256 TEXT, declared_rows INTEGER,
 error TEXT,
 PRIMARY KEY (kind, z, charge));
CREATE TABLE species (
 z INTEGER NOT NULL, charge INTEGER NOT NULL, symbol TEXT NOT NULL,
 label TEXT NOT NULL, line_listed INTEGER NOT NULL, level_listed INTEGER NOT NULL,
 line_status TEXT NOT NULL, level_status TEXT NOT NULL,
 line_rows INTEGER NOT NULL, usable_wavelengths INTEGER NOT NULL,
 observed_wavelengths INTEGER NOT NULL, ritz_only_wavelengths INTEGER NOT NULL,
 aki_rows INTEGER NOT NULL, level_rows INTEGER NOT NULL,
 line_parse_errors INTEGER NOT NULL, level_parse_errors INTEGER NOT NULL,
 PRIMARY KEY (z, charge));
CREATE TABLE lines (
 z INTEGER NOT NULL, charge INTEGER NOT NULL, row_index INTEGER NOT NULL,
 observed_nm REAL, observed_unc_nm REAL, ritz_nm REAL, ritz_unc_nm REAL,
 wavelength_nm REAL, wavelength_source TEXT, medium TEXT,
 aki_s1 REAL, aki_accuracy TEXT, lower_energy_cm1 REAL, upper_energy_cm1 REAL,
 lower_configuration TEXT, lower_term TEXT, lower_j TEXT,
 upper_configuration TEXT, upper_term TEXT, upper_j TEXT,
 transition_type TEXT, line_reference TEXT, tp_reference TEXT, source_row INTEGER,
 PRIMARY KEY(z, charge, row_index));
CREATE INDEX lines_wavelength ON lines(z, charge, wavelength_nm);
CREATE TABLE levels (
 z INTEGER NOT NULL, charge INTEGER NOT NULL, row_index INTEGER NOT NULL,
 configuration TEXT, configuration_nist TEXT, term TEXT, j_text TEXT,
 energy_cm1 REAL, uncertainty_cm1 REAL, reference TEXT, flags TEXT,
 bracketed INTEGER NOT NULL, source_row INTEGER,
 energy_raw TEXT, energy_kind TEXT, lande_g_raw TEXT, leading_percentages TEXT,
 PRIMARY KEY(z, charge, row_index));
CREATE INDEX levels_energy ON levels(z, charge, energy_cm1);
CREATE TABLE parse_errors(kind TEXT, z INTEGER, charge INTEGER, detail TEXT);
CREATE TABLE source_count_discrepancies(
 kind TEXT, z INTEGER, charge INTEGER, declared_rows INTEGER,
 parsed_rows INTEGER, PRIMARY KEY(kind,z,charge));
CREATE TABLE ion_summary (
 z INTEGER NOT NULL, charge INTEGER NOT NULL,
 ground_row_index INTEGER, ground_configuration TEXT, ground_term TEXT,
 ground_j TEXT, ground_energy_cm1 REAL, ground_uncertainty_cm1 REAL,
 first_limit_row_index INTEGER, limit_count INTEGER NOT NULL,
 ionization_cm1 REAL, ionization_eV REAL, ionization_uncertainty_cm1 REAL,
 ionization_source TEXT,
 PRIMARY KEY(z, charge));
"""


def _build_database(out: Path, index: dict[str, set[str]], files: list[dict]) -> tuple[dict, list[dict]]:
    by_key = {(f["kind"], f["name"]): f for f in files}
    temp = out / "spectra.sqlite.part"
    if temp.exists():
        temp.unlink()
    db = sqlite3.connect(temp)
    db.executescript(SCHEMA)
    now = datetime.now(timezone.utc).isoformat()
    for key, value in {"source": "NIST ASD v5.12 via Cloudy/Nublado mirror",
                       "mirror": BASE, "built_utc": now,
                       "wavelength_policy": "observed if present, otherwise Ritz; native NIST air/vacuum medium",
                       "intensity_policy": "Aki is a transition probability, not an emitted intensity"}.items():
        db.execute("INSERT INTO metadata VALUES (?,?)", (key, value))
    for f in files:
        db.execute("INSERT INTO source_files VALUES (?,?,?,?,?,?,?,?)",
                   (f["kind"], f["z"], f["charge"], f["url"],
                    f.get("bytes"), f.get("sha256"), f.get("declared_rows"),
                    f.get("error")))
    species_rows: list[dict] = []
    for z in range(1, 119):
        for charge in range(z):
            name = f"asd.{z:03d}{charge:03d}"
            record = {"z": z, "charge": charge, "symbol": element_symbol(z),
                      "label": ion_label(z, charge)}
            counts = {"line_rows": 0, "usable_wavelengths": 0,
                      "observed_wavelengths": 0, "ritz_only_wavelengths": 0,
                      "aki_rows": 0, "level_rows": 0,
                      "line_parse_errors": 0, "level_parse_errors": 0}
            statuses = {}
            for kind in ("lines", "levels"):
                listed = name in index[kind]
                record[f"{kind[:-1] if kind == 'lines' else 'level'}_listed"] = int(listed)
                if not listed:
                    statuses[kind] = "not_in_archive"
                    continue
                source = by_key[(kind, name)]
                if "error" in source:
                    statuses[kind] = "download_error"
                    continue
                try:
                    if kind == "lines":
                        rows, problems = nist_asd.parse_lines_text(
                            source["content"], ion=record["label"], z=z,
                            charge=charge, source=name)
                        counts["line_rows"] = len(rows)
                        for line in rows:
                            wavelength = line.observed_nm or line.ritz_nm
                            mode = ("observed" if line.observed_nm else
                                    "Ritz" if line.ritz_nm else None)
                            if wavelength is not None and wavelength > 0:
                                counts["usable_wavelengths"] += 1
                                counts["observed_wavelengths"] += bool(line.observed_nm)
                                counts["ritz_only_wavelengths"] += bool(
                                    not line.observed_nm and line.ritz_nm)
                            counts["aki_rows"] += line.aki is not None
                            db.execute("INSERT INTO lines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                       (z, charge, line.index, line.observed_nm,
                                        line.observed_unc_nm, line.ritz_nm,
                                        line.ritz_unc_nm, wavelength, mode, line.medium,
                                        line.aki, line.acc, line.ei_cm1, line.ek_cm1,
                                        line.lower_configuration, line.lower_term,
                                        line.lower_j_text, line.upper_configuration,
                                        line.upper_term, line.upper_j_text,
                                        line.transition_type, line.line_reference,
                                        line.tp_reference, line.row))
                    else:
                        rows, problems = nist_asd.parse_levels_text(
                            source["content"], ion=record["label"], z=z,
                            charge=charge, source=name)
                        counts["level_rows"] = len(rows)
                        db.executemany(
                            "INSERT INTO levels VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            ((z, charge, level.index, level.configuration,
                              level.configuration_nist, level.term, level.j_text,
                              level.energy_cm1, level.uncertainty_cm1,
                              level.reference, json.dumps(level.flags),
                              int(level.bracketed), level.row, level.energy_raw,
                              level.energy_kind, level.lande_g_raw,
                              level.leading_percentages) for level in rows))
                    declared = DECLARED_RE[kind].search(source["content"])
                    if declared is None:
                        problems.append(f"{name}: NIST row count missing")
                    elif int(declared.group(1).replace(",", "")) != len(rows):
                        db.execute("""INSERT INTO source_count_discrepancies
                            VALUES (?,?,?,?,?)""",
                            (kind, z, charge,
                             int(declared.group(1).replace(",", "")), len(rows)))
                    for problem in problems:
                        db.execute("INSERT INTO parse_errors VALUES (?,?,?,?)",
                                   (kind, z, charge, problem))
                    counts[f"{kind[:-1] if kind == 'lines' else 'level'}_parse_errors"] = len(problems)
                    statuses[kind] = "parsed" if rows and not problems else "parse_error"
                except Exception as exc:
                    statuses[kind] = "parse_error"
                    counts[f"{kind[:-1] if kind == 'lines' else 'level'}_parse_errors"] += 1
                    db.execute("INSERT INTO parse_errors VALUES (?,?,?,?)",
                               (kind, z, charge, f"{type(exc).__name__}: {exc}"))
            record.update(counts)
            record["line_status"] = statuses["lines"]
            record["level_status"] = statuses["levels"]
            db.execute("INSERT INTO species VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (z, charge, record["symbol"], record["label"],
                        record["line_listed"], record["level_listed"],
                        record["line_status"], record["level_status"],
                        *(record[k] for k in counts)))
            species_rows.append(record)
    ionization_count = 0
    ground_count = 0
    for item in species_rows:
        z, charge = item["z"], item["charge"]
        ground = db.execute("""SELECT row_index,configuration_nist,term,j_text,
                              energy_cm1,uncertainty_cm1 FROM levels
                              WHERE z=? AND charge=? AND energy_cm1 IS NOT NULL
                              AND term NOT LIKE '%Limit%'
                              ORDER BY energy_cm1,row_index LIMIT 1""",
                            (z, charge)).fetchone()
        limits = db.execute("""SELECT row_index,energy_cm1,uncertainty_cm1
                             FROM levels WHERE z=? AND charge=?
                             AND term LIKE '%Limit%' AND energy_cm1 IS NOT NULL
                             ORDER BY energy_cm1,row_index""", (z, charge)).fetchall()
        first = limits[0] if limits else None
        ion_cm1 = first[1] - ground[4] if first and ground else None
        ion_ev = ion_cm1 * HARTREE_EV / HARTREE_CM1 if ion_cm1 is not None else None
        ground_count += ground is not None
        ionization_count += ion_cm1 is not None
        db.execute("INSERT INTO ion_summary VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (z, charge, ground[0] if ground else None,
                    ground[1] if ground else None, ground[2] if ground else None,
                    ground[3] if ground else None, ground[4] if ground else None,
                    ground[5] if ground else None, first[0] if first else None,
                    len(limits), ion_cm1, ion_ev, first[2] if first else None,
                    "NIST level Limit row" if ion_cm1 is not None else None))
    db.commit()
    count_discrepancies = db.execute(
        "SELECT COUNT(*) FROM source_count_discrepancies").fetchone()[0]
    db.close()
    temp.replace(out / "spectra.sqlite")
    summary = {
        "built_utc": now, "source": "NIST ASD v5.12", "mirror": BASE,
        "candidate_ions": len(species_rows),
        "line_files_listed": len(index["lines"]),
        "level_files_listed": len(index["levels"]),
        "line_files_parsed": sum(r["line_status"] == "parsed" for r in species_rows),
        "level_files_parsed": sum(r["level_status"] == "parsed" for r in species_rows),
        "declared_line_rows": sum(f.get("declared_rows") or 0 for f in files
                                  if f["kind"] == "lines"),
        "declared_level_rows": sum(f.get("declared_rows") or 0 for f in files
                                   if f["kind"] == "levels"),
        "lines": sum(r["line_rows"] for r in species_rows),
        "levels": sum(r["level_rows"] for r in species_rows),
        "ground_states": ground_count,
        "ionization_thresholds": ionization_count,
        "usable_wavelengths": sum(r["usable_wavelengths"] for r in species_rows),
        "observed_wavelengths": sum(r["observed_wavelengths"] for r in species_rows),
        "ritz_only_wavelengths": sum(r["ritz_only_wavelengths"] for r in species_rows),
        "aki_rows": sum(r["aki_rows"] for r in species_rows),
        "parse_errors": sum(r["line_parse_errors"] + r["level_parse_errors"]
                            for r in species_rows),
        "source_count_discrepancies": count_discrepancies,
        "download_errors": [f"{f['kind']}/{f['name']}: {f['error']}"
                            for f in files if "error" in f],
    }
    summary["line_archive_complete"] = (
        summary["line_files_parsed"] == summary["line_files_listed"] and
        sum(r["line_parse_errors"] for r in species_rows) == 0)
    summary["level_archive_complete"] = (
        summary["level_files_parsed"] == summary["level_files_listed"] and
        sum(r["level_parse_errors"] for r in species_rows) == 0)
    summary["complete_for_indexed_archive"] = (
        summary["line_archive_complete"] and summary["level_archive_complete"])
    (out / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out / "coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=species_rows[0].keys())
        writer.writeheader()
        writer.writerows(species_rows)
    with (out / "source_manifest.json").open("w") as handle:
        json.dump([{k: v for k, v in f.items() if k != "content"}
                   for f in files], handle, indent=2)
    return summary, species_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    index = inventory(out, args.offline)
    tasks = [(kind, name) for kind in ("lines", "levels") for name in sorted(index[kind])]
    print(f"NIST ASD index: {len(index['lines'])} line files, "
          f"{len(index['levels'])} level files; {len(tasks)} downloads/checks", flush=True)
    files = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one_file, out, kind, name, args.offline): (kind, name)
                   for kind, name in tasks}
        for future in as_completed(futures):
            files.append(future.result())
            if len(files) % 100 == 0 or len(files) == len(tasks):
                failed = sum("error" in f for f in files)
                print(f"  {len(files)}/{len(tasks)} checked; {failed} failures", flush=True)
    summary, _ = _build_database(out, index, files)
    summary = verify_official_holdings(out, summary, offline=args.offline)
    from import_nist_ionization import import_to_database
    summary.update(import_to_database(out, offline=args.offline))
    if (out / "official_line_details").is_dir():
        from import_nist_line_details import main as import_line_details
        details_exit = import_line_details([
            "--out", str(out), "--offline", "--quiet",
            "--workers", str(min(8, args.workers))])
        with sqlite3.connect(out / "spectra.sqlite") as db:
            summary["line_details_imported_ions"] = db.execute(
                """SELECT COUNT(*) FROM nist_line_detail_sources
                   WHERE status='imported'""").fetchone()[0]
            summary["line_details_mismatched_ions"] = db.execute(
                """SELECT COUNT(*) FROM nist_line_detail_sources
                   WHERE status='mismatch'""").fetchone()[0]
        summary["line_details_coverage_complete"] = (
            details_exit == 0
            and summary["line_details_imported_ions"] == summary["line_files_listed"])
    else:
        summary["line_details_imported_ions"] = 0
        summary["line_details_mismatched_ions"] = 0
        summary["line_details_coverage_complete"] = False
    if (out / "official_level_details").is_dir():
        from import_nist_level_details import main as import_level_details
        details_exit = import_level_details([
            "--out", str(out), "--offline", "--quiet",
            "--workers", str(min(8, args.workers))])
        with sqlite3.connect(out / "spectra.sqlite") as db:
            summary["level_details_imported_ions"] = db.execute(
                """SELECT COUNT(*) FROM nist_level_detail_sources
                   WHERE status='imported'""").fetchone()[0]
            summary["level_details_mismatched_ions"] = db.execute(
                """SELECT COUNT(*) FROM nist_level_detail_sources
                   WHERE status='mismatch'""").fetchone()[0]
        summary["level_details_coverage_complete"] = (
            details_exit == 0
            and summary["level_details_imported_ions"] == summary["level_files_listed"])
    else:
        summary["level_details_imported_ions"] = 0
        summary["level_details_mismatched_ions"] = 0
        summary["level_details_coverage_complete"] = False
    if (out / "official_weighted_a").is_dir():
        from import_nist_weighted_a import main as import_weighted_a
        weighted_exit = import_weighted_a([
            "--out", str(out), "--offline", "--quiet",
            "--workers", str(min(8, args.workers))])
        with sqlite3.connect(out / "spectra.sqlite") as db:
            summary["weighted_a_imported_ions"] = db.execute(
                """SELECT COUNT(*) FROM nist_weighted_a_sources
                   WHERE status='imported'""").fetchone()[0]
            summary["weighted_a_mismatched_ions"] = db.execute(
                """SELECT COUNT(*) FROM nist_weighted_a_sources
                   WHERE status='mismatch'""").fetchone()[0]
        summary["weighted_a_coverage_complete"] = (
            weighted_exit == 0
            and summary["weighted_a_imported_ions"] == summary["line_files_listed"])
    else:
        summary["weighted_a_imported_ions"] = 0
        summary["weighted_a_mismatched_ions"] = 0
        summary["weighted_a_coverage_complete"] = False
    isotope_manifests = [out / f"official_isotope_{kind}_holdings.json"
                         for kind in ("lines", "levels")]
    if all(path.is_file() for path in isotope_manifests):
        from import_nist_isotope_records import main as import_isotope_records
        isotope_exit = import_isotope_records([
            "--output", str(out), "--offline", "--workers", str(min(8, args.workers))])
        with sqlite3.connect(out / "spectra.sqlite") as db:
            for kind, manifest_path in zip(("lines", "levels"), isotope_manifests):
                manifest = json.loads(manifest_path.read_text())
                count, total = db.execute("""SELECT COUNT(*),COALESCE(SUM(parsed_rows),0)
                    FROM nist_isotope_sources WHERE kind=? AND status='imported'""",
                    (kind,)).fetchone()
                summary[f"isotope_{kind}_spectra"] = count
                summary[f"isotope_{kind}_rows"] = total
                summary[f"official_isotope_{kind}_holdings"] = manifest["isotope_count"]
                summary[f"official_{kind}_all_holdings"] = manifest["all_count"]
                summary[f"isotope_{kind}_coverage_complete"] = (
                    isotope_exit == 0 and manifest["elements"] == 118
                    and count == manifest["isotope_spectra"]
                    and total == manifest["isotope_count"])
    diagnostics_audit = out / "nist_diagnostics_audit.json"
    if diagnostics_audit.is_file():
        audit = json.loads(diagnostics_audit.read_text())
        diagnostic_probes = [row for row in audit.get("probes", [])
                             if row.get("diag_out") or row.get("line_out") == "4"]
        summary["plasma_diagnostics_status"] = (
            "source_query_error" if diagnostic_probes and all(
                row.get("error_message") == "Query error."
                for row in diagnostic_probes) else "not_imported")
        summary["plasma_diagnostics_imported"] = bool(
            audit.get("diagnostics_imported", False))
        summary["plasma_diagnostics_audit_utc"] = audit.get("audited_at_utc")
    (out / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "download_errors"},
                     indent=2), flush=True)
    # A cached supplementary export is part of this build. Fail if it cannot
    # be restored completely; NIST's six inconsistent level counts are kept
    # as source discrepancies rather than fabricated rows.
    complete = summary["line_holdings_verified"] and not summary["download_errors"]
    for key, directory in (("line_details_coverage_complete", "official_line_details"),
                           ("level_details_coverage_complete", "official_level_details"),
                           ("weighted_a_coverage_complete", "official_weighted_a")):
        if (out / directory).is_dir():
            complete = complete and summary[key]
    if all(path.is_file() for path in isotope_manifests):
        complete = (complete and summary["isotope_lines_coverage_complete"]
                    and summary["isotope_levels_coverage_complete"])
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
