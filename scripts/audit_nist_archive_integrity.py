#!/usr/bin/env python3
"""Verify retained NIST source bytes and per-ion archive counts offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/nist_spectra"


def audit(out: Path) -> dict:
    checked = Counter()
    problems = []

    def path_label(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)

    def verify_file(group: str, key: str, path: Path, expected: str | None,
                    expected_bytes: int | None = None) -> None:
        if not path.is_file():
            problems.append({"group": group, "key": key, "issue": "missing_file",
                             "path": path_label(path)})
            return
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
                size += len(chunk)
        checked[group] += 1
        if not expected or digest.hexdigest() != expected:
            problems.append({"group": group, "key": key, "issue": "sha256_mismatch",
                             "path": path_label(path)})
        if expected_bytes is not None and size != expected_bytes:
            problems.append({"group": group, "key": key,
                             "issue": "byte_count_mismatch", "expected": expected_bytes,
                             "actual": size})

    with sqlite3.connect(out / "spectra.sqlite") as db:
        for kind, z, charge, byte_count, sha256, declared in db.execute(
                """SELECT kind,z,charge,bytes,sha256,declared_rows
                   FROM source_files"""):
            key = f"{z}:{charge}"
            name = f"asd.{z:03d}{charge:03d}"
            raw_path = out / "raw" / kind / name
            if not raw_path.is_file():
                # The original 22-ion benchmark panel lives in the tracked
                # benchmark tree; the later full-table download lives in out.
                raw_path = ROOT / "benchmarks/nist_asd" / kind / name
            verify_file(f"base_{kind}", key,
                        raw_path,
                        sha256, byte_count)
            actual = db.execute(f"SELECT COUNT(*) FROM {kind} WHERE z=? AND charge=?",
                                (z, charge)).fetchone()[0]
            expected = declared
            if kind == "levels":
                expected += db.execute("""SELECT COUNT(*)
                    FROM nist_line_only_level_sources WHERE z=? AND charge=?""",
                    (z, charge)).fetchone()[0]
            if actual != expected:
                problems.append({"group": f"base_{kind}", "key": key,
                                 "issue": "archive_row_count_mismatch",
                                 "expected": expected, "actual": actual})
            checked[f"count_{kind}"] += 1

        detail_groups = (
            ("nist_line_detail_sources", "official_line_details", "line_details"),
            ("nist_level_detail_sources", "official_level_details", "level_details"),
            ("nist_weighted_a_sources", "official_weighted_a", "weighted_a"),
        )
        for table, folder, group in detail_groups:
            for z, charge, sha256, source_rows, matched_rows, status in db.execute(
                    f"SELECT z,charge,sha256,source_rows,matched_rows,status FROM {table}"):
                key = f"{z}:{charge}"
                verify_file(group, key,
                            out / folder / f"asd.{z:03d}{charge:03d}.tsv",
                            sha256)
                if status != "imported" or source_rows != matched_rows:
                    problems.append({"group": group, "key": key,
                                     "issue": "import_not_fully_matched",
                                     "source_rows": source_rows,
                                     "matched_rows": matched_rows,
                                     "status": status})

        for z, charge, sha256, declared, status in db.execute(
                """SELECT z,charge,sha256,declared_rows,status
                   FROM nist_term_order_level_sources"""):
            key = f"{z}:{charge}"
            verify_file("term_order_levels", key,
                        out / "official_level_term_order" /
                        f"asd.{z:03d}{charge:03d}.tsv", sha256)
            actual = db.execute("SELECT COUNT(*) FROM levels WHERE z=? AND charge=?",
                                (z, charge)).fetchone()[0]
            if status != "imported" or actual != declared:
                problems.append({"group": "term_order_levels", "key": key,
                                 "issue": "reconciliation_mismatch",
                                 "expected": declared, "actual": actual,
                                 "status": status})

        for kind, spectrum, sha256, holdings, parsed, status in db.execute(
                """SELECT kind,spectrum,sha256,holdings_rows,parsed_rows,status
                   FROM nist_isotope_sources"""):
            verify_file(f"isotope_{kind}", spectrum,
                        out / "official_isotope_records" / kind /
                        f"{spectrum.replace(' ', '_')}.tsv", sha256)
            actual = db.execute("""SELECT COUNT(*) FROM nist_isotope_records
                WHERE kind=? AND spectrum=?""", (kind, spectrum)).fetchone()[0]
            if status != "imported" or holdings != parsed or parsed != actual:
                problems.append({"group": f"isotope_{kind}", "key": spectrum,
                                 "issue": "isotope_row_count_mismatch",
                                 "holdings": holdings, "parsed": parsed,
                                 "actual": actual, "status": status})

        for kind, listed_column, detail_tables in (
                ("lines", "line_listed",
                 ("nist_line_detail_sources", "nist_weighted_a_sources")),
                ("levels", "level_listed", ("nist_level_detail_sources",))):
            listed = db.execute(f"SELECT SUM({listed_column}) FROM species").fetchone()[0]
            archived = db.execute("SELECT COUNT(*) FROM source_files WHERE kind=?",
                                  (kind,)).fetchone()[0]
            if listed != archived:
                problems.append({"group": f"base_{kind}", "issue": "catalog_gap",
                                 "listed": listed, "archived": archived})
            for detail_table in detail_tables:
                detailed = db.execute(f"SELECT COUNT(*) FROM {detail_table}").fetchone()[0]
                if detailed != archived:
                    problems.append({"group": detail_table,
                                     "issue": "supplemental_source_gap",
                                     "base_spectra": archived,
                                     "supplemental_spectra": detailed})

            holdings_file = out / f"official_isotope_{kind}_holdings.json"
            if holdings_file.is_file():
                manifest = json.loads(holdings_file.read_text())
                source_count, source_rows = db.execute("""SELECT COUNT(*),
                    COALESCE(SUM(parsed_rows),0) FROM nist_isotope_sources
                    WHERE kind=?""", (kind,)).fetchone()
                if (source_count != manifest["isotope_spectra"]
                        or source_rows != manifest["isotope_count"]):
                    problems.append({"group": f"isotope_{kind}",
                                     "issue": "holdings_manifest_gap",
                                     "expected_spectra": manifest["isotope_spectra"],
                                     "source_spectra": source_count,
                                     "expected_rows": manifest["isotope_count"],
                                     "source_rows": source_rows})

        totals = {
            "standard_lines": db.execute("SELECT COUNT(*) FROM lines").fetchone()[0],
            "standard_levels": db.execute("SELECT COUNT(*) FROM levels").fetchone()[0],
            "line_only_level_ids": db.execute(
                "SELECT COUNT(*) FROM nist_line_only_level_sources").fetchone()[0],
            "isotope_lines": db.execute("""SELECT COUNT(*) FROM nist_isotope_records
                WHERE kind='lines'""").fetchone()[0],
            "isotope_levels": db.execute("""SELECT COUNT(*) FROM nist_isotope_records
                WHERE kind='levels'""").fetchone()[0],
        }
        historical = db.execute(
            "SELECT COUNT(*) FROM source_count_discrepancies").fetchone()[0]
        reconciled = db.execute("""SELECT COUNT(*)
            FROM nist_term_order_level_sources WHERE status='imported'""").fetchone()[0]

    return {"audited_utc": datetime.now(timezone.utc).isoformat(),
            "source_files_hashed": dict(sorted(checked.items())),
            "archive_rows": totals,
            "historical_energy_order_count_discrepancies": historical,
            "term_order_reconciled_spectra": reconciled,
            "issues": problems,
            "source_integrity_verified": not problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.out)
    report = args.report or args.out / "nist_archive_integrity.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "source_files_hashed", "archive_rows", "source_integrity_verified")},
        indent=2))
    return 0 if result["source_integrity_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
