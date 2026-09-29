#!/usr/bin/env python3
"""Audit NIST's displayed level weight against the angular-momentum identity.

The archive keeps NIST's ``g`` column verbatim.  A simple J has physical
statistical weight 2J+1; disagreements are source values to flag, not silently
rewrite.  Ambiguous or unresolved J labels are excluded from this identity
check.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from fractions import Fraction
from pathlib import Path
from atomx.reference_data.matching import weight_from_j

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "output/nist_spectra"
FIELDS = ("z", "charge", "row_index", "configuration", "term", "j_text",
          "nist_g_raw", "derived_2j_plus_1", "energy_cm1", "level_id",
          "official_source_url")
LINE_FIELDS = ("z", "charge", "row_index", "side", "j_text", "nist_g_raw",
               "derived_2j_plus_1", "official_source_url")


def audit(archive: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    by_ion = Counter()
    mismatches = []
    line_counts = Counter()
    line_by_ion = Counter()
    line_mismatches = []
    with sqlite3.connect(archive) as db:
        for (z, charge, row_index, configuration, term, j_text, g_raw,
             energy_cm1, level_id, url) in db.execute("""
                SELECT l.z,l.charge,l.row_index,l.configuration,l.term,
                       l.j_text,l.statistical_weight_raw,l.energy_cm1,
                       l.level_id,s.url FROM levels l
                LEFT JOIN nist_level_detail_sources s
                  ON s.z=l.z AND s.charge=l.charge
                ORDER BY l.z,l.charge,l.row_index"""):
            expected = weight_from_j(j_text)
            if expected is None or not (g_raw or "").strip():
                counts["not_checkable"] += 1
                continue
            try:
                actual = Fraction(g_raw.strip())
            except (ValueError, ZeroDivisionError):
                counts["not_checkable"] += 1
                continue
            counts["checkable"] += 1
            if actual == expected:
                counts["agree"] += 1
                continue
            counts["disagree"] += 1
            by_ion[f"{z}:{charge}"] += 1
            mismatches.append(dict(
                z=z, charge=charge, row_index=row_index,
                configuration=configuration, term=term, j_text=j_text,
                nist_g_raw=g_raw, derived_2j_plus_1=expected,
                energy_cm1=energy_cm1, level_id=level_id,
                official_source_url=url))
        for z, charge, row_index, lower_j, lower_g, upper_j, upper_g, url in db.execute("""
                SELECT l.z,l.charge,l.row_index,l.lower_j,l.lower_g_raw,
                       l.upper_j,l.upper_g_raw,s.url FROM lines l
                LEFT JOIN nist_line_detail_sources s
                  ON s.z=l.z AND s.charge=l.charge
                ORDER BY l.z,l.charge,l.row_index"""):
            for side, j_text, g_raw in (("lower", lower_j, lower_g),
                                        ("upper", upper_j, upper_g)):
                expected = weight_from_j(j_text)
                if expected is None or not (g_raw or "").strip():
                    line_counts["not_checkable"] += 1
                    continue
                try:
                    actual = Fraction(g_raw.strip())
                except (ValueError, ZeroDivisionError):
                    line_counts["not_checkable"] += 1
                    continue
                line_counts["checkable"] += 1
                if actual == expected:
                    line_counts["agree"] += 1
                    continue
                line_counts["disagree"] += 1
                line_by_ion[f"{z}:{charge}"] += 1
                line_mismatches.append(dict(
                    z=z, charge=charge, row_index=row_index, side=side,
                    j_text=j_text, nist_g_raw=g_raw,
                    derived_2j_plus_1=expected, official_source_url=url))
    path = out / "nist_level_weight_discrepancies.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(mismatches)
    line_path = out / "nist_line_weight_discrepancies.csv"
    with line_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LINE_FIELDS)
        writer.writeheader()
        writer.writerows(line_mismatches)
    summary = dict(**counts, affected_spectra=len(by_ion),
                   by_ion=dict(by_ion), csv_path=str(path),
                   line_checkable=line_counts["checkable"],
                   line_agree=line_counts["agree"],
                   line_disagree=line_counts["disagree"],
                   line_affected_spectra=len(line_by_ion),
                   line_by_ion=dict(line_by_ion),
                   line_csv_path=str(line_path))
    (out / "nist_level_weight_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.output / "spectra.sqlite", args.output),
                     indent=2))


if __name__ == "__main__":
    main()
