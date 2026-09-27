#!/usr/bin/env python3
"""Find NIST line rows whose stored wavenumber disagrees with vacuum wavelength."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "output/nist_spectra/spectra.sqlite"
OUTPUT = ROOT / "output/nist_spectra"


def discrepancy(wavelength_nm: float | None, uncertainty_nm: float | None,
                wavenumber_raw: str | None) -> tuple[float, float, float | None] | None:
    """Return stored-minus-derived cm⁻¹, fractional gap, and listed-uncertainty multiples."""
    try:
        wavelength = float(wavelength_nm)
        wavenumber = float(wavenumber_raw)
    except (TypeError, ValueError):
        return None
    if (not math.isfinite(wavelength) or wavelength <= 0 or
            not math.isfinite(wavenumber) or wavenumber <= 0):
        return None
    derived = 1e7 / wavelength
    delta = wavenumber - derived
    relative = abs(delta) / derived
    multiple = None
    if (uncertainty_nm is not None and math.isfinite(uncertainty_nm)
            and uncertainty_nm > 0):
        wavelength_uncertainty_cm1 = 1e7 * uncertainty_nm / wavelength ** 2
        multiple = abs(delta) / wavelength_uncertainty_cm1
    return delta, relative, multiple


def audit(archive: Path = ARCHIVE, *, min_relative: float = 0.001,
          min_uncertainty_multiple: float = 10) -> tuple[dict, list[dict]]:
    flagged = []
    checked = 0
    with sqlite3.connect(archive) as db:
        for (z, charge, row_index, observed_nm, observed_unc_nm, medium,
             wavenumber_raw, line_reference, url, sha256) in db.execute("""
             SELECT l.z,l.charge,l.row_index,l.observed_nm,l.observed_unc_nm,
                    l.medium,l.wavenumber_raw,l.line_reference,s.url,s.sha256
             FROM lines l LEFT JOIN nist_line_detail_sources s
               ON s.z=l.z AND s.charge=l.charge
             WHERE l.observed_nm IS NOT NULL
             ORDER BY l.z,l.charge,l.row_index"""):
            if medium != "vacuum":
                continue
            result = discrepancy(observed_nm, observed_unc_nm, wavenumber_raw)
            if result is None:
                continue
            checked += 1
            delta, relative, multiple = result
            large_vs_uncertainty = (multiple is not None and
                                    multiple >= min_uncertainty_multiple)
            if relative < min_relative and not large_vs_uncertainty:
                continue
            flagged.append({
                "z": z, "charge": charge, "row_index": row_index,
                "observed_nm": observed_nm,
                "observed_unc_nm": observed_unc_nm,
                "stored_wavenumber_cm1": wavenumber_raw,
                "from_wavelength_cm1": 1e7 / observed_nm,
                "stored_minus_derived_cm1": delta,
                "relative_difference_percent": 100 * relative,
                "wavelength_uncertainty_multiples": multiple,
                "large_vs_listed_wavelength_uncertainty": large_vs_uncertainty,
                "line_reference": line_reference,
                "official_source_url": url,
                "official_source_sha256": sha256,
            })
    summary = {
        "checked_observed_vacuum_rows": checked,
        "relative_difference_threshold_percent": 100 * min_relative,
        "wavelength_uncertainty_multiple_threshold": min_uncertainty_multiple,
        "flagged_rows": len(flagged),
        "flagged_over_relative_threshold": sum(
            abs(row["relative_difference_percent"]) >= 100 * min_relative
            for row in flagged),
        "flagged_large_vs_listed_wavelength_uncertainty": sum(
            bool(row["large_vs_listed_wavelength_uncertainty"])
            for row in flagged),
        "interpretation": ("Source-field consistency check only. The stored "
                           "wavenumber and wavelength are preserved unchanged; "
                           "the wavenumber can have independent precision."),
    }
    return summary, flagged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary, flagged = audit(args.archive)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "nist_line_wavenumber_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    with (args.out / "nist_line_wavenumber_discrepancies.csv").open(
            "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "z", "charge", "row_index", "observed_nm", "observed_unc_nm",
            "stored_wavenumber_cm1", "from_wavelength_cm1",
            "stored_minus_derived_cm1", "relative_difference_percent",
            "wavelength_uncertainty_multiples",
            "large_vs_listed_wavelength_uncertainty", "line_reference",
            "official_source_url", "official_source_sha256"])
        writer.writeheader()
        writer.writerows(flagged)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
