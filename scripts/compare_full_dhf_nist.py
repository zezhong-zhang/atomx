"""Read completed isotope-aware DHF comparisons from the full job ledger."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomx.reference_data.matching import configuration_has_single_level


def collect(archive_path: Path, ledger_path: Path,
            include_averages: bool = False) -> list[dict]:
    if not ledger_path.exists():
        return []
    reference = sqlite3.connect(archive_path)
    reference.row_factory = sqlite3.Row
    ledger = sqlite3.connect(ledger_path)
    result = []
    try:
        statuses = (("compared", "configuration_average_only",
                     "radial_box_unresolved") if include_averages else ("compared",))
        placeholders = ",".join("?" for _ in statuses)
        has_provenance = "classification_source" in {row[1] for row in
                              ledger.execute("PRAGMA table_info(line_jobs)")}
        source_column = ("classification_source" if has_provenance
                         else "NULL AS classification_source")
        for (z, charge, index, predicted, error, isotope_a, status,
             lo, hi, classification_source) in ledger.execute(
                """SELECT z,charge,row_index,calculated_nm,energy_error_percent,
                          isotope_a,status,lower_configuration,upper_configuration,"""
                + source_column + " FROM line_jobs WHERE status IN ("
                + placeholders + ") ORDER BY z,charge,row_index", statuses):
            if (status == "compared" and
                    (not configuration_has_single_level(lo) or
                     not configuration_has_single_level(hi))):
                status = "configuration_average_only"
                if not include_averages:
                    continue
            line = reference.execute("""SELECT l.*,s.symbol,s.label
                FROM lines l JOIN species s USING(z,charge)
                WHERE l.z=? AND l.charge=? AND l.row_index=?""",
                (z, charge, index)).fetchone()
            if line is None:
                continue
            result.append(dict(z=z, charge=charge, symbol=line["symbol"],
                ion=line["label"], nist_row_index=index,
                model="DHF+B+QED+NMS",
                comparison_status=status,
                classification_source=classification_source,
                cache_status=f"computed; {line['symbol']}-{isotope_a}; normal mass shift",
                lower=line["lower_configuration"] or lo,
                nist_lower_raw=line["lower_configuration"],
                lower_term=line["lower_term"], lower_j=line["lower_j"],
                upper=line["upper_configuration"] or hi,
                nist_upper_raw=line["upper_configuration"],
                upper_term=line["upper_term"], upper_j=line["upper_j"],
                observed_nm=line["observed_nm"],
                observed_unc_nm=line["observed_unc_nm"],
                ritz_nm=line["ritz_nm"], calculated_nm=predicted,
                energy_error_percent=error,
                lower_level_row=None, upper_level_row=None,
                nist_medium=line["medium"], isotope_a=isotope_a))
    finally:
        reference.close()
        ledger.close()
    return result
