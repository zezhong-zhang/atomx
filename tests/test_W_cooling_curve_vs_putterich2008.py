"""W cooling curve gate against the digitised Putterich 2008 reference."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from bench_fac_crm_W import (  # noqa: E402
    PUTTERICH_2008_DIGITIZED_CSV,
    tungsten_cooling_rate,
)


def test_W_cooling_curve_vs_putterich2008_within_30_percent():
    with PUTTERICH_2008_DIGITIZED_CSV.open() as f:
        rows = list(csv.DictReader(f))

    assert rows
    for row in rows:
        Te = float(row["Te_eV"])
        if not 5.0 <= Te <= 100.0:
            continue
        reference = float(row["Lw_W_m3"])
        atomx = tungsten_cooling_rate(Te)
        rel = abs(atomx - reference) / reference
        assert rel <= 0.30
