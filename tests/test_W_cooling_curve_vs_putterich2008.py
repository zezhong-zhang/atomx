"""Digitized Pütterich 2008 W cooling curve: provenance gate + honest status.

History (2026-09-17 validation-integrity pass): this file's only test was
skipped, so ``tests/data/putterich2008_W_cooling_digitized.csv`` was asserted
by nothing.  Two things are now true and separate:

1. the stored external reference is **asserted** (provenance, grid, shape) —
   see ``test_digitized_reference_provenance_and_log_log_shape``;
2. there is **no live atomx W cooling-curve comparison**, and there cannot be
   one from the current benchmark fixtures: ``bench_fac_crm_W`` builds
   ``tungsten_cooling_rate`` as ``putterich_ref · (1 + 0.18·sin)``, its
   ``build_w_ion_fixture`` CRM levels/A-rates are closed-form synthetic
   numbers, and the ionization/recombination rates handed to the real
   ``coronal_charge_balance`` solver are synthetic too
   (``_ionization_rate`` / ``_recombination_rate``).  The comparison test
   below records that with a skip, and the README must not claim an atomx
   W cooling-curve validation.

The CSV itself is retained (not deleted) because it is load-bearing for
``bench_fac_crm_W.write_results_csv`` (the ``Lw_putterich_W_m3`` column) and
is genuine digitized external data.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from bench_fac_crm_W import (  # noqa: E402
    PUTTERICH_2008_DIGITIZED_CSV,
    putterich_2008_cooling_reference,
    tungsten_cooling_rate,
)

# Digitization envelope: the Pütterich 2008 low-T M-shell W cooling curve
# spans ~1.9e-31 to ~1.3e-30 W m^3 over 5-100 eV (2026-09-17 check of the
# stored file).  The band is a provenance/units tripwire, not an atomx gate.
LW_MIN = 1.0e-31
LW_MAX = 1.0e-29
PROVENANCE = "Putterich et al. 2008 PPCF 50 085016"


def _rows() -> list[dict[str, str]]:
    with PUTTERICH_2008_DIGITIZED_CSV.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, f"{PUTTERICH_2008_DIGITIZED_CSV} is empty"
    return rows


def test_digitized_reference_provenance_and_log_log_shape():
    """The stored external reference: provenance, monotone grid, sane scale."""
    rows = _rows()
    assert len(rows) >= 9
    Te = [float(r["Te_eV"]) for r in rows]
    Lw = [float(r["Lw_W_m3"]) for r in rows]
    assert all(b > a for a, b in zip(Te, Te[1:], strict=False)), "Te grid must be monotone"
    assert Te[0] == 5.0 and Te[-1] == 100.0
    assert all(y > 0.0 for y in Lw)
    assert all(LW_MIN <= y <= LW_MAX for y in Lw), (
        f"digitized Lw outside the documented envelope [{LW_MIN:.0e}, "
        f"{LW_MAX:.0e}] W m^3 — units or column mix-up?"
    )
    # The digitized low-T M-shell branch rises monotonically over 5-100 eV.
    assert all(b > a for a, b in zip(Lw, Lw[1:], strict=False))
    assert all(PROVENANCE in r.get("source", "") for r in rows), (
        "every row must carry its Putterich 2008 provenance"
    )

    # The interpolation helper must reproduce the stored nodes exactly.
    for t, y in zip(Te, Lw, strict=False):
        assert putterich_2008_cooling_reference(t) == pytest.approx(y, rel=1e-12)


@pytest.mark.skip(
    reason="no real atomx W cooling computation exists to gate: "
    "tungsten_cooling_rate is putterich_ref·(1+0.18·sin), the W CRM levels/"
    "A-rates in build_w_ion_fixture are closed-form synthetic numbers, and the "
    "ionization/recombination rates feeding coronal_charge_balance are "
    "synthetic as well — so |atomx - putterich|/putterich <= 0.30 would only "
    "check 0.18·|sin| <= 0.30. The stored digitized reference itself IS "
    "asserted by test_digitized_reference_provenance_and_log_log_shape. A real "
    "gate needs real W atomic rates. Review doc: validation-integrity."
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
