"""Stored Pütterich-2008 W cooling reference: provenance gate + honest status.

What this file does **not** do
-----------------------------
There is no atomx-vs-published W cooling-curve validation here and there
cannot be one from the current benchmark fixtures: ``bench_fac_crm_W``
exposes ``tungsten_cooling_rate`` as ``putterich_ref · (1 + 0.18·sin(ln T))``
(a documented synthetic placeholder), its ``build_w_ion_fixture`` CRM
levels/A-rates are closed-form synthetic numbers, and the
ionization/recombination rates handed to ``coronal_charge_balance`` are
synthetic too.  The old test in this file asserted
``|atomx − putterich|/putterich <= 0.30``, i.e. ``0.18·|sin| <= 0.30``;
measured, that deviation is 17.99 % at 5 eV against the 30 % gate, so the
gate's only content was the engineered offset.  It is now replaced by the
placeholder identity below, which states what the benchmark number is
instead of presenting it as physics.

Provenance status of the stored reference (checked 2026-09)
----------------------------------------------------------
Every row carries the ``Putterich et al. 2008 PPCF 50 085016`` citation,
but the values have 9 significant figures (``1.85782573e-31``), which a
digitisation of a published figure cannot produce, and no generator or
analytic fit is committed.  Six candidate closed forms were fitted to the
nine nodes (power law, ``A T^n e^{-B/T}``, ``A T^n e^{-B/T^2}``,
``A(1-e^{-(T/B)^n})``, ``A T^n/(B+T^n)``, ``A + B e^{-C/T}``): the best
reaches only 8.8e-2 maximum relative error, so the file is **not** the
output of a simple published analytic fit either.  The numbers therefore
cannot be reproduced or verified inside this repository; only their
*internal* consistency is gated here.  The CSV is retained (not deleted)
because ``bench_fac_crm_W.write_results_csv`` reads it for its
``Lw_putterich_W_m3`` column.
"""

from __future__ import annotations

import csv
import math
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

# Digitisation envelope: the Pütterich 2008 low-T M-shell W cooling curve
# spans ~1.9e-31 to ~1.3e-30 W m^3 over 5-100 eV (2026-09-17 check of the
# stored file).  The band is a provenance/units tripwire, not an atomx gate.
LW_MIN = 1.0e-31
LW_MAX = 1.0e-29
PROVENANCE = "Putterich et al. 2008 PPCF 50 085016"

# The placeholder's deterministic perturbation amplitude, from
# bench_fac_crm_W.tungsten_cooling_rate.
PLACEHOLDER_AMPLITUDE = 0.18


def _rows() -> list[dict[str, str]]:
    with PUTTERICH_2008_DIGITIZED_CSV.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, f"{PUTTERICH_2008_DIGITIZED_CSV} is empty"
    return rows


def test_stored_reference_provenance_and_log_log_shape():
    """The stored external reference: provenance, monotone grid, sane scale."""
    rows = _rows()
    assert len(rows) >= 9
    Te = [float(r["Te_eV"]) for r in rows]
    Lw = [float(r["Lw_W_m3"]) for r in rows]
    assert all(b > a for a, b in zip(Te, Te[1:], strict=False)), "Te grid must be monotone"
    assert Te[0] == 5.0 and Te[-1] == 100.0
    assert all(y > 0.0 for y in Lw)
    assert all(LW_MIN <= y <= LW_MAX for y in Lw), (
        f"stored Lw outside the documented envelope [{LW_MIN:.0e}, "
        f"{LW_MAX:.0e}] W m^3 — units or column mix-up?"
    )
    # The stored low-T M-shell branch rises monotonically over 5-100 eV.
    assert all(b > a for a, b in zip(Lw, Lw[1:], strict=False))
    assert all(PROVENANCE in r.get("source", "") for r in rows), (
        "every row must carry its Putterich 2008 provenance"
    )
    # ... while stating that the numbers themselves are unverifiable here:
    # 9 significant figures rule out a figure digitisation and no generator
    # is committed (see the module docstring).
    assert all(len(r["Lw_W_m3"].split("e")[0].replace(".", "").lstrip("0")) > 4
               for r in rows)

    # The interpolation helper must reproduce the stored nodes exactly.
    for t, y in zip(Te, Lw, strict=False):
        assert putterich_2008_cooling_reference(t) == pytest.approx(y, rel=1e-12, abs=0.0)


def test_benchmark_atomx_column_is_the_documented_synthetic_placeholder():
    """``tungsten_cooling_rate(T) == ref(T)·(1 + 0.18·sin(ln T))`` — plumbing.

    This is the *definition* of the benchmark's "atomx" W cooling number
    (bench_fac_crm_W, docstring: "SYNTHETIC PLACEHOLDER — NOT real atomx
    output"), asserted here at machine precision so that a real atomx
    computation replacing it trips this test and forces the claim to be
    re-stated.  Measured deviation from the stored reference over the nine
    committed nodes: 17.99 % at 5 eV (worst), 2.62 % at 20 eV, i.e. exactly
    the analytic amplitude 0.18·|sin(ln T)| — which is why the previous
    "within 30 %" gate here was satisfied by construction.
    """
    worst = 0.0
    worst_at = None
    for row in _rows():
        Te = float(row["Te_eV"])
        reference = float(row["Lw_W_m3"])
        atomx = tungsten_cooling_rate(Te)
        assert pytest.approx(
            reference * (1.0 + PLACEHOLDER_AMPLITUDE * math.sin(math.log(Te))),
            rel=1e-12, abs=0.0) == atomx
        rel = abs(atomx - reference) / reference
        assert rel <= PLACEHOLDER_AMPLITUDE
        if rel > worst:
            worst, worst_at = rel, Te
    assert worst_at == 5.0
    assert pytest.approx(0.17987, rel=1e-3, abs=0.0) == worst
