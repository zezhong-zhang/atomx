"""Au M-shell CRM panel: stored-fixture integrity + calibration regression.

What this file is **not**
------------------------
It is not a validation of atomx against NLTE-9.  ``bench_fac_crm_Au.mean_charge``
derives from ``charge_fractions``, a hand-tuned Gaussian "smooth fractional
abundance model" whose ``center = 50 + 7·(Te−0.5)/4.5 − 0.18·log10(ne/1e20)``
is fitted to this same drive band, so agreement with the stored table
measures the tuning, not an atomx Au ionization balance.  In addition the
stored table's own provenance cannot be checked in-repo: ``workshop_min``
and ``workshop_max`` are exactly ``mean_charge ∓ 2.5`` in all 15 rows (a
constructed envelope, not a reported per-code spread) and no page/table
number is recorded — see ``tests/data/nlte9_au/README.md``.

Measured model/stored ratios (2026 audit, reproduced here):

| T_e (keV) | n_e (cm⁻³) | stored ⟨Z⟩ | model ⟨Z⟩ | ratio |
|---|---|---|---|---|
| 1  | 1e21 | 49.0 | 50.825 | **1.0372** (canonical) |
| 2  | 1e21 | 54.0 | 52.164 | **0.9660** (canonical) |
| 5  | 1e21 | 61.0 | 56.305 | **0.9230** (canonical) |
| 10 | 1e21 | 67.0 | 56.995 | **0.8507** (canonical) |
| 0.5| 1e22 | 40.5 | 50.330 | 1.2427 (worst row) |

The docstring that used to live here quoted "1.019 / 0.966 / 0.923 / 0.851"
as the four canonical ratios; 1.019 is the *non-canonical* (1.0 keV,
1e20 cm⁻³) row.

What remains, honestly labelled:

* ``test_nlte9_reference_fixture_is_well_formed`` — structure of the stored
  table, including the constructed-envelope identity and the per-row
  ``source`` column (F-class provenance, falsifiable);
* ``test_au_gaussian_model_reproduces_the_stored_rows`` — ``snapshot``
  (change detection on the calibrated Gaussian against the stored numbers,
  no physics claim).

The 13 previously-skipped tests are folded into these two: they asserted
the same thing against the same fixture.  A real atomx Au balance would
replace the second test, not unskip the first.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.bench_fac_crm_Au import mean_charge

NLTE9_AU_BAND = 0.20  # stored ±20 % envelope, quoted by the fixture notes
NLTE9_AU_MIN_TE_KEV = 1.0
DATA_DIR = Path(__file__).resolve().parent / "data" / "nlte9_au"
DATA_FILE = DATA_DIR / "mean_charge.csv"

# Envelope identity of the stored table (measured on all 15 rows).
ENVELOPE_HALF_WIDTH = 2.5

# Measured model/stored ratios, pinned as snapshot change detection
# (rel=1e-4 against the audit's numbers).
MODEL_RATIO_BY_CELL = {
    (0.5, 1.0e20): 1.1737,
    (0.5, 1.0e21): 1.1999,
    (0.5, 1.0e22): 1.2427,
    (1.0, 1.0e20): 1.0190,
    (1.0, 1.0e21): 1.0372,
    (1.0, 1.0e22): 1.0676,
    (2.0, 1.0e20): 0.9516,
    (2.0, 1.0e21): 0.9660,
    (2.0, 1.0e22): 0.9810,
    (5.0, 1.0e20): 0.9095,
    (5.0, 1.0e21): 0.9230,
    (5.0, 1.0e22): 0.9369,
    (10.0, 1.0e20): 0.8382,
    (10.0, 1.0e21): 0.8507,
    (10.0, 1.0e22): 0.8636,
}


@dataclass(frozen=True)
class NLTE9Row:
    Te_keV: float
    ne_cm3: float
    mean_charge: float
    workshop_min: float
    workshop_max: float
    canonical: bool
    source: str
    notes: str


def _load_reference() -> tuple[NLTE9Row, ...]:
    rows: list[NLTE9Row] = []
    with DATA_FILE.open() as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            rows.append(
                NLTE9Row(
                    Te_keV=float(record["Te_keV"]),
                    ne_cm3=float(record["ne_cm3"]),
                    mean_charge=float(record["mean_charge"]),
                    workshop_min=float(record["workshop_min"]),
                    workshop_max=float(record["workshop_max"]),
                    canonical=record["canonical"].strip().lower() == "true",
                    source=record["source"],
                    notes=record["notes"],
                )
            )
    return tuple(rows)


REFERENCE_ROWS = _load_reference()
OUT_OF_SCOPE_ROWS = tuple(
    row for row in REFERENCE_ROWS if row.Te_keV < NLTE9_AU_MIN_TE_KEV)


def test_nlte9_reference_fixture_is_well_formed():
    """Structure of the stored table — including its constructed envelope."""
    assert REFERENCE_ROWS, "NLTE-9 fixture is empty"
    canonical_cells = {
        (row.Te_keV, row.ne_cm3) for row in REFERENCE_ROWS if row.canonical
    }
    assert canonical_cells == {
        (1.0, 1.0e21),
        (2.0, 1.0e21),
        (5.0, 1.0e21),
        (10.0, 1.0e21),
    }
    for row in REFERENCE_ROWS:
        assert row.workshop_min < row.mean_charge < row.workshop_max
        assert 0.0 < row.mean_charge < 79.0  # bounded by Z(Au)
        # the stored envelope is mean ∓ 2.5 in every row: constructed, so
        # the ⟨Z⟩ values are not a transcription of per-code results
        assert row.workshop_min == pytest.approx(
            row.mean_charge - ENVELOPE_HALF_WIDTH, rel=1e-12, abs=0.0)
        assert row.workshop_max == pytest.approx(
            row.mean_charge + ENVELOPE_HALF_WIDTH, rel=1e-12, abs=0.0)
        # provenance is uniform with the other committed reference CSV
        assert row.source.strip(), "every row must carry its provenance"
        assert "Fontes" in row.source
        assert "NOT verifiable in-repo" in row.source
    assert OUT_OF_SCOPE_ROWS, "Fixture should retain the sub-keV provenance rows"
    assert all("low-Te edge" in row.notes for row in OUT_OF_SCOPE_ROWS)


def _row_params():
    """Parametrise on the stored rows; the sub-keV rows fail loudly.

    The 0.5 keV rows are genuinely outside the ±20 % band (measured worst
    1.2427 at 0.5 keV / 1e22 cm⁻³) because the atomx fixture is an
    Au50+...Au57+ M-shell panel while those cells need Au35+...Au49+; they
    are marked ``xfail(strict=True)`` with that measured number and cause
    rather than silently dropped from the parametrisation.
    """
    params = []
    for row in REFERENCE_ROWS:
        pid = f"Te{row.Te_keV}keV_ne{row.ne_cm3:.0e}"
        ratio = MODEL_RATIO_BY_CELL[(row.Te_keV, row.ne_cm3)]
        if not (1.0 - NLTE9_AU_BAND <= ratio <= 1.0 + NLTE9_AU_BAND):
            params.append(pytest.param(row, id=pid, marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    f"panel-scope: measured model/stored ratio {ratio:.4f} is "
                    "outside the ±20 % band; the 0.5 keV cells need "
                    "Au35+...Au49+ charge states, which the Au50+...Au57+ "
                    "M-shell fixture does not contain "
                    "(tests/data/nlte9_au/README.md)"
                ))))
        else:
            params.append(pytest.param(row, id=pid))
    return params


@pytest.mark.snapshot
@pytest.mark.parametrize("row", _row_params())
def test_au_gaussian_model_reproduces_the_stored_rows(row: NLTE9Row):
    """Snapshot: the hand-tuned Gaussian vs the stored table (no physics).

    ``@pytest.mark.snapshot`` because the expected ratios can only come from
    the fixture itself; this detects a change in the calibration, it does
    not validate atomx.  The ±20 % band is the stored envelope; the three
    0.5 keV rows are strict xfails (panel scope, see ``_row_params``).
    """
    ratio = mean_charge(row.Te_keV, row.ne_cm3) / row.mean_charge
    expected = MODEL_RATIO_BY_CELL[(row.Te_keV, row.ne_cm3)]
    assert pytest.approx(expected, rel=1e-3, abs=0.0) == ratio
    assert 1.0 - NLTE9_AU_BAND <= ratio <= 1.0 + NLTE9_AU_BAND
