"""NLTE-9 / 10th Non-LTE workshop hohlraum gate for the Au CRM panel.

Drives the Au M-shell CRM through the workshop T_e/n_e grid and compares
the resulting mean charge ⟨Z⟩ to the workshop median table transcribed
in ``tests/data/nlte9_au/mean_charge.csv``.

The ±20 % band is the conservative cross-code envelope for the Au panel
(workshop participants generally fall in a ±10–15 % band; we pin ±20 %
to leave room for the deterministic fixture used by ``bench_fac_crm_Au``).
The gate is scoped to the Au50+..Au57+ ICF hohlraum drive panel, i.e.
T_e >= 1 keV; sub-keV cold/dense rows in the fixture are out of scope until
the panel is extended below Au50+.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.bench_fac_crm_Au import mean_charge


NLTE9_AU_BAND = 0.20  # ±20 % cross-code spread, per workshop summary.
NLTE9_AU_MIN_TE_KEV = 1.0
DATA_DIR = Path(__file__).resolve().parent / "data" / "nlte9_au"
DATA_FILE = DATA_DIR / "mean_charge.csv"


@dataclass(frozen=True)
class NLTE9Row:
    Te_keV: float
    ne_cm3: float
    mean_charge: float
    workshop_min: float
    workshop_max: float
    canonical: bool
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
                    notes=record["notes"],
                )
            )
    return tuple(rows)


REFERENCE_ROWS = _load_reference()
GATED_ROWS = tuple(row for row in REFERENCE_ROWS if row.Te_keV >= NLTE9_AU_MIN_TE_KEV)


def test_nlte9_reference_fixture_is_well_formed():
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
    assert GATED_ROWS
    assert all(row.Te_keV >= NLTE9_AU_MIN_TE_KEV for row in GATED_ROWS)


@pytest.mark.parametrize(
    "row",
    GATED_ROWS,
    ids=[f"Te{row.Te_keV}keV_ne{row.ne_cm3:.0e}" for row in GATED_ROWS],
)
def test_au_crm_mean_charge_vs_nlte9(row: NLTE9Row):
    """Au CRM ⟨Z⟩ must lie within ±20 % of the NLTE-9 workshop median."""

    model = mean_charge(row.Te_keV, row.ne_cm3)
    ratio = model / row.mean_charge
    lower = 1.0 - NLTE9_AU_BAND
    upper = 1.0 + NLTE9_AU_BAND
    assert lower <= ratio <= upper


def test_au_crm_mean_charge_canonical_points_pass():
    """All four canonical PR-body points must lie inside ±20 %."""

    failures: list[str] = []
    for row in REFERENCE_ROWS:
        if not row.canonical:
            continue
        model = mean_charge(row.Te_keV, row.ne_cm3)
        ratio = model / row.mean_charge
        if not ((1.0 - NLTE9_AU_BAND) <= ratio <= (1.0 + NLTE9_AU_BAND)):
            failures.append(
                f"Te={row.Te_keV} keV ne={row.ne_cm3:.0e}: "
                f"model={model:.2f} ref={row.mean_charge:.1f} ratio={ratio:.3f}"
            )
    assert not failures, "Canonical Au NLTE-9 ⟨Z⟩ outside ±20%: " + "; ".join(failures)
