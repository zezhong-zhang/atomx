"""Fe XVII KLL Auger-width panel pinned to FAC ``AITable``.

The full FAC DR fixture contains 194 resonances from the
``Structure -> AITable -> TRTable`` workflow.  This test freezes five KLL
autoionizing rows from that fixture so changes to Auger-width plumbing have a
small, reviewable FAC reference panel with the epic's 10% tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atomx.scattering.autoionization import AutoionizationState


FIXTURE_PATH = Path(__file__).parent / "data" / "fe17_dr_fac.json"
FAC_AITABLE_TOLERANCE = 0.10

# Representative KLL rows from FAC 1.1.5 AITable.  The fixture labels use FAC's
# compact notation; ``2*7.3*2`` identifies the Delta-n=0 KLL block used here.
FE17_KLL_AUGER_PANEL = [
    {
        "ilev": 24,
        "E_resonance_eV": 264.37,
        "J": 0.5,
        "gamma_auger_per_s": 2.7544e13,
    },
    {
        "ilev": 32,
        "E_resonance_eV": 287.19,
        "J": 0.5,
        "gamma_auger_per_s": 4.1004e14,
    },
    {
        "ilev": 63,
        "E_resonance_eV": 314.58,
        "J": 1.5,
        "gamma_auger_per_s": 7.1429e13,
    },
    {
        "ilev": 72,
        "E_resonance_eV": 321.38,
        "J": 1.5,
        "gamma_auger_per_s": 3.1258e14,
    },
    {
        "ilev": 162,
        "E_resonance_eV": 391.09,
        "J": 3.5,
        "gamma_auger_per_s": 9.4244e13,
    },
]


@pytest.fixture(scope="module")
def fac_resonances_by_ilev() -> dict[int, dict]:
    data = json.loads(FIXTURE_PATH.read_text())
    rows = {}
    for row in data["resonances"]:
        prefix, _sep, _rest = row["label"].partition(":")
        if not prefix.startswith("FAC ILEV "):
            continue
        rows[int(prefix.removeprefix("FAC ILEV "))] = row
    return rows


@pytest.mark.parametrize("reference", FE17_KLL_AUGER_PANEL)
def test_fe17_kll_auger_width_panel_matches_fac_aitable(
    fac_resonances_by_ilev: dict[int, dict],
    reference: dict,
):
    row = fac_resonances_by_ilev[reference["ilev"]]
    assert "2*7.3*2" in row["label"], row["label"]

    state = AutoionizationState(
        label=row["label"],
        E_resonance_eV=float(row["E_resonance_eV"]),
        J=float(row["J"]),
        gamma_auger_per_s=float(row["gamma_auger_per_s"]),
        a_radiative_per_s=float(row["a_radiative_per_s"]),
    )

    assert state.E_resonance_eV == pytest.approx(reference["E_resonance_eV"])
    assert state.J == pytest.approx(reference["J"])
    assert state.gamma_auger_per_s == pytest.approx(
        reference["gamma_auger_per_s"],
        rel=FAC_AITABLE_TOLERANCE,
    )
