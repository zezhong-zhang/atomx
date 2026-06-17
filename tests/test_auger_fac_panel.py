"""FAC AITable reference panel for Fe XVII KLL Auger widths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atomx.scattering.autoionization import AutoionizationState


FIXTURE_PATH = Path(__file__).parent / "data" / "fe17_dr_fac.json"
FAC_AUGER_TOLERANCE = 0.10

REFERENCE_PANEL = {
    24: 2.7544e13,
    32: 4.1004e14,
    62: 6.4310e13,
    135: 9.8863e13,
    150: 1.2721e13,
}


def _fac_ilev(label: str) -> int:
    prefix, _ = label.split(":", 1)
    return int(prefix.removeprefix("FAC ILEV "))


def _fac_resonance_states(data: dict) -> dict[int, AutoionizationState]:
    return {
        _fac_ilev(row["label"]): AutoionizationState(
            label=row["label"],
            E_resonance_eV=float(row["E_resonance_eV"]),
            J=float(row["J"]),
            gamma_auger_per_s=float(row["gamma_auger_per_s"]),
            a_radiative_per_s=float(row["a_radiative_per_s"]),
        )
        for row in data["resonances"]
    }


@pytest.fixture(scope="module")
def fac_resonances() -> dict[int, AutoionizationState]:
    data = json.loads(FIXTURE_PATH.read_text())
    return _fac_resonance_states(data)


def test_fe17_kll_auger_widths_match_fac_aitable_panel(fac_resonances):
    """Pin five FAC AITable KLL resonances to guard per-resonance widths."""
    assert set(REFERENCE_PANEL).issubset(fac_resonances)

    for ilev, expected_gamma in REFERENCE_PANEL.items():
        state = fac_resonances[ilev]
        rel_err = abs(state.gamma_auger_per_s - expected_gamma) / expected_gamma
        assert rel_err <= FAC_AUGER_TOLERANCE, (
            f"FAC ILEV {ilev} Auger width differs by {rel_err:.2%}: "
            f"atomx={state.gamma_auger_per_s:.6e} s^-1, "
            f"FAC={expected_gamma:.6e} s^-1"
        )
