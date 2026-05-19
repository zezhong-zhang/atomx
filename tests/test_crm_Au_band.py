"""Au M-band UTA acceptance tests."""

from __future__ import annotations

import pytest

from benchmarks.bench_fac_crm_Au import (
    AU52_BAUCHE_ARNOULT_REFERENCE,
    CHARGE_STATES,
    M_BAND_EV,
    au_m_shell_configs,
    build_iso_panel,
    mband_moments,
)


def test_au_iso_panel_covers_requested_charge_states():
    panel = build_iso_panel()
    assert tuple(stage.charge for stage in panel) == CHARGE_STATES
    for stage in panel:
        assert stage.atom.Z == 79
        assert stage.atom.charge == stage.charge
        assert stage.uta_mode is True
        assert getattr(stage.atom, "uta_mode") is True
        assert len(stage.configs) >= 15
        assert stage.configs == au_m_shell_configs(stage.charge)


def test_au52_uta_moments_match_bauche_arnoult_gate():
    moments = mband_moments(52)
    ref = AU52_BAUCHE_ARNOULT_REFERENCE
    assert moments.centroid_eV == pytest.approx(ref.centroid_eV, rel=0.02)
    assert moments.variance_eV2 == pytest.approx(ref.variance_eV2, rel=0.02)
    assert moments.n_lines >= 3


def test_au_mband_transitions_stay_in_icf_window():
    stage = [candidate for candidate in build_iso_panel() if candidate.charge == 52][0]
    for transition in stage.crm.transitions:
        energy = (
            stage.crm.levels[transition.upper].energy_eV
            - stage.crm.levels[transition.lower].energy_eV
        )
        assert M_BAND_EV[0] <= energy <= M_BAND_EV[1]
