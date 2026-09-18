"""Au M-band UTA acceptance tests.

Honest status (2026-09-17 validation-integrity pass): ``mband_moments(52)``
does not compute anything physical — ``bench_fac_crm_Au.au52_mband_lines``
constructs a three-line sample *from* ``AU52_BAUCHE_ARNOULT_REFERENCE``
(centroid, centroid ± sqrt(2·variance), weights 1/4, 1/2, 1/4), so the
Bauche-Arnoult moment gate matches by construction and is skipped.  The
remaining tests assert fixture structure only; ``build_stage_crm`` also builds
its levels from those synthetic moments, so the ICF-window test below is a
property of the fixture, not a validation against the literature.
"""

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
        assert stage.atom.uta_mode is True
        assert len(stage.configs) >= 15
        assert stage.configs == au_m_shell_configs(stage.charge)


@pytest.mark.skip(
    reason="mband_moments(52) is not a physics computation: "
    "bench_fac_crm_Au.au52_mband_lines builds a 3-line sample from "
    "AU52_BAUCHE_ARNOULT_REFERENCE itself (centroid ± sqrt(2)*sigma, weights "
    "0.25/0.50/0.25), so the UTA moments reproduce the reference first moments "
    "by construction. Validates the fixture, not atomx. Needs a real atomx Au "
    "M-band UTA/CI calculation. Review doc: validation-integrity."
)
def test_au52_uta_moments_match_bauche_arnoult_gate():
    moments = mband_moments(52)
    ref = AU52_BAUCHE_ARNOULT_REFERENCE
    assert moments.centroid_eV == pytest.approx(ref.centroid_eV, rel=0.02)
    assert moments.variance_eV2 == pytest.approx(ref.variance_eV2, rel=0.02)
    assert moments.n_lines >= 3


def test_au_mband_transitions_stay_in_icf_window():
    """Fixture property: synthetic UTA levels sit inside the ICF M-band window.

    Structural only — the levels come from ``mband_moments``, which is the
    synthetic 3-line sample described above.
    """
    stage = [candidate for candidate in build_iso_panel() if candidate.charge == 52][0]
    for transition in stage.crm.transitions:
        energy = (
            stage.crm.levels[transition.upper].energy_eV
            - stage.crm.levels[transition.lower].energy_eV
        )
        assert M_BAND_EV[0] <= energy <= M_BAND_EV[1]
