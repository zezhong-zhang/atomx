from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

# The "fac"/"chianti"/"atomx" columns in bench_fac_crm_Fe are all fabricated
# from one Gaussian (atomx = chianti·(1+0.045·sin), fac = chianti·(1+0.035·ramp)).
# Any test asserting "atomx ≈ fac/chianti" therefore validates the fabrication,
# not atomx physics.  Such tests are skipped until real CHIANTI reference data
# and a real atomx coronal_charge_balance run replace the synthetic fixture.
_SYNTHETIC_REASON = (
    "asserts fabricated self-agreement over synthetic sin/ramp-perturbed "
    "fixtures (see bench_fac_crm_Fe._reference_balance); needs real CHIANTI "
    "reference + real atomx CRM run. Review doc: validation-integrity."
)

from bench_fac_crm_Fe import (  # noqa: E402
    TEMPERATURES_KEV,
    coronal_balance_rows,
    fe_iso_stage_panel,
    fractional_abundance_rms,
)


def test_fe_iso_stage_panel_covers_xray_charge_states():
    panel = fe_iso_stage_panel()

    assert [stage.charge for stage in panel] == list(range(16, 26))
    assert panel[0].roman == "Fe XVII"
    assert panel[-1].roman == "Fe XXVI"
    assert all(1 <= len(stage.configs) <= 20 for stage in panel)
    assert panel[0].configs[0] == "1s2 2s2 2p6"


def test_fe_coronal_balance_fixture_shape_and_normalisation():
    rows = coronal_balance_rows()

    assert len(rows) == len(TEMPERATURES_KEV) * 10
    for Te in TEMPERATURES_KEV:
        subset = [row for row in rows if row["Te_keV"] == Te]
        assert len(subset) == 10
        assert abs(sum(float(row["atomx"]) for row in subset) - 1.0) < 1.0e-12
        assert abs(sum(float(row["fac"]) for row in subset) - 1.0) < 1.0e-12
        assert abs(sum(float(row["chianti_v11"]) for row in subset) - 1.0) < 1.0e-12


@pytest.mark.skip(reason=_SYNTHETIC_REASON)
def test_fe_coronal_balance_meets_fac_gate():
    assert fractional_abundance_rms("fac") <= 0.15
    assert fractional_abundance_rms("chianti_v11") <= 0.15


@pytest.mark.skip(reason=_SYNTHETIC_REASON)
def test_fe_700ev_charge_state_distribution_matches_fac_within_20_percent():
    rows = [row for row in coronal_balance_rows() if row["Te_eV"] == 700.0]
    assert len(rows) == 10
    for row in rows:
        fac = float(row["fac"])
        atomx = float(row["atomx"])
        if fac > 1.0e-6:
            assert abs(atomx - fac) / fac <= 0.20
