from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

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

    assert len(rows) == 6 * 10
    for Te in TEMPERATURES_KEV:
        subset = [row for row in rows if row["Te_keV"] == Te]
        assert len(subset) == 10
        assert abs(sum(float(row["atomx"]) for row in subset) - 1.0) < 1.0e-12
        assert abs(sum(float(row["fac"]) for row in subset) - 1.0) < 1.0e-12
        assert abs(sum(float(row["chianti_v11"]) for row in subset) - 1.0) < 1.0e-12


def test_fe_coronal_balance_meets_fac_gate():
    assert fractional_abundance_rms("fac") <= 0.15
    assert fractional_abundance_rms("chianti_v11") <= 0.15
