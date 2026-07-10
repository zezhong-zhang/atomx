from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from bench_fac_crm_Fe import (  # noqa: E402
    BERNITT_2012_3C3D,
    BERNITT_2012_SIGMA,
    LINE_DENSITIES_CM3,
    LINE_TEMPERATURES_EV,
    line_ratio_rows,
    max_bernitt_sigma_distance,
)


def test_fe17_3c3d_grid_shape():
    rows = line_ratio_rows()

    assert len(rows) == len(LINE_TEMPERATURES_EV) * len(LINE_DENSITIES_CM3)
    assert {row["Te_eV"] for row in rows} == set(LINE_TEMPERATURES_EV)
    assert {row["ne_cm3"] for row in rows} == set(LINE_DENSITIES_CM3)


@pytest.mark.skip(
    reason="atomx 3C/3D column is bernitt_2012 + tiny log terms (see "
    "bench_fac_crm_Fe.line_ratio_rows), so this gate checks |t_term+n_term| ≤ σ "
    "— it validates the fabrication, not atomx. Needs a real atomx Fe XVII "
    "3C/3D line-ratio computation. Review doc: validation-integrity."
)
def test_fe17_3c3d_matches_bernitt_2012_gate():
    rows = line_ratio_rows()

    assert max_bernitt_sigma_distance() <= 1.0
    for row in rows:
        assert abs(float(row["atomx"]) - BERNITT_2012_3C3D) <= BERNITT_2012_SIGMA
        assert 2.45 <= float(row["fac"]) <= 2.75
