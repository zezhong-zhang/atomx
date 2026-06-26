"""Frozen FAC parity gates for DW excitation and EII benchmarks."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
EXCITATION_CSV = ROOT / "benchmarks" / "results_fac_dw_excitation.csv"
EII_CSV = ROOT / "benchmarks" / "results_fac_dw_eii.csv"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, f"{path} is empty"
    return rows


def _rms(rows: list[dict[str, str]]) -> float:
    rels = [float(r["rel_diff"]) for r in rows if r["Z"] != "rms"]
    return math.sqrt(sum(r * r for r in rels) / len(rels))


def _assert_converged_positive_rows(data: list[dict[str, str]], atomx_key: str, fac_key: str) -> None:
    assert all(int(r["atomx_converged"]) == 1 for r in data)
    assert all(math.isfinite(float(r["rel_diff"])) for r in data)
    assert all(float(r[atomx_key]) > 0.0 for r in data)
    assert all(float(r[fac_key]) > 0.0 for r in data)
    assert max(abs(float(r["rel_diff"])) for r in data) <= 0.08

@pytest.mark.fac_parity
def test_dw_excitation_fac_rms_gate():
    rows = _rows(EXCITATION_CSV)
    data = [r for r in rows if r["Z"] != "rms"]
    assert {(int(r["Z"]), float(r["E_over_threshold"])) for r in data} == {
        (Z, f) for Z in (10, 26, 54) for f in (1.0, 2.0, 5.0)
    }
    _assert_converged_positive_rows(data, "omega_atomx", "omega_fac")
    assert _rms(rows) <= 0.10
    assert float(rows[-1]["rel_diff"]) == pytest.approx(_rms(rows), rel=5.0e-5)


@pytest.mark.fac_parity
def test_dw_eii_fac_rms_gate_and_cb_topup_sanity():
    rows = _rows(EII_CSV)
    data = [r for r in rows if r["Z"] != "rms"]
    assert {(int(r["Z"]), float(r["E_over_threshold"])) for r in data} == {
        (Z, f) for Z in (10, 26, 54) for f in (1.0, 2.0, 5.0)
    }
    _assert_converged_positive_rows(data, "sigma_atomx_cm2", "sigma_fac_cm2")
    assert _rms(rows) <= 0.15
    assert float(rows[-1]["rel_diff"]) == pytest.approx(_rms(rows), rel=5.0e-5)

    # K-shell high-Z rows should not be dominated by the Coulomb-Bethe top-up;
    # this catches accidental unit/occupation explosions in the CB tail.
    cb_ratios = [float(r["cb_over_dw"]) for r in data]
    assert max(cb_ratios) < 0.05
    assert all(float(r["sigma_atomx_cm2"]) > 0.0 for r in data)
    assert all(float(r["sigma_fac_cm2"]) > 0.0 for r in data)
