"""Frozen FAC-parity gate.

The golden was generated with the benchmark matrix, including ``sfac`` columns,
but CI intentionally reruns only atomx and compares against frozen atomx values.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from fac_matrix import CSV_COLUMNS, run_atomx_matrix_cell  # noqa: E402

GOLDEN = ROOT / "benchmarks" / "reference" / "fac_matrix_2026-05-13.csv"


def _load_golden() -> list[dict]:
    with GOLDEN.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows, f"{GOLDEN} is empty"
    assert rows[0].keys() == set(CSV_COLUMNS)
    return rows


def _relerr(actual: float, expected: float) -> float:
    scale = max(abs(expected), 1.0e-300)
    return abs(actual - expected) / scale


@pytest.mark.fac_parity
def test_frozen_fac_parity_matrix_atomx_columns():
    golden_rows = _load_golden()
    frozen_time_by_z = defaultdict(float)
    actual_time_by_z = defaultdict(float)

    for golden in golden_rows:
        Z = int(golden["Z"])
        iso = golden["iso"]
        actual = run_atomx_matrix_cell(Z, iso)

        expected_E = float(golden["atomx_E_eV"])
        expected_E1 = float(golden["atomx_first_E1_eV"])
        expected_A = float(golden["atomx_first_A_s"])
        expected_gauge = float(golden["atomx_gauge_ratio_max"])

        assert _relerr(float(actual["atomx_E_eV"]), expected_E) <= 1.0e-3, (Z, iso, "E")
        assert _relerr(float(actual["atomx_first_E1_eV"]), expected_E1) <= 1.0e-3, (
            Z,
            iso,
            "first E1 energy",
        )
        if expected_A == 0.0:
            assert abs(float(actual["atomx_first_A_s"])) <= 1.0e-30, (Z, iso, "first A")
        else:
            assert _relerr(float(actual["atomx_first_A_s"]), expected_A) <= 5.0e-2, (
                Z,
                iso,
                "first A",
            )

        # Gauge-ratio tripwire: a 10 % perturbation in the radial /
        # angular pipeline shifts the worst-channel |A_v/A_l - 1| past
        # this bound.  Closed-shell ground configs with no s↔p E1
        # channel record exactly 0 by construction; treat them as a
        # strict-equality check so that any spurious non-zero entry from
        # a structure-code regression trips immediately.
        actual_gauge = float(actual["atomx_gauge_ratio_max"])
        if expected_gauge == 0.0:
            assert actual_gauge == 0.0, (Z, iso, "gauge_ratio_max non-zero")
        else:
            assert _relerr(actual_gauge, expected_gauge) <= 5.0e-2, (
                Z,
                iso,
                "gauge_ratio_max",
            )

        frozen_time_by_z[Z] += float(golden["t_atomx"])
        actual_time_by_z[Z] += float(actual["t_atomx"])

    # Timing budget: 15× frozen baseline. Golden was baselined before
    # J-resolved CSF expansion (4 → 37 CSFs) and gauge='both', which
    # fundamentally increased per-cell work. TODO: rebaseline golden
    # timing column and tighten back to 1.5×.
    for Z, actual_seconds in actual_time_by_z.items():
        assert actual_seconds <= 15.0 * frozen_time_by_z[Z], (Z, actual_seconds, frozen_time_by_z[Z])
