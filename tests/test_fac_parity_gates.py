"""Frozen atomx **self-regression** snapshot (misnamed "FAC-parity" gate).

⚠️ This is NOT a validation against FAC.  The golden CSV was generated with
the benchmark matrix (including ``sfac`` columns), but this test reruns only
atomx and compares against the **frozen** ``atomx_*`` columns — i.e. it checks
that atomx has not drifted from its own past output.  The ``fac_*`` columns are
**never asserted**; they disagree with the atomx columns by orders of magnitude
in places (e.g. Z=10 Ne-like ``atomx_first_A_s`` = 3.59e6 vs ``fac_first_A_s`` =
6.85e-3), reflecting an unreconciled units/definition mismatch, so asserting
them here would be meaningless without first fixing that.  Treat a green run as
"atomx is self-consistent with its snapshot", **not** "atomx agrees with FAC".
See ``docs/CRITICAL_REVIEW_2026-07-09.md`` §V1.  (The ``fac_parity`` marker name
is retained only because CI/README select on it.)
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from fac_matrix import CSV_COLUMNS, run_atomx_matrix_cell  # noqa: E402

GOLDEN = ROOT / "benchmarks" / "reference" / "fac_matrix_2026-05-26.csv"
TIMING_MULTIPLIER = float(os.environ.get("FAC_PARITY_TIMING_MULT", "4.0"))


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
def test_frozen_atomx_self_regression_snapshot():
    """Atomx self-regression: fresh atomx vs frozen atomx_* columns only.

    NOT a FAC comparison — see module docstring / review §V1.
    """
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

    # Timing budget: the golden timing column was captured on a faster
    # developer machine. Keep the budget configurable so slower agent/CI
    # workers can still enforce that runtime remains within a bounded factor.
    for Z, actual_seconds in actual_time_by_z.items():
        assert actual_seconds <= TIMING_MULTIPLIER * frozen_time_by_z[Z], (
            Z,
            actual_seconds,
            frozen_time_by_z[Z],
            TIMING_MULTIPLIER,
        )
