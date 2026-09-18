"""FAC parity tests for the DR driver (HYP-146, HYP-530).

``dr_calculate`` (no hooks) drives the ab-initio determinant-CI manifold
pipeline (``ResonanceManifold.dr_levels``) since 2026-07-16; the full
end-to-end build vs the clean-room FAC dr33 reference lives in
``tests/test_dr_manifold_e2e.py`` (slow, ~1-2 min).  This module keeps
the fast gates:

1. ``test_kernel_parity_with_fac`` — feed FAC's own AI rates, TR rates,
   and resonance energies into atomx's ``dr_rate_coefficient`` Burgess
   kernel and verify it reproduces the reference α_DR within 1 % at every
   gate temperature.  This is the test that protects the DR kernel.

2. ``test_benchmark_csv_gate`` — gate the committed
   ``benchmarks/results_fac_dr.csv`` produced by
   ``benchmarks/bench_fac_dr.py`` (manifold pipeline vs the clean-room
   FAC dr33 reference ``tests/data/fe17_dr33_fac_cleanroom.json``:
   SetSE/VP/MS/Breit off, AICut 0).  Gates: rms |rel_diff| <= 0.35 and
   every T >= 5 MK point within alpha ratio [0.8, 1.35].  The 1-MK point
   (ratio ~0.49) is Boltzmann suppression from the documented ~+40 eV
   frozen-orbital E_res offset, not a width error — it dominates the rms
   and is why the target is 0.35 rather than the generic 0.20.

3. ``test_end_to_end_driver_returns_positive_alpha`` — smoke test of the
   hook path (external widths through the Burgess kernel).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from atomx.scattering.autoionization import (
    AutoionizationState,
    dr_calculate,
    dr_rate_coefficient,
)

FIXTURE_PATH = Path(__file__).parent / "data" / "fe17_dr_fac.json"
RESULTS_CSV = Path(__file__).parent.parent / "benchmarks" / "results_fac_dr.csv"
KERNEL_TOLERANCE = 0.01  # 1 % parity for the Burgess kernel itself
RMS_GATE = 0.35          # dominated by the documented 1-MK E_res offset
RATIO_GATE = (0.8, 1.35)  # per-point gate for T >= 5 MK


@pytest.fixture(scope="module")
def fac_fixture() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(f"FAC reference fixture missing: {FIXTURE_PATH}")
    data = json.loads(FIXTURE_PATH.read_text())
    if data.get("n_fac_resonances", 0) == 0:
        pytest.skip("FAC fixture has zero resonances — regenerate the fixture")
    return data


def _fac_resonance_states(data: dict) -> list[AutoionizationState]:
    return [
        AutoionizationState(
            label=row["label"],
            E_resonance_eV=float(row["E_resonance_eV"]),
            J=float(row["J"]),
            gamma_auger_per_s=float(row["gamma_auger_per_s"]),
            a_radiative_per_s=float(row["a_radiative_per_s"]),
        )
        for row in data["resonances"]
    ]


# ---------------------------------------------------------------------------
# 1.  Kernel parity — feeds FAC widths through atomx's Burgess kernel
# ---------------------------------------------------------------------------

@pytest.mark.external_fac
def test_kernel_parity_with_fac(fac_fixture):
    states = _fac_resonance_states(fac_fixture)
    T_eV = np.asarray(fac_fixture["T_eV"], dtype=float)
    rc = dr_rate_coefficient(
        states,
        T_eV,
        initial_state_g=fac_fixture.get("initial_state_g", 1.0),
        initial_state_label="Fe XVII 2p6 J=0 (FAC reference)",
    )
    expected = np.asarray(fac_fixture["alpha_fac_cm3_s"], dtype=float)
    np.testing.assert_allclose(
        rc.alpha_cm3_s, expected, rtol=KERNEL_TOLERANCE,
        err_msg=(
            "atomx Burgess kernel disagrees with FAC AITable+TRTable combine "
            "above 1 % — investigate dr_rate_coefficient prefactor or units."
        ),
    )


# ---------------------------------------------------------------------------
# 2.  Benchmark CSV gate — manifold pipeline vs clean-room FAC dr33
# ---------------------------------------------------------------------------

def _load_results_csv():
    if not RESULTS_CSV.exists():
        pytest.skip(
            f"benchmark CSV missing: {RESULTS_CSV} — regenerate with "
            "benchmarks/bench_fac_dr.py"
        )
    with RESULTS_CSV.open() as f:
        rows = list(csv.DictReader(f))
    data_rows = [r for r in rows if r["T_MK"] != "rms"]
    rms_rows = [r for r in rows if r["T_MK"] == "rms"]
    assert data_rows, "results_fac_dr.csv has no temperature rows"
    assert rms_rows, "results_fac_dr.csv has no rms row"
    return data_rows, rms_rows[0]


@pytest.mark.external_fac
def test_benchmark_csv_gate():
    """rms <= 0.35 and every T >= 5 MK point within alpha ratio [0.8, 1.35].

    Measured 2026-07-16 (manifold pipeline vs clean-room FAC dr33):
    ratios 0.49 / 0.99 / 1.08 / 1.17 at 1 / 5 / 10 / 50 MK, rms ~0.27.
    The 1-MK point carries the documented ~+40 eV frozen-orbital E_res
    offset through exp(-E/kT) and is excluded from the per-point gate.
    """
    data_rows, rms_row = _load_results_csv()
    assert len(data_rows) == 4

    rms = float(rms_row["rel_diff"])
    assert rms <= RMS_GATE, (
        f"DR benchmark rms |rel_diff| = {rms:.4f} exceeds {RMS_GATE} — "
        "regression in the manifold pipeline or the Burgess kernel"
    )

    lo, hi = RATIO_GATE
    for row in data_rows:
        T_mk = float(row["T_MK"])
        alpha_atomx = float(row["alpha_atomx_cm3_s"])
        alpha_fac = float(row["alpha_fac_cm3_s"])
        assert alpha_fac > 0.0
        ratio = alpha_atomx / alpha_fac
        if T_mk >= 5.0:
            assert lo <= ratio <= hi, (
                f"T = {T_mk} MK: alpha_atomx/alpha_FAC = {ratio:.3f} "
                f"outside [{lo}, {hi}]"
            )
        else:
            # 1 MK: frozen-orbital E_res offset — sanity-band only.
            assert 0.3 <= ratio <= 1.0, (
                f"T = {T_mk} MK: alpha ratio {ratio:.3f} outside the "
                "documented offset band [0.3, 1.0]"
            )


@pytest.mark.external_fac
def test_benchmark_csv_reference_is_cleanroom():
    """The CSV must be benchmarked against the clean-room FAC reference."""
    data_rows, rms_row = _load_results_csv()
    assert all("clean-room FAC" in r["note"] for r in data_rows), (
        "results_fac_dr.csv note column no longer labels the clean-room "
        "FAC reference — was the benchmark regenerated with a different "
        "reference?"
    )
    assert "0.35" in rms_row["note"]


# ---------------------------------------------------------------------------
# 3.  Hook-path smoke test (external widths through the Burgess kernel)
# ---------------------------------------------------------------------------

# Lightweight Atom-like input; inlined so the test does not import from the
# benchmarks tree.
class _FeXVIIModel:
    Z = 26
    charge = 16
    converged = True
    _config_strings = ["1s2 2s2 2p6"]
    result = None
    orbital_energies = {
        "1s1/2": -260.0,
        "2s1/2": -47.0,
        "2p1/2": -45.0,
        "2p3/2": -44.6,
        "3s1/2": -18.0,
        "3p1/2": -16.0,
        "3p3/2": -15.8,
        "3d3/2": -14.4,
        "3d5/2": -14.4,
        "4s1/2": -10.0,
        "4p1/2": -9.5,
        "4p3/2": -9.4,
        "4d3/2": -9.0,
        "4d5/2": -9.0,
    }


def test_end_to_end_driver_returns_positive_alpha(fac_fixture):
    """Smoke test: dr_calculate produces finite positive α with external hooks."""
    T_eV = np.asarray(fac_fixture["T_eV"], dtype=float)
    # Simple positive width hooks: exercises the legacy enumeration path
    # without building an MCDHF result or the manifold pipeline.
    rc = dr_calculate(
        _FeXVIIModel(),
        "1s2 2s2 2p6",
        n_max=4,
        l_max=2,
        T_eV=T_eV,
        auger_rate_fn=lambda _spec, _e: 1e13,
        radiative_rate_fn=lambda _spec, _e: 1e10,
    )
    assert rc.alpha_cm3_s.shape == T_eV.shape
    assert np.all(rc.alpha_cm3_s > 0.0), "expected α_DR > 0 at every gate T_e"
    assert len(rc.resonances) > 0, "enumeration should produce resonances"
