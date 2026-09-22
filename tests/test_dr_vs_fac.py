"""FAC parity tests for the DR driver (HYP-146, HYP-530).

``dr_calculate`` (no hooks) drives the ab-initio determinant-CI manifold
pipeline (``ResonanceManifold.dr_levels``) since 2026-07-16; the full
end-to-end build vs the clean-room FAC dr33 reference lives in
``tests/test_dr_manifold_e2e.py`` (slow, ~1-2 min).  This module keeps
the fast gates plus one live recompute:

1. ``test_kernel_reproduces_recorded_alpha`` *(snapshot)* — feeds the
   fixture's stored widths through atomx's ``dr_rate_coefficient`` Burgess
   kernel and checks it still returns the number the fixture recorded.
   **This is a same-kernel round-trip, not a FAC parity test**: the
   fixture's ``alpha_fac_cm3_s`` was itself produced by combining FAC's
   AITable+TRTable widths through this kernel, so the prefactor cancels.
   The measured ratio is constant to six digits at every temperature
   (0.99970901, = the RYDBERG_EV**1.5 unit conversion), hence the 1e-3
   gate; the absolute scale is gated by
   ``tests/test_eii_ea.py::test_detailed_balance_capture_flux_is_shared_by_dr_and_re``
   against the Saha coefficient.

2. ``test_benchmark_csv_gate`` — gate the committed
   ``benchmarks/results_fac_dr.csv`` produced by
   ``benchmarks/bench_fac_dr.py`` (manifold pipeline vs the clean-room
   FAC dr33 reference ``tests/data/fe17_dr33_fac_cleanroom.json``:
   SetSE/VP/MS/Breit off, AICut 0).  Gates: rms |rel_diff| <= 0.35 and
   every T >= 5 MK point within alpha ratio [0.8, 1.35].  The 1-MK point
   (ratio ~0.49) is Boltzmann suppression from the documented ~+40 eV
   frozen-orbital E_res offset, not a width error — it dominates the rms
   and is why the target is 0.35 rather than the generic 0.20.

   The CSV was re-recorded on 2026-09-22 (both columns, same FAC
   fixture): its ``alpha_fac_cm3_s`` and ``alpha_atomx_cm3_s`` had been
   frozen on 2026-07-16, before the R7 asymmetric-mean-field fix and the
   DR prefactor fix.  Because *both* columns of this file are produced by
   the same kernel from the same external fixture, the prefactor change
   scales both by RYDBERG_EV**1.5 = 50.186 and leaves ``rel_diff``
   unchanged (0.2723 -> 0.2718); no external FAC number was touched.
   ``test_live_atomx_dr_matches_the_recorded_panel`` recomputes the atomx
   column here so a physics regression trips even if the CSV is stale.

3. ``test_dr_fixture_pair_discrepancy_is_documented`` — the two
   FAC-derived Fe XVII fixtures in ``tests/data/`` disagree by up to 3.9x
   at 1 MK.  They are different FAC runs (dr33+dr34+KLM with AICut 1e-4
   vs dr33 with AICut 0) and must not be mixed; this test pins the
   measured gap so the adjudication cannot silently change.

4. ``test_end_to_end_driver_returns_positive_alpha`` — smoke test of the
   hook path (external widths through the Burgess kernel).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from atomx.constants import RYDBERG_EV
from atomx.scattering.autoionization import (
    AutoionizationState,
    dr_calculate,
    dr_rate_coefficient,
)

FIXTURE_PATH = Path(__file__).parent / "data" / "fe17_dr_fac.json"
FIXTURE_CLEAN_PATH = Path(__file__).parent / "data" / "fe17_dr33_fac_cleanroom.json"
RESULTS_CSV = Path(__file__).parent.parent / "benchmarks" / "results_fac_dr.csv"
# Same-kernel round-trip reproducibility (NOT a parity claim): measured
# residual 1 - 0.99970901 = 2.9e-4 at every temperature, so 1e-3 is 3.4x
# the measurement (the old 1 % gate was 34x it).
KERNEL_TOLERANCE = 1.0e-3
RMS_GATE = 0.35          # dominated by the documented 1-MK E_res offset
RATIO_GATE = (0.8, 1.35)  # per-point gate for T >= 5 MK

# ``fe17_dr_fac.json::alpha_fac_cm3_s`` was produced by combining FAC's
# AITable+TRTable widths through this same Burgess kernel (see the fixture's
# ``source`` note), *before* the eV-convention prefactor fix: its values
# carry the Rydberg-temperature prefactor, RYDBERG_EV**1.5 = 50.186x below
# the eV convention now used by ``dr_rate_coefficient`` (review C5).  The
# comparison converts the fixture; the absolute scale is gated independently
# in tests/test_plasma_physics.py — a same-kernel reference cannot gate the
# prefactor itself.
FIXTURE_UNIT_FACTOR = RYDBERG_EV ** 1.5


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
            label=str(row.get("label", row.get("ilev", "?"))),
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

@pytest.mark.snapshot
def test_kernel_reproduces_recorded_alpha(fac_fixture):
    states = _fac_resonance_states(fac_fixture)
    T_eV = np.asarray(fac_fixture["T_eV"], dtype=float)
    rc = dr_rate_coefficient(
        states,
        T_eV,
        initial_state_g=fac_fixture.get("initial_state_g", 1.0),
        initial_state_label="Fe XVII 2p6 J=0 (FAC reference)",
    )
    expected = (np.asarray(fac_fixture["alpha_fac_cm3_s"], dtype=float)
                * FIXTURE_UNIT_FACTOR)
    np.testing.assert_allclose(
        rc.alpha_cm3_s, expected, rtol=KERNEL_TOLERANCE,
        err_msg=(
            "atomx Burgess kernel disagrees with FAC AITable+TRTable combine "
            "above 1 % — investigate dr_rate_coefficient prefactor or units."
        ),
    )


# ---------------------------------------------------------------------------
# 2.  The two FAC-derived Fe XVII fixtures are NOT interchangeable
# ---------------------------------------------------------------------------

@pytest.mark.snapshot
def test_dr_fixture_pair_discrepancy_is_documented():
    """The two FAC-derived Fe XVII fixtures differ by 1.8x-3.9x; pin the gap.

    ``fe17_dr_fac.json`` (194 resonances) is the older AITable+TRTable dump
    (dr33+dr34+KLM, AICut 1e-4); ``fe17_dr33_fac_cleanroom.json`` (236
    resonances) is the clean-room dr33 deck with SetSE/VP/MS/Breit off and
    AICut 0 that ``bench_fac_dr.py`` and ``test_dr_manifold_e2e.py`` use.
    Both were combined through the same Burgess kernel, so the pipeline is
    simultaneously "0.49 of FAC" (cleanroom) and "1.9x FAC" (old fixture) —
    they cannot both be FAC.  Adjudication: the cleanroom deck is the
    reference (AICut 0 keeps the weak resonances the 1e-4 cut discards, and
    its provenance is dated and complete); the median Auger width is 24x
    smaller there because most of the extra resonances are weak.

    Measured 2026-09-22 (this kernel): kernel(old)/kernel(cleanroom) =
    0.257 / 0.450 / 0.510 / 0.569 at 1 / 5 / 10 / 50 MK; median
    Gamma_auger 1.826e13 (old) vs 7.717e11 s^-1 (cleanroom).
    """
    old = json.loads(FIXTURE_PATH.read_text())
    clean = json.loads(FIXTURE_CLEAN_PATH.read_text())
    assert len(old["resonances"]) == 194
    assert len(clean["resonances"]) == 236
    T_eV = np.asarray(old["T_eV"], dtype=float)  # ambient grid: both fixtures
    k_old = dr_rate_coefficient(
        _fac_resonance_states(old), T_eV, initial_state_g=1.0).alpha_cm3_s
    k_clean = dr_rate_coefficient(
        _fac_resonance_states(clean), T_eV, initial_state_g=1.0).alpha_cm3_s
    ratio = np.asarray(k_old, dtype=float) / np.asarray(k_clean, dtype=float)
    np.testing.assert_allclose(ratio, [0.2569, 0.4499, 0.5099, 0.5693],
                               rtol=0.02)
    med_old = float(np.median([r["gamma_auger_per_s"]
                               for r in old["resonances"]]))
    med_clean = float(np.median([r["gamma_auger_per_s"]
                                 for r in clean["resonances"]]))
    assert med_old / med_clean == pytest.approx(23.66, rel=0.05)
    # the clean-room fixture is the one bench_fac_dr.py compares against
    from bench_fac_dr import FIXTURE_CLEAN  # noqa: E402
    assert FIXTURE_CLEAN == FIXTURE_CLEAN_PATH
    assert "AICut 0" in clean["provenance"]


# ---------------------------------------------------------------------------
# 3.  Benchmark CSV gate — manifold pipeline vs clean-room FAC dr33
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


@pytest.mark.external_fac
def test_live_atomx_dr_matches_the_recorded_panel():
    """Recompute the manifold pipeline live and gate it against the panel.

    Live-vs-recorded atomx <= 1e-4 (same code path as
    ``benchmarks/bench_fac_dr.py``, so this is reproducibility), and the live
    alpha_DR vs the recorded *external* FAC column within 10 % of the values
    measured on 2026-09-22 (0.486 / 0.978 / 1.074 / 1.158 at 1 / 5 / 10 /
    50 MK).  Before this test the CSV's own ``rel_diff`` column was the only
    gate, so deleting the manifold pipeline could not fail anything.
    """
    import bench_fac_dr

    data_rows, _ = _load_results_csv()
    rc = bench_fac_dr.run_atomx()
    live = np.asarray(rc.alpha_cm3_s, dtype=float)
    assert live.shape == (len(data_rows),)
    measured = {1.0: 0.486, 5.0: 0.978, 10.0: 1.074, 50.0: 1.158}
    rels = []
    for row, value in zip(data_rows, live, strict=True):
        T_mk = float(row["T_MK"])
        recorded = float(row["alpha_atomx_cm3_s"])
        drift = abs(value - recorded) / recorded
        assert drift <= 1.0e-4, (
            f"T = {T_mk} MK: live alpha_atomx = {value:.6e} vs recorded "
            f"{recorded:.6e} (rel={drift:.2e}) — regenerate the CSV"
        )
        fac = float(row["alpha_fac_cm3_s"])
        ratio = value / fac
        rels.append((value - fac) / fac)
        centre = measured[T_mk]
        assert abs(ratio - centre) <= 0.10 * centre, (
            f"T = {T_mk} MK: live alpha_atomx/alpha_FAC = {ratio:.3f}, "
            f"measured {centre:.3f} ± 10 %"
        )
    rms = float(np.sqrt(np.mean(np.square(rels))))
    assert rms <= RMS_GATE, f"live-vs-FAC rms {rms:.4f} exceeds {RMS_GATE}"


# ---------------------------------------------------------------------------
# 4.  Hook-path smoke test (external widths through the Burgess kernel)
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
