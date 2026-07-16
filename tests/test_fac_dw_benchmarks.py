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
def test_dw_excitation_benchmark_structure_and_live_fac():
    """Pin the DW-excitation benchmark plumbing against LIVE FAC.

    The previous frozen CSV asserted a fabricated ~4 % parity that matched
    *neither* code.  ``bench_fac_dw_excitation.py`` now runs sfac live and
    sums FAC's He-like 1s2 -> 1s2p collision strengths over the multiplet at
    matched E/threshold points.  This test pins the grid coverage and that
    live FAC produced positive collision strengths; the physics parity gate
    is the test below.
    """
    rows = _rows(EXCITATION_CSV)
    data = [r for r in rows if r["Z"] != "rms"]
    assert {(int(r["Z"]), float(r["E_over_threshold"])) for r in data} == {
        (Z, f) for Z in (10, 26, 54) for f in (1.05, 2.0, 5.0)
    }
    assert all(int(r["atomx_converged"]) == 1 for r in data)
    assert all(float(r["omega_fac"]) > 0.0 for r in data)


@pytest.mark.fac_parity
def test_dw_excitation_fac_rms_gate():
    """DW excitation parity vs LIVE FAC: rms <= 8%, per-point <= 18%.

    The 2026-07-14 rewrite made ``dw_excitation_cross_section`` reproduce
    FAC's relativistic DW physics exactly: Omega = 2*pi^2 * sum over
    channels of (2J+1)|D_J - E_J|^2 with delta(E)-normalized shooting
    continuum waves in FAC's (N-1)/N mean potential, exact m-summation
    angular coefficients (direct + exchange), all jj destination pairs,
    and a geometric high-l tail.  Validated per-partial-wave against
    FAC's SetCEPWFile dump (H-like 1s->2p, 1s->3d) and per-level against
    FAC CE tables; the He-like panel (Z=10/26/54, E/th=1.05/2/5,
    multiplet-summed) agrees to rms ~6%.
    """
    rows = _rows(EXCITATION_CSV)
    data = [r for r in rows if r["Z"] != "rms"]
    assert all(int(r["atomx_converged"]) == 1 for r in data)
    assert all(float(r["omega_atomx"]) > 0.0 for r in data)
    assert all(float(r["omega_fac"]) > 0.0 for r in data)
    # 2026-07-16: with Grant's relativistic delta(E) normalization
    # (A^2 = (E+2c^2)/(pi c^2 k_rel); the old sqrt(2/(pi k_nonrel))
    # target was low by sqrt(1+E/2c^2) per wave — 9% at 200 keV) the
    # panel rms is 3.7% and every point is within 6.7% (Z=54 E/th=5
    # went -11% -> +1.6%).
    assert max(abs(float(r["rel_diff"])) for r in data) <= 0.10
    assert _rms(rows) <= 0.05


@pytest.mark.fac_parity
def test_dw_eii_fac_rms_gate():
    """DW EII parity vs LIVE FAC in pure-DW CI mode (SetCIQkMode('dw')).

    The 2026-07-14 rewrite put ``_eii_dw_total`` on the same FAC-exact
    machinery as DW excitation (delta(E)-normalized shooting continuum in
    the FAC (N-1)/N mean potential, exact m-summation angular
    coefficients, T_J = D_J - E_J, sigma = pi/p^2 * q_a/(2j_a+1) *
    Int dE1 2*pi^2 Sum (2J+1)|T_J|^2).  The old box-continuum kernel
    swung 4 orders of magnitude across 200 eV; the old frozen CSV columns
    were fabricated (matched neither code), and the old FAC driver passed
    absolute energies into a threshold-relative grid.

    2026-07-15: two reference/physics fixes.  (1) The FAC user grid is
    now pinned at the requested factors — the old run let FAC pick its
    sparse default grid and interpolated across the steep threshold
    rise, mis-stating sigma_fac(f=1.5) by -40% for He-like Fe.  (2) The
    atomx ejected-energy window is now FAC-equivalent ([~1.5e-3 T, T/2],
    log-spaced low nodes): the old 0.05T..0.49T window missed the
    dominant soft-ejection region (30-45% of the integral at high E) and
    partially cancelled against a geometric-tail overshoot.

    2026-07-16: (3) the FAC reference uses EXACT (mode-1) Slater
    integrals (SetCIPWOptions qr=36) — FAC's default phase-amplitude
    mode is 4-14% low (Z=26..54) while atomx is exact; (4) Grant's
    relativistic delta(E) wave normalization fixed the high-energy
    deficit; (5) the INTERNAL grid is pinned (SetCIEGrid): FAC computes
    DW only on its internal egrid and evaluates user points through the
    4-parameter CIRadialQkFromFit form, which mis-stated
    sigma(f=1.5, Z=54) by -44% — the entire apparent "near-threshold
    interference excess" (+83%) was that fit artifact.  Direct-to-direct
    with converged settings (kappa_max=10, 25 ejected-energy nodes):
    rms 2.0%, every point within 3.1% at all Z from threshold to 5x.
    """
    rows = _rows(EII_CSV)
    data = [r for r in rows if r["Z"] != "rms"]
    assert {(int(r["Z"]), float(r["E_over_threshold"])) for r in data} == {
        (Z, f) for Z in (10, 26, 54) for f in (1.5, 2.0, 5.0)
    }
    assert all(int(r["atomx_converged"]) == 1 for r in data)
    assert all(float(r["sigma_atomx_cm2"]) > 0.0 for r in data)
    assert all(float(r["sigma_fac_cm2"]) > 0.0 for r in data)
    assert max(abs(float(r["rel_diff"])) for r in data) <= 0.06
    assert _rms(rows) <= 0.035
