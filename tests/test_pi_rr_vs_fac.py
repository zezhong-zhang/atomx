"""Live gates for the FAC PI/RR panel (``benchmarks/results_fac_pi_rr.csv``).

The S5 audit found that the file named ``test_rr_vs_fac.py`` compared atomx
against an *internal* hydrogenic chain and that the real FAC panel produced by
``benchmarks/bench_fac_pi_rr.py`` was referenced by **zero** tests.  This
module is that missing gate.

Panel
-----
``results_fac_pi_rr.csv`` holds four RR series computed with sfac 1.1.5
(``RRTable``, ``T_e`` on FAC's own user grid), one row per (case, transition,
electron energy):

* ``Fe25+ -> Fe24+ (1s2)``      capture into He-like Fe XXIV 1s2
                                (free level = H-like Fe XXV 1s, 2J+1 = 2);
* ``Fe24+ -> Fe23+ (1s2 2s)``   capture into 2s1/2 / 2p1/2 / 2p3/2 of
                                Li-like Fe XXIII (free level = 1s2, 2J+1 = 1,
                                level-resolved; ILEV 7/8/9 from FAC's
                                ``.lev`` table).

Reproducibility note (measured 2026-09-22)
------------------------------------------
The bench's original call sequence could no longer be executed at all:
``Atom(engine=None)`` resolves to the *shooting* engine, which leaves
``SCFResult._bsp_data = None`` so ``rr_cross_section`` raises ValueError, and
the FAC residual-core continuum model is a shooting model on which the RR
kernel supports only the E1 multipole (NotImplementedError with the default
E1+M1+E2).  The driver now pins ``engine="bspline"`` and ``multipoles=("E1",)``
and refuses to overwrite the committed CSV when a case raises.  With that
configuration the live atomx column reproduces the regenerated recorded column
to <1e-9 and sits **1.5-1.6x above** the FAC RR cross section over
1 eV - 16 keV (measured rel_diff +0.52 ... +0.59), i.e. the panel is a live
external comparison again, not a frozen one.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from bench_fac_pi_rr import (  # noqa: E402
    A0_CM2_X1E20,
    FAC_CONTINUUM,
)

from atomx import Atom  # noqa: E402
from atomx.scattering.radiative_recombination import rr_cross_section  # noqa: E402

CSV_PATH = ROOT / "benchmarks" / "results_fac_pi_rr.csv"

# FAC free-level degeneracies 2J+1 of the recombining ion (FAC's printed RR
# cross section is divided by this, see faclib/dbase.c:PrintRRTable).
G_INITIAL = {
    "Fe25+ -> Fe24+(1s²) RR": 2.0,   # H-like Fe XXV 1s (2J+1 = 2)
    "Fe24+ -> Fe23+(1s²2s) RR": 1.0,  # He-like Fe XXIV 1s2 (2J+1 = 1)
}
# Li-like bound ILEV -> capture orbital (bench_fac_pi_rr._capture_target).
LI_CAPTURE = {7: (["2s1/2"], "1s2 2s1"), 8: (["2p1/2"], "1s2 2p1"),
              9: (["2p3/2"], "1s2 2p1")}
# Active per-series gate on |sigma_atomx/sigma_FAC - 1| for the Li-like
# captures (measured maxima 2026-09-22: 0.577 / 0.233 / 0.259 for ILEV 7/8/9;
# the gate sits 5-10 % above the measurement).
FAC_ENVELOPE_BY_ILEV = {7: 0.62, 8: 0.28, 9: 0.30}
# The He-like K-shell series is NOT inside any usable band: measured
# atomx/FAC - 1 = +0.587 / +1.137 / +2.022 / +3.521 at 1 / 15 / 70 / 200 eV.
K_SHELL_MEASURED = {1.0: 0.587, 15.116: 1.137, 70.291: 2.022, 200.0: 3.521}
# The parity claim the panel does NOT meet (see the strict-xfail test below).
FAC_PARITY = 0.15


def _load_panel() -> list[dict]:
    if not CSV_PATH.exists():
        pytest.skip(f"FAC PI/RR panel missing: {CSV_PATH}")
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, f"{CSV_PATH} has no rows — the panel was truncated"
    for row in rows:
        row["E_eV"] = float(row["E_eV"])
        row["sigma_atomx_1e20cm2"] = float(row["sigma_atomx_1e20cm2"])
        row["sigma_fac_1e20cm2"] = float(row["sigma_fac_1e20cm2"])
        row["rel_diff"] = float(row["rel_diff"])
        row["ilev"] = int(row["transition"].rsplit("->", 1)[1])
    return rows


@pytest.fixture(scope="module")
def panel() -> list[dict]:
    return _load_panel()


@pytest.fixture(scope="module")
def live_atomx() -> dict[tuple[str, int, float], float]:
    """Recompute the atomx column exactly as ``bench_fac_pi_rr`` now does."""
    out: dict[tuple[str, int, float], float] = {}
    he = Atom(Z=26, config="1s2", corrections="none", engine="bspline")
    he.solve()
    li = {
        "1s2 2s1": Atom(Z=26, config="1s2 2s1", corrections="none",
                        engine="bspline"),
        "1s2 2p1": Atom(Z=26, config="1s2 2p1", corrections="none",
                        engine="bspline"),
    }
    for atom in li.values():
        atom.solve()
    for row in _load_panel():
        case, ilev, e_eV = row["case"], row["ilev"], row["E_eV"]
        if case.startswith("Fe25+"):
            res, capture = he._result, None
        else:
            orbitals, config = LI_CAPTURE[ilev]
            res, capture = li[config]._result, orbitals
        rr = rr_cross_section(
            res, np.array([e_eV]), g_initial=G_INITIAL[case],
            capture_orbitals=capture, continuum_model=FAC_CONTINUUM,
            multipoles=("E1",),
        )
        out[(case, ilev, e_eV)] = float(rr.sigma_total_a0_sq[0]) * A0_CM2_X1E20
    return out


@pytest.mark.external_fac
def test_panel_records_the_external_fac_column(panel):
    """Structure + self-consistency of the stored FAC panel (no atomx)."""
    assert len(panel) == 24
    assert {row["case"] for row in panel} == set(G_INITIAL)
    for row in panel:
        assert row["sigma_fac_1e20cm2"] > 0.0, row
        assert row["sigma_atomx_1e20cm2"] > 0.0, row
        rel = (row["sigma_atomx_1e20cm2"] / row["sigma_fac_1e20cm2"] - 1.0)
        assert rel == pytest.approx(row["rel_diff"], abs=1.0e-9), (
            f"{row['case']} {row['transition']} E={row['E_eV']}: recorded "
            f"rel_diff {row['rel_diff']} does not equal the ratio of the "
            f"recorded columns ({rel}) — the CSV was hand-edited"
        )
    # the worst point of the panel is the 200 eV He-like K-shell row, and its
    # recorded value is the measured one (see K_SHELL_MEASURED below)
    worst = max(panel, key=lambda r: abs(r["rel_diff"]))
    assert worst["case"].startswith("Fe25+") and worst["E_eV"] == 200.0, worst
    assert worst["rel_diff"] == pytest.approx(K_SHELL_MEASURED[200.0], abs=2e-3)


@pytest.mark.snapshot
def test_live_rr_reproduces_the_recorded_atomx_column(panel, live_atomx):
    """Live atomx (bench settings) vs the recorded atomx column, 1e-3.

    ``snapshot``: this side is atomx's own recorded output.  Measured
    2026-09-22: worst relative drift 1.2e-9.
    """
    worst = 0.0
    for row in panel:
        key = (row["case"], row["ilev"], row["E_eV"])
        live = live_atomx[key]
        rel = abs(live - row["sigma_atomx_1e20cm2"]) / row["sigma_atomx_1e20cm2"]
        worst = max(worst, rel)
        assert rel <= 1.0e-3, (
            f"{key}: live sigma_RR={live:.6e} vs recorded "
            f"{row['sigma_atomx_1e20cm2']:.6e} (rel={rel:.2e}) — the RR kernel "
            "or the panel's engine/multipole settings changed"
        )
    assert worst <= 1.0e-3


@pytest.mark.external_fac
def test_live_li_like_rr_against_fac_is_inside_the_measured_envelope(
    panel, live_atomx
):
    """LIVE Li-like RR vs the stored FAC column: +0.58 / +0.23 / +0.26.

    The three capture levels (2s1/2, 2p1/2, 2p3/2 of Fe XXIII) are 50-60 %
    (2s) and 23-26 % (2p) ABOVE the FAC RRTable value with the currently
    runnable E1-only B-spline path — a constant scale offset across
    50 eV - 16 keV, not a threshold effect.  The gate documents that state
    with 5-10 % headroom.
    """
    for row in panel:
        if row["case"].startswith("Fe25+"):
            continue
        key = (row["case"], row["ilev"], row["E_eV"])
        rel = live_atomx[key] / row["sigma_fac_1e20cm2"] - 1.0
        limit = FAC_ENVELOPE_BY_ILEV[row["ilev"]]
        assert abs(rel) <= limit, (
            f"{key}: live atomx/FAC - 1 = {rel:+.3f} outside ±{limit} "
            f"(atomx={live_atomx[key]:.6e}, FAC={row['sigma_fac_1e20cm2']:.6e})"
        )


@pytest.mark.external_fac
@pytest.mark.xfail(
    strict=True,
    reason=(
        "MEASURED live atomx/FAC - 1 for the He-like Fe25+ -> Fe24+ 1s2 K-shell "
        "series = +0.587 / +1.137 / +1.301 / +2.022 / +2.628 / +3.521 at "
        "1 / 15.1 / 36.9 / 70.3 / 121.6 / 200 eV (2026-09-22), i.e. atomx is "
        "1.6x to 4.5x the FAC RRTable cross section and the discrepancy GROWS "
        "with energy.  This is the only currently runnable configuration "
        "(engine='bspline' + multipoles=('E1',): the FAC residual-core "
        "continuum is a shooting model on which the RR kernel supports only "
        "E1, and Atom(engine=None) now resolves to shooting, which leaves "
        "_bsp_data = None and raises ValueError).  A growing ratio points at "
        "the 1s bound-free dipole matrix element / continuum normalisation "
        "rather than at the RR prefactor; closing it needs the K-shell RR "
        "kernel fixed against the FAC RRTable, not a wider band."
    ),
)
def test_live_k_shell_rr_agrees_with_fac_to_15_percent(panel, live_atomx):
    """The parity claim the K-shell series should support: |atomx/FAC-1| <= 15 %."""
    for row in panel:
        if not row["case"].startswith("Fe25+"):
            continue
        key = (row["case"], row["ilev"], row["E_eV"])
        rel = live_atomx[key] / row["sigma_fac_1e20cm2"] - 1.0
        assert abs(rel) <= FAC_PARITY, (
            f"{key}: |atomx/FAC - 1| = {abs(rel):.3f} exceeds {FAC_PARITY:.0%}"
        )
