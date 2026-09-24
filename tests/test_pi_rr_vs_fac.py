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

Validation update (2026-09-23)
------------------------------
The continuum uses native-grid Hartree quadrature and keeps the nuclear
field separate from interpolated screening. The residual-core exchange
model no longer removes a second electron's self-field. Each Li-like
final level has its own occupied jj orbital. The external FAC column and
accuracy bands are unchanged: K-shell errors are about 4%, Li-like 2s
errors up to about 12%, and 2p errors up to about 23% on this panel after
the exact Dirac-derivative continuum normalization. These are different
local-exchange/HF models, so this comparison is not an exact identity.
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
    atomx_li_capture_results,
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
# Existing external accuracy bands, preserved through the kernel repairs.
FAC_ENVELOPE_BY_ILEV = {7: 0.62, 8: 0.28, 9: 0.30}
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
    li = atomx_li_capture_results()
    for row in _load_panel():
        case, ilev, e_eV = row["case"], row["ilev"], row["E_eV"]
        if case.startswith("Fe25+"):
            res, capture = he._result, None
        else:
            orbitals, _ = LI_CAPTURE[ilev]
            res, capture = li[{7: -1, 8: 1, 9: -2}[ilev]], orbitals
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


@pytest.mark.snapshot
def test_live_rr_reproduces_the_recorded_atomx_column(panel, live_atomx):
    """Self-snapshot gate for the corrected E1-only residual-core calculation."""
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
    """Live, separately optimized final levels vs the unchanged FAC accuracy bands."""
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
