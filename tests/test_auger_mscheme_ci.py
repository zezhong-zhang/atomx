"""Determinant-CI Auger widths vs FAC (Fe XVII KLL, 2p5 3d2 manifold).

Validates :mod:`atomx.scattering.autoionization.mscheme_ci`:

- the manifold trace (mixing-invariant sum of Auger widths over all
  J=7/2 levels of 2p^5 3d^2) matches FAC's AITable sum, and
- the strongest level's width matches FAC's strongest J=7/2 level,

both with the FAC-like orbital basis (ground-config solve + frozen-core
virtuals).  The trace test is the physics gate: it is independent of CI
mixing, so it pins the amplitude machinery (Slater-Condon signs, hole
conjugation, 2*pi golden-rule constant, delta(E) continuum) in
isolation.  FAC reference: tests/data/fe17_dr_fac.json
(AITable, Fe XVII 2p5 3d2 -> 2p6 + eps).

Reference-subset definition (2026-09-17 validation-integrity pass)
-----------------------------------------------------------------
The reference numbers are now **derived in-test from the fixture** instead of
being opaque literals.  The subset is exactly the manifold under test: FAC
levels whose label contains ``2p5.3d2`` with J = 7/2 (five levels:
ILEV 162/166/170/179/190; trace 5.35668e14 s^-1, strongest 3.0806e14 s^-1).
The previously unstated alternative — all J = 7/2 fixture rows containing
``3*2`` (12 levels, which additionally includes the 2s-hole ``2s1 2p6 3d2``
manifold that this test does NOT build) — sums to 6.40504e14 s^-1, 19.6 %
higher; ``test_fac_reference_subset_is_the_2p5_3d2_manifold`` pins both so the
distinction can never drift again.  Measured atomx/fixture: trace 0.893,
strongest level 0.958.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.constants import HARTREE_EV  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "fe17_dr_fac.json"

# The two candidate reference subsets, selected by label + two_J.  Both are
# recomputed from the fixture in ``fac_j72``; the literals here only document
# the values seen on 2026-09-17.
MANIFOLD_LABEL = "2p5.3d2"   # the 2p^5 3d^2 configuration this test builds
BROADER_LABEL = "3*2"        # any two captured n=3 electrons (includes 2s holes)
TWO_J72 = 7


def _j72_rows(data: dict, label_token: str) -> list[dict]:
    return [
        row
        for row in data["resonances"]
        if float(row["J"]) == TWO_J72 / 2.0 and label_token in row["label"]
    ]


@pytest.fixture(scope="module")
def fac_j72() -> dict:
    """FAC AITable J=7/2 reference, derived from the fixture by definition."""
    if not FIXTURE.exists():
        pytest.skip(f"FAC AITable fixture missing: {FIXTURE}")
    data = json.loads(FIXTURE.read_text())
    manifold = _j72_rows(data, MANIFOLD_LABEL)
    broader = _j72_rows(data, BROADER_LABEL)
    if not manifold:
        pytest.skip("fixture carries no 2p5.3d2 J=7/2 levels — regenerate it")
    return {
        "manifold_rows": manifold,
        "trace": sum(float(r["gamma_auger_per_s"]) for r in manifold),
        "top": max(float(r["gamma_auger_per_s"]) for r in manifold),
        "broader_rows": broader,
        "broader_trace": sum(float(r["gamma_auger_per_s"]) for r in broader),
    }


@pytest.fixture(scope="module")
def manifold_j72():
    from atomx.methods.mbpt import build_virtual_basis
    from atomx.numerics.bspline.evaluate import eval_P, eval_Q
    from atomx.numerics.quadrature import make_fine_grid, p_rel
    from atomx.scattering.autoionization.mscheme_ci import ResonanceManifold

    g = Atom(Z=26, charge=16, configs=["1s2 2s2 2p6"], breit=False, qed=False)
    g.solve()
    res = g._result
    bsp = getattr(res, "_bsp_data", None) or res.bsp_data
    p_max = p_rel(500.0 / HARTREE_EV)
    r_fine = make_fine_grid(float(bsp["t"][-1]), p_max)
    dr_fine = np.gradient(r_fine)
    gr = bsp["gr"].ravel()
    order = np.argsort(gr)
    grs = gr[order]

    def tofine(P, Q):
        Pg = eval_P(P, bsp).ravel()[order]
        Qg = eval_Q(Q, bsp).ravel()[order]
        return np.interp(r_fine, grs, Pg), np.interp(r_fine, grs, Qg)

    orbital_map = {}
    orbital_eps = {}
    for kappa, lab in ((1, "2p1/2"), (-2, "2p3/2")):
        P, Q, _ = res.orbitals_raw[kappa][0]
        orbital_map[(2, kappa)] = tofine(P, Q)
        orbital_eps[(2, kappa)] = float(res.orbital_energies[lab])
    for kappa in (2, -3):
        _occ, virt = build_virtual_basis(res, kappa)
        P, Q, e = virt[0]
        orbital_map[(3, kappa)] = tofine(P, Q)
        orbital_eps[(3, kappa)] = float(e)

    groups = [([(2, 1), (2, -2)], 5), ([(3, 2), (3, -3)], 2)]
    M = ResonanceManifold(res, groups, r_fine=r_fine, dr_fine=dr_fine,
                          orbital_map=orbital_map,
                          orbital_energies=orbital_eps)
    levels = M.auger_widths(7, 409.0, Z=26, bsp_data=bsp)
    return levels


@pytest.mark.external_fac
def test_fac_reference_subset_is_the_2p5_3d2_manifold(fac_j72):
    """Pin which fixture subset is the reference (and which is not).

    The manifold under test is 2p^5 3d^2 J=7/2: five AITable levels, trace
    5.35668e14 s^-1.  The broader "all J=7/2 rows with two captured n=3
    electrons" subset (12 levels, adding the 2s-hole 2s1 2p6 3d2 manifold)
    sums to 6.40504e14 s^-1 and must NOT be used as the reference.
    """
    assert len(fac_j72["manifold_rows"]) == 5
    assert all(
        MANIFOLD_LABEL in row["label"] and float(row["J"]) == 3.5
        for row in fac_j72["manifold_rows"]
    )
    assert fac_j72["trace"] == pytest.approx(5.35668e14, rel=1e-5)
    assert fac_j72["top"] == pytest.approx(3.0806e14, rel=1e-5)
    # The confusable broader subset, recorded explicitly.
    assert len(fac_j72["broader_rows"]) == 12
    assert fac_j72["broader_trace"] == pytest.approx(6.40504e14, rel=1e-5)
    assert fac_j72["broader_trace"] / fac_j72["trace"] == pytest.approx(1.196, rel=2e-3)


@pytest.mark.external_fac
def test_manifold_dimension(manifold_j72):
    # 2p^5 3d^2 supports exactly 9 J=7/2 basis states.
    assert len(manifold_j72) == 9


@pytest.mark.external_fac
def test_trace_matches_fac(manifold_j72, fac_j72):
    """Mixing-invariant manifold trace vs the fixture's 2p5.3d2 J=7/2 sum.

    Gate 20 % (measured 2026-09-17: atomx 4.785e14 / fixture 5.357e14 = 0.893).
    The old hardcoded 5.357e14 was correct for this subset but unlabelled, so
    the 19.6 %-larger broader subset was indistinguishable inside a 20 % band.
    """
    trace = sum(L.gamma_auger_per_s for L in manifold_j72)
    assert trace == pytest.approx(fac_j72["trace"], rel=0.20)


@pytest.mark.external_fac
def test_strongest_level_matches_fac(manifold_j72, fac_j72):
    """Strongest J=7/2 level width vs the fixture's strongest (<= 35 %).

    Measured 2026-09-17: atomx 2.950e14 / fixture 3.081e14 = 0.958.  The
    residual is the orbital-basis difference between atomx's V^{N-1} virtuals
    and FAC's optimized DFS potential, plus dr34/2s-hole mixing outside the
    manifold.
    """
    top = max(L.gamma_auger_per_s for L in manifold_j72)
    assert top == pytest.approx(fac_j72["top"], rel=0.35)


@pytest.mark.external_fac
def test_level_energy_spread(manifold_j72):
    """The manifold spans ~18 eV in FAC; atomx within a factor ~1.6."""
    spread = max(L.energy_au for L in manifold_j72) * HARTREE_EV
    assert 10.0 <= spread <= 32.0
