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
from atomx.scattering.autoionization.mscheme_ci import (  # noqa: E402
    AugerContinuumError,
    ResonanceManifold,
)


FIXTURE = Path(__file__).parent / "data" / "fe17_dr_fac.json"

# FAC AITable sums for 2p5.3d2 J=7/2 (from the fixture; FAC lists levels
# above its AI cut, so the reference trace is a slight lower bound).
FAC_TRACE_J72 = 5.357e14
FAC_TOP_J72 = 3.081e14


def test_manifold_auger_widths_surface_continuum_failure(monkeypatch):
    import atomx.scattering.autoionization.mscheme_ci as mscheme_ci

    result = type("Result", (), {"subshells": []})()
    manifold = ResonanceManifold.__new__(ResonanceManifold)
    manifold.result = result
    manifold.r_fine = np.array([0.0, 1.0])
    manifold.group_bits = [1]
    manifold.V_func = lambda r: np.zeros_like(np.asarray(r, dtype=float))

    monkeypatch.setattr(manifold, "determinants", lambda _tM: [0])
    monkeypatch.setattr(manifold, "hamiltonian", lambda _dets: np.array([[0.0]]))
    monkeypatch.setattr(manifold, "j2_matrix", lambda _dets: np.array([[0.75]]))
    monkeypatch.setattr(manifold, "parity_of_det", lambda _det: 1)
    monkeypatch.setattr(
        mscheme_ci,
        "_continuum_on_grid",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unit failure")),
    )

    with pytest.raises(AugerContinuumError, match="Auger continuum failed"):
        manifold.auger_widths(
            1,
            10.0,
            Z=1,
            bsp_data={"t": np.array([1.0])},
            shooting_n_grid=3000,
        )


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


@pytest.mark.fac_parity
def test_manifold_dimension(manifold_j72):
    # 2p^5 3d^2 supports exactly 9 J=7/2 basis states.
    assert len(manifold_j72) == 9


@pytest.mark.fac_parity
def test_trace_matches_fac(manifold_j72):
    """Mixing-invariant manifold trace vs FAC AITable sum (<= 20%)."""
    trace = sum(L.gamma_auger_per_s for L in manifold_j72)
    assert trace == pytest.approx(FAC_TRACE_J72, rel=0.20)


@pytest.mark.fac_parity
def test_strongest_level_matches_fac(manifold_j72):
    """Strongest J=7/2 level width vs FAC (<= 35%; the residual is the
    orbital-basis difference between atomx's V^{N-1} virtuals and FAC's
    optimized DFS potential, plus dr34/2s-hole mixing outside the
    manifold)."""
    top = max(L.gamma_auger_per_s for L in manifold_j72)
    assert top == pytest.approx(FAC_TOP_J72, rel=0.35)


@pytest.mark.fac_parity
def test_level_energy_spread(manifold_j72):
    """The manifold spans ~18 eV in FAC; atomx within a factor ~1.6."""
    spread = max(L.energy_au for L in manifold_j72) * HARTREE_EV
    assert 10.0 <= spread <= 32.0
