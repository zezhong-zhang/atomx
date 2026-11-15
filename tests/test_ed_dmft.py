"""ED (d + one bath per cubic orbital) impurity solver: exact anchors.

B1 bath decoupled (V = 0): G_imp equals the independent Hubbard-I (atomic, spherical-basis)
   solver pole for pole — checks the spherical->cubic rotation of the Slater tensor and the
   bit-string ED against an independent implementation.
B2 U = 0: Sigma == 0 identically (non-interacting Anderson model).
B3 pole bookkeeping: G_imp == 1/(z - eps - Delta(z) - Sigma(z)) at arbitrary z with the
   bath pole of Delta explicitly cancelled; Sigma residues positive (causal).
"""

import numpy as np
import pytest

from atomx.embedding import dmft
from atomx.embedding import ed_dmft as ed


@pytest.fixture(scope="module")
def tab8():
    return ed.interaction_table(ed.coulomb_cubic(dmft.slater_from_UJ(8.0, 1.0)))


def _G(poles, z):
    p, r = poles
    return (r[None, :] / (z[:, None] - p[None, :])).sum(1)


def test_B1_decoupled_bath_equals_hubbard_I(tab8):
    sl = dmft.slater_from_UJ(8.0, 1.0)
    eps = np.array([1.3, 0, 0, 1.3, 0]) - 56.0
    sol = ed.solve_ed(eps, np.full(5, -200.0), np.zeros(5), sl, -4.0, table=tab8)
    hi = dmft.solve_hubbard_I(eps, sl, -4.0)
    z = np.array([0.3 + 0.1j, -7 + 0.2j, 3 + 0.05j, 20 + 1j])
    for a in range(5):
        assert np.allclose(_G(sol.poles[a], z), _G(hi.poles[a], z), atol=1e-8)


def test_B2_noninteracting_sigma_vanishes():
    tab0 = ed.interaction_table(ed.coulomb_cubic(dict(F0=0.0, F2=0.0, F4=0.0)))
    sol = ed.solve_ed(
        np.array([-1.0, -2, -2, 1, -2]),
        np.array([-4.0, -5, -5, -4, -5]),
        np.array([2.0, 1, 1, 2, 1]),
        dict(F0=0.0, F2=0.0, F4=0.0),
        0.0,
        table=tab0,
    )
    assert np.abs(sol.sigma_inf).max() < 1e-9
    for _zs, R in sol.sigma_poles:
        assert len(R) == 0 or R.max() < 1e-9


def test_B3_dyson_identity_with_bath(tab8):
    sl = dmft.slater_from_UJ(8.0, 1.0)
    eps = np.array([1.3, 0, 0, 1.3, 0]) - 56.0 + 0.0
    eb = np.array([-60.0, -61.0, -61.0, -60.0, -61.0])
    Vb = np.array([2.0, 1.0, 1.0, 2.0, 1.0])
    sol = ed.solve_ed(eps, eb, Vb, sl, -4.0, table=tab8)
    z = np.array([0.3 + 0.1j, -7 + 0.2j, 3 + 0.05j, -60.5 + 0.3j])
    for a in range(5):
        zs, R = sol.sigma_poles[a]
        assert (R > 0).all()
        Sig = sol.sigma_inf[a] + (R[None, :] / (z[:, None] - zs[None, :])).sum(1)
        Delta = Vb[a] ** 2 / (z - eb[a])
        assert np.allclose(
            _G(sol.poles[a], z), 1.0 / (z - eps[a] - Delta - Sig), atol=1e-8
        )


def test_B4_projected_bath_recovers_ligand_and_ignores_high_energy():
    """Synthetic Delta = V1^2/(z-e1) + V2^2/(z-e2) with a ligand pole e1 inside the window and a
    large high-energy pole e2 outside it: the projection must return V1, e1 (the Matsubara fit
    is pulled toward the dominant high-energy weight, which is the failure mode it replaces)."""
    mu = 0.0
    e1, v1, e2, v2 = -4.0, 1.2, 16.0, 2.2

    def delta(z):
        z = np.atleast_1d(z)
        d = v1**2 / (z - e1) + v2**2 / (z - e2)
        return np.tile(d, (5, 1))

    eb, V, c = ed.project_bath(delta, mu, window=(-10.0, 0.0), eta=0.05, dw=0.005)
    assert np.allclose(V, v1, rtol=0.02)
    assert np.allclose(eb, e1, atol=0.02)
    # static remainder = high-energy pole seen at the reference energy
    assert np.allclose(c, np.real(v2**2 / (0.5j - e2)), atol=0.05)
