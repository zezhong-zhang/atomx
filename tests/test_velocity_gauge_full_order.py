"""Full-order velocity gauge for EL (L >= 1) — Grant 2007 §8.4 validation.

Exercises the derivative-free velocity-gauge radial integral
:func:`~atomx.properties.transitions.multipole.radial_eL_velocity` for all
multipole ranks L = 1…4, covering every (κ_i, κ_f) channel accessible in
hydrogenic Dirac–Coulomb states.  For exact eigenstates of the common
Hamiltonian the velocity gauge must reproduce the length gauge to numerical
precision.  The tolerance is per multipole rank (measured at the default
ns=60/ks=9, 2026-09-25):

    L=1  |A_v/A_l − 1| ≤ 4.2e-10   -> gate 1e-8   (worst: p_{3/2}→d_{3/2})
    L=2  |A_v/A_l − 1| ≤ 2.9e-10   -> gate 1e-8
    L=3  |A_v/A_l − 1| ≤ 4.3e-07   -> gate 1e-5
    L=4  |A_v/A_l − 1| ≤ 3.7e-04   -> gate 1e-3

The old file used a flat 1e-2, which is 7 orders looser than the theorem
holds at L=1.  The J-dependence is the O((ω r)²) truncation of the
long-wavelength reduction, so the gate grows with L.
"""

from __future__ import annotations

import pytest

from atomx.numerics.analytic.hydrogenic_dirac import hydrogenic_dirac_wavefunction
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.multipole import radial_eL, radial_eL_velocity
from atomx.properties.transitions.rates import einstein_A_eL

#: Per-rank tolerance for |A_v/A_l - 1| (see the module docstring).
GAUGE_TOL = {1: 1e-8, 2: 1e-8, 3: 1e-5, 4: 1e-3}


def _gauge_ratio(n_lo: int, k_lo: int, n_up: int, k_up: int,
                 Z: int, L: int = 1) -> float:
    P_lo, Q_lo, bsp, E_lo = hydrogenic_dirac_wavefunction(n_lo, k_lo, Z)
    P_up, Q_up, _, E_up = hydrogenic_dirac_wavefunction(n_up, k_up, Z)

    omega = abs(E_up - E_lo)
    rme_l = radial_eL(P_up, Q_up, P_lo, Q_lo, bsp, L)
    rme_v = radial_eL_velocity(P_up, Q_up, P_lo, Q_lo, bsp, L,
                                kappa_i=k_up, kappa_f=k_lo, omega=omega)
    ang = angular_eL(k_up, k_lo, L)
    tj_up = 2 * abs(k_up) - 1

    S_l = (ang * rme_l) ** 2
    S_v = (ang * rme_v) ** 2
    A_l = einstein_A_eL(omega, S_l, L, tj_up)
    A_v = einstein_A_eL(omega, S_v, L, tj_up)
    dev = abs(A_v / A_l - 1.0)
    tol = GAUGE_TOL[L]
    assert dev < tol, (
        f"|A_v/A_l - 1| = {dev:.3e} exceeds {tol:.0e} for "
        f"(n,kappa) {n_lo},{k_lo} -> {n_up},{k_up} at Z={Z}, L={L}")
    return dev


# ---------------------------------------------------------------------------
# 1. E1 (L = 1) — all four channels: s↔p_{1/2}, s↔p_{3/2}
# ---------------------------------------------------------------------------

class TestE1FullOrder:

    @pytest.mark.parametrize("Z", [1, 2, 5, 8])
    def test_s_to_p12(self, Z):
        _gauge_ratio(1, -1, 2, 1, Z, L=1)

    @pytest.mark.parametrize("Z", [1, 2, 5, 8])
    def test_s_to_p32(self, Z):
        _gauge_ratio(1, -1, 2, -2, Z, L=1)

    @pytest.mark.parametrize("Z", [1, 2])
    def test_p12_to_d32(self, Z):
        _gauge_ratio(2, 1, 3, 2, Z, L=1)

    @pytest.mark.parametrize("Z", [1, 2])
    def test_p32_to_d32(self, Z):
        _gauge_ratio(2, -2, 3, 2, Z, L=1)

    @pytest.mark.parametrize("Z", [1, 2])
    def test_p32_to_d52(self, Z):
        _gauge_ratio(2, -2, 3, -3, Z, L=1)


# ---------------------------------------------------------------------------
# 2. E2 (L = 2)
# ---------------------------------------------------------------------------

class TestE2FullOrder:

    @pytest.mark.parametrize("Z", [1, 2])
    def test_s_to_d32(self, Z):
        _gauge_ratio(1, -1, 3, 2, Z, L=2)

    @pytest.mark.parametrize("Z", [1, 2])
    def test_s_to_d52(self, Z):
        _gauge_ratio(1, -1, 3, -3, Z, L=2)

    @pytest.mark.parametrize("Z", [1, 2])
    def test_p32_to_f52(self, Z):
        _gauge_ratio(2, -2, 4, 3, Z, L=2)

    @pytest.mark.parametrize("Z", [1, 2])
    def test_p32_to_f72(self, Z):
        _gauge_ratio(2, -2, 4, -4, Z, L=2)


# ---------------------------------------------------------------------------
# 3. E3 (L = 3)
# ---------------------------------------------------------------------------

class TestE3FullOrder:

    def test_s_to_f52(self):
        _gauge_ratio(1, -1, 4, 3, 1, L=3)

    def test_s_to_f72(self):
        _gauge_ratio(1, -1, 4, -4, 1, L=3)


# ---------------------------------------------------------------------------
# 4. E4 (L = 4)
# ---------------------------------------------------------------------------

class TestE4FullOrder:

    def test_s_to_g72(self):
        _gauge_ratio(1, -1, 5, 4, 1, L=4)

    def test_s_to_g92(self):
        _gauge_ratio(1, -1, 5, -5, 1, L=4)
