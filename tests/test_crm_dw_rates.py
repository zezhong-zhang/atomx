"""CRM consumes tabulated DW collision strengths (review M1 fix).

The rate matrix prefers a tabulated Omega(E) (from the FAC-verified DW
machinery) over the van Regemorter gf formula; van Regemorter remains the
explicit fallback when no table is attached.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx.constants import BOLTZMANN_EV  # noqa: E402
from atomx.plasma.crm import (  # noqa: E402
    CollisionalRadiativeModel,
    CRMLevel,
    CRMTransition,
    effective_collision_strength,
)


def test_effective_collision_strength_constant_omega_exact():
    # Upsilon(const Omega = c) = c, to machine precision (piecewise-exact
    # kernel, not trapezoid-on-exp).
    E = np.linspace(50.0, 5000.0, 17)
    om = np.full_like(E, 0.831)
    for T in (5.0, 50.0, 500.0):
        assert effective_collision_strength(E, om, 50.0, T) == pytest.approx(
            0.831, rel=1e-12)


def test_effective_collision_strength_below_threshold_empty():
    E = np.array([10.0, 20.0])
    om = np.array([1.0, 1.0])
    assert effective_collision_strength(E, om, 100.0, 10.0) == 0.0


def _two_level(omega_table):
    levels = [CRMLevel("g", 0.0, g=2),
              CRMLevel("e", 100.0, g=4)]
    kw = {}
    if omega_table is not None:
        kw = {"omega_E_eV": omega_table[0], "omega": omega_table[1]}
    trs = [CRMTransition(upper=1, lower=0, A_rate=1e9, gf=0.5, **kw)]
    return CollisionalRadiativeModel(levels, trs)


def test_rate_matrix_uses_omega_table_over_van_regemorter():
    E = np.linspace(100.0, 5000.0, 30)
    om = np.full_like(E, 1.0)
    m_dw = _two_level((E, om))
    m_vr = _two_level(None)
    Te = 200.0
    R_dw = m_dw._build_rate_matrix(1.0e12, Te)
    R_vr = m_vr._build_rate_matrix(1.0e12, Te)
    # analytic rate for Upsilon = 1:
    # C = 8.629e-6/(g_l sqrt(T_K)) exp(-dE/kT)
    T_K = Te / BOLTZMANN_EV
    C = 8.6291e-6 / (2.0 * np.sqrt(T_K)) * np.exp(-100.0 / Te)
    assert R_dw[1, 0] == pytest.approx(1.0e12 * C, rel=1e-10)
    # the fallback gives a different (van Regemorter) number
    assert R_vr[1, 0] > 0.0
    assert R_vr[1, 0] != pytest.approx(R_dw[1, 0], rel=0.05)


def test_rate_matrix_detailed_balance_with_omega():
    E = np.linspace(100.0, 8000.0, 40)
    om = 0.3 + 1e-5 * E
    m = _two_level((E, om))
    Te = 350.0
    R = m._build_rate_matrix(5.0e13, Te)
    C_up = R[1, 0] / 5.0e13
    C_down = (R[0, 1] - 1e9) / 5.0e13  # subtract A_rate
    # detailed balance: C_down = C_up (g_l/g_u) exp(dE/kT)
    assert C_down == pytest.approx(
        C_up * (2.0 / 4.0) * np.exp(100.0 / Te), rel=1e-10)
