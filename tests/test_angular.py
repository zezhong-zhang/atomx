"""M2 tests: Angular momentum algebra validated against sympy exact values.

Uses sympy.physics.wigner as the reference oracle.
Our convention: all arguments are TWICE the angular momentum (integer storage).
"""

import numpy as np
import pytest
from sympy import Rational as R
from sympy.physics.wigner import (
    clebsch_gordan as sympy_cg,
)
from sympy.physics.wigner import (
    wigner_3j as sympy_3j,
)
from sympy.physics.wigner import (
    wigner_6j as sympy_6j,
)

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.wigner import wigner_3j, wigner_6j


def _ref_3j(j1, j2, j3, m1, m2, m3):
    """Reference 3j via sympy (accepts half-integer Rationals)."""
    return float(sympy_3j(R(j1, 1), R(j2, 1), R(j3, 1),
                          R(m1, 1), R(m2, 1), R(m3, 1)))


def _ref_3j_t(tj1, tj2, tj3, tm1, tm2, tm3):
    """Reference 3j from twice-j values."""
    return float(sympy_3j(R(tj1, 2), R(tj2, 2), R(tj3, 2),
                          R(tm1, 2), R(tm2, 2), R(tm3, 2)))


def _ref_6j_t(tj1, tj2, tj3, tl1, tl2, tl3):
    """Reference 6j from twice-j values."""
    return float(sympy_6j(R(tj1, 2), R(tj2, 2), R(tj3, 2),
                          R(tl1, 2), R(tl2, 2), R(tl3, 2)))


def _ref_cg_t(tj1, tm1, tj2, tm2, tj3, tm3):
    """Reference CG from twice-j values."""
    return float(sympy_cg(R(tj1, 2), R(tj2, 2), R(tj3, 2),
                          R(tm1, 2), R(tm2, 2), R(tm3, 2)))


class TestWigner3j:
    """Wigner 3j symbol tests against sympy."""

    @pytest.mark.parametrize("tj1,tj2,tj3,tm1,tm2,tm3", [
        (0, 0, 0, 0, 0, 0),          # trivial
        (1, 1, 0, 1, -1, 0),         # half-integer
        (1, 1, 2, 1, -1, 0),         # half-integer
        (1, 1, 2, 1, 1, -2),         # half-integer
        (2, 2, 0, 0, 0, 0),          # integer j=1
        (2, 2, 2, 2, 0, -2),         # integer
        (2, 2, 4, 0, 0, 0),          # j=1,1,2
        (2, 4, 4, 2, -2, 0),         # j=1,2,2
        (3, 3, 2, 1, -1, 0),         # j=3/2,3/2,1
        (4, 4, 4, 0, 0, 0),          # j=2,2,2
        (6, 4, 2, 0, 0, 0),          # j=3,2,1
    ])
    def test_against_sympy(self, tj1, tj2, tj3, tm1, tm2, tm3):
        val = wigner_3j(tj1, tj2, tj3, tm1, tm2, tm3)
        ref = _ref_3j_t(tj1, tj2, tj3, tm1, tm2, tm3)
        assert val == pytest.approx(ref, abs=1e-12), \
            f"3j({tj1},{tj2},{tj3};{tm1},{tm2},{tm3}): got {val}, expected {ref}"

    def test_selection_rule_m_sum(self):
        assert wigner_3j(2, 2, 2, 2, 2, 2) == 0.0

    def test_selection_rule_triangle(self):
        assert wigner_3j(0, 0, 4, 0, 0, 0) == 0.0

    def test_orthogonality(self):
        """Sum_{m1,m2} (2j3+1) |3j(j1,j2,j3;m1,m2,m3)|^2 = 1."""
        tj1, tj2, tj3, tm3 = 2, 2, 2, 0
        total = 0.0
        for tm1 in range(-tj1, tj1 + 1, 2):
            tm2 = -tm3 - tm1
            if abs(tm2) <= tj2:
                total += wigner_3j(tj1, tj2, tj3, tm1, tm2, tm3) ** 2
        assert total == pytest.approx(1.0 / (tj3 + 1), abs=1e-10)


class TestWigner6j:
    """Wigner 6j symbol tests against sympy."""

    @pytest.mark.parametrize("tj1,tj2,tj3,tl1,tl2,tl3", [
        (1, 1, 2, 1, 1, 2),          # {1/2,1/2,1;1/2,1/2,1}
        (2, 2, 2, 2, 2, 2),          # {1,1,1;1,1,1}
        (2, 2, 0, 2, 2, 0),          # with zeros
        (1, 1, 0, 1, 1, 0),          # half-integer with zeros
        (2, 4, 4, 2, 2, 4),          # mixed
        (3, 3, 2, 1, 1, 2),          # j=3/2 involved
        (4, 4, 4, 4, 4, 4),          # j=2,2,2;2,2,2
    ])
    def test_against_sympy(self, tj1, tj2, tj3, tl1, tl2, tl3):
        val = wigner_6j(tj1, tj2, tj3, tl1, tl2, tl3)
        ref = _ref_6j_t(tj1, tj2, tj3, tl1, tl2, tl3)
        assert val == pytest.approx(ref, abs=1e-10), \
            f"6j: got {val}, expected {ref}"

    def test_triangle_violation(self):
        assert wigner_6j(0, 0, 2, 0, 0, 0) == 0.0


class TestClebschGordan:
    """Clebsch-Gordan coefficient tests against sympy."""

    @pytest.mark.parametrize("tj1,tm1,tj2,tm2,tj3,tm3", [
        (2, 0, 0, 0, 2, 0),          # <1,0;0,0|1,0>
        (1, 1, 1, -1, 2, 0),         # <1/2,1/2;1/2,-1/2|1,0>
        (1, 1, 1, -1, 0, 0),         # <1/2,1/2;1/2,-1/2|0,0>
        (2, 2, 2, 0, 2, 2),          # <1,1;1,0|1,1>
        (2, 2, 2, -2, 4, 0),         # <1,1;1,-1|2,0>
        (3, 1, 1, 1, 4, 2),          # <3/2,1/2;1/2,1/2|2,1>
        (4, 2, 2, 0, 4, 2),          # <2,1;1,0|2,1>
    ])
    def test_against_sympy(self, tj1, tm1, tj2, tm2, tj3, tm3):
        val = clebsch_gordan(tj1, tm1, tj2, tm2, tj3, tm3)
        ref = _ref_cg_t(tj1, tm1, tj2, tm2, tj3, tm3)
        assert val == pytest.approx(ref, abs=1e-10), \
            f"CG({tj1},{tm1};{tj2},{tm2}|{tj3},{tm3}): got {val}, expected {ref}"

    def test_selection_rule(self):
        """CG vanishes if m1 + m2 != m3."""
        assert clebsch_gordan(2, 2, 2, 0, 2, 0) == 0.0

    def test_normalization(self):
        """Sum_{m1,m2} |CG|^2 = 1 for fixed j3,m3."""
        tj1, tj2, tj3, tm3 = 2, 2, 2, 0
        total = 0.0
        for tm1 in range(-tj1, tj1 + 1, 2):
            tm2 = tm3 - tm1
            if abs(tm2) <= tj2:
                total += clebsch_gordan(tj1, tm1, tj2, tm2, tj3, tm3) ** 2
        assert total == pytest.approx(1.0, abs=1e-10)


class TestCowanGrantAugerCoefficient:
    def test_reduced_C_matches_grant_convention(self):
        from atomx.angular.coefficients import reduced_C

        assert reduced_C(-1, -1, 0) == pytest.approx(-np.sqrt(2.0))
        assert reduced_C(-1, -2, 1) == pytest.approx(2.0 / np.sqrt(3.0))
        assert reduced_C(-1, -1, 1) == 0.0

    def test_cowan_grant_B_nonzero_closed_channel(self):
        from atomx.angular.coefficients import cowan_grant_auger_B

        val = cowan_grant_auger_B(-1, 1, 1, -1, 1, J_res=0.0, J_final=0.5)
        assert val == pytest.approx(np.sqrt(2.0) / 3.0, abs=1e-12)

    def test_cowan_grant_B_gates_final_continuum_coupling(self):
        from atomx.angular.coefficients import cowan_grant_auger_B

        assert cowan_grant_auger_B(-1, 1, 1, 1, 1, J_res=0.5, J_final=0.0) == 0.0
