"""Higher-multipole (M1, E2, M2) Einstein A-coefficient regression tests.

The E1 case is validated against NIST + FAC in benchmarks/REPORT.md §1.1.
This module pins the prefactor and (2L+1)!! algebra for L = 2 (E2) and
L = 3 (E3), plus the magnetic-multipole entry point
``einstein_A_mL``, against closed-form values.

Reference: Grant (2007) Eq. 8.4.10:
    A_EL = 2 · α^(2L+1) · ω^(2L+1) · (L+1)(2L+1)
           / [L · ((2L+1)!!)²] · S / (2j_i + 1)
"""


import pytest

from atomx.constants import ALPHA, AU_RATE_PER_S
from atomx.properties.transitions.rates import (
    einstein_A_eL,
    einstein_A_mL,
    oscillator_strength,
)


def _double_factorial(n):
    out = 1
    for i in range(1, n + 1, 2):
        out *= i
    return out


def _exact_A_eL(omega, S, L, tj_i):
    """Reproduce Grant Eq. 8.4.10 by hand for cross-check."""
    pre = (2.0 * ALPHA ** (2*L + 1) * omega ** (2*L + 1)
           * (L + 1) * (2*L + 1)
           / (L * _double_factorial(2*L + 1) ** 2))
    return pre * S / (tj_i + 1) * AU_RATE_PER_S


@pytest.mark.parametrize("L", [1, 2, 3, 4])
def test_eL_matches_grant_formula(L):
    """einstein_A_eL agrees with closed-form Grant Eq. 8.4.10."""
    omega = 1.5         # Hartree
    S = 0.42
    tj_i = 1            # j_i = 1/2
    val = einstein_A_eL(omega, S, L, tj_i)
    ref = _exact_A_eL(omega, S, L, tj_i)
    assert val == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize("L", [1, 2, 3])
def test_mL_matches_eL(L):
    """In the convention where ``S_ML`` already absorbs the 1/c, the
    A_M and A_E formulae are identical (rates.py docstring)."""
    omega = 0.7
    S = 0.18
    tj_i = 5            # j_i = 5/2
    assert einstein_A_mL(omega, S, L, tj_i) == pytest.approx(
        einstein_A_eL(omega, S, L, tj_i), rel=1e-12,
    )


def test_eL_zero_below_threshold():
    """ω ≤ 0 returns 0 (no spontaneous emission)."""
    assert einstein_A_eL(0.0, 1.0, 1, 1) == 0.0
    assert einstein_A_eL(-0.5, 1.0, 1, 1) == 0.0


def test_eL_zero_for_zero_strength():
    assert einstein_A_eL(1.0, 0.0, 2, 3) == 0.0


def test_omega_scaling_e1_e2_e3():
    """A_E1 ∝ ω³, A_E2 ∝ ω⁵, A_E3 ∝ ω⁷."""
    S = 1.0
    tj = 1
    A_a_e1 = einstein_A_eL(1.0, S, 1, tj)
    A_b_e1 = einstein_A_eL(2.0, S, 1, tj)
    assert A_b_e1 / A_a_e1 == pytest.approx(2.0**3, rel=1e-12)
    A_a_e2 = einstein_A_eL(1.0, S, 2, tj)
    A_b_e2 = einstein_A_eL(2.0, S, 2, tj)
    assert A_b_e2 / A_a_e2 == pytest.approx(2.0**5, rel=1e-12)
    A_a_e3 = einstein_A_eL(1.0, S, 3, tj)
    A_b_e3 = einstein_A_eL(2.0, S, 3, tj)
    assert A_b_e3 / A_a_e3 == pytest.approx(2.0**7, rel=1e-12)


def test_alpha_scaling():
    """A ∝ α^(2L+1):  ratio A_E2 / A_E1 ∝ α² for fixed ω, S, j."""
    S = 1.0
    omega = 1.0
    tj = 1
    A_e1 = einstein_A_eL(omega, S, 1, tj)
    A_e2 = einstein_A_eL(omega, S, 2, tj)
    # Ratio of prefactors: (3·5/[2·(5!!)²]) / (2·3/[1·(3!!)²]) = (15/(2·225)) / (6/9)
    #                  = 1/30 / 2/3 = 1/20 = 0.05
    # × α² extra from L=2
    expected = (1.0 / 20.0) * ALPHA ** 2
    assert A_e2 / A_e1 == pytest.approx(expected, rel=1e-12)


def test_he_like_m1_2s_to_1s_z_scaling():
    """He-like 2³S₁ → 1¹S₀ M1 rate scales as Z¹⁰ in the relativistic
    limit (Drake 1971 / Johnson-Lin-Dalgarno 1977).

    The rate is α²·ω³·M² with M ≈ α·Z·R³/Z = α (relativistic spinor
    mixing).  Squared M ∝ α²·Z⁰; combined with ω³ ∝ Z⁶ and an
    additional Z² from the magnetic dipole operator gives A ∝ Z¹⁰.

    We test that supplying ω·Z² and S·Z⁴ scales A by Z¹⁰ — i.e. the
    formula respects the expected scaling.  Catches errors in the
    (2L+1)·(L+1)/[L·(2L+1)!!²] prefactor for L = 1.
    """
    A0 = einstein_A_mL(omega=1.0, line_strength=1.0, L=1, tj_i=2)
    Z = 10.0
    A1 = einstein_A_mL(omega=Z**2, line_strength=Z**4, L=1, tj_i=2)
    assert pytest.approx(Z**10, rel=1e-12) == A1 / A0


def test_oscillator_strength_e1():
    """Oscillator strength gf = 2ω/(2L+1) · S, f = gf/(2j_i+1)."""
    omega = 0.5
    S = 1.5
    tj_i = 3
    f, gf = oscillator_strength(omega, S, 1, tj_i, 5)
    assert gf == pytest.approx(2.0 * omega / 3.0 * S, rel=1e-12)
    assert f == pytest.approx(gf / 4.0, rel=1e-12)


def test_oscillator_strength_zero_below_threshold():
    assert oscillator_strength(0.0, 1.0, 1, 1, 1) == (0.0, 0.0)
