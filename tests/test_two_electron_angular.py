"""Angular coefficients for two-electron target close coupling."""

import math

from atomx.scattering.r_matrix.angular_cc import reduced_C_LS
from atomx.scattering.two_electron.angular import (
    direct_coefficient,
    target_reduced_multipole,
)


def test_target_reduced_multipole_1s2_closed_shell():
    """For 1s^2 the k=0 reduced matrix element is 2 (two electrons)."""
    M = target_reduced_multipole(
        l1_bra=0, l2_bra=0, L_t_bra=0,
        l1_ket=0, l2_ket=0, L_t_ket=0,
        k=0,
    )
    assert abs(M - 2.0) < 1e-12


def test_target_reduced_multipole_1s2p_diagonal_scalar():
    """For 1s2p ^1P the k=0 reduced matrix element is 2*sqrt(3)."""
    M = target_reduced_multipole(
        l1_bra=0, l2_bra=1, L_t_bra=1,
        l1_ket=0, l2_ket=1, L_t_ket=1,
        k=0,
    )
    assert abs(M - 2.0 * math.sqrt(3.0)) < 1e-12


def test_target_reduced_multipole_1s2_to_1s2p():
    """1s^2 -> 1s2p couples only through the 2p electron with k=1."""
    M = target_reduced_multipole(
        l1_bra=0, l2_bra=1, L_t_bra=1,
        l1_ket=0, l2_ket=0, L_t_ket=0,
        k=1,
    )
    # <1s||C^1||2p> = reduced_C_LS(0,1,1) = -1.
    assert abs(M - (-1.0)) < 1e-12


def test_direct_coefficient_reduces_to_one_electron_target():
    """When the target reduced C^k equals the one-electron value, the
    composite-target direct coefficient matches the original
    :func:`angular_cc.direct_terms` result.
    """
    from atomx.scattering.r_matrix.angular_cc import direct_terms

    l_t, l_c, L = 1, 2, 3
    terms_one = {t.k: t.coefficient for t in direct_terms(l_t, l_c, l_t, l_c, L)}
    for k, coeff_one in terms_one.items():
        rm_t = reduced_C_LS(l_t, k, l_t)
        coeff_two = direct_coefficient(
            l_t, l_c, l_t, l_c, L, k=k, target_reduced_c=rm_t
        )
        assert abs(coeff_two - coeff_one) < 1e-12, f"k={k} mismatch"
