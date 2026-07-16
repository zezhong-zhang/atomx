"""Angular coefficients for the Dirac-Fock equations.

For the average-of-configuration Dirac-Fock, the two-electron interaction
between subshells a and b at multipole order k requires angular coefficients:

Direct:  f^k(a,b) — selects k=0 for closed-shell average
Exchange: g^k(a,b) — involves Wigner 3j symbols with selection rules on k

References:
    Grant, "Relativistic Quantum Theory", Eqs. 7.16-7.18
    Desclaux, Comp. Phys. Comm. 9, 31 (1975), Eq. 12
"""

import math
from functools import lru_cache

import numpy as np
from atomx.angular.wigner import wigner_3j
from atomx.angular.wigner import wigner_6j


def _l_from_kappa(kappa: int) -> int:
    return kappa if kappa > 0 else -kappa - 1


def _twoj_from_kappa(kappa: int) -> int:
    return 2 * abs(kappa) - 1


def _twoj(value: float | int) -> int:
    return int(round(2.0 * float(value)))


@lru_cache(maxsize=None)
def reduced_C(kappa_a: int, kappa_b: int, k: int) -> float:
    """Reduced matrix element ``<kappa_a || C^k || kappa_b>``.

    Uses Grant's relativistic convention:

    ``(-1)^(j_a+1/2) sqrt((2j_a+1)(2j_b+1))
      (j_a k j_b; -1/2 0 1/2)``.

    (lru_cached; pure function of small integers.)
    """
    l_a = _l_from_kappa(kappa_a)
    l_b = _l_from_kappa(kappa_b)
    if (l_a + k + l_b) % 2 != 0:
        return 0.0

    tj_a = _twoj_from_kappa(kappa_a)
    tj_b = _twoj_from_kappa(kappa_b)
    tk = 2 * int(k)
    if tk < abs(tj_a - tj_b) or tk > tj_a + tj_b:
        return 0.0

    w3j = wigner_3j(tj_a, tk, tj_b, -1, 0, 1)
    if abs(w3j) < 1e-15:
        return 0.0

    phase_exp = (tj_a + 1) // 2
    phase = 1.0 if phase_exp % 2 == 0 else -1.0
    return phase * math.sqrt((tj_a + 1) * (tj_b + 1)) * w3j


def cowan_grant_auger_B(
    kappa_a: int,
    kappa_b: int,
    kappa_c: int,
    kappa_eps: int,
    k: int,
    J_res: float,
    J_final: float = 0.0,
) -> float:
    """Cowan-Grant J-resolved Auger angular coefficient.

    Parameters use the channel convention ``a,b -> c,eps`` and ordinary
    angular momenta for ``J_res`` and ``J_final``.  The 6j recouples the
    two one-electron Coulomb vertices at rank ``k``; ``J_final`` gates the
    final residual level coupled to the continuum electron.  The phase is
    intentionally omitted because this coarse rate path squares each channel
    contribution independently.  A zero return means a triangle/parity rule
    forbids the requested coupling.
    """
    C_ac = reduced_C(kappa_a, kappa_c, k)
    C_be = reduced_C(kappa_b, kappa_eps, k)
    if abs(C_ac) < 1e-15 or abs(C_be) < 1e-15:
        return 0.0

    tj_a = _twoj_from_kappa(kappa_a)
    tj_b = _twoj_from_kappa(kappa_b)
    tj_c = _twoj_from_kappa(kappa_c)
    tj_e = _twoj_from_kappa(kappa_eps)
    tJ_res = _twoj(J_res)
    tJ_final = _twoj(J_final)

    # The continuum electron couples to the residual final level to conserve
    # the resonance J.  For the common closed-shell final state J_final=0,
    # this requires j_eps = J_res.
    if tJ_res < abs(tJ_final - tj_e) or tJ_res > tJ_final + tj_e:
        return 0.0
    if (tJ_res + tJ_final + tj_e) % 2 != 0:
        return 0.0

    sixj = wigner_6j(tj_a, tj_b, tJ_res, tj_e, tj_c, 2 * int(k))
    if abs(sixj) < 1e-15:
        return 0.0

    return (
        math.sqrt((tJ_res + 1) * (tJ_final + 1))
        * sixj
        * C_ac
        * C_be
    )


def reduced_cl_j2(tj_a, tk, tj_b):
    """FAC-compatible signed reduced C^k angular coefficient.

    Arguments use the twice-angular-momentum convention.  FAC's
    ``ReducedCL(ja, k, jb)`` is the signed square root whose square is the
    ``exchange_lambda`` angular factor for the corresponding kappa pair.
    """
    w3j = wigner_3j(tj_a, tk, tj_b, 1, 0, -1)
    if abs(w3j) < 1e-15:
        return 0.0
    phase_exp = (tj_a + 1) // 2
    phase = -1.0 if phase_exp % 2 else 1.0
    return float(phase * np.sqrt((tj_a + 1) * (tj_b + 1)) * w3j)


def exchange_lambda(kappa_a, kappa_b, k):
    """Angular coefficient Lambda(k, kappa_a, kappa_b) for the exchange interaction.

    Lambda = (2*j_a+1)*(2*j_b+1) * (j_a  k  j_b)^2
                                    (-1/2 0  1/2)

    Parameters
    ----------
    kappa_a, kappa_b : int
        Relativistic quantum numbers.
    k : int
        Multipole order.

    Returns
    -------
    float
        Angular coefficient. Zero if selection rules forbid this k.
    """
    # j = |kappa| - 1/2, so 2*j = 2*|kappa| - 1
    tj_a = 2 * abs(kappa_a) - 1  # twice j_a
    tj_b = 2 * abs(kappa_b) - 1  # twice j_b
    tk = 2 * k                    # twice k (k is integer)

    # 3j symbol: (j_a  k  j_b; -1/2  0  1/2)
    # In twice-j convention: (tj_a  tk  tj_b;  -1   0   1)
    w3j = wigner_3j(tj_a, tk, tj_b, -1, 0, 1)

    if abs(w3j) < 1e-15:
        return 0.0

    return (tj_a + 1) * (tj_b + 1) * w3j * w3j


def exchange_k_range(kappa_a, kappa_b):
    """Return the range of allowed k values for exchange between kappa_a, kappa_b.

    Triangle rule: |j_a - j_b| <= k <= j_a + j_b
    Parity: k + l_a + l_b must be even

    Parameters
    ----------
    kappa_a, kappa_b : int

    Returns
    -------
    k_values : list of int
        Allowed multipole orders.
    """
    l_a = kappa_a if kappa_a > 0 else -kappa_a - 1
    l_b = kappa_b if kappa_b > 0 else -kappa_b - 1
    j_a = abs(kappa_a) - 0.5
    j_b = abs(kappa_b) - 0.5

    k_min = int(abs(j_a - j_b))
    k_max = int(j_a + j_b)

    return [k for k in range(k_min, k_max + 1)
            if (k + l_a + l_b) % 2 == 0]


def direct_coefficient(q_b):
    """Direct (Hartree/Coulomb) coefficient for closed-shell average.

    For the average-of-configuration, only k=0 contributes to the
    direct potential, with coefficient = q_b (occupation of shell b).

    Parameters
    ----------
    q_b : int
        Occupation of shell b.

    Returns
    -------
    float
    """
    return float(q_b)


def exchange_coefficient(kappa_a, kappa_b, k, q_b):
    """Exchange coefficient for the Fock matrix of orbital a from orbital b.

    c^k(a,b) = -q_b / ((2*j_a + 1)*(2*j_b + 1)) * Lambda(k, kappa_a, kappa_b)

    Matches DBSR_HF av_energy_coef: coef(j,i,k) = -q_i*q_j * Cjkj^2/((jbs(i)+1)*(jbs(j)+1))
    after dividing by q_i (the target orbital occupation).

    Parameters
    ----------
    kappa_a, kappa_b : int
    k : int
    q_b : int or float

    Returns
    -------
    float
    """
    tj_a = 2 * abs(kappa_a) - 1
    tj_b = 2 * abs(kappa_b) - 1
    lam = exchange_lambda(kappa_a, kappa_b, k)
    if abs(lam) < 1e-15:
        return 0.0
    return -q_b / ((tj_a + 1.0) * (tj_b + 1.0)) * lam
