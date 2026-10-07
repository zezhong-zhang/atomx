"""Shared one- and two-electron angular coefficients.

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

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.recoupling import l_of_kappa, twoj_of_kappa
from atomx.angular.wigner import wigner_3j, wigner_6j


def _twoj(value: float | int) -> int:
    return int(round(2.0 * float(value)))


@lru_cache(maxsize=None)
def reduced_C_jj(kappa_a: int, kappa_b: int, k: int) -> float:
    """Reduced C tensor in Grant's jj-coupling phase convention.

    Its phase is (-1)^(j_a+1/2) multiplying the 3j projections
    (-1/2, 0, +1/2). GRASP CRE and FAC ReducedCL instead use
    (+1/2, 0, -1/2), as implemented by :func:`reduced_cl_j2`.
    The two conventions differ by (-1)^(j_a+j_b+k); their squared
    coefficients agree, but mixed operators need explicit conversion.
    The LS-coupled reduced C element is a different function:
    :func:`atomx.scattering.r_matrix.angular_cc.reduced_C_LS`.

    In particular this function gives -sqrt(2j+1) at k=0. Do not combine
    it directly with a standard reduced Pauli spin tensor: the recoil
    SMS2/CRE ratio must use C and sigma in the same convention. Existing
    CI angular contractions retain this convention consistently; changing
    the common tensor requires migrating that entire contraction chain.

    Cached as a pure function of the integer angular quantum numbers.
    """
    l_a = l_of_kappa(kappa_a)
    l_b = l_of_kappa(kappa_b)
    if (l_a + k + l_b) % 2 != 0:
        return 0.0

    tj_a = twoj_of_kappa(kappa_a)
    tj_b = twoj_of_kappa(kappa_b)
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
    C_ac = reduced_C_jj(kappa_a, kappa_c, k)
    C_be = reduced_C_jj(kappa_b, kappa_eps, k)
    if abs(C_ac) < 1e-15 or abs(C_be) < 1e-15:
        return 0.0

    tj_a = twoj_of_kappa(kappa_a)
    tj_b = twoj_of_kappa(kappa_b)
    tj_c = twoj_of_kappa(kappa_c)
    tj_e = twoj_of_kappa(kappa_eps)
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


def reduced_matrix_ls(k: int, l: int, lp: int) -> float:
    """LS (non-relativistic) reduced matrix element ``<l||K||l'>``.

    XATOM ``reduced_matrix(K, l, l')`` (coulomb.f90:271, Son-Young-Santra
    PRA 83, 033402 Eq. (11) with the erratum applied):

        <l||K||l'> = (-1)^(l-l'+K) sqrt(2l'+1) <l' 0 K 0 | l 0>

    Zero when the triangle or m=0 parity selection rules forbid the term.
    """
    if abs(l - lp) > k or l + lp < k or (l + lp + k) % 2:
        return 0.0
    return (
        (-1.0) ** (l - lp + k)
        * math.sqrt(2 * lp + 1)
        * clebsch_gordan(2 * lp, 0, 2 * k, 0, 2 * l, 0)
    )


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
    tj_a = twoj_of_kappa(kappa_a)
    tj_b = twoj_of_kappa(kappa_b)
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
    l_a = l_of_kappa(kappa_a)
    l_b = l_of_kappa(kappa_b)
    tj_a = twoj_of_kappa(kappa_a)
    tj_b = twoj_of_kappa(kappa_b)

    k_min = abs(tj_a - tj_b) // 2
    k_max = (tj_a + tj_b) // 2

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
    tj_a = twoj_of_kappa(kappa_a)
    tj_b = twoj_of_kappa(kappa_b)
    lam = exchange_lambda(kappa_a, kappa_b, k)
    if abs(lam) < 1e-15:
        return 0.0
    return -q_b / ((tj_a + 1.0) * (tj_b + 1.0)) * lam


# ---------------------------------------------------------------------------
# m-resolved angular coefficients (shared by the continuum/DW scattering code
# and the CI matrix elements)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def ckm(kappa_c: int, tmc: int, kappa_a: int, tma: int, k: int, tq: int) -> float:
    """``<j_c m_c | C^k_q | j_a m_a>`` (doubled m and q), Grant's ``<||C^k||>``.

    This is FAC's ``ckm`` — the m-resolved angular factor multiplying the
    Slater integral ``R^k`` in every two-electron matrix element.  It lives in
    the angular layer so the scattering (continuum/DW) and CI codes share one
    implementation.

    The Wigner–Eckart step here is the textbook one,
    ``(-1)^(j_c-m_c) (j_c k j_a; -m_c q m_a)``, but ``reduced_C_jj`` carries
    *Grant's* phase, so the product is Grant's coefficient, not the textbook
    one — in particular ``ckm(k, m, k, m, 0, 0) == -1`` for every ``k``/``m``
    (see :func:`reduced_C_jj`).  That is deliberate and pinned by
    ``test_ckm_identity_sign_pins_grant_convention``.
    """
    red = reduced_C_jj(kappa_c, kappa_a, k)
    if red == 0.0:
        return 0.0
    tjc, tja = twoj_of_kappa(kappa_c), twoj_of_kappa(kappa_a)
    w = wigner_3j(tjc, 2 * k, tja, -tmc, tq, tma)
    if abs(w) < 1e-15:
        return 0.0
    phase = 1.0 if ((tjc - tmc) // 2) % 2 == 0 else -1.0
    return phase * w * red


@lru_cache(maxsize=131072)
def coupled_coulomb_direct(kappa_c, kappa_d, kappa_a, kappa_b, tJ, k):
    """Angular factor of ``<(c d) JM | C^k(1)·C^k(2) | (a b) JM>``.

    ``tJ`` is twice the total angular momentum; ``k`` is the integer tensor
    rank. The spin-angular orbitals use :func:`reduced_C_jj`'s phase convention.
    Racah recoupling eliminates the magnetic-sublevel sums exactly:

    ``(-1)^(j_a+j_d+J) {j_c j_d J; j_b j_a k} <c||C^k||a><d||C^k||b>``.

    The radial Slater integral and antisymmetrization are excluded. This
    coefficient applies equally to bound and continuum orbitals and is
    independent of M. Forbidden triangles or parities give zero.

    Bound the memo: a large continuum channel space otherwise retains
    millions of coefficients after its packed channel table is constructed.
    """
    ca = reduced_C_jj(kappa_c, kappa_a, k)
    db = reduced_C_jj(kappa_d, kappa_b, k)
    if ca == 0.0 or db == 0.0:
        return 0.0
    tja, tjb = twoj_of_kappa(kappa_a), twoj_of_kappa(kappa_b)
    tjc, tjd = twoj_of_kappa(kappa_c), twoj_of_kappa(kappa_d)
    sixj = wigner_6j(tjc, tjd, tJ, tjb, tja, 2 * k)
    phase = -1.0 if ((tja + tjd + tJ) // 2) % 2 else 1.0
    return phase * sixj * ca * db


@lru_cache(maxsize=131072)
def coupled_coulomb_exchange(kappa_c, kappa_d, kappa_a, kappa_b, tJ, k):
    """Exchange angular factor with vertices ``d <- a`` and ``c <- b``.

    Uses the same ordered bra/ket couplings as :func:`coupled_coulomb_direct`.
    Swapping the coupled bra contributes ``(-1)^(j_c+j_d-J)``. This is the
    permutation phase, not the fermionic minus sign: the antisymmetrized
    amplitude is ``D*R_direct - E*R_exchange``.
    """
    tjc, tjd = twoj_of_kappa(kappa_c), twoj_of_kappa(kappa_d)
    phase = -1.0 if ((tjc + tjd - tJ) // 2) % 2 else 1.0
    return phase * coupled_coulomb_direct(
        kappa_d, kappa_c, kappa_a, kappa_b, tJ, k)
