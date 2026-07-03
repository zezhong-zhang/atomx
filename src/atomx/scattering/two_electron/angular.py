"""LS-coupled angular coefficients for two-electron target close coupling.

This module provides the reduced multipole matrix elements of the target
state that are needed when the projectile couples to a composite (two-
electron) target.  The conventions match :mod:`atomx.scattering.r_matrix.angular_cc`:
``C^k`` is the renormalised spherical harmonic and reduced matrix elements
are computed with the usual Condon-Shortley phase.
"""

from __future__ import annotations

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.wigner import wigner_3j, wigner_6j
from atomx.scattering.r_matrix.angular_cc import reduced_C


def _electron_contribution(
    l_bra: int,
    l_ket: int,
    L_t_bra: int,
    L_t_ket: int,
    l_spectator: int,
    k: int,
) -> float:
    """Reduced matrix element <(l_bra l_spectator)L_t_bra || C^k || (l_ket l_spectator)L_t_ket>.

    The active electron is the first member of each coupled pair; the
    spectator angular momentum is ``l_spectator``.  The formula is the
    standard spectator recoupling (Cowan 1981, eq. 11.20).
    """
    if (l_bra + k + l_ket) % 2 != 0:
        return 0.0
    rm = reduced_C(l_bra, k, l_ket)
    if rm == 0.0:
        return 0.0
    six = wigner_6j(
        2 * l_bra,
        2 * L_t_bra,
        2 * l_spectator,
        2 * L_t_ket,
        2 * l_ket,
        2 * k,
    )
    if six == 0.0:
        return 0.0
    phase = -1.0 if (l_bra + l_spectator + L_t_bra + k) % 2 else 1.0
    return (
        phase
        * ((2 * L_t_bra + 1) * (2 * L_t_ket + 1)) ** 0.5
        * six
        * rm
    )


def target_reduced_multipole(
    l1_bra: int,
    l2_bra: int,
    L_t_bra: int,
    l1_ket: int,
    l2_ket: int,
    L_t_ket: int,
    k: int,
) -> float:
    """``<L_t_bra || Σ_a C^k(a) || L_t_ket>`` for a two-electron target.

    The target orbital state is ``|(l1 l2) L_t>``.  Both electrons are
    summed; for equivalent electrons this automatically gives the correct
    occupancy factor (e.g. 2 for ``1s^2`` when ``k=0``).
    """
    total = 0.0
    # Electron 1 changes l1, electron 2 is spectator.
    if l2_bra == l2_ket:
        total += _electron_contribution(
            l1_bra, l1_ket, L_t_bra, L_t_ket, l2_ket, k
        )
    # Electron 2 changes l2, electron 1 is spectator.
    if l1_bra == l1_ket:
        total += _electron_contribution(
            l2_bra, l2_ket, L_t_bra, L_t_ket, l1_ket, k
        )
    return total


def direct_coefficient(
    l_t_bra: int,
    l_c_bra: int,
    l_t_ket: int,
    l_c_ket: int,
    L_total: int,
    k: int,
    target_reduced_c: float,
) -> float:
    """Direct multipole coefficient ``v_k^d`` for a composite target.

    This is the same formula as :func:`angular_cc.direct_terms`, except the
    one-electron target reduced matrix element ``<l_t_bra||C^k||l_t_ket>``
    is replaced by the supplied two-electron value ``target_reduced_c``.
    """
    rm_c = reduced_C(l_c_bra, k, l_c_ket)
    if rm_c == 0.0:
        return 0.0
    six = wigner_6j(
        2 * l_t_bra,
        2 * l_c_bra,
        2 * L_total,
        2 * l_c_ket,
        2 * l_t_ket,
        2 * k,
    )
    if six == 0.0:
        return 0.0
    phase = -1.0 if (l_t_bra + l_c_ket + L_total) % 2 else 1.0
    return phase * target_reduced_c * rm_c * six


# ---------------------------------------------------------------------------
# Brute-force LS-coupled three-electron angular coefficients.
# ---------------------------------------------------------------------------

_PERMS = ("I", "P13", "P23")


def _perm_identity():
    return (0, 1, 2)  # maps new coord -> old coord index


def _perm_p13():
    return (2, 1, 0)


def _perm_p23():
    return (0, 2, 1)


def _perm_vector(perm: str):
    if perm == "I":
        return _perm_identity()
    if perm == "P13":
        return _perm_p13()
    if perm == "P23":
        return _perm_p23()
    raise ValueError(perm)


def _perm_sign(perm: str) -> int:
    return 0 if perm == "I" else 1


def _orbital_cg_expansion(
    l1: int,
    l2: int,
    L_t: int,
    l_c: int,
    L_total: int,
    M_L: int,
) -> list[tuple[int, int, int, float]]:
    """Return ``(m1, m2, m_c, coeff)`` for ``|((l1 l2)L_t, l_c)L_total M_L>``.

    All angular-momentum arguments are physical (integer) values.
    """
    terms: list[tuple[int, int, int, float]] = []
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            M_t = m1 + m2
            if abs(M_t) > L_t:
                continue
            cg1 = clebsch_gordan(
                2 * l1, 2 * m1, 2 * l2, 2 * m2, 2 * L_t, 2 * M_t
            )
            if cg1 == 0.0:
                continue
            for m_c in range(-l_c, l_c + 1):
                if M_t + m_c != M_L:
                    continue
                cg2 = clebsch_gordan(
                    2 * L_t, 2 * M_t, 2 * l_c, 2 * m_c, 2 * L_total, 2 * M_L
                )
                if cg2 == 0.0:
                    continue
                coeff = cg1 * cg2
                if coeff != 0.0:
                    terms.append((m1, m2, m_c, coeff))
    return terms


def _spin_cg_expansion(
    S_t: int,
    S_total: float,
    M_S: float,
) -> list[tuple[int, int, int, float]]:
    """Return ``(mu1, mu2, mu_c, coeff)`` for ``|((1/2 1/2)S_t, 1/2)S_total M_S>``.

    ``S_t`` is an integer (0 or 1); ``S_total`` and ``M_S`` are physical
    half-integer values.  The stored magnetic quantum numbers are doubled.
    """
    terms: list[tuple[int, int, int, float]] = []
    tS_t = 2 * S_t
    tS_total = int(round(2 * S_total))
    tM_S = int(round(2 * M_S))
    s = 1  # doubled spin 1/2
    for mu1 in (-1, 1):
        for mu2 in (-1, 1):
            tM_st = mu1 + mu2
            if abs(tM_st) > tS_t:
                continue
            cg1 = clebsch_gordan(s, mu1, s, mu2, tS_t, tM_st)
            if cg1 == 0.0:
                continue
            for mu_c in (-1, 1):
                if tM_st + mu_c != tM_S:
                    continue
                cg2 = clebsch_gordan(
                    tS_t, tM_st, s, mu_c, tS_total, tM_S
                )
                if cg2 == 0.0:
                    continue
                coeff = cg1 * cg2
                if coeff != 0.0:
                    terms.append((mu1, mu2, mu_c, coeff))
    return terms


def _apply_perm_to_terms(
    terms: list[tuple[int, int, int, float]], perm: str
) -> list[tuple[int, int, int, float]]:
    """Apply a particle permutation to an uncoupled expansion.

    ``terms`` holds ``(x1, x2, x3, coeff)`` for coordinates 1, 2, 3.
    The returned list is the expansion of ``P |state>``.
    """
    p = _perm_vector(perm)
    result: list[tuple[int, int, int, float]] = []
    for x1, x2, x3, coeff in terms:
        vals = (x1, x2, x3)
        result.append((vals[p[0]], vals[p[1]], vals[p[2]], coeff))
    return result


def _spin_overlap(
    spin_j_terms: list[tuple[int, int, int, float]],
    spin_i_terms: list[tuple[int, int, int, float]],
    perm_j: str,
    perm_i: str,
) -> float:
    """Spin overlap ``<perm_j spin_j | perm_i spin_i>``.

    All Clebsch-Gordan coefficients are real, so the overlap is a plain
    sum over matching uncoupled spin basis states.
    """
    perm_j_terms = _apply_perm_to_terms(spin_j_terms, perm_j)
    perm_i_terms = _apply_perm_to_terms(spin_i_terms, perm_i)
    # Build lookup for ket coefficients.
    ket_map: dict[tuple[int, int, int], float] = {}
    for mu1, mu2, mu_c, coeff in perm_i_terms:
        ket_map[(mu1, mu2, mu_c)] = coeff
    total = 0.0
    for mu1, mu2, mu_c, coeff in perm_j_terms:
        total += coeff * ket_map.get((mu1, mu2, mu_c), 0.0)
    return total


def _C_matrix_element(
    l_bra: int,
    m_bra: int,
    l_ket: int,
    m_ket: int,
    k: int,
    q: int,
) -> float:
    """``<l_bra m_bra | C^k_q | l_ket m_ket>`` using Wigner-Eckart."""
    if m_ket + q != m_bra:
        return 0.0
    rm = reduced_C(l_bra, k, l_ket)
    if rm == 0.0:
        return 0.0
    threej = wigner_3j(
        2 * l_bra, 2 * k, 2 * l_ket,
        -2 * m_bra, 2 * q, 2 * m_ket,
    )
    if threej == 0.0:
        return 0.0
    phase = -1.0 if (l_bra - m_bra) % 2 else 1.0
    return phase * threej * rm


def _orbital_at_coord(
    l1: int, l2: int, l_c: int, coord: int, perm: str
) -> int:
    """Orbital angular momentum at ``coord`` after applying ``perm``.

    ``coord`` is 1, 2 or 3.  The permutation maps the new coordinate to
    the original electron index (target 1, target 2, projectile 3).
    """
    p = _perm_vector(perm)
    original = p[coord - 1]
    return (l1, l2, l_c)[original]


def _classify_shape(
    la_j: int,
    la_i: int,
    lc_j: int,
    lc_i: int,
    l1_j: int,
    l2_j: int,
    l1_i: int,
    l2_i: int,
    lc_j0: int,
    lc_i0: int,
) -> str | None:
    """Map a radial integral ``R^k(la_j, la_i; lc_j, lc_i)`` to a shape key.

    The radial integral is symmetric under swapping the two pairs, so both
    ``(A,B;C,D)`` and ``(C,D;A,B)`` are tried.
    """
    direct_patterns = (
        ((l1_j, l1_i, lc_j0, lc_i0), "direct_1"),
        ((l2_j, l2_i, lc_j0, lc_i0), "direct_2"),
    )
    exchange_patterns = (
        ((l1_j, lc_i0, lc_j0, l1_i), "exchange_1"),
        ((l2_j, lc_i0, lc_j0, l2_i), "exchange_2"),
    )
    for patterns in (direct_patterns, exchange_patterns):
        for pattern, key in patterns:
            if (la_j, la_i, lc_j, lc_i) == pattern:
                return key
            if (lc_j, lc_i, la_j, la_i) == pattern:
                return key
    return None


def two_electron_cc_coefficients(
    l1_bra: int,
    l2_bra: int,
    L_t_bra: int,
    S_t_bra: int,
    l_c_bra: int,
    l1_ket: int,
    l2_ket: int,
    L_t_ket: int,
    S_t_ket: int,
    l_c_ket: int,
    L_total: int,
    S_total: float,
) -> dict[str, list[tuple[int, float]]]:
    """Brute-force LS-coupled angular coefficients for (two-electron target + projectile).

    Returns a dictionary with keys ``direct_1``, ``direct_2``,
    ``exchange_1``, ``exchange_2``.  Each value is a list of ``(k,
    coefficient)`` pairs for the corresponding radial integral shape.

    The method expands the coupled orbital and spin states in the uncoupled
    ``|m1 m2 m_c>`` / ``|mu1 mu2 mu_c>`` basis, applies the three-electron
    antisymmetrizer ``(I - P13 - P23)/sqrt(3)`` to both bra and ket, and
    evaluates the matrix element of ``sum_a C^k(a) . C^k(c)`` (a = target
    electron index, c = projectile coordinate 3).  Results are accumulated
    by the orbitals paired at the active coordinates.

    Conventions match :mod:`atomx.scattering.r_matrix.angular_cc`: the
    returned coefficients are the ordinary LS-coupled matrix elements of the
    scalar multipole operators, ready to multiply the corresponding Slater
    integrals.
    """
    # Use M_L = L_total and M_S = S_total; scalar-operator matrix elements
    # are independent of the chosen magnetic sublevels.
    M_L = L_total
    M_S = S_total

    orb_j = _orbital_cg_expansion(
        l1_bra, l2_bra, L_t_bra, l_c_bra, L_total, M_L
    )
    orb_i = _orbital_cg_expansion(
        l1_ket, l2_ket, L_t_ket, l_c_ket, L_total, M_L
    )
    spin_j = _spin_cg_expansion(S_t_bra, S_total, M_S)
    spin_i = _spin_cg_expansion(S_t_ket, S_total, M_S)

    # Maximum k to consider.
    k_max = max(
        l1_bra + l1_ket,
        l2_bra + l2_ket,
        l_c_bra + l_c_ket,
        l1_bra + l_c_ket,
        l_c_bra + l1_ket,
        l2_bra + l_c_ket,
        l_c_bra + l2_ket,
    )

    accum: dict[str, dict[int, float]] = {
        "direct_1": {},
        "direct_2": {},
        "exchange_1": {},
        "exchange_2": {},
    }

    for perm_j in _PERMS:
        sign_j = -1.0 if _perm_sign(perm_j) else 1.0
        for perm_i in _PERMS:
            sign_i = -1.0 if _perm_sign(perm_i) else 1.0
            overall_sign = sign_j * sign_i / 3.0

            spin_ov = _spin_overlap(spin_j, spin_i, perm_j, perm_i)
            if abs(spin_ov) < 1e-15:
                continue

            orb_j_p = _apply_perm_to_terms(orb_j, perm_j)
            orb_i_p = _apply_perm_to_terms(orb_i, perm_i)

            for a in (1, 2):  # target electron index in the operator
                c_coord = 3
                b = 2 if a == 1 else 1  # spectator target coordinate

                la_j = _orbital_at_coord(l1_bra, l2_bra, l_c_bra, a, perm_j)
                la_i = _orbital_at_coord(l1_ket, l2_ket, l_c_ket, a, perm_i)
                lc_j = _orbital_at_coord(l1_bra, l2_bra, l_c_bra, c_coord, perm_j)
                lc_i = _orbital_at_coord(l1_ket, l2_ket, l_c_ket, c_coord, perm_i)
                lb_j = _orbital_at_coord(l1_bra, l2_bra, l_c_bra, b, perm_j)
                lb_i = _orbital_at_coord(l1_ket, l2_ket, l_c_ket, b, perm_i)

                if lb_j != lb_i:
                    # Orthogonality of the spectator orbital.
                    continue

                shape = _classify_shape(
                    la_j, la_i, lc_j, lc_i,
                    l1_bra, l2_bra, l1_ket, l2_ket,
                    l_c_bra, l_c_ket,
                )
                if shape is None:
                    # Term does not correspond to one of the four
                    # close-coupling shapes; skip it.
                    continue

                for k in range(k_max + 1):
                    rm_a = reduced_C(la_j, k, la_i)
                    if rm_a == 0.0:
                        continue
                    rm_c = reduced_C(lc_j, k, lc_i)
                    if rm_c == 0.0:
                        continue

                    total = 0.0
                    for m1_j, m2_j, mc_j, c_j in orb_j_p:
                        for m1_i, m2_i, mc_i, c_i in orb_i_p:
                            m_j = (m1_j, m2_j, mc_j)
                            m_i = (m1_i, m2_i, mc_i)
                            if m_j[b - 1] != m_i[b - 1]:
                                continue
                            ma_j = m_j[a - 1]
                            ma_i = m_i[a - 1]
                            mc_j_coord = m_j[c_coord - 1]
                            mc_i_coord = m_i[c_coord - 1]

                            me = 0.0
                            for q in range(-k, k + 1):
                                me += (
                                    (-1) ** q
                                    * _C_matrix_element(
                                        la_j, ma_j, la_i, ma_i, k, q
                                    )
                                    * _C_matrix_element(
                                        lc_j, mc_j_coord, lc_i, mc_i_coord, k, -q
                                    )
                                )
                            if me != 0.0:
                                total += c_j * c_i * me

                    if total != 0.0:
                        coeff = overall_sign * spin_ov * total
                        accum[shape][k] = accum[shape].get(k, 0.0) + coeff

    # Convert accumulators to sorted lists and drop numerical zeros.
    result: dict[str, list[tuple[int, float]]] = {}
    for key in ("direct_1", "direct_2", "exchange_1", "exchange_2"):
        terms = [
            (k, v)
            for k, v in sorted(accum[key].items())
            if abs(v) > 1e-12
        ]
        result[key] = terms
    return result
