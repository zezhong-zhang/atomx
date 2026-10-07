"""Angular coefficients for the two-electron close-coupling expansion.

For a (target, projectile) two-electron LS-coupled basis the two-electron
Coulomb operator ``1/r_{12}`` reduces, channel by channel, to weighted
multipole y^k integrals. Following the Cowan (1981) / Burke (2011) §5.3
convention,

    < t_j c_j; L S | 1/r_{12} | t_i c_i; L S >_A
       = Σ_k [ v_k^d(j i; L) R^k(t_j t_i ; c_j c_i)
             - (-1)^S v_k^x(j i; L) R^k(t_j t_i ; c_i c_j) ]

with reduced angular factors

    v_k^d(j i; L) = (-1)^{l_{t,j} + l_{c,i} + L}
                    * <l_{t,j} || C^k || l_{t,i}>
                    * <l_{c,j} || C^k || l_{c,i}>
                    * { l_{t,j}  l_{c,j}  L  ;
                        l_{c,i}  l_{t,i}  k  }

    v_k^x(j i; L) = (-1)^L
                    * <l_{t,j} || C^k || l_{c,i}>
                    * <l_{c,j} || C^k || l_{t,i}>
                    * { l_{t,j}  l_{c,j}  L  ;
                        l_{t,i}  l_{c,i}  k  }

where ``< l || C^k || l' > = (-1)^l sqrt((2l+1)(2l'+1)) (l k l'; 0 0 0)``
is the reduced matrix element of the renormalised spherical harmonic
``C^k_q = sqrt(4π/(2k+1)) Y_kq``.

The operator action on a channel function ``F_i(r)`` is then

    [V^d F]_j(r) = sum_i sum_k v_k^d(ji;L) y_k^{t_j t_i}(r) F_i(r)
    [W^x F]_j(r) = sum_i sum_k v_k^x(ji;L,S) P_{t_i}(r)
                   * integral P_{t_j}(r') F_i(r')
                              (r_<^k / r_>^{k+1}) dr'

(``v_k^x`` returned by :func:`exchange_terms` already absorbs the
``-(-1)^S`` antisymmetriser sign so callers can simply add the term.)

The non-local exchange term swaps the bound/continuum roles in the
two-electron coordinate integral; the (-1)^S sign carries the singlet/
triplet split. (Singlet S=0 → exchange enters with a -1 prefactor; triplet
S=1 → +1.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from atomx.angular.wigner import wigner_3j, wigner_6j


def reduced_C_LS(l: int, k: int, l_prime: int) -> float:
    """``< l || C^k || l' > = (-1)^l sqrt((2l+1)(2l'+1)) (l k l'; 0 0 0)``.

    LS-coupling convention (integer ``l``); the jj-coupling analogue is
    :func:`atomx.angular.coefficients.reduced_C_jj` — a different function
    with a different phase, do not mix them.
    """
    # Parity selection: l + k + l' must be even, otherwise the 3j with all
    # m = 0 vanishes.
    if (l + k + l_prime) % 2 != 0:
        return 0.0
    threej = wigner_3j(2 * l, 2 * k, 2 * l_prime, 0, 0, 0)
    return ((-1) ** l) * math.sqrt((2 * l + 1) * (2 * l_prime + 1)) * threej


def _sixj(j1: int, j2: int, j3: int, l1: int, l2: int, l3: int) -> float:
    """6-j with integer arguments (no half-integer angular momenta here)."""
    return wigner_6j(2 * j1, 2 * j2, 2 * j3, 2 * l1, 2 * l2, 2 * l3)


@dataclass(frozen=True)
class AngularTerm:
    """One (k, coefficient) pair contributing to a direct or exchange term."""

    k: int
    coefficient: float


def direct_terms(
    l_tj: int, l_cj: int, l_ti: int, l_ci: int, L: int
) -> List[AngularTerm]:
    """Multipole expansion of the direct two-electron CC matrix element.

    Returns ``[(k, v_k^d)]`` over all ``k`` with non-vanishing coefficient.
    """
    if (l_tj + l_ti) % 2 != (l_ci + l_cj) % 2:
        # Conservation of the two-electron parity for a Coulomb operator
        # which conserves each individual ``l + l'`` parity.
        return []
    k_min_t = abs(l_tj - l_ti)
    k_max_t = l_tj + l_ti
    k_min_c = abs(l_cj - l_ci)
    k_max_c = l_cj + l_ci
    k_min = max(k_min_t, k_min_c)
    k_max = min(k_max_t, k_max_c)

    terms: List[AngularTerm] = []
    for k in range(k_min, k_max + 1):
        rm_t = reduced_C_LS(l_tj, k, l_ti)
        if rm_t == 0.0:
            continue
        rm_c = reduced_C_LS(l_cj, k, l_ci)
        if rm_c == 0.0:
            continue
        sj = _sixj(l_tj, l_cj, L, l_ci, l_ti, k)
        if sj == 0.0:
            continue
        coef = ((-1) ** (l_tj + l_ci + L)) * rm_t * rm_c * sj
        if coef != 0.0:
            terms.append(AngularTerm(k=k, coefficient=coef))
    return terms


def exchange_terms(
    l_tj: int, l_cj: int, l_ti: int, l_ci: int, L: int, S: float
) -> List[AngularTerm]:
    """Multipole expansion of the exchange CC matrix element.

    Returns ``[(k, v_k^x)]`` (already including the ``-(-1)^S`` sign so
    callers add ``[K F](r) = sum_k v_k^x P_{t_j}(r) * <P_{t_i}|y^k| F>``).

    ``S`` is the physical total spin of the two active electrons.  For a
    one-electron target this is an integer (0 or 1).  Half-integer total
    spins (composite targets) require the recoupled exchange kernel that is
    implemented under Milestone 2.
    """
    if (l_tj + l_ci) % 2 != (l_cj + l_ti) % 2:
        return []
    if int(round(2 * S)) % 2 != 0:
        raise NotImplementedError(
            "Half-integer total spin requires the recoupled exchange kernel."
        )
    # The two reduced matrix elements connect (l_t_j, l_c_i) and
    # (l_c_j, l_t_i) — different from the direct term.
    k_min = max(abs(l_tj - l_ci), abs(l_cj - l_ti))
    k_max = min(l_tj + l_ci, l_cj + l_ti)

    # Sign comes from the antisymmetriser. With our convention the
    # exchange contribution enters the channel Hamiltonian with sign
    # (-1)^S — so the singlet (S=0) integral has the *same* sign as the
    # direct integral, while the triplet (S=1) exchange has opposite
    # sign. (This reproduces the textbook e–H static-exchange split
    # where the ³S phase shift exceeds the ¹S phase shift.)
    spin_sign = (-1) ** S  # +1 for singlet, -1 for triplet

    terms: List[AngularTerm] = []
    for k in range(k_min, k_max + 1):
        rm1 = reduced_C_LS(l_tj, k, l_ci)
        if rm1 == 0.0:
            continue
        rm2 = reduced_C_LS(l_cj, k, l_ti)
        if rm2 == 0.0:
            continue
        sj = _sixj(l_tj, l_cj, L, l_ti, l_ci, k)
        if sj == 0.0:
            continue
        # The exchange phase derives from the antisymmetriser plus the
        # rewriting of the angular reduction; the overall (-1)^L collects
        # to the form used in Burke 2011 eq. (5.84). See module docstring.
        coef = spin_sign * ((-1) ** L) * rm1 * rm2 * sj
        if coef != 0.0:
            terms.append(AngularTerm(k=k, coefficient=coef))
    return terms
