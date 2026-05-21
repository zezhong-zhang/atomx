"""CSF-level angular selection and FAC-compatible population factors.

This module is the shared home for angular logic that depends on complete
configuration-state functions rather than a single orbital.  It deliberately
keeps radial integrals out of the API so structure, transitions, scattering,
and autoionization can reuse the same CSF tensor rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atomx.angular.coefficients import exchange_k_range
from atomx.angular.coefficients import exchange_lambda


@dataclass(frozen=True)
class OneBodyCsfCoupling:
    """Angular/population metadata for a one-body CSF transition."""

    source: tuple[int, int]
    destination: tuple[int, int]
    source_occupation: float
    destination_occupation: float
    population_factor: float


@dataclass(frozen=True)
class TwoBodyCsfCoupling:
    """Metadata for a two-body Coulomb coupling between CSFs.

    ``sources`` are the two subshells depleted in the ket; ``destinations`` are
    the two subshells populated in the bra.  ``direct_k`` corresponds to
    pairings source[0]->destination[0], source[1]->destination[1]; ``exchange_k``
    corresponds to source[0]->destination[1], source[1]->destination[0].
    """

    sources: tuple[tuple[int, int], tuple[int, int]]
    destinations: tuple[tuple[int, int], tuple[int, int]]
    direct_k: tuple[int, ...]
    exchange_k: tuple[int, ...]


def csf_twice_j(csf) -> int:
    """Infer twice the total J for a CSF.

    Explicit J-resolved CSFs carry ``total_J2`` and labels such as
    ``... J=0`` (legacy form) or ``... [J=0/2]`` (current
    ``enumerate_jj_csfs`` form, including the J=0 closed-shell case).
    Legacy compact CSFs may carry neither; for those, this falls back
    to the open-shell estimate used by the older transition calculator.
    """
    explicit = int(getattr(csf, "total_J2", 0))
    if csf_has_explicit_j(csf):
        return explicit

    open_shells = []
    for sh in getattr(csf, "subshells", []):
        j2 = 2 * abs(sh.kappa) - 1
        capacity = j2 + 1
        if 1e-6 < sh.occupation < capacity - 1e-6:
            open_shells.append((sh, j2))
    if not open_shells:
        return 0
    _, j2 = min(
        open_shells,
        key=lambda item: abs(item[0].occupation - (item[1] + 1) / 2.0),
    )
    return int(j2)


def csf_parity(csf) -> int:
    """Return CSF parity as +1 or -1."""
    return int(getattr(csf, "parity", 1))


def csf_has_explicit_j(csf) -> bool:
    """Whether a CSF carries explicit total-J metadata.

    Accepts either a non-default ``total_J2`` attribute or a label
    containing ``J=`` in either the legacy form ``... J=2/2`` or the
    bracketed form ``... [J=2/2]`` emitted by
    :func:`atomx.methods.mcdhf.config.enumerate_jj_csfs`.  The bracketed
    form is essential for closed-shell J=0 CSFs whose ``total_J2`` is
    zero — without this they were silently bypassing the many-electron
    multipole selection-rule filter in the MCDHF transition path.
    """
    if int(getattr(csf, "total_J2", 0)) > 0:
        return True
    label = str(getattr(csf, "label", ""))
    return " J=" in label or "[J=" in label


def multipole_allows_state_transition(tj_i: int, parity_i: int,
                                      tj_f: int, parity_f: int,
                                      L: int, operator_type: str) -> bool:
    """Many-electron J/parity selection rule for electric or magnetic tensors."""
    tL = 2 * L
    if tL < abs(tj_i - tj_f) or tL > tj_i + tj_f:
        return False
    if tj_i == 0 and tj_f == 0:
        return False
    parity_product = parity_i * parity_f
    if operator_type == "E":
        return parity_product == (1 if L % 2 == 0 else -1)
    if operator_type == "M":
        return parity_product == (-1 if L % 2 == 0 else 1)
    return False


def subshell_capacity(kappa: int) -> int:
    """Return the relativistic subshell capacity 2j+1."""
    return 2 * abs(kappa)


def fac_uta_population_factor(source_occ: float, dest_occ: float,
                              source_kappa: int, dest_kappa: int) -> float:
    """FAC UTA population factor for a one-body promotion.

    FAC's configuration-average transition path multiplies the one-electron
    reduced matrix element by

        sqrt(q_source * vacancies_dest / (g_source * g_dest))

    where ``g = 2j+1``.  This keeps closed-shell -> empty-shell lines at
    unit strength while correctly suppressing partially occupied or blocked
    channels.
    """
    g_source = subshell_capacity(source_kappa)
    g_dest = subshell_capacity(dest_kappa)
    if g_source <= 0 or g_dest <= 0:
        return 0.0
    q_source = max(0.0, float(source_occ))
    vacancies_dest = max(0.0, g_dest - float(dest_occ))
    if q_source <= 0.0 or vacancies_dest <= 0.0:
        return 0.0
    return float(np.sqrt(q_source * vacancies_dest / (g_source * g_dest)))


def one_body_csf_coupling(csf_bra, csf_ket, L: int,
                          operator_type: str) -> OneBodyCsfCoupling | None:
    """Return one-body CSF coupling metadata, or ``None`` if forbidden.

    ``csf_ket`` is the initial CSF and ``csf_bra`` is the final CSF.  The two
    CSFs must differ by exactly one electron promoted from a source subshell to
    a destination subshell.  Explicit J-resolved CSFs are additionally filtered
    by many-electron tensor selection rules.
    """
    if csf_has_explicit_j(csf_bra) and csf_has_explicit_j(csf_ket):
        if not multipole_allows_state_transition(
            csf_twice_j(csf_ket), csf_parity(csf_ket),
            csf_twice_j(csf_bra), csf_parity(csf_bra),
            L, operator_type,
        ):
            return None

    occ_bra = {(sh.n, sh.kappa): sh.occupation for sh in csf_bra.subshells}
    occ_ket = {(sh.n, sh.kappa): sh.occupation for sh in csf_ket.subshells}
    diffs = []
    for key in set(occ_bra) | set(occ_ket):
        q_bra = occ_bra.get(key, 0.0)
        q_ket = occ_ket.get(key, 0.0)
        if abs(q_bra - q_ket) > 0.01:
            diffs.append((key, q_bra, q_ket))
    if len(diffs) != 2:
        return None

    (key_a, q_bra_a, q_ket_a), (key_b, q_bra_b, q_ket_b) = diffs
    if q_bra_a < q_ket_a:
        source = key_a
        destination = key_b
        source_occ = q_ket_a
        dest_occ = q_ket_b
    else:
        source = key_b
        destination = key_a
        source_occ = q_ket_b
        dest_occ = q_ket_a

    population = fac_uta_population_factor(
        source_occ, dest_occ, source[1], destination[1],
    )
    if population <= 0.0:
        return None
    return OneBodyCsfCoupling(
        source=source,
        destination=destination,
        source_occupation=float(source_occ),
        destination_occupation=float(dest_occ),
        population_factor=population,
    )


def _expanded_occupation_changes(csf_bra, csf_ket, tol: float = 0.01):
    occ_bra = {(sh.n, sh.kappa): sh.occupation for sh in csf_bra.subshells}
    occ_ket = {(sh.n, sh.kappa): sh.occupation for sh in csf_ket.subshells}
    sources = []
    destinations = []
    for key in sorted(set(occ_bra) | set(occ_ket)):
        delta = occ_bra.get(key, 0.0) - occ_ket.get(key, 0.0)
        if abs(delta) <= tol:
            continue
        n = int(round(abs(delta)))
        if abs(abs(delta) - n) > tol:
            return None, None
        if delta < 0:
            sources.extend([key] * n)
        else:
            destinations.extend([key] * n)
    return sources, destinations


def _intersect_k_ranges(kappa_a: int, kappa_b: int,
                        kappa_c: int, kappa_d: int) -> tuple[int, ...]:
    left = set(exchange_k_range(kappa_a, kappa_b))
    right = set(exchange_k_range(kappa_c, kappa_d))
    return tuple(sorted(left & right))


def coulomb_pair_k_values(source_1: tuple[int, int], source_2: tuple[int, int],
                          dest_1: tuple[int, int], dest_2: tuple[int, int],
                          exchange: bool = False) -> tuple[int, ...]:
    """Allowed Coulomb multipoles for a two-electron orbital pairing.

    The scalar Coulomb operator couples two one-electron spherical tensors of
    the same rank k.  This helper only enforces orbital angular selection; the
    complete CSF coefficient is supplied by the RCFP/recoupling layer.
    """
    if exchange:
        return _intersect_k_ranges(source_1[1], dest_2[1], source_2[1], dest_1[1])
    return _intersect_k_ranges(source_1[1], dest_1[1], source_2[1], dest_2[1])


def orbital_tensor_coupling_coeff(left_a: tuple[int, int], left_b: tuple[int, int],
                                  right_a: tuple[int, int], right_b: tuple[int, int],
                                  k: int) -> float:
    """Reduced orbital tensor coefficient for a Coulomb pair rank.

    This is the common reduced one-electron tensor magnitude used as the
    process-independent pre-factor before the full RCFP layer supplies term-
    specific signs and parentage coefficients.  It replaces earlier ad-hoc
    "direct k=0" handling with the same rank-normalized convention used by
    the average-configuration exchange coefficients.
    """
    if k not in exchange_k_range(left_a[1], left_b[1]):
        return 0.0
    if k not in exchange_k_range(right_a[1], right_b[1]):
        return 0.0
    lam_left = exchange_lambda(left_a[1], left_b[1], k)
    lam_right = exchange_lambda(right_a[1], right_b[1], k)
    if lam_left <= 1e-15 or lam_right <= 1e-15:
        return 0.0
    g = (
        subshell_capacity(left_a[1])
        * subshell_capacity(left_b[1])
        * subshell_capacity(right_a[1])
        * subshell_capacity(right_b[1])
    )
    if g <= 0:
        return 0.0
    return float(np.sqrt(lam_left * lam_right / g))


def two_body_csf_coupling(csf_bra, csf_ket) -> TwoBodyCsfCoupling | None:
    """Return orbital/rank metadata for a two-body CSF coupling.

    This is the process-independent front door for Coulomb Hamiltonian and
    Auger/DR matrix elements.  It detects exactly two-electron changes and
    returns the allowed direct/exchange k ranks.  Full FAC-equivalent numerical
    coefficients still require the RCFP layer; callers should treat empty rank
    lists as a forbidden coupling.
    """
    if csf_has_explicit_j(csf_bra) and csf_has_explicit_j(csf_ket):
        if csf_twice_j(csf_bra) != csf_twice_j(csf_ket):
            return None
        if csf_parity(csf_bra) != csf_parity(csf_ket):
            return None

    sources, destinations = _expanded_occupation_changes(csf_bra, csf_ket)
    if sources is None or len(sources) != 2 or len(destinations) != 2:
        return None

    direct_k = coulomb_pair_k_values(sources[0], sources[1], destinations[0], destinations[1])
    exchange_k = coulomb_pair_k_values(
        sources[0], sources[1], destinations[0], destinations[1],
        exchange=True,
    )
    if not direct_k and not exchange_k:
        return None
    return TwoBodyCsfCoupling(
        sources=(sources[0], sources[1]),
        destinations=(destinations[0], destinations[1]),
        direct_k=direct_k,
        exchange_k=exchange_k,
    )
