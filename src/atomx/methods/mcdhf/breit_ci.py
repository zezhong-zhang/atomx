"""Breit interaction matrix elements in the m-scheme determinant basis.

Why this module exists
----------------------
The Breit interaction is *not* a scalar correction: the spin-other-orbit,
spin-spin and orbit-orbit parts depend on how the individual electron
spins and orbital currents are aligned, so they split levels inside a
``(2J, parity)`` block.  Evaluating ``corrections.breit.compute_breit_energy``
once on the averaged orbital set and adding the same number to every CI
state (the historical MCDHF behaviour) therefore shifts every level
identically and leaves fine structure untouched.  Measured on He-like
C V, that path left the 1s2p 3P fine structure at the Dirac-Coulomb
value 13.74 cm-1 against NIST's observed 123.29 cm-1.

This module supplies the missing operator.  The frequency-independent
Breit interaction

    B(1,2) = -alpha_1.alpha_2 / r_12                        (Gaunt)
             + 1/2 [ alpha_1.alpha_2 / r_12
                     - (alpha_1.r_12)(alpha_2.r_12)/r_12^3 ]   (retardation)

is expanded in photon multipoles with
``1/r_12 = sum_L r_<^L / r_>^{L+1} C^L(1).C^L(2)`` and the vector
operators are contracted in the *uncoupled* m-scheme,

    (alpha_1.alpha_2)(C^L_1.C^L_2) =
        sum_{mu,q} (-1)^{mu+q} [alpha_mu C^L_q]_1 [alpha_-mu C^L_-q]_2,

so that every two-body matrix element factorises into a product of two
one-body spinor matrix elements times a Slater-type radial integral
``int int f_a f_c (r_<^L/r_>^{L+1}) f_b f_d`` over the Dirac components
``f in {P, Q}``.  The one-body matrix elements are evaluated by explicit
sums over the ``(l, m_l) x (1/2, m_s)`` decomposition of the spinor
spherical harmonics, so no reduced-matrix-element table is needed and no
convention can drift.

What is included and what is not
--------------------------------
* ``gaunt_terms`` — the **complete Gaunt interaction**, all photon
  multipoles L (the sum terminates by angular selection rules).  This is
  the dominant part of the Breit interaction and carries all of its
  non-scalar content.
* The **retardation** term is *not* implemented here.  It is not
  approximated by zero: callers that ask for it get an explicit
  :class:`BreitOperatorError` (see ``breit_provenance``), because an
  omitted Breit term folded in silently is exactly the defect this
  module replaces.  Its measured size is recorded in
  :data:`RETARDATION_NOTE`.

Validation
----------
``tests/test_breit_operator.py`` checks the angular algebra against the
repo's independent Coulomb kernel, checks a hand-computable Gaunt matrix
element, and compares closed-shell totals with DBSR_HF (the reference
implementation of ``corrections.breit.compute_breit_energy``): Ne
1s2 2s2 2p6 gives 0.017340 Ha here against 0.016644 Ha total Breit from
DBSR_HF — the 4 % difference is the omitted retardation.
"""

from __future__ import annotations

import math
from functools import lru_cache

from atomx.angular.wigner import wigner_3j

__all__ = [
    "BreitOperatorError",
    "RETARDATION_NOTE",
    "gaunt_angular_terms",
    "gaunt_terms",
    "l_of_kappa",
    "twoj_of_kappa",
]


class BreitOperatorError(ValueError):
    """A requested Breit term is not implemented.

    Raised rather than silently returning zero: an omitted Breit term
    changes fine structure, and a zero would hide that.
    """


#: Measured size of the omitted retardation term, from DBSR_HF runs on
#: the same closed-shell systems (full Breit vs the Gaunt operator
#: implemented here):
#:
#:   Ne 1s2            0.012110 (DBSR) vs 0.012114 (Gaunt)   -0.04 %
#:   Ne 1s2 2s2        0.013869 (DBSR) vs 0.014074 (Gaunt)   -1.5 %
#:   Ne 1s2 2s2 2p6    0.016644 (DBSR) vs 0.017340 (Gaunt)   -4.2 %
#:
#: i.e. the retardation *reduces* the Gaunt total by a few per cent for
#: light/medium Z.  For fine structure the relevant comparison is the
#: C V 3P splitting reported by ``MCDHF`` provenance.
RETARDATION_NOTE = (
    "Breit: exact Gaunt operator in the (2J, parity) CI blocks; the "
    "frequency-independent retardation term is NOT included (measured "
    "to be ~1-4 % of the Gaunt total on closed-shell Ne, "
    "corrections.breit.RETARDATION_NOTE)."
)


def twoj_of_kappa(kappa: int) -> int:
    """``2j`` for a Dirac quantum number ``kappa``."""
    return 2 * abs(int(kappa)) - 1


def l_of_kappa(kappa: int) -> int:
    """Orbital angular momentum of the *large* component of ``kappa``."""
    kappa = int(kappa)
    return kappa if kappa > 0 else -kappa - 1


# ---------------------------------------------------------------------------
# Spinor spherical harmonics and Pauli matrices
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _spinor_coeffs(kappa: int, m2: int):
    """``chi_{kappa m} = sum c (Y_{l m_l} xi_{m_s})`` as (c, l, 2m_l, 2m_s)."""
    l = l_of_kappa(kappa)
    j2 = twoj_of_kappa(kappa)
    out = []
    for ml2 in range(-2 * l, 2 * l + 1, 2):
        ms2 = m2 - ml2
        if abs(ms2) != 1:
            continue
        w = wigner_3j(2 * l, 1, j2, ml2, ms2, -m2)
        if w == 0.0:
            continue
        phase = 1.0 if ((2 * l - 1 + m2) // 2) % 2 == 0 else -1.0
        c = phase * math.sqrt(j2 + 1.0) * w
        if c:
            out.append((c, l, ml2, ms2))
    return tuple(out)


@lru_cache(maxsize=None)
def _sigma_mu(mu: int, msa2: int, msb2: int) -> complex:
    """``<m_sa| sigma_mu |m_sb>`` for the spherical Pauli components."""
    sx = ((0, 1), (1, 0))
    sy = ((0, -1j), (1j, 0))
    sz = ((1, 0), (0, -1))
    if mu == 0:
        mat = sz
    elif mu == 1:
        mat = tuple(
            tuple(-(sx[i][j] + 1j * sy[i][j]) / math.sqrt(2.0) for j in range(2))
            for i in range(2)
        )
    elif mu == -1:
        mat = tuple(
            tuple((sx[i][j] - 1j * sy[i][j]) / math.sqrt(2.0) for j in range(2))
            for i in range(2)
        )
    else:
        raise BreitOperatorError(f"alpha_mu needs mu in (-1, 0, 1), got {mu}")
    return mat[0 if msa2 == 1 else 1][0 if msb2 == 1 else 1]


@lru_cache(maxsize=None)
def _clm(l: int, ml2: int, L: int, q2: int, lp: int, mlp2: int) -> float:
    """``<l m_l| C^L_q |l' m_l'>`` with ``C^L = sqrt(4pi/(2L+1)) Y_L``."""
    if ml2 - q2 != mlp2:
        return 0.0
    w = wigner_3j(2 * l, 2 * L, 2 * lp, -ml2, q2, mlp2)
    if w == 0.0:
        return 0.0
    phase = 1.0 if ((2 * l - ml2) // 2) % 2 == 0 else -1.0
    red = ((-1.0) ** l) * math.sqrt((2 * l + 1) * (2 * lp + 1)) * \
        wigner_3j(2 * l, 2 * L, 2 * lp, 0, 0, 0)
    return phase * w * red


@lru_cache(maxsize=None)
def _one_body_terms(kappa_a: int, m2a: int, kappa_c: int, m2c: int,
                    L: int, q2: int, mu: int):
    """Component terms of ``<a| alpha_mu C^L_q |c>``.

    Returns ``((X_a, X_c, coeff), ...)`` with ``X`` in ``{"P", "Q"}``: the
    coefficient multiplies the radial integral ``int f_a f_c dr``.  The
    small component of a Dirac spinor is ``i Q chi_{-kappa}`` in Grant's
    convention (the same one the repo's radial solver integrates), which
    is the source of the relative ``i`` between the two terms.
    """
    out = []
    for (xa, xc, fac) in (("P", "Q", -1j), ("Q", "P", 1j)):
        ka = kappa_a if xa == "P" else -kappa_a
        kc = kappa_c if xc == "P" else -kappa_c
        # Angular element only: the spin-angular factor is independent of
        # the radial functions.
        tot = 0.0
        for ca, la, mla2, msa2 in _spinor_coeffs(ka, m2a):
            for cc, lc, mlc2, msc2 in _spinor_coeffs(kc, m2c):
                sp = _sigma_mu(mu, msa2, msc2)
                if sp == 0:
                    continue
                ang = _clm(la, mla2, L, q2, lc, mlc2)
                if ang == 0.0:
                    continue
                tot += (ca * cc * sp * ang).real
        if tot != 0.0:
            out.append((xa, xc, fac * tot))
    return tuple(out)


# ---------------------------------------------------------------------------
# Two-body terms
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def gaunt_angular_terms(kappa_a: int, m2a: int, kappa_b: int, m2b: int,
                        kappa_c: int, m2c: int, kappa_d: int, m2d: int):
    """``(L, X_a, X_c, X_b, X_d, coeff)`` terms of ``<ab|G|cd>``.

    The radial integral each term multiplies is

        ``int int f_a(r1) f_c(r1) [r_<^L/r_>^{L+1}] f_b(r2) f_d(r2)``

    with ``f = P`` for ``X = "P"`` and ``f = Q`` for ``X = "Q"``.  ``coeff``
    already contains the ``sum_{mu,q}`` angular sum, so the matrix element is
    ``sum_terms coeff * integral``.
    """
    if m2a + m2b != m2c + m2d:
        return ()
    j2a = twoj_of_kappa(kappa_a)
    j2b = twoj_of_kappa(kappa_b)
    j2c = twoj_of_kappa(kappa_c)
    j2d = twoj_of_kappa(kappa_d)
    l_max = min((j2a + j2c) // 2, (j2b + j2d) // 2)
    out = []
    for L in range(l_max + 1):
        qs = range(-2 * L, 2 * L + 1, 2) if L else (0,)
        for q2 in qs:
            for mu in (-1, 0, 1):
                phase = -1.0 if (mu + q2 // 2) % 2 else 1.0
                t1 = _one_body_terms(kappa_a, m2a, kappa_c, m2c, L, q2, mu)
                if not t1:
                    continue
                t2 = _one_body_terms(kappa_b, m2b, kappa_d, m2d,
                                     L, -q2, -mu)
                if not t2:
                    continue
                for (xa, xc, ca) in t1:
                    for (xb, xd, cb) in t2:
                        out.append((L, xa, xc, xb, xd, -phase * ca * cb))
    return tuple(out)


def gaunt_terms(ia, ib, ic, id_, coeff, spinor_labels):
    """Radial-key terms of ``coeff * <ia ib|G|ic id>`` for the exact CI.

    ``spinor_labels`` maps a determinant bit index to ``(n, kappa, 2m)``
    (the layout :mod:`atomx.methods.mcdhf.exact_ci` uses).  Keys have the
    shape ``("g", L, (X_a, n_a, kappa_a), (X_c, n_c, kappa_c),
    (X_b, n_b, kappa_b), (X_d, n_d, kappa_d))``.
    """
    na, ka, ma = spinor_labels[ia]
    nb, kb, mb = spinor_labels[ib]
    nc, kc, mc = spinor_labels[ic]
    nd, kd, md = spinor_labels[id_]
    terms = gaunt_angular_terms(ka, ma, kb, mb, kc, mc, kd, md)
    if not terms:
        return ()
    key_a, key_c = (na, ka), (nc, kc)
    key_b, key_d = (nb, kb), (nd, kd)
    # ``ang`` is a product of two purely imaginary one-body factors, so it
    # is real-valued; take the real part explicitly so the sparse table
    # stays float64 (the values are exact reals, not rounded complex ones).
    return tuple(
        (("g", L, (xa,) + key_a, (xc,) + key_c, (xb,) + key_b, (xd,) + key_d),
         float((coeff * ang).real))
        for (L, xa, xc, xb, xd, ang) in terms)


def retardation_terms(*_args, **_kwargs):
    """Not implemented — the omitted Breit term, refused explicitly."""
    raise BreitOperatorError(
        "the frequency-independent retardation term of the Breit "
        "interaction is not implemented; only the exact Gaunt operator is "
        "available.  It must not be silently dropped: request it and this "
        "raises instead of returning zero.  See "
        "atomx.methods.mcdhf.breit_ci.RETARDATION_NOTE for the measured "
        "size of the omission.")
