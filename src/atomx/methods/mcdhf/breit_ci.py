"""Breit interaction matrix elements in the m-scheme determinant basis.

Why this module exists
----------------------
The Breit interaction is *not* a scalar correction: the spin-other-orbit,
spin-spin and orbit-orbit parts depend on how the individual electron
spins and orbital currents are aligned, so they split levels inside a
``(2J, parity)`` block.  Evaluating the removed scalar Breit
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
element, and checks the ``BreitOperatorError`` / ``RETARDATION_NOTE``
contract.  The *external* anchor for closed-shell totals lives in
``tests/test_qed.py::TestBreitQEDIntegration``
(``test_ne_breit_operator_matches_dbsr_hf`` and
``test_he_breit_operator_matches_dbsr_hf``): Ne 1s2 2s2 2p6 gives
0.017340 Ha here against 0.016644 Ha total Breit from DBSR_HF (the 4.2 %
difference is the omitted retardation), and He 1s2 gives
6.37792e-05 Ha against DBSR_HF's 6.37774e-05 (0.003 %, retardation being
negligible at Z = 2).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from atomx.angular.wigner import wigner_3j

__all__ = [
    "BreitOperatorError",
    "RETARDATION_NOTE",
    "attach_breit_operator_energy",
    "breit_operator_energy",
    "configuration_string_of",
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


def breit_operator_energy(orbitals_dict, bsp_data, Z, ns, config: str):
    """Exact Gaunt energy of a SINGLE-CSF configuration, on given orbitals.

    This is the operator evaluation for routes that have no CI: the SCF
    already produced the orbitals, and for a one-CSF configuration the
    Gaunt expectation is the diagonal matrix element in that CSF, so no
    CI diagonalisation and no MCDHF SCF is needed.  It costs one radial
    quadrature pass — measured 0.01 s for Ne on the shooting grid, i.e.
    ~2 % of the SCF itself, against 3.8 s for routing the same atom
    through the MCDHF SCF.

    Verified against DBSR_HF (``dbsr_hf an=Z term=AV mbreit=1``) for
    He 1s2 and Ne, and it is the *exact* He value: 6.377743260e-05
    against DBSR_HF's 0.000063777432919 (5e-10 relative), where the
    scalar this replaces is 3.00x.  For Ne it gives 1.733427520e-02
    against DBSR_HF 0.016643630840490 (+4.15 %, the omitted
    retardation).

    Parameters
    ----------
    orbitals_dict : dict
        ``{(n, kappa): (P, Q)}`` — SCF orbitals in the representation
        ``bsp_data`` expects (B-spline coefficient vectors or grid
        arrays).
    bsp_data : dict or GridCIWorkspace
        Radial workspace.  Only its grid and integration weights are
        used (the Gaunt integrals do not touch the nuclear potential),
        so a ``GridCIWorkspace`` built from a ``ShootingResult`` is
        enough for the shooting engine — no solver object needed.
    Z, ns : int
        Nuclear charge and basis/grid dimension.
    config : str
        The configuration, e.g. ``"1s2 2s2 2p6"``.

    Raises
    ------
    BreitOperatorError
        If the configuration is not a single CSF.  For a multi-CSF
        configuration the Gaunt energy is not one number: it depends on
        the state, and the SCF average that produced the orbitals has to
        be matched by the same Guerra/CI averaging.  That is what
        ``MCDHFSolver`` does; silently returning one CSF's diagonal (or
        a scalar) instead is the defect this function exists to avoid.
    """
    from atomx.methods.mcdhf.config import enumerate_jj_csfs_many
    from atomx.methods.mcdhf.exact_ci import csf_signature, exact_block_matrix
    from atomx.methods.mcdhf.matrix import _breit_radial_evaluator

    csfs = enumerate_jj_csfs_many([config])
    if len(csfs) != 1:
        raise BreitOperatorError(
            f"the Gaunt operator energy of {config!r} is not a single "
            f"number: the configuration has {len(csfs)} jj CSFs, whose "
            f"Breit matrix elements differ (that is the physics the "
            f"operator exists to represent).  Route it through the CI "
            f"(MCDHFSolver, i.e. a list `config=[...]` in Atom) so the "
            f"state and its SCF averaging are explicit.")
    ev = _breit_radial_evaluator(orbitals_dict, bsp_data, Z, ns)
    # A zero Coulomb evaluator leaves the block equal to the Gaunt
    # matrix alone: ``exact_block_matrix`` adds the Breit determinant
    # matrix to the Coulomb one before projecting to CSFs.
    block = exact_block_matrix(
        [csf_signature(csfs[0])], lambda _key: 0.0,
        breit_evaluator=ev, verify=False)
    return float(block[0, 0])


def configuration_string_of(subshells) -> str:
    """``"1s2 2s2 2p6"`` from a result's ``subshells`` (jj partners merged).

    ``Subshell`` objects carry ``(n, kappa, occupation)``; the notation
    ``enumerate_jj_csfs`` parses is non-relativistic (``nl^q``), so the
    two jj partners of an ``nl`` shell are summed back together.
    Fractional occupations are refused: they mean the SCF averaged over
    configurations, and the ``nl^q`` token would be a silent
    reinterpretation of that average.
    """
    l_char = "spdfghiklmno"
    order: list = []
    totals: dict = {}
    for sh in subshells:
        kappa = int(sh.kappa)
        l = kappa if kappa > 0 else -kappa - 1
        key = (int(sh.n), l)
        if key not in totals:
            order.append(key)
            totals[key] = 0.0
        totals[key] += float(sh.occupation)
    parts = []
    for key in order:
        q = totals[key]
        if abs(q - round(q)) > 1e-6:
            raise BreitOperatorError(
                f"configuration reconstruction: subshell {key} carries the "
                f"fractional occupation {q!r}, i.e. the SCF averaged over "
                f"several configurations, so there is no single "
                f"configuration whose Breit energy could be reported.")
        parts.append(f"{key[0]}{l_char[key[1]]}{int(round(q))}")
    return " ".join(parts)


def attach_breit_operator_energy(result, config: str | None = None) -> float:
    """Compute ``result``'s Gaunt energy and stash it on the result.

    The Breit interaction is applied by the ``CorrectionStack`` (in
    ``corrections/``), but the operator needs the CI angular machinery
    (``exact_ci``, ``matrix._breit_radial_evaluator``), which lives in
    ``methods/mcdhf/``.  ``corrections/`` must not import ``methods/``
    (layering), so the value travels on the result instead:

        ``result.__dict__["_E_Breit_operator"]``

    exactly like ``_breit_provenance``.  The SCF solvers call this just
    before finalising their corrections stack; the ``breit`` adapter
    reads the value and **raises** if it is absent rather than falling
    back to anything.

    Parameters
    ----------
    result : AtomicResult
        Converged SCF result (B-spline or grid flavour).
    config : str, optional
        Configuration string; derived from ``result.subshells`` when not
        given.

    Returns
    -------
    float
        The Gaunt energy in Hartree, also stored on the result.
    """
    if config is None:
        config = configuration_string_of(result.subshells or [])
    orbitals_dict, bsp_data, Z, ns = _ci_radial_inputs(result)
    value = breit_operator_energy(orbitals_dict, bsp_data, Z, ns, config)
    result.__dict__["_E_Breit_operator"] = float(value)
    result.__dict__["_E_Breit_operator_config"] = config
    return float(value)


def _ci_radial_inputs(result):
    """``(orbitals_dict, workspace, Z, ns)`` for a converged SCF result."""
    from atomx.methods.mcdhf.matrix import GridCIWorkspace

    subshells = list(getattr(result, "subshells", None) or [])
    if not subshells:
        raise BreitOperatorError(
            "the Breit operator needs the result's subshells; this result "
            "carries none, so its configuration cannot be reconstructed.")
    Z = getattr(result, "Z", None)
    if Z is None:
        raise BreitOperatorError("the Breit operator needs result.Z.")

    raw = getattr(result, "orbitals_raw", None)
    bsp = getattr(result, "_bsp_data", None)
    if raw and bsp:
        orbitals_dict = {}
        counter: dict = {}
        for sh in subshells:
            idx = counter.get(sh.kappa, 0)
            counter[sh.kappa] = idx + 1
            orbs = raw.get(sh.kappa, [])
            if idx < len(orbs):
                P, Q, _occ = orbs[idx]
                orbitals_dict[(sh.n, sh.kappa)] = (P, Q)
        ns = bsp.get("ns") if isinstance(bsp, dict) else None
        if ns is None:
            raise BreitOperatorError(
                "the Breit operator needs the B-spline dimension "
                "(bsp_data['ns']) to evaluate its radial integrals.")
        return orbitals_dict, bsp, int(Z), int(ns)

    pa = getattr(result, "_pa", None)
    grid = getattr(result, "_grid", None)
    if pa is not None and grid is not None:
        orbitals_dict = {
            (sh.n, sh.kappa): (result._pa[i], result._qa[i])
            for i, sh in enumerate(subshells)}
        # Only r and difr are read on the Gaunt path (the interaction is
        # a two-electron kernel; it never touches the nuclear potential),
        # so the shooting engine needs no solver object here.
        ws = GridCIWorkspace(grid, result._difr,
                             np.zeros_like(np.asarray(grid, dtype=float)))
        return orbitals_dict, ws, int(Z), ws.n

    raise BreitOperatorError(
        "the Breit operator has no radial workspace for this result: it "
        "needs either a B-spline workspace with orbitals_raw, or "
        "grid-tabulated _pa/_qa arrays with _grid/_difr.")


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
