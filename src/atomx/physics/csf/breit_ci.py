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

The complete static operator
----------------------------
``breit_terms`` combines the complete Gaunt multipoles with retardation
from transition-current divergences (see ``retardation.py``). The vector
alpha permits photon L up to j_a+j_c+1, not just j_a+j_c. All terms enter
the determinant matrix before projection to J-coupled CSFs, including
configuration interference. This is the frequency-independent Breit
approximation; frequency-dependent transverse exchange is not included.

Validation includes independent DBSR_HF closed-shell totals, GRASP RCI,
and analytic multipole/current identities.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.recoupling import l_of_kappa, twoj_of_kappa
from atomx.angular.wigner import wigner_3j
from atomx.config.subshell import L_LETTERS

__all__ = [
    "BreitOperatorError",
    "RETARDATION_NOTE",
    "attach_breit_operator_energy",
    "breit_configuration_average",
    "breit_operator_energy",
    "configuration_string_of",
    "gaunt_angular_terms",
    "gaunt_terms",
    "breit_terms",
    "retardation_terms",
    "l_of_kappa",
    "twoj_of_kappa",
]


class BreitOperatorError(ValueError):
    """A requested Breit term is not implemented.

    Raised rather than silently returning zero: an omitted Breit term
    changes fine structure, and a zero would hide that.
    """


RETARDATION_NOTE = (
    "Breit: complete Gaunt plus frequency-independent retardation in "
    "the (2J, parity) CI blocks, evaluated on Dirac-Coulomb optimized "
    "orbitals. Frequency-dependent transverse exchange is not included."
)



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
        c = clebsch_gordan(2 * l, ml2, 1, ms2, j2, m2)
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
    # alpha is rank 1: a photon L can couple to electronic rank L-1.
    l_max = min((j2a + j2c) // 2, (j2b + j2d) // 2) + 1
    out = {}
    for L in range(l_max + 1):
        # The vector-spinor matrix element conserves m_a-m_c=q+mu.
        # Only three (q,mu) pairs can contribute, independently of L.
        # Descending mu retains the original ascending-q summation order.
        for mu in (1, 0, -1):
            q2 = m2a - m2c - 2 * mu
            if abs(q2) <= 2 * L:
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
                        key = (L, xa, xc, xb, xd)
                        out[key] = out.get(key, 0j) - phase * ca * cb
    # All q,mu contributions with this key multiply the same radial
    # integral. Contract once in angular space, before principal quantum
    # numbers and determinant signs duplicate the terms across CI blocks.
    # Only exact zeros are removed: no angular or radial cutoff is added.
    return tuple((*key, value) for key, value in out.items() if value != 0.)


def breit_operator_energy(orbitals_dict, bsp_data, Z, ns, config: str, *,
                          orbital_energies=None):
    """Full static Breit energy of a SINGLE-CSF configuration, on given orbitals.

    This is the operator evaluation for routes that have no CI: the SCF
    already produced the orbitals, and for a one-CSF configuration the
    static Breit expectation is the diagonal matrix element in that CSF, so no
    CI diagonalisation and no MCDHF SCF is needed.  It costs one radial
    quadrature pass — measured 0.01 s for Ne on the shooting grid, i.e.
    ~2 % of the SCF itself, against 3.8 s for routing the same atom
    through the MCDHF SCF.

    Independent DBSR_HF closed-shell references are checked in
    ``tests/test_static_breit_retardation.py`` for He, Ne and Ar.
    The Ne value includes static retardation (~0.016643631 Ha), rather
    than the older incomplete Gaunt-only result.

    Parameters
    ----------
    orbitals_dict : dict
        ``{(n, kappa): (P, Q)}`` — SCF orbitals in the representation
        ``bsp_data`` expects (B-spline coefficient vectors or grid
        arrays).
    bsp_data : dict or GridCIWorkspace
        Radial workspace.  Only its grid and integration weights are
        used (the static Breit integrals do not touch the nuclear potential),
        so a ``GridCIWorkspace`` built from a ``ShootingResult`` is
        enough for the shooting engine — no solver object needed.
    Z, ns : int
        Nuclear charge and basis/grid dimension.
    config : str
        The configuration, e.g. ``"1s2 2s2 2p6"``.
    orbital_energies : mapping, optional
        With a complete ``(n, kappa) -> Hartree`` map, replace static Breit
        by real finite-frequency transverse exchange on these spinors.

    Raises
    ------
    BreitOperatorError
        If the configuration is not a single CSF.  For a multi-CSF
        configuration the static Breit energy is not one number: it depends on
        the state, and the SCF average that produced the orbitals has to
        be matched by the same Guerra/CI averaging.  That is what
        ``MCDHFSolver`` does; silently returning one CSF's diagonal (or
        a scalar) instead is the defect this function exists to avoid.
    """
    from atomx.angular.csf_hamiltonian import csf_signature, exact_block_matrix
    from atomx.config.csf import enumerate_jj_csfs_many
    from atomx.physics.csf.matrix import _breit_radial_evaluator

    csfs = enumerate_jj_csfs_many([config])
    if len(csfs) != 1:
        raise BreitOperatorError(
            f"the static Breit operator energy of {config!r} is not a single "
            f"number: the configuration has {len(csfs)} jj CSFs, whose "
            f"Breit matrix elements differ (that is the physics the "
            f"operator exists to represent).  Route it through the CI "
            f"(MCDHFSolver, i.e. a list `config=[...]` in Atom) so the "
            f"state and its SCF averaging are explicit.")
    if orbital_energies is None:
        ev = _breit_radial_evaluator(orbitals_dict, bsp_data, Z, ns)
    else:
        from atomx.physics.csf.transverse import radial_evaluator
        ev = radial_evaluator(orbitals_dict, bsp_data, orbital_energies)
    # A zero Coulomb evaluator leaves the block equal to the static Breit
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
    order: list = []
    totals: dict = {}
    for sh in subshells:
        kappa = int(sh.kappa)
        l = l_of_kappa(kappa)
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
        parts.append(f"{key[0]}{L_LETTERS[key[1]]}{int(round(q))}")
    return " ".join(parts)


def breit_configuration_average(orbitals_dict, bsp_data, Z, ns, jj_configs, *,
                                orbital_energies=None):
    """Transverse expectation in an explicit, spherical jj-configuration ensemble.

    With an orbital-energy mapping use the finite-frequency operator;
    otherwise use full static Breit. The ensemble contraction is shared.

    Within each integer-occupation jj configuration all magnetic determinants
    have equal probability. A distinct spinor pair in shell i has probability
    q_i(q_i-1)/(g_i(g_i-1)); in different shells it is q_i*q_j/(g_i*g_j).
    Contract these probabilities with <ab|G|ab>-<ab|G|ba>. This is the trace
    of the same two-body operator used by CI, without constructing any CSFs
    or determinant matrices. Configuration weights retain correlations
    between the occupations of jj partners (products of mean occupations
    would be incorrect).
    """
    from collections import defaultdict

    from atomx.physics.csf.matrix import _breit_radial_evaluator

    configs = list(jj_configs)
    if not configs:
        raise BreitOperatorError("the static Breit ensemble needs jj configurations")
    shells = configs[0].subshells
    keys = [(sh.n, sh.kappa) for sh in shells]
    weights = np.asarray([cfg.weight for cfg in configs], dtype=float)
    if (not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)):
        raise BreitOperatorError("static Breit ensemble weights must sum to one")
    degeneracy = np.asarray([2 * abs(sh.kappa) for sh in shells])
    pair_counts = np.zeros((len(shells), len(shells)))
    for cfg in configs:
        if [(sh.n, sh.kappa) for sh in cfg.subshells] != keys:
            raise BreitOperatorError("static Breit ensemble subshells must be aligned")
        q = np.asarray([sh.occupation for sh in cfg.subshells], dtype=float)
        if (not np.all(np.isfinite(q)) or np.any(q < 0)
                or np.any(q > degeneracy) or np.any(q != np.rint(q))):
            raise BreitOperatorError("static Breit ensemble needs integer occupations")
        counts = np.outer(q, q)
        counts[np.diag_indices_from(counts)] -= q
        pair_counts += cfg.weight * counts
    denominator = np.outer(degeneracy, degeneracy)
    denominator[np.diag_indices_from(denominator)] -= degeneracy
    probabilities = pair_counts / denominator

    labels = []
    shell_indices = []
    for i, (n, kappa) in enumerate(keys):
        for m2 in range(-twoj_of_kappa(kappa), twoj_of_kappa(kappa) + 1, 2):
            labels.append((n, kappa, m2))
            shell_indices.append(i)
    coefficients = defaultdict(float)
    for a, i in enumerate(shell_indices):
        for b in range(a):
            probability = probabilities[i, shell_indices[b]]
            if probability == 0:
                continue
            for c, d, sign in ((a, b, 1.0), (b, a, -1.0)):
                for key, coeff in breit_terms(a, b, c, d, sign, labels):
                    coefficients[key] += probability * coeff
    if orbital_energies is None:
        evaluate = _breit_radial_evaluator(orbitals_dict, bsp_data, Z, ns)
    else:
        from atomx.physics.csf.transverse import radial_evaluator
        evaluate = radial_evaluator(orbitals_dict, bsp_data, orbital_energies)
    return math.fsum(coeff * evaluate(key) for key, coeff in coefficients.items()
                     if coeff != 0.0)


def attach_breit_operator_energy(result, config: str | None = None,
                                *, jj_configs=None) -> float:
    """Compute ``result``'s static Breit energy and stash it on the result.

    The static operator backend of ``CorrectionStack`` reads a precomputed
    expectation. Angular contractions live in ``angular/`` and radial
    operators in ``physics/csf/``; no numerical or correction module imports
    an SCF method. The expectation is stored as:

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
    jj_configs : sequence of JJConfig, optional
        Explicit spherical configuration ensemble used by the SCF. Evaluate
        its static Breit trace directly; no state-specific CI is implied.

    Returns
    -------
    float
        The static Breit energy in Hartree, also stored on the result.
    """
    if config is None and jj_configs is None:
        config = configuration_string_of(result.subshells or [])
    orbitals_dict, bsp_data, Z, ns = _ci_radial_inputs(result)
    if jj_configs is None:
        value = breit_operator_energy(orbitals_dict, bsp_data, Z, ns, config)
    else:
        value = breit_configuration_average(
            orbitals_dict, bsp_data, Z, ns, jj_configs)
        result.__dict__["_E_Breit_operator_averaging"] = "jj_configuration_ensemble"
    result.__dict__["_E_Breit_operator"] = float(value)
    result.__dict__["_E_Breit_operator_config"] = config
    return float(value)


def _ci_radial_inputs(result):
    """``(orbitals_dict, workspace, Z, ns)`` for a converged SCF result."""
    from atomx.physics.csf.matrix import GridCIWorkspace

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
        # Only r and difr are read on the static Breit path (the interaction is
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
    (the layout :mod:`atomx.angular.csf_hamiltonian` uses).  Keys have the
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


def retardation_terms(*args, **kwargs):
    """Static Breit retardation terms in the same determinant layout as Gaunt."""
    from atomx.physics.csf.retardation import retardation_terms as terms
    return terms(*args, **kwargs)


def breit_terms(*args, **kwargs):
    """Complete frequency-independent Breit: Gaunt plus retardation."""
    return gaunt_terms(*args, **kwargs) + retardation_terms(*args, **kwargs)
