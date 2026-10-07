"""An m-scheme (Slater-Condon) relativistic CI oracle for the atomx CI matrix.

Purpose
-------
``atomx.physics.csf.matrix.build_ci_matrix`` evaluates CSF matrix
elements through a jj-coupled angular kernel.  This module rebuilds the CI
in the **m-scheme determinant basis** so that the *assembly* of that matrix
has an independent check.

What this oracle is, and is not (2026-09 audit, finding F3)
-----------------------------------------------------------
It is:

* an independent **assembly**: ``H`` is built from explicit Slater-Condon
  rules over determinants, restricted to the determinant space the atomx
  CSF list spans, and diagonalised — no coefficients of fractional
  parentage, no coupling trees, no CSF projection;
* an independent **algebra table**: every 3j symbol comes from
  :mod:`sympy.physics.wigner` (exact rationals), and ``ck_red``/``_w3j``
  are written here rather than imported;
* a check on the **CSF-to-determinant map**: the sorted spectrum of
  ``build_ci_matrix`` and of this construction must coincide.

It is NOT independent of the angular *formula* it certifies, and this
docstring used to claim otherwise:

* ``exact_ci.two_body_angular`` is a line-by-line transcription of
  ``_angular_terms`` below — the same ``q2 % 2`` guard, the same ``kmax``,
  the same ``(-1)**((2j_a-2m_a)/2 + (2j_b-2m_b)/2 + q)`` phase, the same
  term order.  A sign or factor error in that formula is invisible to the
  comparison;
* the determinant sign convention is shared by design
  (``exact_ci._ops_sign`` vs ``_apply_ops`` here);
* the occupation patterns come from the repo's own ``enumerate_jj_csfs``
  (:func:`patterns_from_configs`), and ``matrix._h_one_body``,
  ``_norm_sq`` and ``_rk`` are shared — so the *radial* integrals are the
  same numbers on both sides by construction.

Provenance, for anyone reading the history: this file does not exist at the
fix commit ``04cd274`` (``git ls-tree 04cd274 benchmarks/``) although the
tests that commit added import it; its only commit is ``2007d36``, a
descendant.  It was written in the same working session as the fix it
certifies, so its agreement is not pre-existing external evidence.

The genuinely independent evidence for the CI Hamiltonian lives in
``tests/test_ci_grid_kernel.py`` (exact Condon-Shortley hydrogenic
``F^k``/``G^k`` closed forms), ``tests/test_ci_analytic_slater_condon.py``
(the orbital-free ``E(1P1) - E(3P1) = 2 G^1/3`` identity), the
variational/Hylleraas-Undheim-MacDonald ladder in
``tests/test_first_principles_invariants.py``, and the NIST/DBSR_HF level
comparisons.

What the m-scheme route does need is the same one-electron basis, the same
radial integrals and the same **nuclear potential**: the one-body blocks are
memoised per ``(ns, kappa, Z)`` and take the potential from
``matrix._SLATER_MEMO["_nuclear_potential"]``, so a caller comparing against
a finite-nucleus CI must set that key (at Z = 26 the point-nucleus
difference is 4.4 mHa, four orders above the 1e-8 tolerances used).

The convention implemented here is

    <ab|r12^-1|cd> = sum_k R^k(ac,bd)
        * (-1)^(j_a-m_a+j_b-m_b) * sum_q (-1)^q
        * (j_a k j_c; -m_a q m_c) (j_b k j_d; -m_b -q m_d)
        * <kappa_a||C^k||kappa_c> <kappa_b||C^k||kappa_d>

with ``<kappa_a||C^k||kappa_c>`` in Grant's convention.

Validation of this oracle
-------------------------
* He ``1s2`` and Be ``1s2 2s2`` (one determinant each) reproduce the
  production ``build_ci_matrix`` diagonal — and hence the DFSolver total
  energy — to 1.8e-15 / 7.1e-15 Ha;
* with only ``2p_1/2`` populated, the two ``M = 0`` determinants are
  exactly degenerate, as they must be when there is no spin-orbit
  partner: ``2s_1/2 2p_1/2`` gives ``3P0``/``3P1`` at the same energy
  in the LS limit;
* **rotational invariance** — a scalar Hamiltonian's level energies cannot
  depend on which ``2M`` sector they are computed in, so ``mscheme_levels``
  at ``M2 = 0`` and at ``M2 = 2J`` must agree on every common level.  This
  invariant is what exposed the dropped-exchange-integral bug fixed
  2026-09-21 (see the comment in ``mscheme_levels``): the shortcut made the
  Be ``1s2 2s1 2p1`` J=1 level move by 40 mHa between ``M2 = 0`` and
  ``M2 = 2``.  Anyone re-deriving numbers from this module should keep that
  check in mind.

Usage
-----
>>> from mscheme_ci_reference import (patterns_from_configs,
...                                   mscheme_levels)
>>> pats = patterns_from_configs(["1s2 2s2", "1s2 2p2"])
>>> evals, n_det = mscheme_levels(orbitals_dict, bsp_data, Z=4, ns=ns,
...                               patterns=pats, M2=0)

``orbitals_dict`` / ``bsp_data`` / ``ns`` are exactly the objects handed
to :func:`atomx.methods.ci.ci_solve` (e.g. captured from
``MCDHFSolver``), so the comparison is apples-to-apples: same orbitals,
same space, same radial integrals.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
from sympy import Rational
from sympy import sqrt as _ssqrt
from sympy.physics.wigner import wigner_3j

__all__ = [
    "ck_red",
    "mscheme_levels",
    "patterns_from_configs",
    "twoj",
]


# ---------------------------------------------------------------------------
# Spinor angular data.  ``m`` is stored DOUBLED (2m), so m2 is always odd
# and the half-integer arithmetic stays in integers.
# ---------------------------------------------------------------------------

def twoj(kappa: int) -> int:
    """``2j`` for a Dirac quantum number ``kappa``."""
    return 2 * abs(int(kappa)) - 1


def l_of(kappa: int) -> int:
    """Orbital angular momentum ``l`` for a Dirac ``kappa``."""
    kappa = int(kappa)
    return kappa if kappa > 0 else -kappa - 1


def ck_red(kappa_a: int, kappa_c: int, k: int) -> float:
    """``<kappa_a || C^k || kappa_c>`` (Grant's convention).

    ``C^k_q = sqrt(4 pi / (2k+1)) Y_kq``.  Zero unless parity allows
    (``l_a + l_c + k`` even) and the triangle rule holds.
    """
    ja2, jc2 = twoj(kappa_a), twoj(kappa_c)
    if (l_of(kappa_a) + l_of(kappa_c) + k) % 2:
        return 0.0
    if 2 * k < abs(ja2 - jc2) or 2 * k > ja2 + jc2:
        return 0.0
    val = ((-1) ** (Rational(ja2, 2) + Rational(1, 2))
           * _ssqrt((ja2 + 1) * (jc2 + 1))
           * wigner_3j(Rational(ja2, 2), Rational(jc2, 2), k,
                       Rational(1, 2), Rational(-1, 2), 0))
    return float(val)


def _w3j(j1_2: int, j2_2: int, j3_2: int,
         m1_2: int, m2_2: int, m3_2: int) -> float:
    return float(wigner_3j(Rational(j1_2, 2), Rational(j2_2, 2),
                           Rational(j3_2, 2), Rational(m1_2, 2),
                           Rational(m2_2, 2), Rational(m3_2, 2)))


@lru_cache(maxsize=None)
def _angular_terms(ka, m2a, kb, m2b, kc, m2c, kd, m2d):
    """``(k, angular coefficient)`` pairs for ``<ab|V|cd>``.

    Cached: the angular factor depends only on the spinor labels, so the
    radial work is done once per distinct integral.
    """
    if m2a + m2b != m2c + m2d:
        return ()
    q2 = m2a - m2c
    if q2 % 2:
        return ()
    q = q2 // 2
    out = []
    kmax = min((twoj(ka) + twoj(kc)) // 2, (twoj(kb) + twoj(kd)) // 2)
    for k in range(kmax + 1):
        ca, cb = ck_red(ka, kc, k), ck_red(kb, kd, k)
        if ca == 0.0 or cb == 0.0:
            continue
        t1 = _w3j(twoj(ka), 2 * k, twoj(kc), -m2a, q2, m2c)
        t2 = _w3j(twoj(kb), 2 * k, twoj(kd), -m2b, -q2, m2d)
        if t1 == 0.0 or t2 == 0.0:
            continue
        phase = (-1) ** ((twoj(ka) - m2a) // 2
                         + (twoj(kb) - m2b) // 2 + q)
        out.append((k, phase * t1 * t2 * ca * cb))
    return tuple(out)


# ---------------------------------------------------------------------------
# Determinant machinery
# ---------------------------------------------------------------------------

def _apply_ops(occupied, ops):
    """Apply ``ops`` right-to-left to a determinant; return (sign, occ).

    ``ops`` is a list of ``(spinor_index, create)``.  ``(0, None)`` means
    the operator annihilated an empty spinor or created on an occupied
    one, i.e. the determinant vanishes.
    """
    occ = set(occupied)
    sign = 1
    for idx, create in reversed(ops):
        if create:
            if idx in occ:
                return 0, None
            sign *= (-1) ** sum(1 for o in occ if o < idx)
            occ.add(idx)
        else:
            if idx not in occ:
                return 0, None
            sign *= (-1) ** sum(1 for o in occ if o < idx)
            occ.discard(idx)
    return sign, occ


def _spinors(orbitals_dict):
    out = []
    for (n, kappa) in sorted(orbitals_dict):
        for m2 in range(-twoj(kappa), twoj(kappa) + 1, 2):
            out.append((n, kappa, m2))
    return out


def patterns_from_configs(config_strings):
    """Occupation patterns of every jj-CSF of every NR configuration.

    Returns a list of ``((n, kappa, q), ...)`` tuples — the per-configuration
    determinant spaces whose union is the CI space.
    """
    from atomx.config.csf import enumerate_jj_csfs

    pats = []
    for cfg in config_strings:
        for csf in enumerate_jj_csfs(cfg):
            t = tuple(sorted((sh.n, sh.kappa, int(round(sh.occupation)))
                             for sh in csf.subshells))
            if t not in pats:
                pats.append(t)
    return pats


def mscheme_levels(orbitals_dict, bsp_data, Z, ns, patterns, M2=0):
    """Exact CI level energies in the m-scheme basis of ``patterns``.

    Parameters
    ----------
    orbitals_dict, bsp_data, ns, Z
        Exactly what is handed to :func:`atomx.methods.ci.ci_solve`.
    patterns
        Occupation patterns, e.g. from :func:`patterns_from_configs`.
    M2
        Total ``2M`` of the determinant basis.  ``M2 = 0`` collects every
        level with integer ``J``; use ``M2 = 2J`` to isolate a block.

    Returns
    -------
    (eigenvalues, n_determinants)
        Ascending eigenvalues and the determinant-space dimension.
    """
    import atomx.physics.csf.matrix as MAT

    spinors = _spinors(orbitals_dict)
    owner = [(n, kappa) for (n, kappa, _m2) in spinors]
    patset = {tuple(sorted(p)) for p in patterns}
    n_elec = sum(q for (_n, _k, q) in patterns[0])

    dets = []
    for combo in itertools.combinations(range(len(spinors)), n_elec):
        counts = {}
        for i in combo:
            counts[owner[i]] = counts.get(owner[i], 0) + 1
        t = tuple(sorted((key[0], key[1], q) for key, q in counts.items() if q))
        if t not in patset:
            continue
        if M2 is not None and sum(spinors[i][2] for i in combo) != M2:
            continue
        dets.append(combo)

    index = {d: i for i, d in enumerate(dets)}
    H = np.zeros((len(dets), len(dets)))

    def add(col, ops, coef):
        if coef == 0.0:
            return
        sign, occ = _apply_ops(dets[col], ops)
        if not sign:
            return
        row = index.get(tuple(sorted(occ)))
        if row is not None:
            # Terms leaving the truncated space are dropped, exactly as
            # they are in the CSF representation.
            H[row, col] += sign * coef

    for col, det in enumerate(dets):
        # one-body: sum_ab h_ab a+_a a_b
        for c in det:
            for a in range(len(spinors)):
                na, ka, m2a = spinors[a]
                nc, kc, m2c = spinors[c]
                if ka != kc or m2a != m2c:
                    continue
                Pa, Qa = orbitals_dict[(na, ka)]
                Pc, Qc = orbitals_dict[(nc, kc)]
                h = MAT._h_one_body(Pa, Qa, Pc, Qc, ka, Z, bsp_data, ns)
                h /= MAT._norm_sq(Pc, Qc, kc, Z, bsp_data, ns)
                add(col, [(a, True), (c, False)], h)
        # two-body: sum_{p<q, r<s} (<pq|V|rs> - <pq|V|sr>) a+_p a+_q a_s a_r
        for c, d in itertools.combinations(det, 2):
            for a, b in itertools.combinations(range(len(spinors)), 2):
                v = _v_element(spinors[a], spinors[b], spinors[c], spinors[d],
                               orbitals_dict, bsp_data, Z, ns)
                v2 = _v_element(spinors[a], spinors[b], spinors[d], spinors[c],
                                orbitals_dict, bsp_data, Z, ns)
                # The antisymmetrised element is ``v - v2``: BOTH integrals
                # must be evaluated before the pair can be skipped.  This
                # used to read ``if v == 0.0: continue``, which dropped the
                # exchange integral whenever the *direct* one vanished by
                # angular selection rules.  That is not a symmetry — the
                # exchange integral is a different Slater integral — and the
                # error is visible as a breakdown of rotational invariance:
                # for Be 1s2 2s1 2p1 the J=1 level came out at -14.490934 Ha
                # from the 2M=0 block but -14.530753 Ha from the 2M=2 block,
                # while the exact (and the 2M=4, one-determinant) value is
                # -14.513834 Ha.  A scalar Hamiltonian cannot depend on the
                # 2M sector, so the shortcut was wrong; measurements quoted
                # from this oracle before 2026-09-21 that touch open-shell
                # off-diagonal elements (Be singles -14.595092, Be SD
                # fac_frozen -14.654637) carry the error.
                if v == 0.0 and v2 == 0.0:
                    continue
                add(col, [(a, True), (b, True), (d, False), (c, False)],
                    v - v2)
    return np.linalg.eigvalsh(H), len(dets)


def _v_element(a, b, c, d, orbitals_dict, bsp_data, Z, ns):
    """``<a b|1/r12|c d>`` for spinor tuples ``(n, kappa, 2m)``."""
    import atomx.physics.csf.matrix as MAT

    na, ka, m2a = a
    nb, kb, m2b = b
    nc, kc, m2c = c
    nd, kd, m2d = d
    terms = _angular_terms(ka, m2a, kb, m2b, kc, m2c, kd, m2d)
    if not terms:
        return 0.0
    Pa, Qa = orbitals_dict[(na, ka)]
    Pc, Qc = orbitals_dict[(nc, kc)]
    Pb, Qb = orbitals_dict[(nb, kb)]
    Pd, Qd = orbitals_dict[(nd, kd)]
    total = 0.0
    for k, coef in terms:
        total += coef * MAT._rk(Pa, Qa, Pc, Qc, Pb, Qb, Pd, Qd, bsp_data, k)
    return total
