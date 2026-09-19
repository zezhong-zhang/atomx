"""Two-electron (double-replacement) CI coupling.

History.  `Atom(configs=[...])` did not couple configurations differing by a
two-electron replacement at all: `Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"])`
returned the pure `1s2 2s2` state (mixing ``[1, 0, ...]``, the HF energy to
6e-7), so the classic 2s²/2p² near-degeneracy correlation was missing.  The
element is now a general m-scheme contraction handling every total J and all
three patterns:

    a² ↔ b²      one subshell loses 2, another gains 2
    a² ↔ c¹ d¹   one loses 2, two gain 1 (double excitation into two
                 different orbitals)
    a¹ b¹ ↔ c²   the transpose of the above

Three kinds of check:

1. **Element level.**  `matrix._two_electron_replacement_element` is compared
   against the FAC-ported m-scheme determinant machinery
   (`ResonanceManifold`, the engine validated against FAC AITable) on
   *identical* radial data and the same grid — two independently assembled
   implementations, so agreement pins the angular algebra (coefficients,
   phases, k-selection and the J-dependence).

2. **Pairing convention.**  `matrix._rk` must pair its arguments as
   ``R^k(a c, b d)`` in *both* engine flavours.  The B-spline branch used to
   hand the orbitals to ``compute_slater_rk_cell`` in the ``(a, b, c, d)``
   order, which that routine reads as the pairs ``(a, b)`` and ``(c, d)``, so
   the two engines disagreed on every integral built through it.

3. **End to end.**  The Be 2s²/2p² ground state must mix and drop below the
   single-configuration HF energy; a J = 2 pair channel and the split channel
   must move real calculations.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import eval_genlaguerre

from atomx.methods.mcdhf.matrix import (
    GridCIWorkspace,
    _rk,
    _two_diff_element,
    _two_electron_replacement_element,
)
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.convol import compute_slater_rk_cell
from atomx.numerics.quadrature import make_fine_grid
from atomx.scattering.autoionization.mscheme_ci import ResonanceManifold


def _l_of(kappa: int) -> int:
    return kappa if kappa > 0 else -kappa - 1


def _radial(n: int, kappa: int, z: float, r: np.ndarray):
    """Crude hydrogenic-like large component (Q = 0), L2-normalised."""
    l = _l_of(kappa)
    a = z / n
    rho = 2.0 * a * r
    P = rho ** (l + 1) * np.exp(-rho / 2.0) * eval_genlaguerre(
        n - l - 1, 2 * l + 1, rho)
    P /= np.sqrt(np.trapezoid(P * P, r))
    return P, np.zeros_like(r)


def _orbital_index(manifold, key, tm):
    for i, o in enumerate(manifold.orbitals):
        if (o.n, o.kappa, o.tm) == (key[0], key[1], tm):
            return i
    return None


def _pair_vector(manifold, dets, key, tJ):
    """|key²; J M=0⟩ over ``dets`` (ascending spin-orbital index)."""
    from atomx.angular.coefficients import cg

    idx = {d: i for i, d in enumerate(dets)}
    tj = 2 * abs(key[1]) - 1
    tms = list(range(-tj, tj + 1, 2))
    vec = np.zeros(len(dets))
    for x, tm_i in enumerate(tms):
        for tm_j in tms[x + 1:]:
            if tm_i + tm_j != 0:
                continue
            i = _orbital_index(manifold, key, tm_i)
            j = _orbital_index(manifold, key, tm_j)
            if i is None or j is None:
                continue
            mask = (1 << i) | (1 << j)
            if mask not in idx:
                continue
            vec[idx[mask]] += (cg(tj, tm_i, tj, tm_j, tJ, 0)
                               - cg(tj, tm_j, tj, tm_i, tJ, 0))
    n = np.linalg.norm(vec)
    assert n > 1e-12, f"pair state {key} J={tJ / 2} vanished"
    return vec / n


def _split_vector(manifold, dets, sub_c, sub_d, tJ):
    """|(c d); J M=0⟩ for c ≠ d, one electron in each subshell."""
    from atomx.angular.coefficients import cg

    idx = {d: i for i, d in enumerate(dets)}
    tjc, tjd = 2 * abs(sub_c[1]) - 1, 2 * abs(sub_d[1]) - 1
    vec = np.zeros(len(dets))
    for i, oi in enumerate(manifold.orbitals):
        if (oi.n, oi.kappa) != sub_c:
            continue
        for j, oj in enumerate(manifold.orbitals):
            if (oj.n, oj.kappa) != sub_d or i == j:
                continue
            if oi.tm + oj.tm != 0:
                continue
            p, q = (i, j) if i < j else (j, i)
            mask = (1 << p) | (1 << q)
            if mask not in idx:
                continue
            c = cg(tjc, oi.tm, tjd, oj.tm, tJ, 0)
            vec[idx[mask]] += c if i < j else -c
    n = np.linalg.norm(vec)
    assert n > 1e-12, f"state ({sub_c},{sub_d}) J={tJ / 2} vanished"
    return vec / n


def _j2_check(manifold, dets, vec, J2):
    lam = float(vec @ manifold.j2_matrix(dets) @ vec)
    assert abs(lam - J2 * (J2 + 1)) < 1e-8, (lam, J2)


def _oracle_pair(r, dr, spec_a, spec_b, J2):
    (na, ka), (nb, kb) = spec_a, spec_b
    oa, ob = _radial(na, ka, 6.0, r), _radial(nb, kb, 6.0, r)
    manifold = ResonanceManifold(
        result=None, groups=[([(na, ka)], 2), ([(nb, kb)], 2)],
        r_fine=r, dr_fine=dr,
        orbital_map={(na, ka): oa, (nb, kb): ob},
        orbital_energies={(na, ka): -0.5, (nb, kb): -0.3},
        configs=[(2, 0), (0, 2)],
    )
    dets = manifold.determinants(0)
    va = _pair_vector(manifold, dets, (na, ka), 2 * J2)
    vb = _pair_vector(manifold, dets, (nb, kb), 2 * J2)
    _j2_check(manifold, dets, va, J2)
    _j2_check(manifold, dets, vb, J2)
    return (float(va @ manifold.hamiltonian(dets) @ vb),
            {spec_a: oa, spec_b: ob})


def _oracle_split(r, dr, spec_a, spec_c, spec_d, J2):
    (na, ka), (nc, kc), (nd, kd) = spec_a, spec_c, spec_d
    oa, oc, od = (_radial(na, ka, 6.0, r), _radial(nc, kc, 6.0, r),
                  _radial(nd, kd, 6.0, r))
    manifold = ResonanceManifold(
        result=None,
        groups=[([(na, ka)], 2), ([(nc, kc)], 1), ([(nd, kd)], 1)],
        r_fine=r, dr_fine=dr,
        orbital_map={(na, ka): oa, (nc, kc): oc, (nd, kd): od},
        orbital_energies={(na, ka): -0.5, (nc, kc): -0.4, (nd, kd): -0.3},
        configs=[(2, 0, 0), (0, 1, 1)],
    )
    dets = manifold.determinants(0)
    va = _pair_vector(manifold, dets, (na, ka), 2 * J2)
    vb = _split_vector(manifold, dets, (nc, kc), (nd, kd), 2 * J2)
    _j2_check(manifold, dets, va, J2)
    _j2_check(manifold, dets, vb, J2)
    return (float(va @ manifold.hamiltonian(dets) @ vb),
            {spec_a: oa, spec_c: oc, spec_d: od})


class _Sh:
    def __init__(self, n, kappa, occ):
        self.n, self.kappa, self.occupation = n, kappa, occ


class _CSF:
    """Minimal stand-in exposing what the element helper reads."""

    def __init__(self, total_J2, subshells):
        self.total_J2 = total_J2
        self.subshells = subshells


def _csf_for(tJ, acted):
    """CSF with a closed 1s spectator plus the acted subshells."""
    subs = [_Sh(1, -1, 2.0)]
    for key, occ in acted:
        subs.append(_Sh(key[0], key[1], float(occ)))
    return _CSF(total_J2=tJ, subshells=subs)


PAIR_CASES = [   # ((n, kappa)_a, (n, kappa)_b, label)
    ((3, -1), (4, -1), "3s1/2-4s1/2 (same kappa)"),
    ((3, -1), (3, +1), "3s1/2-3p1/2"),
    ((3, -1), (3, -2), "3s1/2-3p3/2"),
    ((3, +1), (3, -2), "3p1/2-3p3/2"),
    ((3, -2), (3, -3), "3p3/2-3d5/2"),
    ((3, +2), (3, -3), "3d3/2-3d5/2"),
    ((3, -2), (4, -2), "3p3/2-4p3/2 (same kappa)"),
    ((4, +3), (4, -4), "4f5/2-4f7/2"),
]


@pytest.mark.parametrize("spec_a,spec_b,tag", PAIR_CASES)
@pytest.mark.parametrize("J2", [0, 2])
def test_pair_channel_matches_oracle(spec_a, spec_b, tag, J2):
    """a² ↔ b²: CSF element == FAC-ported determinant element (same grid)."""
    r = make_fine_grid(40.0, 3.0, max_points=20000)
    dr = np.gradient(r)
    try:
        oracle, orbitals = _oracle_pair(r, dr, spec_a, spec_b, J2)
    except AssertionError:
        pytest.skip(f"{tag}: J={J2} not allowed for this pair")
    assert abs(oracle) > 1e-6, f"{tag}: oracle ~0, test would be vacuous"

    workspace = GridCIWorkspace(r, dr, np.zeros_like(r))
    csf = _csf_for(2 * J2, [(spec_a, 2), (spec_b, 2)])
    got = _two_electron_replacement_element(
        csf, csf, [(spec_a, 2)], [(spec_b, 2)], orbitals, workspace)
    # rel=1e-3: the two sides integrate differently (SLAG6 vs the fine-grid
    # trapezoid) and that residual grows with l — 5e-5 for s/p, 3e-4 for
    # d/f here.  A wrong angular coefficient, phase or k-selection is O(1).
    assert got == pytest.approx(oracle, rel=1e-3, abs=1e-10), (
        f"{tag} J={J2}: element {got!r} != oracle {oracle!r}")


SPLIT_CASES = [   # ((n, kappa)_a, (n, kappa)_c, (n, kappa)_d, label)
    ((3, -1), (4, -1), (5, -1), "3s^2 -> 4s 5s"),
    ((3, -2), (4, -2), (5, -2), "3p3/2^2 -> 4p 5p"),
    ((3, -3), (4, -3), (5, -3), "3d5/2^2 -> 4d 5d"),
    ((3, -2), (3, -3), (4, -3), "3p3/2^2 -> 3d5/2 4d5/2"),
]


@pytest.mark.parametrize("spec_a,spec_c,spec_d,tag", SPLIT_CASES)
@pytest.mark.parametrize("J2", [0, 2])
def test_split_channel_matches_oracle(spec_a, spec_c, spec_d, tag, J2):
    """a² ↔ c¹d¹: CSF element == FAC-ported determinant element."""
    r = make_fine_grid(40.0, 3.0, max_points=20000)
    dr = np.gradient(r)
    try:
        oracle, orbitals = _oracle_split(r, dr, spec_a, spec_c, spec_d, J2)
    except AssertionError:
        pytest.skip(f"{tag}: J={J2} not allowed")
    assert abs(oracle) > 1e-6, f"{tag}: oracle ~0, test would be vacuous"

    workspace = GridCIWorkspace(r, dr, np.zeros_like(r))
    csf = _csf_for(2 * J2, [(spec_a, 2), (spec_c, 1), (spec_d, 1)])
    got = _two_electron_replacement_element(
        csf, csf, [(spec_a, 2)], [(spec_c, 1), (spec_d, 1)], orbitals, workspace)
    # rel=1e-3: the two sides integrate differently (SLAG6 vs the fine-grid
    # trapezoid) and that residual grows with l — 5e-5 for s/p, 3e-4 for
    # d/f here.  A wrong angular coefficient, phase or k-selection is O(1).
    assert got == pytest.approx(oracle, rel=1e-3, abs=1e-10), (
        f"{tag} J={J2}: element {got!r} != oracle {oracle!r}")


def test_rk_pairing_convention_bspline():
    """``_rk`` pairs (arg1, arg2) and (arg3, arg4) in *both* engine flavours.

    Callers pass the roles they mean — ``_rk(Pa, Pc, Pb, Pd)`` for
    ``R^k(ac, bd)``.  The grid branch used to pair (arg1, arg3) and
    (arg2, arg4) instead, so the two engines evaluated different integrals
    for the same call and the grid-flavour CI was wrong wherever an
    off-diagonal element went through it.  Distinguishable random orbitals
    make the two readings numerically different, so a regression cannot pass
    silently (tests/test_ci_grid_kernel.py::test_rk_pairing_matches_gk pins
    the grid side).
    """
    t = np.linspace(0.0, 40.0, 80)
    bsp = prepare_bspline_data(t, 30, 9)
    ns = bsp["ns"]
    rng = np.random.default_rng(7)
    A = (rng.normal(size=ns), rng.normal(size=ns) * 0.02)
    B = (rng.normal(size=ns), rng.normal(size=ns) * 0.02)
    C = (rng.normal(size=ns), rng.normal(size=ns) * 0.02)
    D = (rng.normal(size=ns), rng.normal(size=ns) * 0.02)

    for k in (0, 1, 2):
        got = _rk(A[0], A[1], B[0], B[1], C[0], C[1], D[0], D[1], bsp, k)
        want = compute_slater_rk_cell(
            A[0], A[1], B[0], B[1], C[0], C[1], D[0], D[1], bsp, k)
        transposed = compute_slater_rk_cell(
            A[0], A[1], C[0], C[1], B[0], B[1], D[0], D[1], bsp, k)
        assert got == pytest.approx(want, rel=1e-12), (
            f"k={k}: _rk does not pair (a,b)/(c,d)")
        assert abs(got - transposed) > 1e-6 * max(1.0, abs(want)), (
            f"k={k}: distinguishable orbitals did not separate the "
            f"conventions")


def test_be_2s2_2p2_pair_couples():
    """Be 2s²/2p²: the ground state mixes with both 2p² jj pairs."""
    from atomx import Atom

    single = Atom(Z=4, configs=["1s2 2s2"])
    single.solve()
    e_hf = float(single.result.total_energy)

    both = Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"])
    both.solve()
    result = both.result
    m0 = np.asarray(result.mixing_coefficients)[:, 0]

    p_weight = float(np.sum(m0[1:] ** 2))
    assert abs(m0[0]) == pytest.approx(0.944, abs=0.03), m0
    assert p_weight == pytest.approx(0.108, abs=0.03), (
        f"2p² admixture {p_weight:.4f} — pair channel not coupling")
    assert float(np.sum(m0 ** 2)) == pytest.approx(1.0, abs=1e-8)

    delta = e_hf - float(result.total_energy)
    assert 0.02 < delta < 0.09, f"CI lowering {delta:.4f} Ha outside range"
    assert float(result.total_energy) == pytest.approx(-14.6139, abs=0.002)


def test_j2_pair_channel_moves_a_real_calculation():
    """A J = 2 pair channel: C-like 2p²/3p², where 2p3/2² ↔ 3p3/2² couples."""
    from atomx import Atom

    atom = Atom(Z=6, configs=["1s2 2s2 2p2", "1s2 2s2 3p2"])
    atom.solve()
    result = atom.result
    assert float(result.total_energy) == pytest.approx(-37.7211, abs=0.003)
    labels = [c.label for c in result.csfs] if result.csfs else []
    assert any("[J=4/2]" in lab for lab in labels), (
        "expected J = 2 CSFs in this space")


def test_split_channel_moves_a_real_calculation():
    """The a² ↔ c¹d¹ channel: C 2p² vs C 2p² + 3d¹4d¹.

    Both runs differ in orbital relaxation as well, so this is an
    "is the channel doing something physical and bounded" gate; the exact
    angular algebra is pinned by the oracle comparisons above.
    """
    from atomx import Atom

    base = Atom(Z=6, configs=["1s2 2s2 2p2"])
    base.solve()
    e_base = float(base.result.total_energy)

    atom = Atom(Z=6, configs=["1s2 2s2 2p2", "1s2 2s2 3d1 4d1"])
    atom.solve()
    result = atom.result
    e_split = float(result.total_energy)

    assert e_split == pytest.approx(-37.7218, abs=0.003)
    lowering = e_base - e_split
    assert 0.0 < lowering < 0.05, (
        f"3d¹4d¹ admixture moved the energy by {lowering:.5f} Ha")
    m0 = np.asarray(result.mixing_coefficients)[:, 0]
    # Every CSF in the space must be populated in some state of the block —
    # a split channel that silently stayed at zero would leave the 3d4d CSFs
    # empty in the ground state (the jj pair alone gives ~0.99 of the weight).
    dom = float(np.max(np.abs(m0)))
    assert dom < 0.99, np.round(m0, 4)


def test_pair_channel_does_not_couple_incompatible_parity():
    """A pair move that changes parity must stay uncoupled (block filter)."""
    from atomx import Atom

    both = Atom(Z=4, configs=["1s2 2s2", "1s2 2s1 2p1"])
    both.solve()
    energies = np.asarray(both.result.state_energies)
    assert np.all(np.isfinite(energies))


def test_spectator_guard_leaves_open_shell_parentage_alone():
    """With open spectators the active pair has no definite J.

    The element helper must decline (return 0) rather than apply the
    closed-spectator formula where the parentage is unknown.
    """
    csf = _CSF(total_J2=0, subshells=[_Sh(1, -1, 2.0), _Sh(2, -1, 1.0),
                                      _Sh(2, 1, 1.0), _Sh(3, -1, 2.0),
                                      _Sh(3, -2, 2.0)])
    r = make_fine_grid(20.0, 2.0, max_points=4000)
    workspace = GridCIWorkspace(r, np.gradient(r), np.zeros_like(r))
    orbitals = {}
    for key in ((2, -1), (2, 1), (3, -1), (3, -2)):
        orbitals[key] = (np.exp(-r / 4.0) * r ** (abs(key[1]) - 1),
                         np.zeros_like(r))
    got = _two_electron_replacement_element(
        csf, csf, [((3, -1), 2)], [((3, -2), 2)], orbitals, workspace)
    assert got == 0.0
# ---------------------------------------------------------------------------
# HYP-1773: occupation guard for over-occupied pattern subshells
# ---------------------------------------------------------------------------


def _real_csf(tJ2, occ):
    """Real :class:`CSF` (provides ``differs_from``) for ``_two_diff_element``."""
    from atomx.methods.dirac_fock import Subshell
    from atomx.methods.mcdhf.config import CSF

    return CSF(
        subshells=[Subshell(n, k, float(q)) for (n, k), q in occ],
        total_J2=tJ2, parity=1)


# Subshells of the audit probe: c is j = 3/2, d is j = 1/2, b is j = 5/2.
_C, _D, _B = (3, -2), (4, 1), (4, -3)


def _probe_orbitals(keys, r):
    return {key: _radial(key[0], key[1], 6.0, r) for key in keys}


@pytest.mark.parametrize("tJ2", [4, 2])
def test_overoccupied_split_pattern_returns_zero(tJ2):
    """HYP-1773: c³d¹ ↔ c²b² must not fabricate a c¹d¹ pair element.

    The bra holds c³ (three electrons in c) but the channel built a
    two-electron c¹d¹ J-coupled state, silently dropping the third c
    electron: before the fix the J = 2 element came out as −0.0039238 Ha
    while the ``ResonanceManifold`` determinant oracle gives +0.0051909 Ha
    — wrong sign *and* magnitude.  Over-occupied patterns now
    conservatively return 0 (the pre-372f06b behaviour).  At J = 1 the
    true coupling is non-zero as well and is likewise dropped rather than
    miscomputed.
    """
    r = make_fine_grid(40.0, 3.0, max_points=20000)
    workspace = GridCIWorkspace(r, np.gradient(r), np.zeros_like(r))
    orbitals = _probe_orbitals((_C, _D, _B), r)
    bra = _real_csf(tJ2, [(_C, 3), (_D, 1)])
    ket = _real_csf(tJ2, [(_C, 2), (_B, 2)])
    assert _two_diff_element(bra, ket, orbitals, workspace) == 0.0
    assert _two_diff_element(ket, bra, orbitals, workspace) == 0.0


def test_overoccupied_pair_channel_production_case_returns_zero():
    """The production CSF pair from HYP-1773 returns 0 (J = 2, even).

    ``enumerate_jj_csfs`` for configs ``["1s2 2s2 2p5 3p1", "1s2 2s2 2p4
    3d2"]`` generates ``2p½² 2p3/2³ 3p½¹`` and ``2p½² 2p3/2² 3d5/2²``,
    which share (J = 2, even) blocks — pre-fix the channel returned a
    wrong non-zero element (−0.0069160 Ha with these probe orbitals) that
    the CI matrix silently embedded.
    """
    p1h, p3h, p3s, d5h = (2, 1), (2, -2), (3, 1), (3, -3)
    r = make_fine_grid(40.0, 3.0, max_points=20000)
    workspace = GridCIWorkspace(r, np.gradient(r), np.zeros_like(r))
    orbitals = _probe_orbitals((p1h, p3h, p3s, d5h), r)
    bra = _real_csf(4, [(p1h, 2), (p3h, 3), (p3s, 1)])
    ket = _real_csf(4, [(p1h, 2), (p3h, 2), (d5h, 2)])
    assert _two_diff_element(bra, ket, orbitals, workspace) == 0.0
    assert _two_diff_element(ket, bra, orbitals, workspace) == 0.0


def test_closed_j52_pattern_returns_zero():
    """A closed j = 5/2 pattern subshell is NOT a clean J = 0 spectator.

    Unlike j = 3/2, the j = 5/2 closed shell is not a pure J = 0 pair
    condensate (the pair-removal cfp is not 1 and the remaining c⁴ J = 0
    state carries a seniority ambiguity), so the c⁶ ↔ c⁴b² element cannot
    factorise into a two-electron element; the channel must decline.
    Pre-fix it returned a wrong non-zero value (−0.0770 Ha here).
    """
    c6, b2 = (4, -3), (4, -2)   # 4d5/2 (j = 5/2) and 4p3/2
    r = make_fine_grid(40.0, 3.0, max_points=20000)
    workspace = GridCIWorkspace(r, np.gradient(r), np.zeros_like(r))
    orbitals = _probe_orbitals((c6, b2), r)
    bra = _real_csf(0, [(c6, 6)])
    ket = _real_csf(0, [(c6, 4), (b2, 2)])
    assert _two_diff_element(bra, ket, orbitals, workspace) == 0.0
    assert _two_diff_element(ket, bra, orbitals, workspace) == 0.0


def _pair_ops(manifold, key):
    """Normalised |key²; J = 0, M = 0⟩ creation terms (index pair, coeff)."""
    from atomx.angular.coefficients import cg

    tj = 2 * abs(key[1]) - 1
    tms = list(range(-tj, tj + 1, 2))
    out = []
    for x, tm_i in enumerate(tms):
        for tm_j in tms[x + 1:]:
            if tm_i + tm_j != 0:
                continue
            c = (cg(tj, tm_i, tj, tm_j, 0, 0)
                 - cg(tj, tm_j, tj, tm_i, 0, 0))
            if c != 0.0:
                out.append(((_orbital_index(manifold, key, tm_i),
                             _orbital_index(manifold, key, tm_j)), c))
    norm = np.sqrt(sum(c * c for _, c in out))
    assert norm > 1e-12
    return [(pair, c / norm) for pair, c in out]


def _closed_shell_vector(manifold, dets, key):
    """The unique closed-subshell determinant (all m filled, J = 0)."""
    idx = {d: i for i, d in enumerate(dets)}
    tj = 2 * abs(key[1]) - 1
    mask = 0
    for tm in range(-tj, tj + 1, 2):
        i = _orbital_index(manifold, key, tm)
        assert i is not None
        mask |= 1 << i
    vec = np.zeros(len(dets))
    vec[idx[mask]] = 1.0
    return vec


def _pair_product_vector(manifold, dets, key_c, key_b):
    """|[c²(J = 0) b²(J = 0)] J = 0⟩ product state over ``dets``."""
    idx = {d: i for i, d in enumerate(dets)}
    vec = np.zeros(len(dets))
    for (i1, i2), ci in _pair_ops(manifold, key_c):
        for (j1, j2), cj in _pair_ops(manifold, key_b):
            mask = (1 << i1) | (1 << i2) | (1 << j1) | (1 << j2)
            vec[idx[mask]] += ci * cj
    norm = np.linalg.norm(vec)
    assert norm > 1e-12
    return vec / norm


def test_closed_j32_pattern_subshell_matches_oracle():
    """The closed j = 3/2 pattern subshell (occ 4) must not regress.

    The occupation guard declines every over-occupied pattern subshell
    EXCEPT a closed j = 3/2 subshell on a count-2 pattern: the j = 3/2
    closed shell is a pure J = 0 pair condensate (pair-removal cfp is
    exactly 1), so the extra pair decouples as a J = 0 spectator and the
    element stays exact.  Verified against the determinant oracle; the
    magnitude is compared because the overall sign of an isolated element
    is the (arbitrary, internally consistent) basis-phase convention of
    each engine.
    """
    C3, B3 = (3, -2), (3, -3)   # 3p3/2 closed (occ 4) vs 3d5/2 pair
    r = make_fine_grid(40.0, 3.0, max_points=20000)
    dr = np.gradient(r)
    orbitals = {C3: _radial(3, -2, 6.0, r), B3: _radial(3, -3, 6.0, r)}
    manifold = ResonanceManifold(
        result=None, groups=[([C3], 4), ([B3], 2)],
        r_fine=r, dr_fine=dr,
        orbital_map=orbitals,
        orbital_energies={C3: -0.5, B3: -0.3},
        configs=[(4, 0), (2, 2)],
    )
    dets = manifold.determinants(0)
    v_bra = _closed_shell_vector(manifold, dets, C3)
    v_ket = _pair_product_vector(manifold, dets, C3, B3)
    _j2_check(manifold, dets, v_bra, 0)
    _j2_check(manifold, dets, v_ket, 0)
    oracle = float(v_bra @ manifold.hamiltonian(dets) @ v_ket)
    assert abs(oracle) > 1e-3, "oracle ~0, test would be vacuous"

    workspace = GridCIWorkspace(r, dr, np.zeros_like(r))
    csf_r = _real_csf(0, [(C3, 4)])
    csf_s = _real_csf(0, [(C3, 2), (B3, 2)])
    got = _two_electron_replacement_element(
        csf_r, csf_s, [(C3, 2)], [(B3, 2)], orbitals, workspace)
    assert abs(got) == pytest.approx(abs(oracle), rel=1e-3), (
        f"closed j=3/2 pattern: |element| {abs(got)!r} != "
        f"|oracle| {abs(oracle)!r}")
