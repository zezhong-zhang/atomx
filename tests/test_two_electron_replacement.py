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

1. **Element level — DEAD CODE, coverage of a diagnostic path only.**
   ``matrix._two_electron_replacement_element`` is reachable only through
   ``_two_diff_element`` -> ``_ci_matrix_element`` ->
   ``_build_ci_matrix_approximate``, and that legacy kernel has **no caller
   anywhere** in ``src/``, ``tests/``, ``benchmarks/`` or ``scripts/``
   (audit finding F5).  The comparison itself is real — the helper is
   checked against the FAC-ported m-scheme determinant machinery
   (``ResonanceManifold``, the engine validated against FAC AITable) on
   *identical* radial data — but it says nothing about the production CI
   Hamiltonian, so every test that only reaches the helper carries
   ``@pytest.mark.snapshot`` and this note.  The production two-electron
   coupling is covered by ``test_be_2s2_2p2_pair_couples`` /
   ``test_j2_pair_channel_moves_a_real_calculation`` below and by the
   exact-matrix gates in ``test_ci_matrix_vs_mscheme_reference.py``.

2. **Pairing convention.**  ``matrix._rk`` must pair its arguments as
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

from atomx.angular.csf_recoupling import allowed_coupled_J2
from atomx.methods.mcdhf.matrix import (
    GridCIWorkspace,
    _fk,
    _gk,
    _rk,
    _two_electron_replacement_element,
)
from atomx.numerics.bspline.basis import prepare_bspline_data
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


def _equivalent_J2_allowed(kappa: int, J2: int) -> bool:
    """Can a *closed* ``j^2`` pair couple to ``J2``?

    Two equivalent electrons couple antisymmetrically, so only even ``J``
    up to ``2j - 1`` survive: ``j = 1/2`` gives ``J = 0`` only, ``j = 3/2``
    gives ``0, 2``, ``j = 5/2`` gives ``0, 2, 4``.
    """
    return J2 % 2 == 0 and 2 * abs(kappa) - 2 >= J2


def _split_J2_allowed(kappa_c: int, kappa_d: int, J2: int) -> bool:
    """Triangle rule for two electrons in *different* subshells."""
    return J2 in allowed_coupled_J2(2 * abs(kappa_c) - 1,
                                    2 * abs(kappa_d) - 1)


@pytest.mark.snapshot
@pytest.mark.parametrize("spec_a,spec_b,tag", PAIR_CASES)
@pytest.mark.parametrize("J2", [0, 2])
def test_pair_channel_matches_oracle(spec_a, spec_b, tag, J2):
    """a² ↔ b²: CSF element == FAC-ported determinant element (same grid).

    Snapshot-marked dead-code coverage: the helper under test lives only on
    the unreachable ``_build_ci_matrix_approximate`` chain (module
    docstring, audit F5).  The skip for an impossible ``J`` is decided from
    the coupling rule *up front* — the old ``except AssertionError:
    pytest.skip`` also swallowed ``_j2_check``, the oracle's own
    ``<J^2> = J(J+1)`` self-check, turning a broken state construction into
    green skips.
    """
    if not _equivalent_J2_allowed(spec_a[1], J2):
        pytest.skip(f"{tag}: j^2 cannot couple to J={J2}")
    r = make_fine_grid(40.0, 3.0, max_points=20000)
    dr = np.gradient(r)
    oracle, orbitals = _oracle_pair(r, dr, spec_a, spec_b, J2)
    assert abs(oracle) > 1e-6, f"{tag}: oracle ~0, test would be vacuous"

    workspace = GridCIWorkspace(r, dr, np.zeros_like(r))
    csf = _csf_for(2 * J2, [(spec_a, 2), (spec_b, 2)])
    got = _two_electron_replacement_element(
        csf, [(spec_a, 2)], [(spec_b, 2)], orbitals, workspace)
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


@pytest.mark.snapshot
@pytest.mark.parametrize("spec_a,spec_c,spec_d,tag", SPLIT_CASES)
@pytest.mark.parametrize("J2", [0, 2])
def test_split_channel_matches_oracle(spec_a, spec_c, spec_d, tag, J2):
    """a² ↔ c¹d¹: CSF element == FAC-ported determinant element.

    Snapshot-marked dead-code coverage, as above.
    """
    if not _equivalent_J2_allowed(spec_a[1], J2):
        pytest.skip(f"{tag}: a^2 cannot couple to J={J2}")
    r = make_fine_grid(40.0, 3.0, max_points=20000)
    dr = np.gradient(r)
    oracle, orbitals = _oracle_split(r, dr, spec_a, spec_c, spec_d, J2)
    assert abs(oracle) > 1e-6, f"{tag}: oracle ~0, test would be vacuous"

    workspace = GridCIWorkspace(r, dr, np.zeros_like(r))
    csf = _csf_for(2 * J2, [(spec_a, 2), (spec_c, 1), (spec_d, 1)])
    got = _two_electron_replacement_element(
        csf, [(spec_a, 2)], [(spec_c, 1), (spec_d, 1)], orbitals, workspace)
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
    off-diagonal element went through it.

    On the B-spline branch ``_rk`` *is* ``compute_slater_rk_cell(a, b, c, d)``
    (``matrix._rk``), so comparing exactly those two calls is a delegation
    check that cannot fail — the previous version of this test did that, and
    that assertion is deleted.  The content is the pair-role identity:
    ``_rk(a, b, a, b) = R^k(ab, ab) = G^k(a, b)``, whereas the (a, c)(b, d)
    reading would give ``R^k(aa, bb) = F^k(a, b)``.  Both are evaluated
    below (through the *different* ``_gk``/``_fk`` entry points, which pass
    the orbitals to the kernel in a different order) and are 18-95x apart
    for these orbitals, so the pairing cannot pass silently.  The grid
    flavour of the same identity, against exact hydrogenic closed forms, is
    in ``tests/test_ci_grid_kernel.py::test_rk_pairs_its_arguments_as_ab_cd``.
    """
    t = np.linspace(0.0, 40.0, 80)
    bsp = prepare_bspline_data(t, 30, 9)
    ns = bsp["ns"]
    rng = np.random.default_rng(7)
    A = (rng.normal(size=ns), rng.normal(size=ns) * 0.02)
    B = (rng.normal(size=ns), rng.normal(size=ns) * 0.02)

    for k in (0, 1, 2):
        got = _rk(A[0], A[1], B[0], B[1], A[0], A[1], B[0], B[1], bsp, k)
        gk = _gk(A[0], A[1], B[0], B[1], bsp, k)
        fk = _fk(A[0], A[1], B[0], B[1], bsp, k)
        assert got == pytest.approx(gk, rel=1e-12), (
            f"k={k}: _rk(a,b,a,b) = {got!r} is not R^k(ab,ab) = G^k = {gk!r}")
        assert abs(got - fk) > 1e-6 * max(1.0, abs(gk)), (
            f"k={k}: the (a,c)(b,d) reading F^k = {fk!r} is indistinguishable "
            f"from _rk = {got!r}; the identity above is then vacuous")


@pytest.mark.snapshot
def test_j2_pair_channel_moves_a_real_calculation():
    """A J = 2 pair channel: C-like 2p²/3p², where 2p3/2² ↔ 3p3/2² couples.

    Snapshot-marked: the total energy is atomx's own recorded value
    (including the Breit operator, which the 2026-09 audit's Breit-operator
    migration is changing — the orchestrator owns that re-pin, not this
    test).  The structural assertion (J = 2 CSFs present) is the part that
    holds regardless.
    """
    from atomx import Atom

    atom = Atom(Z=6, config=["1s2 2s2 2p2", "1s2 2s2 3p2"])
    atom.solve()
    result = atom.result
    # Re-pinned with the exact CI matrix (was -37.7211 with the old
    # configuration-average kernel, which is a spurious 26 mHa gain: the
    # matrix now matches the independent m-scheme oracle to 4e-14 at these
    # converged orbitals, and the value sits just below the C HF energy as a
    # single pair-channel admixture should).
    # Re-pinned 2026-09-21 for the Gaunt Breit OPERATOR, now the default.
    # The previous value encoded the scalar
    # ``corrections.breit.compute_breit_energy``, which is ~3.0x too
    # large against DBSR_HF (``dbsr_hf an=Z term=AV mbreit=1``): Ne
    # 1s2 2s2 2p6 gives DBSR_HF 0.016643629692418 Ha, the Gaunt
    # operator 0.017340 (+4.2%, its documented omitted retardation)
    # and the scalar 0.050449 (3.03x); He 1s2 gives the same 3.01x, so
    # the scalar's error is structure-independent.
    assert float(result.total_energy) == pytest.approx(-37.7005, abs=0.003)
    labels = [c.label for c in result.csfs] if result.csfs else []
    assert any("[J=4/2]" in lab for lab in labels), (
        "expected J = 2 CSFs in this space")


@pytest.mark.snapshot
def test_split_channel_moves_a_real_calculation():
    """The a² ↔ c¹d¹ channel: C 2p² vs C 2p² + 3d¹4d¹.

    Both runs differ in orbital relaxation as well, so this is an
    "is the channel doing something physical and bounded" gate; the exact
    angular algebra is pinned by the oracle comparisons above.  The total
    energy is snapshot-marked (atomx's own value, Breit included).
    """
    from atomx import Atom

    base = Atom(Z=6, config=["1s2 2s2 2p2"])
    base.solve()
    e_base = float(base.result.total_energy)

    atom = Atom(Z=6, config=["1s2 2s2 2p2", "1s2 2s2 3d1 4d1"])
    atom.solve()
    result = atom.result
    e_split = float(result.total_energy)

    # Re-pinned with the exact CI matrix (was -37.7218); see the note in
    # test_j2_pair_channel_moves_a_real_calculation.
    # Re-pinned 2026-09-21 for the Gaunt Breit OPERATOR, now the default.
    # The previous value encoded the scalar
    # ``corrections.breit.compute_breit_energy``, which is ~3.0x too
    # large against DBSR_HF (``dbsr_hf an=Z term=AV mbreit=1``): Ne
    # 1s2 2s2 2p6 gives DBSR_HF 0.016643629692418 Ha, the Gaunt
    # operator 0.017340 (+4.2%, its documented omitted retardation)
    # and the scalar 0.050449 (3.03x); He 1s2 gives the same 3.01x, so
    # the scalar's error is structure-independent.
    assert e_split == pytest.approx(-37.7002, abs=0.003)
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
    """A pair move that changes parity must be *exactly* uncoupled.

    ``1s2 2s2`` (even) and ``1s2 2s1 2p1`` (odd) are connected by a
    one-electron move that flips parity, which is a selection rule: the CI
    matrix element must be identically zero.  The assertion here used to be
    ``np.all(np.isfinite(energies))``, which would pass with the parity
    filter deleted as long as nothing returned NaN.
    """
    from atomx import Atom
    from atomx.methods.mcdhf.matrix import build_ci_matrix

    both = Atom(Z=4, config=["1s2 2s2", "1s2 2s1 2p1"])
    both.solve()
    result = both.result
    H = build_ci_matrix(result.csfs, result.orbitals_dict, None,
                        result.bsp_data, result.Z, result.bsp_data["ns"])
    csfs = result.csfs
    cross = [(i, j) for i, a in enumerate(csfs) for j, b in enumerate(csfs)
             if a.parity != b.parity]
    assert cross, "fixture must contain CSFs of both parities"
    worst = max(abs(float(H[i, j])) for i, j in cross)
    assert worst == 0.0, (
        f"parity-forbidden CI element is {worst!r}, not exactly zero")


@pytest.mark.snapshot
def test_be_2s2_2p2_pair_couples():
    """Be 2s²/2p²: the ground state mixes with both 2p² jj pairs.

    Snapshot-marked: the mixing weights (0.944 / 0.108) and the total
    energy are atomx's own recorded output, not external numbers.  The
    physics content is the *coupling* — a non-zero 2p² admixture, and a
    lowering bounded away from zero — which the assertions below bracket.
    """
    from atomx import Atom

    single = Atom(Z=4, config=["1s2 2s2"])
    single.solve()
    e_hf = float(single.result.total_energy)

    both = Atom(Z=4, config=["1s2 2s2", "1s2 2p2"])
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
    # Re-pinned with the exact CI matrix (was -14.6139 with the old
    # configuration-average kernel, which also put this space 2.6 mHa below
    # the independent m-scheme oracle on identical orbitals).  The matrix
    # built here now matches that oracle to 1e-14 at these converged
    # orbitals, so the value is the exact Dirac-Coulomb CI energy of this
    # 6-CSF space in these orbitals; the residual difference to the
    # literature 2-configuration MCHF value (-14.6166, orbitals relaxed for
    # that space) is an orbital-relaxation effect, not a matrix element.
    # NOTE (2026-09 audit + Breit-operator migration): this total energy
    # includes the Breit correction; if the Breit implementation changes,
    # the orchestrator owns re-pinning this literal, not this test.
    assert float(result.total_energy) == pytest.approx(-14.6115, abs=0.002)


@pytest.mark.snapshot
def test_spectator_guard_leaves_open_shell_parentage_alone():
    """With open spectators the active pair has no definite J.

    The element helper must decline (return 0) rather than apply the
    closed-spectator formula where the parentage is unknown.  Snapshot
    marker: like the element-level tests above, this only reaches
    ``_two_electron_replacement_element`` on the unreachable legacy chain.
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
        csf, [((3, -1), 2)], [((3, -2), 2)], orbitals, workspace)
    assert got == 0.0
