"""Two-electron (double-replacement) CI coupling.

History.  `Atom(configs=[...])` did not couple configurations differing by a
two-electron replacement at all: `Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"])`
returned the pure `1s2 2s2` state (mixing ``[1, 0, ...]``, the HF energy to
6e-7), so the classic 2s²/2p² near-degeneracy correlation was missing.  The
production kernel covers all three patterns:

    a² ↔ b²      one subshell loses 2, another gains 2
    a² ↔ c¹ d¹   one loses 2, two gain 1 (double excitation into two
                 different orbitals)
    a¹ b¹ ↔ c²   the transpose of the above

Two kinds of check remain, both on the production path:

1. **Pairing convention.**  ``matrix._rk`` must pair its arguments as
   ``R^k(a c, b d)`` in *both* engine flavours.  The B-spline branch used to
   hand the orbitals to ``compute_slater_rk_cell`` in the ``(a, b, c, d)``
   order, which that routine reads as the pairs ``(a, b)`` and ``(c, d)``, so
   the two engines disagreed on every integral built through it.

2. **End to end.**  The Be 2s²/2p² ground state must mix and drop below the
   single-configuration HF energy; a J = 2 pair channel and the split channel
   must move real calculations.

Removed with the legacy kernel (2026-10-05 review, finding E3): the
element-level tests of ``matrix._two_electron_replacement_element`` /
``matrix._two_diff_element`` (oracle comparisons against
``ResonanceManifold`` and the over-occupation guards).  Those helpers were
reachable only through the known-wrong legacy CI assembly, which had no
production caller; the tests themselves were marked dead-code coverage.
The production two-electron coupling is gated by the end-to-end tests here
and by the exact-matrix oracle in
``tests/test_ci_matrix_vs_mscheme_reference.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.physics.csf.matrix import (
    _fk,
    _gk,
    _rk,
)


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
    after correlated orbital optimization and Gaunt/model-QED corrections.
    The structural assertion (J = 2 CSFs present) is independent of the
    recorded energy.
    """
    from atomx import Atom

    atom = Atom(Z=6, config=["1s2 2s2 2p2", "1s2 2s2 3p2"])
    atom.solve()
    result = atom.result
    # Self snapshot after constrained optimization of the full correlated
    # Dirac-Coulomb functional, followed by Gaunt/model-QED corrections.
    assert atom.converged
    assert float(result.total_energy) == pytest.approx(-37.70484425926792, abs=1e-5)
    labels = [c.label for c in result.csfs] if result.csfs else []
    assert any("[J=4/2]" in lab for lab in labels), (
        "expected J = 2 CSFs in this space")


@pytest.mark.snapshot
def test_split_channel_moves_a_real_calculation():
    """The a² ↔ c¹d¹ channel: C 2p² vs C 2p² + 3d¹4d¹.

    Both runs differ in orbital relaxation as well, so this is an
    "is the channel doing something physical and bounded" gate; the exact
    angular algebra is pinned by the exact-matrix oracle tests.  The total
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

    # Self snapshot after constrained optimization of the full correlated
    # Dirac-Coulomb functional, followed by Gaunt/model-QED corrections.
    assert atom.converged
    assert e_split == pytest.approx(-37.70568011522897, abs=1e-5)
    lowering = e_base - e_split
    assert 0.0 < lowering < 0.05, (
        f"3d¹4d¹ admixture moved the energy by {lowering:.5f} Ha")
    m0 = np.asarray(result.mixing_coefficients)[:, 0]
    # Check the actual added configurations. A cap on the largest single
    # coefficient also depends on jj recoupling within the reference and
    # cannot establish whether the split channel participates.
    split = [i for i, csf in enumerate(result.csfs)
             if any(sh.n == 4 and sh.l == 2 and sh.occupation > 0
                    for sh in csf.subshells)]
    assert split
    assert float(np.sum(m0[split] ** 2)) > 1e-4, np.round(m0, 4)


def test_pair_channel_does_not_couple_incompatible_parity():
    """A pair move that changes parity must be *exactly* uncoupled.

    ``1s2 2s2`` (even) and ``1s2 2s1 2p1`` (odd) are connected by a
    one-electron move that flips parity, which is a selection rule: the CI
    matrix element must be identically zero.  The assertion here used to be
    ``np.all(np.isfinite(energies))``, which would pass with the parity
    filter deleted as long as nothing returned NaN.
    """
    from atomx import Atom
    from atomx.physics.csf.matrix import build_ci_matrix

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
    # Self snapshot after constrained optimization of the full correlated
    # Dirac-Coulomb functional, followed by Gaunt/model-QED corrections.
    assert both.converged
    assert float(result.total_energy) == pytest.approx(-14.61859192, abs=1e-5)
