"""Scalar Fock fields match independent signed magnetic contractions."""

import numpy as np
import pytest

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.physics.csf.frozen_core import closed_core_one_body, closed_shell_one_body
from atomx.physics.csf.radial_snapshot import CIRadialSnapshot
from atomx.physics.csf.spinor_integrals import build_spinor_integrals


@pytest.mark.parametrize(
    "kinetic_balance,nucleus", [(False, "point"), (True, "point"), (True, "fermi")]
)
def test_overlapping_core_and_rotated_signed_fock(kinetic_balance, nucleus):
    occupied = ((1, -1), (2, -1))
    core = DFSolver(
        4,
        [Subshell(*a, 2) for a in occupied],
        ns=30,
        ks=7,
        kinetic_balance=kinetic_balance,
        nuclear=nucleus,
    ).solve()
    waves = {a: core.orbitals_by_subshell()[a][:2] for a in occupied}
    virtual = []
    # d/f targets constrain exchange ranks beyond the old s/p-only anchors.
    for k in (-1, 1, -2, 2, -3, 3, -4):
        _, roots = build_virtual_basis(
            core, k, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        a = (max([n for n, ka in occupied if ka == k] + [abs(k)]) + 1, k)
        virtual.append(a)
        waves[a] = roots[0][:2]
    # Keep the occupied subspace complete, with noncanonical off-diagonal Fock.
    u = np.linalg.qr(np.random.default_rng(701).normal(size=(2, 2)))[0]
    rotated = [np.column_stack([waves[a][i] for a in occupied]) @ u for i in (0, 1)]
    for j, a in enumerate(occupied):
        waves[a] = (rotated[0][:, j], rotated[1][:, j])
    native = build_spinor_integrals(
        waves,
        core._bsp_data,
        4,
        30,
        occupied_keys=occupied,
        virtual_keys=virtual,
        nuclear_potential=core._nuclear_potential,
        representation="on_demand",
    )
    snapshot = CIRadialSnapshot(
        waves, core._bsp_data, 4, 30, nuclear_potential=core._nuclear_potential
    )
    radial = snapshot.evaluator_for(
        waves, core._bsp_data, 4, 30, nuclear_potential=core._nuclear_potential
    )
    analytic = closed_shell_one_body(waves, {a: 2 for a in occupied}, radial)
    assert abs(analytic[occupied[0], occupied[1]]) > 1e-3
    for (a, b), value in analytic.items():
        for m in range(-2 * abs(a[1]) + 1, 2 * abs(a[1]), 2):
            p, q = native.labels.index((*a, m)), native.labels.index((*b, m))
            # Explicit signed ERIs, rather than the factory's Fock matrix,
            # remain the independent oracle if construction is optimized.
            explicit = native.one_body_Ha[p, q] + sum(
                native.eri(p, i, q, i) for i in range(native.noccupied)
            )
            assert abs(value - explicit) < 2e-11
    frozen = closed_core_one_body(virtual, {a: 2 for a in occupied}, radial)
    assert all(frozen[a, b] == analytic[a, b] for a, b in frozen)
    with pytest.raises(ValueError, match="disjoint"):
        closed_core_one_body(waves, {a: 2 for a in occupied}, radial)


def test_full_core_self_interaction_and_fractional_rejection():
    a = (1, -1)

    def radial(key):
        return -2.0 if key[0] == "h" else 1.0

    # Two occupied 1s substates: 2*J minus one exchange contribution.
    assert closed_shell_one_body([a], {a: 2}, radial)[a, a] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="closed subshell"):
        closed_shell_one_body([a], {a: 1}, radial)
