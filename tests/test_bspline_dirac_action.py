"""Dirac differential action versus independent Galerkin assembly and polynomials."""

import numpy as np
import pytest

from atomx.constants import C_SQ, C
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import (
    apply_dirac_hamiltonian,
    build_dirac_matrices,
)
from atomx.numerics.bspline.evaluate import eval_P, eval_Q
from atomx.physics.dirac import Dirac


@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("kappa", [-1, 1, -2])
def test_native_action_equals_independent_Galerkin_matrix_with_zero_boundaries(
    balance, kappa
):
    data = prepare_bspline_data(
        np.array([0.0] * 4 + [2.0] * 4), 4, 4, kinetic_balance=balance
    )
    p = np.array([0.0, 0.7, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.4, 0.0])
    if not balance:
        p[2] = 0.2
    a = np.array([0.0, 0.3, 0.0, 0.0])
    b = np.array([0.0, 0.9, 0.1, 0.0])
    if not balance:
        a[2] = -0.6

    def potential(r):
        return -10.0 / np.sqrt(r * r + 0.3**2)

    hp, hq = apply_dirac_hamiltonian(p, q, kappa, data, potential=potential(data["gr"]))
    differential = np.sum(data["grw"] * (eval_P(a, data) * hp + eval_Q(b, data) * hq))
    matrix, _ = build_dirac_matrices(4, kappa, 10, data, potential)
    galerkin = np.concatenate([a, b]) @ matrix @ np.concatenate([p, q])
    assert differential == pytest.approx(galerkin, rel=1e-13, abs=1e-12)


def test_RKB_polynomials_pin_derivatives_rest_energy_and_small_component_padding():
    data = prepare_bspline_data(
        np.array([0.0] * 4 + [2.0] * 4), 4, 4, kinetic_balance=True
    )
    r = data["gr"]
    t = r / 2
    # P is quadratic, Q cubic. Deliberately huge padded P coefficient
    # must not leak into either its values or derivatives.
    p = np.array([0.0, 1.0, 0.0, 1e8])
    q = np.array([0.0, 0.0, 1.0, 0.0])
    hp, hq = apply_dirac_hamiltonian(p, q, -1, data, potential=-10 / r)
    pv, qv = 2 * t * (1 - t), 3 * t * t * (1 - t)
    dp, dq = 1 - 2 * t, 3 * t - 4.5 * t * t
    np.testing.assert_allclose(hp, -10 / r * pv + C * (-dq - qv / r), atol=1e-13)
    np.testing.assert_allclose(
        hq, C * (dp - pv / r) + (-10 / r - 2 * C_SQ) * qv, atol=2e-12
    )
    phase = np.exp(0.37j)
    ap, aq = apply_dirac_hamiltonian(phase * p, phase * q, -1, data, potential=-10 / r)
    np.testing.assert_allclose(ap, phase * hp, atol=1e-13)
    np.testing.assert_allclose(aq, phase * hq, atol=2e-12)


def test_physics_action_rejects_origin_and_inconsistent_representation():
    values = np.ones(3)
    with pytest.raises(ValueError, match="aligned"):
        Dirac().radial_action(
            values, values, values, values, np.array([0, 1, 2]), values, -1
        )
    with pytest.raises(ValueError, match="aligned"):
        Dirac().radial_action(values, values, values[:2], values, values, values, -1)
