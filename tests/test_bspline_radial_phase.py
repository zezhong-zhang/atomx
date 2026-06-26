"""Deterministic radial-spinor phase convention tests."""

import numpy as np

from atomx.numerics.bspline.dirac_assembly import (
    DiracResult,
    canonicalize_radial_phase,
)


def test_canonicalize_radial_phase_makes_largest_large_component_positive():
    P = np.array([0.1, -3.0, 2.0])
    Q = np.array([0.2, -0.4, 0.6])

    P_fix, Q_fix = canonicalize_radial_phase(P, Q)

    assert P_fix[1] > 0.0
    np.testing.assert_allclose(P_fix, -P)
    np.testing.assert_allclose(Q_fix, -Q)


def test_dirac_result_get_orbital_returns_canonical_phase():
    coeffs = np.zeros((6, 2))
    coeffs[:, 0] = np.array([-0.1, -2.0, 0.5, -0.01, -0.2, 0.03])
    coeffs[:, 1] = np.array([0.2, 1.0, -0.3, 0.02, 0.1, -0.01])
    result = DiracResult(
        kappa=-1,
        eigenvalues=np.array([-2.0, 1.0]),
        bound_energies=np.array([-2.0]),
        coefficients=coeffs,
        ns=3,
        nsp=3,
        nsq=3,
    )

    P, Q, E = result.get_orbital(0)

    assert E == -2.0
    assert P[1] > 0.0
    np.testing.assert_allclose(P, np.array([0.1, 2.0, -0.5]))
    np.testing.assert_allclose(Q, np.array([0.01, 0.2, -0.03]))
