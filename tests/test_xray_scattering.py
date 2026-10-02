"""Analytic atomic and free-electron anchors for X-ray scattering."""

import numpy as np
import pytest
from scipy.special import roots_legendre

from atomx.constants import ALPHA, BOHR_RADIUS_M, HARTREE_EV
from atomx.scattering.xray import (
    AtomicScattering,
    ScatteringOrbital,
    klein_nishina,
    thomson,
)


def hydrogen(occupation=1):
    # Analytic, nonrelativistic 1s radial wavefunction, with Gauss quadrature.
    nodes, weights = roots_legendre(500)
    r, w = 20 * (nodes + 1), 20 * weights
    p = 2 * r * np.exp(-r)
    return AtomicScattering(
        r, w, [ScatteringOrbital(1, -1, occupation, p, np.zeros_like(p))]
    )


def test_hydrogen_form_factor_and_inelastic_closure():
    model = hydrogen()
    q = np.array([0, 0.2, 1, 3, 8])
    exact_f = (1 + q**2 / 4) ** -2
    np.testing.assert_allclose(model.form_factor(q), exact_f, atol=3e-13)
    # For a single 1s electron the occupied projector is exactly F^2.
    np.testing.assert_allclose(model.incoherent_function(q), 1 - exact_f**2, atol=5e-13)
    double = hydrogen(2)
    np.testing.assert_allclose(
        double.incoherent_function(q), 2 * (1 - exact_f**2), atol=1e-12
    )


def test_free_electron_integrated_cross_sections():
    mu, weights = roots_legendre(200)
    theta = np.arccos(mu)
    r2 = (ALPHA**2 * BOHR_RADIUS_M) ** 2 / 1e-28
    np.testing.assert_allclose(
        2 * np.pi * (weights @ thomson(theta)), 8 * np.pi / 3 * r2
    )
    for energy in (1000.0, 100000.0, 1e6):
        x = energy / HARTREE_EV * ALPHA**2
        # Analytic Klein–Nishina total, independent of angular quadrature.
        exact = (
            2
            * np.pi
            * r2
            * (
                (1 + x) / x**2 * (2 * (1 + x) / (1 + 2 * x) - np.log1p(2 * x) / x)
                + np.log1p(2 * x) / (2 * x)
                - (1 + 3 * x) / (1 + 2 * x) ** 2
            )
        )
        np.testing.assert_allclose(
            2 * np.pi * (weights @ klein_nishina(energy, theta)), exact, rtol=1e-8
        )


def test_open_shell_rejects_unequal_jj_fractions():
    r = np.linspace(0.01, 10, 100)
    p = r * np.exp(-r)
    with pytest.raises(ValueError, match="equal jj"):
        AtomicScattering(
            r,
            np.ones(100) * 0.1,
            [
                ScatteringOrbital(2, 1, 1, p, p * 0),
                ScatteringOrbital(2, -2, 4, p, p * 0),
            ],
        )


def test_solved_hydrogen_result_against_analytic_density():
    from atomx.methods.dirac_fock import DFSolver, Subshell

    solver = DFSolver(
        1, [Subshell(1, -1, 1)], ns=70, ks=9, r_max=30, nuclear="point", exchange="hf"
    )
    result = solver.solve(max_iter=100, tol=1e-9, mode="orbital")
    assert result.converged
    model = AtomicScattering.from_result(result)
    q = np.array([0.0, 0.2, 1.0, 3.0])
    # Allow the known relativistic departure from the analytic Schrödinger
    # 1s density; this checks solver → grid → observable together.
    np.testing.assert_allclose(
        model.form_factor(q), (1 + q * q / 4) ** -2, rtol=2e-4, atol=2e-6
    )


@pytest.mark.parametrize("q", [-1, np.inf, np.nan])
def test_invalid_momentum(q):
    with pytest.raises(ValueError):
        hydrogen().form_factor(q)


@pytest.mark.parametrize("occupation", [2.0, 6.0])
def test_isotropic_p_shell_against_ls_projection(occupation):
    from scipy.special import spherical_jn

    nodes, weights = roots_legendre(500)
    r, w = 20 * (nodes + 1), 20 * weights
    # Hydrogenic 2p radial; jj partners share the nonrelativistic radial.
    p = r**2 * np.exp(-r / 2) / (2 * np.sqrt(6))
    model = AtomicScattering(
        r,
        w,
        [
            ScatteringOrbital(2, 1, occupation / 3, p, p * 0),
            ScatteringOrbital(2, -2, 2 * occupation / 3, p, p * 0),
        ],
    )
    q = np.array([0.0, 0.2, 1.0, 3.0])
    R0 = spherical_jn(0, q[:, None] * r) @ (w * p * p)
    R2 = spherical_jn(2, q[:, None] * r) @ (w * p * p)
    # Scalar LS trace for l=1: coefficients 6 at L=0, 12 at L=2,
    # with the intra-shell occupation weight q/6 (not (q/6)^2).
    expected = occupation * (1 - R0**2 - 2 * R2**2)
    np.testing.assert_allclose(model.incoherent_function(q), expected, atol=2e-12)
