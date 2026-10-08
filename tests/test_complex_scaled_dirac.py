"""Independent Sommerfeld, relativistic dispersion and c-product gates."""

import numpy as np
import pytest

from atomx.constants import C_SQ, C
from atomx.numerics.bspline.complex_dirac import ComplexScaledDiracWorkspace
from atomx.numerics.bspline.knots import make_exponential_knots
from atomx.physics.dirac import Dirac, hydrogenic_dirac_energy


def workspace(kappa=-1, Z=10, ns=75):
    return ComplexScaledDiracWorkspace(make_exponential_knots(ns, 9, 1e-4, 40), 9, kappa, Z)


@pytest.mark.parametrize('kappa', [-1, 1, -2, 2, -3])
def test_rotated_hydrogenic_bound_roots_match_sommerfeld(kappa):
    w = workspace(kappa)
    n = abs(kappa)+(kappa > 0)
    exact = hydrogenic_dirac_energy(n, kappa, 10)
    for theta in (0, .2, .4):
        result = w.solve(theta)
        root = np.argmin(abs(result.energies-exact))
        # Absolute complex error gates energy AND artificial bound width.
        assert abs(result.energies[root]-exact) < 2e-8
        assert abs(result.angle_slopes[root]) < 2e-7
        assert result.residuals.max() < 1e-12


def test_free_continuum_obeys_relativistic_dilation_on_both_branches():
    w = workspace(Z=0)
    real, rotated = w.solve(0), w.solve(.3)
    for E in real.energies[real.energies.real > 0][:6].real:
        p2c2 = E*(E+2*C_SQ)
        root = np.sqrt(C_SQ**2 + np.exp(-.6j)*p2c2)
        positive = np.exp(-.6j)*p2c2/(root+C_SQ)
        assert positive.imag < 0
        assert np.min(abs(rotated.energies-positive)) < 2e-8
    # At fixed kappa the sea's dominant component has different orbital l,
    # hence different finite-box momenta. Rotate its OWN real-angle roots.
    for E in real.energies[real.energies.real < -2*C_SQ][-6:].real:
        root = np.sqrt(C_SQ**2 + np.exp(-.6j)*E*(E+2*C_SQ))
        negative = -C_SQ-root
        assert negative.imag > 0
        assert np.min(abs(rotated.energies-negative)) < 2e-8
    # Rotating free box states is not a physical decay-width measurement.
    assert not hasattr(rotated, 'widths')


def test_complex_symmetric_metric_uses_transpose_and_retains_nonhermiticity():
    w = workspace()
    H, S = w.matrices(.3)
    np.testing.assert_allclose(H, H.T, atol=0, rtol=0)
    np.testing.assert_allclose(S, S.T, atol=0, rtol=0)
    assert np.linalg.norm(H-H.conj().T) > 1
    assert np.linalg.norm(S-S.conj().T) > .01
    _, S0 = w.matrices(0)
    np.testing.assert_allclose(S0, np.eye(w.dimension), atol=1e-11)
    o = w.solve(.3)
    np.testing.assert_allclose(np.einsum('ij,ij->j', o.coefficients, S@o.coefficients), 1, atol=1e-12)
    for v in (o.energies, o.coefficients, o.residuals, o.angle_slopes, w.knots):
        assert not v.flags.writeable


def test_angle_slopes_against_independent_eigenvalue_difference():
    w = workspace(Z=0)
    o = w.solve(.3)
    idx = np.flatnonzero(o.energies.real > 0)[0]
    E = o.energies[idx]
    h = .001
    sides = []
    for theta in (.3-h, .3+h):
        e = w.solve(theta).energies
        sides.append(e[np.argmin(abs(e-E))])
    fd = (sides[1]-sides[0])/(2*h)
    assert abs(fd-o.angle_slopes[idx]) < 2e-8
    dH, dS = w.angle_derivatives(.3)
    for a, index in ((dH, 0), (dS, 1)):
        fd_mat = (w.matrices(.3+1e-5)[index]-w.matrices(.3-1e-5)[index])/(2e-5)
        np.testing.assert_allclose(a, fd_mat, atol=.01, rtol=1e-7)


def test_polynomial_action_scales_kinetic_and_coulomb_but_not_mass():
    r = np.linspace(.1, 2, 17)
    P, Q = r*r, r*r*r
    theta = .3
    z = np.exp(-1j*theta)
    hp, hq = Dirac().radial_action(P, Q, 2*r, 3*r*r, r, -10*z/r, -1, dilation_angle=theta)
    np.testing.assert_allclose(hp, z*(-10*P/r+C*(-3*r*r-Q/r)), atol=1e-12)
    np.testing.assert_allclose(hq, z*(-10*Q/r+C*(2*r-P/r))-2*C_SQ*Q, atol=1e-10)
    p, q = Dirac().radial_action(P, Q, 2*r, 3*r*r, r, -10/r, -1)
    assert np.max(abs(hq-z*q)) > 1000  # catches scaling the rest mass


@pytest.mark.parametrize('theta', [np.nan, np.inf, 1j, np.pi/2, [0]])
def test_invalid_angles_fail(theta):
    with pytest.raises(ValueError, match='theta'):
        workspace(ns=35).solve(theta)


def test_owned_knots_and_fresh_matrices_prevent_stale_cache():
    knots = make_exponential_knots(40, 9, 1e-4, 40)
    w = ComplexScaledDiracWorkspace(knots, 9, -1, 10)
    before = w.matrices(.3)
    knots[9] *= .9
    before[0][:] = 0
    assert np.linalg.norm(w.matrices(.3)[0]) > 1
    np.testing.assert_array_equal(w.matrices(.3)[1], before[1])
    with pytest.raises(ValueError, match='quadrature'):
        ComplexScaledDiracWorkspace(w.knots, 9, -1, 10, n_gauss=9)
    with pytest.raises(ValueError, match='subcritical'):
        ComplexScaledDiracWorkspace(w.knots, 9, -1, 150)
