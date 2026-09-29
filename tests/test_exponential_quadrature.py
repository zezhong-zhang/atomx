"""Analytic stiff-source controls independent of atomic energy snapshots."""
import numpy as np
import pytest

from atomx.numerics.shooting.exponential_quadrature import (
    apply_exponential_prefix, exponential_prefix_weights,
)


@pytest.mark.parametrize('slope', [.01, .1, 1., 2., 5., 20., 100.])
def test_exponential_source_matches_exact_integral(slope):
    x = np.arange(100.)
    logs = slope*x
    expected = -np.expm1(-logs)/slope
    result = apply_exponential_prefix(np.ones_like(x), *exponential_prefix_weights(logs))
    np.testing.assert_allclose(result, expected, rtol=5e-12, atol=2e-13)
    shifted = apply_exponential_prefix(np.ones_like(x),
                                      *exponential_prefix_weights(logs+10000.))
    np.testing.assert_allclose(result, shifted, rtol=2e-11, atol=2e-13)


@pytest.mark.parametrize('slope', [2., 20.])
def test_signed_polynomial_source_has_analytic_antiderivative(slope):
    x = np.arange(100.)
    polynomial = np.polynomial.Polynomial([-.4, .01, -.0001, 2e-6, -1e-8, 1e-10])
    primitive = sum(((-1)**k/slope**(k+1))*polynomial.deriv(k)
                    for k in range(6))
    expected = primitive(x)-np.exp(-slope*x)*primitive(0.)
    result = apply_exponential_prefix(polynomial(x),
                                     *exponential_prefix_weights(slope*x))
    np.testing.assert_allclose(result, expected, rtol=5e-12, atol=2e-13)


def test_source_map_is_linear_and_rejects_unresolved_exponents():
    x = np.arange(100.)
    prepared = exponential_prefix_weights(2*x)
    a, b = np.sin(.1*x), np.cos(.03*x)
    np.testing.assert_allclose(apply_exponential_prefix(a+3*b, *prepared),
        apply_exponential_prefix(a, *prepared)+3*apply_exponential_prefix(b, *prepared),
        rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError, match='too coarse'):
        exponential_prefix_weights(101*x)


def test_zero_rate_preserves_amplitude_direction_cancellation():
    x = np.arange(100.)
    polynomial = np.polynomial.Polynomial([.1, -.01, .0002, -1e-6])
    logs = .5*np.sin(.1*x)
    source = polynomial(x)*np.exp(-logs)
    expected = np.exp(-logs)*(polynomial.integ()(x)-polynomial.integ()(0.))
    actual = apply_exponential_prefix(source,
        *exponential_prefix_weights(logs, np.zeros_like(logs)))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_green_reciprocity_and_energy_response_identity():
    from atomx.numerics.shooting.dirac_response import DiracResponse
    from atomx.numerics.shooting.grid import sgrid
    r, jacobian, n = sgrid(30., 1e-7, .1, 2000)
    potential = np.full(n, -2.)
    a = np.array([r*np.exp(-r), .01*r*r*np.exp(-r)])
    b = np.array([r*r*np.exp(-.8*r), -.02*r*np.exp(-.8*r)])
    energy, step = -5., 1e-4
    response = DiracResponse(energy, -1, r, jacobian, potential, boundary='box')
    ga, gb = response.apply(a), response.apply(b)
    left = np.sum(a*gb*response.weights)
    right = np.sum(ga*b*response.weights)
    assert left == pytest.approx(right, rel=1e-10, abs=1e-13)
    plus = DiracResponse(energy+step, -1, r, jacobian, potential, boundary='box').apply(a)
    minus = DiracResponse(energy-step, -1, r, jacobian, potential, boundary='box').apply(a)
    derivative = (plus-minus)/(2*step)
    expected = response.apply(ga)  # d(H-e)^-1/de = (H-e)^-2
    relative = np.sqrt(np.sum((derivative-expected)**2*response.weights)
                       /np.sum(expected**2*response.weights))
    assert relative < 1e-7
