"""Analytic radial contractions and discrete symmetry of Dirac operators."""
import math

import numpy as np
import pytest

from atomx.numerics.shooting.constants import SL
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.operators import _ds_derivative, one_body_element


@pytest.mark.parametrize('degree', range(9))
@pytest.mark.parametrize('points', [9, 10, 11, 12, 13, 41])
def test_derivative_is_exact_for_stencil_polynomials(degree, points):
    # This probes every boundary row as well as the centered stencil.
    x = np.linspace(-.5, .5, points)
    expected = (np.zeros_like(x) if degree == 0
                else degree*x**(degree-1)/(points-1))
    np.testing.assert_allclose(_ds_derivative(x**degree), expected, atol=3e-14)


@pytest.mark.parametrize('kappa', [-3, -1, 1, 3])
def test_dirac_contraction_refines_to_analytic_integral(kappa):
    ell = kappa if kappa > 0 else -kappa-1
    alpha, beta, z = 1.7, .004, 5.
    power = 2*(ell+1)

    def moment(j):
        return math.factorial(j)/(2*alpha)**(j+1)

    norm = moment(power)+beta**2*moment(power+2)
    exact = (SL*beta*(2*kappa-1)*moment(power)
             - z*(moment(power-1)+beta**2*moment(power+1))
             - 2*SL**2*beta**2*moment(power+2))/norm
    errors = []
    for points in (400, 1600):
        r, jacobian, n = sgrid(30., 1e-7, .1, points)
        p = r**(ell+1)*np.exp(-alpha*r)/np.sqrt(norm)
        q = beta*r*p
        actual = one_body_element(p, q, p, q, kappa, r, jacobian, np.full(n, -z))
        errors.append(abs(actual-exact))
        # Copies use separate derivative buffers; strided views are valid too.
        views = np.empty((n, 2))
        views[:, 0], views[:, 1] = p, q
        copied = one_body_element(p.copy(), q.copy(), p, q, kappa,
                                  r, jacobian, np.full(n, -z))
        strided = one_body_element(views[:, 0], views[:, 1], views[:, 0], views[:, 1],
                                   kappa, r, jacobian, np.full(n, -z))
        assert copied == pytest.approx(actual, abs=2e-13)
        assert strided == pytest.approx(actual, abs=2e-13)
    assert errors[1] < 1e-9
    assert errors[1] < errors[0]/100


def test_dirac_contraction_is_symmetric_for_distinct_spinors():
    r, jacobian, n = sgrid(30., 1e-7, .1, 1200)
    pa, pb = r*np.exp(-r), r*(1-r)*np.exp(-r/2)
    qa, qb = .01*r*pa, -.02*pb
    potential = np.full(n, -3.)
    ab = one_body_element(pa, qa, pb, qb, -1, r, jacobian, potential)
    ba = one_body_element(pb, qb, pa, qa, -1, r, jacobian, potential)
    assert ab == pytest.approx(ba, abs=2e-13)
