"""Hydrogenic Schrödinger roots from varied guesses and failed-root contracts."""
import numpy as np
import pytest
from scipy.special import eval_genlaguerre, factorial

from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.quadrature import slag6
from atomx.numerics.shooting.schrodinger_bound import sbound
from atomx.numerics.shooting.spline import vint


@pytest.mark.parametrize('z', [1., 2., 10.])
@pytest.mark.parametrize('n,ell', [(2, 1), (3, 2), (3, 1), (4, 0)])
@pytest.mark.parametrize('guess', [-.1, -1.])
def test_coulomb_roots_and_waves_do_not_depend_on_starting_energy(z, n, ell, guess):
    r, jacobian, count = sgrid(150., 1e-7, .5, 1600)
    energy, ier, p, q, _ = sbound(guess, 1e-11, n, ell,
        *vint(r, np.full(count, -z), count), r, count)
    assert ier == 0
    assert energy == pytest.approx(-z*z/(2*n*n), rel=2e-10, abs=5e-12)
    rho = 2*z*r/n
    radial = (2*z/n)**1.5*np.sqrt(factorial(n-ell-1)/(2*n*factorial(n+ell)))
    expected = radial*r*np.exp(-rho/2)*rho**ell*eval_genlaguerre(n-ell-1, 2*ell+1, rho)
    np.testing.assert_allclose(p, expected, rtol=2e-8, atol=2e-10)
    # Q is dP/dr on this engine, not the small Dirac component.
    laguerre = eval_genlaguerre(n-ell-1, 2*ell+1, rho)
    derivative = (-eval_genlaguerre(n-ell-2, 2*ell+2, rho)
                  if n-ell-1 else np.zeros_like(rho))
    expected_q = (radial*np.exp(-rho/2)*rho**ell
                  *((ell+1-rho/2)*laguerre+rho*derivative))
    np.testing.assert_allclose(q, expected_q, rtol=2e-8, atol=2e-9)
    norm = slag6(1., p*p*jacobian, count)[-1]
    assert norm == pytest.approx(1., rel=2e-9)
    assert np.all(np.isfinite(q))


def test_unresolved_box_returns_error_without_an_orbital():
    r, _, count = sgrid(5., 1e-7, .01, 400)
    _, ier, p, q, last = sbound(-.1, 1e-11, 4, 0,
        *vint(r, np.full(count, -1.), count), r, count)
    assert ier != 0
    np.testing.assert_array_equal(p, np.zeros(count))
    np.testing.assert_array_equal(q, np.zeros(count))
    assert last == 0


def test_unconverged_refinement_cannot_return_an_orbital(monkeypatch):
    import atomx.numerics.shooting.schrodinger_bound as bound

    # Deliberately nonstationary responses move E by a fixed .001 Ha per
    # trial. They never satisfy the matching tolerance or exhaust the box.
    def outward(*args):
        index = args[4]
        args[-2][index], args[-1][index] = 1., 0.
        return 0, 1., 0
    def inward(*args):
        index = args[3]
        args[-2][index], args[-1][index] = 1., -.004
        return 1., len(args[-2])-1, 0
    monkeypatch.setattr(bound, 'soutw', outward)
    monkeypatch.setattr(bound, 'sinw', inward)
    r, _, count = sgrid(75., 1e-7, .5, 1000)
    _, ier, p, q, last = bound.sbound.py_func(-.5, 1e-11, 1, 0,
        *vint(r, np.full(count, -1.), count), r, count)
    assert ier == 7
    assert not np.any(p) and not np.any(q)
    assert last == 0
