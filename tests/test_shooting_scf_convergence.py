"""Energy plateaus from damping must not certify orbital convergence."""
import numpy as np
import pytest

from atomx.numerics.shooting import scf
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.quadrature import slag6


@pytest.mark.parametrize('fixed_point', [False, True])
def test_coupled_scf_checks_undamped_orbital_update(monkeypatch, fixed_point):
    r, dr, n = sgrid(30., 1e-7, .1, 500)
    def normalized(values):
        return values / np.sqrt(slag6(1., values**2*dr, n)[-1])
    initial = normalized(r*np.exp(-r))
    target = initial.copy() if fixed_point else normalized(r*np.exp(-.5*r))
    pa, qa, ev = [initial.copy()], [np.zeros(n)], np.array([-.5])
    monkeypatch.setattr(scf, 'solve_inhomogeneous_orbital',
                        lambda *args, **kwargs: (target.copy(), np.zeros(n), -.5))
    result = scf.iterate_nonlocal_orbitals(
        pa, qa, ev, ns=[1], kappas=[-1], groups={-1:[0]}, r=r, difr=dr,
        initial_energy=-.5, max_iter=4, tolerance=1e-12, damping=.9999,
        constraint_mode='coupled', first_sweep=False,
        field=lambda *args: (np.full(n,-1.), np.zeros(n), np.zeros(n)),
        evaluate=lambda *args: (-.5, (0.,0.,0.)), stationarity=lambda *args: 0.)
    assert result.converged == fixed_point
    if fixed_point:
        assert result.orbital_update_norm < 1e-12
        assert result.iterations == 2
    else:
        assert result.orbital_update_norm > .1
        assert result.iterations == 4


def test_explicit_orbital_accuracy_is_independent_of_energy_tolerance(monkeypatch):
    """A stationary energy can hide a first-order error in a property."""
    r, dr, n = sgrid(30., 1e-7, .1, 500)
    def normalize(p):
        return p/np.sqrt(slag6(1.,p*p*dr,n)[-1])
    initial = normalize(r*np.exp(-r))
    target = normalize(initial+1e-6*r*r*np.exp(-r))
    monkeypatch.setattr(scf, 'solve_inhomogeneous_orbital',
                        lambda *args, **kwargs: (target.copy(),np.zeros(n),-.5))
    def solve(orbital_tolerance):
        return scf.iterate_nonlocal_orbitals(
            [initial.copy()],[np.zeros(n)],np.array([-.5]),ns=[1],kappas=[-1],
            groups={-1:[0]},r=r,difr=dr,initial_energy=-.5,max_iter=4,
            tolerance=1e-8,damping=.9999,constraint_mode='coupled',first_sweep=False,
            field=lambda *args:(np.full(n,-1.),np.zeros(n),np.zeros(n)),
            evaluate=lambda *args:(-.5,(0.,0.,0.)),stationarity=lambda *args:0.,
            orbital_tolerance=orbital_tolerance)
    assert solve(None).converged
    assert not solve(1e-9).converged
    with pytest.raises(ValueError,match='orbital_tolerance'):
        solve(0.)
