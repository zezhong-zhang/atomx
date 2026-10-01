"""Finite-photon operators: independent FAC data and local-Dirac identities."""
import json
from pathlib import Path

import numpy as np
import pytest

from atomx.config.subshell import Subshell
from atomx.numerics.radial_workspace import make_grid_workspace
from atomx.numerics.shooting.local import ShootingSolver
from atomx.numerics.special import scaled_spherical_jn
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.multipole import radial_eL, radial_eL_velocity
from atomx.properties.transitions.rates import einstein_A_eL


@pytest.fixture(scope='module', params=[1, 10, 82])
def hydrogenic(request):
    z = request.param
    states = {}
    for n, k in [(1, -1)]+[(l+1, k) for l in range(1, 5) for k in (l, -l-1)]:
        states[n, k] = ShootingSolver(z, [Subshell(n, k, 1)], nuclear='point',
                                      n_grid=2000, r_max=125/z).solve(tol=1e-11)
        assert states[n, k].converged
    return z, states


@pytest.mark.parametrize('rank', [1, 2, 3, 4])
@pytest.mark.parametrize('upper_branch', [False, True])
def test_electric_gauges_reciprocity_and_fac(hydrogenic, rank, upper_branch):
    z, states = hydrogenic
    k = -rank-1 if upper_branch else rank
    lo, up = states[1, -1], states[rank+1, k]
    workspace = lo.radial_state.correction_workspace()
    lower = (lo._pa[0, 1:], lo._qa[0, 1:])
    upper = (up._pa[0, 1:], up._qa[0, 1:])
    omega = up.total_energy-lo.total_energy
    length = radial_eL(*lower, *upper, workspace, rank,
                       kappa_i=-1, kappa_f=k, omega=omega)
    velocity = radial_eL_velocity(*lower, *upper, workspace, rank,
                                  kappa_i=-1, kappa_f=k, omega=omega)
    assert velocity == pytest.approx(length, rel=1e-8)
    for kernel, ref in [(radial_eL, length), (radial_eL_velocity, velocity)]:
        reverse = kernel(*upper, *lower, workspace, rank,
                         kappa_i=k, kappa_f=-1, omega=-omega)
        assert reverse == pytest.approx(ref, rel=1e-13)
    if z in (10, 82):
        reference = json.loads((Path(__file__).parent/'data'/'fac_photon_reference.json').read_text())
        rows = next(c['rows'] for c in reference['cases'] if c['z'] == z)
        row = next(row for row in rows if row['gauge'] == 2
                   and row['multipole'] == -rank and row['upper'] == [rank+1, k])
        rate = einstein_A_eL(omega, (length*angular_eL(-1, k, rank))**2,
                            rank, 2*abs(k)-1)
        # Independently solved spinors; FAC evaluates each exact line
        # frequency. The stricter common-spinor test isolates the operator.
        assert rate == pytest.approx(row['fac_A_per_s'], rel=1e-5)
    if z == 82:
        static = radial_eL(*lower, *upper, workspace, rank)
        assert abs((length/static)**2-1) > .02  # cannot pass with a static operator


@pytest.mark.parametrize('rank', [1, 2, 3, 4])
def test_static_limit_and_missing_frequency_contract(rank):
    x, w = np.polynomial.legendre.leggauss(80)
    r, w = (x+1)*5, w*5
    p, q = r*np.exp(-r), -.01*r*np.exp(-r)
    ws = make_grid_workspace(r, w)
    static = radial_eL(p, q, p, q, ws, rank)
    finite = radial_eL(p, q, p, q, ws, rank, kappa_i=-1, kappa_f=rank, omega=1e-9)
    assert finite == pytest.approx(static, rel=1e-12)
    with pytest.raises(ValueError, match='both kappas'):
        radial_eL(p, q, p, q, ws, rank, omega=1.)
    with pytest.raises(ValueError, match='both kappas'):
        radial_eL_velocity(p, q, p, q, ws, rank)
    with pytest.raises(ValueError, match='nonzero'):
        radial_eL_velocity(p, q, p, q, ws, rank, kappa_i=-1, kappa_f=rank, omega=0.)


def test_scaled_bessel_is_regular_and_even():
    from scipy.special import factorial2, spherical_jn
    x = np.array([0., 1e-100, .099999, .100001, 1., 10.])
    for rank in range(6):
        values = scaled_spherical_jn(rank, x)
        assert np.all(np.isfinite(values))
        assert values[:2] == pytest.approx([1., 1.], abs=1e-15)
        assert values == pytest.approx(scaled_spherical_jn(rank, -x), rel=1e-15)
        assert values[2:] == pytest.approx(
            factorial2(2*rank+1)*spherical_jn(rank, x[2:])/x[2:]**rank, rel=2e-14)
