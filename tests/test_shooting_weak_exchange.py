"""Weak nonlocal exchange must approach a local eigenstate without pole failure."""
import numpy as np
import pytest

from atomx.numerics.shooting.dirac_bound import dbound
from atomx.numerics.shooting.dirac_orbital import solve_inhomogeneous_orbital
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.quadrature import slag6_weights
from atomx.numerics.shooting.spline import splin0


def test_normalization_search_leaves_preceding_pole_decreasing_branch():
    """An adjacent local pole must not trap the excited-orbital corrector."""
    from scipy.optimize import brentq
    r, jacobian, count = sgrid(75., 1e-7, .5, 3200)
    rv = np.full(count, -2.)
    potential = splin0(r, rv, count, 0., 0.)
    weights = slag6_weights(count)*jacobian
    states = []
    for n, guess in [(1, -2.), (2, -.5)]:
        e, error, p, q, _ = dbound(guess, 1e-11, n, -1, r, rv,
                                  *potential, count, r, count)
        assert error == 0
        phi = np.array([p, q])
        phi /= np.sqrt(np.sum(phi*phi*weights))
        states.append((e, phi))
    e1, phi1 = states[0]
    e2, phi2 = states[1]
    strength = .02
    chi = strength*(phi1+phi2)
    expected = brentq(lambda e: strength**2*((e1-e)**-2+(e2-e)**-2)-1.,
                      (e1+e2)/2, e2-1e-5, xtol=1e-14)
    p, q, energy = solve_inhomogeneous_orbital(e1+.01, 2, -1, rv, *chi,
        r, np.arange(count), jacobian, count)
    assert energy == pytest.approx(expected, abs=2e-9)
    wave = np.array([p, q])
    exact = strength*phi1/(e1-expected)+strength*phi2/(e2-expected)
    assert np.sqrt(np.sum((wave-exact)**2*weights)) < 1e-7


def test_sub_ulp_pole_source_retains_orthogonal_response():
    r, jacobian, count = sgrid(75., 1e-7, .5, 3200)
    rv = np.full(count, -2.)
    potential = splin0(r, rv, count, 0., 0.)
    weights = slag6_weights(count)*jacobian
    states = []
    for n, guess in [(1,-2.), (2,-.5)]:
        energy, error, p, q, _ = dbound(guess,1e-11,n,-1,r,rv,
            *potential,count,r,count)
        assert error == 0
        phi = np.array([p,q])
        phi /= np.sqrt(np.sum(phi*phi*weights))
        states.append((energy,phi))
    e1,phi1 = states[0]
    e2,phi2 = states[1]
    chi = 1e-17*phi1+1e-6*phi2
    p,q,energy = solve_inhomogeneous_orbital(e1-1e-4,1,-1,rv,*chi,
        r,np.arange(count),jacobian,count)
    wave = np.array([p,q])
    if np.sum(wave*phi1*weights)<0:
        wave *= -1
    b = 1e-6/(e2-energy)
    expected = np.sqrt(1-b*b)*phi1+b*phi2
    error = np.sqrt(np.sum((wave-expected)**2*weights))
    assert error < 1e-8
    assert energy == pytest.approx(e1,abs=1e-11)
    # The orthogonal exchange response is retained, not replaced by phi1.
    assert np.sum(wave*phi2*weights) == pytest.approx(b,rel=1e-3,abs=1e-10)


@pytest.mark.parametrize('configuration', ['1s1 10d1','1s1 13i1'])
def test_excited_helium_matches_independent_bspline(configuration):
    from atomx.config.jj_coupling import generate_jj_average, JJConfig
    from atomx.numerics.shooting.hartree_fock import ShootingHFSolver
    from atomx.methods.dirac_fock import DiracFock
    from atomx.config.parsing import parse_nr_shells
    shells,ensemble = generate_jj_average(configuration,averaging='statistical')
    kept = [i for i,s in enumerate(shells) if s.occupation>0]
    shells = [shells[i] for i in kept]
    ensemble = [JJConfig(subshells=[c.subshells[i] for i in kept],weight=c.weight)
                for c in ensemble]
    radius = 5*max(n for n,_,_ in parse_nr_shells(configuration))**2
    shooting = ShootingHFSolver(2,shells,aw=4,nuclear='fermi',jj_configs=ensemble,
        n_grid=3200,r_max=radius).solve(max_iter=200,tol=1e-12)
    reference = DiracFock(breit=False,qed=False,nuclear_mass_number=4,
        starting_potential='gsz',scf_damping=.5,scf_damping_release_residual=1e-10).solve(
        # The diffuse B-spline orbital-change residual stalls near 1e-8.
        # Keep the independent energy gate below unchanged and verify its
        # radial refinement, rather than claiming a 1e-10 orbital residual.
        2,shells,jj_configs=ensemble,max_iter=200,tol=1e-7,
        r_max=radius,grid_hi=.125,grid_he=.0625,grid_hmax=5.)
    assert shooting.converged and reference.converged
    assert shooting.total_energy == pytest.approx(reference.total_energy,abs=2e-9,rel=0)
    if configuration=='1s1 10d1':
        refined = DiracFock(breit=False,qed=False,nuclear_mass_number=4,
            starting_potential='gsz',scf_damping=.5,scf_damping_release_residual=1e-10).solve(
            2,shells,jj_configs=ensemble,max_iter=200,tol=1e-7,
            r_max=radius,grid_hi=.125,grid_he=.0625,grid_hmax=2.5)
        assert refined.converged
        assert reference.total_energy == pytest.approx(refined.total_energy,abs=2e-9,rel=0)
