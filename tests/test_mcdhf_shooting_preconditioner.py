"""The radial accelerator must preserve the full variational problem."""
import numpy as np
import pytest

from atomx.numerics.shooting.preconditioner import RadialGreen
from atomx.methods.mcdhf.variational import CorrelatedOrbitalFunctional, optimize_orbitals
from atomx.methods.mcdhf.config import enumerate_jj_csfs
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.dirac_assembly import solve_dirac


def test_public_option_rejects_incompatible_protocols():
    from atomx import Atom, Correlation
    from atomx.methods.mcdhf.solver import MCDHFSolver
    with pytest.raises(ValueError, match='orbital_preconditioner'):
        Correlation(orbital_preconditioner='unknown')
    with pytest.raises(ValueError, match='mcdhf'):
        Correlation(radial_optimization='fac_frozen', orbital_preconditioner='shooting')
    with pytest.raises(ValueError, match='multi-configuration'):
        Atom(2, config='1s2', correlation=Correlation(orbital_preconditioner='shooting'))
    csfs = [c for cfg in ('1s2', '2s2') for c in enumerate_jj_csfs(cfg)]
    with pytest.raises(ValueError, match='B-spline'):
        MCDHFSolver(2, csfs, engine='shooting', orbital_preconditioner='shooting')


def test_public_option_reaches_correlated_optimizer():
    from atomx import Atom, Basis, Correlation
    result = Atom(2, config=['1s2', '2s2'], nuclear='point',
                  engine='bspline', basis=Basis(ns=35, r_max=30),
                  corrections='none',
                  correlation=Correlation(orbital_preconditioner='shooting')).solve(
                      max_iter=1).result
    report = result.orbital_optimization
    assert report['update_preconditioner'] == 'shooting'
    assert report['functional'] == 'correlated Dirac-Coulomb CI'
    assert np.max(np.diff(report['energy_history'])) <= 1e-12


def test_green_sweeps_match_dirichlet_kernel_and_are_positive():
    r = np.geomspace(1e-6, 9.9, 130)
    weights = np.gradient(r)
    green = RadialGreen(r, weights, 10., shift=.7)
    a = np.sqrt(1.4)
    low, high = np.minimum.outer(r, r), np.maximum.outer(r, r)
    kernel = 2*np.sinh(a*low)*np.sinh(a*(10-high))/(a*np.sinh(a*10))
    rng = np.random.default_rng(413)
    sources = rng.normal(size=(r.size, 3))
    result = green.apply(sources)
    np.testing.assert_allclose(result, kernel @ (weights[:, None]*sources), atol=2e-14)
    bilinear = sources.T @ (weights[:, None]*result)
    np.testing.assert_allclose(bilinear, bilinear.T, atol=2e-14)
    assert np.linalg.eigvalsh(bilinear).min() > 0


def test_green_solves_manufactured_differential_equation():
    errors = []
    for n in (400, 800):
        r = (np.arange(n)+.5)/n*8
        source = np.sin(np.pi*r/8)
        result = RadialGreen(r, np.full(n, 8/n), 8).apply(source[:, None])[:, 0]
        exact = source/(.5+.5*(np.pi/8)**2)
        errors.append(np.max(abs(result-exact)))
    assert errors[1] < errors[0]/3.9
    assert errors[1] < 4e-5


@pytest.mark.parametrize('ell', [1, 2, 4])
def test_centrifugal_green_solves_manufactured_equation(ell):
    # u=r^(l+1)(R-r), regular at the origin and zero at R.
    # (-u''+l(l+1)u/r²)/2 = (l+1)r^l, analytically.
    errors = []
    for n in (400, 800):
        r = (np.arange(n)+.5)/n*4
        exact = r**(ell+1)*(4-r)
        source = (ell+1)*r**ell + .7*exact
        green = RadialGreen(r, np.full(n, 4/n), 4, shift=.7, ell=ell)
        result = green.apply(source[:, None])[:, 0]
        errors.append(np.max(abs(result-exact))/np.max(exact))
    assert errors[1] < errors[0]/3.9
    assert errors[1] < 5e-5


def helium_problem():
    data = prepare_bspline_data(make_knots(35, 9, .005, 30), 35, 9)
    csfs = [c for cfg in ('1s2', '2s2', '2p2')
            for c in enumerate_jj_csfs(cfg) if c.total_J2 == 0 and c.parity == 1]
    functional = CorrelatedOrbitalFunctional(csfs, data, 2)
    orbitals = {}
    for kappa in functional.metric:
        spec = solve_dirac(functional.one_body[kappa], functional.metric[kappa], 35, kappa)
        ell = kappa if kappa > 0 else -kappa-1
        for n, k in functional.keys:
            if k == kappa:
                p, q, _ = spec.get_orbital(n-ell-1+int(kappa > 0))
                orbitals[n, k] = p, q
    return functional, orbitals


def test_correlated_shooting_preserves_constraints_and_convergence_metric(monkeypatch):
    functional, orbitals = helium_problem()
    initial = []
    for mode in ('spectral', 'shooting'):
        _, _, report = optimize_orbitals(functional, orbitals, max_iter=0,
                                         orbital_preconditioner=mode)
        initial.append(report)
    assert initial[0]['orbital_residual'] == pytest.approx(initial[1]['orbital_residual'])
    assert initial[0]['estimated_energy_error'] == pytest.approx(initial[1]['estimated_energy_error'])
    waves, ci, report = optimize_orbitals(functional, orbitals, max_iter=12,
                                          orbital_preconditioner='shooting')
    assert ci.ground_energy < initial[0]['energy_history'][0]-1e-3
    assert np.max(np.diff(report['energy_history'])) <= 1e-12
    for kappa in functional.metric:
        coeff = np.column_stack([np.concatenate(waves[k]) for k in waves if k[1] == kappa])
        metric = functional.metric[kappa]
        np.testing.assert_allclose(coeff.T @ metric @ coeff, np.eye(coeff.shape[1]), atol=2e-12)
        spec = solve_dirac(functional.one_body[kappa], metric, 35, kappa)
        np.testing.assert_allclose(coeff, spec.coefficients @ (spec.coefficients.T @ metric @ coeff), atol=2e-10)

    original = functional.evaluate
    calls = 0
    def reject_first(*args, **kwargs):
        nonlocal calls
        if kwargs.get('gradient') is False:
            calls += 1
            if calls <= 28:
                raise FloatingPointError('test rejection')
        return original(*args, **kwargs)
    monkeypatch.setattr(functional, 'evaluate', reject_first)
    _, fallback_ci, fallback = optimize_orbitals(functional, orbitals, max_iter=1,
                                                orbital_preconditioner='shooting')
    assert fallback['line_search_fallbacks'] == 1
    assert fallback['shooting_fallback_reason'] == 'shooting/secant line search exhausted'
    assert fallback_ci.ground_energy < initial[0]['energy_history'][0]


def test_both_updates_reach_same_correlated_stationary_energy():
    functional, orbitals = helium_problem()
    rows = []
    for mode in ('spectral', 'shooting'):
        _, ci, report = optimize_orbitals(functional, orbitals, max_iter=300,
                                          orbital_preconditioner=mode,
                                          tolerance=1e-5, energy_tolerance=1e-8)
        assert report['converged'], report
        assert report['orbital_residual'] < 1e-5
        rows.append(ci.ground_energy)
    assert abs(rows[1]-rows[0]) < 1e-8
