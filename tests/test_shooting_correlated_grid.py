"""Independent derivatives and boundary-value tests for full shooting work."""
import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs_many
from atomx.methods.csf.grid_functional import (
    GridCSFFunctional,
    coulomb_adjoint,
    derivative_adjoint,
)
from atomx.numerics.shooting.constants import SL
from atomx.numerics.shooting.coulomb import _yk_over_r
from atomx.numerics.shooting.dirac_response import DiracResponse
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.quadrature import slag6, slag6_adjoint, slag6_weights
from atomx.physics.csf import matrix


@pytest.mark.parametrize('n', [3, 6, 9, 83])
def test_cumulative_quadrature_adjoint(n):
    rng = np.random.default_rng(n)
    a, b = rng.normal(size=(2, n))
    assert slag6(.7, a, n)@b == pytest.approx(a@slag6_adjoint(.7, b, n), abs=2e-13)
    assert slag6_weights(n)@a == pytest.approx(slag6(1., a, n)[-1], abs=2e-13)


def workspace(n=250):
    r, difr, _ = sgrid(30., 1e-7, .5, n)
    return matrix.GridCIWorkspace(r, difr, np.full(len(r), -2.))


def test_radial_operator_adjoints():
    ws = workspace()
    rng = np.random.default_rng(37)
    a, b = rng.normal(size=(2, ws.n))
    assert matrix._ds_derivative(a)@b == pytest.approx(a@derivative_adjoint(b), abs=2e-11)
    # Smooth origin-regular probes avoid ill-conditioned large multipole
    # weights from unphysical white-noise sources at r=1e-7.
    for rank in (0, 1, 3):
        a = ws.r**(rank+2)*np.exp(-ws.r)
        b = ws.r**(rank+2)*np.exp(-2*ws.r)*ws.difr
        lhs = b@_yk_over_r(a, rank, ws.r, ws.difr, ws.n)
        rhs = a@coulomb_adjoint(b, rank, ws)
        assert lhs == pytest.approx(rhs, rel=2e-13, abs=1e-13)


def correlated_problem():
    ws = workspace()
    csfs = [c for c in enumerate_jj_csfs_many(['1s2', '1s1 2s1', '2s2', '2p2'])
            if c.total_J2 == 0 and c.parity == 1]
    functional = GridCSFFunctional(csfs, ws, 2)
    orbitals = {}
    r = ws.r
    for n, kappa in functional.keys:
        ell = kappa if kappa > 0 else -kappa-1
        p = r**(ell+1)*np.exp(-r*2/n)
        q = p*(.1+.03*r)/SL
        orbitals[n, kappa] = np.array([p, q])
    for kappa in sorted({k[1] for k in orbitals}):
        keys = [k for k in orbitals if k[1] == kappa]
        x = np.stack([orbitals[k] for k in keys], axis=-1)
        gram = np.einsum('ari,r,arj->ij', x, functional.weights, x)
        e, v = np.linalg.eigh(gram)
        x = np.einsum('ari,ij->arj', x, (v/np.sqrt(e))@v.T)
        for i, key in enumerate(keys):
            orbitals[key] = x[:, :, i].copy()
    return functional, orbitals


def test_full_ci_grid_gradient_matches_energy_and_constraints():
    functional, orbitals = correlated_problem()
    energy, grad, _ = functional.evaluate(orbitals)
    ws = functional.data
    # Independently compare the one-body derivative operator to the existing
    # CI integral, including its symmetric derivative and boundary terms.
    for key, phi in orbitals.items():
        actual = np.sum(phi*functional.one_body_action(phi, key[1]))
        expected = matrix._grid_h_one_body(*phi, *phi, key[1], ws)
        assert actual == pytest.approx(expected, abs=2e-12)
    directions = {k: phi*np.array([np.sin(ws.r), np.cos(ws.r)])
                  for k, phi in orbitals.items()}
    directions, _ = functional.constrained_residual(
        orbitals, {k: v*functional.weights for k, v in directions.items()})
    analytical = sum(np.sum(grad[k]*directions[k]) for k in orbitals)
    h = 1e-5
    plus = functional.retract({k: orbitals[k]+h*directions[k] for k in orbitals})
    minus = functional.retract({k: orbitals[k]-h*directions[k] for k in orbitals})
    finite = (functional.evaluate(plus, gradient=False)[0]
              - functional.evaluate(minus, gradient=False)[0])/(2*h)
    assert analytical == pytest.approx(finite, rel=2e-8, abs=2e-8)
    # Probe the evaluator's explicit h_ab/<b|b> extension just inside its
    # metric validation tolerance. This normal derivative would be wrong
    # if the denominator were silently treated as constant.
    h_normal = 1e-8
    normal_fd = (functional.evaluate({k: (1+h_normal)*v for k, v in orbitals.items()}, gradient=False)[0]
                 - functional.evaluate({k: (1-h_normal)*v for k, v in orbitals.items()}, gradient=False)[0])/(2*h_normal)
    normal_exact = sum(np.sum(grad[k]*orbitals[k]) for k in orbitals)
    assert normal_exact == pytest.approx(normal_fd, rel=2e-6, abs=2e-6)
    assert any(k[0] == 'r' for k in functional.last_weights)
    assert any(k[0] == 'h' and k[1] != k[3] for k in functional.last_weights)
    residual, multipliers = functional.constrained_residual(orbitals, grad)
    for kappa in multipliers:
        keys = [k for k in orbitals if k[1] == kappa]
        cross = np.array([[np.sum(orbitals[a]*residual[b]*functional.weights)
                           for b in keys] for a in keys])
        np.testing.assert_allclose(cross+cross.T, 0., atol=2e-12)
    assert np.isfinite(energy)


@pytest.mark.parametrize('boundary', ['decaying', 'box'])
def test_dirac_green_manufactured_solution_and_coupled_constraints(boundary):
    r, difr, n = sgrid(75., 1e-7, .5, 1000)
    energy, kappa, z = -.7, -1, 1.
    p = r*np.exp(-2*r)
    q = .01*r*r*np.exp(-2*r)
    dp = (1-2*r)*np.exp(-2*r)
    dq = .02*r*(1-r)*np.exp(-2*r)
    inv_r = np.divide(1., r, out=np.zeros_like(r), where=r > 0)
    source = np.array([(-z*inv_r-energy)*p+SL*(-dq+kappa*inv_r*q),
                       SL*(dp+kappa*inv_r*p)+(-z*inv_r-2*SL**2-energy)*q])
    green = DiracResponse(energy, kappa, r, difr, np.full(n, -z), boundary=boundary)
    actual = green.apply(source)
    expected = np.array([p, q])
    relative = np.sqrt(np.sum((actual-expected)**2*green.weights)
                       /np.sum(expected**2*green.weights))
    assert relative < 2e-6
    constraints = np.array([expected, expected*r])
    result, multipliers, report = green.solve(source, constraints)
    assert report['constraint_residual'] < 1e-12
    reconstructed = green.apply(source+np.einsum('i,iar->ar', multipliers, constraints))
    np.testing.assert_allclose(result, reconstructed, atol=1e-12)
    # Post-hoc Gram subtraction would not solve this bordered equation.
    assert np.linalg.norm(multipliers) > 1e-5
    with pytest.raises(np.linalg.LinAlgError):
        green.solve(source, [expected, expected])


def test_box_response_keeps_sources_at_outer_boundary():
    r, difr, n = sgrid(4., 1e-7, .1, 600)
    p, q = r*(4-r), .01*r*r
    dp, dq = 4-2*r, .02*r
    inv_r = np.divide(1., r, out=np.zeros_like(r), where=r > 0)
    energy = -.7
    source = np.array([(-inv_r-energy)*p+SL*(-dq-inv_r*q),
                       SL*(dp-inv_r*p)+(-inv_r-2*SL**2-energy)*q])
    response = DiracResponse(energy, -1, r, difr, -np.ones(n), boundary='box')
    actual = response.apply(source)
    expected = np.array([p, q])
    error = np.sqrt(np.sum((actual-expected)**2*response.weights)
                    /np.sum(expected**2*response.weights))
    assert error < 2e-6
    assert abs(actual[0, -1]) < 1e-12
    assert abs(actual[1, -1]-.16) < 1e-7


def shooting_electron_problem(z=2, nuclear='point'):
    from atomx.numerics.shooting.electron_space import bound_electron_space
    from atomx.numerics.shooting.hartree_fock import ShootingHFSolver
    r, difr, n = sgrid(200./z, 1e-7, .5, 1000)
    ws = matrix.GridCIWorkspace(r, difr, ShootingHFSolver(z, [], nuclear=nuclear)._nuclear_rv(r, difr, n))
    csfs = [c for c in enumerate_jj_csfs_many(['1s2', '2s2', '2p2'])
            if c.total_J2 == 0 and c.parity == 1]
    functional = GridCSFFunctional(csfs, ws, z)
    space, orbitals = bound_electron_space(functional, n_roots=3)
    return functional, orbitals, space


def test_full_grid_shooting_updates_lower_ci_energy_without_leaving_electron_space():
    from atomx.numerics.shooting.orbital_optimization import optimize_grid_shooting
    functional, orbitals, space = shooting_electron_problem()
    waves, ci, report = optimize_grid_shooting(functional, orbitals, space,
        {key: 1.1*min(space.root_energies[key[1]]) for key in functional.keys}, max_iter=8, tolerance=1e-12)
    assert ci.ground_energy < report['energy_history'][0]-1e-4
    assert np.max(np.diff(report['energy_history'])) <= 1e-12
    assert report['maximum_border_constraint_residual'] < 1e-10
    assert not report['converged']
    for key, phi in waves.items():
        np.testing.assert_allclose(phi, space.project(phi, key[1]), atol=2e-12)
    # Deliberately insufficient test basis: the API must record that scope.
    assert 'continuum omitted' in report['electron_space']


def test_heavy_grid_shooting_cannot_drift_out_of_fixed_electron_space():
    from atomx.numerics.shooting.orbital_optimization import optimize_grid_shooting
    functional, orbitals, space = shooting_electron_problem(54, 'fermi')
    leakage = []
    def check(iteration, waves, ci, status):
        leakage.append(max(np.sqrt(np.sum((v-space.project(v, k[1]))**2*functional.weights))
                           for k, v in waves.items()))
    waves, ci, report = optimize_grid_shooting(functional, orbitals, space,
        {k: 1.1*min(space.root_energies[k[1]]) for k in functional.keys},
        max_iter=250, progress_callback=check)
    assert max(leakage) < 1e-11
    assert report['electron_space_leakage'] < 1e-11
    assert np.max(np.diff(report['energy_history'])) <= report['energy_resolution_ha']
    assert report['converged'] == (report['orbital_residual'] < report['orbital_tolerance'])
