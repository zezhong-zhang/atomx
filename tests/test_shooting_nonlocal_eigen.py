"""Nonlocal shooting eigenproblems with analytically known spectral shifts."""
import numpy as np
import pytest

from atomx.numerics.shooting.dirac_bound import dbound
from atomx.numerics.shooting.dirac_orbital import OrbitalSolveError
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.nonlocal_eigen import solve_nonlocal_bound_state
from atomx.numerics.shooting.problem import ShootingRadialProblem


@pytest.fixture(scope='module')
def coulomb_2s():
    r, dr, count = sgrid(75., 1e-7, .1, 1600)
    problem = ShootingRadialProblem(r, dr, np.full(count, -1.))
    e, error, p, q, _ = dbound(-.125, 1e-12, 2, -1, r, problem.rv,
                              *problem.spline, count, r, count)
    assert error == 0
    phi = np.array([p, q])
    phi /= np.sqrt(np.sum(phi*phi*problem.weights))
    return problem, phi, e


def test_rank_one_exchange_has_exact_eigenvalue_shift(coulomb_2s):
    problem, phi, local_energy = coulomb_2s
    strength = .02
    def exchange(psi):
        return strength*phi*np.sum(phi*psi*problem.weights)
    p, q, energy, report = solve_nonlocal_bound_state(
        local_energy-.015, 2, -1, problem, phi, exchange)
    assert abs(energy-(local_energy-strength)) < 2e-9
    assert abs(np.sum((p*p+q*q)*problem.weights)-1.) < 1e-12
    assert report['overlap'] > 1-1e-8
    assert report['green_equation_residual'] < 1e-8
    assert report['dominant_nodes'] == 1


def test_nonlocal_continuation_rejects_wrong_state(coulomb_2s):
    problem, phi, energy = coulomb_2s
    def exchange(psi):
        return .02*phi*np.sum(phi*psi*problem.weights)
    with pytest.raises(OrbitalSolveError, match='accepted root'):
        solve_nonlocal_bound_state(energy-.015, 3, -1, problem, phi, exchange)


def test_inverse_shift_avoids_exact_physical_eigenvalue(coulomb_2s):
    problem, phi, local_energy = coulomb_2s
    def exchange(psi):
        return .02*phi*np.sum(phi*psi*problem.weights)
    _, _, energy, report = solve_nonlocal_bound_state(
        local_energy-.02, 2, -1, problem, phi, exchange)
    assert abs(energy-(local_energy-.02)) < 2e-9
    assert report['inverse_shift'] < local_energy-.02
    assert report['green_equation_residual'] < 1e-8


def test_zero_exchange_limit_does_not_invert_at_a_local_pole(coulomb_2s):
    problem, phi, local_energy = coulomb_2s
    _, _, energy, report = solve_nonlocal_bound_state(
        local_energy, 2, -1, problem, phi, lambda psi: np.zeros_like(psi))
    assert abs(energy-local_energy) < 2e-9
    assert report['overlap'] > 1-1e-8
    assert report['green_equation_residual'] < 1e-8
    assert report['residual_form'] == 'shifted_green_identity'


@pytest.mark.parametrize('strength', [0., .02])
def test_local_anchor_recovers_known_root_from_mixed_seed(coulomb_2s, strength):
    problem, phi, expected_energy = coulomb_2s
    count = len(problem.r)
    _, error, p, q, _ = dbound(-.5, 1e-12, 1, -1, problem.r, problem.rv,
                             *problem.spline, count, problem.r, count)
    assert error == 0
    core = np.array([p, q])
    core /= np.sqrt(np.sum(core * core * problem.weights))
    seed = phi + .6 * core
    seed /= np.sqrt(np.sum(seed * seed * problem.weights))
    p, q, energy, report = solve_nonlocal_bound_state(
        expected_energy-strength, 2, -1, problem, seed,
        lambda psi: strength*phi*np.sum(phi*psi*problem.weights))
    assert abs(energy - (expected_energy-strength)) < 2e-9
    assert report['overlap_with_incoming_orbital'] < .9
    assert len(report['exchange_continuation']) == 4
    assert all(step['overlap'] >= .9 and step['green_equation_residual'] < 1e-7
               for step in report['exchange_continuation'])
