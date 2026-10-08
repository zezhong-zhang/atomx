"""Memory choices retain an independent-particle eigenproblem and its gates."""
from types import SimpleNamespace

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.orbital_optimization import optimize_orbitals


@pytest.mark.parametrize('invalid', [0, -1, True, 1.5])
def test_invalid_history_size_rejects_before_evaluating(invalid):
    with pytest.raises(ValueError, match='history_size'):
        optimize_orbitals(None, {}, history_size=invalid)


def independent_particle_problem():
    data = prepare_bspline_data(make_knots(35, 9, .005, 30), 35, 9)
    # Manufactured two-particle target without electron repulsion: its
    # minimum is exactly twice the one-body Rayleigh energy. N=2 exercises
    # secant updates rather than the optimizer's one-electron exact shortcut.
    class IndependentParticles(CSFOrbitalFunctional):
        def evaluate(self, waves, *, gradient=True, ci_result=None):
            x = np.concatenate(waves[1, -1])
            hx = self.one_body[-1] @ x
            energy = 2*float(x @ hx)
            ci = SimpleNamespace(ground_energy=energy)
            if not gradient:
                return energy, ci
            self.last_occupations = {(1, -1): 2.}
            return energy, {(1, -1): 4*hx}, ci

    functional = IndependentParticles(enumerate_jj_csfs('1s2'), data, 1)
    spectrum = solve_dirac(functional.one_body[-1], functional.metric[-1], 35, -1)
    coefficients = spectrum.coefficients[:, :3] @ np.array([.9, .3, .1])
    coefficients /= np.sqrt(coefficients @ functional.metric[-1] @ coefficients)
    return functional, {(1, -1): (coefficients[:35], coefficients[35:])}, spectrum


def test_explicit_default_preserves_trajectory_bitwise():
    functional, waves, _ = independent_particle_problem()
    a, _, default = optimize_orbitals(functional, waves, max_iter=12)
    b, _, explicit = optimize_orbitals(functional, waves, max_iter=12, history_size=10)
    np.testing.assert_array_equal(default['energy_history'], explicit['energy_history'])
    for left, right in zip(a[1, -1], b[1, -1], strict=True):
        np.testing.assert_array_equal(left, right)


def test_larger_history_reaches_same_positive_energy_eigenstate():
    functional, waves, spectrum = independent_particle_problem()
    final, ci, report = optimize_orbitals(functional, waves, max_iter=100,
                                         history_size=80, tolerance=1e-7)
    assert report['converged'] and report['orbital_residual'] < 1e-7
    assert report['estimated_energy_error'] < 1e-10
    reference = spectrum.coefficients[:, 0]
    # Reevaluate the directly diagonalized eigenvector's Rayleigh energy;
    # raw generalized Dirac eigenvalues have rest-energy cancellation noise.
    exact_basis_energy = reference @ functional.one_body[-1] @ reference
    exact_basis_energy /= reference @ functional.metric[-1] @ reference
    assert ci.ground_energy == pytest.approx(2*exact_basis_energy, abs=2e-12)
    assert report['iterations'] > 1
    assert np.max(np.diff(report['energy_history'])) <= 0
    x = np.concatenate(final[1, -1])
    assert x @ functional.metric[-1] @ x == pytest.approx(1., abs=2e-12)
    # The physical state is unique up to phase, checked against direct Dirac.
    overlap = x @ functional.metric[-1] @ spectrum.coefficients[:, 0]
    assert abs(overlap) > 1-1e-10


def test_endpoint_restart_declares_changed_residual_metric():
    functional, waves, _ = independent_particle_problem()
    final, _, report = optimize_orbitals(functional, waves, max_iter=12)
    _, _, repeated = optimize_orbitals(functional, waves, max_iter=0)
    _, _, restarted = optimize_orbitals(functional, final, max_iter=0)
    assert report['residual_metric_sha256'] == repeated['residual_metric_sha256']
    assert report['residual_metric_sha256'] != restarted['residual_metric_sha256']
