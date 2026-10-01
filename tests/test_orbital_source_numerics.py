"""Representation-independent rotation and normalized source-equation oracles."""
import numpy as np
import pytest

from atomx.numerics.bspline.inhomogeneous import solve_constrained_source
from atomx.numerics.orbital_rotations import rotate_occupied


@pytest.mark.parametrize('saddle', [False, True])
def test_rotation_resolves_odd_coherent_harmonics(saddle):
    pa, qa = np.eye(2), np.zeros((2,2))
    angle = .037
    sign = -1 if saddle else 1
    def energy():
        x, y = pa[0]
        return sign*(1-(np.cos(angle)*x+np.sin(angle)*y)**3)
    rotate_occupied(pa, qa, [(0,1,saddle)], energy, harmonics=(1,2,3,4))
    assert np.arctan2(pa[0,1], pa[0,0]) == pytest.approx(angle, abs=1e-8)
    np.testing.assert_allclose(pa@pa.T, np.eye(2), atol=1e-14)


def test_source_equation_retains_negative_energy_response_and_constraints():
    # Independent manufactured solution with a negative-energy component,
    # an occupied constraint, and a nonidentity Galerkin overlap metric.
    metric = np.diag([2., 3., 4., 5.])
    energies = np.array([-40000., -3., -1., 2.])
    hamiltonian = np.diag(energies*np.diag(metric))
    vector = np.array([1e-5, 0., 1., .02])
    vector /= np.sqrt(vector@metric@vector)
    expected_energy = -1.2
    source = (hamiltonian-expected_energy*metric)@vector
    orbital, energy = solve_constrained_source(hamiltonian, metric, source, vector,
        principal=2, kappa=-1, constraints=[np.array([0.,1/np.sqrt(3),0.,0.])], lower_occupied=1)
    assert energy == pytest.approx(expected_energy, abs=1e-10)
    np.testing.assert_allclose(orbital, vector, atol=1e-10)
    np.testing.assert_allclose((hamiltonian-energy*metric)@orbital, source, atol=1e-10)


def test_excited_source_root_with_residue_at_both_neighboring_poles():
    energies = np.array([-40000.,-4.,-1.,2.])
    metric = np.eye(4)
    hamiltonian = np.diag(energies)
    source = np.array([0.,.1,.2,0.])
    reference = np.array([0.,0.,1.,0.])
    vector,energy = solve_constrained_source(hamiltonian,metric,source,reference,
                                            principal=2,kappa=-1)
    assert -1.3 < energy < -1.
    np.testing.assert_allclose(vector@vector,1.,atol=1e-12)
    np.testing.assert_allclose((hamiltonian-energy*metric)@vector,source,atol=1e-10)
