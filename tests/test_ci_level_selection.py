"""Basis ordering and degenerate-subspace checks for physical initial states."""

from types import SimpleNamespace

import numpy as np
import pytest

from atomx import CILevel, CILevelPopulation, CIStateAverage
from atomx.physics.csf.levels import initial_ci_level_weights


def spectrum(energies=(-3, -2, -1), sectors=((0, 1), (2, -1), (0, 1))):
    return (
        [SimpleNamespace(total_J2=j, parity=p) for j, p in sectors],
        SimpleNamespace(
            state_energies=np.array(energies, dtype=float),
            mixing_coefficients=np.eye(len(energies)),
        ),
    )


def test_physical_populations_survive_global_reordering_without_statistical_factor():
    csfs, result = spectrum()
    level = CILevel(0, 1, 1)
    population = CILevelPopulation(((level, 1e308), (CILevel(2, -1), 1e308)))
    assert level.resolve(csfs, result) == 2
    np.testing.assert_array_equal(population.weights(csfs, result), [0, 0.5, 0.5])
    result.state_energies = result.state_energies[[2, 0, 1]]
    result.mixing_coefficients = result.mixing_coefficients[:, [2, 0, 1]]
    assert level.resolve(csfs, result) == 0
    np.testing.assert_array_equal(population.weights(csfs, result), [0.5, 0, 0.5])
    # The optimization ensemble and physical preparation have separate meanings.
    np.testing.assert_allclose(
        CIStateAverage(statistical=True).weights(csfs, result), [0.2, 0.2, 0.6]
    )


def test_equal_population_is_invariant_under_degenerate_root_rotation():
    csfs, result = spectrum((-3, -3), ((0, 1), (0, 1)))
    theta = 0.6
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    population = CILevelPopulation(((CILevel(0, 1, 0), 2), (CILevel(0, 1, 1), 2)))
    weights = population.weights(csfs, result)
    before = result.mixing_coefficients @ np.diag(weights) @ result.mixing_coefficients.T
    result.mixing_coefficients = rotation
    after = rotation @ np.diag(population.weights(csfs, result)) @ rotation.T
    np.testing.assert_allclose(after, before, atol=1e-15)
    with pytest.raises(ValueError, match="degeneracy"):
        CILevel(0, 1).resolve(csfs, result)
    with pytest.raises(ValueError, match="degeneracy"):
        CILevelPopulation(((CILevel(0, 1, 0), 1), (CILevel(0, 1, 1), 2))).weights(
            csfs, result
        )
    # Legacy integers explicitly address a column of this recorded snapshot.
    np.testing.assert_array_equal(initial_ci_level_weights(0, csfs, result), [1, 0])


@pytest.mark.parametrize("args", [(True, 1), (0, 0), (-1, 1), (0, 1, -1), (0.0, 1)])
def test_selector_rejects_invalid_quantum_numbers(args):
    with pytest.raises(ValueError):
        CILevel(*args)


def test_invalid_or_missing_preparations_and_mixed_symmetry_fail_explicitly():
    csfs, result = spectrum()
    for weight in (0, -1, np.nan, np.inf, True):
        with pytest.raises(ValueError):
            CILevelPopulation(((CILevel(0, 1), weight),))
    with pytest.raises(ValueError, match="duplicate"):
        CILevelPopulation(((CILevel(0, 1), 1), (CILevel(0, 1), 1)))
    with pytest.raises(ValueError, match="outside"):
        CILevel(2, -1, 1).resolve(csfs, result)
    with pytest.raises(ValueError, match="outside"):
        initial_ci_level_weights(20, csfs, result)
    for selection in (False, 1.0, "ground", CIStateAverage()):
        with pytest.raises(TypeError, match="initial_state"):
            initial_ci_level_weights(selection, csfs, result)
    result.mixing_coefficients[:, :2] = np.array([[1, 1], [-1, 1], [0, 0]]) / np.sqrt(2)
    with pytest.raises(ValueError, match="symmetry-pure"):
        CILevel(0, 1).resolve(csfs, result)


def test_unpolarized_population_does_not_require_an_unrequested_full_spectrum():
    csfs, result = spectrum()
    result.state_energies = result.state_energies[:2]
    result.mixing_coefficients = result.mixing_coefficients[:, :2]
    np.testing.assert_array_equal(
        CILevelPopulation(((CILevel(2, -1), 1),)).weights(csfs, result), [0, 1]
    )
    with pytest.raises(ValueError, match="full real orthonormal"):
        CIStateAverage().weights(csfs, result)
