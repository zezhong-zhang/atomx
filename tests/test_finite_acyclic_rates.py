"""Finite positive triangular integration against independent CTMC solutions."""

import numpy as np
import pytest
from scipy.linalg import expm

from atomx.numerics.acyclic_rates import finite_acyclic_rates, integrated_acyclic_rates
from atomx.plasma.population_network import population_generator


def test_branching_all_edge_rewards_and_energy_against_dense_exponential():
    source = np.array([3, 3, 2, 2, 1, 3])
    destination = np.array([2, 1, 0, 1, 0, 1])
    rates = np.array([4., 1., 2., 3., 6., 2.])
    initial = np.array([0., 0., 0., 2.])
    generator = population_generator(4, source, destination, rates).toarray()
    augmented = np.zeros((10, 10))
    augmented[:4, :4] = generator
    for edge, (u, rate) in enumerate(zip(source, rates, strict=True)):
        augmented[4+edge, u] = rate
    exact = expm(.4*augmented) @ np.r_[initial, np.zeros(6)]
    errors = []
    for steps in (256, 1024, 4096):
        actual = finite_acyclic_rates(np.arange(4), source, destination, rates,
                                     initial, np.linspace(0, .4, steps+1))
        observed = np.r_[actual['populations'], actual['transition_counts']]
        errors.append(np.linalg.norm(observed-exact, ord=1))
        assert np.all(observed >= 0)
        energy = np.array([0., 1., 3., 8.])
        emitted = actual['transition_counts'] @ (energy[source]-energy[destination])
        assert emitted + actual['populations'] @ energy == pytest.approx(
            initial @ energy, abs=5e-12)
    assert errors[1] < errors[0]/3.8
    assert errors[2] < errors[1]/3.8
    assert errors[-1] < .001


def test_two_state_finite_survival_is_distinct_from_infinite_time_yield():
    rate, duration = 3., .2
    time = np.linspace(0, duration, 4097)
    actual = finite_acyclic_rates([0, 1], [1], [0], [rate], [0., 1.], time)
    survival = np.exp(-rate*duration)
    assert actual['populations'][1] == pytest.approx(survival, abs=3e-5)
    assert actual['transition_counts'][0] == pytest.approx(1-survival, abs=3e-5)
    assert actual['residence_times'][1] == pytest.approx((1-survival)/rate, abs=1e-5)
    eventual = integrated_acyclic_rates([0, 1], [1], [0], [rate], [0., 1.])
    assert eventual['transition_counts'][0] == 1.
    assert actual['transition_counts'][0] < .5


def test_stiff_fast_and_slow_states_full_counts_and_nonuniform_grid():
    source, destination = np.array([2, 1]), np.array([1, 0])
    rates = np.array([1e250, 2.])
    initial = np.array([0., 0., 1.])
    edges = np.r_[0., np.geomspace(1e-253, 1e-4, 512), np.linspace(.001, 1., 3000)]
    result = finite_acyclic_rates([0, 1, 2], source, destination, rates, initial, edges)
    assert result['populations'][1] == pytest.approx(np.exp(-2.), abs=.0002)
    assert result['transition_counts'][0] == pytest.approx(1., abs=2e-14)
    assert abs(result['particle_defect']) < 1e-12
    np.testing.assert_allclose(result['populations'].sum(), 1., atol=1e-12, rtol=0)


def test_arbitrary_labels_tiny_input_zero_channels_and_terminal_residence():
    initial = np.array([0., 1e-80, 0.])
    result = finite_acyclic_rates([2, 0, 1], [1, 1, 0], [0, 2, 2],
                                 [2., 0., 3.], initial, np.linspace(0, 1., 1025))
    assert result['transition_counts'][1] == 0.
    assert result['populations'].sum() == pytest.approx(1e-80, rel=1e-12, abs=0)
    terminal = finite_acyclic_rates([1, 0], [], [], [], [.3, .7], [0., .2, 1.])
    np.testing.assert_array_equal(terminal['populations'], [.3, .7])
    np.testing.assert_allclose(terminal['residence_times'], [.3, .7], rtol=1e-15)


@pytest.mark.parametrize('grid', [[.1, 1.], [0., 0.], [0., np.nan], [0.], [[0., 1.]],
                                 np.array([0., 1.+1j])])
def test_invalid_temporal_geometry_rejects(grid):
    with pytest.raises(ValueError):
        finite_acyclic_rates([0, 1], [1], [0], [1.], [0., 1.], grid)


def test_complete_resources_and_cycle_reject_without_fast_elimination():
    with pytest.raises(MemoryError):
        finite_acyclic_rates([0, 1], [1], [0], [1.], [0., 1.], [0., 1., 2.], max_steps=1)
    with pytest.raises(MemoryError):
        finite_acyclic_rates([0, 1], [1], [0], [1.], [0., 1.], [0., 1.], memory_bytes=1)
    with pytest.raises(ValueError, match='decrease'):
        finite_acyclic_rates([0, 1], [0], [1], [1.], [0., 1.], [0., 1.])
    with pytest.raises(ValueError, match='resource'):
        finite_acyclic_rates([0, 1], [1], [0], [1.], [0., 1.], [0., 1.], max_steps=True)
