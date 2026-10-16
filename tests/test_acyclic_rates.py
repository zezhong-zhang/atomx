"""Integrated graph flow versus independent continuous-time rate equations."""

import numpy as np
import pytest
from scipy.linalg import expm

from atomx.numerics.acyclic_rates import integrated_acyclic_rates
from atomx.plasma.population_network import population_generator


def test_branching_energy_rewards_and_rate_equation_exponential():
    #3 ->2 and1;2 ->0 and1;1 ->0. A second physical3->1 edge is retained.
    source = np.array([3,3,2,2,1,3])
    destination=np.array([2,1,0,1,0,1])
    rates=np.array([4.,1.,2.,3.,6.,2.])
    initial=np.array([0.,0.,0.,2.])
    actual=integrated_acyclic_rates(np.arange(4),source,destination,rates,initial)
    generator=population_generator(4,source,destination,rates).toarray()
    augmented=np.zeros((10,10))
    augmented[:4,:4]=generator
    for i,(u,rate) in enumerate(zip(source,rates,strict=True)):
        augmented[4+i,u]=rate
    result=expm(30*augmented)@np.r_[initial,np.zeros(6)]
    np.testing.assert_allclose(actual['terminal_weights'],result[:4],atol=2e-13,rtol=2e-13)
    np.testing.assert_allclose(actual['transition_counts'],result[4:],atol=2e-13,rtol=2e-13)
    assert actual['particle_defect']==pytest.approx(0,abs=1e-14)
    energies=np.array([0.,1.,3.,8.])
    emitted=actual['transition_counts']@(energies[source]-energies[destination])
    assert emitted+actual['terminal_weights']@energies==pytest.approx(initial@energies,abs=2e-13)


def test_arbitrary_labels_exact_zero_rates_and_tiny_unnormalized_input():
    order=np.array([2,0,1])
    src=np.array([1,1,0])
    dst=np.array([0,2,2])
    rates=np.array([2.,0.,3.])
    actual=integrated_acyclic_rates(order,src,dst,rates,np.array([0.,1e-80,0.]))
    assert actual['terminal_weights'][2]==pytest.approx(1e-80,rel=1e-14,abs=0)
    np.testing.assert_allclose(actual['transition_counts'],[1e-80,0,1e-80],rtol=1e-14,atol=0)
    assert actual['residence_times'][1]==pytest.approx(.5e-80,rel=1e-14,abs=0)


def test_cyclic_invalid_and_complete_resource_admission():
    with pytest.raises(ValueError,match='decrease'):
        integrated_acyclic_rates([0,1],[0],[1],[1.],[1.,0.])
    with pytest.raises(ValueError,match='complete order'):
        integrated_acyclic_rates([1,1],[1],[0],[1.],[1.,0.])
    with pytest.raises(MemoryError):
        integrated_acyclic_rates([0,1],[1],[0],[1.],[1.,0.],memory_bytes=1)
    with pytest.raises(ValueError):
        integrated_acyclic_rates([0,1],[1],[0],[-1.],[1.,0.])
    with pytest.raises(ValueError,match='total initial'):
        integrated_acyclic_rates([0,1],[],[],[],[1e308,1e308])
    terminal=integrated_acyclic_rates([1,0],[],[],[],[.3,.7])
    np.testing.assert_array_equal(terminal['terminal_weights'],[.3,.7])
