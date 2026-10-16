"""Explicit reservoir campaigns, outside the default essential suite.

Original inputs and acceptance rules moved unchanged from
``test_xfel_multielectron_ci.py`` after hosted timing on 10 October 2026.
"""

import numpy as np
import pytest
from test_xfel_multielectron_ci import build, continuum
from test_xfel_multielectron_ci import joint as joint
from test_xfel_multielectron_ci import neon3 as neon3

from atomx.scattering.multielectron_ci import MultiElectronCIChannels
from atomx.xfel import CoherentGaussianPulse, MultiElectronCoherentModel


def test_absolute_reservoir_certification_reuses_three_electron_signed_model(joint):
    original, pulse = joint
    ladders = {n: {i: int(n == 3 and i != 0) for i in range(len(r.state_energies))}
               for n, r in original.channels.sectors.items()}
    bounded = MultiElectronCoherentModel(original.channels, pulse, photon_numbers=ladders,
        continuum_model=continuum(), radiative_reservoir='geometric_bounded',
        electron_reservoir='clustered_bounded')
    for name in ('decay_operators', 'pi_operators'):
        for a, b in zip(getattr(original.system, name), getattr(bounded.system, name), strict=True):
            np.testing.assert_array_equal(a, b)
    times = np.linspace(pulse.support_fs[0], pulse.support_fs[1]+30, 51)
    a = original.solve(pulse, times_fs=times, after_pulse_fs=30)
    b = bounded.solve(pulse, times_fs=times, after_pulse_fs=30)
    np.testing.assert_array_equal(a.density_matrices, b.density_matrices)
    assert 0 < b.provenance['reservoir_reference_trace_norm_error_bound'] < 1e-3


def test_four_electron_full_sparse_chain_retains_absolute_error_rejection():
    from scipy.sparse import issparse

    value = build(['1s2 2s2', '1s1 2s2 2p1', '1s2 2s1 2p1'])
    provider = MultiElectronCIChannels(value, continuum_model=continuum())
    line = max((l for l in value.transitions(multipoles=['E1']) if l.initial_state_index == 0),
               key=lambda l: l.A_rate_per_s)
    pulse = CoherentGaussianPulse(line.energy_eV, 1e17, 10)
    ladders = {n: {i: int(n == 4 and i != 0) for i in range(len(r.state_energies))}
               for n, r in provider.sectors.items()}
    bounded = MultiElectronCoherentModel(provider, pulse, photon_numbers=ladders,
        continuum_model=continuum(), radiative_reservoir='geometric_bounded',
        electron_reservoir='clustered_bounded')
    assert len(bounded.system.state_ids) == 201
    assert all(issparse(c) for c in bounded.system.decay_operators+bounded.system.pi_operators)
    assert bounded.system.provenance['electron_auger_generator_difference_bound_per_s'] > 1e12
    with pytest.raises(ValueError, match='combined reservoir.*trace-norm budget'):
        bounded.solve(pulse, after_pulse_fs=30)
