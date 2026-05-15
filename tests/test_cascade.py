"""X-ray / decay cascade calculator regression tests."""

import pytest

from atomx.plasma.cascade import (
    CascadeLevel,
    CascadeTransition,
    cascade_yields,
    fluorescence_yield,
    auger_yield,
)


def test_simple_two_level_cascade():
    """One excited level decaying to ground — all population ends in ground."""
    levels = [
        CascadeLevel('upper', 100.0),
        CascadeLevel('ground', 0.0),
    ]
    transitions = [
        CascadeTransition(upper=0, lower=1, rate_per_s=1e10, photon_eV=100.0),
    ]
    res = cascade_yields(levels, transitions, initial_idx=0)
    assert res.terminal_yields == pytest.approx({1: 1.0})
    assert res.photon_yields[(0, 1)] == pytest.approx(1.0)
    assert fluorescence_yield(res) == pytest.approx(1.0)
    assert auger_yield(res) == pytest.approx(0.0)


def test_k_shell_branching():
    """K-hole with radiative + Auger branches: branching ratios respected."""
    levels = [
        CascadeLevel('K-hole'),
        CascadeLevel('L-hole'),
        CascadeLevel('ground'),
    ]
    Gamma_r = 1e14
    Gamma_A = 5e14
    transitions = [
        CascadeTransition(0, 1, Gamma_r, photon_eV=6700.0),
        CascadeTransition(0, 2, Gamma_A, photon_eV=0.0),
        CascadeTransition(1, 2, 1e13, photon_eV=700.0),
    ]
    res = cascade_yields(levels, transitions, initial_idx=0)
    # K → L photon yield = Γ_r / (Γ_r + Γ_A) = 1/6
    assert res.photon_yields[(0, 1)] == pytest.approx(1.0 / 6.0, rel=1e-12)
    assert res.auger_yields[(0, 2)] == pytest.approx(5.0 / 6.0, rel=1e-12)
    # All population ends in ground
    assert res.terminal_yields[2] == pytest.approx(1.0, rel=1e-12)


def test_initial_in_terminal_state():
    """Starting in a level with no decays leaves population intact."""
    levels = [CascadeLevel('a'), CascadeLevel('b')]
    transitions = []
    res = cascade_yields(levels, transitions, initial_idx=0)
    assert res.terminal_yields[0] == pytest.approx(1.0)
    assert fluorescence_yield(res) == 0.0


def test_invalid_initial_idx():
    levels = [CascadeLevel('a'), CascadeLevel('b')]
    with pytest.raises(ValueError):
        cascade_yields(levels, [], initial_idx=5)


def test_self_loop_transition_rejected():
    """A transition i → i is not allowed."""
    levels = [CascadeLevel('a')]
    transitions = [CascadeTransition(upper=0, lower=0, rate_per_s=1.0)]
    with pytest.raises(ValueError):
        cascade_yields(levels, transitions, initial_idx=0)


def test_zero_rate_transition_ignored():
    """Zero-rate transitions don't contribute."""
    levels = [CascadeLevel('a'), CascadeLevel('b')]
    transitions = [
        CascadeTransition(upper=0, lower=1, rate_per_s=0.0, photon_eV=100.0),
    ]
    res = cascade_yields(levels, transitions, initial_idx=0)
    # Without any decay path, population stays in 0
    assert res.terminal_yields[0] == pytest.approx(1.0)


def test_branching_ratio_preserved_through_chain():
    """Multi-step cascade preserves probability."""
    levels = [CascadeLevel(str(i)) for i in range(4)]
    transitions = [
        CascadeTransition(0, 1, 0.6e10, 100.0),
        CascadeTransition(0, 2, 0.4e10, 0.0),     # Auger
        CascadeTransition(1, 3, 1.0e10, 50.0),
        CascadeTransition(2, 3, 1.0e10, 30.0),
    ]
    res = cascade_yields(levels, transitions, initial_idx=0)
    total = sum(res.terminal_yields.values())
    assert total == pytest.approx(1.0, rel=1e-10)
    assert 3 in res.terminal_yields
    assert res.terminal_yields[3] == pytest.approx(1.0, rel=1e-10)
