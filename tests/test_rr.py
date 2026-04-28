"""Smoke tests for radiative recombination via Milne detailed balance."""

from __future__ import annotations

import numpy as np

from atomx import Atom
from atomx.scattering.radiative_recombination import (
    RRResult,
    rr_cross_section,
    rr_rate_coefficient,
)


def test_he_plus_rr_cross_section_falls_with_energy():
    """σ_RR for hydrogenic He⁺(1s) capture decreases monotonically
    with electron energy at low E (Sommerfeld limit ~ E^{-1})."""
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    res = atom._result

    E = np.geomspace(0.1, 100.0, 25)
    r = rr_cross_section(res, E, g_initial=1)
    assert isinstance(r, RRResult)
    assert np.all(r.sigma_total_a0_sq > 0)
    # Monotonic decrease over the chosen grid
    assert np.all(np.diff(r.sigma_total_a0_sq) < 0)


def test_rr_default_degeneracy_matches_closed_core():
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    E = np.geomspace(0.1, 20.0, 8)
    r_default = rr_cross_section(atom._result, E)
    r_explicit = rr_cross_section(atom._result, E, g_initial=1)
    np.testing.assert_allclose(
        r_default.sigma_total_a0_sq,
        r_explicit.sigma_total_a0_sq,
        rtol=0.0,
        atol=0.0,
    )


def test_rr_rate_coefficient_he_plus_order_of_magnitude():
    """α_RR(He²⁺ → He⁺) at T=1 eV should be in the 1e-14 to 1e-12 cm³/s
    window — capture into 1s only (full hydrogenic with Rydberg sum
    is ~5.4e-13 cm³/s for Z=2)."""
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    res = atom._result

    E = np.geomspace(0.05, 200.0, 50)
    r = rr_cross_section(res, E, g_initial=1)
    rate = rr_rate_coefficient(r, np.array([1.0]))
    assert 1e-14 < rate < 1e-12


def test_rr_against_invalid_input():
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    res = atom._result
    # Negative or zero energies should raise
    import pytest
    with pytest.raises(ValueError, match="must all be > 0"):
        rr_cross_section(res, np.array([-1.0, 1.0]))
