"""Born / Coulomb-Born high-energy asymptote regression tests."""

import numpy as np
import pytest

from atomx.scattering.distorted_wave.born import (
    born_excitation_dipole,
    born_excitation_forbidden,
)
from atomx.scattering.distorted_wave.solver import dw_excitation_cross_section
from atomx.scattering.electron_ionization.coulomb_born import (
    bethe_born_ci_cross_section,
)

# ── Coulomb-Born / Bethe-Born CI ────────────────────────────────


def test_bethe_born_ci_zero_below_threshold():
    sigma = bethe_born_ci_cross_section(np.array([5.0, 13.0]), M_sq=0.3, I_eV=13.6)
    assert np.all(sigma == 0)


def test_bethe_born_ci_high_e_asymptote():
    """σ(E) ∝ ln(E)/E at high E (5 % over 4 decades of energy)."""
    E = np.array([1.0e4, 1.0e6, 1.0e8])
    sigma = bethe_born_ci_cross_section(E, M_sq=0.3, I_eV=13.6)
    # σ × E / [ln(2E/I) − 1] should be ~constant at high E
    rel = sigma * E / (np.log(2 * E / 13.6) - 1.0)
    assert (rel.max() - rel.min()) / rel.mean() < 0.05


def test_bethe_born_ci_validates_inputs():
    with pytest.raises(ValueError):
        bethe_born_ci_cross_section(np.array([100.0]), M_sq=-1, I_eV=13.6)
    with pytest.raises(ValueError):
        bethe_born_ci_cross_section(np.array([100.0]), M_sq=0.3, I_eV=0)


def test_bethe_born_ci_h_oom():
    """H 1s ionisation σ at 100 eV is order-of-magnitude correct (~0.5 Mb)."""
    sigma = bethe_born_ci_cross_section(np.array([100.0]), M_sq=0.283, I_eV=13.6)
    sigma_Mb = sigma[0] * 1e16
    # NIST H total e-ionisation σ(100 eV) ≈ 0.65 Mb; Bethe-Born is 30-50 % of that
    assert 0.1 < sigma_Mb < 1.0


# ── Born excitation (dipole) ────────────────────────────────────


def test_born_dipole_zero_below_threshold():
    sigma = born_excitation_dipole(np.array([5.0]), f_ab=0.4, delta_E_eV=10.0)
    assert sigma[0] == 0


def test_born_dipole_validates():
    with pytest.raises(ValueError):
        born_excitation_dipole(np.array([100.0]), f_ab=-0.1, delta_E_eV=10.0)
    with pytest.raises(ValueError):
        born_excitation_dipole(np.array([100.0]), f_ab=0.4, delta_E_eV=0)


def test_born_dipole_log_e_over_e():
    """σ ∝ ln(E/ΔE)/E at high E."""
    E = np.array([100.0, 1000.0, 10000.0])
    sigma = born_excitation_dipole(E, f_ab=0.4, delta_E_eV=10.0)
    rel = sigma * E / np.log(E / 10.0)
    assert (rel.max() - rel.min()) / rel.mean() < 1e-9


def test_born_dipole_f_proportionality():
    """σ scales linearly with the oscillator strength f."""
    E = np.array([1000.0])
    s1 = born_excitation_dipole(E, f_ab=0.1, delta_E_eV=10.0)
    s2 = born_excitation_dipole(E, f_ab=0.2, delta_E_eV=10.0)
    assert s2[0] == pytest.approx(2.0 * s1[0], rel=1e-12)


# ── Born excitation (forbidden L>1) ─────────────────────────────


def test_born_forbidden_e2_falls_as_inv_e():
    """Forbidden L=2 σ ∝ 1/E (no log term)."""
    E = np.array([100.0, 1000.0, 10000.0])
    sigma = born_excitation_forbidden(E, line_strength_au=1.0, delta_E_eV=10.0, L=2)
    rel = sigma * E
    assert (rel.max() - rel.min()) / rel.mean() < 1e-9


def test_born_forbidden_zero_below_threshold():
    sigma = born_excitation_forbidden(np.array([5.0]), 1.0, 10.0)
    assert sigma[0] == 0


def test_born_forbidden_validates():
    with pytest.raises(ValueError):
        born_excitation_forbidden(np.array([100.0]), -0.1, 10.0)
    with pytest.raises(ValueError):
        born_excitation_forbidden(np.array([100.0]), 1.0, 0)
    with pytest.raises(ValueError):
        born_excitation_forbidden(np.array([100.0]), 1.0, 10.0, L=0)


def test_dw_excitation_validates_l_max_before_solve():
    with pytest.raises(ValueError, match="L_max"):
        dw_excitation_cross_section(None, None, np.array([100.0]), L_max=-1)
