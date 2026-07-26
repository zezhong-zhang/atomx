"""High-n DR density suppression via continuum lowering (review M8)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx.scattering.autoionization.auger import AutoionizationState  # noqa: E402
from atomx.scattering.dielectronic_recombination.solver import (  # noqa: E402
    dr_rate_coefficient)


def _temperature_for_screening_u(Z_ion, ne_cm3, u):
    from atomx.constants import BOHR_RADIUS_M, HARTREE_EV
    from atomx.plasma.screening import ion_sphere_radius

    ne_au = ne_cm3 * (BOHR_RADIUS_M * 100) ** 3
    lam_D = ion_sphere_radius(Z_ion, ne_cm3) / u
    return lam_D ** 2 * 4.0 * np.pi * ne_au * HARTREE_EV


def _states():
    # two resonances converging to a 800-eV series limit: a low-n one
    # bound by 500 eV and a high-n Rydberg one bound by only 5 eV.
    mk = lambda E, lab: AutoionizationState(
        label=lab, E_resonance_eV=E, J=1.5,
        gamma_auger_per_s=1e14, a_radiative_per_s=1e13)
    return [mk(300.0, "low-n"), mk(795.0, "high-n")]


def test_zero_density_unchanged():
    T = [50.0, 200.0]
    a0 = dr_rate_coefficient(_states(), T, initial_state_g=1.0)
    a1 = dr_rate_coefficient(_states(), T, initial_state_g=1.0,
                             ne_cm3=None)
    assert np.allclose(a0.alpha_cm3_s, a1.alpha_cm3_s, rtol=0, atol=0)


def test_high_density_drops_high_n_only():
    T = [200.0]
    lo = dr_rate_coefficient(_states(), T, initial_state_g=1.0,
                             ne_cm3=1.0e18, Z_ion=16,
                             series_limit_eV=800.0)
    hi = dr_rate_coefficient(_states(), T, initial_state_g=1.0,
                             ne_cm3=5.0e23, Z_ion=16,
                             series_limit_eV=800.0)
    # low density: both survive (IPD << 5 eV)
    assert lo.per_resonance["high-n"][0] > 0.0
    assert lo.per_resonance["low-n"][0] > 0.0
    # high density: the 5-eV-bound Rydberg resonance dissolves,
    # the 500-eV-bound one survives
    assert hi.per_resonance["high-n"][0] == 0.0
    assert hi.per_resonance["low-n"][0] > 0.0
    assert hi.alpha_cm3_s[0] < lo.alpha_cm3_s[0]


def test_suppression_requires_limit():
    with pytest.raises(ValueError, match="series_limit_eV"):
        dr_rate_coefficient(_states(), [100.0], ne_cm3=1e20)


def test_stewart_pyatt_boundary_does_not_toggle_dr_survival():
    from atomx.plasma.screening import ionization_potential_depression

    Z_ion = 1
    ne_cm3 = 1.0e23
    temps = [
        _temperature_for_screening_u(Z_ion, ne_cm3, 9.99),
        _temperature_for_screening_u(Z_ion, ne_cm3, 10.0),
    ]
    ipd = np.array([
        ionization_potential_depression(Z_ion, ne_cm3, t) for t in temps
    ])
    binding_eV = 2.0 * float(ipd.max())
    series_limit_eV = binding_eV + 0.1
    state = AutoionizationState(
        label="boundary-bound", E_resonance_eV=series_limit_eV - binding_eV,
        J=1.5, gamma_auger_per_s=1e14, a_radiative_per_s=1e13)

    rate = dr_rate_coefficient(
        [state], temps, initial_state_g=1.0, ne_cm3=ne_cm3,
        Z_ion=Z_ion, series_limit_eV=series_limit_eV)

    assert rate.per_resonance["boundary-bound"][0] > 0.0
    assert rate.per_resonance["boundary-bound"][1] > 0.0
