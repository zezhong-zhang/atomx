"""High-n DR density suppression via continuum lowering (review M8)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx.scattering.autoionization.auger import AutoionizationState  # noqa: E402
from atomx.scattering.dielectronic_recombination.solver import (  # noqa: E402
    dr_rate_coefficient,
)


def _states():
    # two resonances converging to a 800-eV series limit: a low-n one
    # bound by 500 eV and a high-n Rydberg one bound by only 5 eV.
    def mk(E, lab):
        return AutoionizationState(
            label=lab, E_resonance_eV=E, J=1.5,
            gamma_auger_per_s=1e14, a_radiative_per_s=1e13)
    return [mk(300.0, "low-n"), mk(795.0, "high-n")]


def _saha_capture_prefactor_cm3(temperature_eV):
    """(h^2 / (2 pi m_e kT))^{3/2} in cm^3, CODATA 2022, written out here."""
    planck_j_s = 6.626_070_15e-34
    m_e_kg = 9.109_383_7139e-31
    ev_j = 1.602_176_634e-19
    return ((planck_j_s ** 2
             / (2.0 * math.pi * m_e_kg * temperature_eV * ev_j)) ** 1.5
            * 1.0e6)


def test_zero_density_matches_the_burgess_closed_form():
    """At zero density the rate must be the Burgess formula, by hand.

    Replaces the old ``test_zero_density_unchanged``, which compared two calls
    that differed only by ``ne_cm3=None`` (a literal no-op).  The expected
    value is the detailed-balance (Saha) capture coefficient times the
    branching ratio, both written out in the test:

        alpha_DR = (h^2/(2 pi m_e kT))^{3/2} (g_d/2g_i)
                   Gamma_a Gamma_r/(Gamma_a + Gamma_r) exp(-E_d/kT),

    with g_d = 2J+1 = 4 for the J = 3/2 fixture resonances.  Both spellings of
    "no density" (the default and an explicit ``ne_cm3 = 0.0``) must give it.
    """
    states = _states()
    for temperature_eV in (50.0, 200.0):
        prefactor = _saha_capture_prefactor_cm3(temperature_eV)
        expected = sum(
            prefactor * (s.g / 2.0)
            * (s.gamma_auger_per_s * s.a_radiative_per_s / s.total_width_per_s)
            * math.exp(-s.E_resonance_eV / temperature_eV)
            for s in states
        )
        for ne_cm3 in (None, 0.0):
            got = float(dr_rate_coefficient(
                states, [temperature_eV], initial_state_g=1.0,
                ne_cm3=ne_cm3).alpha_cm3_s[0])
            assert got == pytest.approx(expected, rel=1e-12), (
                f"T = {temperature_eV} eV, ne_cm3 = {ne_cm3}: got {got:.6e}, "
                f"Burgess closed form {expected:.6e}"
            )


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


def test_density_suppression_rejects_nonfinite_ipd(monkeypatch):
    import atomx.plasma.screening as screening

    monkeypatch.setattr(
        screening, "ionization_potential_depression", lambda *args, **kw: np.nan)

    with pytest.raises(ValueError, match="non-finite IPD"):
        dr_rate_coefficient(
            _states(),
            [100.0],
            ne_cm3=1e20,
            Z_ion=16,
            series_limit_eV=800.0,
        )


def test_density_suppression_rejects_nonfinite_binding_energy():
    state = AutoionizationState(
        label="bad-energy",
        E_resonance_eV=np.nan,
        J=1.5,
        gamma_auger_per_s=1e14,
        a_radiative_per_s=1e13,
    )

    with pytest.raises(ValueError, match="binding energy"):
        dr_rate_coefficient(
            [state],
            [100.0],
            ne_cm3=1e20,
            Z_ion=16,
            series_limit_eV=800.0,
        )
