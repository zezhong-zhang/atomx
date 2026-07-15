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
