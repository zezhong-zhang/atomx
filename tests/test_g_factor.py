"""Smoke tests for the bound-electron g-factor."""

from __future__ import annotations

import math

import pytest

from atomx import Atom
from atomx.prop.g_factor import (
    GFactorResult,
    g_factor,
    g_factor_dirac_hydrogenic,
    g_factor_orbital,
    g_factor_qed,
    g_factor_finite_nuclear_size,
)


def test_hydrogen_dirac_breit_value():
    """Hydrogenic 1s_{1/2} Breit formula at Z=1: g = (2/3)(1+2√(1-α²))."""
    g = g_factor_dirac_hydrogenic(1, 1, -1)
    from atomx.constants import ALPHA
    breit = (2.0 / 3.0) * (1.0 + 2.0 * math.sqrt(1.0 - ALPHA ** 2))
    assert abs(g - breit) < 1e-12


def test_hydrogen_total_g_close_to_experiment():
    """Hydrogen bound g = 2.002 283 845 at the leading order."""
    g_d = g_factor_dirac_hydrogenic(1, 1, -1)
    g_q = g_factor_qed(1, -1)
    # Experimental total ≈ 2.002 283 845
    assert abs((g_d + g_q) - 2.0022838) < 5e-6


def test_he_plus_scf_matches_hydrogenic():
    """SCF orbital g-factor for hydrogen-like He⁺ matches the
    closed-form Breit-Margolis value."""
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    g_orb = g_factor_orbital(atom._result, "1s1/2")
    g_hyd = g_factor_dirac_hydrogenic(2, 1, -1)
    assert abs(g_orb - g_hyd) < 1e-8


def test_g_factor_top_level():
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    r = g_factor(atom._result, "1s1/2", R_rms_fm=1.6, include_fns=True)
    assert isinstance(r, GFactorResult)
    assert r.label == "1s1/2"
    assert 1.99 < r.g_total < 2.01
    assert r.g_qed > 0
    assert r.g_fns < 0  # FNS lowers the bound g-factor


def test_fns_zero_for_p_states():
    assert g_factor_finite_nuclear_size(50, kappa=+1, R_rms_fm=4.0) == 0.0
    assert g_factor_finite_nuclear_size(50, kappa=-2, R_rms_fm=4.0) == 0.0


def test_g_factor_unknown_label_raises():
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    with pytest.raises(KeyError):
        g_factor(atom._result, "99x9/2", R_rms_fm=1.6)


def test_g_factor_fns_requires_radius():
    atom = Atom(Z=2, config="1s1", breit=False, qed=False)
    atom.solve()
    with pytest.raises(ValueError, match="R_rms_fm"):
        g_factor(atom._result, "1s1/2", include_fns=True)
