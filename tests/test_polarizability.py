"""Smoke tests for dipole polarisability."""

from __future__ import annotations

import numpy as np
import pytest

from atomx import Atom
from atomx.properties.polarizability import (
    PolarizabilityResult,
    _bbr_shift_from_alpha,
    dipole_polarizability,
)


def test_helium_polarizability_reasonable_magnitude():
    """He α(0) at the HF level lies in the 0.7-1.5 a₀³ window for any
    reasonable B-spline basis.  Exact Helium α(0) ≈ 1.383 a₀³.
    """
    atom = Atom(Z=2, config="1s2", numerics="bspline", breit=False, qed=False)
    atom.solve()
    res = atom._result

    r = dipole_polarizability(res, kappa_max=2, n_virt_max=15)
    assert isinstance(r, PolarizabilityResult)
    assert 0.5 < r.alpha_static < 1.6
    assert r.per_orbital["1s1/2"] == pytest.approx(r.alpha_static, rel=1e-12)


def test_polarizability_grows_with_basis():
    atom = Atom(Z=2, config="1s2", numerics="bspline", breit=False, qed=False)
    atom.solve()
    res = atom._result

    a1 = dipole_polarizability(res, kappa_max=1, n_virt_max=10).alpha_static
    a2 = dipole_polarizability(res, kappa_max=3, n_virt_max=30).alpha_static
    # Larger partial-wave basis → α(0) closer to the limit (always
    # increases for length gauge with V^{N-1} virtuals).
    assert a2 >= a1


def test_dynamic_polarizability_below_resonance():
    """α(ω) is positive and increasing for ω below the lowest
    transition (1s→2p for He, ~0.78 hartree at HF level).
    """
    atom = Atom(Z=2, config="1s2", numerics="bspline", breit=False, qed=False)
    atom.solve()
    res = atom._result

    omega = np.array([0.0, 0.05, 0.1])
    r = dipole_polarizability(res, omega=omega, kappa_max=2, n_virt_max=15)
    assert r.alpha_dynamic is not None
    a = r.alpha_dynamic
    assert np.all(a > 0)
    assert a[0] < a[1] < a[2]


def test_bbr_dynamic_constant_alpha_matches_static_limit():
    T_K = 300.0
    alpha0 = 1.25
    from atomx.constants import HARTREE_EV
    kT_au = (T_K / 11604.5) / HARTREE_EV
    x = np.linspace(1e-4, 60.0, 2000)
    omega = x * kT_au
    shift = _bbr_shift_from_alpha(omega, np.full_like(omega, alpha0), T_K)
    E_field_au = 8.319 / 5.14220674763e9
    expected = -0.5 * E_field_au * E_field_au * alpha0
    assert shift == pytest.approx(expected, rel=1e-12)
