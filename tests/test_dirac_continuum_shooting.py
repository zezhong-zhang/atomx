"""Direct ODE Dirac-continuum integrator regression tests."""

import math

import numpy as np
import pytest

from atomx.numerics.shooting.dirac_continuum import (
    solve_dirac_continuum_shooting,
)


def _V_hydrogen(r):
    return -1.0 / np.where(r > 0, r, 1e-30)


def _V_zero(r):
    return np.zeros_like(r)


def test_H_continuum_envelope_close_to_norm():
    """Asymptotic envelope of P matches √(2/(πk)) within 5 %."""
    E = 1.0
    r, P, Q, delta = solve_dirac_continuum_shooting(
        E_au=E, kappa=-2, V_func=_V_hydrogen, r_max=40.0, n_grid=4000,
    )
    k = math.sqrt(2 * E)
    expected = math.sqrt(2.0 / (math.pi * k))
    # WKB amplitude from the outer 25 % of the grid
    P_outer = P[-len(r) // 4:]
    A_meas = math.sqrt(2.0 * float(np.mean(P_outer ** 2)))
    assert A_meas == pytest.approx(expected, rel=5e-2)


def test_high_energy_coulomb_envelope_converges_with_grid_refinement():
    """Refinement removes high-energy spatial aliasing.

    A Coulomb continuum has a known delta(E)-normalised asymptotic
    amplitude.  The error must decrease as the Salvat mesh resolves more
    points per wavelength; this catches a return to box pseudostates or an
    under-sampled exact-energy shooting wave.
    """
    E = 50.0
    k = math.sqrt(2.0 * E)
    expected = math.sqrt(2.0 / (math.pi * k))
    relative_errors = []

    for n_grid in (400, 800, 1600, 3200):
        r, P, _, _ = solve_dirac_continuum_shooting(
            E_au=E,
            kappa=-2,
            V_func=_V_hydrogen,
            r_max=40.0,
            n_grid=n_grid,
            Z_residual=1.0,
        )
        outer = P[r > 30.0]
        measured = math.sqrt(2.0 * float(np.mean(outer ** 2)))
        relative_errors.append(abs(measured / expected - 1.0))

    assert np.all(np.diff(relative_errors) < 0.0), relative_errors
    assert relative_errors[-1] < 2.0e-2, relative_errors


def test_phase_shift_finite():
    r, P, Q, delta = solve_dirac_continuum_shooting(
        E_au=2.0, kappa=-1, V_func=_V_hydrogen, r_max=30.0, n_grid=3000,
    )
    assert math.isfinite(delta)
    assert -math.pi <= delta <= math.pi


def test_returns_consistent_shapes():
    r, P, Q, delta = solve_dirac_continuum_shooting(
        E_au=1.0, kappa=-1, V_func=_V_zero, r_max=20.0, n_grid=2000,
    )
    assert r.shape == P.shape == Q.shape
    assert len(r) == 2000


def test_invalid_inputs():
    with pytest.raises(ValueError):
        solve_dirac_continuum_shooting(E_au=-1.0, kappa=-1, V_func=_V_zero)
    with pytest.raises(ValueError):
        solve_dirac_continuum_shooting(E_au=1.0, kappa=0, V_func=_V_zero)


def test_relativistic_small_component_suppressed():
    """Q/P ratio at moderate r should be ~Zα for hydrogen ε p
    (Q is the small component)."""
    r, P, Q, _ = solve_dirac_continuum_shooting(
        E_au=1.0, kappa=-2, V_func=_V_hydrogen, r_max=40.0, n_grid=4000,
    )
    # Skip first 100 points (small r where P,Q are still small)
    # and check the typical magnitude ratio at r > 10 a₀
    mask = r > 10.0
    P_outer = P[mask]
    Q_outer = Q[mask]
    P_max = float(np.max(np.abs(P_outer)))
    Q_max = float(np.max(np.abs(Q_outer)))
    # Q should be suppressed by ~α relative to P in the non-relativistic limit
    assert Q_max < 0.1 * P_max
