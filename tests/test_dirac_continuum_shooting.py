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


class TestCoulombPhaseShift:
    """``delta`` must be a phase shift, not a function of the box radius.

    For a pure Coulomb potential ``-Z/r`` the integrated solution *is* the
    regular Coulomb function, so the phase extracted against

        ``P(r) ~ A sin(k r - l pi/2 - eta ln(2 k r) + sigma_l + delta)``

    must be zero once the matching radius is asymptotic (``k r >> eta^2``).
    The fit previously used a plane-wave asymptotic, dropping both the
    ``-eta ln(2 k r)`` term and the Coulomb phase ``sigma_l``; for
    ``Z_res = 1, k = 0.271, r_max = 50`` the log term alone is 12.2 rad, so
    the returned "phase shift" was dominated by the box radius.  See
    ``docs/PHYSICS_NUMERICS_REVIEW_2026-09-19.md`` finding M15.
    """

    @pytest.mark.parametrize(
        "Z, k, r_max",
        [(1.0, 2.0, 200.0), (1.0, 5.0, 200.0), (1.0, 10.0, 100.0),
         (10.0, 10.0, 200.0)],
    )
    def test_pure_coulomb_phase_is_zero(self, Z, k, r_max):

        from atomx.numerics.shooting.dirac_continuum import (
            solve_dirac_continuum_shooting,
        )

        E = 0.5 * k * k
        _r, _P, _Q, delta = solve_dirac_continuum_shooting(
            E, -1, lambda r: -Z / r, r_max=float(r_max), n_grid=8000,
            Z_residual=Z,
        )
        assert abs(delta) < 0.05, (
            f"Z={Z}, k={k}: pure-Coulomb delta = {delta:+.4f} rad; the "
            f"Coulomb log/phase terms are missing from the asymptotic fit"
        )

    def test_delta_no_longer_scales_with_the_box(self):
        """The same physical state matched at two radii must agree.

        With the plane-wave fit the two answers differ by
        ``eta ln(r2/r1)`` — 2.8 rad for Z=1, k=0.5, r: 100 -> 1000.
        """
        from atomx.numerics.shooting.dirac_continuum import (
            solve_dirac_continuum_shooting,
        )

        Z, k = 1.0, 0.5
        E = 0.5 * k * k
        deltas = []
        for r_max in (200.0, 800.0):
            _r, _P, _Q, d = solve_dirac_continuum_shooting(
                E, -1, lambda r: -Z / r, r_max=r_max, n_grid=8000,
                Z_residual=Z,
            )
            deltas.append(d)
        # eta ln(2 k r) changes by eta*ln(4) = -2.77 rad between these
        # radii, so a plane-wave fit would show a difference of that order.
        assert abs(deltas[0] - deltas[1]) < 0.15, deltas
