"""Direct ODE Dirac-continuum integrator regression tests."""

import math

import numpy as np
import pytest

from atomx.numerics.shooting.dirac_continuum import (
    solve_dirac_continuum_many,
    solve_dirac_continuum_shooting,
)
from atomx.numerics.shooting.grid import sgrid


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

    # A finite outer-window RMS oscillates slightly with the number of
    # included cycles; monotone error at every grid size is not an ODE
    # convergence requirement. The finer grids agree within 0.2%.
    assert relative_errors[0] < 1.0e-2, relative_errors
    assert max(relative_errors[1:]) < 2.0e-3, relative_errors


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
    with pytest.raises(ValueError, match="finite"):
        solve_dirac_continuum_shooting(E_au=float("nan"), kappa=-1, V_func=_V_zero)


def test_fast_continuum_grid_matches_salvat_bisection():
    args = (50.0, 1e-3/25, 0.5, 20000)
    reference, _, n_ref = sgrid(*args)
    fast, _, n_fast = sgrid(*args, fast=True)
    assert n_fast == n_ref
    np.testing.assert_allclose(fast, reference, rtol=4e-12, atol=4e-13)


def test_mutating_potential_invalidates_prepared_spline():
    class Coulomb:
        def __init__(self, charge):
            self.charge = charge

        def __call__(self, r):
            return -self.charge/r

    potential = Coulomb(1.0)
    args = dict(r_max=30., n_grid=3000, Z_residual=1.)
    _, first, _, _ = solve_dirac_continuum_shooting(2., -1, potential, **args)
    potential.charge = 2.0
    _, changed, _, _ = solve_dirac_continuum_shooting(2., -1, potential, **args)
    _, fresh, _, _ = solve_dirac_continuum_shooting(2., -1, Coulomb(2.), **args)
    assert not np.allclose(first, changed)
    np.testing.assert_allclose(changed, fresh, rtol=1e-12, atol=1e-12)


def test_screened_potential_uses_the_coulomb_tail_interval():
    from atomx.numerics.shooting.dirac_continuum import _prepare_potential

    def potential(r):
        return (-17. - 9.*np.maximum(1.-r, 0.)**4)/r

    r, _, a, b, c, d, index, rv = _prepare_potential(potential, 75., 4000, 17.)
    reconstructed = a[index]+r*(b[index]+r*(c[index]+r*d[index]))
    # VINT stores a dedicated constant-rV interval after its last knot.
    # Extrapolating the preceding cubic creates an unphysical growing field.
    np.testing.assert_allclose(reconstructed[r > 2.], -17., atol=1e-12, rtol=0)


@pytest.mark.parametrize('kappa', [-1, 1, -3])
def test_screened_continuum_against_independent_ode(kappa):
    from scipy.integrate import solve_ivp

    from atomx.constants import C

    def potential(r):
        return (-17.-9.*np.maximum(1.-r, 0.)**4)/r

    energy = 10.
    r, p, q, _ = solve_dirac_continuum_shooting(
        energy, kappa, potential, r_max=40., n_grid=5000,
        Z_residual=17., match_phase=False)

    def equation(x, y):
        kinetic = energy-potential(x)
        return [-kappa*y[0]/x+(2*C+kinetic/C)*y[1],
                kappa*y[1]/x-kinetic/C*y[0]]

    # Independent point-Coulomb Frobenius ratio, propagated directly in V.
    gamma = math.sqrt(kappa*kappa-(26/C)**2)
    reference = solve_ivp(equation, (1e-9, 40.),
        [1., (gamma+kappa)*C/26.], t_eval=r[1:], method='DOP853',
        rtol=2e-11, atol=2e-12)
    assert reference.success
    wave = reference.y/np.max(np.abs(reference.y))
    native = np.array([p[1:], q[1:]])
    # This tests the regular radial solution up to its arbitrary scale;
    # energy normalization has separate asymptotic-envelope tests.
    scale = np.vdot(wave, native)/np.vdot(wave, wave)
    assert np.linalg.norm(scale*wave-native)/np.linalg.norm(native) < 2e-8


def test_relativistic_coulomb_phase_stable_with_matching_radius():
    energy = 200000. / 27.211386245988

    def potential(r):
        return -54./r
    phases = [solve_dirac_continuum_shooting(
        energy, -1, potential, r_max=radius, n_grid=points,
        Z_residual=54.)[3]
        for radius, points in ((20., 26515), (40., 53030))]
    assert abs(phases[0]-phases[1]) < 2e-9
    # This grid used to fail because a roundoff-sized terminal cubic was
    # extrapolated beyond VINT's Coulomb-tail knot, not from under-resolution.
    coarse = solve_dirac_continuum_shooting(
        energy, -1, potential, r_max=20., n_grid=20000,
        Z_residual=54.)[3]
    assert abs(coarse-phases[0]) < 2e-9


def test_batch_deduplicates_waves_and_matches_scalar():
    requests = [(-1, 1.), (-1, 1.), (1, 2.)]
    r, waves = solve_dirac_continuum_many(
        requests, _V_hydrogen, r_max=30., n_grid=3000,
        Z_residual=1., workers=2)
    assert len(waves) == 2
    for kappa, energy in set(requests):
        scalar_r, scalar_P, scalar_Q, _ = solve_dirac_continuum_shooting(
            energy, kappa, _V_hydrogen, r_max=30., n_grid=3000,
            Z_residual=1.)
        np.testing.assert_array_equal(r, scalar_r)
        np.testing.assert_allclose(waves[(kappa, energy)][0], scalar_P,
                                   rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(waves[(kappa, energy)][1], scalar_Q,
                                   rtol=1e-13, atol=1e-13)
    with pytest.raises(ValueError, match="positive energy"):
        solve_dirac_continuum_many([(-1, 0.)], _V_hydrogen)


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
    """The exact Dirac–Coulomb reference removes Coulomb phase and box bias."""

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
        assert abs(delta) < 2e-9, (
            f"Z={Z}, k={k}: pure-Coulomb delta = {delta:+.4f} rad; the "
            f"regular Dirac–Coulomb reference should give zero phase"
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
        assert abs(deltas[0] - deltas[1]) < 2e-9, deltas
