"""Finite-polynomial differences preserve the declared CI functional."""

import numpy as np
import pytest

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.dirac_assembly import solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.physics.csf.state_average import CIStateAverage


def problem(ns, configs=("1s1 2s1",)):
    data = prepare_bspline_data(make_knots(ns, 9, .005, 30), ns, 9)
    csfs = [csf for config in configs for csf in enumerate_jj_csfs(config)]
    functional = CSFOrbitalFunctional(csfs, data, 4, state_average=CIStateAverage())
    eigen = solve_dirac(functional.one_body[-1], functional.metric[-1], ns, -1)
    seed = eigen.coefficients[:, :len(functional.keys)]
    waves = {key: (seed[:ns, i].copy(), seed[ns:, i].copy())
             for i, key in enumerate(functional.keys)}
    return functional, waves, eigen.coefficients


@pytest.mark.parametrize("ns", [35, 120])
def test_complete_quartic_change_matches_rediagonalized_CI(ns):
    functional, waves, spectrum = problem(ns)
    # A positive-energy, same-kappa orthonormal trial with both orbitals changed.
    old = np.column_stack([np.concatenate(waves[key]) for key in functional.keys])
    trial = old @ np.array([[np.cos(.08), -np.sin(.08)],
                           [np.sin(.08), np.cos(.08)]])
    trial = trial * np.cos(.03) + spectrum[:, 2:4] * np.sin(.03)
    newer = {key: (trial[:ns, i], trial[ns:, i])
             for i, key in enumerate(functional.keys)}
    energy, _, ci = functional.evaluate(waves)
    updated, _, other = functional.evaluate(newer)
    delta = functional.fixed_density_energy_difference(waves, newer, ci, other)
    assert delta == pytest.approx(updated - energy, abs=3e-11)
    # Full finite polynomial, not a truncated first/second-order correction.
    reverse = functional.fixed_density_energy_difference(newer, waves, other, ci)
    assert reverse == pytest.approx(-delta, abs=1e-13)


def test_unresolved_large_constant_does_not_hide_small_physical_change():
    functional, waves, spectrum = problem(35)
    _, gradient, ci = functional.evaluate(waves)
    key = (1, -1)
    direction = spectrum[:, 3]
    derivative = float(gradient[key] @ direction)
    step = 1e-8
    old = np.concatenate(waves[key])
    y = (old + step * direction) / np.sqrt(1 + step**2)
    newer = dict(waves)
    newer[key] = y[:35], y[35:]
    _, other = functional.evaluate(newer, gradient=False)
    delta = functional.fixed_density_energy_difference(waves, newer, ci, other)
    assert delta == pytest.approx(step * derivative, rel=3e-5, abs=2e-16)
    assert delta != 0
    # The same density/space can include a constant spectator-core offset;
    # its subtraction in double precision would completely lose this step.
    assert float(1e12 + delta) - 1e12 == 0


def test_CI_mixing_response_is_not_replaced_by_a_frozen_density():
    functional, waves, spectrum = problem(35, ("1s2", "2s2"))
    _, _, ci = functional.evaluate(waves)
    newer = dict(waves)
    vector = .9 * spectrum[:, 0] + np.sqrt(1 - .9**2) * spectrum[:, 2]
    newer[1, -1] = vector[:35], vector[35:]
    _, trial = functional.evaluate(newer, gradient=False)
    assert functional.fixed_density_energy_difference(waves, newer, ci, trial) is None


def test_constraint_centering_removes_only_roundoff_normal_extension():
    functional, waves, _ = problem(35)
    _, _, ci = functional.evaluate(waves)
    newer = dict(waves)
    newer[1, -1] = tuple(part * (1 + 6e-15) for part in waves[1, -1])
    _, trial = functional.evaluate(newer, gradient=False)
    complete = functional.fixed_density_energy_difference(waves, newer, ci, trial)
    centered = functional.constrained_energy_difference(waves, newer, ci, trial)
    assert abs(complete) > 1e-15
    # Scaling is a pure normal change in exact arithmetic. Storing its
    # products also introduces tiny tangent rounding, so demanding zero
    # physical change would test the wrong state. Remove the dominant normal
    # contribution; retain that rounded state's residual tangent change.
    assert abs(centered) < abs(complete) * 1e-3
    changed = dict(waves)
    changed[1, -1] = tuple(part * (1 + 1e-8) for part in waves[1, -1])
    _, bad = functional.evaluate(changed, gradient=False)
    assert functional.constrained_energy_difference(waves, changed, ci, bad) is None


@pytest.mark.parametrize("ns", [35, 120])
def test_constraint_centering_retains_true_polar_step_curvature(ns):
    functional, waves, spectrum = problem(ns)
    newer = dict(waves)
    old = np.concatenate(waves[1, -1])
    angle = .08
    new = np.cos(angle) * old + np.sin(angle) * spectrum[:, 3]
    newer[1, -1] = new[:ns], new[ns:]
    before, _, ci = functional.evaluate(waves)
    after, other = functional.evaluate(newer, gradient=False)
    centered = functional.constrained_energy_difference(waves, newer, ci, other)
    assert centered == pytest.approx(after - before, abs=3e-11)
    assert abs(centered) > 1e-3


def test_explicit_linear_map_increment_has_exact_zero_and_tangent_limit():
    functional, waves, spectrum = problem(35)
    _, gradient, ci = functional.evaluate(waves)
    zeros = {key: tuple(np.zeros_like(part) for part in waves[key])
             for key in functional.keys}
    assert functional.constrained_energy_difference(
        waves, waves, ci, ci, increments=zeros) == 0
    key = (1, -1)
    angle = 1e-9
    old = np.concatenate(waves[key])
    # Complete polar rotation; form its exact finite coordinate increment
    # without subtracting a second reconstructed orbital from the reference.
    change = (np.cos(angle) - 1) * old + np.sin(angle) * spectrum[:, 3]
    increments = dict(zeros)
    increments[key] = change[:35], change[35:]
    newer = dict(waves)
    value = old + change
    newer[key] = value[:35], value[35:]
    _, other = functional.evaluate(newer, gradient=False)
    delta = functional.constrained_energy_difference(
        waves, newer, ci, other, increments=increments)
    assert delta == pytest.approx(float(gradient[key] @ change), rel=2e-6, abs=2e-17)
