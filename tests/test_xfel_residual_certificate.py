"""Analytic residual-bound oracle and exact signed coordinate closure."""

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from atomx.xfel import CoherentGaussianPulse, CoherentSystem
from atomx.xfel.coherent_subspace import reachable_coordinates
from atomx.xfel.reservoir_errors import lindblad_difference_bound


def decay_model(relative_error=1e-5):
    b, c = np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    b[0, 1] = np.sqrt(1e16)
    c[0, 1] = np.sqrt(1e16 * (1 + relative_error))
    return CoherentSystem(
        {"g": 0.0, "e": 1.0},
        {"g": 0, "e": 0},
        np.zeros((2, 2)),
        charges={"g": 0, "e": 0},
        decay_operators=[b],
        reference_decay_operators=[c],
        reference_pi_operators=[],
        provenance={
            "radiative_arithmetic_generator_difference_bound_per_s": 0.0,
            "radiative_arithmetic_trace_norm_error_budget": 1e-3,
            "electron_pi_generator_difference_bound_cm2": 0.0,
            "electron_auger_generator_difference_bound_per_s": lindblad_difference_bound(
                c, b
            ),
            "reservoir_trace_norm_error_budget": 1e-3,
        },
    )


@pytest.mark.parametrize("integrator", ["DOP853", "BDF"])
def test_fast_decay_residual_bound_matches_exact_integral_and_reference(integrator):
    model = decay_model()
    pulse = CoherentGaussianPulse(100, 0, 10)
    rho = np.diag([0.0, 1.0])
    start, stop = pulse.support_fs
    times = np.r_[start, start + 0.01, start + 0.1, start + 1, stop]
    with pytest.raises(ValueError, match="combined reservoir"):
        model.solve(pulse, rho, times_fs=times)
    result = model.solve(
        pulse,
        rho,
        times_fs=times,
        reservoir_certificate="residual",
        density_representation="reachable",
        max_step_fs=0.1,
        rtol=1e-11,
        atol=1e-14,
        integrator=integrator,
    )
    t = times - start
    expected = np.array([np.diag([1 - np.exp(-10 * x), np.exp(-10 * x)]) for x in t])
    np.testing.assert_allclose(result.density_matrices, expected, rtol=0, atol=2e-11)
    # Nuclear residual bound sqrt(2)||R||_F = 2 delta_rate p_e.
    exact_bound = 2e-5 * (1 - np.exp(-10 * (stop - start)))
    assert result.provenance["reservoir_residual_bound_estimate"] == pytest.approx(
        exact_bound, abs=2e-11
    )
    reference = np.array(
        [np.diag([1 - np.exp(-10.0001 * x), np.exp(-10.0001 * x)]) for x in t]
    )
    actual = max(
        np.linalg.svd(a - b, compute_uv=False).sum()
        for a, b in zip(expected, reference, strict=True)
    )
    assert (
        actual < result.provenance["reservoir_reference_trace_norm_error_bound"] < 1e-3
    )
    assert (
        result.provenance["reservoir_uniform_reference_trace_norm_error_bound"] > 1e-3
    )
    assert result.provenance["density_active_coordinates"] == 2
    # Output stop is not the integration stop; certificate still covers full T.
    early = model.solve(
        pulse,
        rho,
        times_fs=times[:3],
        reservoir_certificate="residual",
        density_representation="reachable",
        max_step_fs=0.1,
        rtol=1e-11,
        atol=1e-14,
        integrator=integrator,
    )
    assert early.provenance["reservoir_residual_bound_estimate"] == pytest.approx(
        exact_bound, abs=2e-11
    )


def test_residual_mode_does_not_remove_error_or_resolution_gates():
    pulse = CoherentGaussianPulse(100, 0, 10)
    with pytest.raises(ValueError, match="residual reservoir.*budget"):
        decay_model(0.01).solve(
            pulse,
            np.diag([0.0, 1.0]),
            reservoir_certificate="residual",
            density_representation="reachable",
            max_step_fs=0.1,
        )
    start, stop = pulse.support_fs
    with pytest.raises(RuntimeError, match="time refinement"):
        decay_model().solve(
            pulse,
            np.diag([0.0, 1.0]),
            times_fs=[start, start + 0.1, start + 0.2, stop],
            reservoir_certificate="residual",
            density_representation="reachable",
            rtol=1e-3,
            atol=1e-5,
        )
    plain = CoherentSystem(
        {"g": 0},
        {"g": 0},
        np.zeros((1, 1)),
        charges={"g": 0},
        provenance={"explicit": "no reference"},
    )
    with pytest.raises(ValueError, match="explicit original-energy"):
        plain.solve(pulse, np.ones((1, 1)), reservoir_certificate="residual")


def test_residual_integral_with_radiative_growth_matches_closed_form():
    # An additional bounded radiative perturbation grows the error envelope;
    # keep both this growth and the occupation-dependent electron forcing.
    model = decay_model()
    eta_per_s = 1e9
    model.provenance["radiative_arithmetic_generator_difference_bound_per_s"] = (
        eta_per_s
    )
    pulse = CoherentGaussianPulse(100, 0, 10)
    start, stop = pulse.support_fs
    result = model.solve(
        pulse,
        np.diag([0.0, 1.0]),
        times_fs=[start, start + 0.1, stop],
        reservoir_certificate="residual",
        density_representation="reachable",
        integrator="BDF",
        rtol=1e-11,
        atol=1e-14,
    )
    duration = stop - start
    eta = eta_per_s * 1e-15
    rate = 10.0
    relative_error = 1e-5
    # Direct integration of exp(-rate*t) forcing with exp(eta*t) growth.
    exact = np.expm1(eta * duration) + 2 * relative_error * rate / (rate + eta) * (
        np.exp(eta * duration) - np.exp(-rate * duration)
    )
    assert result.provenance["reservoir_residual_bound_estimate"] == pytest.approx(
        exact, abs=2e-11
    )
    assert result.provenance[
        "radiative_arithmetic_trace_norm_error_bound"
    ] == pytest.approx(np.expm1(eta * duration), abs=1e-15)


def test_exact_reachability_uses_union_reference_and_no_magnitude_cutoff():
    a = np.zeros((4, 4), complex)
    a[1, 0] = 1
    a[2, 1] = 1e-80
    b = np.zeros_like(a)
    b[1, 0] = -1
    initial = np.array([1, 0, 0, 0])
    # Separate affine terms cancel at one parameter choice, not for all fields.
    np.testing.assert_array_equal(
        reachable_coordinates(initial, [csc_matrix(a), csc_matrix(b)]), [0, 1, 2]
    )
    reference = np.zeros_like(a)
    reference[3, 2] = 1e-60
    np.testing.assert_array_equal(
        reachable_coordinates(initial, [a, b, reference]), [0, 1, 2, 3]
    )
    with pytest.raises(ValueError, match="matching finite"):
        reachable_coordinates(initial, [np.zeros((2, 2))])


@pytest.mark.parametrize("integrator", ["DOP853", "BDF"])
def test_reachable_signed_drive_loss_equals_full_density_and_phase_covariance(
    integrator,
):
    dipole = np.zeros((3, 3), complex)
    dipole[1, 0], dipole[2, 0] = 0.2, -0.13j
    decay = np.zeros((3, 3), complex)
    decay[0, 1], decay[0, 2] = 1e6, -2j * 1e6
    reference = decay.copy()
    reference[0, 1] *= 1.0001
    pulse = CoherentGaussianPulse(100, 1e16, 10, phase_rad=0.4)

    def build(d, b, c):
        return CoherentSystem(
            {"g": 0.0, "a": 100.1, "b": 100.3},
            {"g": 0, "a": 1, "b": 1},
            d,
            charges={"g": 0, "a": 0, "b": 0},
            decay_operators=[b],
            reference_decay_operators=[c],
            reference_pi_operators=[],
            provenance={
                "radiative_arithmetic_generator_difference_bound_per_s": 0.0,
                "radiative_arithmetic_trace_norm_error_budget": 1e-3,
                "electron_pi_generator_difference_bound_cm2": 0.0,
                "electron_auger_generator_difference_bound_per_s": lindblad_difference_bound(
                    c, b
                ),
                "reservoir_trace_norm_error_budget": 1e-3,
            },
        )

    model = build(dipole, decay, reference)
    psi = np.array([1, 0.2j, 0.1])
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    times = np.linspace(*pulse.support_fs, 21)
    full = model.solve(pulse, rho, times_fs=times, density_representation="full")
    reachable = model.solve(
        pulse, rho, times_fs=times, density_representation="reachable"
    )
    np.testing.assert_allclose(
        full.density_matrices, reachable.density_matrices, rtol=0, atol=2e-10
    )
    a = model.solve(
        pulse,
        rho,
        times_fs=times,
        density_representation="reachable",
        reservoir_certificate="residual",
        integrator=integrator,
        rtol=1e-11,
        atol=1e-14,
    )
    u = np.diag([1, 1j, -1])
    changed = build(
        u @ dipole @ u.conj().T, u @ decay @ u.conj().T, u @ reference @ u.conj().T
    )
    b = changed.solve(
        pulse,
        u @ rho @ u.conj().T,
        times_fs=times,
        density_representation="reachable",
        reservoir_certificate="residual",
        integrator=integrator,
        rtol=1e-11,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        b.density_matrices, u @ a.density_matrices @ u.conj().T, rtol=0, atol=2e-10
    )
    assert b.provenance["reservoir_residual_bound_estimate"] == pytest.approx(
        a.provenance["reservoir_residual_bound_estimate"], abs=2e-12
    )


def test_real_residual_jacobian_matches_independent_finite_difference():
    from atomx.xfel.reservoir_errors import real_residual_jacobian

    rng = np.random.default_rng(83)
    generator = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    difference = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    x = np.r_[state.real, state.imag, 0.2]

    def independent(y):
        v = y[:4] + 1j * y[4:8]
        derivative = generator @ v
        residual = difference @ v
        value = 0.003 * (y[-1] + 1) + np.sqrt(2) * np.sqrt(
            sum(abs(c) ** 2 for c in residual)
        )
        return np.r_[derivative.real, derivative.imag, value]

    columns = []
    for column in range(9):
        delta = np.zeros(9)
        delta[column] = 1e-6
        columns.append((independent(x + delta) - independent(x - delta)) / (2e-6))
    numerical = np.column_stack(columns)
    jacobian = real_residual_jacobian(
        generator, difference, state, dimension=2, eta_fs=0.003
    )
    np.testing.assert_allclose(jacobian.toarray(), numerical, rtol=0, atol=3e-8)
