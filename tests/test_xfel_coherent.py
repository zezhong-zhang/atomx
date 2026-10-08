"""Independent field units, angular sums and exact coherent limits."""

import numpy as np
import pytest
from scipy.integrate import quad
from sympy import Rational
from sympy.physics.wigner import wigner_3j

from atomx.constants import AU_RATE_PER_S, BOHR_RADIUS_M, HARTREE_EV, HARTREE_J, units
from atomx.xfel import CoherentGaussianPulse, CoherentSystem, GaussianPulse
from atomx.xfel.coherent import magnetic_e1_component


def two_level(dipole=1, decay=0):
    jump = np.array([[0, np.sqrt(decay)], [0, 0]], complex)
    return CoherentSystem(
        {"g": 0, "e": 100},
        {"g": 0, "e": 1},
        [[0, 0], [dipole, 0]],
        charges={"g": 0, "e": 0},
        decay_operators=[jump],
        provenance={"control": "two level"},
    )


def area(pulse, dipole=1):
    return quad(
        lambda t: pulse.electric_field_au(t).real * dipole * AU_RATE_PER_S * 1e-15,
        *pulse.support_fs,
        epsabs=1e-10,
    )[0]


def test_coherent_field_matches_independent_si_intensity_and_time_bandwidth():
    pulse = CoherentGaussianPulse(100, 1e17, 10)
    field_si = abs(pulse.electric_field_au(0)) * HARTREE_J / (units._e * BOHR_RADIUS_M)
    intensity_si = 0.5 * units._eps0 * units._c * field_si**2
    photon_intensity_si = pulse.flux(0) * 1e4 * 100 * units._e
    assert intensity_si == pytest.approx(photon_intensity_si, rel=2e-9)
    assert pulse.bandwidth_fwhm_eV * 10 == pytest.approx(
        4 * np.log(2) * units._hbar / units._e * 1e15
    )
    fluence = quad(lambda t: pulse.flux(t) * 1e-15, *pulse.support_fs)[0]
    assert fluence == pytest.approx(1e17, rel=1e-12)


def test_pi_and_two_pi_gaussian_pulses_obey_exact_area_theorem():
    base = CoherentGaussianPulse(100, 1e15, 10)
    fluence = 1e15 * (np.pi / area(base)) ** 2
    system = two_level()
    for multiple in (1, 2):
        pulse = CoherentGaussianPulse(100, fluence * multiple**2, 10)
        result = system.solve(pulse, [[1, 0], [0, 0]])
        expected = np.sin(area(pulse) / 2) ** 2
        assert result.populations[-1, 1] == pytest.approx(expected, abs=2e-9)
        assert result.provenance["max_trace_error"] < 1e-12
        assert result.provenance["minimum_density_eigenvalue"] > -1e-9


def test_decay_and_off_diagonal_coherence_match_closed_lindblad_solution():
    gamma = 1e13
    system = two_level(dipole=0, decay=gamma)
    pulse = CoherentGaussianPulse(100, 0, 10)
    result = system.solve(pulse, [[0.5, 0.5], [0.5, 0.5]])
    elapsed_s = (result.times_fs - result.times_fs[0]) * 1e-15
    np.testing.assert_allclose(
        result.populations[:, 1], 0.5 * np.exp(-gamma * elapsed_s), atol=1e-12
    )
    np.testing.assert_allclose(
        result.density_matrices[:, 0, 1],
        0.5 * np.exp(-gamma * elapsed_s / 2),
        atol=1e-12,
    )


def test_complex_carrier_phase_changes_coherence_not_population():
    pulse = CoherentGaussianPulse(100, 1e15, 10)
    shifted = CoherentGaussianPulse(100, 1e15, 10, phase_rad=np.pi / 2)
    system = two_level()
    first = system.solve(pulse, [[1, 0], [0, 0]])
    second = system.solve(shifted, [[1, 0], [0, 0]])
    np.testing.assert_allclose(first.populations, second.populations, atol=1e-12)
    np.testing.assert_allclose(
        second.density_matrices[:, 1, 0],
        1j * first.density_matrices[:, 1, 0],
        atol=1e-12,
    )


def test_magnetic_dipoles_match_sympy_and_independent_einstein_sums():
    for jl, ju in ((1, 1), (1, 3), (3, 5)):
        for mu in range(-ju, ju + 1, 2):
            total = 0
            for ml in range(-jl, jl + 1, 2):
                for q in (-1, 0, 1):
                    value = magnetic_e1_component(jl, ml, ju, mu, q, 1.7)
                    expected = (
                        float(
                            (-1) ** ((jl - ml) // 2)
                            * wigner_3j(
                                Rational(jl, 2),
                                1,
                                Rational(ju, 2),
                                -Rational(ml, 2),
                                q,
                                Rational(mu, 2),
                            )
                        )
                        * 1.7
                    )
                    assert value == pytest.approx(expected, abs=1e-14)
                    total += value**2
            assert total == pytest.approx(1.7**2 / (ju + 1), rel=1e-14)


def test_pi_jump_preserves_trace_and_accumulates_charge():
    system = CoherentSystem(
        {"g": 0, "ion": 150},
        {"g": 0, "ion": 0},
        np.zeros((2, 2)),
        charges={"g": 0, "ion": 1},
        pi_operators=[[[0, 0], [1e-9, 0]]],
        provenance={"control": "PI"},
    )
    pulse = CoherentGaussianPulse(100, 1e17, 10)
    result = system.solve(pulse, [[1, 0], [0, 0]])
    assert result.populations[-1, 0] == pytest.approx(np.exp(-0.1), rel=1e-10)
    assert result.mean_charge[-1] == pytest.approx(1 - np.exp(-0.1), rel=1e-10)


def test_coherent_model_rejects_invalid_density_and_implicit_rate_pulses():
    system = two_level()
    pulse = CoherentGaussianPulse(100, 1e15, 10)
    with pytest.raises(ValueError, match="complex-field"):
        system.solve(GaussianPulse(100, 1e15, 10), [[1, 0], [0, 0]])
    for density in ([[1, 1], [0, 0]], [[1.1, 0], [0, -0.1]], [[0.4, 0], [0, 0.4]]):
        with pytest.raises(ValueError, match="initial density"):
            system.solve(pulse, density)
    with pytest.raises(ValueError, match="rotating-wave"):
        two_level(100).solve(CoherentGaussianPulse(100, 1e20, 10), [[1, 0], [0, 0]])
    with pytest.raises(ValueError, match="raise one photon"):
        CoherentSystem(
            {"g": 0, "e": 100},
            {"g": 0, "e": 0},
            [[0, 0], [1, 0]],
            charges={"g": 0, "e": 0},
            provenance={"control": "bad ladder"},
        )


def test_shared_decay_operator_preserves_dark_state_coherence():
    dipole = np.zeros((3, 3))
    jump = np.zeros((3, 3), complex)
    jump[0, 1:] = np.sqrt(1e13)
    system = CoherentSystem(
        {"g": 0, "a": 100, "b": 100},
        {"g": 0, "a": 1, "b": 1},
        dipole,
        charges={"g": 0, "a": 0, "b": 0},
        decay_operators=[jump],
        provenance={"control": "degenerate shared bath"},
    )
    dark = np.array([0, 1, -1]) / np.sqrt(2)
    density = np.outer(dark, dark)
    result = system.solve(CoherentGaussianPulse(100, 0, 10), density)
    np.testing.assert_allclose(
        result.density_matrices,
        np.broadcast_to(density, result.density_matrices.shape),
        atol=1e-12,
    )
    bright = np.array([0, 1, 1]) / np.sqrt(2)
    result = system.solve(CoherentGaussianPulse(100, 0, 10), np.outer(bright, bright))
    elapsed = (result.times_fs - result.times_fs[0]) * 1e-15
    np.testing.assert_allclose(
        result.populations[:, 1:].sum(axis=1), np.exp(-2e13 * elapsed), atol=1e-12
    )


def test_relative_dipole_phase_produces_bright_superposition():
    system = CoherentSystem(
        {"g": 0, "a": 100, "b": 100},
        {"g": 0, "a": 1, "b": 1},
        [[0, 0, 0], [1, 0, 0], [1j, 0, 0]],
        charges={"g": 0, "a": 0, "b": 0},
        provenance={"control": "signed bright state"},
    )
    base = CoherentGaussianPulse(100, 1e15, 10)
    pulse = CoherentGaussianPulse(
        100, 1e15 * (np.pi / (np.sqrt(2) * area(base))) ** 2, 10
    )
    result = system.solve(pulse, np.diag([1, 0, 0]))
    np.testing.assert_allclose(result.populations[-1], [0, 0.5, 0.5], atol=1e-9)
    assert result.density_matrices[-1, 1, 2] == pytest.approx(-0.5j, abs=1e-9)


def test_detuning_rotates_free_coherence_at_native_hbar():
    system = CoherentSystem(
        {"g": 0, "e": 100.1},
        {"g": 0, "e": 1},
        np.zeros((2, 2)),
        charges={"g": 0, "e": 0},
        provenance={"control": "detuning"},
    )
    result = system.solve(CoherentGaussianPulse(100, 0, 10), [[0.5, 0.5], [0.5, 0.5]])
    dt = result.times_fs - result.times_fs[0]
    expected = 0.5 * np.exp(1j * 0.1 / HARTREE_EV * AU_RATE_PER_S * 1e-15 * dt)
    np.testing.assert_allclose(result.density_matrices[:, 0, 1], expected, atol=1e-10)
