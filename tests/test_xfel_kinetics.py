"""Independent analytic gates for finite pulse dynamics and imaging."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from atomx.xfel import (
    GaussianPulse,
    KineticGraph,
    KineticTransition,
    ResonantLine,
    ScatteringFactorBank,
    TabulatedPulse,
    add_resonances,
)
from atomx.xfel.native_channels import shakeoff_probabilities


def two_state(sigma=1e-18):
    return KineticGraph(
        {"a": 0, "": 1},
        [KineticTransition("a", "", sigma, "pi", 1, 1)],
        initial_state="a",
        photon_energy_eV=100,
    )


@pytest.mark.parametrize("fwhm", [1.0, 10.0, 100.0])
def test_gaussian_fluence_and_exponential_survival(fwhm):
    pulse = GaussianPulse(100, 2e18, fwhm)
    fluence = quad(lambda t: pulse.flux(t) * 1e-15, *pulse.support_fs, epsabs=1e4)[0]
    assert fluence == pytest.approx(2e18, rel=1e-12)
    result = two_state().solve(pulse)
    assert result.populations[-1, 0] == pytest.approx(np.exp(-2), rel=1e-8)
    assert result.mean_charge[-1] == pytest.approx(1 - np.exp(-2), rel=1e-8)
    assert result.photons_absorbed[-1] == pytest.approx(
        result.mean_charge[-1], rel=1e-10
    )


def test_finite_auger_branching_matches_matrix_exponential():
    graph = KineticGraph(
        {"a": 0, "hole": 1, "": 2},
        [
            KineticTransition("a", "hole", 1e-18, "pi", 1, 1),
            KineticTransition("hole", "", 3e13, "auger", 1, 0),
        ],
        initial_state="a",
        photon_energy_eV=100,
    )
    pulse = TabulatedPulse(100, [0, 100], [2e31, 2e31])
    result = graph.solve(pulse, times_fs=np.linspace(0, 100, 101))
    matrix = np.array([[-2e13, 0, 0], [2e13, -3e13, 0], [0, 3e13, 0]])
    expected = expm(matrix * 100e-15) @ np.array([1, 0, 0])
    assert result.populations[-1] == pytest.approx(expected, rel=1e-8, abs=1e-10)
    assert result.electrons_emitted[-1] == pytest.approx(
        result.mean_charge[-1], rel=1e-10
    )


def test_pulse_structure_matters_with_finite_lifetimes():
    graph = KineticGraph(
        {"a": 0, "h": 1, "": 2},
        [
            KineticTransition("a", "h", 1e-18, "pi", 1, 1),
            KineticTransition("h", "", 1e13, "auger", 1, 0),
        ],
        initial_state="a",
        photon_energy_eV=100,
    )
    short = graph.solve(GaussianPulse(100, 1e18, 1))
    long = graph.solve(GaussianPulse(100, 1e18, 100))
    assert long.mean_charge[-1] > short.mean_charge[-1] + 0.2


def test_imaging_uses_mean_intensity_instead_of_squared_mean():
    pulse = GaussianPulse(100, 1e18, 10)
    dynamics = two_state().solve(pulse)
    bank = ScatteringFactorBank(
        [0],
        {"a": [10.0], "": [0.0]},
        photon_energy_eV=100,
        provenance={"control": "two-state analytic"},
    )
    signal = bank.observe(dynamics, pulse)
    assert signal.mean_intensity[:, 0] == pytest.approx(
        100 * dynamics.populations[:, 0]
    )
    assert signal.fluctuation_intensity[-1, 0] > 20
    assert (
        signal.pulse_average_intensity[0] > abs(signal.pulse_average_amplitude[0]) ** 2
    )


def test_missing_imaging_state_is_not_silently_zero():
    pulse = GaussianPulse(100, 1e18, 10)
    bank = ScatteringFactorBank(
        [0], {"a": [1.0]}, photon_energy_eV=100, provenance={"test": True}
    )
    with pytest.raises(ValueError, match="missing transient"):
        bank.observe(two_state().solve(pulse), pulse)


def test_line_area_and_bandwidth_conserve_oscillator_strength():
    line = ResonantLine("a", "x", 10.0, 0.4, 1e14, 2, 4, "analytic control")
    # Integral of a Lorentz profile over the real energy axis equals one.
    width = 6.582119569e-16 * 1e14 / 2
    integral = quad(
        lambda x: (
            line.integrated_sigma_cm2_eV * width / (np.pi * (x * x + width * width))
        ),
        -np.inf,
        np.inf,
        epsabs=0,
        epsrel=1e-9,
    )[0]
    assert integral == pytest.approx(line.integrated_sigma_cm2_eV, rel=1e-9)
    assert line.sigma_cm2(10, bandwidth_fwhm_eV=1) < line.sigma_cm2(10)


def test_resonant_stimulated_emission_has_correct_degeneracy_ratio():
    graph = KineticGraph(
        {"a": 0, "x": 0, "": 1},
        [KineticTransition("x", "a", 1e9, "radiative", 0, 0)],
        initial_state="a",
        photon_energy_eV=10,
    )
    line = ResonantLine("a", "x", 10, 0.4, 1e9, 2, 4, "test line")
    graph = add_resonances(graph, [line], bandwidth_fwhm_eV=0.1)
    edges = [e for e in graph.transitions if e.kind in graph.light_kinds]
    assert edges[1].coefficient / edges[0].coefficient == pytest.approx(0.5)
    with pytest.raises(ValueError, match="strong-field"):
        graph.solve(GaussianPulse(10, 1e19, 100, bandwidth_fwhm_eV=0.1))
    with pytest.raises(ValueError, match="time-bandwidth"):
        graph.solve(GaussianPulse(10, 1e13, 1, bandwidth_fwhm_eV=0.1))
    with pytest.raises(ValueError, match="bandwidth"):
        graph.solve(GaussianPulse(10, 1e18, 10))


def test_shakeoff_transcription_population_energy_and_norm_gates():
    q = {(2, -1): 2, (2, -2): 4}
    overlap = {(2, -1): 0.99, (2, -2): 0.98}
    threshold = {(2, -1): 60.0, (2, -2): 30.0}
    all_open = shakeoff_probabilities(
        q,
        overlap,
        available_energy_eV=100,
        thresholds_eV=threshold,
        provenance="XATOM equation",
    )
    assert all_open[(2, -1)] == pytest.approx(2 * (1 - 0.99**2))
    closed = shakeoff_probabilities(
        q,
        overlap,
        available_energy_eV=40,
        thresholds_eV=threshold,
        provenance="XATOM equation",
    )
    assert (2, -1) not in closed
    with pytest.raises(ValueError, match="exceeds unity"):
        shakeoff_probabilities(
            q,
            {s: 0 for s in q},
            available_energy_eV=100,
            thresholds_eV=threshold,
            provenance="invalid approximation",
        )


def test_graph_refuses_charge_or_unit_type_mismatch():
    with pytest.raises(ValueError, match="electron/charge"):
        KineticGraph(
            {"a": 0, "": 1},
            [KineticTransition("a", "", 1e14, "radiative")],
            initial_state="a",
            photon_energy_eV=100,
        )
