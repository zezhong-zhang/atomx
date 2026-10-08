"""Incoherent resonance pumping with explicit level population and decay.

An oscillator-strength line has integrated cross section 2*pi^2*alpha*f
in atomic units. A causal response with both frequency poles includes
upper/lower natural widths and Gaussian photon bandwidth. This model does not describe
coherent Rabi dynamics or overlapping Fano-channel interference.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import roots_legendre, wofz

from atomx.constants import ALPHA, AU_RATE_PER_S, BOHR_RADIUS_M, HARTREE_EV
from atomx.xfel.kinetics import KineticGraph, KineticTransition

_HBAR_EV_S = HARTREE_EV / AU_RATE_PER_S
_LINE_AREA_CM2_EV = 2 * np.pi**2 * ALPHA * (BOHR_RADIUS_M * 100) ** 2 * HARTREE_EV


@dataclass(frozen=True)
class ResonantLine:
    lower: str
    upper: str
    energy_eV: float
    oscillator_strength: float
    total_decay_per_s: float
    g_lower: float
    g_upper: float
    provenance: str
    lower_decay_per_s: float = 0.0

    def __post_init__(self):
        if (
            not np.isfinite(
                [
                    self.energy_eV,
                    self.oscillator_strength,
                    self.total_decay_per_s,
                    self.g_lower,
                    self.g_upper,
                    self.lower_decay_per_s,
                ]
            ).all()
            or min(self.energy_eV, self.total_decay_per_s, self.g_lower, self.g_upper)
            <= 0
            or self.oscillator_strength < 0
            or self.lower_decay_per_s < 0
            or not self.provenance
        ):
            raise ValueError(
                "line requires physical energies, widths, weights and provenance"
            )
        if not np.isfinite(self.half_width_eV) or self.half_width_eV <= 0:
            raise ValueError("natural width is numerically unresolved")

    @classmethod
    def from_einstein_A(
        cls,
        lower,
        upper,
        energy_eV,
        A_per_s,
        *,
        total_decay_per_s,
        g_lower,
        g_upper,
        provenance,
        lower_decay_per_s=0.0,
    ):
        """Einstein absorption/emission relation in the same E1 convention."""
        if not np.isfinite(A_per_s) or A_per_s < 0 or total_decay_per_s < A_per_s:
            raise ValueError("total decay width must contain this radiative branch")
        omega = energy_eV / HARTREE_EV
        f = A_per_s / AU_RATE_PER_S * (g_upper / g_lower) / (2 * ALPHA**3 * omega**2)
        return cls(
            lower,
            upper,
            energy_eV,
            f,
            total_decay_per_s,
            g_lower,
            g_upper,
            provenance,
            lower_decay_per_s,
        )

    @property
    def integrated_sigma_cm2_eV(self):
        return self.oscillator_strength * _LINE_AREA_CM2_EV

    @property
    def half_width_eV(self):
        return 0.5 * _HBAR_EV_S * (self.total_decay_per_s + self.lower_decay_per_s)

    @staticmethod
    def _pole_average(delta, gamma, sigma):
        if sigma == 0:
            return 1 / (delta - 1j * gamma)
        return (
            1j
            * np.sqrt(np.pi / 2)
            / sigma
            * np.conj(wofz((delta + 1j * gamma) / (np.sqrt(2) * sigma)))
        )

    @staticmethod
    def _smooth_average(function, energy, sigma):
        nodes, weights = roots_legendre(64)
        z = 9 * nodes
        return weights @ (
            function(energy + sigma * z) * 9 * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
        )

    def anomalous_factor(self, energy_eV, *, bandwidth_fwhm_eV=0.0):
        """Causal oscillator response f*w0²/(w²-w0²-i*Gamma*w).

        Includes the negative-frequency pole, so f1(0)=-f and the
        positive-frequency TRK integral equals f. A Gaussian convolution
        uses the analytic complex pole average, even for narrow lines.
        This returns a mean amplitude; spectral intensity must be averaged
        separately rather than squaring this mean for a broadband pulse.
        """
        self._validate_probe(energy_eV, bandwidth_fwhm_eV, allow_zero=True)
        gamma = self.half_width_eV
        if bandwidth_fwhm_eV == 0:
            return (
                self.oscillator_strength
                * self.energy_eV**2
                / (energy_eV**2 - self.energy_eV**2 - 2j * gamma * energy_eV)
            )
        sigma = bandwidth_fwhm_eV / np.sqrt(8 * np.log(2))
        if energy_eV <= 9 * sigma:
            raise ValueError("Gaussian photon spectrum extends to nonpositive energies")
        if gamma >= self.energy_eV:
            return self._smooth_average(
                lambda e: (
                    self.oscillator_strength
                    * self.energy_eV**2
                    / (e**2 - self.energy_eV**2 - 2j * gamma * e)
                ),
                energy_eV,
                sigma,
            )
        pole = np.sqrt(self.energy_eV**2 - gamma**2)
        return (
            self.oscillator_strength
            * self.energy_eV**2
            / (2 * pole)
            * (
                self._pole_average(energy_eV - pole, gamma, sigma)
                - self._pole_average(energy_eV + pole, gamma, sigma)
            )
        )

    @staticmethod
    def _validate_probe(energy, bandwidth, *, allow_zero=False):
        if (
            not np.isfinite([energy, bandwidth]).all()
            or bandwidth < 0
            or energy < 0
            or (energy == 0 and not allow_zero)
        ):
            raise ValueError("invalid photon energy or bandwidth")

    def sigma_cm2(self, energy_eV, *, bandwidth_fwhm_eV=0.0):
        self._validate_probe(energy_eV, bandwidth_fwhm_eV)
        sigma = bandwidth_fwhm_eV / np.sqrt(8 * np.log(2))
        gamma = self.half_width_eV
        scale = 4 * np.pi * ALPHA * HARTREE_EV * (BOHR_RADIUS_M * 100) ** 2

        def natural(energy):
            return (
                scale
                * self.oscillator_strength
                * self.energy_eV**2
                * 2
                * gamma
                / ((energy**2 - self.energy_eV**2) ** 2 + (2 * gamma * energy) ** 2)
            )

        if sigma == 0:
            return float(natural(energy_eV))
        if energy_eV <= 9 * sigma:
            raise ValueError("Gaussian photon spectrum extends to nonpositive energies")
        if gamma >= self.energy_eV or abs(energy_eV - self.energy_eV) > 9 * sigma:
            return float(self._smooth_average(natural, energy_eV, sigma))
        pole = np.sqrt(self.energy_eV**2 - gamma**2)
        value = (
            scale
            * self.oscillator_strength
            / (2 * pole)
            * np.imag(
                (pole - 1j * gamma) * self._pole_average(energy_eV - pole, gamma, sigma)
                + (pole + 1j * gamma)
                * self._pole_average(energy_eV + pole, gamma, sigma)
            )
        )
        if value < 0 or not np.isfinite(value):
            raise RuntimeError(
                "causal line convolution produced an invalid cross section"
            )
        return float(value)


def add_resonances(graph, lines, *, bandwidth_fwhm_eV):
    """Add absorption and stimulated emission to existing excited states.

    Excited-state PI and spontaneous decay must already be present and
    graph-closed. No resonance is folded into an irreversible PI edge.
    """
    previous_bandwidth = graph.provenance.get("resonance_bandwidth_fwhm_eV")
    if previous_bandwidth is not None and not np.isclose(
        previous_bandwidth, bandwidth_fwhm_eV, rtol=0, atol=1e-12
    ):
        raise ValueError("cannot combine different resonance bandwidths")
    edges = list(graph.transitions)
    seen = {(e.source, e.target) for e in edges if e.kind == "resonant_absorption"}
    validity = list(graph.provenance.get("resonance_validity", []))
    for line in lines:
        if (line.lower, line.upper) in seen:
            raise ValueError("duplicate resonant transition")
        seen.add((line.lower, line.upper))
        if line.lower not in graph.charges or line.upper not in graph.charges:
            raise ValueError("resonance requires explicit lower and excited states")
        decay = sum(
            e.coefficient
            for e in graph.transitions
            if e.source == line.upper and e.kind in graph.decay_kinds
        )
        if not np.isclose(decay, line.total_decay_per_s, rtol=1e-8, atol=0):
            raise ValueError("line width differs from graph excited-state decay sum")
        lower_decay = sum(
            e.coefficient
            for e in graph.transitions
            if e.source == line.lower and e.kind in graph.decay_kinds
        )
        if not np.isclose(lower_decay, line.lower_decay_per_s, rtol=1e-8, atol=0):
            raise ValueError("line width differs from graph lower-state decay sum")
        sigma = line.sigma_cm2(
            graph.photon_energy_eV, bandwidth_fwhm_eV=bandwidth_fwhm_eV
        )
        validity.append(
            dict(
                peak_pumping_sigma_cm2=sigma * (1 + line.g_lower / line.g_upper),
                correlation_rate_s=max(
                    bandwidth_fwhm_eV / _HBAR_EV_S,
                    (line.total_decay_per_s + line.lower_decay_per_s) / 2,
                ),
            )
        )
        edges.append(
            KineticTransition(
                line.lower,
                line.upper,
                sigma,
                "resonant_absorption",
                0,
                1,
                line.provenance,
            )
        )
        edges.append(
            KineticTransition(
                line.upper,
                line.lower,
                sigma * line.g_lower / line.g_upper,
                "stimulated_emission",
                0,
                -1,
                line.provenance,
            )
        )
    return KineticGraph(
        graph.charges,
        edges,
        initial_state=graph.initial_state,
        photon_energy_eV=graph.photon_energy_eV,
        provenance=dict(
            graph.provenance,
            resonance_bandwidth_fwhm_eV=bandwidth_fwhm_eV,
            resonance_validity=validity,
            resonance_model="incoherent causal oscillator pumping",
        ),
    )
