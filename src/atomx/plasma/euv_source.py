"""Energy-conserving reduced laser/EUV source with fixed-charge LTE closure.

This is a homogeneous, symmetric, one-temperature expanding slab, not a
spatially resolved ablation or multi-charge radiation-hydrodynamics solver.
Absolute line powers, Einstein absorption including stimulated emission,
stationary angular transfer, excitation heat capacity and pressure work share
one state and one energy ledger. See docs/EUV_SOURCE_MODEL.md for validity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix

from atomx.constants import CODATA, DEFAULT_CODATA_VERSION
from atomx.plasma.laser_absorption import (
    critical_density_per_m3,
    drude_slab_fractions,
    electron_ion_collision_frequency,
)
from atomx.spectrum.slab_transfer import slab_escape_factor, slab_escaped_flux

_C = CODATA[DEFAULT_CODATA_VERSION]
EV_J = _C["_e"]
HC_EV_NM = _C["_hplanck"] * _C["_c"] / EV_J * 1e9


class LTELinePlasma:
    """Complete declared level partition and an explicitly scoped E1 line list.

    No renormalization to the line endpoints, strength screening, energy shifts
    or empirically scaled powers. Uniform energy bins conserve integrated
    power. A specified Gaussian FWHM is a phenomenological redistribution
    closure, not a calculated Stark width or an instrumental CE correction.
    """

    def __init__(self, energies_eV, weights, upper, lower, A_per_s, *, charge,
                 ion_mass_amu, energy_edges_eV, fwhm_eV, source_id, emission_scope,
                 memory_cap_bytes=2 * 1024**3):
        self.energy = np.array(energies_eV, dtype=float, copy=True)
        self.g = np.array(weights, dtype=float, copy=True)
        upper, lower = np.asarray(upper), np.asarray(lower)
        rates = np.asarray(A_per_s, dtype=float)
        edges = np.array(energy_edges_eV, dtype=float, copy=True)
        if (self.energy.ndim != 1 or not self.energy.size or self.g.shape != self.energy.shape
                or np.any(~np.isfinite(self.energy)) or np.any(self.energy < 0)
                or self.energy.min() != 0 or np.any(~np.isfinite(self.g)) or np.any(self.g <= 0)
                or upper.ndim != 1 or upper.shape != lower.shape or rates.shape != upper.shape
                or upper.dtype.kind not in "iu" or lower.dtype.kind not in "iu"
                or np.any(upper < 0) or np.any(lower < 0)
                or np.any(upper >= self.energy.size) or np.any(lower >= self.energy.size)
                or np.any(~np.isfinite(rates)) or np.any(rates < 0)
                or not isinstance(charge, (int, np.integer)) or charge < 1
                or not np.isfinite(ion_mass_amu) or ion_mass_amu <= 0
                or not isinstance(source_id, str) or not source_id
                or not isinstance(emission_scope, str) or not emission_scope):
            raise ValueError("invalid level, line, ion or provenance inputs")
        if (edges.ndim != 1 or edges.size < 3 or edges[0] < 0
                or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0)
                or not np.allclose(np.diff(edges), edges[1] - edges[0], rtol=1e-10, atol=1e-14)
                or not np.isfinite(fwhm_eV) or fwhm_eV < 0):
            raise ValueError("finite increasing uniform positive-energy bins and nonnegative width required")
        # Conservative construction/sort/sparse scratch admission, before COO allocation.
        estimate = 240 * rates.size + 64 * self.energy.size + 64 * edges.size
        if estimate > memory_cap_bytes:
            raise MemoryError(f"line-table planning bound {estimate} > cap {memory_cap_bytes}; no lines removed")
        gap = self.energy[upper] - self.energy[lower]
        if np.any(gap <= 0) or np.any(gap < edges[0]) or np.any(gap >= edges[-1]):
            raise ValueError("every positive photon energy must lie inside the declared grid")
        self.edges = edges
        self.centers = (edges[:-1] + edges[1:]) / 2
        self.width_Hz = np.diff(edges) * EV_J / _C["_hplanck"]
        self.charge = int(charge)
        self.mass_kg = ion_mass_amu * _C["_amu"]
        self.source_id = source_id
        self.emission_scope = emission_scope
        self.n_lines = rates.size
        self.fwhm_eV = fwhm_eV
        self.construction_bound_bytes = estimate
        rows = np.searchsorted(edges, gap, side="right") - 1
        self.upper = np.array(upper, copy=True)
        self.lower = np.array(lower, copy=True)
        self.A_per_s = np.array(rates, copy=True)
        self._line_bins = rows
        shape = (self.centers.size, self.energy.size)

        def sparse(values, columns):
            return coo_matrix((values, (rows, columns)), shape=shape).tocsr()

        frequency = gap * EV_J / _C["_hplanck"]
        absorption = rates * _C["_c"]**2 / (8 * np.pi * frequency**2)
        self._emission = sparse(rates * gap * EV_J, upper)
        self._absorption = sparse(absorption * self.g[upper] / self.g[lower], lower)
        self._stimulated = sparse(absorption, upper)
        self._sigma_bins = fwhm_eV / np.sqrt(8 * np.log(2)) / (edges[1] - edges[0])
        # Column-normalized finite-grid positive Gaussian response. Every photon
        # remains in the declared positive-energy grid, including at boundaries.
        self._profile_norm = self._smooth(np.ones(self.centers.size))
        for array in (self.energy, self.g, self.edges, self.centers, self.width_Hz,
                      self.upper, self.lower, self.A_per_s, self._line_bins):
            array.setflags(write=False)

    def _smooth(self, array):
        if self._sigma_bins == 0:
            return array
        return gaussian_filter1d(array, self._sigma_bins, mode="constant", truncate=7.0)

    def _profile(self, array):
        return self._smooth(array / self._profile_norm)

    def lte(self, temperature_eV):
        """p, per-ion internal energy/eV and dU/dT (T measured in eV).

        U includes both electron/ion ideal-gas thermal energy and all supplied
        excitation states; charge ionization energies are constant here and
        excluded together with the initial preparation energy of that charge.
        """
        if not np.isfinite(temperature_eV) or temperature_eV <= 0:
            raise ValueError("positive finite temperature required")
        p = self.g * np.exp(-self.energy / temperature_eV)
        p /= p.sum()
        mean = float(p @ self.energy)
        variance = float(p @ (self.energy - mean)**2)
        thermal = 1.5 * (1 + self.charge)
        return p, thermal * temperature_eV + mean, thermal + variance / temperature_eV**2

    def coefficients(self, populations, ion_density_per_m3):
        """Absolute 4pi W/m³/bin and net absorption/m from common Einstein A.

        Einstein factors use each physical line center before binning. Finite
        profiles therefore satisfy Kirchhoff at line centers; wing/bin errors
        in a Planck comparison require width/grid convergence, not rescaling.
        Inverted populations (negative net opacity) reject this passive model.
        """
        p = np.asarray(populations, dtype=float)
        if (p.shape != self.energy.shape or np.any(~np.isfinite(p)) or np.any(p < 0)
                or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)
                or not np.isfinite(ion_density_per_m3) or ion_density_per_m3 <= 0):
            raise ValueError("normalized nonnegative populations and positive density required")
        emit = self._profile(self._emission @ p) * ion_density_per_m3
        lower = self._absorption @ p
        upper = self._stimulated @ p
        net = lower - upper
        tolerance = 8 * np.finfo(float).eps * max(float(np.max(lower, initial=0)), 1e-300)
        if net.min(initial=0) < -tolerance:
            raise ValueError("population inversion: passive slab does not support masers")
        # Only subtraction roundoff after an explicit gate; no physical inversion clipped.
        absorb = self._profile(np.maximum(net, 0)) * ion_density_per_m3 / self.width_Hz
        return emit, absorb

    def band_weights(self, wavelength_min_nm=13.365, wavelength_max_nm=13.635):
        if not (0 < wavelength_min_nm < wavelength_max_nm):
            raise ValueError("positive ordered band wavelengths required")
        lo, hi = HC_EV_NM / wavelength_max_nm, HC_EV_NM / wavelength_min_nm
        return np.maximum(0, np.minimum(self.edges[1:], hi) - np.maximum(self.edges[:-1], lo)) / np.diff(self.edges)

    def line_escape_factors(self, optical_depth):
        """Profile-averaged factors adjoint to the exact binned emission map.

        A one-zone non-LTE escape-probability closure: trapped photons cycle
        locally in their originating transition. Overlapping-line conversion,
        radiation pumping and spatial redistribution need a full CR/transfer
        solve and are not supplied by these factors.
        """
        beta = slab_escape_factor(optical_depth)
        if beta.shape != self.centers.shape:
            raise ValueError("one optical depth per spectral bin required")
        return (self._smooth(beta) / self._profile_norm)[self._line_bins]


@dataclass(frozen=True)
class SlabSourceConfig:
    """SI geometry/density, eV common temperature; no implicit unit conversions."""

    ion_density_per_m3: float
    thickness_m: float
    temperature_eV: float
    laser_wavelength_m: float
    coulomb_log: float
    duration_s: float
    expansion: bool = True
    max_critical_density_ratio: float = 0.8
    min_temperature_eV: float = 1.0
    rtol: float = 2e-7
    max_rhs_evaluations: int = 20000


def evolve_lte_slab(plasma, config, laser_intensity_W_m2, *, sample_times_s=None):
    """Evolve LTE excitation/heat, symmetric homologous expansion and energy flow.

    Ldot=2v, K/A=M v²/6, vdot=6P/M, so Kdot/A=P Ldot exactly.
    All EUV self-absorption is returned to matter in the stationary closure;
    only escaped photons cool it. Laser intensity is a caller-supplied function
    of seconds. Initial preheat is in the ledger, never attributed to this laser.
    """
    return _evolve_slab(plasma, config, laser_intensity_W_m2, sample_times_s=sample_times_s)


def evolve_cr_slab(plasma, config, laser_intensity_W_m2, crm, initial_populations,
                   *, sample_times_s=None, max_dynamic_levels=256):
    """Fixed-charge non-LTE alternative using the existing CRM's physical rates.

    Small stiff systems only: the joint temperature/population Radau Jacobian
    is dense. The level cap rejects instead of trimming the model. Multi-charge
    channels/DR, radiative continua and gf-based collision guesses are rejected;
    every line must match in identity, order and A. Explicit incident-energy
    Omega tables are required for every included transition. Forbidden or
    additional collisional channels remain a declared limitation of this CRM.
    """
    if plasma.energy.size > max_dynamic_levels:
        raise MemoryError("joint non-LTE level cap exceeded; sparse Jacobian coupling required, no levels removed")
    from atomx.plasma.population_network import omega_collision_pair

    if crm.charge != plasma.charge:
        raise ValueError("explicit matching CRM charge required")
    if crm.bound_free_channels or crm.dr_satellite_sources or crm.autoionization_channels:
        raise ValueError("coupled source currently requires a fixed charge; no bound-free/DR/autoionization energy closure")
    if crm.n_levels != plasma.energy.size or len(crm.transitions) != plasma.n_lines:
        raise ValueError("CRM and radiation table must share every level and transition")
    for i, level in enumerate(crm.levels):
        if level.energy_eV != plasma.energy[i] or level.g != plasma.g[i] or level.autoionization_rate_s != 0:
            raise ValueError("matched CRM energies, statistical weights and stable levels required")
    for i, tr in enumerate(crm.transitions):
        if tr.upper != plasma.upper[i] or tr.lower != plasma.lower[i] or tr.A_rate != plasma.A_per_s[i]:
            raise ValueError("CRM/radiation transition identity or Einstein A mismatch")
        if tr.omega is None or tr.omega_E_eV is None:
            raise ValueError("explicit collision strengths required; no gf-based fallback in the coupled source")
        omega_collision_pair(tr.omega_E_eV, tr.omega,
                             plasma.energy[tr.upper] - plasma.energy[tr.lower],
                             config.temperature_eV, plasma.g[tr.lower], plasma.g[tr.upper])
    initial = np.asarray(initial_populations, dtype=float)
    # Shared coefficient validation covers normalization, positivity and inversion.
    plasma.coefficients(initial, config.ion_density_per_m3)
    return _evolve_slab(plasma, config, laser_intensity_W_m2, sample_times_s=sample_times_s,
                        crm=crm, initial_populations=initial)


def _evolve_slab(plasma, config, laser_intensity_W_m2, *, sample_times_s, crm=None, initial_populations=None):
    c = config
    for name in ("ion_density_per_m3", "thickness_m", "temperature_eV", "laser_wavelength_m",
                 "coulomb_log", "duration_s", "max_critical_density_ratio", "min_temperature_eV", "rtol"):
        if not np.isfinite(getattr(c, name)) or getattr(c, name) <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if c.temperature_eV <= c.min_temperature_eV or c.max_rhs_evaluations < 1:
        raise ValueError("initial temperature above floor and positive RHS cap required")
    initial_ratio = (plasma.charge * c.ion_density_per_m3
                     / critical_density_per_m3(c.laser_wavelength_m))
    if initial_ratio > c.max_critical_density_ratio:
        raise ValueError(f"homogeneous corona density ratio {initial_ratio:.6g} exceeds declared cap; no density reduction")
    N = c.ion_density_per_m3 * c.thickness_m
    mass_areal = N * plasma.mass_kg
    band = plasma.band_weights()
    evaluations = 0

    def dynamic_populations(y):
        # Eliminate p_0: the finite-difference Jacobian then stays in the
        # particle-conserving tangent space. No post-step normalization.
        return np.concatenate(([1 - np.sum(y[9:])], y[9:]))

    def rhs(t_scaled, y):
        nonlocal evaluations
        evaluations += 1
        if evaluations > c.max_rhs_evaluations:
            raise RuntimeError("source RHS budget exceeded; no partial efficiency certified")
        T, L, v = y[:3]
        if crm is None:
            p, _, cv = plasma.lte(T)
        else:
            p, cv = dynamic_populations(y), 1.5 * (1 + plasma.charge)
        ni = N / L
        emissivity, opacity = plasma.coefficients(p, ni)
        escaped = slab_escaped_flux(emissivity, opacity, L)
        intensity = float(laser_intensity_W_m2(t_scaled * c.duration_s))
        if not np.isfinite(intensity) or intensity < 0:
            raise ValueError("laser intensity must be finite and nonnegative")
        nu = electron_ion_collision_frequency(ni * plasma.charge, T, plasma.charge, c.coulomb_log)
        fractions = drude_slab_fractions(ni * plasma.charge, nu, c.laser_wavelength_m, L)
        absorbed = intensity * fractions["absorption"]
        pressure = ni * (1 + plasma.charge) * T * EV_J
        Ldot = 2 * v if c.expansion else 0.0
        vdot = 6 * pressure / mass_areal if c.expansion else 0.0
        cooling = float(escaped.sum())
        pdot = np.empty(0)
        if crm is not None:
            factors = plasma.line_escape_factors(opacity * L)
            generator = crm.rate_matrix(ni * plasma.charge / 1e6, T, sparse=True, radiative_factors=factors)
            pdot = generator @ p
        excitation_change = float(plasma.energy @ pdot) if crm is not None else 0.0
        Tdot = (absorbed - cooling - pressure * Ldot - N * EV_J * excitation_change) / (N * EV_J * cv)
        # Time scaled to [0,1] for integration/finite difference conditioning.
        return np.concatenate(([Tdot, Ldot, vdot, intensity, absorbed,
                                intensity * fractions["reflection"], intensity * fractions["transmission"],
                                cooling, float(band @ escaped)], pdot[1:])) * c.duration_s

    def cold_event(t, y):
        return y[0] - c.min_temperature_eV

    cold_event.terminal = True
    cold_event.direction = -1
    y0 = np.zeros(9 + (0 if crm is None else plasma.energy.size - 1))
    y0[:2] = c.temperature_eV, c.thickness_m
    if crm is not None:
        y0[9:] = initial_populations[1:]

    def internal_energy(y):
        if crm is None:
            return N * EV_J * plasma.lte(y[0])[1]
        return N * EV_J * (1.5 * (1 + plasma.charge) * y[0] + plasma.energy @ dynamic_populations(y))

    initial_internal = internal_energy(y0)
    if sample_times_s is None:
        sample_times_s = np.linspace(0, c.duration_s, 201)
    times = np.asarray(sample_times_s, dtype=float)
    if (times.ndim != 1 or times.size < 2 or times[0] != 0 or times[-1] != c.duration_s
            or np.any(np.diff(times) <= 0)):
        raise ValueError("sample times must increase from zero through the full duration")
    atol = np.array([1e-7, c.thickness_m * 1e-9, 1e-5] + [max(initial_internal, 1) * 1e-10] * 6
                    + ([] if crm is None else [1e-12] * (plasma.energy.size - 1)))
    solution = solve_ivp(rhs, (0, 1), y0, t_eval=times / c.duration_s,
                         method="Radau", rtol=c.rtol, atol=atol, max_step=0.025,
                         events=cold_event)
    if not solution.success or solution.t[-1] != 1:
        raise RuntimeError(f"source evolution incomplete (including temperature-floor event): {solution.message}")
    y = solution.y
    internal = np.array([internal_energy(column) for column in y.T])
    kinetic = mass_areal * y[2]**2 / 6
    residual = initial_internal + y[3] - internal - kinetic - y[5] - y[6] - y[7]
    scale = initial_internal + y[3]
    defect = float(np.max(np.abs(residual) / scale))
    if defect > max(5e-6, 20 * c.rtol):
        raise ArithmeticError(f"energy ledger failed: relative defect {defect:.6g}")
    all_populations = None if crm is None else np.column_stack([dynamic_populations(column) for column in y.T])
    initial_pop = plasma.lte(c.temperature_eV)[0] if crm is None else all_populations[:, 0]
    final_pop = plasma.lte(y[0, -1])[0] if crm is None else all_populations[:, -1]
    if crm is not None and (all_populations.min() < 0 or np.max(np.abs(all_populations.sum(axis=0) - 1)) > 1e-10):
        raise ArithmeticError("joint non-LTE populations failed positivity/conservation gate")
    output_state = y if crm is None else np.vstack((y[:9], all_populations))
    state_rows = ("temperature_eV", "thickness_m", "edge_velocity_m_s", "incident_J_m2",
                  "absorbed_J_m2", "reflected_J_m2", "transmitted_J_m2",
                  "escaped_both_faces_J_m2", "band_both_faces_J_m2")
    if crm is not None:
        state_rows += tuple(f"population:{i}" for i in range(plasma.energy.size))
    return dict(time_s=solution.t * c.duration_s, state=output_state, internal_J_m2=internal,
                kinetic_J_m2=kinetic, ledger_residual_J_m2=residual,
                initial_populations=initial_pop, final_populations=final_pop,
                initial_internal_J_m2=initial_internal, maximum_relative_energy_defect=defect,
                initial_critical_density_ratio=initial_ratio, rhs_evaluations=evaluations,
                source_id=plasma.source_id, emission_scope=plasma.emission_scope,
                population_closure="fixed-charge LTE" if crm is None else "fixed-charge CRM with local escape probability",
                state_rows=state_rows)


def integrated_escaped_spectrum(plasma, evolution, ion_column_per_m2, *, relative_gate=5e-4):
    """Independent sampled-spectrum quadrature, checked against ODE ledgers.

    Returns both-face J/m²/bin. Refuses insufficient temporal sampling rather
    than scaling the spectrum to the ledger. Supports either source closure.
    """
    times, state = evolution["time_s"], evolution["state"]
    if (times.size < 3 or times.size % 2 != 1
            or not np.allclose(np.diff(times), times[1] - times[0], rtol=1e-10, atol=0)):
        raise ValueError("spectrum replay requires an odd, uniform time grid of at least three points")
    weights = np.full(times.size, 2 * (times[1] - times[0]) / 3)
    weights[1:-1:2] *= 2
    weights[[0, -1]] *= 0.5
    integral = np.zeros(plasma.centers.size)
    for weight, column in zip(weights, state.T, strict=True):
        p = plasma.lte(column[0])[0] if state.shape[0] == 9 else column[9:]
        eps, alpha = plasma.coefficients(p, ion_column_per_m2 / column[1])
        integral += weight * slab_escaped_flux(eps, alpha, column[1])
    ledger = state[[7, 8], -1]
    replay = np.array([integral.sum(), integral @ plasma.band_weights()])
    errors = np.abs(replay - ledger) / np.maximum(np.abs(ledger), 1e-100)
    if np.any(errors > relative_gate):
        raise ArithmeticError(f"sampled spectrum/ODE energy mismatch {errors}; refine sample times")
    return integral, dict(total_relative_error=float(errors[0]), band_relative_error=float(errors[1]),
                          relative_gate=relative_gate, sample_points=times.size)
