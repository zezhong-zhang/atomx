"""Conservative smooth-flow 1D material/laser/radiation coupling.

Preformed ideal plasma, plane-parallel Lagrangian cells, Saha internal
equilibrium, one electron/ion temperature. Vacuum mechanical boundaries;
stationary multigroup radiation and coherent layered Drude wave deposition
(an undercritical single-pass alternative is explicit).
This does not model a solid target, shocks, ablation mass feed, laser refraction,
continuum lowering or two-temperature kinetic relaxation.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from atomx.constants import CODATA, DEFAULT_CODATA_VERSION
from atomx.plasma.laser_absorption import (
    critical_density_per_m3,
    drude_layer_stack,
    drude_slab_fractions,
    electron_ion_collision_frequency,
)

_C = CODATA[DEFAULT_CODATA_VERSION]


def spitzer_face_heat_flux(temperatures_eV, electron_densities_per_m3, effective_charges,
                            cell_widths_m, coulomb_log, flux_limit):
    """Internal electron heat flux W/m², closed ends, explicit saturation limit.

    Classical kappa=1.84e-10 T_K^(5/2)/(Z lnLambda) W/(m K).
    Harmonic face conductivity; |q| <= f n_e kT sqrt(kT/m_e), evaluated
    conservatively on the less permissive neighboring cell. A closure for
    weakly coupled plasma, not a calibrated high-Z transport coefficient.
    """
    T, ne, z, dz = (np.asarray(x, dtype=float) for x in
                    (temperatures_eV, electron_densities_per_m3, effective_charges, cell_widths_m))
    if (T.ndim != 1 or T.shape != ne.shape or T.shape != z.shape or T.shape != dz.shape
            or any(np.any(~np.isfinite(a)) or np.any(a <= 0) for a in (T, ne, z, dz))
            or not np.isfinite(coulomb_log) or coulomb_log <= 0
            or not np.isfinite(flux_limit) or not 0 < flux_limit <= 1):
        raise ValueError("positive finite material data/log and explicit flux limit in (0,1] required")
    K = T * _C["_e"] / _C["_k"]
    kappa = 1.84e-10 * K**2.5 / (z * coulomb_log)
    harmonic = 2 * kappa[:-1] * kappa[1:] / (kappa[:-1] + kappa[1:])
    classical = -harmonic * np.diff(K) / (0.5 * (dz[:-1] + dz[1:]))
    saturation = flux_limit * ne * T * _C["_e"] * np.sqrt(T * _C["_e"] / _C["_me"])
    bound = np.minimum(saturation[:-1], saturation[1:])
    result = np.zeros(T.size + 1)
    result[1:-1] = classical / (1 + np.abs(classical) / bound)
    return result


def evolve_lagrangian_slab(model, cell_widths_m, ion_densities_per_m3, temperatures_eV, *,
                            duration_s, laser_intensity_W_m2, laser_wavelength_m, coulomb_log,
                            conduction_flux_limit, angular_order=24, sample_times_s=None,
                            rtol=2e-7, max_cells=64, max_rhs_evaluations=10000,
                            max_critical_density_ratio=0.8, laser_transport="wave"):
    """Advance material nodes/temperatures and an incident-to-escaped ledger.

    A model is an explicitly sourced MixtureRadiation. The density derivative
    of its EOS is retained: equilibrium ionization changes during compression
    and expansion contribute to internal energy. No post-hoc energy repair.
    """
    dz0, ni0, T0 = (np.array(x, dtype=float) for x in
                   (cell_widths_m, ion_densities_per_m3, temperatures_eV))
    if (dz0.ndim != 1 or dz0.size < 1 or dz0.shape != ni0.shape or dz0.shape != T0.shape
            or any(np.any(~np.isfinite(v)) or np.any(v <= 0) for v in (dz0, ni0, T0))
            or not all(np.isfinite(v) and v > 0 for v in
                       (duration_s, laser_wavelength_m, coulomb_log, rtol, max_critical_density_ratio))):
        raise ValueError("positive cell geometry, material state and source controls required")
    n = dz0.size
    if n > max_cells:
        raise MemoryError("joint source cell cap exceeded; no spatial cells removed")
    if conduction_flux_limit is not None and not 0 < conduction_flux_limit <= 1:
        raise ValueError("explicit conduction limiter in (0,1], or None to disable")
    if laser_transport not in ("wave", "single-pass"):
        raise ValueError("explicit wave or single-pass laser transport required")
    columns = ni0 * dz0
    cell_mass = columns * model.mass_kg
    node_mass = np.concatenate(([cell_mass[0] / 2], (cell_mass[:-1] + cell_mass[1:]) / 2,
                                [cell_mass[-1] / 2]))
    nodes = n + 1
    temperature_start = 2 * nodes
    ledger_start = temperature_start + n
    y0 = np.zeros(ledger_start + 7)
    y0[:nodes] = np.concatenate(([0.], np.cumsum(dz0)))
    y0[temperature_start:ledger_start] = T0
    nc = critical_density_per_m3(laser_wavelength_m)
    band = model.band_weights()
    evaluations = 0

    def material(y):
        widths = np.diff(y[:nodes])
        if np.any(widths <= 0):
            raise ArithmeticError("Lagrangian cells crossed; shock-capable transport required")
        densities = columns / widths
        temps = y[temperature_start:ledger_start]
        cells = [model.cell(T, ni) for T, ni in zip(temps, densities, strict=True)]
        return widths, densities, temps, cells

    def rhs(time_scaled, y):
        nonlocal evaluations
        evaluations += 1
        if evaluations > max_rhs_evaluations:
            raise RuntimeError("spatial source RHS budget exceeded; no partial CE certified")
        widths, densities, temps, cells = material(y)
        eq = [c["equilibrium"] for c in cells]
        ne = np.array([e["electron_density_per_m3"] for e in eq])
        if laser_transport == "single-pass" and np.max(ne / nc) > max_critical_density_ratio:
            raise ValueError("spatial source exceeds declared undercritical laser domain; no density clamp")
        zeff = np.array([e["mean_charge_squared"] / e["mean_charge"] for e in eq])
        collisions = [electron_ion_collision_frequency(d, T, z, coulomb_log)
                      for d, T, z in zip(ne, temps, zeff, strict=True)]
        intensity = float(laser_intensity_W_m2(time_scaled * duration_s))
        if not np.isfinite(intensity) or intensity < 0:
            raise ValueError("incident laser intensity must be finite and nonnegative")
        if laser_transport == "wave":
            laser = drude_layer_stack(ne, collisions, laser_wavelength_m, widths)
            reflection = intensity * laser["reflection"]
            forward = intensity * laser["transmission"]
            absorbed = intensity * laser["absorption_by_cell"]
        else:
            laser = [drude_slab_fractions(d, nu, laser_wavelength_m, w)
                     for d, nu, w in zip(ne, collisions, widths, strict=True)]
            reflection = intensity * laser[0]["reflection"]
            forward = intensity - reflection
            absorbed = np.zeros(n)
            for i in range(n):
                outgoing = forward * np.exp(-laser[i]["attenuation_per_m"] * widths[i])
                absorbed[i] = forward - outgoing
                forward = outgoing
        from atomx.spectrum.slab_transfer import layered_transfer

        transfer = layered_transfer(np.array([c["emissivity_W_m3_bin"] for c in cells]),
                                     np.array([c["absorption_per_m"] for c in cells]), widths,
                                     angular_order=angular_order)
        radiation = transfer["cell_net_loss_W_m2_bin"].sum(axis=1)
        pressures = np.array([e["pressure_Pa"] for e in eq])
        velocities = y[nodes:temperature_start]
        width_rate = np.diff(velocities)
        forces = np.concatenate(([-pressures[0]], -np.diff(pressures), [pressures[-1]]))
        heat_flux = np.zeros(n + 1) if conduction_flux_limit is None else spitzer_face_heat_flux(
            temps, ne, zeff, widths, coulomb_log, conduction_flux_limit)
        cv = np.array([e["heat_capacity_eV_per_eV"] for e in eq])
        density_response = np.array([e["energy_derivative_log_density_eV"] for e in eq])
        Tdot = ((absorbed - radiation - pressures * width_rate - np.diff(heat_flux))
                / (columns * _C["_e"]) + density_response * width_rate / widths) / cv
        left, right = transfer["left_flux_W_m2_bin"], transfer["right_flux_W_m2_bin"]
        return np.concatenate((velocities, forces / node_mass, Tdot,
            [intensity, reflection, forward, absorbed.sum(), left.sum(), right.sum(), band @ (left + right)])) * duration_s

    def internal(y):
        widths = np.diff(y[:nodes])
        if np.any(widths <= 0):
            raise ArithmeticError("Lagrangian cells crossed")
        states = [model.eos.equilibrium(T, ni) for T, ni in
                  zip(y[temperature_start:ledger_start], columns / widths, strict=True)]
        return float(columns @ np.array([e["internal_energy_eV"] for e in states]) * _C["_e"])

    initial_internal = internal(y0)
    times = np.linspace(0, duration_s, 101) if sample_times_s is None else np.asarray(sample_times_s)
    if times.ndim != 1 or times.size < 2 or times[0] != 0 or times[-1] != duration_s or np.any(np.diff(times) <= 0):
        raise ValueError("ordered sample times must span the complete pulse")
    atol = np.concatenate((np.full(nodes, dz0.min() * 1e-9), np.full(nodes, 1e-5),
                           np.full(n, 1e-7), np.full(7, max(initial_internal, 1) * 1e-10)))
    solution = solve_ivp(rhs, (0, 1), y0, method="Radau", t_eval=times / duration_s,
                         rtol=rtol, atol=atol, max_step=.025)
    if not solution.success:
        raise RuntimeError("spatial source integration failed: " + solution.message)
    internal_energy = np.array([internal(column) for column in solution.y.T])
    kinetic = .5 * np.sum(node_mass[:, None] * solution.y[nodes:temperature_start]**2, axis=0)
    ledger = solution.y[ledger_start:]
    defect = initial_internal + ledger[0] - internal_energy - kinetic - ledger[1] - ledger[2] - ledger[4] - ledger[5]
    relative_error = float(np.max(abs(defect) / (initial_internal + ledger[0])))
    if relative_error > max(1e-5, 20 * rtol):
        raise ArithmeticError(f"spatial source energy gate failed: {relative_error}")
    return dict(time_s=solution.t * duration_s, positions_m=solution.y[:nodes],
                velocities_m_s=solution.y[nodes:temperature_start],
                temperatures_eV=solution.y[temperature_start:ledger_start],
                ledger_J_m2=ledger, internal_J_m2=internal_energy, kinetic_J_m2=kinetic,
                energy_balance_defect_J_m2=defect, maximum_relative_energy_defect=relative_error,
                ion_columns_per_m2=columns, source_id=model.source_id, rhs_evaluations=evaluations,
                ledger_rows=("incident", "reflected", "transmitted", "absorbed", "escaped_left",
                             "escaped_right", "escaped_band_both"),
                angular_order=angular_order, conduction_flux_limit=conduction_flux_limit,
                laser_transport=laser_transport,
                closure="one-temperature finite Saha EOS; smooth inviscid Lagrangian plane plasma")
