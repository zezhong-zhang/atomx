"""Linear collisional Drude deposition in a slab or plane-parallel layer stack.

The single-pass slab omits rear-face reflection; the coherent wave stack retains
all interfaces and permits critical/overdense cells. Neither supplies oblique
ray refraction, non-Maxwellian or strong-field corrections.
The Coulomb logarithm is an explicit closure input, not a fitted CE parameter.
"""

from __future__ import annotations

import numpy as np

from atomx.constants import CODATA, DEFAULT_CODATA_VERSION

_C = CODATA[DEFAULT_CODATA_VERSION]
_EPS0 = 1 / (_C["_mu0"] * _C["_c"]**2)


def critical_density_per_m3(wavelength_m):
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("wavelength must be finite and positive")
    omega = 2 * np.pi * _C["_c"] / wavelength_m
    return _EPS0 * _C["_me"] * omega**2 / _C["_e"]**2


def electron_ion_collision_frequency(ne_per_m3, temperature_eV, charge, coulomb_log):
    """Classical Maxwellian electron-ion momentum collision frequency, s^-1.

    4 sqrt(2 pi)/3 * n_e Z (e²/4 pi eps0)² lnLambda / [sqrt(m_e) (kT)^1.5].
    Classical weak-coupling validity must be assessed separately by the caller.
    """
    if not all(np.isfinite(x) and x > 0
               for x in (ne_per_m3, temperature_eV, charge, coulomb_log)):
        raise ValueError("positive finite density, temperature, charge and Coulomb logarithm required")
    coulomb = _C["_e"]**2 / (4 * np.pi * _EPS0)
    return (4 * np.sqrt(2 * np.pi) / 3 * ne_per_m3 * charge * coulomb**2
            * coulomb_log / (np.sqrt(_C["_me"]) * (temperature_eV * _C["_e"])**1.5))


def drude_slab_fractions(ne_per_m3, collision_frequency_per_s, wavelength_m, thickness_m):
    """Return absorption, reflection, transmission and intensity attenuation.

    The passive square-root branch of eps=1-wp²/[w(w+i nu)] gives alpha=2 Im(k).
    A+R+T=1. An overdense homogeneous slab is mathematically defined here,
    but the coupled source workflow separately rejects an inadmissible corona.
    """
    if not all(np.isfinite(x) and x >= 0 for x in (ne_per_m3, collision_frequency_per_s)):
        raise ValueError("density and collision frequency must be finite and nonnegative")
    if not np.isfinite(thickness_m) or thickness_m <= 0:
        raise ValueError("thickness must be finite and positive")
    nc = critical_density_per_m3(wavelength_m)
    omega = 2 * np.pi * _C["_c"] / wavelength_m
    index = np.sqrt(complex(1 - (ne_per_m3 / nc) / (1 + 1j * collision_frequency_per_s / omega)))
    reflection = float(abs((1 - index) / (1 + index))**2)
    attenuation = float(2 * omega * index.imag / _C["_c"])
    absorption = float((1 - reflection) * -np.expm1(-attenuation * thickness_m))
    transmission = float((1 - reflection) * np.exp(-attenuation * thickness_m))
    return dict(absorption=absorption, reflection=reflection, transmission=transmission,
                attenuation_per_m=attenuation, critical_density_ratio=ne_per_m3 / nc)


def drude_layer_stack(ne_per_m3, collision_frequencies_per_s, wavelength_m, cell_widths_m):
    """Normal-incidence coherent wave deposition, including critical/overdense cells.

    Vacuum on both sides. Stable reflection recursion and decaying propagation
    factors avoid the exponentially growing entries of transfer matrices.
    Poynting-flux differences give the individual cell absorption; no critical
    density clamp, imposed absorption fraction or skin-layer guess is applied.
    The supplied local Drude dielectric remains a collisional constitutive model,
    not a treatment of nonlocal/strong-field or oblique laser physics.
    """
    ne, nu, dz = (np.asarray(v, dtype=float) for v in
                  (ne_per_m3, collision_frequencies_per_s, cell_widths_m))
    if (ne.ndim != 1 or ne.size < 1 or ne.shape != nu.shape or ne.shape != dz.shape
            or any(np.any(~np.isfinite(v)) for v in (ne, nu, dz))
            or np.any(ne < 0) or np.any(nu < 0) or np.any(dz <= 0)):
        raise ValueError("nonnegative density/collision rates and positive matched layer widths required")
    nc = critical_density_per_m3(wavelength_m)
    omega = 2 * np.pi * _C["_c"] / wavelength_m
    interior = np.sqrt(1 - (ne / nc).astype(complex) / (1 + 1j * nu / omega))
    # An exactly zero dielectric has a singular traveling-wave representation.
    if np.any(abs(interior) < 1e-14):
        raise ValueError("exact lossless critical layer requires a non-traveling-wave representation")
    indices = np.concatenate(([1 + 0j], interior, [1 + 0j]))
    reflection_interface = (indices[:-1] - indices[1:]) / (indices[:-1] + indices[1:])
    phase = np.exp(1j * omega / _C["_c"] * interior * dz)
    reflected_at_left = np.zeros(ne.size + 1, dtype=complex)
    reflected_at_right = np.zeros(ne.size, dtype=complex)
    for j in range(ne.size - 1, -1, -1):
        r = reflection_interface[j + 1]
        downstream = reflected_at_left[j + 1]
        reflected_at_right[j] = (r + downstream) / (1 + r * downstream)
        reflected_at_left[j] = reflected_at_right[j] * phase[j]**2
    r0 = reflection_interface[0]
    reflected = (r0 + reflected_at_left[0]) / (1 + r0 * reflected_at_left[0])
    forward_right = 1 + 0j
    absorption = np.zeros(ne.size)
    fields_left, fields_right = [], []

    def flux(E, H):
        return float(np.real(E * np.conjugate(H)))

    for j, index in enumerate(interior):
        left_index = indices[j]
        amplitude = (2 * left_index / (left_index + index) * forward_right
                     / (1 + reflection_interface[j] * reflected_at_left[j]))
        back_left = reflected_at_left[j] * amplitude
        forward_right = amplitude * phase[j]
        back_right = reflected_at_right[j] * forward_right
        E_left, H_left = amplitude + back_left, index * (amplitude - back_left)
        E_right, H_right = forward_right + back_right, index * (forward_right - back_right)
        absorption[j] = flux(E_left, H_left) - flux(E_right, H_right)
        fields_left.append(E_left)
        fields_right.append(E_right)
    last = indices[-2]
    transmitted = 2 * last / (last + 1) * forward_right
    R, T = float(abs(reflected)**2), float(abs(transmitted)**2)
    raw_defect = float(1 - R - T - absorption.sum())
    if absorption.min(initial=0) < -1e-12 or abs(raw_defect) > 2e-11:
        raise ArithmeticError("layered wave deposition failed passivity/energy conservation")
    # Remove only explicitly bounded Poynting subtraction roundoff.
    absorption = np.maximum(absorption, 0)
    return dict(absorption_by_cell=absorption, reflection=R, transmission=T,
                complex_reflection_amplitude=reflected, complex_transmission_amplitude=transmitted,
                electric_field_left=np.array(fields_left), electric_field_right=np.array(fields_right),
                critical_density_ratio=ne / nc,
                energy_balance_defect=float(1 - R - T - absorption.sum()))
