"""Stationary, homogeneous, isotropic slab transfer in SI units.

No incident EUV, scattering, velocity gradients or radiation energy storage.
The angular integral is exact; a normal-ray transmission is not an escape
probability. Spectral arrays contain bin integrals, never peak-normalized data.
"""

from __future__ import annotations

import numpy as np
from scipy.special import exp1


def slab_escape_factor(optical_depth):
    """Both-face escaped power / optically thin power for a uniform slab.

    beta(tau) = [1/2 - E_3(tau)] / tau. In particular beta(0)=1 and
    beta(tau >> 1)=1/(2 tau). This integrates the full hemisphere, including
    grazing rays. The expression avoids cancellation in the thin limit.
    """
    tau = np.asarray(optical_depth, dtype=float)
    if np.any(~np.isfinite(tau)) or np.any(tau < 0):
        raise ValueError("optical depths must be finite and nonnegative; masers unsupported")
    out = np.ones_like(tau)
    active = tau > 0
    x = tau[active]
    out[active] = 0.5 * (-np.expm1(-x) / x + np.exp(-x) - x * exp1(x))
    return out


def slab_escaped_flux(emissivity_W_m3_bin, absorption_per_m, thickness_m):
    """Power through BOTH faces per unit area in W/m² per spectral bin.

    Emissivity is integrated over 4 pi and over each bin. The flux through
    either face of this symmetric slab is half the result. Absorption includes
    stimulated emission. Reabsorbed photons are not a separate cooling loss.
    """
    emission = np.asarray(emissivity_W_m3_bin, dtype=float)
    absorption = np.asarray(absorption_per_m, dtype=float)
    if emission.shape != absorption.shape:
        raise ValueError("emission and absorption must have identical bin shapes")
    if (not np.isfinite(thickness_m) or thickness_m <= 0
            or np.any(~np.isfinite(emission)) or np.any(emission < 0)):
        raise ValueError("positive thickness and finite nonnegative emission required")
    return emission * thickness_m * slab_escape_factor(absorption * thickness_m)


def layered_transfer(emissivity_W_m3_bin, absorption_per_m, cell_widths_m, *,
                     angular_order=32, incident_left_W_m2_sr_bin=None,
                     incident_right_W_m2_sr_bin=None, memory_cap_bytes=1024**3):
    """Stationary discrete-angle transport through nonuniform plane-parallel cells.

    Arrays are (cells, spectral bins). Each cell has a constant source and net
    absorption; formal cell integration is exact, angular quadrature is explicit.
    Returns both face fluxes, mean intensity and signed material radiative loss
    per cell. A cool foreground can heat, so cell losses are never clipped.
    No scattering, Doppler redistribution, radiation pressure or stored photons.
    """
    from numpy.polynomial.legendre import leggauss

    eps = np.asarray(emissivity_W_m3_bin, dtype=float)
    alpha = np.asarray(absorption_per_m, dtype=float)
    dz = np.asarray(cell_widths_m, dtype=float)
    if (eps.ndim != 2 or eps.size == 0 or alpha.shape != eps.shape or dz.shape != (eps.shape[0],)
            or np.any(~np.isfinite(eps)) or np.any(eps < 0)
            or np.any(~np.isfinite(alpha)) or np.any(alpha < 0)
            or np.any(~np.isfinite(dz)) or np.any(dz <= 0)
            or not isinstance(angular_order, (int, np.integer)) or angular_order < 2):
        raise ValueError("finite nonnegative spectral sources/opacity and positive cell widths required")
    bound = 112 * eps.size + 128 * eps.shape[1]
    if bound > memory_cap_bytes:
        raise MemoryError(f"layered transfer planning bound {bound} exceeds cap; no cells/bins removed")
    bins = eps.shape[1]

    def boundary(value):
        v = np.zeros(bins) if value is None else np.asarray(value, dtype=float)
        if v.shape != (bins,) or np.any(~np.isfinite(v)) or np.any(v < 0):
            raise ValueError("isotropic boundary intensity must be finite nonnegative per bin")
        return v

    left, right = boundary(incident_left_W_m2_sr_bin), boundary(incident_right_W_m2_sr_bin)
    points, weights = leggauss(angular_order)
    mus, weights = (points + 1) / 2, weights / 2
    mean = np.zeros_like(eps)
    loss = np.zeros_like(eps)
    faces = np.zeros((2, bins))
    incident_flux = np.pi * (left + right)
    for mu, weight in zip(mus, weights, strict=True):
        x = alpha * dz[:, None] / mu
        atten = np.exp(-x)
        phi = np.ones_like(x)
        positive = x > 0
        phi[positive] = -np.expm1(-x[positive]) / x[positive]
        # (1-phi)/x, including a transparent cell's linear intensity profile.
        psi = np.empty_like(x)
        small = x < 1e-4
        psi[small] = 0.5 - x[small] / 6 + x[small]**2 / 24 - x[small]**3 / 120
        psi[~small] = (1 - phi[~small]) / x[~small]
        emission_path = eps / (4 * np.pi) * dz[:, None] / mu
        for direction in (1, -1):
            intensity = left.copy() if direction == 1 else right.copy()
            order = range(eps.shape[0]) if direction == 1 else range(eps.shape[0] - 1, -1, -1)
            for cell in order:
                average = intensity * phi[cell] + emission_path[cell] * psi[cell]
                emitted = intensity * atten[cell] + emission_path[cell] * phi[cell]
                mean[cell] += 0.5 * weight * average
                loss[cell] += 2 * np.pi * weight * mu * (emitted - intensity)
                intensity = emitted
            faces[1 if direction == 1 else 0] += 2 * np.pi * weight * mu * intensity
    energy_balance = loss.sum(axis=0) - faces.sum(axis=0) + incident_flux
    return dict(left_flux_W_m2_bin=faces[0], right_flux_W_m2_bin=faces[1],
                mean_intensity_W_m2_sr_bin=mean, cell_net_loss_W_m2_bin=loss,
                energy_balance_defect_W_m2_bin=energy_balance, angular_order=angular_order,
                planning_bound_bytes=bound)


def trace_ray(emissivity_W_m3_bin, absorption_per_m, path_lengths_m, *, incident_W_m2_sr_bin=None):
    """Directional radiance along explicit cell path lengths, W/m²/sr/bin.

    Unlike hemisphere flux, this is suitable for a calibrated spectrometer's
    line of sight. Path lengths already include the viewing geometry; no
    additional cosine factor is inferred. Cells run from far side to observer.
    """
    eps, alpha, lengths = (np.asarray(v, dtype=float) for v in
                            (emissivity_W_m3_bin, absorption_per_m, path_lengths_m))
    if (eps.ndim != 2 or eps.size == 0 or alpha.shape != eps.shape or lengths.shape != (eps.shape[0],)
            or any(np.any(~np.isfinite(v)) for v in (eps, alpha, lengths))
            or np.any(eps < 0) or np.any(alpha < 0) or np.any(lengths <= 0)):
        raise ValueError("positive ray path lengths and finite passive spectral coefficients required")
    intensity = np.zeros(eps.shape[1]) if incident_W_m2_sr_bin is None else np.array(incident_W_m2_sr_bin, dtype=float)
    if intensity.shape != (eps.shape[1],) or np.any(~np.isfinite(intensity)) or np.any(intensity < 0):
        raise ValueError("finite nonnegative incident radiance per bin required")
    for e, a, distance in zip(eps, alpha, lengths, strict=True):
        tau = a * distance
        phi = np.ones_like(tau)
        keep = tau > 0
        phi[keep] = -np.expm1(-tau[keep]) / tau[keep]
        intensity = intensity * np.exp(-tau) + e * distance / (4 * np.pi) * phi
    return intensity
