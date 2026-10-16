"""Sparse population propagation with explicitly supplied physical rates.

This module supplies no guessed excitation, ionization or recombination rates.
Columns are source states. Inter-charge edges are allowed; callers must supply
matched state energies, electron balance and energy exchange for a coupled EOS.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import expm_multiply

from atomx.constants import BOLTZMANN_EV
from atomx.plasma.crm import effective_collision_strength


def population_generator(n_levels, source, destination, rates_per_s):
    """Construct a conservative CSR generator, adding repeated physical edges."""
    src = np.asarray(source)
    dst = np.asarray(destination)
    rates = np.asarray(rates_per_s, dtype=float)
    if (not isinstance(n_levels, (int, np.integer)) or n_levels < 1
            or src.ndim != 1 or src.shape != dst.shape or src.shape != rates.shape
            or src.dtype.kind not in "iu" or dst.dtype.kind not in "iu"
            or np.any(src < 0) or np.any(dst < 0)
            or np.any(src >= n_levels) or np.any(dst >= n_levels)
            or np.any(src == dst) or np.any(~np.isfinite(rates)) or np.any(rates < 0)):
        raise ValueError("invalid population edges, indices or rates")
    off = coo_matrix((rates, (dst, src)), shape=(n_levels, n_levels)).tocsr()
    return off - diags(np.asarray(off.sum(axis=0)).ravel(), format="csr")


def advance_populations(generator, populations, duration_s):
    """Constant-rate exponential propagation; no clipping or renormalization.

    Positivity and particle conservation are checked, with roundoff reported
    through the returned values rather than hidden by a repair.
    """
    p = np.asarray(populations, dtype=float)
    if (p.ndim != 1 or generator.shape != (p.size, p.size)
            or np.any(~np.isfinite(p)) or np.any(p < 0)
            or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)
            or not np.isfinite(duration_s) or duration_s < 0):
        raise ValueError("invalid populations or duration")
    coo = generator.tocoo()
    if (np.any(~np.isfinite(coo.data))
            or np.any(coo.data[coo.row != coo.col] < 0)
            or np.max(np.abs(np.asarray(generator.sum(axis=0))))
            > 1e-12 * max(1.0, np.max(np.abs(coo.data), initial=0.0))):
        raise ValueError("matrix must be a conservative population generator")
    out = expm_multiply(generator * duration_s, p, traceA=generator.diagonal().sum() * duration_s)
    if out.min() < -1e-12 or abs(out.sum() - p.sum()) > 1e-10:
        raise ArithmeticError("population propagation failed conservation/positivity gate")
    return out


def omega_collision_pair(incident_energy_eV, omega, gap_eV, temperature_eV, g_lower, g_upper):
    """Up/down electron-impact coefficients in cm³/s from an explicit Omega table.

    Reuses the CRM's incident-energy convention and piecewise-linear Maxwell
    average. De-excitation survives excitation Boltzmann underflow. No gf-based
    approximation is substituted for a missing collision-strength table.
    """
    E = np.asarray(incident_energy_eV, dtype=float)
    om = np.asarray(omega, dtype=float)
    if (E.ndim != 1 or E.size < 2 or E.shape != om.shape
            or np.any(~np.isfinite(E)) or np.any(~np.isfinite(om))
            or np.any(np.diff(E) <= 0) or E[0] < gap_eV or np.any(om < 0)
            or not all(np.isfinite(x) and x > 0
                       for x in (gap_eV, temperature_eV, g_lower, g_upper))):
        raise ValueError("ordered incident-energy Omega table above a positive threshold required")
    ups = effective_collision_strength(E, om, gap_eV, temperature_eV)
    base = 8.6291e-6 * ups / np.sqrt(temperature_eV / BOLTZMANN_EV)
    return base / g_lower * np.exp(-gap_eV / temperature_eV), base / g_upper
