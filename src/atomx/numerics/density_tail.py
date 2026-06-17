"""Post-processing for numerically noisy radial density tails."""

from __future__ import annotations

import numpy as np


def smooth_exponential_tail(
    r,
    rho,
    *,
    noise_floor: float = 1.0e-18,
    min_fit_points: int = 8,
    max_fit_points: int = 32,
    output_cutoff: float = 1.0e-300,
) -> np.ndarray:
    """Replace a numerically noisy density tail with a fitted exponential.

    Dirac-Fock B-spline densities are well behaved through the physical tail,
    then some elements hit a solver noise floor and start oscillating at
    densities that are irrelevant to integrals but damaging to log-space fits.
    This routine detects the first below-floor tail upturn and extrapolates
    ``log(rho) = a*r + b`` from the preceding clean segment.

    Clean monotone tails are returned unchanged.
    """
    r_arr = np.asarray(r, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    out = rho_arr.copy()

    too_short = r_arr.size < min_fit_points + 2
    if r_arr.shape != rho_arr.shape or r_arr.ndim != 1 or too_short:
        return out
    if noise_floor <= 0.0 or not np.isfinite(noise_floor):
        return out

    finite_positive = np.isfinite(r_arr) & np.isfinite(rho_arr) & (rho_arr > 0.0)
    if finite_positive.sum() < min_fit_points + 2:
        return out

    peak_idx = int(np.nanargmax(np.where(finite_positive, rho_arr, -np.inf)))
    if peak_idx >= r_arr.size - min_fit_points - 1:
        return out

    tail_start = peak_idx + 1
    tail = np.arange(tail_start, r_arr.size)
    below_floor = tail[finite_positive[tail] & (rho_arr[tail] <= noise_floor)]
    if below_floor.size == 0:
        return out

    # Once a below-floor upturn is seen, replace from the start of that
    # below-floor run. B-spline noise commonly appears as alternating up/down
    # wiggles once the true density is gone.
    noisy_idx = None
    for idx in below_floor:
        if idx == 0 or not finite_positive[idx - 1]:
            continue
        if rho_arr[idx] > rho_arr[idx - 1]:
            noisy_idx = int(below_floor[0])
            break
    if noisy_idx is None:
        return out

    candidates = np.arange(tail_start, noisy_idx)
    candidates = candidates[
        finite_positive[candidates] & (r_arr[candidates] < r_arr[noisy_idx])
    ]
    if candidates.size < min_fit_points:
        return out

    fit_idx = candidates[-max_fit_points:]
    if fit_idx.size < min_fit_points:
        return out

    x_fit = r_arr[fit_idx]
    y_fit = np.log(rho_arr[fit_idx])
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope >= 0.0:
        return out

    replace = np.arange(noisy_idx, r_arr.size)
    replace = replace[
        np.isfinite(r_arr[replace]) & (r_arr[replace] >= r_arr[noisy_idx])
    ]
    if replace.size == 0:
        return out

    log_floor = np.log(output_cutoff) if output_cutoff > 0.0 else -np.inf
    log_tail = slope * r_arr[replace] + intercept
    if np.isfinite(log_floor):
        log_tail = np.maximum(log_tail, log_floor)
    tail_values = np.exp(log_tail)

    # Do not let the fitted splice introduce a visible upward jump. If the
    # unconstrained fit starts too high, shift the whole exponential down so
    # the log-linear slope is preserved.
    prev = out[noisy_idx - 1]
    if np.isfinite(prev) and prev > 0.0 and tail_values[0] > prev:
        tail_values = tail_values * (prev / tail_values[0])

    out[replace] = tail_values
    return out
