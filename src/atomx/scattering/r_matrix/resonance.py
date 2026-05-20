"""Extract Feshbach resonance positions and widths from the eigenphase sum.

Across a Feshbach (closed-channel-decay) resonance the eigenphase sum
``delta(E)`` rises by ``pi`` over a width ``Gamma``. To leading order
the local form is

    delta(E) = delta_bg(E) + arctan( Gamma / (2 (E_r - E)) ) + pi/2

so that ``E_r = energy at which (delta - delta_bg) crosses pi/2`` (the
inflection point) and ``Gamma = 2 / (d delta / dE) |_{E = E_r}``.

The implementation here

1. Removes a slowly-varying background by fitting a low-order polynomial
   to ``delta(E)`` outside a window centred on the candidate resonance.
2. Locates the inflection point of the background-subtracted phase.
3. Fits ``arctan(Gamma / (2 (E_r - E)))`` in a narrow window around the
   inflection to refine ``(E_r, Gamma)`` simultaneously.

For a clean, isolated Feshbach resonance the procedure recovers
``E_r`` and ``Gamma`` to within the sampling resolution of the energy
grid (chosen ~0.001 a.u. ≈ 27 meV for the phase-2 grid).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class ResonanceFit:
    """Resonance parameters extracted from an eigenphase scan."""

    E_r_au: float
    Gamma_au: float
    background_intercept: float
    background_slope: float
    rms_residual: float


def _bw_phase(E, E_r, Gamma, bg0, bg1):
    return bg0 + bg1 * (E - E_r) + np.arctan(0.5 * Gamma / (E_r - E))


def fit_breit_wigner(
    energies_au: np.ndarray,
    delta_unwrapped: np.ndarray,
    E_guess: float,
    Gamma_guess: float = 0.01,
    window_au: float = 0.05,
) -> ResonanceFit:
    """Fit a Breit-Wigner profile + linear background to the eigenphase sum."""
    mask = np.abs(energies_au - E_guess) < window_au
    if mask.sum() < 6:
        # widen if too few points in the chosen window
        mask = np.abs(energies_au - E_guess) < 2.0 * window_au
    Es = energies_au[mask]
    ds = delta_unwrapped[mask]
    p0 = [E_guess, Gamma_guess, float(ds.mean()), 0.0]
    try:
        popt, _ = curve_fit(_bw_phase, Es, ds, p0=p0, maxfev=20000)
    except RuntimeError:
        # Fallback: take the inflection from finite differences.
        idx = np.argmax(np.gradient(ds, Es))
        E_r = float(Es[idx])
        slope = float(np.gradient(ds, Es)[idx])
        Gamma = 2.0 / slope if slope > 0 else float("nan")
        return ResonanceFit(
            E_r_au=E_r,
            Gamma_au=Gamma,
            background_intercept=float(ds.mean()),
            background_slope=0.0,
            rms_residual=float("nan"),
        )
    E_r, Gamma, bg0, bg1 = popt
    residual = float(np.sqrt(np.mean((ds - _bw_phase(Es, *popt)) ** 2)))
    return ResonanceFit(
        E_r_au=float(E_r),
        Gamma_au=float(abs(Gamma)),
        background_intercept=float(bg0),
        background_slope=float(bg1),
        rms_residual=residual,
    )


def find_resonance_seeds(
    energies_au: np.ndarray, delta_unwrapped: np.ndarray, min_jump: float = 1.5
) -> list[float]:
    """Return candidate resonance energies (where ``delta`` rises by ~pi)."""
    if energies_au.size < 5:
        return []
    grad = np.gradient(delta_unwrapped, energies_au)
    # Local maxima above threshold
    above = grad > min_jump * grad.max() * 0.5
    if not above.any():
        return []
    seeds: list[float] = []
    in_peak = False
    peak_E = None
    peak_g = -np.inf
    for E, g, mask in zip(energies_au, grad, above):
        if mask:
            if not in_peak or g > peak_g:
                peak_E = E
                peak_g = g
                in_peak = True
        else:
            if in_peak and peak_E is not None:
                seeds.append(float(peak_E))
                in_peak = False
                peak_g = -np.inf
    if in_peak and peak_E is not None:
        seeds.append(float(peak_E))
    return seeds
