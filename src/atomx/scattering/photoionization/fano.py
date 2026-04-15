"""Fano (Beutler-Fano) autoionization-resonance profile.

A single isolated autoionizing resonance in the photoionization
continuum is described, after Fano (1961), by

    σ(E) = σ_bg(E) · ρ² · (q + ε)² / (1 + ε²)  +  σ_bg(E) · (1 − ρ²)   (1)

    ε = 2 (E − E_res) / Γ                                             (2)

where

    * ``E_res`` — resonance central energy,
    * ``Γ``     — resonance total width (Auger + radiative),
    * ``q``     — Fano asymmetry parameter (dimensionless; q = ∞ gives
                  a Lorentzian peak, q = 0 a symmetric "window"
                  profile, q ≈ 1 a classic asymmetric shape),
    * ``ρ²``    — fraction of the continuum channels coupled to the
                  resonance (0 ≤ ρ² ≤ 1; ρ² = 1 recovers the single-
                  channel limit, often assumed in first analyses).

The full cross section is the sum of independent resonance
contributions on the smooth background provided by e.g.
:func:`atomx.numerics.bspline.continuum.photoionization_cross_section`.

Parameter conventions
---------------------
* Energies (E, E_res, Γ) in the same unit, which becomes the unit for
  the returned cross-section grid.  Any consistent choice works.
* q may be positive or negative; its magnitude controls asymmetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class FanoResonance:
    """Parameters of a single Fano autoionization resonance.

    Attributes
    ----------
    E_res_eV : float
        Resonance energy (eV).
    gamma_eV : float
        Full total width Γ (eV) — same as ``Γ_Auger + Γ_radiative``.
    q : float
        Fano asymmetry parameter.
    rho_sq : float, default 1.0
        Fraction of the continuum coupled to the resonance (0 ≤ ρ² ≤ 1).
    label : str, optional
        Informational label.
    """

    E_res_eV: float
    gamma_eV: float
    q: float
    rho_sq: float = 1.0
    label: str = ""

    def __post_init__(self):
        if self.gamma_eV <= 0:
            raise ValueError(f"gamma must be positive; got {self.gamma_eV}.")
        if not (0.0 <= self.rho_sq <= 1.0):
            raise ValueError(f"rho_sq must be in [0, 1]; got {self.rho_sq}.")


# ---------------------------------------------------------------------------
# Dimensionless Fano factor
# ---------------------------------------------------------------------------

def fano_profile(
    E_eV,
    E_res_eV: float,
    gamma_eV: float,
    q: float,
    rho_sq: float = 1.0,
) -> np.ndarray:
    """Multiplicative Fano factor ``F(E) = σ/σ_bg`` on an energy grid.

    Implements eq. (1) in the module docstring — the multiplicative
    "ρ² · (q + ε)² / (1 + ε²) + (1 − ρ²)" factor to be applied to a
    non-resonant background.

    Parameters
    ----------
    E_eV : array_like
        Photon energies (eV) — same shape as the background array.
    E_res_eV : float
        Resonance centre (eV).
    gamma_eV : float
        Total width Γ (eV).
    q : float
        Fano asymmetry parameter.
    rho_sq : float, default 1.0
        Continuum-coupling strength.

    Returns
    -------
    np.ndarray
        Same shape as ``E_eV``; multiply element-wise by the
        background cross-section.
    """
    if gamma_eV <= 0:
        raise ValueError(f"gamma must be positive; got {gamma_eV}.")
    if not (0.0 <= rho_sq <= 1.0):
        raise ValueError(f"rho_sq must be in [0, 1]; got {rho_sq}.")

    E = np.asarray(E_eV, dtype=float)
    eps = 2.0 * (E - E_res_eV) / gamma_eV
    fano = (q + eps) ** 2 / (1.0 + eps ** 2)
    return rho_sq * fano + (1.0 - rho_sq)


# ---------------------------------------------------------------------------
# Apply a list of resonances to a background cross section
# ---------------------------------------------------------------------------

def apply_resonances(
    sigma_bg: np.ndarray,
    E_eV: np.ndarray,
    resonances: Sequence[FanoResonance],
) -> np.ndarray:
    """Overlay a list of Fano resonances on a background cross section.

    Resonances are applied **multiplicatively** (independent-resonance
    approximation).  For strongly overlapping resonances a full
    multi-channel Fano treatment is required (not implemented here —
    noted as a TODO for a future release).

    Parameters
    ----------
    sigma_bg : np.ndarray
        Non-resonant (background) photoionization cross section on the
        energy grid ``E_eV``.  Any units are fine (Mb, a₀², cm², …);
        the returned array preserves them.
    E_eV : np.ndarray
        Photon energies (eV), same length as ``sigma_bg``.
    resonances : iterable of :class:`FanoResonance`
        List of resonances to overlay.

    Returns
    -------
    np.ndarray
        Decorated cross section σ(E), same shape and units as
        ``sigma_bg``.
    """
    sigma_bg = np.asarray(sigma_bg, dtype=float)
    E_eV = np.asarray(E_eV, dtype=float)
    if sigma_bg.shape != E_eV.shape:
        raise ValueError(
            f"sigma_bg and E_eV must have matching shapes; "
            f"got {sigma_bg.shape} vs {E_eV.shape}."
        )

    sigma = sigma_bg.copy()
    for r in resonances:
        factor = fano_profile(
            E_eV, r.E_res_eV, r.gamma_eV, r.q, rho_sq=r.rho_sq,
        )
        sigma = sigma * factor
    return sigma
