"""Elementary line-profile functions (Gaussian, Lorentzian, Voigt).

All profiles are normalised so that ``∫ f(ν) dν = 1``, with the
frequency axis in whatever unit (Hz, MHz, cm⁻¹, …) the caller
uses consistently for both the argument ``x`` and the widths.

References
----------
F. Schreier, *JQSRT* **48**, 743 (1992) — Voigt profile numerics.
D. W. Posener, *Aust. J. Phys.* **12**, 184 (1959) — Voigt convolution.
H. R. Griem, *Plasma Spectroscopy* (McGraw-Hill, 1964) — physical
    origins of each broadening mechanism.
"""

from __future__ import annotations

from math import log, pi, sqrt

import numpy as np
from scipy.special import voigt_profile

# Avogadro × m_u × c² in Joules per kg × ... actually we just need:
# Doppler σ / ν₀  =  sqrt(k T / (M c²))
_KB_J_PER_K = 1.380649e-23          # Boltzmann, J/K (exact, CODATA 2019)
_C_M_PER_S = 299792458.0            # speed of light, m/s (exact)
_U_KG = 1.66053906892e-27           # atomic mass unit, kg (CODATA 2022)


# ---------------------------------------------------------------------------
# Gaussian (Doppler)
# ---------------------------------------------------------------------------

def gaussian(x, x0: float, sigma: float) -> np.ndarray:
    """Normalised Gaussian with standard deviation ``σ``.

    ``f(x) = (1 / (σ √(2π))) · exp(-(x−x₀)²/(2σ²))``.
    Units: whatever the caller uses consistently for ``x``, ``x0`` and
    ``sigma``.
    """
    x = np.asarray(x, dtype=float)
    z = (x - x0) / sigma
    return np.exp(-0.5 * z * z) / (sigma * sqrt(2.0 * pi))


def doppler_width(
    nu0: float,
    T_kelvin: float,
    mass_amu: float,
) -> float:
    """Doppler Gaussian σ (1 σ) from temperature and atomic mass.

        σ = ν₀ · √(k_B T / (M c²)).

    Returns the width in the same frequency unit as ``nu0``
    (Hz, MHz, cm⁻¹, …).

    Parameters
    ----------
    nu0
        Rest-frame line frequency (any consistent unit).
    T_kelvin
        Temperature in Kelvin.
    mass_amu
        Atomic mass in unified atomic-mass units.
    """
    if T_kelvin <= 0 or mass_amu <= 0:
        raise ValueError("T and mass must both be positive.")
    # dimensionless  v/c:
    v_over_c = sqrt(_KB_J_PER_K * T_kelvin / (mass_amu * _U_KG * _C_M_PER_S ** 2))
    return nu0 * v_over_c


# ---------------------------------------------------------------------------
# Lorentzian (natural / pressure)
# ---------------------------------------------------------------------------

def lorentzian(x, x0: float, gamma: float) -> np.ndarray:
    """Normalised Lorentzian with half-width-at-half-maximum ``γ``.

    ``f(x) = (γ / π) / ((x − x₀)² + γ²)``.
    """
    x = np.asarray(x, dtype=float)
    d = x - x0
    return gamma / (pi * (d * d + gamma * gamma))


def natural_width(A_einstein: float) -> float:
    """Lorentzian HWHM ``Γ = A / (4π)`` from the Einstein A coefficient.

    The full Lorentzian FWHM is ``Γ_full = A / (2π)``; the returned
    HWHM value is what :func:`lorentzian` expects.

    Units: ``A`` in frequency units (same as ``x``) → ``Γ`` in the same.
    """
    return A_einstein / (4.0 * pi)


# ---------------------------------------------------------------------------
# Voigt = convolution(Gaussian, Lorentzian)
# ---------------------------------------------------------------------------

def voigt(x, x0: float, sigma: float, gamma: float) -> np.ndarray:
    """Normalised Voigt profile with Gaussian σ and Lorentzian γ (HWHM).

    ``V(x) = ∫ G(u; σ) L(x − u; γ) du`` with both factors unit-area
    normalised.  Evaluated via :func:`scipy.special.voigt_profile`.
    """
    return voigt_profile(np.asarray(x, dtype=float) - x0, sigma, gamma)


# ---------------------------------------------------------------------------
# FWHM helpers
# ---------------------------------------------------------------------------

def gaussian_fwhm(sigma: float) -> float:
    """Gaussian FWHM from standard deviation:  2√(2 ln 2) · σ ≈ 2.3548 σ."""
    return 2.0 * sqrt(2.0 * log(2.0)) * sigma


def lorentzian_fwhm(gamma: float) -> float:
    """Lorentzian FWHM from HWHM:  2 · γ."""
    return 2.0 * gamma


def voigt_fwhm(sigma: float, gamma: float) -> float:
    """Olivero–Longbothum (1977) approximation to the Voigt FWHM.

        f_V  ≈  0.5346 · f_L  +  √(0.2166 · f_L² + f_G²)

    where ``f_L = 2γ`` and ``f_G = 2√(2 ln 2) σ`` are the Lorentzian
    and Gaussian FWHMs.  Accurate to ~0.02 %.
    """
    fL = lorentzian_fwhm(gamma)
    fG = gaussian_fwhm(sigma)
    return 0.5346 * fL + sqrt(0.2166 * fL * fL + fG * fG)
