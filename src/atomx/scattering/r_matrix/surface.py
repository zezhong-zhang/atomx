"""Surface matching: R-matrix at the boundary -> phase shift -> cross section.

Outside ``r > a`` the single-channel radial function for a short-range
potential is a linear combination of the regular and irregular
Riccati-Bessel functions::

    u(r) = N [cos(δ) S_l(kr) + sin(δ) C_l(kr)],
    S_l(x) = x j_l(x),    C_l(x) = -x y_l(x),

with ``u(x→∞) ~ sin(kr - lπ/2 + δ)``. Imposing the R-matrix matching
condition at ``r = a`` with Bloch constant ``b = 0``
(``u(a) = a · u'(a) · R(E)``) gives

    tan(δ_l) = [S_l(ka) − k a · R(E) · S_l'(ka)]
              / [k a · R(E) · C_l'(ka) − C_l(ka)],

where the primes denote derivatives with respect to the argument
``x = kr``. This matches Burke (2011), eq. (3.1.74), specialised to one
open channel with no Coulomb tail.
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def r_matrix_from_eigenpairs(
    eigenvalues_au: np.ndarray,
    surface_amplitudes: np.ndarray,
    a_au: float,
    energies_au: np.ndarray,
) -> np.ndarray:
    """Spectral sum ``R(E) = (1/2a) Σ_n w_n² / (E_n − E)``.

    The result has the same shape as ``energies_au``.
    """
    diff = eigenvalues_au[None, :] - energies_au[:, None]
    return (0.5 / a_au) * np.sum((surface_amplitudes ** 2)[None, :] / diff, axis=1)


def _riccati_bessel(l: int, x: np.ndarray):
    """Riccati-Bessel ``S_l, C_l`` and their argument-derivatives ``S'_l, C'_l``."""
    j = spherical_jn(l, x)
    y = spherical_yn(l, x)
    jp = spherical_jn(l, x, derivative=True)
    yp = spherical_yn(l, x, derivative=True)
    S = x * j
    C = -x * y
    Sp = j + x * jp
    Cp = -(y + x * yp)
    return S, C, Sp, Cp


def phase_shift_from_R(
    R_E: np.ndarray, energies_au: np.ndarray, a_au: float, l: int
) -> tuple[np.ndarray, np.ndarray]:
    """Single-channel phase shift ``δ_l(E)`` from the boundary R-matrix.

    Returns ``(delta_rad, k_au)`` with ``δ_l`` wrapped to ``(−π/2, π/2]``
    via ``arctan`` — the principal Wigner phase shift, sufficient for the
    unitarity-bounded cross-section formula used here.
    """
    k = np.sqrt(2.0 * energies_au)
    x = k * a_au
    S, C, Sp, Cp = _riccati_bessel(l, x)
    rak = R_E * a_au * k
    num = S - rak * Sp
    den = rak * Cp - C
    delta = np.arctan2(num, den)
    # Fold into (−π/2, π/2]; arctan2 spans (−π, π] but the matching is
    # tan-valued, so a ±π flip is unphysical here.
    delta = np.where(delta > np.pi / 2.0, delta - np.pi, delta)
    delta = np.where(delta <= -np.pi / 2.0, delta + np.pi, delta)
    return delta, k


def partial_wave_cross_section_au(
    delta_rad: np.ndarray, k_au: np.ndarray, l: int
) -> np.ndarray:
    """``σ_l(E) = (4π/k²)(2l+1) sin²(δ_l)`` in units of ``a₀²``."""
    return (4.0 * np.pi / (k_au * k_au)) * (2 * l + 1) * np.sin(delta_rad) ** 2
