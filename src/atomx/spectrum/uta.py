"""Spectrum-facing UTA helpers.

This module keeps the heavy-Z spectrum path explicit while reusing the
configuration-average UTA moment implementation in :mod:`atomx.uta`.
"""

from atomx.constants import HC_EV_NM
from atomx.uta import UTAArray, uta_moments, uta_simple_from_levels


def add_uta_array_to_spectrum(
    spectrum,
    lines: list[tuple[float, float]],
    *,
    lam_min_nm: float,
    lam_max_nm: float,
) -> UTAArray | None:
    """Render one unresolved transition array onto a spectrum.

    ``lines`` are ``(transition_energy_eV, strength)`` pairs.  The emitted
    array conserves integrated strength by adding one Gaussian line at the
    UTA centroid with width from the second central moment.
    """
    moments = uta_moments(lines)
    lam_nm = moments.centroid_nm
    if lam_nm < lam_min_nm or lam_nm > lam_max_nm:
        return None
    intensity = moments.total_strength * moments.centroid_eV
    sigma_nm = _uta_sigma_nm(moments.centroid_eV, moments.std_eV)
    spectrum.add_line(lam_nm, intensity, sigma_nm=sigma_nm)
    return moments


def _uta_sigma_nm(centroid_eV: float, std_eV: float) -> float:
    if centroid_eV <= 0 or std_eV <= 0:
        return 0.0
    return HC_EV_NM * std_eV / (centroid_eV ** 2)


__all__ = [
    "UTAArray",
    "add_uta_array_to_spectrum",
    "uta_moments",
    "uta_simple_from_levels",
]
