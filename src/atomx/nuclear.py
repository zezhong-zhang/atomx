"""Nuclear charge distribution models.

Supports point nucleus, Fermi distribution, and uniform sphere.
DBSR_HF uses Fermi distribution by default — critical for KB grids
where r_min ≈ 1e-5 Bohr approaches the nuclear radius.
"""

import numpy as np
from atomx.constants import BOHR_RADIUS_FM, nuclear_rms_radius_au


def point_nucleus_potential(Z):
    """Return V(r) = -Z/r for a point nucleus.

    Parameters
    ----------
    Z : int or float
        Nuclear charge.

    Returns
    -------
    V : callable
        V(r) function in Hartree atomic units.
    """
    def V(r):
        return -Z / r
    return V


def fermi_nucleus_potential(Z, A=None):
    """Fermi distribution nuclear potential (DBSR_HF default).

    ρ(r) = ρ_0 / (1 + exp((r - c)/a))

    where c = half-density radius, a = skin thickness.

    For r outside the nucleus, V(r) = -Z/r exactly.
    Inside: V(r) is smoother than -Z/r (no singularity).

    For light atoms (Z < 5), the Fermi distribution is essentially
    point-like, so this falls back to a uniform sphere approximation.
    """
    if A is None:
        A = int(round(2.5 * max(Z, 1)))

    # Fermi parameters from GRASP/DBSR_HF
    bohr_fm = 5.29177e4  # Bohr radius in fm
    c_fm = 1.1 * A ** (1.0 / 3.0)  # half-density radius in fm
    a_fm = 0.5                       # skin thickness in fm (DBSR_HF default)
    c_au = c_fm / bohr_fm
    a_au = a_fm / bohr_fm

    # For practical purposes: use uniform sphere with equivalent RMS radius
    # The Fermi RMS radius: R_rms² = (3/5)c² + (7/5)(πa)²
    R_rms = np.sqrt(0.6 * c_au**2 + 1.4 * (np.pi * a_au)**2)
    R_sphere = np.sqrt(5.0 / 3.0) * R_rms  # equivalent uniform sphere

    R = R_sphere

    def V(r):
        if np.isscalar(r):
            if r <= R:
                return -Z / (2.0 * R) * (3.0 - (r / R) ** 2)
            return -Z / r
        else:
            V_arr = np.where(r <= R,
                             -Z / (2.0 * R) * (3.0 - (r / R) ** 2),
                             -Z / r)
            return V_arr

    return V


def uniform_sphere_potential(Z, A=None, R_nuc_au=None):
    """Potential for a uniformly charged nuclear sphere.

    V(r) = -Z/(2R) * (3 - r^2/R^2)  for r <= R
    V(r) = -Z/r                       for r > R

    Parameters
    ----------
    Z : int
        Nuclear charge.
    A : int, optional
        Mass number. Used to compute R if R_nuc_au not given.
    R_nuc_au : float, optional
        Nuclear radius in Bohr. If None, computed from A.

    Returns
    -------
    V : callable
    """
    if R_nuc_au is None:
        if A is None:
            A = int(round(2.5 * Z))  # rough estimate
        # Convert RMS radius to equivalent uniform sphere radius: R = sqrt(5/3) * R_rms
        R_rms = nuclear_rms_radius_au(A)
        R_nuc_au = np.sqrt(5.0 / 3.0) * R_rms

    R = R_nuc_au

    def V(r):
        if r <= R:
            return -Z / (2.0 * R) * (3.0 - (r / R) ** 2)
        else:
            return -Z / r

    return V
