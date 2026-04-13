"""Nuclear charge distribution models.

Supports point nucleus and finite-size models (Fermi, uniform sphere).
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
