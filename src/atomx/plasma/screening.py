"""Plasma screening models for dense plasma environments.

Modifies the nuclear/electron potential to account for plasma effects:
- Debye-Huckel: weak coupling (low density, high temperature)
- Ion-sphere: strong coupling (high density)
- Stewart-Pyatt: smooth interpolation between the two limits

These models return potential functions or modification matrices that
slot into the existing Dirac equation solver.

Reference:
- Stewart & Pyatt, ApJ 144, 1203 (1966)
- Gu, FAC manual - plasma screening modes
"""

import numpy as np

from atomx.constants import BOHR_RADIUS_M, HARTREE_EV
from atomx.numerics.bspline.integrals import _assemble_integral

# Conversion factors
CM_TO_BOHR = 1.0 / (BOHR_RADIUS_M * 100)  # 1 cm in Bohr radii
EV_TO_HARTREE = 1.0 / HARTREE_EV


def debye_length(ne_cm3, Te_eV):
    """Debye screening length in Bohr radii.

    lambda_D = sqrt(kT / (4*pi*n_e)) in atomic units.

    Parameters
    ----------
    ne_cm3 : float
        Electron density in cm^-3.
    Te_eV : float
        Electron temperature in eV.

    Returns
    -------
    float
        Debye length in Bohr radii.
    """
    # Convert to atomic units
    ne_au = ne_cm3 * (BOHR_RADIUS_M * 100) ** 3  # density in a_0^{-3}
    kT_au = Te_eV * EV_TO_HARTREE  # temperature in Hartree

    if ne_au <= 0 or kT_au <= 0:
        return np.inf

    return np.sqrt(kT_au / (4.0 * np.pi * ne_au))


def ion_sphere_radius(Z_ion, ne_cm3):
    """Ion-sphere radius in Bohr radii.

    R_IS = (3*Z_ion / (4*pi*n_e))^{1/3}

    Parameters
    ----------
    Z_ion : int
        Ion charge (net charge, not nuclear charge).
    ne_cm3 : float
        Free electron density in cm^-3.

    Returns
    -------
    float
        Ion-sphere radius in Bohr radii.
    """
    ne_au = ne_cm3 * (BOHR_RADIUS_M * 100) ** 3
    if ne_au <= 0 or Z_ion <= 0:
        return np.inf
    return (3.0 * Z_ion / (4.0 * np.pi * ne_au)) ** (1.0 / 3.0)


def debye_potential(Z, ne_cm3, Te_eV):
    """Debye-screened nuclear potential.

    V(r) = -Z * exp(-r/lambda_D) / r

    For weakly-coupled plasmas where the coupling parameter
    Gamma = Z^2 / (R_IS * kT) << 1.

    Parameters
    ----------
    Z : int
        Nuclear charge.
    ne_cm3 : float
        Electron density in cm^-3.
    Te_eV : float
        Electron temperature in eV.

    Returns
    -------
    callable
        V(r) function that takes r in Bohr and returns potential in Hartree.
    """
    lam_D = debye_length(ne_cm3, Te_eV)

    def V(r):
        r_safe = np.maximum(r, 1e-30)
        return -Z * np.exp(-r_safe / lam_D) / r_safe

    return V


def ion_sphere_potential(Z, Z_ion, ne_cm3):
    """Ion-sphere screened potential.

    V(r) = -Z/r + Z_ion/(2*R_IS) * (3 - r^2/R_IS^2)   for r < R_IS
    V(r) = -(Z - Z_ion)/r                                for r >= R_IS

    For strongly-coupled plasmas. Inside the ion sphere, the
    electron density is uniform and exactly neutralizes the ion.

    Parameters
    ----------
    Z : int
        Nuclear charge.
    Z_ion : int
        Net ion charge (number of free electrons per ion).
    ne_cm3 : float
        Free electron density.

    Returns
    -------
    callable
        V(r) function.
    """
    R_IS = ion_sphere_radius(Z_ion, ne_cm3)

    def V(r):
        r_safe = np.maximum(r, 1e-30)
        V_out = -(Z - Z_ion) / r_safe
        V_in = -Z / r_safe + Z_ion / (2.0 * R_IS) * (3.0 - r_safe ** 2 / R_IS ** 2)
        return np.where(r_safe < R_IS, V_in, V_out)

    return V


def stewart_pyatt_potential(Z, Z_ion, ne_cm3, Te_eV):
    """Stewart-Pyatt screened potential.

    Interpolates between Debye and ion-sphere limits using the
    Stewart-Pyatt model. The transition parameter is:

    u = R_IS / lambda_D

    When u << 1: Debye limit (weak coupling)
    When u >> 1: ion-sphere limit (strong coupling)

    Parameters
    ----------
    Z : int
    Z_ion : int
    ne_cm3 : float
    Te_eV : float

    Returns
    -------
    callable
        V(r) function.
    """
    lam_D = debye_length(ne_cm3, Te_eV)
    R_IS = ion_sphere_radius(Z_ion, ne_cm3)

    # Coupling parameter
    u = R_IS / lam_D if lam_D > 0 else np.inf

    # Stewart-Pyatt interpolation: effective screening length
    # lambda_eff = lambda_D * f(u) where f interpolates between
    # f(0) = 1 (Debye) and f(inf) ~ R_IS/lambda_D (ion-sphere)
    if u < 0.1:
        # Debye limit
        return debye_potential(Z, ne_cm3, Te_eV)
    elif u > 10:
        # Ion-sphere limit
        return ion_sphere_potential(Z, Z_ion, ne_cm3)
    else:
        # Interpolation: effective lambda
        # SP formula: lambda_eff = lambda_D * [(1 + u^3)^(2/3) - u^2]^(1/2)
        lam_eff = lam_D * np.sqrt(max((1 + u ** 3) ** (2.0 / 3.0) - u ** 2, 0.01))

        def V(r):
            r_safe = np.maximum(r, 1e-30)
            return -Z * np.exp(-r_safe / lam_eff) / r_safe

        return V


def screening_potential_matrix(V_func, Z, bsp_data):
    """Convert a screening potential function to B-spline matrix form.

    Computes the difference V_screened(r) - V_nuclear(r) and assembles
    it into matrices that can be added to the Dirac Hamiltonian.

    Parameters
    ----------
    V_func : callable
        Screened potential V(r) in Hartree.
    Z : int
        Nuclear charge (to subtract bare nuclear potential).
    bsp_data : dict

    Returns
    -------
    V_PP, V_QQ : ndarray
        Screening correction matrices for P and Q components.
    """
    nv = bsp_data["nv"]
    ns = bsp_data["ns"]
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]

    # Screening correction: V_screened - V_nuclear
    V_screen = V_func(gr)
    V_nuc = -Z / np.maximum(gr, 1e-30)
    V_correction = V_screen - V_nuc

    V_w = grw * V_correction

    ksp = bsp_data.get("ksp", bsp_data["ks"])
    bsp_p = bsp_data["bsp_p"]
    V_PP = _assemble_integral(nv, ksp, ksp, ns, bsp_p, bsp_p, V_w)
    V_PP = 0.5 * (V_PP + V_PP.T)

    ksq = bsp_data.get("ksq", bsp_data["ks"])
    bsp_q = bsp_data.get("bsp_q", bsp_p)
    V_QQ = _assemble_integral(nv, ksq, ksq, ns, bsp_q, bsp_q, V_w)
    V_QQ = 0.5 * (V_QQ + V_QQ.T)

    return V_PP, V_QQ


def _stewart_pyatt_ipd_au(Z_ion, lam_D, R_IS):
    """Continuous Stewart-Pyatt IPD in Hartree.

    Stewart-Pyatt writes the continuum lowering as
    ``3 Z e^2 / (2 R) * ((1 + (lambda_D/R)^3)^(2/3)
    - (lambda_D/R)^2)``.  With this module's ``u = R/lambda_D``,
    the equivalent form is
    ``3 Z / (2 lambda_D) * (((1 + u^3)^(2/3) - 1) / u^3)``.
    It tends to ``Z/lambda_D`` for ``u -> 0`` and ``3Z/(2R)`` for
    ``u -> inf`` without any branch point in the physical range.
    """
    if not (Z_ion > 0 and lam_D > 0 and R_IS > 0):
        raise ValueError("Stewart-Pyatt IPD requires positive inputs")
    if not (np.isfinite(Z_ion) and np.isfinite(lam_D) and np.isfinite(R_IS)):
        raise ValueError("Stewart-Pyatt IPD requires finite inputs")

    u = R_IS / lam_D
    if not np.isfinite(u):
        IPD_au = 1.5 * Z_ion / R_IS
    elif u < 1e-5:
        ratio = (2.0 / 3.0) - (1.0 / 9.0) * u ** 3
        IPD_au = 1.5 * Z_ion / lam_D * ratio
    elif u > 1e100:
        IPD_au = 1.5 * Z_ion / R_IS
    else:
        log_u = np.log(u)
        log_u3 = 3.0 * log_u
        log_ratio = (
            np.log(np.expm1((2.0 / 3.0) * np.logaddexp(0.0, log_u3)))
            - log_u3
        )
        IPD_au = 1.5 * Z_ion / lam_D * np.exp(log_ratio)

    if not (np.isfinite(IPD_au) and IPD_au >= 0.0):
        raise ValueError("Stewart-Pyatt IPD evaluation produced non-finite output")
    return IPD_au


def _require_positive_finite(name, value):
    value = float(value)
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return value


def ionization_potential_depression(Z_ion, ne_cm3, Te_eV, model='stewart_pyatt'):
    """Ionization potential depression (IPD) from plasma screening.

    The IPD is the reduction in ionization energy due to plasma effects.

    Parameters
    ----------
    Z_ion : int
    ne_cm3 : float
    Te_eV : float
    model : str
        'debye', 'ion_sphere', or 'stewart_pyatt'.

    Returns
    -------
    float
        IPD in eV.
    """
    Z_ion = _require_positive_finite("Z_ion", Z_ion)
    ne_cm3 = _require_positive_finite("ne_cm3", ne_cm3)
    Te_eV = _require_positive_finite("Te_eV", Te_eV)

    if model == 'debye':
        lam_D = debye_length(ne_cm3, Te_eV)
        if not (np.isfinite(lam_D) and lam_D > 0.0):
            raise ValueError("Debye IPD evaluation produced invalid Debye length")
        IPD_au = Z_ion / lam_D
    elif model == 'ion_sphere':
        R_IS = ion_sphere_radius(Z_ion, ne_cm3)
        if not (np.isfinite(R_IS) and R_IS > 0.0):
            raise ValueError("ion-sphere IPD evaluation produced invalid radius")
        IPD_au = 1.5 * Z_ion / R_IS  # Ecker-Kroll formula
    elif model == 'stewart_pyatt':
        lam_D = debye_length(ne_cm3, Te_eV)
        R_IS = ion_sphere_radius(Z_ion, ne_cm3)
        IPD_au = _stewart_pyatt_ipd_au(Z_ion, lam_D, R_IS)
    else:
        raise ValueError(f"Unknown model: {model}")

    IPD_eV = IPD_au * HARTREE_EV
    if not (np.isfinite(IPD_eV) and IPD_eV >= 0.0):
        raise ValueError("IPD evaluation produced non-finite output")
    return IPD_eV
