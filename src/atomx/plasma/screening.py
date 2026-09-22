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


def stewart_pyatt_screening_length(lam_D, R_IS):
    """Stewart-Pyatt effective screening length in Bohr radii.

    The Stewart-Pyatt interpolation of the screened potential reduces to
    the Debye-Huckel form for weak coupling (``u = R_IS/lam_D -> 0``) and
    to the ion-sphere form for strong coupling (``u -> inf``).  Written as
    a Yukawa potential ``-Z exp(-r/lam_eff)/r``, the ion-sphere limit
    requires the continuum shift of that Yukawa, ``Z/lam_eff``, to equal
    the ion-sphere shift ``3 Z_ion/(2 R_IS)``; for the self-screening case
    ``Z = Z_ion`` this fixes

        lam_eff(u -> inf) = (2/3) R_IS,

    while ``lam_eff(0) = lam_D``.  The closed form used here,

        lam_eff = lam_D * (2/3) u^3 / ((1 + u^3)^{2/3} - 1),

    is the Stewart-Pyatt interpolation itself: it is monotone in u, has
    both limits, and is evaluated with ``log1p``/``expm1`` so the small-u
    cancellation in the denominator never loses significance.

    The previous interpolation factor ``sqrt((1 + u^3)^{2/3} - u^2)``
    *decreased* like ``u^{-1/2}`` for large u, sending ``lam_eff -> 0``
    and the potential to 0 instead of the ion-sphere limit — a 247 eV
    discontinuity at the branch switch.

    Parameters
    ----------
    lam_D : float
        Debye length in Bohr radii.
    R_IS : float
        Ion-sphere radius in Bohr radii.

    Returns
    -------
    float
        Effective screening length in Bohr radii (``lam_D`` when either
        input is non-positive/non-finite).
    """
    lam_D = float(lam_D)
    R_IS = float(R_IS)
    if not np.isfinite(lam_D) or lam_D <= 0.0:
        return np.inf
    if not np.isfinite(R_IS) or R_IS <= 0.0:
        return lam_D
    u = R_IS / lam_D
    if u < 1e-3:
        # (2/3)u^3 / expm1((2/3)log1p(u^3)) = 1 + u^3/6 + O(u^6)
        ratio = 1.0 + u ** 3 / 6.0
    elif u > 1e3:
        # asymptote: lam_eff -> (2/3) R_IS
        ratio = (2.0 / 3.0) * u
    else:
        ratio = (2.0 / 3.0) * u ** 3 / np.expm1(
            (2.0 / 3.0) * np.log1p(u ** 3))
    return lam_D * ratio


def _smoothstep(x):
    """C1 monotone step: 0 for x <= 0, 1 for x >= 1, 3x^2 - 2x^3 between."""
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def stewart_pyatt_potential(Z, Z_ion, ne_cm3, Te_eV):
    """Stewart-Pyatt screened potential.

    Interpolates between the Debye and ion-sphere limits using the
    Stewart-Pyatt model. The transition parameter is:

    u = R_IS / lambda_D

    When u << 1: Debye limit (weak coupling)
    When u >> 1: ion-sphere limit (strong coupling)

    The interpolation is continuous by construction: the screening length
    :func:`stewart_pyatt_screening_length` runs from ``lambda_D`` to
    ``(2/3) R_IS``, and the potential is blended between the Yukawa form
    and the exact ion-sphere potential with a smooth weight in ``log10 u``
    that is 0 at ``u = 0.1`` and 1 at ``u = 10``.  The two limits are
    therefore reproduced exactly — ``V = debye_potential`` for
    ``u <= 0.1`` and ``V = ion_sphere_potential`` for ``u >= 10`` — with
    no jump at either switch (the previous three-branch version jumped by
    ~250 eV at ``u = 10`` because ``lambda_eff`` collapsed to 0).

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
    V_ion_sphere = ion_sphere_potential(Z, Z_ion, ne_cm3)

    # Screening length: lambda_D (Debye) -> (2/3) R_IS (ion sphere)
    lam_sp = stewart_pyatt_screening_length(lam_D, R_IS)

    # Coupling parameter u = R_IS/lambda_D; the interpolation runs over
    # 0.1 <= u <= 10 (one decade either side of u = 1 in log10 u).
    if np.isfinite(lam_D) and lam_D > 0.0:
        u = R_IS / lam_D if np.isfinite(R_IS) else np.inf
    else:
        u = 0.0  # no screening length: weak-coupling end
    if not np.isfinite(u):
        weight = 1.0
    elif u <= 0.0:
        weight = 0.0
    else:
        weight = _smoothstep((np.log10(u) + 1.0) / 2.0)

    if weight <= 0.0:
        lam_eff = lam_D
    elif weight >= 1.0:
        lam_eff = lam_sp
    else:
        lam_eff = (1.0 - weight) * lam_D + weight * lam_sp

    def V(r):
        r_safe = np.maximum(r, 1e-30)
        yukawa = -Z * np.exp(-r_safe / lam_eff) / r_safe
        if weight <= 0.0:
            return yukawa
        if weight >= 1.0:
            return V_ion_sphere(r_safe)
        return (1.0 - weight) * yukawa + weight * V_ion_sphere(r_safe)

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


def screening_shift(Z, r, ne_cm3, Te_eV, model="debye", z_ion=None):
    """Additive screening shift ``delta_V(r) = V_screened(r) - (-Z/r)``.

    This is what an SCF has to add to the *nuclear* potential: the model's
    own central potential minus the bare ``-Z/r``.  The difference is
    finite at the origin (the nuclear charge distribution of
    :mod:`atomx.potentials.nuclear` keeps regularising ``r -> 0``) and
    tends to ``Z/lambda`` at large ``r`` for the Debye model — the
    continuum shift that lowers the ionisation energy.  It is evaluated by
    calling :func:`debye_potential` / :func:`stewart_pyatt_potential`
    directly, so ``screening_shift(...) == V_model(r) + Z/r`` holds by
    construction.

    Parameters
    ----------
    Z : int
        Nuclear charge.
    r : array_like
        Radii in Bohr.
    ne_cm3 : float
        Free-electron density in cm^-3.
    Te_eV : float
        Free-electron temperature in eV.
    model : {"debye", "stewart_pyatt"}
        Screening model.  ``"debye"`` is the Yukawa (Debye-Huckel) limit;
        ``"stewart_pyatt"`` interpolates to the ion-sphere limit and needs
        a positive ``z_ion``.
    z_ion : int, optional
        Net ion charge, i.e. the number of free electrons per ion.  Only
        read by ``"stewart_pyatt"`` (it fixes the ion-sphere radius).

    Returns
    -------
    ndarray
        ``delta_V(r)`` in Hartree on the input grid.
    """
    r = np.asarray(r, dtype=float)
    # ``V_model`` diverges as -Z/r at the origin while the shift stays
    # finite (Z/lambda for Debye), so the difference is a cancellation of
    # two large numbers.  1e-9 Bohr is far below the innermost solver grid
    # point (~1e-7) and keeps the cancellation at the 1e-6 relative level.
    r_safe = np.maximum(r, 1e-9)
    if model == "debye":
        V = debye_potential(Z, ne_cm3, Te_eV)(r_safe)
    elif model == "stewart_pyatt":
        if z_ion is None or int(z_ion) <= 0:
            raise ValueError(
                "screening model 'stewart_pyatt' needs a positive ion "
                f"charge for the ion-sphere radius (got z_ion={z_ion!r})")
        V = stewart_pyatt_potential(Z, int(z_ion), ne_cm3, Te_eV)(r_safe)
    else:
        raise ValueError(
            f"model must be 'debye' or 'stewart_pyatt' (got {model!r})")
    return V + Z / r_safe


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
