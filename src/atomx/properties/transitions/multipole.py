"""Radial multipole matrix elements for radiative transitions.

Computes the radial part of electric (EL) and magnetic (ML) multipole
matrix elements between relativistic (Dirac) orbitals.

In the long-wavelength limit, electric multipole EL in length gauge:
    R_EL = integral [P_f(r)*P_i(r) + Q_f(r)*Q_i(r)] * r^L dr

For electric multipole EL in velocity (Coulomb) gauge:
    R_EL_vel uses a derivative-free combination of large–small products

For magnetic multipole ML (FAC / Grant 2007 Eq. 8.4.17 convention):
    R_ML = integral [P_f(r)*Q_i(r) + Q_f(r)*P_i(r)] * r^L dr

The displayed powers of r are zero-frequency limits. Supplying the photon
energy retains the spherical-Bessel radial operators for electric and magnetic
radiation. Electric operators use the signed energy E_f - E_i; magnetic
operators use its positive magnitude.

No explicit 1/c factor is included: the alpha-suppression of magnetic
vs electric multipole rates arises naturally because the small-component
spinor Q ~ (alpha Z) P, so the cross integral P_f Q_i + Q_f P_i carries
one extra power of alpha relative to the electric P_f P_i + Q_f Q_i
combination.  The angular kappa factor in :func:`~atomx.properties.transitions
.angular.angular_mL` supplies the remaining (kappa_i+kappa_f)/(L+1)
algebraic structure of the magnetic operator (Grant Eq. 8.4.17).

Workspace flavours
------------------
Every integral here is of the form ``∫ f(P_a, Q_a, P_b, Q_b, r) dr`` and
is evaluated through the :mod:`atomx.numerics.radial_workspace` seam, so the
``bsp_data`` argument accepts either radial flavour:

* the B-spline workspace dict (``prepare_bspline_data``) with orbital
  arguments as expansion **coefficient vectors** (``orbitals_raw``); or
* a raw radial-grid workspace (:func:`atomx.numerics.radial_workspace
  .make_grid_workspace` over the shooting engine's ``(r, difr)``) with
  orbital arguments as **grid-tabulated** ``P(r)``/``Q(r)`` arrays.

One implementation of each radial kernel serves both engines.
"""

import numpy as np

from atomx.constants import C
from atomx.numerics.special import scaled_spherical_jn

# One radial-integral implementation serves both engines: the workspace
# seam evaluates B-spline coefficient vectors through the (kinetic-
# balance-aware) basis tables and passes grid-tabulated orbitals through.
from atomx.numerics.radial_workspace import eval_orbital_pair, workspace_arrays


def radial_eL(P_i, Q_i, P_f, Q_f, bsp_data, L=1, *,
              kappa_i=None, kappa_f=None, omega=None):
    """Electric multipole in length (Babushkin) gauge.

    With ``omega=None`` or zero this is the static integral of
    ``(P_f P_i + Q_f Q_i) r**L``. For radiative transitions supply both
    kappas and the SIGNED energy ``omega = E_f - E_i`` in Hartree. In a
    CI contraction use the state energy difference for every orbital pair.
    Interchanging i/f requires reversing omega. The finite-frequency
    operator includes both the charge-density and large–small terms.

    The normalization has the static limit in a_0**L and uses the same
    angular factor and Einstein rate formula as the static operator.
    Both B-spline coefficients and tabulated grid spinors are supported.
    """
    gr, grw = workspace_arrays(bsp_data)
    fi_P, fi_Q = eval_orbital_pair(P_i, Q_i, bsp_data)
    ff_P, ff_Q = eval_orbital_pair(P_f, Q_f, bsp_data)
    rho = ff_P * fi_P + ff_Q * fi_Q
    if omega is None or omega == 0:
        return float(np.sum(grw * rho * gr**L))
    _validate_electric_frequency(kappa_i, kappa_f, omega)
    k = omega / C
    cross_sum = fi_P * ff_Q + fi_Q * ff_P
    cross_difference = fi_P * ff_Q - fi_Q * ff_P
    integrand = (
        rho * gr**L * scaled_spherical_jn(L, k * gr)
        + k * gr**(L + 1) * scaled_spherical_jn(L + 1, k * gr)
        / (2 * L + 3)
        * ((kappa_i - kappa_f) / (L + 1) * cross_sum + cross_difference)
    )
    return float(np.sum(grw * integrand))


def _validate_electric_frequency(kappa_i, kappa_f, omega):
    if kappa_i is None or kappa_f is None or omega is None:
        raise ValueError('electric radiation needs both kappas and signed omega=E_f-E_i')
    if not np.isfinite(omega) or omega == 0:
        raise ValueError('velocity/finite-frequency operator needs finite nonzero omega')


def radial_eL_velocity(P_i, Q_i, P_f, Q_f, bsp_data, L=1,
                       kappa_i=None, kappa_f=None, omega=None, *,
                       long_wavelength=False):
    """Electric multipole in velocity (Coulomb) gauge, all photon orders.

    ``omega = E_f - E_i`` is SIGNED, as in :func:`radial_eL`. Both kappas
    and nonzero omega are required; there is no arbitrary fallback when
    these are missing. ``long_wavelength=True`` retains only the leading
    photon term for explicit diagnostics.

    For eigenstates of a common local Dirac Hamiltonian the two gauges
    agree. Approximate many-electron wavefunctions and nonlocal exchange
    do not in general obey this one-body gauge identity exactly.
    """
    _validate_electric_frequency(kappa_i, kappa_f, omega)
    gr, grw = workspace_arrays(bsp_data)
    fi_P, fi_Q = eval_orbital_pair(P_i, Q_i, bsp_data)
    ff_P, ff_Q = eval_orbital_pair(P_f, Q_f, bsp_data)
    dk = kappa_i - kappa_f
    cross_sum = fi_P * ff_Q + fi_Q * ff_P
    cross_difference = fi_P * ff_Q - fi_Q * ff_P
    k = omega / C
    integrand = gr**(L - 1) / k * (-dk * cross_sum + L * cross_difference)
    if not long_wavelength:
        integrand *= scaled_spherical_jn(L - 1, k * gr)
        integrand += (
            k * gr**(L + 1) * scaled_spherical_jn(L + 1, k * gr)
            / ((2 * L + 1) * (2 * L + 3))
            * (dk * L / (L + 1) * cross_sum + L * cross_difference)
        )
    return float(np.sum(grw * integrand))


def radial_mL(P_i, Q_i, P_f, Q_f, bsp_data, L=1, *, omega=None):
    """Magnetic multipole with optional finite photon wavelength.

    R_ML = integral [P_f(r)*Q_i(r) + Q_f(r)*P_i(r)] * r^L dr

    For L=1 this is the M1 magnetic dipole matrix element.

    Convention: matches Grant (2007) Eq. 8.4.17 / FAC's M_L(omega) in the
    long-wavelength limit. The full magnetic reduced matrix element is
    obtained by combining this with the kappa-dependent angular factor
    in :func:`~atomx.properties.transitions.angular.angular_mL`:

        ⟨f‖t^M_L‖i⟩ = ⟨-kappa_f‖C^L‖kappa_i⟩ · (kappa_i + kappa_f)/(L+1) · R_ML

    No explicit 1/c factor is included — alpha-suppression of the rate
    arises from the Q ~ (alpha Z) P scaling of the small spinor component.

    Parameters
    ----------
    P_i, Q_i, P_f, Q_f : ndarray
        B-spline coefficient vectors or grid-tabulated arrays.
    bsp_data : dict
        B-spline or radial-grid workspace.
    L : int
        Multipole rank.
    omega : float, optional
        Positive photon energy in Hartree. Replaces r^L with
        (2L+1)!! j_L(omega*r/c)/(omega/c)^L. None or zero explicitly
        selects the long-wavelength limit. Inhibited M1 amplitudes need
        the finite-frequency term even at small Z.

    Returns
    -------
    float
        Radial matrix element in atomic units (a_0^L).
    """
    gr, grw = workspace_arrays(bsp_data)

    fi_P, fi_Q = eval_orbital_pair(P_i, Q_i, bsp_data)
    ff_P, ff_Q = eval_orbital_pair(P_f, Q_f, bsp_data)

    radial_power = gr**L
    if omega is not None and omega != 0:
        if not np.isfinite(omega) or omega < 0:
            raise ValueError('photon energy must be finite and nonnegative')
        radial_power *= scaled_spherical_jn(L, omega * gr / C)
    integrand = (ff_P * fi_Q + ff_Q * fi_P) * radial_power
    return np.sum(grw * integrand)
