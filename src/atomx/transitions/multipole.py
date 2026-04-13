"""Radial multipole matrix elements for radiative transitions.

Computes the radial part of electric (EL) and magnetic (ML) multipole
matrix elements between relativistic (Dirac) orbitals represented in
a B-spline basis.

For electric multipole EL in length (Babushkin) gauge:
    R_EL = integral [P_f(r)*P_i(r) + Q_f(r)*Q_i(r)] * r^L dr

For electric multipole EL in velocity (Coulomb) gauge:
    R_EL_vel involves d/dr and kappa-dependent terms

For magnetic multipole ML:
    R_ML = integral [P_f(r)*Q_i(r) + Q_f(r)*P_i(r)] * r^L dr / c
"""

import numpy as np
from atomx.constants import C
from atomx.scf.slater import eval_orbital_on_grid


def _eval_P(coeffs, bsp_data):
    """Evaluate P-component on grid (KB-aware)."""
    ksp = bsp_data.get("ksp", bsp_data["ks"])
    nsp = bsp_data.get("nsp", bsp_data["ns"])
    return eval_orbital_on_grid(coeffs, bsp_data["bsp_p"],
                                 bsp_data["nv"], ksp, nsp)


def _eval_Q(coeffs, bsp_data):
    """Evaluate Q-component on grid (KB-aware)."""
    ksq = bsp_data.get("ksq", bsp_data["ks"])
    nsq = bsp_data.get("nsq", bsp_data["ns"])
    bsp_q = bsp_data.get("bsp_q", bsp_data["bsp_p"])
    return eval_orbital_on_grid(coeffs, bsp_q, bsp_data["nv"], ksq, nsq)


def radial_eL(P_i, Q_i, P_f, Q_f, bsp_data, L=1):
    """Electric 2^L-pole radial matrix element (length/Babushkin gauge).

    R_EL = integral [P_f(r)*P_i(r) + Q_f(r)*Q_i(r)] * r^L dr

    For L=1 this is the standard E1 dipole matrix element.

    Parameters
    ----------
    P_i, Q_i : ndarray
        Initial-state B-spline coefficients (large, small components).
    P_f, Q_f : ndarray
        Final-state B-spline coefficients.
    bsp_data : dict
        B-spline grid data.
    L : int
        Multipole rank (1=dipole, 2=quadrupole, ...).

    Returns
    -------
    float
        Radial matrix element in atomic units (a_0^L).
    """
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]

    fi_P = _eval_P(P_i, bsp_data)
    fi_Q = _eval_Q(Q_i, bsp_data)
    ff_P = _eval_P(P_f, bsp_data)
    ff_Q = _eval_Q(Q_f, bsp_data)

    integrand = (ff_P * fi_P + ff_Q * fi_Q) * gr ** L
    return np.sum(grw * integrand)


def radial_eL_velocity(P_i, Q_i, P_f, Q_f, bsp_data, L=1,
                        kappa_i=None, kappa_f=None):
    """Electric 2^L-pole radial matrix element (velocity/Coulomb gauge).

    For E1 in velocity gauge (Grant, Eq. 9.21):
        R_E1_vel = integral [P_f*Q_i*(kappa_i+1) + Q_f*P_i*(kappa_f+1)
                             - P_f*Q_i + Q_f*P_i] * (1/(omega*r)) dr
    Simplified for general L using Johnson (2007) formulation:
        R_EL_vel = integral [P_f*Q_i - Q_f*P_i] * r^(L-1) / c dr
                   * L / (L+1) + kappa-dependent terms

    For now, implements the leading-order velocity-gauge form.

    Parameters
    ----------
    P_i, Q_i, P_f, Q_f : ndarray
        B-spline coefficients.
    bsp_data : dict
    L : int
        Multipole rank.
    kappa_i, kappa_f : int, optional
        Kappa quantum numbers (needed for exact velocity gauge).

    Returns
    -------
    float
        Velocity-gauge radial matrix element.
    """
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]

    fi_P = _eval_P(P_i, bsp_data)
    fi_Q = _eval_Q(Q_i, bsp_data)
    ff_P = _eval_P(P_f, bsp_data)
    ff_Q = _eval_Q(Q_f, bsp_data)

    # Leading velocity-gauge form: involves cross-terms P*Q
    # For E1, the exact relativistic velocity form (Grant 2007):
    # T_vel = (1/omega) * integral [(kappa_f + kappa_i) * (P_f*Q_i + Q_f*P_i)
    #         + (kappa_f - kappa_i) * (P_f*Q_i - Q_f*P_i)] dr / (2c)
    # Simplified for general L as the leading contribution:
    if kappa_i is not None and kappa_f is not None:
        # Exact E1 velocity gauge using kappa
        term1 = (ff_P * fi_Q + ff_Q * fi_P)  # symmetric
        term2 = (ff_P * fi_Q - ff_Q * fi_P)  # antisymmetric
        integrand = ((kappa_f + kappa_i) * term1
                     + (kappa_f - kappa_i) * term2) / (2.0 * C)
        return np.sum(grw * integrand)
    else:
        # Approximate velocity form without kappa
        integrand = (ff_P * fi_Q - ff_Q * fi_P) * gr ** (L - 1) / C
        return np.sum(grw * integrand)


def radial_mL(P_i, Q_i, P_f, Q_f, bsp_data, L=1):
    """Magnetic 2^L-pole radial matrix element.

    R_ML = integral [P_f(r)*Q_i(r) + Q_f(r)*P_i(r)] * r^L dr / c

    For L=1 this is the M1 magnetic dipole matrix element.

    Parameters
    ----------
    P_i, Q_i, P_f, Q_f : ndarray
        B-spline coefficients.
    bsp_data : dict
    L : int
        Multipole rank.

    Returns
    -------
    float
        Radial matrix element in atomic units.
    """
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]

    fi_P = _eval_P(P_i, bsp_data)
    fi_Q = _eval_Q(Q_i, bsp_data)
    ff_P = _eval_P(P_f, bsp_data)
    ff_Q = _eval_Q(Q_f, bsp_data)

    integrand = (ff_P * fi_Q + ff_Q * fi_P) * gr ** L / C
    return np.sum(grw * integrand)
