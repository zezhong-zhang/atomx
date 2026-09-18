"""Radial multipole matrix elements for radiative transitions.

Computes the radial part of electric (EL) and magnetic (ML) multipole
matrix elements between relativistic (Dirac) orbitals.

For electric multipole EL in length (Babushkin) gauge:
    R_EL = integral [P_f(r)*P_i(r) + Q_f(r)*Q_i(r)] * r^L dr

For electric multipole EL in velocity (Coulomb) gauge:
    R_EL_vel involves d/dr and kappa-dependent terms

For magnetic multipole ML (FAC / Grant 2007 Eq. 8.4.17 convention):
    R_ML = integral [P_f(r)*Q_i(r) + Q_f(r)*P_i(r)] * r^L dr

No explicit 1/c factor is included: the alpha-suppression of magnetic
vs electric multipole rates arises naturally because the small-component
spinor Q ~ (alpha Z) P, so the cross integral P_f Q_i + Q_f P_i carries
one extra power of alpha relative to the electric P_f P_i + Q_f Q_i
combination.  The angular kappa factor in :func:`~atomx.properties.transitions
.angular.angular_mL` supplies the remaining (kappa_i+kappa_f)/(2L+1)
algebraic structure of the magnetic operator (Grant Eq. 8.4.17).

Workspace flavours
------------------
Every integral here is of the form ``∫ f(P_a, Q_a, P_b, Q_b, r) dr`` and
is evaluated through the :mod:`atomx.corrections.workspace` seam, so the
``bsp_data`` argument accepts either radial flavour:

* the B-spline workspace dict (``prepare_bspline_data``) with orbital
  arguments as expansion **coefficient vectors** (``orbitals_raw``); or
* a raw radial-grid workspace (:func:`atomx.corrections.workspace
  .make_grid_workspace` over the shooting engine's ``(r, difr)``) with
  orbital arguments as **grid-tabulated** ``P(r)``/``Q(r)`` arrays.

One implementation of each radial kernel serves both engines.
"""

import numpy as np

from atomx.constants import C

# One radial-integral implementation serves both engines: the workspace
# seam evaluates B-spline coefficient vectors through the (kinetic-
# balance-aware) basis tables and passes grid-tabulated orbitals through.
from atomx.corrections.workspace import eval_orbital_pair, workspace_arrays


def radial_eL(P_i, Q_i, P_f, Q_f, bsp_data, L=1):
    """Electric 2^L-pole radial matrix element (length/Babushkin gauge).

    R_EL = integral [P_f(r)*P_i(r) + Q_f(r)*Q_i(r)] * r^L dr

    For L=1 this is the standard E1 dipole matrix element.

    Parameters
    ----------
    P_i, Q_i : ndarray
        Initial-state large/small components: B-spline coefficient
        vectors for the B-spline workspace flavour, grid-tabulated
        arrays for the radial-grid flavour.
    P_f, Q_f : ndarray
        Final-state components (same flavour as the initial state).
    bsp_data : dict
        B-spline workspace or radial-grid workspace
        (:func:`atomx.corrections.workspace.make_grid_workspace`).
    L : int
        Multipole rank (1=dipole, 2=quadrupole, ...).

    Returns
    -------
    float
        Radial matrix element in atomic units (a_0^L).
    """
    gr, grw = workspace_arrays(bsp_data)

    fi_P, fi_Q = eval_orbital_pair(P_i, Q_i, bsp_data)
    ff_P, ff_Q = eval_orbital_pair(P_f, Q_f, bsp_data)

    integrand = (ff_P * fi_P + ff_Q * fi_Q) * gr ** L
    return np.sum(grw * integrand)


def radial_eL_velocity(P_i, Q_i, P_f, Q_f, bsp_data, L=1,
                        kappa_i=None, kappa_f=None, omega=None):
    """Electric 2^L-pole radial matrix element (velocity/Coulomb gauge).

    Full derivative-free velocity-gauge form (Grant 2007 §8.4, Gu 2008
    §3.2): for exact eigenstates of a common Dirac-Coulomb Hamiltonian
    the velocity-gauge radial integral equals the length-gauge value,
    so that both gauges produce the *same* Einstein A-coefficient when
    combined with the *same* angular factor
    :func:`~atomx.properties.transitions.angular.angular_eL` and rate formula
    :func:`~atomx.properties.transitions.rates.einstein_A_eL`.

    Grant §8.4 derivative-free reduction (all L ≥ 1)::

        R^V(EL) = (c / ω)
                  × [(κ_i − κ_f + L) · B^{(L−1)}
                   + (κ_i − κ_f − L) · A^{(L−1)}]

    where ``A`` and ``B`` are the individual large–small cross integrals
    weighted by ``r^{L−1}``::

        A = ∫ P_f(r) Q_i(r) r^{L−1} dr
        B = ∫ Q_f(r) P_i(r) r^{L−1} dr

    When ``kappa_i``, ``kappa_f``, or ``omega`` are not supplied the
    leading-order fallback is used instead (no gauge-invariance guarantee).

    Parameters
    ----------
    P_i, Q_i, P_f, Q_f : ndarray
        Initial / final, large / small components (B-spline coefficient
        vectors or grid-tabulated arrays — see module docstring).
    bsp_data : dict
        B-spline or radial-grid workspace.
    L : int
        Multipole rank (1 = dipole, 2 = quadrupole, …).
    kappa_i, kappa_f : int, optional
        Dirac quantum numbers.  Required for the full-order form.
    omega : float, optional
        Transition energy in Hartree.  Required for the full-order form.

    Returns
    -------
    float
        Velocity-gauge radial matrix element in atomic units.
    """
    gr, grw = workspace_arrays(bsp_data)

    fi_P, fi_Q = eval_orbital_pair(P_i, Q_i, bsp_data)
    ff_P, ff_Q = eval_orbital_pair(P_f, Q_f, bsp_data)

    if omega is not None and kappa_i is not None and kappa_f is not None:
        dk = kappa_i - kappa_f
        rl = gr ** (L - 1)
        A = ff_P * fi_Q * rl   # P_f Q_i r^{L-1}
        B = ff_Q * fi_P * rl   # Q_f P_i r^{L-1}
        integrand = (dk + L) * A + (dk - L) * B
        return (C / omega) * float(np.sum(grw * integrand))

    # Leading-order fallback (no gauge-invariance guarantee).
    integrand = (ff_P * fi_Q - ff_Q * fi_P) * gr ** (L - 1) / C
    return float(np.sum(grw * integrand))


def radial_mL(P_i, Q_i, P_f, Q_f, bsp_data, L=1):
    """Magnetic 2^L-pole radial matrix element (long-wavelength).

    R_ML = integral [P_f(r)*Q_i(r) + Q_f(r)*P_i(r)] * r^L dr

    For L=1 this is the M1 magnetic dipole matrix element.

    Convention: matches Grant (2007) Eq. 8.4.17 / FAC's M_L(omega) in the
    long-wavelength limit. The full magnetic reduced matrix element is
    obtained by combining this with the kappa-dependent angular factor
    in :func:`~atomx.properties.transitions.angular.angular_mL`:

        ⟨f‖t^M_L‖i⟩ = ⟨f‖C^L‖i⟩ · (kappa_i + kappa_f)/(2L+1) · R_ML

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

    Returns
    -------
    float
        Radial matrix element in atomic units (a_0^L).
    """
    gr, grw = workspace_arrays(bsp_data)

    fi_P, fi_Q = eval_orbital_pair(P_i, Q_i, bsp_data)
    ff_P, ff_Q = eval_orbital_pair(P_f, Q_f, bsp_data)

    integrand = (ff_P * fi_Q + ff_Q * fi_P) * gr ** L
    return np.sum(grw * integrand)
