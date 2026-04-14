"""Continuum Dirac wavefunctions for photoionization cross-sections.

For photoionization, we need the continuum wavefunction at energy E > 0
(electron energy above the ionization threshold). The continuum state
satisfies the same Dirac equation but with positive energy, and is
normalized to unit flux (not L^2 normalizable).

In the B-spline basis, the continuum solution is obtained by solving
the Dirac equation at a specified positive energy, then extracting
the phase shift from the asymptotic form of the wavefunction.

For photoionization cross-section σ(ω):
  σ(ω) = (4π²αω/3) Σ_{κ_f} |<f|r|i>|² × (2j_f+1)

where |f> is the continuum state, |i> the bound orbital, and the
matrix element involves the dipole operator.
"""

import numpy as np
from scipy import linalg
from atomx.constants import C, C_SQ, ALPHA, HARTREE_EV
from atomx.bspline.knots import make_knots
from atomx.bspline.basis import prepare_bspline_data
from atomx.dirac.matrix import build_dirac_matrices
from atomx.bspline.evaluate import eval_orbital_on_grid


def solve_continuum(kappa, energy, Z, bsp_data, V_scf_PP=None, V_scf_QQ=None):
    """Find the continuum wavefunction at given positive energy.

    Solves the Dirac equation (H - E*S)c = 0 where E > 0 is the
    electron kinetic energy. The solution closest to the target energy
    is extracted from the discretized eigenvalue spectrum.

    Parameters
    ----------
    kappa : int
        Angular momentum quantum number of the continuum electron.
    energy : float
        Kinetic energy of the continuum electron in Hartree (> 0).
    Z : int
        Nuclear charge (or effective charge of the residual ion).
    bsp_data : dict
        B-spline data.
    V_scf_PP, V_scf_QQ : ndarray, optional
        Self-consistent potential from bound electrons (Hartree + exchange).
        If None, uses bare nuclear potential.

    Returns
    -------
    P : ndarray, shape (ns,)
        Large component B-spline coefficients (box-normalized).
    Q : ndarray, shape (ns,)
        Small component B-spline coefficients.
    E_actual : float
        Nearest discretized eigenvalue to target energy.
    """
    ns = bsp_data["ns"]
    ms = 2 * ns

    H, S = build_dirac_matrices(ns, kappa, Z, bsp_data)

    # Add SCF potential if provided
    if V_scf_PP is not None:
        H[:ns, :ns] += V_scf_PP
    if V_scf_QQ is not None:
        H[ns:, ns:] += V_scf_QQ

    # Solve full eigenvalue problem (including both bound and continuum)
    nb = 1  # boundary removal
    keep_p = list(range(nb, ns - nb))
    keep_q = list(range(ns + nb, ms - nb))
    keep = keep_p + keep_q
    nk = len(keep)

    H_red = H[np.ix_(keep, keep)]
    S_red = S[np.ix_(keep, keep)]

    eigenvalues, eigenvectors = linalg.eigh(H_red, S_red)

    # Find eigenvalue closest to target energy (in the continuum, E > 0)
    # Exclude Dirac sea (E < -2c²)
    sea_threshold = -2.0 * C_SQ
    physical_mask = eigenvalues > sea_threshold
    phys_evals = eigenvalues[physical_mask]
    phys_evecs = eigenvectors[:, physical_mask]

    # Find closest to target
    idx = np.argmin(np.abs(phys_evals - energy))
    E_actual = phys_evals[idx]

    # Map back to full vectors
    vec_red = phys_evecs[:, idx]
    P = np.zeros(ns)
    Q = np.zeros(ns)
    for ii, k in enumerate(keep_p):
        P[k] = vec_red[ii]
    for ii, k in enumerate(keep_q):
        Q[k - ns] = vec_red[len(keep_p) + ii]

    return P, Q, E_actual


def dipole_matrix_element(P_bound, Q_bound, P_cont, Q_cont, bsp_data):
    """Compute the electric dipole matrix element <f|r|i>.

    For photoionization, the dipole operator in length gauge is r.
    The radial matrix element is:
        <f|r|i> = ∫ [P_f(r) * P_i(r) + Q_f(r) * Q_i(r)] * r dr

    Parameters
    ----------
    P_bound, Q_bound : ndarray
        Bound orbital B-spline coefficients.
    P_cont, Q_cont : ndarray
        Continuum orbital B-spline coefficients.
    bsp_data : dict

    Returns
    -------
    float
        Dipole matrix element.
    """
    nv = bsp_data["nv"]
    ks = bsp_data["ks"]
    ns = bsp_data["ns"]
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    bsp = bsp_data["bsp_p"]

    fi_P = eval_orbital_on_grid(P_bound, bsp, nv, ks, ns)
    fi_Q = eval_orbital_on_grid(Q_bound, bsp, nv, ks, ns)
    ff_P = eval_orbital_on_grid(P_cont, bsp, nv, ks, ns)
    ff_Q = eval_orbital_on_grid(Q_cont, bsp, nv, ks, ns)

    # <f|r|i> = ∫ (P_f P_i + Q_f Q_i) r dr
    integrand = (ff_P * fi_P + ff_Q * fi_Q) * gr
    return np.sum(grw * integrand)


def photoionization_cross_section(bound_P, bound_Q, bound_kappa, bound_energy,
                                   photon_energies, Z, bsp_data,
                                   V_scf_PP=None, V_scf_QQ=None):
    """Compute photoionization cross-section σ(ω) for a bound orbital.

    σ(ω) = (4π²αω/3) Σ_{κ_f} |<f|r|i>|² × angular_factor

    Parameters
    ----------
    bound_P, bound_Q : ndarray
        Bound orbital coefficients.
    bound_kappa : int
        Bound orbital kappa.
    bound_energy : float
        Bound orbital energy (negative, in Hartree).
    photon_energies : ndarray
        Photon energies in Hartree.
    Z : int
    bsp_data : dict
    V_scf_PP, V_scf_QQ : ndarray, optional

    Returns
    -------
    sigma : ndarray
        Cross-section in atomic units (a_0^2) at each photon energy.
    """
    sigma = np.zeros_like(photon_energies)

    # Allowed final-state kappa values (E1 dipole selection rules: Δl = ±1)
    l_bound = bound_kappa if bound_kappa > 0 else -bound_kappa - 1
    j2_bound = 2 * abs(bound_kappa) - 1

    # E1 dipole: final states with l_f = l_bound ± 1
    final_kappas = []
    for l_f in [l_bound - 1, l_bound + 1]:
        if l_f < 0:
            continue
        # j_f = l_f ± 1/2
        final_kappas.append(-(l_f + 1))  # kappa for j = l+1/2
        if l_f > 0:
            final_kappas.append(l_f)      # kappa for j = l-1/2

    for idx, omega in enumerate(photon_energies):
        electron_energy = omega + bound_energy  # kinetic energy of photoelectron
        if electron_energy <= 0:
            continue  # below threshold

        for kf in final_kappas:
            j2_f = 2 * abs(kf) - 1
            angular_factor = (j2_f + 1)  # (2j_f + 1) degeneracy

            Pf, Qf, E_actual = solve_continuum(
                kf, electron_energy, Z, bsp_data, V_scf_PP, V_scf_QQ)

            dme = dipole_matrix_element(bound_P, bound_Q, Pf, Qf, bsp_data)

            # σ = 4π²αω/3 × |<f|r|i>|² × (2j_f+1)
            sigma[idx] += (4.0 * np.pi ** 2 * ALPHA * omega / 3.0
                          * dme ** 2 * angular_factor)

    return sigma
