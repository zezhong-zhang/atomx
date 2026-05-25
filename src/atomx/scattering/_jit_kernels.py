"""Numba-JIT compiled inner-loop kernels for scattering calculations.

These functions extract the tight numerical loops from the DW excitation,
EII DW, and Auger rate computations so that Python-level overhead is
eliminated from the innermost iterations.

All functions here are pure-numerical (no Python objects) and decorated
with ``@nb.njit`` for ahead-of-time compilation.
"""

from __future__ import annotations

import numpy as np
import numba as nb


@nb.njit(cache=True)
def dw_j_sum(
    tj_lo: int, tj_hi: int,
    Rd_k: np.ndarray, Rd_val: np.ndarray,
    Re_k: np.ndarray, Re_val: np.ndarray,
    C_a1: np.ndarray, C_in2: np.ndarray,
    C_a2: np.ndarray, C_in1: np.ndarray,
    tj_a: int, tj_in: int, tj_1: int, tj_2: int,
    wigner6j_table: np.ndarray,
) -> float:
    """Sum over total J for the EII DW inner loop.

    Computes:
        sum_J (2J+1) |M_J^dir - (-1)^{(j1+j2-J)/2} M_J^ex|^2

    Parameters
    ----------
    tj_lo, tj_hi : int
        Range of 2*J values (inclusive, step 2).
    Rd_k, Rd_val : arrays of shape (n_direct,)
        Multipole ranks k and precomputed C_a1[k]*C_in2[k]*R^k for direct.
    Re_k, Re_val : arrays of shape (n_exchange,)
        Same for exchange.
    C_a1, C_in2, C_a2, C_in1 : arrays
        Angular coefficients (not used directly — folded into Rd_val/Re_val).
    tj_a, tj_in, tj_1, tj_2 : int
        2*j values for the four angular momenta.
    wigner6j_table : ndarray of shape (n_J, max(n_direct, n_exchange))
        Precomputed wigner_6j values.  First index is J (indexed by
        (tJ - tj_lo) // 2), second is the k-index.
    """
    n_J = (tj_hi - tj_lo) // 2 + 1
    n_d = Rd_val.shape[0]
    n_e = Re_val.shape[0]
    total = 0.0

    for iJ in range(n_J):
        tJ = tj_lo + 2 * iJ
        phase_sign_J = 1.0
        if ((tj_in + tj_1 + tJ) // 2) % 2 != 0:
            phase_sign_J = -1.0

        MJd = 0.0
        for ik in range(n_d):
            sj = wigner6j_table[iJ, ik]  # direct 6j
            if abs(sj) < 1e-15:
                continue
            MJd += sj * Rd_val[ik]
        MJd *= phase_sign_J

        phase_sign_e = 1.0
        if ((tj_in + tj_2 + tJ) // 2) % 2 != 0:
            phase_sign_e = -1.0

        MJe = 0.0
        for ik in range(n_e):
            sj = wigner6j_table[iJ, n_d + ik]  # exchange 6j
            if abs(sj) < 1e-15:
                continue
            MJe += sj * Re_val[ik]
        MJe *= phase_sign_e

        ph_swap = 1.0
        if ((tj_1 + tj_2 - tJ) // 2) % 2 != 0:
            ph_swap = -1.0

        amp = MJd - ph_swap * MJe
        total += (tJ + 1) * amp * amp

    return total


@nb.njit(cache=True)
def dw_excitation_sigma_sum(
    n_energies: int,
    kappa_inc_arr: np.ndarray,
    kappa_scat_arr: np.ndarray,
    k_arr: np.ndarray,
    slater_direct: np.ndarray,
    slater_exchange: np.ndarray,
    E_inc_hartree: np.ndarray,
    dE_hartree_grid: np.ndarray,
    tj_i: int,
) -> np.ndarray:
    """Compute the partial-wave T-matrix sum for DW excitation.

    Given precomputed Slater integrals for all (kappa_inc, kappa_scat, k)
    combinations at each energy point, assemble sigma(E).

    Parameters
    ----------
    n_energies : int
        Number of energy grid points.
    kappa_inc_arr, kappa_scat_arr : int arrays
        Lists of kappa values for incident and scattered electrons.
    k_arr : int array
        Multipole orders.
    slater_direct : ndarray, shape (n_energies, n_ki, n_ks, n_k)
        Precomputed R^k direct Slater integrals.  NaN = skip.
    slater_exchange : ndarray, shape (n_energies, n_ki, n_ks, n_k)
        Precomputed R^k exchange Slater integrals.  NaN = skip.
    E_inc_hartree : ndarray, shape (n_energies,)
        Incident energies in Hartree.
    dE_hartree_grid : ndarray, shape (n_energies,)
        Transition energies in Hartree.
    tj_i : int
        2*j for initial bound orbital.

    Returns
    -------
    sigma : ndarray, shape (n_energies,)
        Cross section in atomic units (a0^2).
    omega : ndarray, shape (n_energies,)
        Collision strength (dimensionless).
    """
    sigma = np.zeros(n_energies)
    omega = np.zeros(n_energies)
    n_ki = kappa_inc_arr.shape[0]
    n_ks = kappa_scat_arr.shape[0]
    n_k = k_arr.shape[0]

    for idx in range(n_energies):
        E_inc = E_inc_hartree[idx]
        dE = dE_hartree_grid[idx]
        E_scattered = E_inc - dE
        if E_scattered <= 0.0:
            continue
        k_i_sq = 2.0 * E_inc

        sigma_partial = 0.0
        for iki in range(n_ki):
            for iks in range(n_ks):
                T = 0.0
                for ik in range(n_k):
                    Rk_d = slater_direct[idx, iki, iks, ik]
                    Rk_e = slater_exchange[idx, iki, iks, ik]
                    if np.isnan(Rk_d):
                        Rk_d = 0.0
                    if np.isnan(Rk_e):
                        Rk_e = 0.0
                    T += Rk_d - Rk_e
                sigma_partial += T * T

        if k_i_sq > 0.0:
            sigma[idx] = np.pi / k_i_sq * sigma_partial / (tj_i + 1.0)
            omega[idx] = k_i_sq * (tj_i + 1.0) * sigma[idx] / np.pi

    return sigma, omega


@nb.njit(cache=True, fastmath=True)
def trapezoidal_integrate(y: np.ndarray, x: np.ndarray) -> float:
    """Numba-compatible trapezoidal integration."""
    n = y.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total
