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


@nb.njit(cache=True, fastmath=True)
def eii_dw_inner_sum(
    n_channels: int,
    tj_a: int,
    # Per-channel arrays (all length n_channels):
    tj_in_arr: np.ndarray,
    tj_1_arr: np.ndarray,
    tj_2_arr: np.ndarray,
    tJ_lo_arr: np.ndarray,
    tJ_hi_arr: np.ndarray,
    n_direct_arr: np.ndarray,
    n_exchange_arr: np.ndarray,
    # Flat arrays of precomputed direct/exchange products
    direct_products: np.ndarray,
    exchange_products: np.ndarray,
    # Flat arrays of precomputed wigner_6j values
    w6j_direct: np.ndarray,
    w6j_exchange: np.ndarray,
) -> float:
    """JIT-compiled EII DW inner loop: sum over all (kappa_in, kappa_1, kappa_2)
    channels and their J-sums for a single (E_i, E_1) quadrature point.

    The Python caller pre-computes all angular coefficients, Slater integrals,
    and wigner-6j values into flat arrays.  This kernel does the pure-numerical
    assembly with zero Python overhead.

    Returns the accumulated |T|^2 for this E_1 point.
    """
    acc = 0.0
    d_offset = 0
    e_offset = 0
    w6j_d_offset = 0
    w6j_e_offset = 0

    for ch in range(n_channels):
        tj_in = tj_in_arr[ch]
        tj_1 = tj_1_arr[ch]
        tj_2 = tj_2_arr[ch]
        tJ_lo = tJ_lo_arr[ch]
        tJ_hi = tJ_hi_arr[ch]
        n_d = n_direct_arr[ch]
        n_e = n_exchange_arr[ch]

        n_J = (tJ_hi - tJ_lo) // 2 + 1
        J_sum = 0.0

        for iJ in range(n_J):
            tJ = tJ_lo + 2 * iJ

            # Phase for direct: (-1)^{(j_in + j_1 + J)/2}
            phase_d = 1.0
            if ((tj_in + tj_1 + tJ) // 2) % 2 != 0:
                phase_d = -1.0

            MJd = 0.0
            for ik in range(n_d):
                sj = w6j_direct[w6j_d_offset + iJ * n_d + ik]
                if abs(sj) < 1e-15:
                    continue
                MJd += sj * direct_products[d_offset + ik]
            MJd *= phase_d

            # Phase for exchange: (-1)^{(j_in + j_2 + J)/2}
            phase_e = 1.0
            if ((tj_in + tj_2 + tJ) // 2) % 2 != 0:
                phase_e = -1.0

            MJe = 0.0
            for ik in range(n_e):
                sj = w6j_exchange[w6j_e_offset + iJ * n_e + ik]
                if abs(sj) < 1e-15:
                    continue
                MJe += sj * exchange_products[e_offset + ik]
            MJe *= phase_e

            # Phase swap: (-1)^{(j_1 + j_2 - J)/2}
            ph_swap = 1.0
            if ((tj_1 + tj_2 - tJ) // 2) % 2 != 0:
                ph_swap = -1.0

            amp = MJd - ph_swap * MJe
            J_sum += (tJ + 1) * amp * amp

        acc += J_sum

        d_offset += n_d
        e_offset += n_e
        w6j_d_offset += n_J * n_d
        w6j_e_offset += n_J * n_e

    return acc


@nb.njit(cache=True, fastmath=True)
def dw_excitation_inner_sum(
    n_channels: int,
    # Per-channel arrays (all length n_channels):
    tj_in_arr: np.ndarray,
    tj_out_arr: np.ndarray,
    tJ_lo_arr: np.ndarray,
    tJ_hi_arr: np.ndarray,
    n_direct_arr: np.ndarray,
    n_exchange_arr: np.ndarray,
    # Flat arrays of precomputed direct/exchange products
    direct_products: np.ndarray,
    exchange_products: np.ndarray,
    # Flat arrays of precomputed wigner_6j values
    w6j_direct: np.ndarray,
    w6j_exchange: np.ndarray,
) -> float:
    """JIT-compiled DW excitation inner loop: sum over all (kappa_in, kappa_out)
    channels and their J-sums for a single energy point.

    The Python caller pre-computes all angular coefficients (reduced C^k),
    Slater integrals, and wigner-6j values into flat arrays.  This kernel
    does the pure-numerical assembly with zero Python overhead.

    Returns the accumulated |T|^2 for this energy.
    """
    acc = 0.0
    d_offset = 0
    e_offset = 0
    w6j_d_offset = 0
    w6j_e_offset = 0

    for ch in range(n_channels):
        tj_in = tj_in_arr[ch]
        tj_out = tj_out_arr[ch]
        tJ_lo = tJ_lo_arr[ch]
        tJ_hi = tJ_hi_arr[ch]
        n_d = n_direct_arr[ch]
        n_e = n_exchange_arr[ch]

        n_J = (tJ_hi - tJ_lo) // 2 + 1
        J_sum = 0.0

        for iJ in range(n_J):
            tJ = tJ_lo + 2 * iJ

            # Direct amplitude: M_J^direct = (-1)^{(j_i + j_in + J)/2}
            #   * Σ_k 6j{j_bound_i j_in J; j_out j_bound_f k} * C_i_in * C_out_f * Rk_d
            phase_d = 1.0
            if ((tj_in_arr[ch] + tj_out_arr[ch] + tJ) // 2) % 2 != 0:
                phase_d = -1.0

            MJd = 0.0
            for ik in range(n_d):
                sj = w6j_direct[w6j_d_offset + iJ * n_d + ik]
                if abs(sj) < 1e-15:
                    continue
                MJd += sj * direct_products[d_offset + ik]
            MJd *= phase_d

            # Exchange amplitude: M_J^exchange = (-1)^{(j_in + j_out + J)/2}
            #   * Σ_k 6j{j_bound_i j_in J; j_bound_f j_out k} * C_i_out * C_in_f * Rk_e
            phase_e = 1.0
            if ((tj_in + tj_out + tJ) // 2) % 2 != 0:
                phase_e = -1.0

            MJe = 0.0
            for ik in range(n_e):
                sj = w6j_exchange[w6j_e_offset + iJ * n_e + ik]
                if abs(sj) < 1e-15:
                    continue
                MJe += sj * exchange_products[e_offset + ik]
            MJe *= phase_e

            amp = MJd - MJe
            J_sum += (tJ + 1) * amp * amp

        acc += J_sum

        d_offset += n_d
        e_offset += n_e
        w6j_d_offset += n_J * n_d
        w6j_e_offset += n_J * n_e

    return acc


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
