"""Numba-JIT compiled inner-loop kernels for scattering calculations.

These functions extract the tight numerical loops from the DW excitation,
EII DW, and Auger rate computations so that Python-level overhead is
eliminated from the innermost iterations.

All functions here are pure-numerical (no Python objects) and decorated
with ``@nb.njit`` for ahead-of-time compilation.
"""

from __future__ import annotations

import numba as nb
import numpy as np


@nb.njit(cache=True)
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

    No ``fastmath``: measured 0.95× (i.e. 5 % *slower*, 0.50 µs vs 0.47 µs)
    on a 20-channel table, so the flag bought nothing and only perturbed
    the accumulation order.

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


@nb.njit(cache=True, nogil=True)
def eii_dw_channel_jsum(
    Rd: np.ndarray,          # direct Slater integrals, one per value slot
    Re: np.ndarray,          # exchange Slater integrals, one per value slot
    ok_in: np.ndarray,       # per-kappa wave availability at E_i / E_1 / E_2
    ok_1: np.ndarray,
    ok_2: np.ndarray,
    ch_in: np.ndarray,       # per-channel kappa indices + incident l
    ch_1: np.ndarray,
    ch_2: np.ndarray,
    ch_lin: np.ndarray,
    ch_jr0: np.ndarray,      # [jr0, jr1) J-row span per channel
    ch_jr1: np.ndarray,
    jw: np.ndarray,          # (tJ + 1) per J row
    jd0: np.ndarray,         # [d0, d1) direct-entry span per J row
    jd1: np.ndarray,
    je0: np.ndarray,         # [e0, e1) exchange-entry span per J row
    je1: np.ndarray,
    d_slot: np.ndarray,      # (value slot, angular) per entry
    d_ang: np.ndarray,
    e_slot: np.ndarray,
    e_ang: np.ndarray,
    n_l: int,
) -> np.ndarray:
    """EII DW J-sum over the packed channel table for one (E_i, E_1) node.

    Accumulation order matches the original per-channel Python loop
    exactly (channels, then J rows, then entries) — value invariance.
    No fastmath: T = D - E must round identically.
    """
    prof_l = np.zeros(n_l)
    for c in range(ch_in.shape[0]):
        if not (ok_in[ch_in[c]] and ok_1[ch_1[c]] and ok_2[ch_2[c]]):
            continue
        acc = 0.0
        for jr in range(ch_jr0[c], ch_jr1[c]):
            D = 0.0
            for e in range(jd0[jr], jd1[jr]):
                D += Rd[d_slot[e]] * d_ang[e]
            E_x = 0.0
            for e in range(je0[jr], je1[jr]):
                E_x += Re[e_slot[e]] * e_ang[e]
            # Slater-Condon for the antisymmetrized outgoing pair:
            # T = D - E exactly (see _eii_dw_total).
            T = D - E_x
            if T != 0.0:
                acc += jw[jr] * T * T
        prof_l[ch_lin[c]] += acc
    return prof_l


@nb.njit(cache=True)
def dw_excitation_angular_sum(
    n_channels: int,
    tj_i: int,
    tj_f: int,
    tj_in_arr: np.ndarray,
    tj_out_arr: np.ndarray,
    tJ_lo_arr: np.ndarray,
    tJ_hi_arr: np.ndarray,
    n_direct_arr: np.ndarray,
    n_exchange_arr: np.ndarray,
    direct_products: np.ndarray,
    exchange_products: np.ndarray,
    w6j_direct: np.ndarray,
    w6j_exchange: np.ndarray,
) -> float:
    """JIT-compiled DW excitation angular sum.

    For one energy point, sum over all (κ_in, κ_out) channels and their
    J-sums to compute Σ_{ch,J} (2J+1)|A_J|², the collision strength.

    Following Gu 2008 / Grant Ch. 12 j-j coupled angular reduction, ``tj_i``
    and ``tj_f`` are twice the target-state angular momenta, while ``tj_in``
    and ``tj_out`` are projectile partial-wave angular momenta:
      M_J^direct  = (-1)^{(j_in+J_f+J)/2} Σ_k 6j(J_i,j_in,J,j_out,J_f,k) · D_k
      M_J^exchange = (-1)^{(j_in+j_out+J)/2} Σ_k 6j(J_i,j_in,J,J_f,j_out,k) · E_k
      A_J = M_J^direct − (-1)^{(J_f+j_out−J)/2} · M_J^exchange

    where D_k = ⟨target_i‖C^k‖target_f⟩·⟨κ_in‖C^k‖κ_out⟩·R^k(i,in;f,out)
    and   E_k = ⟨κ_i‖C^k‖κ_out⟩·⟨κ_in‖C^k‖κ_f⟩·R^k(i,in;out,f).
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

            # Direct phase: (-1)^{(j_in + j_f + J)}
            phase_d = 1.0
            if ((tj_in + tj_f + tJ) // 2) % 2 != 0:
                phase_d = -1.0

            MJd = 0.0
            for ik in range(n_d):
                sj = w6j_direct[w6j_d_offset + iJ * n_d + ik]
                if abs(sj) < 1e-15:
                    continue
                MJd += sj * direct_products[d_offset + ik]
            MJd *= phase_d

            # Exchange phase: (-1)^{(j_in + j_out + J)}
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

            # Antisymmetrization phase: (-1)^{(j_f + j_out - J)}
            ph_swap = 1.0
            if ((tj_f + tj_out - tJ) // 2) % 2 != 0:
                ph_swap = -1.0

            amp = MJd - ph_swap * MJe
            J_sum += (tJ + 1) * amp * amp

        acc += J_sum

        d_offset += n_d
        e_offset += n_e
        w6j_d_offset += n_J * n_d
        w6j_e_offset += n_J * n_e

    return acc


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

    Returns sigma (a0^2) and omega (dimensionless) arrays.
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
    """Numba-compatible trapezoidal integration.

    ``fastmath`` is a measured 2.2× win here (0.17 µs vs 0.37 µs): the
    contiguous two-term-per-node reduction vectorises once reassociation
    is allowed, and every other fastmath site in this module was dropped
    for lack of such a win.
    """
    n = y.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n - 1):
        total += 0.5 * (y[i] + y[i + 1]) * (x[i + 1] - x[i])
    return total
