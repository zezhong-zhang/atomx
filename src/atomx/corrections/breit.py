"""Full Breit interaction energy correction.

The Breit operator in the multipole expansion has contributions at
three ranks v = L-1, L, L+1 for each multipole order L, with 8
angular components S(1)..S(8) corresponding to different pairings
of large (P) and small (Q) components.

The four types of Breit Slater integrals:
  S_ppqq^v(a,b) = ∫∫ P_a(r1)P_b(r1) [r_<^v/r_>^{v+1}] Q_a(r2)Q_b(r2) dr
  S_pqqp^v(a,b) = ∫∫ P_a(r1)Q_b(r1) [r_<^v/r_>^{v+1}] Q_a(r2)P_b(r2) dr
  (and symmetric partners obtained by swapping P↔Q)

Reference: Grant & Pyper, J.Phys.B 9, 761 (1976); DBSR_HF hf_energy_BR.f90.
"""

import numpy as np
from atomx.constants import ALPHA, C, C_SQ
from atomx.scf.slater import eval_orbital_on_grid, compute_yk, compute_slater_rk
from atomx.angular.wigner import wigner_3j
from atomx.scf.angular_coeffs import exchange_lambda, exchange_k_range


def _breit_smu_coefficients(kappa_a, kappa_b, kappa_c, kappa_d, L):
    """Compute the 8 Breit angular coefficients S(1)..S(8) for multipole L.

    Returns dict with keys v: L-1, L, L+1 → list of 8 S-coefficients.
    Based on DBSR_HF SMU function (hf_energy_BR.f90 lines 79-115).
    """
    K = kappa_c - kappa_a
    Kp = kappa_d - kappa_b
    results = {}

    # v = L: diagonal term
    if L > 0:
        smu = -(kappa_a + kappa_c) * (kappa_b + kappa_d) / (L * (L + 1))
        results[L] = [smu] * 8  # all 8 components equal

    # v = L-1
    if L >= 2:
        bp = (L + 1) / (2 * (2 * L - 1))
        cp = -(L - 2) / (2 * L * (2 * L - 1))
        s = [0.0] * 8
        s[0] = (L + K) * (bp + cp * Kp)
        s[1] = (L + Kp) * (bp + cp * K)
        s[2] = (L - K) * (bp - cp * Kp)
        s[3] = (L - Kp) * (bp - cp * K)
        s[4] = -(L + K) * (bp - cp * Kp)
        s[5] = -(L - Kp) * (bp + cp * K)
        s[6] = -(L - K) * (bp + cp * Kp)
        s[7] = -(L + Kp) * (bp - cp * K)
        results[L - 1] = s

    # v = L+1
    b = L / (2 * (2 * L + 3))
    c = (L + 3) / ((2 * L + 2) * (2 * L + 3))
    il = -L - 1
    s = [0.0] * 8
    s[0] = (il + Kp) * (b + c * K)
    s[1] = (il + K) * (b + c * Kp)
    s[2] = (il - Kp) * (b - c * K)
    s[3] = (il - K) * (b - c * Kp)
    s[4] = -(il - Kp) * (b + c * K)
    s[5] = -(il + K) * (b - c * Kp)
    s[6] = -(il + Kp) * (b - c * K)
    s[7] = -(il - K) * (b + c * Kp)
    results[L + 1] = s

    return results


def compute_breit_energy(orbitals, bsp_data, Z):
    """Compute full Breit interaction energy.

    Includes all 8 angular components at each multipole L,
    with contributions at v = L-1, L, L+1.

    The 8 integrals correspond to the four density pairings:
    S1..S4: "direct" type — S^v(P_a P_b, Q_a Q_b) etc.
    S5..S8: "exchange" type — S^v(P_a Q_b, Q_a P_b) etc.
    """
    nv = bsp_data["nv"]
    ns = bsp_data["ns"]
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    ksp = bsp_data.get("ksp", bsp_data["ks"])
    ksq = bsp_data.get("ksq", bsp_data["ks"])
    nsp = bsp_data.get("nsp", ns)
    nsq = bsp_data.get("nsq", ns)
    bsp_p = bsp_data["bsp_p"]
    bsp_q = bsp_data.get("bsp_q", bsp_p)
    n_gauss = gr.shape[1]

    E_Breit = 0.0

    for i, (Pa, Qa, oa, ka, la) in enumerate(orbitals):
        tj_a = 2 * abs(ka) - 1
        fa_P = eval_orbital_on_grid(Pa, bsp_p, nv, ksp, nsp)
        fa_Q = eval_orbital_on_grid(Qa, bsp_q, nv, ksq, nsq)

        for j, (Pb, Qb, ob, kb, lb) in enumerate(orbitals):
            tj_b = 2 * abs(kb) - 1
            fb_P = eval_orbital_on_grid(Pb, bsp_p, nv, ksp, nsp)
            fb_Q = eval_orbital_on_grid(Qb, bsp_q, nv, ksq, nsq)

            # Weight for average-of-configuration
            w_ab = oa * ob / ((tj_a + 1.0) * (tj_b + 1.0))

            # Loop over multipole L
            L_max = abs(ka) + abs(kb)
            for L in range(1, L_max + 1):
                # Check angular selection rule: 3j triangle for (j_a, L, j_b)
                if 2 * L > tj_a + tj_b or 2 * L < abs(tj_a - tj_b):
                    continue

                # Get 8 S-coefficients for each v = L-1, L, L+1
                smu_dict = _breit_smu_coefficients(ka, kb, ka, kb, L)

                for v, s_coeffs in smu_dict.items():
                    if v < 0:
                        continue
                    if all(abs(s) < 1e-15 for s in s_coeffs):
                        continue

                    # Parity check on v
                    l_a = ka if ka > 0 else -ka - 1
                    l_b = kb if kb > 0 else -kb - 1

                    # Compute the 8 Breit integrals for this (a,b,v):
                    # Direct-type densities (indices 0-3):
                    #   PP density at r1, QQ density at r2 (and swaps)
                    # Exchange-type densities (indices 4-7):
                    #   PQ density at r1, QP density at r2 (and swaps)

                    # For efficiency, compute the needed Y^k potentials
                    # S1: <PaPb | v | QaQb>
                    rho_PP = fa_P * fb_P
                    rho_QQ = fa_Q * fb_Q
                    rho_PQ = fa_P * fb_Q
                    rho_QP = fa_Q * fb_P

                    yk_QQ = compute_yk(rho_QQ, gr, grw, nv, n_gauss, v)
                    yk_PP = compute_yk(rho_PP, gr, grw, nv, n_gauss, v)
                    yk_QP = compute_yk(rho_QP, gr, grw, nv, n_gauss, v)
                    yk_PQ = compute_yk(rho_PQ, gr, grw, nv, n_gauss, v)

                    # Direct integrals (S1-S4 pattern)
                    R_ppqq = np.sum(grw * rho_PP * yk_QQ)  # S1, S2
                    R_qqpp = np.sum(grw * rho_QQ * yk_PP)  # S3, S4
                    # Exchange integrals (S5-S8 pattern)
                    R_pqqp = np.sum(grw * rho_PQ * yk_QP)  # S5, S6
                    R_qppq = np.sum(grw * rho_QP * yk_PQ)  # S7, S8

                    # Combine with angular coefficients
                    contrib = (s_coeffs[0] * R_ppqq + s_coeffs[1] * R_ppqq
                             + s_coeffs[2] * R_qqpp + s_coeffs[3] * R_qqpp
                             + s_coeffs[4] * R_pqqp + s_coeffs[5] * R_pqqp
                             + s_coeffs[6] * R_qppq + s_coeffs[7] * R_qppq)

                    E_Breit -= w_ab * contrib / C_SQ

    return 0.5 * E_Breit
