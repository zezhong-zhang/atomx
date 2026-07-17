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

Radial quadrature (2026-07 fix)
-------------------------------
The Y^v assembly uses the canonical arbitrary-grid kernel
:func:`atomx.numerics.quadrature.yk_on_grid` (trapezoidal cumulatives,
second order on any grid) on the flattened radial nodes of whichever
workspace the orbitals live on (B-spline Gauss nodes or a shooting
grid).  The previous implementation used the B-spline SCF helper
``compute_yk``, whose rectangle-rule cumulatives double-count the
diagonal cell (r' = r) — the exact O(h) bias class documented in
``numerics/quadrature.py`` from the Auger verification campaign.  For
the strongly cancelling Breit combinations that bias inflated the total
by ~3%: Ne I (point nucleus) gave 0.05197 Ha where the dense-grid
converged value is 0.050429 Ha (cubic-spline resampling of the same
DHF orbitals onto 5k/20k/80k-point grids, all agreeing to 1e-6, and
independently reproduced at 0.05044 Ha by the ODE-shooting engine).
"""

import numpy as np
from atomx.corrections.workspace import (
    eval_orbital_pair,
    workspace_arrays,
)
from atomx.numerics.quadrature import yk_on_grid


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

    Parameters
    ----------
    orbitals : list of (P, Q, occ, kappa, label) tuples
    bsp_data : dict
        B-spline workspace (``P``/``Q`` are coefficient vectors), or a
        radial-grid workspace from
        :func:`atomx.corrections.workspace.make_grid_workspace`
        (``P``/``Q`` are grid arrays) — e.g. the ODE-shooting engine.
    Z : int
    """
    gr, grw = workspace_arrays(bsp_data)
    # Flatten to 1D grid order for the arbitrary-grid Y^v kernel
    # (B-spline workspaces carry (nv, n_gauss) arrays; the flattened
    # Gauss nodes are strictly increasing).
    r_flat = np.ravel(gr)
    w_flat = np.ravel(grw)

    # Evaluate every orbital on the radial nodes once.
    evaluated = []
    for (Pa, Qa, oa, ka, la) in orbitals:
        fP, fQ = eval_orbital_pair(Pa, Qa, bsp_data)
        evaluated.append((np.ravel(fP), np.ravel(fQ), oa, ka, la))

    E_Breit = 0.0

    for i, (fa_P, fa_Q, oa, ka, la) in enumerate(evaluated):
        tj_a = 2 * abs(ka) - 1

        for j, (fb_P, fb_Q, ob, kb, lb) in enumerate(evaluated):
            tj_b = 2 * abs(kb) - 1

            # Weight for average-of-configuration
            w_ab = oa * ob / ((tj_a + 1.0) * (tj_b + 1.0))

            # The four cross densities are L/v-independent.
            rho_PP = fa_P * fb_P
            rho_QQ = fa_Q * fb_Q
            rho_PQ = fa_P * fb_Q
            rho_QP = fa_Q * fb_P

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

                    # Compute the 8 Breit integrals for this (a,b,v):
                    # Direct-type densities (indices 0-3):
                    #   PP density at r1, QQ density at r2 (and swaps)
                    # Exchange-type densities (indices 4-7):
                    #   PQ density at r1, QP density at r2 (and swaps)
                    yk_QQ = yk_on_grid(rho_QQ, r_flat, v)
                    yk_PP = yk_on_grid(rho_PP, r_flat, v)
                    yk_QP = yk_on_grid(rho_QP, r_flat, v)
                    yk_PQ = yk_on_grid(rho_PQ, r_flat, v)

                    # Direct integrals (S1-S4 pattern)
                    R_ppqq = np.sum(w_flat * rho_PP * yk_QQ)  # S1, S2
                    R_qqpp = np.sum(w_flat * rho_QQ * yk_PP)  # S3, S4
                    # Exchange integrals (S5-S8 pattern)
                    R_pqqp = np.sum(w_flat * rho_PQ * yk_QP)  # S5, S6
                    R_qppq = np.sum(w_flat * rho_QP * yk_PQ)  # S7, S8

                    # Combine with angular coefficients
                    contrib = (s_coeffs[0] * R_ppqq + s_coeffs[1] * R_ppqq
                             + s_coeffs[2] * R_qqpp + s_coeffs[3] * R_qqpp
                             + s_coeffs[4] * R_pqqp + s_coeffs[5] * R_pqqp
                             + s_coeffs[6] * R_qppq + s_coeffs[7] * R_qppq)

                    # Breit Slater integrals on the native Dirac spinor
                    # (P, Q) already carry the Q/c ~ Zα suppression: the
                    # PPQQ, PQQP, etc. combinations are intrinsically
                    # O(α²) relative to Coulomb.  The DBSR_HF reference
                    # (``hf_energy_BR.f90``) accumulates ``crk·sk_ppqq``
                    # without any further c² division.  atomx's original
                    # ``/ C_SQ`` factor therefore over-suppressed Breit
                    # by another α² (Z=26 He-like gave 3.8e-5 Ha instead
                    # of ~0.7 Ha).
                    E_Breit -= w_ab * contrib

    return 0.5 * E_Breit
