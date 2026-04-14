"""Two-electron Slater integrals and exchange matrix kernel.

Performance-critical module — the exchange matrix computation is a single
Numba kernel eliminating all Python-level loop overhead.

Shared utilities (eval_orbital_on_grid, compute_yk, compute_slater_rk) have
been extracted to bspline.evaluate and are re-exported here for compatibility.
"""

import numpy as np
import numba as nb

# Re-export shared utilities (moved to bspline.evaluate)
from atomx.bspline.evaluate import (  # noqa: F401
    _compute_yk_flat,
    compute_yk,
    eval_orbital_on_grid,
    eval_P as _eval_P,
    eval_Q as _eval_Q,
    compute_slater_rk,
)


# ---------------------------------------------------------------------------
# Exchange matrix: FULL Numba kernel (the critical hot path)
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _exchange_matrix_kernel(fb, bsp, gr_flat, grw_flat, nv, n_gauss, ks, ns, k):
    """Compute the full ns×ns exchange matrix K^k in a single Numba call.

    K_{m,n} = ∫ [B_m(r)*fb(r)] * Y^k_{fb·B_n}(r) dr

    All loops native — no Python overhead.

    Parameters
    ----------
    fb : ndarray (nv, n_gauss)  — orbital on grid
    bsp : ndarray (nv, n_gauss, ks) — B-spline values
    gr_flat, grw_flat : ndarray (nv*n_gauss,) — flat grid
    nv, n_gauss, ks, ns, k : int
    """
    total_pts = nv * n_gauss
    K = np.zeros((ns, ns))

    # Pre-compute r^k and r^{-(k+1)}
    rk = np.empty(total_pts)
    rkm = np.empty(total_pts)
    for i in range(total_pts):
        rk[i] = gr_flat[i] ** k
        rkm[i] = gr_flat[i] ** (-(k + 1))

    # Pre-compute integrand: w(r) * B_m_local(r) * fb(r) for each (iv, gauss, s)
    # integrand[iv*n_gauss+g, s] = grw * bsp * fb
    integrand = np.empty((total_pts, ks))
    for iv in range(nv):
        for g in range(n_gauss):
            idx = iv * n_gauss + g
            wfb = grw_flat[idx] * fb[iv, g]
            for s in range(ks):
                integrand[idx, s] = wfb * bsp[iv, g, s]

    # Product density buffer (reused per n_idx)
    rho = np.empty(total_pts)

    for n_idx in range(ns):
        # Build product density: rho(r) = fb(r) * B_n(r)
        # B_n is nonzero in intervals [max(0, n-ks+1), min(nv-1, n)]
        for i in range(total_pts):
            rho[i] = 0.0
        iv_lo = n_idx - ks + 1
        if iv_lo < 0:
            iv_lo = 0
        iv_hi = n_idx + 1
        if iv_hi > nv:
            iv_hi = nv

        for iv in range(iv_lo, iv_hi):
            local_s = n_idx - iv
            for g in range(n_gauss):
                idx = iv * n_gauss + g
                rho[idx] = fb[iv, g] * bsp[iv, g, local_s]

        # Y^k of product density (outward/inward integration)
        # Outward
        i_out = np.empty(total_pts)
        s_val = 0.0
        for i in range(total_pts):
            s_val += grw_flat[i] * rho[i] * rk[i]
            i_out[i] = s_val

        # Inward
        i_in = np.empty(total_pts)
        s_val = 0.0
        for i in range(total_pts - 1, -1, -1):
            s_val += grw_flat[i] * rho[i] * rkm[i]
            i_in[i] = s_val

        # Y^k
        yk = np.empty(total_pts)
        for i in range(total_pts):
            yk[i] = rkm[i] * i_out[i] + rk[i] * i_in[i]

        # Assemble: K[m, n] += Σ_g integrand[iv*ng+g, s] * yk[iv*ng+g]
        # where m = iv + s
        for iv in range(nv):
            for s in range(ks):
                m_idx = iv + s
                if m_idx < ns:
                    acc = 0.0
                    for g in range(n_gauss):
                        acc += integrand[iv * n_gauss + g, s] * yk[iv * n_gauss + g]
                    K[m_idx, n_idx] += acc

    return K


def compute_exchange_matrix(orb_b_P, orb_b_Q, bsp_data, k):
    """Non-local exchange matrix K^k — single Numba kernel.

    Handles kinetic balance: K_PP uses ksp-order basis, K_QQ uses ksq-order.
    """
    nv = bsp_data["nv"]
    ns = bsp_data["ns"]
    gr = bsp_data["gr"]
    grw = bsp_data["grw"]
    n_gauss = gr.shape[1]
    gr_flat = gr.ravel().copy()
    grw_flat = grw.ravel().copy()

    # P-block: use ksp-order B-splines
    ksp = bsp_data.get("ksp", bsp_data["ks"])
    nsp = bsp_data.get("nsp", ns)
    bsp_p = bsp_data["bsp_p"]
    fb_P = eval_orbital_on_grid(orb_b_P, bsp_p, nv, ksp, nsp)
    K_PP = _exchange_matrix_kernel(fb_P, bsp_p, gr_flat, grw_flat, nv, n_gauss, ksp, nsp, k)

    # Q-block: use ksq-order B-splines
    ksq = bsp_data.get("ksq", bsp_data["ks"])
    nsq = bsp_data.get("nsq", ns)
    bsp_q = bsp_data.get("bsp_q", bsp_p)
    fb_Q = eval_orbital_on_grid(orb_b_Q, bsp_q, nv, ksq, nsq)
    K_QQ = _exchange_matrix_kernel(fb_Q, bsp_q, gr_flat, grw_flat, nv, n_gauss, ksq, nsq, k)

    return K_PP, K_QQ


# ---------------------------------------------------------------------------
# Slater integrals F^k and G^k (vectorized numpy)
# ---------------------------------------------------------------------------

def compute_slater_fk(orb_a_P, orb_a_Q, orb_b_P, orb_b_Q, bsp_data, k):
    """Direct Slater integral F^k(a,b) = R^k(aa,bb)."""
    return compute_slater_rk(orb_a_P, orb_a_Q, orb_a_P, orb_a_Q,
                              orb_b_P, orb_b_Q, orb_b_P, orb_b_Q, bsp_data, k)


def compute_slater_gk(orb_a_P, orb_a_Q, orb_b_P, orb_b_Q, bsp_data, k):
    """Exchange Slater integral G^k(a,b) = R^k(ab,ba)."""
    return compute_slater_rk(orb_a_P, orb_a_Q, orb_b_P, orb_b_Q,
                              orb_b_P, orb_b_Q, orb_a_P, orb_a_Q, bsp_data, k)
