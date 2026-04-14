"""Evaluate B-spline orbital expansions and Y^k potentials on grid.

These utilities are used throughout the pipeline (scf, mcdhf, transitions,
collisions, corrections) to convert B-spline coefficient vectors into
function values on the quadrature grid.

Performance note: _compute_yk_flat is @nb.njit compiled.
The exchange matrix kernel remains in scf/slater.py (the true hot path).
"""

import numpy as np
import numba as nb


# ---------------------------------------------------------------------------
# Y^k potential: outward/inward cumulative integration (Numba)
# ---------------------------------------------------------------------------

@nb.njit(cache=True)
def _compute_yk_flat(rho_flat, r_flat, w_flat, n, k):
    """Y^k on flattened grid. Sequential cumsum — inherently serial."""
    # Pre-compute r^k and r^{-(k+1)} once
    rk = np.empty(n)
    rkm = np.empty(n)
    for i in range(n):
        rk[i] = r_flat[i] ** k
        rkm[i] = r_flat[i] ** (-(k + 1))

    # Outward: I_out(r) = integral_0^r rho r'^k dr'
    i_out = np.empty(n)
    s = 0.0
    for i in range(n):
        s += w_flat[i] * rho_flat[i] * rk[i]
        i_out[i] = s

    # Inward: I_in(r) = integral_r^inf rho r'^{-(k+1)} dr'
    i_in = np.empty(n)
    s = 0.0
    for i in range(n - 1, -1, -1):
        s += w_flat[i] * rho_flat[i] * rkm[i]
        i_in[i] = s

    # Y^k(r) = r^{-(k+1)} I_out + r^k I_in
    yk = np.empty(n)
    for i in range(n):
        yk[i] = rkm[i] * i_out[i] + rk[i] * i_in[i]
    return yk


def compute_yk(rho, gr, grw, nv, n_gauss, k):
    """Compute Y^k potential via outward/inward integration."""
    yk_flat = _compute_yk_flat(rho.ravel(), gr.ravel(), grw.ravel(),
                                nv * n_gauss, k)
    return yk_flat.reshape(nv, n_gauss)


# ---------------------------------------------------------------------------
# Orbital evaluation on grid (vectorized numpy)
# ---------------------------------------------------------------------------

def eval_orbital_on_grid(coeffs, bsp, nv, ks, ns):
    """Evaluate B-spline expansion on quadrature grid.

    f(iv, m) = sum_s coeffs[iv+s] * bsp(iv, m, s)

    Uses numpy einsum for vectorized evaluation.
    """
    gi = np.arange(nv)[:, None] + np.arange(ks)[None, :]  # (nv, ks)
    gi_safe = np.clip(gi, 0, ns - 1)
    c = np.where(gi < ns, coeffs[gi_safe], 0.0)  # (nv, ks)
    return np.einsum('is,ims->im', c, bsp)


# ---------------------------------------------------------------------------
# Component evaluation helpers (KB-aware)
# ---------------------------------------------------------------------------

def eval_P(coeffs, bsp_data):
    """Evaluate P-component (large) on grid, KB-aware."""
    ksp = bsp_data.get("ksp", bsp_data["ks"])
    nsp = bsp_data.get("nsp", bsp_data["ns"])
    return eval_orbital_on_grid(coeffs, bsp_data["bsp_p"],
                                 bsp_data["nv"], ksp, nsp)


def eval_Q(coeffs, bsp_data):
    """Evaluate Q-component (small) on grid, KB-aware."""
    ksq = bsp_data.get("ksq", bsp_data["ks"])
    nsq = bsp_data.get("nsq", bsp_data["ns"])
    bsp_q = bsp_data.get("bsp_q", bsp_data["bsp_p"])
    return eval_orbital_on_grid(coeffs, bsp_q, bsp_data["nv"], ksq, nsq)


# ---------------------------------------------------------------------------
# General Slater integral R^k (KB-aware)
# ---------------------------------------------------------------------------

def compute_slater_rk(Pa, Qa, Pb, Qb, Pc, Qc, Pd, Qd, bsp_data, k):
    """General Slater integral R^k(ab,cd) — KB-aware.

    Uses ksp-order for P components, ksq-order for Q components.
    """
    gr, grw = bsp_data["gr"], bsp_data["grw"]
    nv = bsp_data["nv"]

    fa = eval_P(Pa, bsp_data)
    ga = eval_Q(Qa, bsp_data)
    fb = eval_P(Pb, bsp_data)
    gb = eval_Q(Qb, bsp_data)
    fc = eval_P(Pc, bsp_data)
    gc = eval_Q(Qc, bsp_data)
    fd = eval_P(Pd, bsp_data)
    gd = eval_Q(Qd, bsp_data)

    rho_ab = fa * fb + ga * gb
    rho_cd = fc * fd + gc * gd
    yk_cd = compute_yk(rho_cd, gr, grw, nv, gr.shape[1], k)
    return np.sum(grw * rho_ab * yk_cd)
