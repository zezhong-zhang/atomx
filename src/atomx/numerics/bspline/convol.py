"""Cell-based B-spline two-electron integrals (DBSR_HF algorithm).

Implements the cell algorithm for R^k integrals:
    R^k(ab,cd) = ∫∫ ρ_ab(r₁) [r_<^k / r_>^{k+1}] ρ_cd(r₂) dr₁ dr₂

Algorithm (Zatsarinny, CPC 174, 728, 2006, Sec. 3.2):
  1. Compute 1D moments <B_i B_j | r^k> per knot interval
  2. For diagonal cells (both electrons in same interval): nested 2D quadrature
     via the 'triang' algorithm (exact for the B-spline representation)
  3. For off-diagonal cells (electrons in different intervals): factor into
     product of 1D moments  moment_k × moment_{-(k+1)}
  4. Assemble full rkb(i, j, ip, jp) array in symmetric upper-band storage
  5. Contract rkb with density matrices to get Fock contributions

Four integral types handle kinetic balance (P vs Q components):
  type 1: pppp  (both pairs P-basis)
  type 2: qqqq  (both pairs Q-basis)
  type 3: pqpq  (left P, right Q)
  type 4: qpqp  (left Q, right P)

Convolution modes (icase):
  1: R(. a ; . b)  — contract right pair → banded result (direct)
  2: R(a . ; b .)  — contract left pair  → banded result (direct)
  3: R(. a ; b .)  — mixed contraction   → full result   (exchange)
  4: R(a . ; . b)  — mixed contraction   → full result   (exchange)

Fastmath policy: ``fastmath=True`` is kept only where a per-kernel A/B
measurement (same kernel, identical inputs, compiled with and without the
flag at ns=113, ks=9, kinetic balance) shows a >1.2× win.  That leaves
exactly three kernels in this module — ``_convol_case1_t`` (3.4×, the
batch-DHF hot path), ``_sum_amb_full`` (3.9×) and
``_accumulate_direct_density_batch`` (1.23×).  Every other site was
neutral or actively harmful *and* perturbs results bitwise, so the flag
was removed: ``_convol_case1_mixed`` 0.69×, ``_convol_case1`` 0.77×,
``_sum_amb_banded`` 0.77×, ``_convol_case4`` 0.78×,
``_convol_case2_mixed`` 0.79×, ``_convol_case2`` 0.83×, and
``_convol_case4_t``/``_density_full``/``_banded_to_full_add`` ≈1.0×
(1.00 = equal speed).  Consequences for numerics are documented at each
kernel: with the flag gone, ``_convol_case4_t`` now reproduces
``_convol_case4`` bit-for-bit (verified on Fe-size banks), while
``_convol_case1_t`` — which keeps the flag for its 3.4× win — differs
from ``_convol_case1`` by ~5e-15 relative per element.
"""

from collections import OrderedDict
from collections.abc import Mapping

import numba as nb
import numpy as np
from numba import prange as _prange

# ══════════════════════════════════════════════════════════════════════
# 1. Moments: <B_i B_j | r^k> per interval
# ══════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def _compute_moments_sym(bsp, gr, grw, nv, ks, k):
    """Symmetric moments: packed upper-triangle (i, j) with j >= i.

    rkm[ii, iv] = Σ_m grw[iv,m] * gr[iv,m]^k * B_{i}(m) * B_{j}(m)

    ii packs (i, j) with j >= i into linear index:
      ii(i, j) = i*ks - i*(i-1)//2 + (j - i)  [0-indexed]
    Total entries: kk = ks*(ks+1)//2

    Returns rkm: shape (kk, nv)
    """
    n_gauss = gr.shape[1]
    kk = ks * (ks + 1) // 2
    rkm = np.zeros((kk, nv))

    for iv in range(nv):
        # Gauss weights × r^k
        gw = np.empty(n_gauss)
        for m in range(n_gauss):
            if k > 0:
                gw[m] = grw[iv, m] * gr[iv, m] ** k
            elif k < 0:
                gw[m] = grw[iv, m] / gr[iv, m] ** (-k)
            else:
                gw[m] = grw[iv, m]

        ii = 0
        for i in range(ks):
            for j in range(i, ks):
                s = 0.0
                for m in range(n_gauss):
                    s += bsp[iv, m, i] * bsp[iv, m, j] * gw[m]
                rkm[ii, iv] = s
                ii += 1

    return rkm


@nb.njit(cache=True)
def _compute_moments_nonsym(bsp1, bsp2, gr, grw, nv, ks1, ks2, k):
    """Non-symmetric moments for mixed-basis types (pqpq, qpqp).

    rkm[ii, iv] where ii packs all (i, j) with i=0..ks1-1, j=0..ks2-1.
    Total entries: kk = ks1 * ks2

    Returns rkm: shape (kk, nv)
    """
    n_gauss = gr.shape[1]
    kk = ks1 * ks2
    rkm = np.zeros((kk, nv))

    for iv in range(nv):
        gw = np.empty(n_gauss)
        for m in range(n_gauss):
            if k > 0:
                gw[m] = grw[iv, m] * gr[iv, m] ** k
            elif k < 0:
                gw[m] = grw[iv, m] / gr[iv, m] ** (-k)
            else:
                gw[m] = grw[iv, m]

        ii = 0
        for i in range(ks1):
            for j in range(ks2):
                s = 0.0
                for m in range(n_gauss):
                    s += bsp1[iv, m, i] * bsp2[iv, m, j] * gw[m]
                rkm[ii, iv] = s
                ii += 1

    return rkm


# ══════════════════════════════════════════════════════════════════════
# 2. Diagonal cell (triang): nested 2D quadrature
# ══════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def _triang_sym(bsp, gr, grw, t, nv, ks, ns, k):
    """Diagonal cell R^k integrals for all intervals (symmetric basis).

    Exact reproduction of DBSR_HF triang_pppp algorithm:
    For interval iv where both electrons are in the same cell:
    rkd[ii, jj, iv] = ∫∫_{cell iv} B_i*B_{ip}(r₁) [r_<^k/r_>^{k+1}] B_j*B_{jp}(r₂) dr₁ dr₂

    Algorithm:
    - Outer loop: Gauss points m in interval iv (r₂ = gr[iv,m])
    - Inner: sub-quadrature from xbase to gr[iv,m] (r₁ direction)
    - INTP[j, jp, m] = ∫_{xbase}^{gr[iv,m]} B_j(r₁)*B_jp(r₁)*r₁^k dr₁
    - Combine: a[ii,jj] = Σ_m B_i(m)*B_ip(m)*grw(m)/r₂^{k+1} * INTP[j,jp,m]
    - Symmetrize: rkd = a + a^T (swaps r_< and r_> roles)

    Parameters
    ----------
    bsp : (nv, n_gauss, ks)  -- B-spline values at main Gauss points
    gr, grw : (nv, n_gauss)  -- main quadrature points and weights
    t : knot sequence (use tp for P-component when KB is active)
    nv, ks, ns, k : int

    Returns
    -------
    rkd : (kk, kk, nv) where kk = ks*(ks+1)//2
    """
    n_gauss = gr.shape[1]
    kk = ks * (ks + 1) // 2
    rkd = np.zeros((kk, kk, nv))

    # Sub-quadrature: ks Gauss-Legendre points on [0, 1] (DBSR_HF convention)
    n_sub = ks
    x_01 = np.empty(n_sub)
    w_01 = np.empty(n_sub)
    _gauss_legendre_01(n_sub, x_01, w_01)

    # Pre-allocate work arrays (reused per interval)
    bsp_sub = np.empty((n_sub, ks))
    gw_rk = np.empty(n_sub)
    INTP = np.empty((ks, ks, n_gauss))
    ow = np.empty(n_gauss)
    bw = np.empty(n_gauss)

    for iv in range(nv):
        # xbase = left boundary of interval iv (DBSR_HF: tp(iv + ksp - 1))
        left = ks - 1 + iv
        xbase = t[left]

        # --- Inner integration: compute INTP[j, jp, m] ---
        for m in range(n_gauss):
            r_upper = gr[iv, m]
            span = r_upper - xbase

            if span < 1e-30:
                for j in range(ks):
                    for jp in range(j, ks):
                        INTP[j, jp, m] = 0.0
                continue

            # Map [0,1] sub-quadrature -> [xbase, gr(iv,m)]
            # Evaluate B-splines at each sub-point (DBSR_HF: calls zbsplvd)
            for n in range(n_sub):
                gx_n = xbase + span * x_01[n]
                _bsplvb_inplace(t, ks, gx_n, left, bsp_sub[n, :])
                # Inner weight: Jacobian * GL weight * r^k
                w = span * w_01[n]
                if k > 0:
                    gw_rk[n] = w * gx_n ** k
                elif k < 0:
                    gw_rk[n] = w / gx_n ** (-k)
                else:
                    gw_rk[n] = w

            # INTP[j, jp, m] = Σ_n B_j(r_n) * B_jp(r_n) * r_n^k * w_n
            for j in range(ks):
                for jp in range(j, ks):
                    s = 0.0
                    for n in range(n_sub):
                        s += bsp_sub[n, j] * bsp_sub[n, jp] * gw_rk[n]
                    INTP[j, jp, m] = s

        # --- Outer integration ---
        # Outer weight: grw(iv,m) / gr(iv,m)^{k+1}  (the 1/r_>^{k+1} factor)
        # Assemble a[ii,jj] = Σ_m B_i(m)*B_ip(m) * outer_weight(m) * INTP[j,jp,m]
        # outer_w depends only on m — hoist it out of the kk² pair loops
        # (and the B_i*B_ip*outer_w product out of the j-pair loop); the
        # per-term operand order is unchanged, so the sums are bit-identical.
        for m in range(n_gauss):
            ow[m] = grw[iv, m] / gr[iv, m] ** (k + 1)
        a = np.zeros((kk, kk))
        ii = 0
        for i in range(ks):
            for ip in range(i, ks):
                for m in range(n_gauss):
                    bw[m] = bsp[iv, m, i] * bsp[iv, m, ip] * ow[m]
                jj = 0
                for j in range(ks):
                    for jp in range(j, ks):
                        s = 0.0
                        for m in range(n_gauss):
                            s += bw[m] * INTP[j, jp, m]
                        a[ii, jj] = s
                        jj += 1
                ii += 1

        # Symmetrize: rkd = a + a^T (accounts for both r_< / r_> orderings)
        for ii in range(kk):
            for jj in range(kk):
                rkd[ii, jj, iv] = a[ii, jj] + a[jj, ii]

    return rkd


@nb.njit(cache=True)
def _gauss_legendre_01(n, x, w):
    """Gauss-Legendre points on [0, 1] via eigenvalue method.

    Good enough for n <= 20.  Populates x[0..n-1], w[0..n-1].
    """
    # Use the companion matrix approach for GL quadrature
    # For small n, use pre-computed values
    if n == 1:
        x[0] = 0.5
        w[0] = 1.0
        return
    # Golub-Welsch: symmetric tridiagonal matrix
    beta = np.empty(n - 1)
    for i in range(n - 1):
        j = i + 1
        beta[i] = j / np.sqrt(4.0 * j * j - 1.0)

    # Build tridiagonal and solve (simple QR-like via Jacobi iteration)
    # For Numba compatibility, do explicit eigenvalue computation
    # Actually, use the fact that GL zeros are well-tabulated for small n
    # Let's use Newton iteration on Legendre polynomials
    for i in range(n):
        # Initial guess for i-th root
        xi = np.cos(np.pi * (i + 0.75) / (n + 0.5))
        for _iter in range(50):
            p0 = 1.0
            p1 = xi
            for j in range(2, n + 1):
                p2 = ((2 * j - 1) * xi * p1 - (j - 1) * p0) / j
                p0 = p1
                p1 = p2
            # p1 = P_n(xi), derivative:
            dp = n * (xi * p1 - p0) / (xi * xi - 1.0)
            dx = p1 / dp
            xi -= dx
            if abs(dx) < 1e-15:
                break
        # Map [-1,1] → [0,1]
        x[i] = 0.5 * (xi + 1.0)
        w[i] = 1.0 / ((1.0 - xi * xi) * dp * dp)


@nb.njit(cache=True)
def _bsplvb_inplace(t, ks, x, left, b):
    """Evaluate B-splines at x, writing into pre-allocated array b[0..ks-1].

    Same algorithm as bsplvb but writes in-place (Numba-friendly).
    """
    b[0] = 1.0
    for i in range(1, ks):
        b[i] = 0.0

    dr = np.empty(ks)
    dl = np.empty(ks)

    for j in range(1, ks):
        dr[j] = t[left + j] - x
        dl[j] = x - t[left + 1 - j]
        saved = 0.0
        for r in range(j):
            term = b[r] / (dr[r + 1] + dl[j - r])
            b[r] = saved + dr[r + 1] * term
            saved = dl[j - r] * term
        b[j] = saved


# ══════════════════════════════════════════════════════════════════════
# 3. Assemble full rkb array
# ══════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def _assemble_rkb_sym(rkd1, rkd2, rkd_diag, nv, ks, ns):
    """Assemble rkb(i, j, ip, jp) from moments + diagonal cells.

    Storage: rkb[i, j, ip, jp] where
      i = global B-spline index (0-indexed, 0..ns-1)
      j = global B-spline index (0-indexed, 0..ns-1)
      ip = band offset (0-indexed, 0..ks-1): actual second index = i + ip
      jp = band offset (0-indexed, 0..ks-1): actual second index = j + jp

    Off-diagonal cells (iv < jv):
      c = rkd1[ii, iv] * rkd2[jj, jv]    (r^k at smaller r, r^{-(k+1)} at larger r)
    Off-diagonal (iv > jv): swap roles
      c = rkd1[jj, jv] * rkd2[ii, iv]
    Diagonal (iv == jv):
      c = rkd_diag[ii, jj, iv]

    Parameters
    ----------
    rkd1 : (kk, nv)  — r^k moments
    rkd2 : (kk, nv)  — r^{-(k+1)} moments
    rkd_diag : (kk, kk, nv) — diagonal cell integrals
    nv, ks, ns : int

    Returns
    -------
    rkb : (ns, ns, ks, ks)
    """
    rkb = np.zeros((ns, ns, ks, ks))

    for jv in range(nv):
        jj = 0
        for jh in range(ks):
            j = jv + jh  # global index of first function in j-pair
            for jhp in range(jh, ks):
                jp = jhp - jh  # band offset (0-indexed)

                for iv in range(nv):
                    ii = 0
                    for ih in range(ks):
                        i = iv + ih
                        for ihp in range(ih, ks):
                            ip = ihp - ih  # band offset (0-indexed)

                            if iv < jv:
                                c = rkd1[ii, iv] * rkd2[jj, jv]
                            elif iv > jv:
                                c = rkd1[jj, jv] * rkd2[ii, iv]
                            else:
                                c = rkd_diag[ii, jj, iv]

                            if i < ns and j < ns and ip < ks and jp < ks:
                                rkb[i, j, ip, jp] += c

                            ii += 1
                jj += 1

    return rkb


# ══════════════════════════════════════════════════════════════════════
# 3b. Mixed-basis (pqpq/qpqp) cell integrals
# ══════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def _triang_pqpq(bsp_p, bsp_q, gr, grw, tp, tq, nv, ksp, ksq, ns, k):
    """Diagonal cell R^k integrals for pqpq type (P-pair × Q-pair).

    DBSR_HF triang_pqpq: outer integral uses P-basis, inner uses Q-basis
    (and vice versa via the transpose symmetrization).

    Packed indices: ii packs (i, ip) with ip >= i for P-pair (ksp*(ksp+1)//2)
                    jj packs (j, jp) with jp >= j for Q-pair (ksq*(ksq+1)//2)

    Returns rkd: (kkp, kkq, nv) where kkp = ksp*(ksp+1)//2, kkq = ksq*(ksq+1)//2
    """
    n_gauss = gr.shape[1]
    kkp = ksp * (ksp + 1) // 2
    kkq = ksq * (ksq + 1) // 2
    rkd = np.zeros((kkp, kkq, nv))

    # Sub-quadrature on [0, 1]
    n_sub = max(ksp, ksq)
    x_01 = np.empty(n_sub)
    w_01 = np.empty(n_sub)
    _gauss_legendre_01(n_sub, x_01, w_01)

    bsp_sub_q = np.empty((n_sub, ksq))
    gw_rk = np.empty(n_sub)
    INTP = np.empty((ksq, ksq, n_gauss))
    ow = np.empty(n_gauss)
    bw = np.empty(n_gauss)

    for iv in range(nv):
        left_q = ksq - 1 + iv
        xbase = tq[left_q]

        # --- Inner: Q-basis sub-quadrature for INTP[j, jp, m] ---
        for m in range(n_gauss):
            r_upper = gr[iv, m]
            span = r_upper - xbase
            if span < 1e-30:
                for j in range(ksq):
                    for jp in range(j, ksq):
                        INTP[j, jp, m] = 0.0
                continue

            for n in range(n_sub):
                gx_n = xbase + span * x_01[n]
                _bsplvb_inplace(tq, ksq, gx_n, left_q, bsp_sub_q[n, :])
                w = span * w_01[n]
                if k > 0:
                    gw_rk[n] = w * gx_n ** k
                elif k < 0:
                    gw_rk[n] = w / gx_n ** (-k)
                else:
                    gw_rk[n] = w

            for j in range(ksq):
                for jp in range(j, ksq):
                    s = 0.0
                    for n in range(n_sub):
                        s += bsp_sub_q[n, j] * bsp_sub_q[n, jp] * gw_rk[n]
                    INTP[j, jp, m] = s

        # --- Outer: P-basis with 1/r^{k+1} weight ---
        # outer_w depends only on m — hoist it (and the B_i*B_ip*outer_w
        # product) out of the pair loops; operand order per term is
        # unchanged, so the sums are bit-identical.
        for m in range(n_gauss):
            ow[m] = grw[iv, m] / gr[iv, m] ** (k + 1)
        a = np.zeros((kkp, kkq))
        ii = 0
        for i in range(ksp):
            for ip in range(i, ksp):
                for m in range(n_gauss):
                    bw[m] = bsp_p[iv, m, i] * bsp_p[iv, m, ip] * ow[m]
                jj = 0
                for j in range(ksq):
                    for jp in range(j, ksq):
                        s = 0.0
                        for m in range(n_gauss):
                            s += bw[m] * INTP[j, jp, m]
                        a[ii, jj] = s
                        jj += 1
                ii += 1

        # Also compute the transpose part: P inner, Q outer
        # INTP_p[i, ip, m] for P-basis inner integral
        bsp_sub_p = np.empty((n_sub, ksp))
        left_p = ksp - 1 + iv
        xbase_p = tp[left_p]
        INTP_p = np.empty((ksp, ksp, n_gauss))

        for m in range(n_gauss):
            r_upper = gr[iv, m]
            span = r_upper - xbase_p
            if span < 1e-30:
                for i2 in range(ksp):
                    for ip2 in range(i2, ksp):
                        INTP_p[i2, ip2, m] = 0.0
                continue

            for n in range(n_sub):
                gx_n = xbase_p + span * x_01[n]
                _bsplvb_inplace(tp, ksp, gx_n, left_p, bsp_sub_p[n, :])
                w = span * w_01[n]
                if k > 0:
                    gw_rk[n] = w * gx_n ** k
                elif k < 0:
                    gw_rk[n] = w / gx_n ** (-k)
                else:
                    gw_rk[n] = w

            for i2 in range(ksp):
                for ip2 in range(i2, ksp):
                    s = 0.0
                    for n in range(n_sub):
                        s += bsp_sub_p[n, i2] * bsp_sub_p[n, ip2] * gw_rk[n]
                    INTP_p[i2, ip2, m] = s

        # Transpose part: Q outer, P inner (same hoist as above)
        aT = np.zeros((kkp, kkq))
        jj = 0
        for j in range(ksq):
            for jp in range(j, ksq):
                for m in range(n_gauss):
                    bw[m] = bsp_q[iv, m, j] * bsp_q[iv, m, jp] * ow[m]
                ii = 0
                for i in range(ksp):
                    for ip in range(i, ksp):
                        s = 0.0
                        for m in range(n_gauss):
                            s += bw[m] * INTP_p[i, ip, m]
                        aT[ii, jj] = s
                        ii += 1
                jj += 1

        # Combine: rkd = a + aT (both r_< / r_> orderings)
        for ii in range(kkp):
            for jj in range(kkq):
                rkd[ii, jj, iv] = a[ii, jj] + aT[ii, jj]

    return rkd


@nb.njit(cache=True)
def _assemble_rkb_pqpq(rkd1_pp, rkd2_qq, rkd3_qq, rkd4_pp,
                         rkd_diag, nv, ksp, ksq, nsp, nsq, ns):
    """Assemble rkb for pqpq type (P-pair at r₁, Q-pair at r₂).

    Off-diagonal cells:
      iv < jv: P at smaller r, Q at larger r
               c = rkd1_pp[ii, iv] * rkd2_qq[jj, jv]
      iv > jv: Q at smaller r, P at larger r
               c = rkd3_qq[jj, jv] * rkd4_pp[ii, iv]
    Diagonal (iv == jv):
               c = rkd_diag[ii, jj, iv]

    Returns rkb: (ns, ns, ksp, ksq)
    """
    rkb = np.zeros((ns, ns, ksp, ksq))

    for jv in range(nv):
        jj = 0
        for jh in range(ksq):
            j = jv + jh
            for jhp in range(jh, ksq):
                jp = jhp - jh

                for iv in range(nv):
                    ii = 0
                    for ih in range(ksp):
                        i = iv + ih
                        for ihp in range(ih, ksp):
                            ip = ihp - ih

                            if iv < jv:
                                c = rkd1_pp[ii, iv] * rkd2_qq[jj, jv]
                            elif iv > jv:
                                c = rkd3_qq[jj, jv] * rkd4_pp[ii, iv]
                            else:
                                c = rkd_diag[ii, jj, iv]

                            if i < ns and j < ns and ip < ksp and jp < ksq:
                                rkb[i, j, ip, jp] += c

                            ii += 1
                jj += 1

    return rkb


def _build_rkb_pqpq(bsp_p, bsp_q, gr, grw, tp, tq, nv, ksp, ksq,
                      nsp, nsq, ns, k):
    """Build rkb for pqpq type: P-pair × Q-pair (DBSR_HF mrk_pqpq)."""
    # 4 moment arrays (DBSR_HF uses different moment types for each off-diag case)
    rkd1_pp = _compute_moments_sym(bsp_p, gr, grw, nv, ksp, k)        # P r^k
    rkd2_qq = _compute_moments_sym(bsp_q, gr, grw, nv, ksq, -(k + 1)) # Q r^{-(k+1)}
    rkd3_qq = _compute_moments_sym(bsp_q, gr, grw, nv, ksq, k)        # Q r^k
    rkd4_pp = _compute_moments_sym(bsp_p, gr, grw, nv, ksp, -(k + 1)) # P r^{-(k+1)}
    # Diagonal cells
    rkd_diag = _triang_pqpq(bsp_p, bsp_q, gr, grw, tp, tq, nv, ksp, ksq, ns, k)
    return _assemble_rkb_pqpq(rkd1_pp, rkd2_qq, rkd3_qq, rkd4_pp,
                               rkd_diag, nv, ksp, ksq, nsp, nsq, ns)


def _build_rkb_qpqp(bsp_p, bsp_q, gr, grw, tp, tq, nv, ksp, ksq,
                      nsp, nsq, ns, k):
    """Build rkb for qpqp type: Q-pair × P-pair (swap of pqpq)."""
    rkd1_qq = _compute_moments_sym(bsp_q, gr, grw, nv, ksq, k)
    rkd2_pp = _compute_moments_sym(bsp_p, gr, grw, nv, ksp, -(k + 1))
    rkd3_pp = _compute_moments_sym(bsp_p, gr, grw, nv, ksp, k)
    rkd4_qq = _compute_moments_sym(bsp_q, gr, grw, nv, ksq, -(k + 1))
    rkd_diag = _triang_pqpq(bsp_q, bsp_p, gr, grw, tq, tp, nv, ksq, ksp, ns, k)
    return _assemble_rkb_pqpq(rkd1_qq, rkd2_pp, rkd3_pp, rkd4_qq,
                               rkd_diag, nv, ksq, ksp, nsq, nsp, ns)


# ══════════════════════════════════════════════════════════════════════
# 4. Density matrices
# ══════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def _accumulate_density_sym(dd, p1, p2, ns, ks, coeff):
    """Accumulate symmetric banded density into dd: dd += coeff * outer_sym(p1, p2).

    Avoids allocating a temporary array per orbital.  Out-of-range reads
    are zero (see :func:`_density_sym`).
    """
    n1 = min(p1.shape[0], ns)
    n2 = min(p2.shape[0], ns)
    for i in range(min(n1, n2)):
        dd[i, 0] += coeff * p1[i] * p2[i]
        for s in range(1, ks):
            ip = i + s
            if ip >= ns:
                break
            v = 0.0
            if ip < n2:
                v += p1[i] * p2[ip]
            if ip < n1:
                v += p1[ip] * p2[i]
            dd[i, s] += coeff * v


@nb.njit(cache=True)
def _accumulate_density_full(xx, p1, p2, ns, coeff):
    """Accumulate full density into xx: xx += coeff * outer(p1, p2).

    Avoids allocating a temporary array per orbital.  Out-of-range reads
    are zero (see :func:`_density_full`).
    """
    n1 = min(p1.shape[0], ns)
    n2 = min(p2.shape[0], ns)
    for i in range(n1):
        for j in range(n2):
            xx[i, j] += coeff * p1[i] * p2[j]


@nb.njit(cache=True, fastmath=True)
def _accumulate_direct_density_batch(dd, comp_stack, coeffs, target_i, ns, ks):
    """Batched symmetric direct density — single numba call per (k, block).

    ``fastmath`` is kept on measured evidence only: 1.23× here (2.28 µs vs
    2.80 µs at ns=113).  Its two siblings
    :func:`_accumulate_density_sym` (1.07×) and
    :func:`_accumulate_exchange_density_batch` (1.07×) do not clear that
    bar and no longer carry the flag.

    Replaces a Python-level loop over sources that called
    :func:`_accumulate_density_sym` once per source. Each call to that
    helper crosses the Python→numba boundary (~1 µs) and forces
    ``a_direct[flat_j]`` scalar extraction; at ~900 k iterations per Pb
    SCF the overhead dominated the Fock build.

    Parameters
    ----------
    dd : (ns, ks) float64, modified in place.
    comp_stack : (n_orb, ns) float64
        Stacked P_j (or Q_j) for all source orbitals.
    coeffs : (n_orb,) float64
        a(target_i, j, k); zeros skip that source.
    target_i : int
        DBSR_HF ``c = c + c`` doubling applies only to j == target_i.
    """
    n_orb = coeffs.shape[0]
    for j in range(n_orb):
        c = coeffs[j]
        if c == 0.0:
            continue
        if j == target_i:
            c = c + c
        for i in range(ns):
            pi = comp_stack[j, i]
            dd[i, 0] += c * pi * pi
            for s in range(1, ks):
                if i + s < ns:
                    pis = comp_stack[j, i + s]
                    dd[i, s] += c * (pi * pis + pis * pi)


@nb.njit(cache=True)
def _accumulate_exchange_density_batch(xx, comp_ii_stack, comp_jj_stack,
                                        coeffs, ns):
    """Batched full-outer exchange density — single numba call per (k, block).

    Parameters
    ----------
    xx : (ns, ns) float64, modified in place.
    comp_ii_stack : (n_orb, ns) float64
    comp_jj_stack : (n_orb, ns) float64
    coeffs : (n_orb,) float64
        b(target_i, j, k); zeros skip that source.
    """
    n_orb = coeffs.shape[0]
    for j in range(n_orb):
        c = coeffs[j]
        if c == 0.0:
            continue
        for i in range(ns):
            p1i = c * comp_ii_stack[j, i]
            for k in range(ns):
                xx[i, k] += p1i * comp_jj_stack[j, k]


@nb.njit(cache=True)
def _density_sym(p1, p2, ns, ks):
    """Symmetric upper-band density: d[i, s] for s = 0..ks-1.

    d[i, 0]   = p1[i] * p2[i]                    (diagonal)
    d[i, s>0] = p1[i] * p2[i+s] + p1[i+s] * p2[i]  (symmetrized off-diagonal)

    ``ns`` is the row extent of the requested band matrix and may be the
    *larger* of the two basis orders (``bd["ns"] = max(nsp, nsq)``).  Reads
    past the actual length of ``p1``/``p2`` are treated as zero — the
    shorter component is zero-padded — rather than reading out of bounds
    (numba performs no bounds checking; the callers pass ``P[:ns_full]``,
    which is shorter than ``ns_full`` whenever the P/Q orders differ under
    kinetic balance).

    Numerics: with ``len(p1) == len(p2) == ns`` the guards are inert and
    the kernel accumulates the plain band construction's two products in
    the same order, so the two agree bit-for-bit (the leading ``0.0 + x``
    is exact under IEEE for finite ``x``).  That bit-identity holds only
    because this kernel carries **no** ``fastmath``: it was removed after
    measuring fastmath at a 1.06× speedup (plain 0.89 µs vs 0.85 µs at
    ns=113, ks=9) — not worth the reassociation it licenses.  The same
    applies to :func:`_density_full`, which fastmath did not speed up at
    all (0.99×).
    """
    n1 = p1.shape[0]
    n2 = p2.shape[0]
    d = np.zeros((ns, ks))
    for i in range(min(n1, ns)):
        if i < n2:
            d[i, 0] = p1[i] * p2[i]
        for s in range(1, ks):
            ip = i + s
            if ip >= ns:
                break
            v = 0.0
            if ip < n2:
                v += p1[i] * p2[ip]
            if ip < n1 and i < n2:
                v += p1[ip] * p2[i]
            d[i, s] = v
    return d


@nb.njit(cache=True)
def _density_full(p1, p2, ns):
    """Full density matrix: d[i, j] = p1[i] * p2[j].

    ``ns`` is the requested matrix extent and may exceed the length of
    either component (see :func:`_density_sym`): out-of-range reads are
    zero, i.e. the shorter component is zero-padded.  Identical to the
    plain outer product when ``len(p1) == len(p2) == ns``.
    """
    n1 = min(p1.shape[0], ns)
    n2 = min(p2.shape[0], ns)
    d = np.zeros((ns, ns))
    for i in range(n1):
        for j in range(n2):
            d[i, j] = p1[i] * p2[j]
    return d


# ══════════════════════════════════════════════════════════════════════
# 5. Convolution: contract rkb with density
# ══════════════════════════════════════════════════════════════════════
#
# These kernels are intentionally SERIAL (plain ``range``, no ``prange`` /
# ``parallel=True``).  They are called only from the SCF matrix build on
# atomic-structure-sized bases (ns ~ 40-150), where each ``prange`` iteration
# is only O(ks²) work.  Parallelising that fine-grained loop across a large
# thread pool is *net-negative*: measured Ar (ns=60) DHF solve was 21.0 s on
# 32 threads vs 2.2 s serial — a 9.4× slowdown from thread fork/join overhead,
# with byte-identical energies.  The serial/parallel crossover for these
# kernels is ns ≳ 290, beyond any realistic atomic basis.  Do NOT re-add
# ``prange`` here without re-measuring against the small-ns regime.

@nb.njit(cache=True)
def _convol_case1(rkb, d, ns, ks):
    """icase=1: R(. a ; . b) — contract right pair (j, jp) with density.

    a[i, ip] = Σ_{j, jp} rkb[i, j, ip, jp] * d[j, jp]

    Result is banded: a[i, ip] with ip = 0..ks-1.
    Serial over flattened output index (see section note: small-ns regime).
    """
    a = np.zeros((ns, ks))
    n_out = ns * ks
    for idx in range(n_out):
        i = idx // ks
        ip = idx % ks
        if i + ip >= ns:
            continue
        s = 0.0
        for jp in range(ks):
            for j in range(ns):
                if j + jp >= ns:
                    continue
                s += rkb[i, j, ip, jp] * d[j, jp]
        a[i, ip] = s
    return a


@nb.njit(cache=True)
def _banded_to_full_add(V, v_banded, ns, ks, coeff):
    """Add banded matrix v_banded(ns, ks) to full matrix V(ns, ns) with coefficient.

    V[i, i+s] += coeff * v_banded[i, s]  (and symmetric: V[i+s, i] += ...)
    """
    for s in range(ks):
        for i in range(ns - s):
            val = coeff * v_banded[i, s]
            V[i, i + s] += val
            if s > 0:
                V[i + s, i] += val


@nb.njit(cache=True)
def _convol_case1_mixed(rkb, d_right, ns, ks_left, ks_right):
    """icase=1 for mixed-basis rkb(ns, ns, ks_left, ks_right).

    Contract right pair (j, jp) with density d_right(ns, ks_right).
    Result is banded: a(ns, ks_left).
    """
    a = np.zeros((ns, ks_left))
    n_out = ns * ks_left
    for idx in range(n_out):
        i = idx // ks_left
        ip = idx % ks_left
        if i + ip >= ns:
            continue
        s = 0.0
        for jp in range(ks_right):
            for j in range(ns):
                if j + jp >= ns:
                    continue
                s += rkb[i, j, ip, jp] * d_right[j, jp]
        a[i, ip] = s
    return a


@nb.njit(cache=True)
def _convol_case2(rkb, d, ns, ks):
    """icase=2: R(a . ; b .) — contract left pair (i, ip) with density.

    a[j, jp] = Σ_{i, ip} rkb[i, j, ip, jp] * d[i, ip]

    Result is banded: a[j, jp].
    """
    a = np.zeros((ns, ks))
    n_out = ns * ks
    for idx in range(n_out):
        j = idx // ks
        jp = idx % ks
        if j + jp >= ns:
            continue
        s = 0.0
        for ip in range(ks):
            for i in range(ns):
                if i + ip >= ns:
                    continue
                s += rkb[i, j, ip, jp] * d[i, ip]
        a[j, jp] = s
    return a


@nb.njit(cache=True)
def _convol_case4(rkb, d, ns, ks, ks_right=None):
    """icase=4: R(a . ; . b) — exchange contraction (gather formulation).

    Computes: a[row, col] by gathering all (i,j,ip,jp) contributions
    where the scatter would write to (row, col).

    Result is FULL matrix: a[ns, ns].
    Serial over output rows (see section note: small-ns regime).

    NOTE (2026-07-10): a scatter reformulation (iterate band offsets, four
    ``a[...] +=`` writes per rkb element) micro-benchmarks ~2× faster on
    cache-hot random data, but is ~1.6× *slower* in-context — its scattered
    read-modify-writes to the ns×ns output thrash cache under real solver
    memory pressure, whereas this gather accumulates each output element in
    a register and writes once. Keep the gather; do not "optimise" to scatter
    without an in-context full-solve measurement.
    """
    if ks_right is None:
        ks_right = ks
    a = np.zeros((ns, ns))
    for row in range(ns):
        for col in range(ns):
            s = 0.0
            for ip in range(ks):
                i = row - ip
                if i < 0:
                    continue
                for jp in range(ks_right):
                    j_val = col  # a[row, col] ← a[ii, j] with ii=i+ip=row, j=col
                    jj = j_val + jp
                    if jj >= ns:
                        continue
                    c = rkb[i, j_val, ip, jp]
                    # Primary: a[i+ip, j] += c * d[i, j+jp]
                    s += c * d[i, jj]
            for ip in range(1, ks):
                # a[i, j] += c * d[ii, jj]  → row=i, so i=row, ii=row+ip
                ii = row + ip
                if ii >= ns:
                    continue
                for jp in range(ks_right):
                    j_val = col
                    jj = j_val + jp
                    if jj >= ns:
                        continue
                    c = rkb[row, j_val, ip, jp]
                    s += c * d[ii, jj]
            for jp in range(1, ks_right):
                # a[ii, jj] += c * d[i, j]  → row=ii=i+ip, col=jj=j+jp
                j_val = col - jp
                if j_val < 0:
                    continue
                for ip in range(ks):
                    i = row - ip
                    if i < 0:
                        continue
                    c = rkb[i, j_val, ip, jp]
                    s += c * d[i, j_val]
            for ip in range(1, ks):
                for jp in range(1, ks_right):
                    # a[i, jj] += c * d[ii, j]  → row=i, col=jj=j+jp
                    j_val = col - jp
                    if j_val < 0:
                        continue
                    ii = row + ip
                    if ii >= ns:
                        continue
                    c = rkb[row, j_val, ip, jp]
                    s += c * d[ii, j_val]
            a[row, col] = s
    return a


# ── Transposed-layout contractions (batch-DHF hot path) ─────────────
#
# ``_convol_case1_t`` / ``_convol_case4_t`` are drop-in reformulations of
# ``_convol_case1(_mixed)`` / ``_convol_case4`` that read the R^k bank in
# the transposed layout rkbT[ip, jp, i, j] (see ``ConvolCache.get_rkb_t``)
# so every innermost loop is unit-stride and SIMD-friendly — measured ~3×
# faster per call at atomx sizes (ns≈113, ks≈9).
#
# Numerics: each output element accumulates exactly the same terms in the
# same source order as the reference kernels (family 1→4 for case 4, jp
# outer / j inner for case 1).  ``_convol_case4_t`` therefore carries no
# ``fastmath`` and is bit-identical to ``_convol_case4`` — fastmath bought
# it nothing (0.97×, i.e. 3 % slower) and only licensed reassociation.
# ``_convol_case1_t`` *keeps* ``fastmath`` because there it is a genuine
# 3.4× win (104 µs vs 361 µs), and its results are consequently NOT
# bit-identical to ``_convol_case1``: measured 4.8e-15 relative per
# element, ≤1e-13 relative in converged energies.  A caller that must
# reproduce ``_convol_case1`` bit-for-bit (validation paths) has to use
# the untransposed kernel.

@nb.njit(cache=True, fastmath=True)
def _convol_case1_t(rkbT, d_right, ns, ks_left, ks_right):
    """icase=1 on the transposed bank: a[i, ip] = Σ rkbT[ip,jp,i,j]·d[j,jp].

    Handles both the symmetric (ks_left == ks_right) and mixed cases;
    result is banded a(ns, ks_left) like ``_convol_case1(_mixed)``.

    ``fastmath`` is deliberate here (3.4× measured); see the block comment
    above — results may differ from ``_convol_case1`` in the last bits.
    """
    # Transpose the banded density so the j-reduction is unit-stride in
    # both operands (d_right is small: ns × ks_right).
    dT = np.empty((ks_right, ns))
    for j in range(ns):
        for jp in range(ks_right):
            dT[jp, j] = d_right[j, jp]
    a = np.zeros((ns, ks_left))
    for ip in range(ks_left):
        for i in range(ns - ip):
            s = 0.0
            for jp in range(ks_right):
                row = rkbT[ip, jp, i]
                drow = dT[jp]
                for j in range(ns - jp):
                    s += row[j] * drow[j]
            a[i, ip] = s
    return a


@nb.njit(cache=True)
def _convol_case4_t(rkbT, d, ns, ks, ks_right=None, symmetric_output=False):
    """icase=4 on the transposed bank with unit-stride output columns.

    For equal-component Coulomb exchange, both the density and Coulomb
    kernel are symmetric. ``symmetric_output`` computes the lower triangle
    only and mirrors it, saving nearly half of this contraction. Mixed
    large/small-component blocks still require the complete matrix.
    """
    if ks_right is None:
        ks_right = ks
    a = np.zeros((ns, ns))
    for row in range(ns):
        buf = a[row]
        col_limit = row + 1 if symmetric_output else ns
        # family 1: a[row, col] += rkb[row-ip, col, ip, jp] * d[row-ip, col+jp]
        for ip in range(ks):
            i = row - ip
            if i < 0:
                continue
            for jp in range(ks_right):
                bank = rkbT[ip, jp, i]
                drow = d[i]
                for col in range(min(ns - jp, col_limit)):
                    buf[col] += bank[col] * drow[col + jp]
        # family 2: a[row, col] += rkb[row, col, ip, jp] * d[row+ip, col+jp]
        for ip in range(1, ks):
            ii = row + ip
            if ii >= ns:
                continue
            for jp in range(ks_right):
                bank = rkbT[ip, jp, row]
                drow = d[ii]
                for col in range(min(ns - jp, col_limit)):
                    buf[col] += bank[col] * drow[col + jp]
        # family 3: a[row, col] += rkb[row-ip, col-jp, ip, jp] * d[row-ip, col-jp]
        for jp in range(1, ks_right):
            for ip in range(ks):
                i = row - ip
                if i < 0:
                    continue
                bank = rkbT[ip, jp, i]
                drow = d[i]
                for col in range(jp, col_limit):
                    buf[col] += bank[col - jp] * drow[col - jp]
        # family 4: a[row, col] += rkb[row, col-jp, ip, jp] * d[row+ip, col-jp]
        for ip in range(1, ks):
            ii = row + ip
            if ii >= ns:
                continue
            for jp in range(1, ks_right):
                bank = rkbT[ip, jp, row]
                drow = d[ii]
                for col in range(jp, col_limit):
                    buf[col] += bank[col - jp] * drow[col - jp]
        if symmetric_output:
            for col in range(row):
                a[col, row] = buf[col]
    return a


@nb.njit(cache=True)
def _global_pair_moments(local, nv, ks, ns):
    """Sum cell moments for each global upper-band B-spline pair."""
    moments = np.zeros((ks, ns))
    for iv in range(nv):
        local_pair = 0
        for first in range(ks):
            i = iv + first
            for second in range(first, ks):
                offset = second - first
                if i + offset < ns:
                    moments[offset, i] += local[local_pair, iv]
                local_pair += 1
    return moments


@nb.njit(cache=True)
def _assemble_near_rkb(rising_left, falling_right, rising_right,
                       falling_left, diagonal, nv, ks_left, ks_right, ns):
    """Assemble only Coulomb pair integrals with overlapping supports.

    The full cell builder visits every pair of radial cells. Here a pair
    of global B-spline products can overlap only in a strip of at most
    ``ks_left + ks_right - 1`` second-pair indices. Bound the first-cell
    loop accordingly while preserving the original cell summation order.
    """
    near = np.zeros((ks_left, ks_right, ns,
                     ks_left + ks_right - 1))
    for jv in range(nv):
        right_local = 0
        for jh in range(ks_right):
            j = jv + jh
            for jhp in range(jh, ks_right):
                jp = jhp - jh
                if j < ns:
                    # For some left pair offset ip, overlap requires
                    # j-ks_right+jp+1 <= i <= j+ks_left-ip-1.
                    # Since i=iv+ih, ih<ks_left, these bounds cover every
                    # contributing first-cell index without a global scan.
                    iv_start = max(0, j - ks_right + jp - ks_left + 2)
                    iv_stop = min(nv, j + ks_left)
                    for iv in range(iv_start, iv_stop):
                        left_local = 0
                        for ih in range(ks_left):
                            i = iv + ih
                            for ihp in range(ih, ks_left):
                                ip = ihp - ih
                                first_j = i + ip - ks_left + 1
                                last_j = i + ks_right - jp - 1
                                if (i < ns and first_j <= j <= last_j):
                                    if iv < jv:
                                        value = (rising_left[left_local, iv]
                                                 * falling_right[right_local, jv])
                                    elif iv > jv:
                                        value = (rising_right[right_local, jv]
                                                 * falling_left[left_local, iv])
                                    else:
                                        value = diagonal[left_local, right_local, iv]
                                    near[ip, jp, i, j - first_j] += value
                                left_local += 1
                right_local += 1
    return near


@nb.njit(cache=True)
def _convol_case1_semiseparable_t(near, rising_left, falling_left,
                                  rising_right, falling_right, density,
                                  ns, ks_left, ks_right):
    """Direct Coulomb contraction using exact prefix/suffix pair moments.

    Outside the overlapping B-spline support strip, the Coulomb kernel
    separates into two radial moments. Prefix and suffix sums contract all
    distant right-hand pairs in O(ns*ks_right), leaving only the local
    overlap strip to visit for each left-hand pair. A suffix is accumulated
    directly to avoid subtracting nearly equal cumulative sums in the tail.
    """
    prefix = np.zeros((ks_right, ns + 1))
    suffix = np.zeros((ks_right, ns + 1))
    for jp in range(ks_right):
        end = ns - jp
        for j in range(end):
            prefix[jp, j + 1] = (prefix[jp, j]
                                   + rising_right[jp, j] * density[j, jp])
        for j in range(end - 1, -1, -1):
            suffix[jp, j] = (suffix[jp, j + 1]
                             + falling_right[jp, j] * density[j, jp])
    result = np.zeros((ns, ks_left))
    for ip in range(ks_left):
        for i in range(ns - ip):
            first_near = i + ip - ks_left + 1
            value = 0.0
            for jp in range(ks_right):
                end = ns - jp
                before = min(end, max(0, first_near))
                after = min(end, max(0, i + ks_right - jp))
                value += falling_left[ip, i] * prefix[jp, before]
                value += rising_left[ip, i] * suffix[jp, after]
                for j in range(max(0, first_near), after):
                    value += (near[ip, jp, i, j - first_near]
                              * density[j, jp])
            result[i, ip] = value
    return result


@nb.njit(cache=True, inline="always")
def _add_semiseparable_exchange_row(buf, density_row, near, rising,
                                    falling, i, ip, jp, ks, j_stop,
                                    output_shift, density_shift):
    """Stream the two factored radial regions and the exact overlap strip."""
    first_near = i + ip - ks + 1
    last_near = i + ks - jp - 1
    first_end = min(j_stop, first_near)
    inside = falling[ip, i]
    for j in range(first_end):
        buf[j + output_shift] += (
            inside * rising[jp, j] * density_row[j + density_shift])
    for j in range(max(0, first_near), min(j_stop, last_near + 1)):
        buf[j + output_shift] += (
            near[ip, jp, i, j - first_near]
            * density_row[j + density_shift])
    outside = rising[ip, i]
    for j in range(max(0, last_near + 1), j_stop):
        buf[j + output_shift] += (
            outside * falling[jp, j] * density_row[j + density_shift])


@nb.njit(cache=True)
def _convol_case4_semiseparable_t(near, rising, falling, density, ns, ks):
    """Exact symmetric exchange from Coulomb pair moments and local cells.

    For disjoint radial supports, ``r_<^k/r_>^(k+1)`` is a product of
    two one-dimensional pair moments. Only pairs whose B-spline supports
    overlap need the original cell integral. This computes one matrix
    triangle and mirrors it, like ``_convol_case4_t(..., True)``.
    """
    result = np.zeros((ns, ns))
    for row in range(ns):
        buf = result[row]
        limit = row + 1
        for ip in range(ks):
            i = row - ip
            if i < 0:
                continue
            for jp in range(ks):
                _add_semiseparable_exchange_row(
                    buf, density[i], near, rising, falling, i, ip, jp,
                    ks, min(ns - jp, limit), 0, jp)
        for ip in range(1, ks):
            other = row + ip
            if other >= ns:
                continue
            for jp in range(ks):
                _add_semiseparable_exchange_row(
                    buf, density[other], near, rising, falling, row,
                    ip, jp, ks, min(ns - jp, limit), 0, jp)
        for jp in range(1, ks):
            for ip in range(ks):
                i = row - ip
                if i < 0:
                    continue
                _add_semiseparable_exchange_row(
                    buf, density[i], near, rising, falling, i, ip, jp,
                    ks, limit - jp, jp, 0)
        for ip in range(1, ks):
            other = row + ip
            if other >= ns:
                continue
            for jp in range(1, ks):
                _add_semiseparable_exchange_row(
                    buf, density[other], near, rising, falling, row,
                    ip, jp, ks, limit - jp, jp, 0)
        for col in range(row):
            result[col, row] = buf[col]
    return result


@nb.njit(cache=True, inline="always")
def _add_semiseparable_mixed_row(buf, density_row, near, rising_left,
                                  falling_left, rising_right, falling_right,
                                  i, ip, jp, ks_left, ks_right, j_stop,
                                  output_shift, density_shift):
    """Stream one mixed-component exchange row in radial order."""
    first_near = i + ip - ks_left + 1
    last_near = i + ks_right - jp - 1
    for j in range(min(j_stop, first_near)):
        buf[j + output_shift] += (
            falling_left[ip, i] * rising_right[jp, j]
            * density_row[j + density_shift])
    for j in range(max(0, first_near), min(j_stop, last_near + 1)):
        buf[j + output_shift] += (
            near[ip, jp, i, j - first_near]
            * density_row[j + density_shift])
    for j in range(max(0, last_near + 1), j_stop):
        buf[j + output_shift] += (
            rising_left[ip, i] * falling_right[jp, j]
            * density_row[j + density_shift])


@nb.njit(cache=True)
def _convol_case4_semiseparable_mixed_t(
        near, rising_left, falling_left, rising_right, falling_right,
        density, ns, ks_left, ks_right):
    """Exact nonsymmetric P/Q exchange without a full Coulomb bank."""
    result = np.zeros((ns, ns))
    for row in range(ns):
        buf = result[row]
        for ip in range(ks_left):
            i = row - ip
            if i < 0:
                continue
            for jp in range(ks_right):
                _add_semiseparable_mixed_row(
                    buf, density[i], near, rising_left, falling_left,
                    rising_right, falling_right, i, ip, jp, ks_left,
                    ks_right, ns - jp, 0, jp)
        for ip in range(1, ks_left):
            other = row + ip
            if other >= ns:
                continue
            for jp in range(ks_right):
                _add_semiseparable_mixed_row(
                    buf, density[other], near, rising_left, falling_left,
                    rising_right, falling_right, row, ip, jp, ks_left,
                    ks_right, ns - jp, 0, jp)
        for jp in range(1, ks_right):
            for ip in range(ks_left):
                i = row - ip
                if i < 0:
                    continue
                _add_semiseparable_mixed_row(
                    buf, density[i], near, rising_left, falling_left,
                    rising_right, falling_right, i, ip, jp, ks_left,
                    ks_right, ns - jp, jp, 0)
        for ip in range(1, ks_left):
            other = row + ip
            if other >= ns:
                continue
            for jp in range(1, ks_right):
                _add_semiseparable_mixed_row(
                    buf, density[other], near, rising_left, falling_left,
                    rising_right, falling_right, row, ip, jp, ks_left,
                    ks_right, ns - jp, jp, 0)
    return result


# ══════════════════════════════════════════════════════════════════════
# 6. Trace contraction: SUM_AmB
# ══════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def _convol_case2_mixed(rkb, d_left, ns, ks_left, ks_right):
    """icase=2 for mixed-basis rkb(ns, ns, ks_left, ks_right).

    Contract left pair (i, ip) with banded density d_left(ns, ks_left).
    Result is banded: a(ns, ks_right).

    HYP-1590: the left-pair contraction is bounded by the density's own
    row count (``n_left``), NOT by ``ns``.  Under kinetic balance
    ``nsp = ns - 1``, so a caller passing ``ns`` rows for a P-density
    with only ``nsp`` rows made the old loop read past the end of
    ``d_left`` — numba nopython mode performs no bounds checking, so the
    kernel consumed whatever garbage float64s happened to follow the
    array on the heap.  R^k rows beyond the P/Q basis size are built
    exactly zero (same trimming rule as :func:`_rkb_bilinear`), so
    ``0 * garbage`` was silently benign — until the garbage was a NaN
    bit pattern, when ``0 * NaN = NaN`` poisoned the whole Slater
    integral.  Heap contents depend on allocator state, which is why the
    failure only surfaced under heavy box load.  Trimming is exact.
    """
    n_left = d_left.shape[0]
    a = np.zeros((ns, ks_right))
    n_out = ns * ks_right
    for idx in range(n_out):
        j = idx // ks_right
        jp = idx % ks_right
        if j + jp >= ns:
            continue
        s = 0.0
        for ip in range(ks_left):
            for i in range(n_left):
                if i + ip >= n_left:
                    continue
                s += rkb[i, j, ip, jp] * d_left[i, ip]
        a[j, jp] = s
    return a


@nb.njit(cache=True)
def _sum_amb_banded(a, b, ns, ks):
    """Trace contraction of two banded matrices: Σ_{i,s} a[i,s] * b[i,s]."""
    s = 0.0
    for ip in range(ks):
        for i in range(ns):
            if i + ip < ns:
                s += a[i, ip] * b[i, ip]
    return s


@nb.njit(cache=True, fastmath=True)
def _sum_amb_full(a, b, ns):
    """Trace contraction of two full matrices: Σ_{i,j} a[i,j] * b[i,j].

    ``fastmath`` is a measured 3.9× win (1.9 µs vs 7.2 µs at ns=113): the
    guard-free contiguous reduction vectorises once reassociation is
    allowed.  Its banded sibling :func:`_sum_amb_banded` is the opposite —
    fastmath made it 30 % *slower* (0.89 µs vs 0.69 µs), so it no longer
    carries the flag.
    """
    s = 0.0
    for i in range(ns):
        for j in range(ns):
            s += a[i, j] * b[i, j]
    return s


# ══════════════════════════════════════════════════════════════════════
# 7. High-level interface
# ══════════════════════════════════════════════════════════════════════

# ── Process-wide bank registry ──────────────────────────────────────
#
# The rkb banks depend only on the B-spline *geometry* (knot vector, basis
# orders, grid) — never on the orbital coefficients or on the solver
# instance consuming them — so every :class:`ConvolCache` built on an
# equal-content workspace can share one bank dict.  Keying on object
# identity is not enough: the MCDHF driver rebuilds the ``bsp_data`` dicts
# for every outer iteration (fresh objects, bit-identical content), and
# each DFSolver used to lazily create its own cache, so the same 20 banks
# were rebuilt once per outer iteration (~14 % of an MCDHF solve, measured
# 0.40 s of 3.20 s at ns=60 and 1.44 s of 10.40 s at ns=113).  Keying on
# the grid content makes the banks reusable across those solvers while
# staying exact: any geometry difference (including a different ``tp``
# under kinetic balance, or a different Gauss quadrature) changes the key.
_BANK_CACHE_LIMIT = 4          # distinct grids kept alive (each holds ~20 banks)
# At a large radial box, one R^k bank can occupy hundreds of megabytes.
# SCF uses the transposed layout; its original layout can be a zero-copy view
# when the final-energy Slater contractions need it. Avoid retaining two full
# copies per bank for the diffuse/Rydberg bases used by the NIST run.
_TRANSPOSE_ONLY_NS = 256
# The factored symmetric exchange overtakes the contiguous full bank once
# the latter ceases to fit comfortably in cache. Smaller bases keep the
# simpler bank contraction, which remains faster there.
SEMISEPARABLE_EXCHANGE_NS = 320
# Mixed exchange is slightly slower per warm contraction at 333 splines,
# but avoids the last O(ns^2) full banks and improves cold-start/RSS.
SEMISEPARABLE_MIXED_EXCHANGE_NS = 320
# Direct/Slater contractions benefit earlier than exchange: on the standard
# 113-spline grid the four-component scalar contraction is 13--30x faster
# with factors (ranks 0, 2, 6), with agreement within 2e-12 relative. Keep
# the separately measured exchange thresholds above unchanged.
SEMISEPARABLE_DIRECT_NS = 96
_RKB_BANKS: OrderedDict = OrderedDict()


def grid_bank_key(bsp_data):
    """Content key identifying the rkb banks built on ``bsp_data``.

    The bank content is a pure function of the knot sequences and basis
    orders.  ``t``/``tp`` are hashed by content (``tobytes``) rather than
    by ``id()``: the MCDHF driver hands each outer iteration a freshly
    built — but bit-identical — ``bsp_data`` dict, so an id()-keyed cache
    would miss every time and rebuild all 20 banks.

    ``gr``/``grw`` are derived from ``(t, ns, ks, n_gauss)``; the
    quadrature *shape* enters the key so a caller changing ``n_gauss``
    without changing the knots still gets its own banks.
    """
    t = bsp_data.get("t") if isinstance(bsp_data, Mapping) else None
    if t is None:
        raise TypeError(
            "ConvolCache needs a B-spline workspace mapping carrying a 't' "
            f"knot vector (got {type(bsp_data).__name__})")
    t = np.ascontiguousarray(t)
    tp = bsp_data.get("tp")
    if tp is None or np.array_equal(tp, t):
        tp_bytes = b""
    else:
        tp_bytes = np.ascontiguousarray(tp).tobytes()
    gr = bsp_data.get("gr")
    return (
        int(bsp_data["nv"]),
        int(bsp_data.get("ks", bsp_data.get("ksq", 0))),
        int(bsp_data.get("ksp", bsp_data.get("ks", 0))),
        int(bsp_data.get("ksq", bsp_data.get("ks", 0))),
        int(bsp_data.get("nsp", bsp_data.get("ns", 0))),
        int(bsp_data.get("nsq", bsp_data.get("ns", 0))),
        int(bsp_data.get("ns", 0)),
        t.shape,
        t.tobytes(),
        tp_bytes,
        None if gr is None else np.shape(gr),
    )


def shared_banks(bsp_data, key=None):
    """Return the process-wide bank dict for ``bsp_data``'s geometry.

    All caches over an equal-content grid return the *same* dict object,
    so both the untransposed and the transposed banks are built once per
    process.  Bounded LRU: at most ``_BANK_CACHE_LIMIT`` grids are
    retained; eviction is safe because the dict is a pure memo (a live
    :class:`ConvolCache` keeps its own reference and simply recomputes a
    bank if it is ever dropped).
    """
    if key is None:
        key = grid_bank_key(bsp_data)
    banks = _RKB_BANKS.get(key)
    if banks is None:
        banks = {}
        _RKB_BANKS[key] = banks
        while len(_RKB_BANKS) > _BANK_CACHE_LIMIT:
            _RKB_BANKS.popitem(last=False)
    else:
        _RKB_BANKS.move_to_end(key)
    return banks


class ConvolCache:
    """Cache for cell-based R^k integrals.

    Stores pre-computed rkb arrays for each (k, itype) combination.
    Reuses arrays across direct potential, exchange, and energy computation.

    ``_cache`` is shared process-wide between every cache built on an
    equal-content grid (see :func:`shared_banks`), so an MCDHF solve — which
    constructs a fresh :class:`~atomx.methods.dirac_fock.DFSolver` per outer
    iteration — builds each of the 20 banks once instead of once per
    iteration.  Banks are read-only to every consumer.

    A cache owns no bank individually, so there is no per-instance
    "clear": the only teardown operation is :meth:`release_shared_banks`,
    which drops the whole grid's banks for every cache at once.
    """

    def __init__(self, bsp_data):
        self.bsp_data = bsp_data
        self.grid_key = grid_bank_key(bsp_data)
        self._cache = shared_banks(bsp_data, self.grid_key)

    def same_grid(self, bsp_data) -> bool:
        """True if ``bsp_data`` has the same bank geometry as this cache."""
        try:
            return grid_bank_key(bsp_data) == self.grid_key
        except (KeyError, TypeError):
            return False

    def get_rkb(self, k, itype="pppp"):
        """Get or compute rkb array for multipole order k and integral type."""
        key = (k, itype)
        if key in self._cache:
            return self._cache[key]
        transposed = self._cache.get((k, itype, "T"))
        if transposed is not None:
            rkbT, ns, ks = transposed
            return rkbT.transpose(2, 3, 0, 1), ns, ks

        bd = self.bsp_data
        nv = bd["nv"]
        gr = bd["gr"]
        grw = bd["grw"]
        t = bd["t"]

        ns_full = bd["ns"]  # = max(nsp, nsq) — DBSR_HF uses this for all types

        if itype == "pppp":
            ks = bd.get("ksp", bd["ks"])
            ns = ns_full  # DBSR_HF: rkb uses ns (not nsp) for first two dims
            bsp = bd["bsp_p"]
            tp = bd.get("tp", t)
            rkb = _build_rkb_sym(bsp, gr, grw, tp, nv, ks, ns, k)
        elif itype == "qqqq":
            ks = bd.get("ksq", bd["ks"])
            ns = ns_full
            bsp = bd.get("bsp_q", bd["bsp_p"])
            rkb = _build_rkb_sym(bsp, gr, grw, t, nv, ks, ns, k)
        elif itype == "pqpq":
            ksp = bd.get("ksp", bd["ks"])
            ksq = bd.get("ksq", bd["ks"])
            nsp = bd.get("nsp", bd["ns"])
            nsq = bd.get("nsq", bd["ns"])
            ns_loc = bd["ns"]
            bsp_p = bd["bsp_p"]
            bsp_q = bd.get("bsp_q", bsp_p)
            tp = bd.get("tp", t)
            rkb = _build_rkb_pqpq(bsp_p, bsp_q, gr, grw, tp, t,
                                   nv, ksp, ksq, nsp, nsq, ns_loc, k)
            ks = ksp  # left-pair order for storage
            ns_loc_out = ns_loc
        elif itype == "qpqp":
            ksp = bd.get("ksp", bd["ks"])
            ksq = bd.get("ksq", bd["ks"])
            nsp = bd.get("nsp", bd["ns"])
            nsq = bd.get("nsq", bd["ns"])
            ns_loc = bd["ns"]
            bsp_p = bd["bsp_p"]
            bsp_q = bd.get("bsp_q", bsp_p)
            tp = bd.get("tp", t)
            rkb = _build_rkb_qpqp(bsp_p, bsp_q, gr, grw, tp, t,
                                   nv, ksp, ksq, nsp, nsq, ns_loc, k)
            ks = ksq  # left-pair order for storage
            ns_loc_out = ns_loc
        else:
            raise ValueError(f"Unknown integral type: {itype}")

        if itype in ("pqpq", "qpqp"):
            self._cache[key] = (rkb, ns_loc_out, ks)
            return rkb, ns_loc_out, ks

        self._cache[key] = (rkb, ns, ks)
        return rkb, ns, ks

    def get_rkb_t(self, k, itype="pppp"):
        """Transposed rkb bank for the ``_convol_case*_t`` kernels.

        Layout: rkbT[ip, jp, i, j] = rkb[i, j, ip, jp], C-contiguous so the
        kernels' innermost j/col loops are unit-stride. For compact bases,
        it is cached alongside the original bank. For bases with at least
        ``_TRANSPOSE_ONLY_NS`` splines, the original copy is discarded and
        :meth:`get_rkb` supplies a zero-copy transposed view to the less
        frequent final-energy consumers. The one-time transpose is
        amortised over the SCF iterations.
        """
        key = (k, itype, "T")
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        rkb, ns, ks = self.get_rkb(k, itype)
        rkbT = np.ascontiguousarray(rkb.transpose(2, 3, 0, 1))
        self._cache[key] = (rkbT, ns, ks)
        if self.bsp_data["ns"] >= _TRANSPOSE_ONLY_NS:
            self._cache.pop((k, itype), None)
        return rkbT, ns, ks

    def _component_pair_moments(self, k, component):
        """Return local and global moments shared by every Coulomb block."""
        key = (k, component, "pair_moments")
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        bd = self.bsp_data
        ks = bd.get("ksp" if component == "p" else "ksq", bd["ks"])
        bsp = (bd["bsp_p"] if component == "p"
               else bd.get("bsp_q", bd["bsp_p"]))
        local_rising = _compute_moments_sym(
            bsp, bd["gr"], bd["grw"], bd["nv"], ks, k)
        local_falling = _compute_moments_sym(
            bsp, bd["gr"], bd["grw"], bd["nv"], ks, -(k + 1))
        rising = _global_pair_moments(local_rising, bd["nv"], ks, bd["ns"])
        falling = _global_pair_moments(local_falling, bd["nv"], ks, bd["ns"])
        self._cache[key] = (local_rising, local_falling, rising, falling)
        return self._cache[key]

    def get_coulomb_factors(self, k, itype="pppp"):
        """Cache exact overlapping cells and factored distant pair moments.

        For disjoint radial supports, the Coulomb kernel separates into
        ``r^k`` and ``r^(-k-1)`` moments. This identity applies to equal
        and mixed P/Q component pairs. Build the overlap strip directly
        from cell moments, without allocating the full four-index bank.
        """
        components = {"pppp": ("p", "p"), "qqqq": ("q", "q"),
                      "pqpq": ("p", "q"), "qpqp": ("q", "p")}
        if itype not in components:
            raise ValueError(f"Unknown integral type: {itype}")
        key = (k, itype, "semiseparable")
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        left, right = components[itype]
        bd = self.bsp_data
        ns, nv = bd["ns"], bd["nv"]
        local_rising_left, local_falling_left, rising_left, falling_left = (
            self._component_pair_moments(k, left))
        local_rising_right, local_falling_right, rising_right, falling_right = (
            self._component_pair_moments(k, right))
        ks_left = rising_left.shape[0]
        ks_right = rising_right.shape[0]
        bsp_left = (bd["bsp_p"] if left == "p"
                    else bd.get("bsp_q", bd["bsp_p"]))
        knots_left = bd.get("tp", bd["t"]) if left == "p" else bd["t"]
        if left == right:
            diagonal = _triang_sym(
                bsp_left, bd["gr"], bd["grw"], knots_left, nv,
                ks_left, ns, k)
        else:
            bsp_right = (bd["bsp_p"] if right == "p"
                         else bd.get("bsp_q", bd["bsp_p"]))
            knots_right = (bd.get("tp", bd["t"])
                           if right == "p" else bd["t"])
            diagonal = _triang_pqpq(
                bsp_left, bsp_right, bd["gr"], bd["grw"],
                knots_left, knots_right, nv, ks_left, ks_right, ns, k)
        near = _assemble_near_rkb(
            local_rising_left, local_falling_right, local_rising_right,
            local_falling_left, diagonal, nv, ks_left, ks_right, ns)
        self._cache[key] = (near, rising_left, falling_left,
                            rising_right, falling_right)
        return self._cache[key]

    def get_symmetric_exchange_factors(self, k, itype="pppp"):
        """Return the same-component subset needed by symmetric exchange."""
        if itype not in ("pppp", "qqqq"):
            raise ValueError("symmetric exchange factors need equal components")
        near, rising, falling, _, _ = self.get_coulomb_factors(k, itype)
        return near, rising, falling

    def release_shared_banks(self):
        """Drop this grid's banks, for this cache and every other live one.

        ``_cache`` is the process-wide bank dict of this grid's content
        key (see :func:`shared_banks`), so releasing here frees the banks
        for *all* caches sharing the grid — that is the intent: immediate
        memory relief between large solves.  Safety does not depend on the
        caller being the last user: banks are pure functions of the grid
        geometry, so any cache that keeps running lazily rebuilds
        bit-identical content on its next access, at the rebuild cost the
        sharing exists to avoid.  Single-threaded by design: every
        ``ConvolCache`` consumer runs in one thread (parallel drivers use
        process pools, where each process holds its own registry).
        """
        self._cache.clear()


def _build_rkb_sym(bsp, gr, grw, t, nv, ks, ns, k):
    """Build rkb for a symmetric basis type (pppp or qqqq)."""
    rkd1 = _compute_moments_sym(bsp, gr, grw, nv, ks, k)
    rkd2 = _compute_moments_sym(bsp, gr, grw, nv, ks, -(k + 1))
    rkd_diag = _triang_sym(bsp, gr, grw, t, nv, ks, ns, k)
    return _assemble_rkb_sym(rkd1, rkd2, rkd_diag, nv, ks, ns)


# ── Direct potential via convolution ────────────────────────────────

def compute_direct_potential_cell(orb_P, orb_Q, bsp_data, k, cache=None):
    """Direct matrices from the complete density P²+Q² in either component.

    Both output blocks feel both source components. In particular the
    QQ block contains the dominant P² source, and the PP block contains
    the relativistic Q² contribution. The matrices use the padded
    ``bsp_data['ns']`` layout of the Dirac eigenproblem.
    """
    bd = bsp_data
    ns = bd["ns"]
    orders = {"p": bd.get("ksp", bd["ks"]), "q": bd.get("ksq", bd["ks"])}
    if cache is None:
        cache = ConvolCache(bd)
    densities = {c: _density_sym(v, v, ns, orders[c])
                 for c, v in (("p", orb_P), ("q", orb_Q))}
    blocks = []
    for a in ("p", "q"):
        banded = np.zeros((ns, orders[a]))
        for b in ("p", "q"):
            bank, _, _ = cache.get_rkb_t(k, a+b+a+b)
            banded += _convol_case1_t(bank, densities[b], ns, orders[a], orders[b])
        full = np.zeros((ns, ns))
        _banded_to_full_add(full, banded, ns, orders[a], 1.)
        blocks.append(full)
    return tuple(blocks)


# ── Exchange matrix via convolution ─────────────────────────────────

def compute_exchange_matrix_cell(orb_a_P, orb_a_Q, orb_b_P, orb_b_Q,
                                  bsp_data, k, cache=None):
    """All four Dirac exchange blocks, in order PP, QQ, PQ, QP.

    Coulomb exchange acts on the complete spinor pair density. Its mixed
    blocks P(r)Q(s) and Q(r)P(s) are required even though the *direct*
    Coulomb field is diagonal in Dirac components. Dropping them changes
    the Fock Hamiltonian and destroys occupied/virtual orthogonality.
    Coefficients and returned blocks use the padded ``bsp_data['ns']``
    layout of the generalized Dirac eigenproblem.
    """
    bd = bsp_data
    ns = bd["ns"]
    orders = {"p": bd.get("ksp", bd["ks"]), "q": bd.get("ksq", bd["ks"])}
    left = {"p": orb_a_P, "q": orb_a_Q}
    right = {"p": orb_b_P, "q": orb_b_Q}
    if cache is None:
        cache = ConvolCache(bd)
    diagonal = orb_a_P is orb_b_P and orb_a_Q is orb_b_Q
    blocks = {}
    for a, b in (("p", "p"), ("q", "q"), ("p", "q"), ("q", "p")):
        if diagonal and (a, b) == ("q", "p"):
            blocks[a, b] = blocks["p", "q"].T.copy()
            continue
        bank, _, _ = cache.get_rkb_t(k, a+b+a+b)
        density = _density_full(left[a], right[b], ns)
        blocks[a, b] = _convol_case4_t(bank, density, ns, orders[a], orders[b],
                                      symmetric_output=diagonal and a == b)
    return tuple(blocks[key] for key in (("p", "p"), ("q", "q"), ("p", "q"), ("q", "p")))


# ── Slater integrals via cell algorithm ────────────────────────────

def compute_slater_rk_cell(Pa, Qa, Pb, Qb, Pc, Qc, Pd, Qd, bsp_data, k,
                            cache=None):
    """Scalar Slater integral R^k(ab, cd) via cell algorithm.

    R^k = ∫∫ ρ_ab(r₁) [r_<^k / r_>^{k+1}] ρ_cd(r₂) dr₁ dr₂

    where ρ_ab = P_a·P_b + Q_a·Q_b.

    Uses the exact cell algorithm for all four component types:
    pppp + qqqq + pqpq + qpqp.

    The evaluation is wrapped in a transient-NaN tripwire (HYP-1590): under
    extreme CPU oversubscription a single serial kernel call has been
    observed to return NaN from finite inputs while an immediate identical
    re-evaluation recovers (reproducer: 8 concurrent he_dhf solves at box
    load ~200, NaN only on each process's first solve, E_J component).
    The result check is O(1) per call; the retry path is cold.
    """
    last = None
    for _attempt in range(3):
        result = _compute_slater_rk_cell_once(
            Pa, Qa, Pb, Qb, Pc, Qc, Pd, Qd, bsp_data, k, cache=cache)
        if np.isfinite(result):
            return result
        last = result
    raise FloatingPointError(
        "compute_slater_rk_cell: non-finite R^k on 3 consecutive "
        f"evaluations (k={k}, value={last}) — transient numerical "
        "corruption did not recover (HYP-1590)."
    )


def _compute_slater_rk_cell_once(Pa, Qa, Pb, Qb, Pc, Qc, Pd, Qd, bsp_data, k,
                                  cache=None):
    """Single evaluation behind :func:`compute_slater_rk_cell`."""
    bd = bsp_data
    ksp = bd.get("ksp", bd["ks"])
    nsp = bd.get("nsp", bd["ns"])
    ksq = bd.get("ksq", bd["ks"])
    nsq = bd.get("nsq", bd["ns"])

    if bd["ns"] >= SEMISEPARABLE_DIRECT_NS:
        if cache is None:
            cache = ConvolCache(bd)
        ns = bd["ns"]
        left_p = _density_sym(Pa, Pb, ns, ksp)
        left_q = _density_sym(Qa, Qb, ns, ksq)
        right_p = _density_sym(Pc, Pd, ns, ksp)
        right_q = _density_sym(Qc, Qd, ns, ksq)
        value = 0.0
        for itype, left, right in (
                ("pppp", left_p, right_p),
                ("qqqq", left_q, right_q),
                ("pqpq", left_p, right_q),
                ("qpqp", left_q, right_p)):
            factors = cache.get_coulomb_factors(k, itype)
            contracted = _convol_case1_semiseparable_t(
                *factors, right, ns, factors[1].shape[0],
                factors[3].shape[0])
            value += _sum_amb_banded(
                contracted, left, ns, factors[1].shape[0])
        return value

    result = 0.0

    # pppp contribution: R^k_PP(P_a·P_b, P_c·P_d)
    if cache is not None:
        rkb_pp, _, _ = cache.get_rkb(k, "pppp")
    else:
        rkb_pp = _build_rkb_sym(
            bd["bsp_p"], bd["gr"], bd["grw"], bd["t"],
            bd["nv"], ksp, nsp, k)

    d_ab_P = _density_sym(Pa, Pb, nsp, ksp)
    conv_P = _convol_case2(rkb_pp, d_ab_P, nsp, ksp)
    d_cd_P = _density_sym(Pc, Pd, nsp, ksp)
    result += _sum_amb_banded(conv_P, d_cd_P, nsp, ksp)

    # qqqq contribution: R^k_QQ(Q_a·Q_b, Q_c·Q_d)
    if cache is not None:
        rkb_qq, _, _ = cache.get_rkb(k, "qqqq")
    else:
        bsp_q = bd.get("bsp_q", bd["bsp_p"])
        rkb_qq = _build_rkb_sym(
            bsp_q, bd["gr"], bd["grw"], bd["t"],
            bd["nv"], ksq, nsq, k)

    d_ab_Q = _density_sym(Qa, Qb, nsq, ksq)
    conv_Q = _convol_case2(rkb_qq, d_ab_Q, nsq, ksq)
    d_cd_Q = _density_sym(Qc, Qd, nsq, ksq)
    result += _sum_amb_banded(conv_Q, d_cd_Q, nsq, ksq)

    # Cross-component contributions (pqpq + qpqp) via cell algorithm
    ns_loc = bd["ns"]

    # pqpq: P_a*P_b at r₁ with Q_c*Q_d at r₂
    if cache is not None:
        rkb_pq, ns_pq, ks_pq = cache.get_rkb(k, "pqpq")
    else:
        bsp_q = bd.get("bsp_q", bd["bsp_p"])
        tp = bd.get("tp", bd["t"])
        rkb_pq = _build_rkb_pqpq(
            bd["bsp_p"], bsp_q, bd["gr"], bd["grw"], tp, bd["t"],
            bd["nv"], ksp, ksq, nsp, nsq, ns_loc, k)

    d_ab_P_pq = _density_sym(Pa, Pb, nsp, ksp)
    conv_pq = _convol_case2_mixed(rkb_pq, d_ab_P_pq, ns_loc, ksp, ksq)
    d_cd_Q_pq = _density_sym(Qc, Qd, nsq, ksq)
    # Trace: need to align dimensions (both are banded with ksq bands, ns_loc rows)
    result += _sum_amb_banded(conv_pq, d_cd_Q_pq, min(ns_loc, nsq), ksq)

    # qpqp: Q_a*Q_b at r₁ with P_c*P_d at r₂
    if cache is not None:
        rkb_qp, ns_qp, ks_qp = cache.get_rkb(k, "qpqp")
    else:
        bsp_q = bd.get("bsp_q", bd["bsp_p"])
        tp = bd.get("tp", bd["t"])
        rkb_qp = _build_rkb_qpqp(
            bd["bsp_p"], bsp_q, bd["gr"], bd["grw"], tp, bd["t"],
            bd["nv"], ksp, ksq, nsp, nsq, ns_loc, k)

    d_ab_Q_qp = _density_sym(Qa, Qb, nsq, ksq)
    conv_qp = _convol_case2_mixed(rkb_qp, d_ab_Q_qp, ns_loc, ksq, ksp)
    d_cd_P_qp = _density_sym(Pc, Pd, nsp, ksp)
    result += _sum_amb_banded(conv_qp, d_cd_P_qp, min(ns_loc, nsp), ksp)

    return result


def compute_slater_fk_cell(Pa, Qa, Pb, Qb, bsp_data, k, cache=None):
    """Direct Slater integral F^k(a,b) = R^k(aa, bb) via cell algorithm."""
    return compute_slater_rk_cell(Pa, Qa, Pa, Qa, Pb, Qb, Pb, Qb,
                                   bsp_data, k, cache)


def compute_slater_gk_cell(Pa, Qa, Pb, Qb, bsp_data, k, cache=None):
    """Exchange Slater integral G^k(a,b) = R^k(ab, ba) via cell algorithm."""
    return compute_slater_rk_cell(Pa, Qa, Pb, Qb, Pb, Qb, Pa, Qa,
                                   bsp_data, k, cache)


# ── Batched Slater integrals: bilinear form in the slot-2/4 orbitals ─


@nb.njit(cache=True)
def _rkb_bilinear(rkb, wa, wb, n_left, n_right, ks_left, ks_right, ns_out):
    """Fold the fixed orbital pair (wa, wb) into one rkb bank.

    M[m, n] = Σ_{i,ip,j,jp} rkb[i,j,ip,jp] · σ_wa(i,ip;m) · σ_wb(j,jp;n)

    where σ_w(i,ip;m) is the symmetrised-band factor of :func:`_density_sym`
    (w[i]·δ_{m,i+ip} + [ip>0]·w[i+ip]·δ_{m,i}), so that for any coefficient
    vectors v_l, v_r the block value equals the scalar path's
    _convol_case2 → _sum_amb_banded chain: v_lᵀ · M · v_r.
    Domains i+ip < n_left, j+jp < n_right match the scalar path (rkb rows
    beyond the P/Q basis size are built zero, so trimming is exact).
    """
    # Stage 1: contract the left band pair — same O(ns²·ks²) as one
    # _convol_case2 call.
    T = np.zeros((ns_out, n_right, ks_right))
    for i in range(n_left):
        for j in range(n_right):
            for ip in range(ks_left):
                if i + ip >= n_left:
                    break
                wi = wa[i]
                wip = wa[i + ip]
                for jp in range(ks_right):
                    if j + jp >= n_right:
                        break
                    c = rkb[i, j, ip, jp]
                    T[i, j, jp] += c * wip
                    if ip > 0:
                        T[i + ip, j, jp] += c * wi
    # Stage 2: contract the right band pair — O(ns²·ks), negligible.
    M = np.zeros((ns_out, ns_out))
    for m in range(ns_out):
        for j in range(n_right):
            wj = wb[j]
            for jp in range(ks_right):
                if j + jp >= n_right:
                    break
                t = T[m, j, jp]
                M[m, j] += t * wb[j + jp]
                if jp > 0:
                    M[m, j + jp] += t * wj
    return M


def build_slater_rk_bilinear(Pa, Qa, Pb, Qb, bsp_data, k, cache=None):
    """Bilinear-form matrices for R^k(a·, b·) with orbitals (a, b) fixed.

    R^k(a r, b s) = Prᵀ·M_PP·Ps + Qrᵀ·M_QQ·Qs + Prᵀ·M_PQ·Qs + Qrᵀ·M_QP·Ps

    for ANY orbital pair (r, s) — :func:`compute_slater_rk_cell` with the
    density/convolution/trace contractions refactored as a bilinear form
    in the slot-2/slot-4 coefficient vectors (same per-block summation
    domains).  Build cost ≈ one scalar call; each subsequent evaluation is
    O(ns²) (:func:`eval_slater_rk_bilinear`) or a GEMM over a whole table
    of pairs (:func:`compute_slater_rk_table`) — the R^k(a r, b s) tables
    in MBPT(2) and the pp-ladder come essentially for free.
    """
    bd = bsp_data
    if cache is None:
        cache = ConvolCache(bd)
    ksp = bd.get("ksp", bd["ks"])
    nsp = bd.get("nsp", bd["ns"])
    ksq = bd.get("ksq", bd["ks"])
    nsq = bd.get("nsq", bd["ns"])
    ns_loc = bd["ns"]

    rkb_pp, _, _ = cache.get_rkb(k, "pppp")
    rkb_qq, _, _ = cache.get_rkb(k, "qqqq")
    rkb_pq, _, _ = cache.get_rkb(k, "pqpq")
    rkb_qp, _, _ = cache.get_rkb(k, "qpqp")

    M_PP = _rkb_bilinear(rkb_pp, Pa, Pb, nsp, nsp, ksp, ksp, ns_loc)
    M_QQ = _rkb_bilinear(rkb_qq, Qa, Qb, nsq, nsq, ksq, ksq, ns_loc)
    M_PQ = _rkb_bilinear(rkb_pq, Pa, Qb, min(ns_loc, nsp), min(ns_loc, nsq),
                         ksp, ksq, ns_loc)
    M_QP = _rkb_bilinear(rkb_qp, Qa, Pb, min(ns_loc, nsq), min(ns_loc, nsp),
                         ksq, ksp, ns_loc)
    return M_PP, M_QQ, M_PQ, M_QP


def eval_slater_rk_bilinear(M, Pl, Ql, Pr, Qr):
    """Single R^k(a l, b r) from :func:`build_slater_rk_bilinear` matrices."""
    M_PP, M_QQ, M_PQ, M_QP = M
    return float(Pl @ (M_PP @ Pr) + Ql @ (M_QQ @ Qr)
                 + Pl @ (M_PQ @ Qr) + Ql @ (M_QP @ Pr))


def compute_slater_rk_table(M, CP_l, CQ_l, CP_r, CQ_r):
    """Full table R^k(a l, b r) over stacked coefficient columns via GEMM.

    ``CP_l``/``CQ_l`` and ``CP_r``/``CQ_r`` are (ns, n) matrices with one
    orbital per column; returns the (n_l, n_r) table.
    """
    M_PP, M_QQ, M_PQ, M_QP = M
    return (CP_l.T @ M_PP @ CP_r + CQ_l.T @ M_QQ @ CQ_r
            + CP_l.T @ M_PQ @ CQ_r + CQ_l.T @ M_QP @ CP_r)


# ---------------------------------------------------------------------------
# Batched cell kernels (performance plan tier-3a, 2026-10-03)
# ---------------------------------------------------------------------------

@nb.njit(parallel=True, cache=True)
def _rk_cell_batch_sym(rkb, p1_stack, p2_stack, p3_stack, p4_stack, ns, ks):
    """Batched pppp/qqqq-term cell R^k for many integrals (prange).

    Element b: symmetrised band density(p1, p2), case-2 contraction
    with ``rkb``, trace against density(p3, p4) — the exact loop
    structure of the scalar path in :func:`_compute_slater_rk_cell_once`
    (``_density_sym`` + ``_convol_case2`` + ``_sum_amb_banded``),
    evaluated for all batch elements in parallel.  Coefficient stacks
    must be zero-padded to a common length covering ``ns``.
    """
    b_n = p1_stack.shape[0]
    out = np.zeros(b_n)
    for b in _prange(b_n):
        p1 = p1_stack[b]
        p2 = p2_stack[b]
        p3 = p3_stack[b]
        p4 = p4_stack[b]
        acc = 0.0
        for j in range(ns):
            for jp in range(ks):
                if j + jp >= ns:
                    continue
                a_jjp = 0.0
                for ip in range(ks):
                    for i in range(ns):
                        if i + ip >= ns:
                            continue
                        d_ab = p1[i] * p2[i + ip]
                        if ip > 0:
                            d_ab += p1[i + ip] * p2[i]
                        a_jjp += rkb[i, j, ip, jp] * d_ab
                d_cd = p3[j] * p4[j + jp]
                if jp > 0:
                    d_cd += p3[j + jp] * p4[j]
                acc += a_jjp * d_cd
        out[b] = acc
    return out


@nb.njit(parallel=True, cache=True)
def _rk_cell_batch_mixed(rkb, p1_stack, p2_stack, q1_stack, q2_stack,
                          ns_c, ks_left, ks_right, ns_t):
    """Batched pqpq/qpqp-term cell contraction for many integrals.

    Element b: band density(p1, p2) (ks_left basis) contracted through
    ``rkb`` to (ns_c, ks_right) bands, traced against band density(q1,
    q2) (ks_right basis) over the first ns_t rows — the exact mixed
    structure of :func:`_compute_slater_rk_cell_once` (HYP-1590
    zero-pad semantics preserved via padded stacks).
    """
    b_n = p1_stack.shape[0]
    out = np.zeros(b_n)
    for b in _prange(b_n):
        p1 = p1_stack[b]
        p2 = p2_stack[b]
        q1 = q1_stack[b]
        q2 = q2_stack[b]
        acc = 0.0
        for j in range(ns_c):
            for jp in range(ks_right):
                if j + jp >= ns_c:
                    continue
                a_jjp = 0.0
                for ip in range(ks_left):
                    for i in range(ns_c):
                        if i + ip >= ns_c:
                            continue
                        d_ab = p1[i] * p2[i + ip]
                        if ip > 0:
                            d_ab += p1[i + ip] * p2[i]
                        a_jjp += rkb[i, j, ip, jp] * d_ab
                if j < ns_t:
                    d_cd = q1[j] * q2[j + jp]
                    if jp > 0:
                        d_cd += q1[j + jp] * q2[j]
                    acc += a_jjp * d_cd
        out[b] = acc
    return out


def compute_slater_rk_cell_batch(orbital_arrays, keys, bsp_data, cache=None):
    """Batched Slater R^k(ab, cd) over many (a, b, c, d, k) tuples.

    ``orbital_arrays`` maps each key used in ``keys`` to the (P, Q)
    COEFFICIENT vectors; ``keys`` is a list of ``(a, b, c, d, k)``
    tuples in that key space.  Groups by k, fetches the SAME cached
    rkb factor tensors as the scalar cell kernel (``cache``), and
    evaluates the four component terms (pppp/qqqq/pqpq/qpqp) with
    prange-batched numba kernels.  Returns a list aligned with
    ``keys``: the same arithmetic as :func:`compute_slater_rk_cell`,
    evaluated in parallel.  This is the performance-plan tier-3a
    infrastructure: the measured CI two-body assembly hotspot is ~100k
    unique cell integrals at ~150 us each (~30 s serial) — the batch
    kernel amortises the same work across cores.
    """
    bd = bsp_data
    nsp = bd.get("nsp", bd["ns"])
    nsq = bd.get("nsq", bd["ns"])
    ksp = bd.get("ksp", bd["ks"])
    ksq = bd.get("ksq", bd["ks"])
    ns_loc = bd["ns"]
    ns_full = max(nsp, nsq)
    if cache is None:
        cache = ConvolCache(bd)

    by_k = {}
    for idx, key in enumerate(keys):
        by_k.setdefault(key[4], []).append(idx)
    result = [0.0] * len(keys)

    for k, idxs in by_k.items():
        rkb_pp, _, _ = cache.get_rkb(k, "pppp")
        rkb_qq, _, _ = cache.get_rkb(k, "qqqq")
        rkb_pq, _, _ = cache.get_rkb(k, "pqpq")
        rkb_qp, _, _ = cache.get_rkb(k, "qpqp")

        # per-role stacked P/Q coefficient matrices (zero-padded)
        roles = ("a", "b", "c", "d")
        stacks = {"P": {}, "Q": {}}
        for comp in ("P", "Q"):
            for role, slot in zip(roles, (0, 1, 2, 3), strict=True):
                mat = np.zeros((len(idxs), ns_full))
                for r, i in enumerate(idxs):
                    key = keys[i]
                    arr = orbital_arrays[key[slot]][0 if comp == "P" else 1]
                    m = min(len(arr), ns_full)
                    mat[r, :m] = arr[:m]
                stacks[comp][role] = mat

        P, Q = stacks["P"], stacks["Q"]
        out_pp = _rk_cell_batch_sym(rkb_pp, P["a"], P["b"], P["c"], P["d"],
                                  nsp, ksp)
        out_qq = _rk_cell_batch_sym(rkb_qq, Q["a"], Q["b"], Q["c"], Q["d"],
                                  nsq, ksq)
        out_pq = _rk_cell_batch_mixed(rkb_pq, P["a"], P["b"], Q["c"], Q["d"],
                                       ns_loc, ksp, ksq, min(ns_loc, nsq))
        out_qp = _rk_cell_batch_mixed(rkb_qp, Q["a"], Q["b"], P["c"], P["d"],
                                     ns_loc, ksq, ksp, min(ns_loc, nsp))
        for r, i in enumerate(idxs):
            result[i] = float(out_pp[r] + out_qq[r] + out_pq[r] + out_qp[r])
    return result
