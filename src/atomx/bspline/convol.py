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
"""

import numpy as np
import numba as nb


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
        a = np.zeros((kk, kk))
        ii = 0
        for i in range(ks):
            for ip in range(i, ks):
                jj = 0
                for j in range(ks):
                    for jp in range(j, ks):
                        s = 0.0
                        for m in range(n_gauss):
                            outer_w = grw[iv, m] / gr[iv, m] ** (k + 1)
                            s += (bsp[iv, m, i] * bsp[iv, m, ip]
                                  * outer_w * INTP[j, jp, m])
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
# 4. Density matrices
# ══════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def _density_sym(p1, p2, ns, ks):
    """Symmetric upper-band density: d[i, s] for s = 0..ks-1.

    d[i, 0]   = p1[i] * p2[i]                    (diagonal)
    d[i, s>0] = p1[i] * p2[i+s] + p1[i+s] * p2[i]  (symmetrized off-diagonal)
    """
    d = np.zeros((ns, ks))
    for i in range(ns):
        d[i, 0] = p1[i] * p2[i]
        for s in range(1, ks):
            if i + s < ns:
                d[i, s] = p1[i] * p2[i + s] + p1[i + s] * p2[i]
    return d


@nb.njit(cache=True)
def _density_full(p1, p2, ns):
    """Full density matrix: d[i, j] = p1[i] * p2[j]."""
    d = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            d[i, j] = p1[i] * p2[j]
    return d


# ══════════════════════════════════════════════════════════════════════
# 5. Convolution: contract rkb with density
# ══════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def _convol_case1(rkb, d, ns, ks):
    """icase=1: R(. a ; . b) — contract right pair (j, jp) with density.

    a[i, ip] = Σ_{j, jp} rkb[i, j, ip, jp] * d[j, jp]

    Result is banded: a[i, ip] with ip = 0..ks-1.
    """
    a = np.zeros((ns, ks))
    for ip in range(ks):
        for i in range(ns):
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
def _convol_case2(rkb, d, ns, ks):
    """icase=2: R(a . ; b .) — contract left pair (i, ip) with density.

    a[j, jp] = Σ_{i, ip} rkb[i, j, ip, jp] * d[i, ip]

    Result is banded: a[j, jp].
    """
    a = np.zeros((ns, ks))
    for jp in range(ks):
        for j in range(ns):
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
def _convol_case4(rkb, d, ns, ks):
    """icase=4: R(a . ; . b) — exchange contraction.

    Contract i (1st index) and jp (4th index) with density d[i, jp_global].
    Result indices: ip_global = i + ip, j.
    a[ip_global, j] = Σ_{i, jp} rkb[i, j, ip, jp] * d[i, j+jp]

    Result is FULL matrix: a[ns, ns].

    In DBSR_HF this scatters into 4 quadrants due to band symmetry:
      rkb[i, j, ip, jp] contributes to:
        a[i+ip, j]    using d[i,   j+jp]
        a[i,    j]    using d[i+ip, j+jp]   (if ip > 0)
        a[i+ip, j+jp] using d[i,   j]       (if jp > 0)
        a[i,    j+jp] using d[i+ip, j]       (if ip > 0 and jp > 0)
    """
    a = np.zeros((ns, ns))
    for jp in range(ks):
        for j in range(ns):
            jj = j + jp
            if jj >= ns:
                continue
            for ip in range(ks):
                for i in range(ns):
                    ii = i + ip
                    if ii >= ns:
                        continue
                    c = rkb[i, j, ip, jp]
                    if abs(c) < 1e-30:
                        continue
                    # Primary contribution
                    a[ii, j] += c * d[i, jj]
                    # Symmetry expansions
                    if ip > 0:
                        a[i, j] += c * d[ii, jj]
                    if jp > 0:
                        a[ii, jj] += c * d[i, j]
                    if ip > 0 and jp > 0:
                        a[i, jj] += c * d[ii, j]
    return a


# ══════════════════════════════════════════════════════════════════════
# 6. Trace contraction: SUM_AmB
# ══════════════════════════════════════════════════════════════════════

@nb.njit(cache=True)
def _sum_amb_banded(a, b, ns, ks):
    """Trace contraction of two banded matrices: Σ_{i,s} a[i,s] * b[i,s]."""
    s = 0.0
    for ip in range(ks):
        for i in range(ns):
            if i + ip < ns:
                s += a[i, ip] * b[i, ip]
    return s


@nb.njit(cache=True)
def _sum_amb_full(a, b, ns):
    """Trace contraction of two full matrices: Σ_{i,j} a[i,j] * b[i,j]."""
    s = 0.0
    for i in range(ns):
        for j in range(ns):
            s += a[i, j] * b[i, j]
    return s


# ══════════════════════════════════════════════════════════════════════
# 7. High-level interface
# ══════════════════════════════════════════════════════════════════════

class ConvolCache:
    """Cache for cell-based R^k integrals.

    Stores pre-computed rkb arrays for each (k, itype) combination.
    Reuses arrays across direct potential, exchange, and energy computation.
    """

    def __init__(self, bsp_data):
        self.bsp_data = bsp_data
        self._cache = {}

    def get_rkb(self, k, itype="pppp"):
        """Get or compute rkb array for multipole order k and integral type."""
        key = (k, itype)
        if key in self._cache:
            return self._cache[key]

        bd = self.bsp_data
        nv = bd["nv"]
        gr = bd["gr"]
        grw = bd["grw"]
        t = bd["t"]

        if itype == "pppp":
            ks = bd.get("ksp", bd["ks"])
            ns = bd.get("nsp", bd["ns"])
            bsp = bd["bsp_p"]
            tp = bd.get("tp", t)  # P-knot sequence (differs from t with KB)
            rkb = _build_rkb_sym(bsp, gr, grw, tp, nv, ks, ns, k)
        elif itype == "qqqq":
            ks = bd.get("ksq", bd["ks"])
            ns = bd.get("nsq", bd["ns"])
            bsp = bd.get("bsp_q", bd["bsp_p"])
            rkb = _build_rkb_sym(bsp, gr, grw, t, nv, ks, ns, k)
        elif itype in ("pqpq", "qpqp"):
            raise NotImplementedError(
                f"Mixed integral type '{itype}' not yet implemented. "
                "Use Y^k approach for cross-component integrals."
            )
        else:
            raise ValueError(f"Unknown integral type: {itype}")

        self._cache[key] = (rkb, ns, ks)
        return rkb, ns, ks

    def clear(self):
        self._cache.clear()


def _build_rkb_sym(bsp, gr, grw, t, nv, ks, ns, k):
    """Build rkb for a symmetric basis type (pppp or qqqq)."""
    rkd1 = _compute_moments_sym(bsp, gr, grw, nv, ks, k)
    rkd2 = _compute_moments_sym(bsp, gr, grw, nv, ks, -(k + 1))
    rkd_diag = _triang_sym(bsp, gr, grw, t, nv, ks, ns, k)
    return _assemble_rkb_sym(rkd1, rkd2, rkd_diag, nv, ks, ns)


# ── Direct potential via convolution ────────────────────────────────

def compute_direct_potential_cell(orb_P, orb_Q, bsp_data, k, cache=None):
    """Direct (Coulomb) potential matrix V^k via cell algorithm.

    V^k_{ij} = Σ_{kl} R^k(ij, kl) * ρ_{kl}

    where ρ_{kl} = P_a[k]*P_a[l] + Q_a[k]*Q_a[l] is the orbital density.

    Returns V_PP (nsp×nsp) and V_QQ (nsq×nsq) diagonal-block contributions.
    """
    bd = bsp_data
    ksp = bd.get("ksp", bd["ks"])
    nsp = bd.get("nsp", bd["ns"])
    ksq = bd.get("ksq", bd["ks"])
    nsq = bd.get("nsq", bd["ns"])

    # PP block: rkb_pppp contracted with P-density
    if cache is not None:
        rkb_pp, ns_pp, ks_pp = cache.get_rkb(k, "pppp")
    else:
        rkb_pp, ns_pp, ks_pp = _build_rkb_sym(
            bd["bsp_p"], bd["gr"], bd["grw"], bd["t"],
            bd["nv"], ksp, nsp, k), nsp, ksp

    d_P = _density_sym(orb_P, orb_P, nsp, ksp)
    v_PP_banded = _convol_case1(rkb_pp, d_P, nsp, ksp)

    # Convert banded → full matrix
    V_PP = np.zeros((nsp, nsp))
    for i in range(nsp):
        for s in range(ksp):
            if i + s < nsp:
                V_PP[i, i + s] += v_PP_banded[i, s]
                if s > 0:
                    V_PP[i + s, i] += v_PP_banded[i, s]

    # QQ block: rkb_qqqq contracted with Q-density
    if cache is not None:
        rkb_qq, ns_qq, ks_qq = cache.get_rkb(k, "qqqq")
    else:
        bsp_q = bd.get("bsp_q", bd["bsp_p"])
        rkb_qq = _build_rkb_sym(
            bsp_q, bd["gr"], bd["grw"], bd["t"],
            bd["nv"], ksq, nsq, k)
        ns_qq, ks_qq = nsq, ksq

    d_Q = _density_sym(orb_Q, orb_Q, nsq, ksq)
    v_QQ_banded = _convol_case1(rkb_qq, d_Q, nsq, ksq)

    V_QQ = np.zeros((nsq, nsq))
    for i in range(nsq):
        for s in range(ksq):
            if i + s < nsq:
                V_QQ[i, i + s] += v_QQ_banded[i, s]
                if s > 0:
                    V_QQ[i + s, i] += v_QQ_banded[i, s]

    return V_PP, V_QQ


# ── Exchange matrix via convolution ─────────────────────────────────

def compute_exchange_matrix_cell(orb_a_P, orb_a_Q, orb_b_P, orb_b_Q,
                                  bsp_data, k, cache=None):
    """Non-local exchange matrix K^k via cell algorithm.

    K^k_{ij} comes from R^k(a·, ·b) contracted with cross-density.

    Returns K_PP (nsp×nsp) and K_QQ (nsq×nsq).
    """
    bd = bsp_data
    ksp = bd.get("ksp", bd["ks"])
    nsp = bd.get("nsp", bd["ns"])
    ksq = bd.get("ksq", bd["ks"])
    nsq = bd.get("nsq", bd["ns"])

    # PP block
    if cache is not None:
        rkb_pp, ns_pp, ks_pp = cache.get_rkb(k, "pppp")
    else:
        rkb_pp = _build_rkb_sym(
            bd["bsp_p"], bd["gr"], bd["grw"], bd["t"],
            bd["nv"], ksp, nsp, k)
        ns_pp, ks_pp = nsp, ksp

    d_P = _density_full(orb_a_P, orb_b_P, nsp)
    K_PP = _convol_case4(rkb_pp, d_P, nsp, ksp)

    # QQ block
    if cache is not None:
        rkb_qq, ns_qq, ks_qq = cache.get_rkb(k, "qqqq")
    else:
        bsp_q = bd.get("bsp_q", bd["bsp_p"])
        rkb_qq = _build_rkb_sym(
            bsp_q, bd["gr"], bd["grw"], bd["t"],
            bd["nv"], ksq, nsq, k)
        ns_qq, ks_qq = nsq, ksq

    d_Q = _density_full(orb_a_Q, orb_b_Q, nsq)
    K_QQ = _convol_case4(rkb_qq, d_Q, nsq, ksq)

    return K_PP, K_QQ


# ── Slater integrals via cell algorithm ────────────────────────────

def compute_slater_rk_cell(Pa, Qa, Pb, Qb, Pc, Qc, Pd, Qd, bsp_data, k,
                            cache=None):
    """Scalar Slater integral R^k(ab, cd) via cell algorithm.

    R^k = ∫∫ ρ_ab(r₁) [r_<^k / r_>^{k+1}] ρ_cd(r₂) dr₁ dr₂

    where ρ_ab = P_a·P_b + Q_a·Q_b.

    Uses all 4 component types: pppp + qqqq + pqpq + qpqp.
    Currently pppp and qqqq are exact cell algorithm; pqpq/qpqp use Y^k fallback.
    """
    bd = bsp_data
    ksp = bd.get("ksp", bd["ks"])
    nsp = bd.get("nsp", bd["ns"])
    ksq = bd.get("ksq", bd["ks"])
    nsq = bd.get("nsq", bd["ns"])

    result = 0.0

    # pppp contribution: R^k_PP(P_a·P_b, P_c·P_d)
    if cache is not None:
        rkb_pp, ns_pp, ks_pp = cache.get_rkb(k, "pppp")
    else:
        rkb_pp = _build_rkb_sym(
            bd["bsp_p"], bd["gr"], bd["grw"], bd["t"],
            bd["nv"], ksp, nsp, k)
        ns_pp, ks_pp = nsp, ksp

    d_ab_P = _density_sym(Pa, Pb, nsp, ksp)
    conv_P = _convol_case2(rkb_pp, d_ab_P, nsp, ksp)
    d_cd_P = _density_sym(Pc, Pd, nsp, ksp)
    result += _sum_amb_banded(conv_P, d_cd_P, nsp, ksp)

    # qqqq contribution: R^k_QQ(Q_a·Q_b, Q_c·Q_d)
    if cache is not None:
        rkb_qq, ns_qq, ks_qq = cache.get_rkb(k, "qqqq")
    else:
        bsp_q = bd.get("bsp_q", bd["bsp_p"])
        rkb_qq = _build_rkb_sym(
            bsp_q, bd["gr"], bd["grw"], bd["t"],
            bd["nv"], ksq, nsq, k)
        ns_qq, ks_qq = nsq, ksq

    d_ab_Q = _density_sym(Qa, Qb, nsq, ksq)
    conv_Q = _convol_case2(rkb_qq, d_ab_Q, nsq, ksq)
    d_cd_Q = _density_sym(Qc, Qd, nsq, ksq)
    result += _sum_amb_banded(conv_Q, d_cd_Q, nsq, ksq)

    # Cross-component contributions (pqpq + qpqp): use Y^k fallback
    # These are typically small corrections (P·Q cross terms)
    from atomx.scf.slater import _eval_P, _eval_Q, compute_yk
    gr, grw = bd["gr"], bd["grw"]
    nv = bd["nv"]
    n_gauss = gr.shape[1]

    fa_P = _eval_P(Pa, bd)
    fa_Q = _eval_Q(Qa, bd)
    fb_P = _eval_P(Pb, bd)
    fb_Q = _eval_Q(Qb, bd)
    fc_P = _eval_P(Pc, bd)
    fc_Q = _eval_Q(Qc, bd)
    fd_P = _eval_P(Pd, bd)
    fd_Q = _eval_Q(Qd, bd)

    # pqpq: ρ_PQ(r₁) × ρ_PQ(r₂) — P_a*P_b at r₁ with Q_c*Q_d at r₂
    rho_PP_ab = fa_P * fb_P
    rho_QQ_cd = fc_Q * fd_Q
    yk_QQ_cd = compute_yk(rho_QQ_cd, gr, grw, nv, n_gauss, k)
    result += np.sum(grw * rho_PP_ab * yk_QQ_cd)

    # qpqp: Q_a*Q_b at r₁ with P_c*P_d at r₂
    rho_QQ_ab = fa_Q * fb_Q
    rho_PP_cd = fc_P * fd_P
    yk_PP_cd = compute_yk(rho_PP_cd, gr, grw, nv, n_gauss, k)
    result += np.sum(grw * rho_QQ_ab * yk_PP_cd)

    return result


def compute_slater_fk_cell(Pa, Qa, Pb, Qb, bsp_data, k, cache=None):
    """Direct Slater integral F^k(a,b) = R^k(aa, bb) via cell algorithm."""
    return compute_slater_rk_cell(Pa, Qa, Pa, Qa, Pb, Qb, Pb, Qb,
                                   bsp_data, k, cache)


def compute_slater_gk_cell(Pa, Qa, Pb, Qb, bsp_data, k, cache=None):
    """Exchange Slater integral G^k(a,b) = R^k(ab, ba) via cell algorithm."""
    return compute_slater_rk_cell(Pa, Qa, Pb, Qb, Pb, Qb, Pa, Qa,
                                   bsp_data, k, cache)
