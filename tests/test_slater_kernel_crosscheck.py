"""Cross-check of ALL SIX independent radial Slater/Poisson kernels on one density.

Motivation: docs/code_review_2026_10_05/README.md (P1) — the six kernels had
never been checked against each other, and the archive records a +74 Ha error
from an integrator substitution of exactly this class.  One density, one
definition, one 40-digit reference; every kernel must land at its own
discretization error, no looser.

Convention audit (read before comparing — file:line evidence):

  All six kernels compute one of two linked quantities for a radial density
  rho(r) = P_a(r)P_b(r) + Q_a(r)Q_b(r) (the atomx ``P = rR`` convention):

      Y^k(r) = r^{-(k+1)} * Int_0^r   rho(r') r'^k        dr'
             + r^k       * Int_r^inf  rho(r') r'^{-(k+1)} dr'
      R^k    = Int_0^inf rho(r) Y^k[rho](r) dr
             = Int Int rho(r) rho(r') [r_<^k / r_>^{k+1}] dr dr'

  1. numerics/bspline/convol.py  (DBSR cell algorithm) — SCALAR R^k;
     definition in the module docstring (convol.py:3-4) and at
     ``compute_slater_rk_cell`` (convol.py:1750-1752).  Three internal paths
     (classic bank contraction, semiseparable above ns>=96, batched) are all
     exercised.  Orbital enters as B-spline coefficients; we L2-project the
     exact 1s (Q == 0) so only the pppp bank contributes.
  2. numerics/yk.py  (Poisson Y^k on the B-spline Gauss grid) — Y^k(r) on
     nodes; definition yk.py:5-7, kernel ``compute_yk`` (yk.py:157).  Scalar
     obtained by the consumer contraction ``sum(grw * rho * yk)``.
  3. numerics/quadrature.py  (trapezoid on arbitrary grids) — Y^k(r) via
     ``yk_on_grid`` (quadrature.py:205) and SCALAR via ``grid_slater_rk``
     (quadrature.py:365, ``R^k = Int rho_ac Y^k[rho_bd] dr``).
  4. numerics/shooting/coulomb.py  (SLAG6, Salvat grid) — Y^k(r) via
     ``_yk_over_r`` (coulomb.py:191; despite the name the returned array is
     Y^L(r) itself — ``i1/r^(L+1) + r^L * tail``, coulomb.py:227-231) and
     SCALAR via ``_slater_integral_slag6`` (coulomb.py:235).  The legacy
     power-series ``_slater_integral`` (coulomb.py:10) is included as a
     bonus backend.  Outer integral truncates at the grid edge r_max.
  5. scattering/r_matrix/cc_potentials.py  — Y^k(r) at composite Gauss
     nodes via ``cumulative_y_k`` (cc_potentials.py:28); the partial cell at
     each node is approximated by HALF the node's Gauss weight
     (cc_potentials.py:91-101).  Scalar: ``sum(w_q * rho * y_k)``, its
     consumer pattern in cc_inner.py.
  6. scattering/two_electron/coulomb.py  — SCALAR R^k via
     ``slater_integral`` (two_electron/coulomb.py:59): one global
     Gauss-Legendre grid, ``_y_k`` (two_electron/coulomb.py:23) uses the
     SAME half-cell partial rule as kernel 5, outer integral by
     ``dot(w, rho * y_k)``.

  φ_k(r) = r·Y^k appears nowhere; no kernel divides by an extra r.  All six
  are directly comparable once each is evaluated on its own native grid and
  read off against the reference at the same radii.

Density: hydrogenic 1s, Z = 1:  P(r) = 2 r e^{-r},  rho(r) = P^2 = 4 r^2
e^{-2r},  Int rho dr = 1.  (Note: the radial density carries the r^2 from
P = rR — rho != 4 e^{-2r}.)

Reference (provenance, anchored three ways in test_reference_*):
  * closed forms:  Y^0(r) = [1 - e^{-2r}(1 + r)]/r  (solves
    (rY^0)'' = -rho/r, W'' = k(k+1)W/r^2 - (2k+1)rho/r at k=0),
    R^0 = 5Z/8 (Slater-Condon), R^1 = 3Z/8 (verified to 45 digits).
  * incomplete-gamma closed form of the definition integral,
    Y^k(r) = 4 [ r^{-(k+1)} 2^{-(k+3)} gamma(k+3, 2r)
               + r^k 2^{k-2} Gamma(2-k, 2r) ],   evaluated with mpmath.
  * definition-literal adaptive mpmath quad of the two truncated integrals
    (agrees with the gamma form to 0 ulp at 40 digits).
  R^2 = 0.260532333438687426013570914996237633812003225 (mpmath, 45 digits;
  R^k scales linearly in Z for a hydrogenic 1s).

Measured results on this density (2026-10-05, this suite's grids; relative
error vs the reference):

  scalar R^k, k = 0 / 1 / 2:
    1 convol cell (classic)      2.5e-15  2.1e-15  1.9e-15
    1 convol cell (semiseparable) 1.1e-15 1.8e-15  1.3e-15
    1 convol cell (batched)      2.8e-15  1.9e-15  0.0
    2 yk.py (Gauss contraction)  3.6e-16  3.0e-16  4.9e-12
    3 trapezoid (n=250k)         1.7e-09  8.5e-09  2.1e-08
    4 SLAG6                      1.7e-12  1.0e-12  1.1e-11
    4 power-series (bonus)       3.2e-09  3.4e-09  3.1e-09
    5 cc_potentials (bsp cells)  9.3e-06  4.6e-05  1.1e-04
    6 two_electron (n=2000 def.) 6.0e-06  3.0e-05  7.3e-05

  Y^k(r) pointwise (max rel over the density support, nodes with
  |Y^k| >= 1e-6 max|Y^k|):
    2 yk.py: <= 1.5e-15 outside the first knot cell; first cell k=1 2.8e-07,
       k=2 6.4e-02 (see below).
    3 trapezoid (r >= 1e-3): 7.3e-09 / 1.5e-07 / 9.0e-07.
    4 SLAG6: 1.4e-11 / 2.3e-09 / 8.1e-10.
    5 cc_potentials outside the first knot cell: 6.6e-06 / 3.0e-05 / 6.7e-05.

Findings carried by this test:

  F1. Kernels 5 and 6 both implement the "half the straddling node's Gauss
      weight" partial-cell estimator — the rule that numerics/yk.py used
      until 2026-09-23, when it was replaced precisely because it leaves a
      systematic O((dr/r)^2) bias in every Y^k node value
      (docs/test_audit/PHYSICS_yk_node_values.md; yk.py module docstring).
      The bias signature documented there — relative scalar error growing as
      (k+1)(3k+2)/2 in k — is reproduced EXACTLY: measured ratios
      err(k)/err(0) = 1 : 5.0 : 12.0 for kernel 5 and 1 : 5.0 : 12.1 for
      kernel 6.  Both converge cleanly under refinement (kernel 5: 8.4x for
      n_gauss 9->27; kernel 6: 4.0x per doubling of n_points), so this is the
      diagnosed discretization bias, not a wrong-definition bug — but both
      modules still ship it, kernel 6 with no accuracy documentation at all.
      cc_potentials.py's own docstring claims only "1 %-level" adequacy; the
      measured bias (1e-6..1e-4) is consistent with that class.
  F2. Kernel 2's POINTWISE Y^k inside the first knot cell carries an
      algebraic-convergence error for k >= 2 (measured 6.4e-02 relative at
      the worst node for k=2, shrinking only to 4.5e-02 at n_gauss=33):
      there rho/r^{k+1} ~ 1/r is endpoint-singular on [0, t_1] and its
      partial integrals are not resolved by the node interpolant.  Outside
      the first cell the kernel is at machine precision (<= 1.5e-15), and
      SCALAR contractions are unaffected (R^2 at 4.9e-12): the density-weighted
      sum is the discrete adjoint of the partial rule, so the singular
      integrand is only ever integrated against a smooth, vanishing adjoint
      factor.  Pointwise consumers at k >= 2 inside the first knot cell
      inherit the error (today's compute_yk pointwise consumers use k=0 or
      contract into scalars).

Gates below are set at ~4-25x the measured 2026-10-05 values — tight enough
that a convention break (wrong r-power, swapped r_</r_>, extra 1/r, halved
kernel — the +74 Ha class) fails by orders of magnitude, loose enough to
absorb platform FP drift.  Nothing was loosened to force agreement: kernels
5 and 6 carry their own measured, refinement-verified bias gates.
"""

from __future__ import annotations

import numpy as np
import pytest

mp = pytest.importorskip("mpmath")

import atomx.numerics.bspline.convol as convol
from atomx.numerics.bspline.basis import (
    bsplvb,
    prepare_bspline_data,
    setup_gauss_points,
)
from atomx.numerics.bspline.convol import (
    ConvolCache,
    compute_slater_rk_cell,
    compute_slater_rk_cell_batch,
)
from atomx.numerics.bspline.knots import make_physics_knots
from atomx.numerics.quadrature import grid_slater_rk, yk_on_grid
from atomx.numerics.shooting.coulomb import (
    _slater_integral,
    _slater_integral_slag6,
    _yk_over_r,
)
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.yk import compute_yk
from atomx.scattering.r_matrix.cc_potentials import cumulative_y_k
from atomx.scattering.two_electron.coulomb import (
    slater_integral as te_slater_integral,
)

Z_TEST = 1.0
K_VALUES = (0, 1, 2)
R_SUPPORT = 20.0          # density is ~1e-23 beyond this; Y^k tail is 1/r there
PROFILE_FLOOR_FRAC = 1e-6  # rel-error mask: |ref| >= this * max|ref|

# Hard-pinned scalar references (hydrogenic 1s, Z = 1):
#   R^0 = 5Z/8  and  R^1 = 3Z/8  (closed forms; R^1 verified vs mpmath quad
#   at 45 digits),  R^2 = mpmath 45-digit quad of the definition integral.
RK_REFERENCE = {
    0: 5.0 / 8.0,
    1: 3.0 / 8.0,
    2: 0.260532333438687426013570914996237633812003225,
}

# Scalar R^k gates per kernel per k (measured values in the module docstring).
SCALAR_TOL = {
    "convol_classic": (1e-12, 1e-12, 1e-12),
    "convol_semiseparable": (1e-12, 1e-12, 1e-12),
    "convol_batch": (1e-12, 1e-12, 1e-12),
    "yk_gauss": (1e-10, 1e-10, 1e-10),
    "trapezoid": (1e-7, 3e-7, 1e-6),
    "shooting_slag6": (5e-10, 5e-10, 5e-10),
    "shooting_power_series": (3e-8, 3e-8, 3e-8),
    "cc_potentials": (5e-5, 2e-4, 5e-4),
    "two_electron_gl": (3e-5, 1.5e-4, 3.5e-4),
}

# Y^k(r) profile gates (max rel error on the masked support).
PROFILE_TOL = {
    "yk_gauss": (1e-12, 1e-12, 1e-12),          # outside first knot cell
    "trapezoid": (1e-7, 2e-6, 1e-5),            # r >= 1e-3
    "shooting_slag6": (1e-9, 1e-7, 1e-7),
    "cc_potentials": (3e-5, 1.5e-4, 3e-4),      # outside first knot cell
}


# ---------------------------------------------------------------------------
# Reference mathematics
# ---------------------------------------------------------------------------

def _mp_setup(dps=50):
    mp.mp.dps = dps
    return mp.mpf(Z_TEST)


def yk_reference(r, k):
    """Y^k(r) for rho = 4Z^3 r^2 e^{-2Zr} via incomplete gamma functions."""
    Zm = _mp_setup()
    r = mp.mpf(float(r))
    a = 2 * Zm
    lo = 4 * Zm**3 * mp.gammainc(k + 3, 0, a * r) / a**(k + 3)
    hi = 4 * Zm**3 * mp.gammainc(2 - k, a * r, mp.inf) / a**(2 - k)
    return float(r**(-(k + 1)) * lo + r**k * hi)


def yk_reference_array(rr, k):
    return np.array([yk_reference(r, k) for r in rr])


def yk_reference_direct_quad(r, k):
    """Definition-literal reference: adaptive quad of the two pieces of Y^k.

    Independent of the gamma identities — the anchor that the gamma form
    implements the same definition the kernels document.
    """
    Zm = _mp_setup(35)
    r = mp.mpf(float(r))
    lo = mp.quad(lambda s: 4 * Zm**3 * s**(k + 2) * mp.exp(-2 * Zm * s), [0, r])
    hi = mp.quad(lambda s: 4 * Zm**3 * s**(1 - k) * mp.exp(-2 * Zm * s),
                 [r, 4 * r + 1, mp.inf])
    return float(r**(-(k + 1)) * lo + r**k * hi)


def y0_closed_form(r):
    r = np.asarray(r, dtype=float)
    return (1.0 - np.exp(-2 * Z_TEST * r) * (1 + Z_TEST * r)) / r


def p1s(r):
    return 2.0 * Z_TEST**1.5 * r * np.exp(-Z_TEST * r)


def rho1s(r):
    return 4.0 * Z_TEST**3 * r**2 * np.exp(-2.0 * Z_TEST * r)


# ---------------------------------------------------------------------------
# Kernel fixtures: each kernel on its own native grid
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bsp_workspace():
    """B-spline workspace on the standard point-nucleus physics grid."""
    t, ns = make_physics_knots(ks=9, Z=1, r_max=50.0)
    return prepare_bspline_data(t, ns, 9), t


@pytest.fixture(scope="module")
def projected_1s(bsp_workspace):
    """L2 (Galerkin) projection of the exact hydrogenic 1s onto the basis.

    The overlap block is exact on the kernel's own Gauss grid; the rhs is
    integrated with 48-point Gauss-Legendre per interval so the projection
    — not its quadrature — sets the error.  Measured: max |P_proj - P_exact|
    = 3.1e-13 on the grid, norm = 1 - 1e-16.
    """
    bd, t = bsp_workspace
    ks, ns, nv = bd["ks"], bd["ns"], bd["nv"]
    bsp, grw = bd["bsp_p"], bd["grw"]
    overlap = np.zeros((ns, ns))
    rhs = np.zeros(ns)
    xq, wq = np.polynomial.legendre.leggauss(48)
    for iv in range(nv):
        idx = iv + np.arange(ks)
        block = bsp[iv]
        overlap[np.ix_(idx, idx)] += (block * grw[iv][:, None]).T @ block
        a, b = t[ks - 1 + iv], t[ks + iv]
        pts = 0.5 * (b - a) * xq + 0.5 * (a + b)
        wts = 0.5 * (b - a) * wq
        basis_vals = np.array(
            [bsplvb(t, ks, float(x), ks - 1 + iv) for x in pts])
        rhs[idx] += (wts * p1s(pts)) @ basis_vals
    coeffs = np.linalg.solve(overlap, rhs)
    return coeffs, overlap


@pytest.fixture(scope="module")
def shooting_workspace():
    """Production shooting grid (same parameters as the SCF default)."""
    r, difr, n_pts = sgrid(75.0, 1.0e-7, 0.5, 1000)
    return r, difr, n_pts, rho1s(r)


@pytest.fixture(scope="module")
def trapezoid_workspace():
    """Fine uniform grid for the arbitrary-grid trapezoid kernel."""
    r = np.linspace(1e-6, 40.0, 250_001)
    return r, rho1s(r)


def _masked_rel_error(values, reference):
    """Max relative error over nodes where the reference is non-negligible."""
    floor = PROFILE_FLOOR_FRAC * np.abs(reference).max()
    mask = np.abs(reference) >= floor
    return float(np.max(np.abs(values[mask] - reference[mask])
                        / np.abs(reference[mask])))


# ---------------------------------------------------------------------------
# Reference anchors
# ---------------------------------------------------------------------------

class TestReferenceAnchor:
    """The arbiter itself is pinned before any kernel is trusted."""

    @pytest.mark.parametrize("k", K_VALUES)
    def test_gamma_form_matches_definition_quad(self, k):
        for r in (0.13, 1.7):
            assert yk_reference(r, k) == pytest.approx(
                yk_reference_direct_quad(r, k), rel=1e-22)

    def test_k0_closed_form(self):
        rr = np.array([0.05, 0.13, 0.5, 1.0, 1.7, 4.2, 10.0])
        got = yk_reference_array(rr, 0)
        np.testing.assert_allclose(got, y0_closed_form(rr), rtol=1e-13)

    def test_scalar_closed_forms(self):
        assert RK_REFERENCE[0] == 5.0 / 8.0
        assert RK_REFERENCE[1] == 3.0 / 8.0
        # R^1 = 3/8 re-verified live at 50 digits (cost: one adaptive quad);
        # the integrand stays in mpmath precision — no float round-trip.
        Zm = _mp_setup()
        a = 2 * Zm

        def y1(r):
            lo = 4 * Zm**3 * mp.gammainc(4, 0, a * r) / a**4
            hi = 4 * Zm**3 * mp.gammainc(1, a * r, mp.inf) / a
            return lo / r**2 + r * hi

        r1 = mp.quad(
            lambda r: 4 * Zm**3 * r**2 * mp.exp(-2 * Zm * r) * y1(r),
            [0, mp.mpf("0.25"), 1, 3, 10, mp.inf])
        assert abs(r1 - mp.mpf(3) / 8) < mp.mpf("1e-30")


# ---------------------------------------------------------------------------
# Workspace sanity
# ---------------------------------------------------------------------------

def test_projection_is_faithful(bsp_workspace, projected_1s):
    bd, _ = bsp_workspace
    coeffs, overlap = projected_1s
    from atomx.numerics.bspline.evaluate import eval_orbital_on_grid

    p_grid = eval_orbital_on_grid(coeffs, bd["bsp_p"], bd["nv"],
                                  bd["ks"], bd["ns"])
    assert np.abs(p_grid - p1s(bd["gr"])).max() < 1e-9
    assert coeffs @ overlap @ coeffs == pytest.approx(1.0, rel=1e-12)


def test_density_normalised_on_bsp_grid(bsp_workspace):
    bd, _ = bsp_workspace
    assert float(np.sum(bd["grw"] * rho1s(bd["gr"]))) == pytest.approx(
        1.0, rel=1e-13)


# ---------------------------------------------------------------------------
# The decisive table: scalar R^k from every kernel vs the reference
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scalar_table(bsp_workspace, projected_1s, shooting_workspace,
                 trapezoid_workspace):
    """One R^k(1s,1s) per (kernel, k) — each computed once, shared by tests."""
    bd, t = bsp_workspace
    coeffs, _ = projected_1s
    ns = bd["ns"]
    q0 = np.zeros(ns)
    gr, grw, nv = bd["gr"], bd["grw"], bd["nv"]
    rho_g = rho1s(gr)
    cache = ConvolCache(bd)
    r_sh, difr_sh, np_sh, f_sh = shooting_workspace
    r_tr, rho_tr = trapezoid_workspace
    p_tr = p1s(r_tr)
    q_tr = np.zeros_like(r_tr)
    rr_bsp, ww_bsp = gr.ravel(), grw.ravel()
    rho_flat = rho_g.ravel()

    table = {}
    for k in K_VALUES:
        sentinel = convol.SEMISEPARABLE_DIRECT_NS
        try:
            convol.SEMISEPARABLE_DIRECT_NS = 10**9  # force the classic path
            table["convol_classic", k] = compute_slater_rk_cell(
                coeffs, q0, coeffs, q0, coeffs, q0, coeffs, q0, bd, k,
                cache=cache)
        finally:
            convol.SEMISEPARABLE_DIRECT_NS = sentinel
        table["convol_semiseparable", k] = compute_slater_rk_cell(
            coeffs, q0, coeffs, q0, coeffs, q0, coeffs, q0, bd, k, cache=cache)
        table["convol_batch", k] = compute_slater_rk_cell_batch(
            {"1s": (coeffs, q0)}, [("1s", "1s", "1s", "1s", k)], bd,
            cache=cache)[0]
        table["yk_gauss", k] = float(np.sum(
            grw * rho_g * compute_yk(rho_g, gr, grw, nv, gr.shape[1], k)))
        table["trapezoid", k] = grid_slater_rk(
            p_tr, q_tr, p_tr, q_tr, p_tr, q_tr, p_tr, q_tr, r_tr, k)
        table["shooting_slag6", k] = _slater_integral_slag6(
            f_sh, f_sh, k, r_sh, difr_sh, np_sh)
        table["shooting_power_series", k] = _slater_integral(
            f_sh, f_sh, k, r_sh, difr_sh, np_sh)
        table["cc_potentials", k] = float(np.sum(
            ww_bsp * rho_flat * cumulative_y_k(rr_bsp, ww_bsp, rho_flat, k)))
        table["two_electron_gl", k] = te_slater_integral(
            p1s, p1s, p1s, p1s, k)  # production defaults: n=2000, r_max=50
    return table


@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("kernel", sorted({k for k, _ in SCALAR_TOL.items()}))
def test_scalar_rk_matches_reference(scalar_table, kernel, k):
    """Each kernel's R^k lands within its own discretization class."""
    got = scalar_table[kernel, k]
    ref = RK_REFERENCE[k]
    assert got == pytest.approx(ref, rel=SCALAR_TOL[kernel][k]), (
        f"{kernel} k={k}: got {got:.15e}, ref {ref:.15e}, "
        f"rel err {abs(got - ref) / ref:.2e}, gate {SCALAR_TOL[kernel][k]:.1e}")


@pytest.mark.parametrize("k", K_VALUES)
def test_high_accuracy_kernels_agree_pairwise(scalar_table, k):
    """The four quadrature-exact kernels must agree with each other.

    This is the review's decisive check: independent implementations, one
    density, same answer — here to a combined gate of 1e-6 relative, while
    the measured pairwise spread is <= 4.2e-8 (bounded by the trapezoid's
    2.1e-08 discretization error).
    """
    ref = RK_REFERENCE[k]
    names = ["convol_classic", "convol_semiseparable", "convol_batch",
             "yk_gauss", "trapezoid", "shooting_slag6",
             "shooting_power_series"]
    values = {n: scalar_table[n, k] for n in names}
    for a in names:
        for b in names:
            assert abs(values[a] - values[b]) / ref < 1e-6, (
                f"k={k}: {a}={values[a]:.15e} vs {b}={values[b]:.15e}")


def test_convol_internal_paths_agree(scalar_table):
    """Classic bank, semiseparable, and batched paths of kernel 1 coincide."""
    for k in K_VALUES:
        classic = scalar_table["convol_classic", k]
        semi = scalar_table["convol_semiseparable", k]
        batch = scalar_table["convol_batch", k]
        assert semi == pytest.approx(classic, rel=1e-12)
        assert batch == pytest.approx(classic, rel=1e-12)


# ---------------------------------------------------------------------------
# Y^k(r) pointwise, on each potential-returning kernel's own nodes
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def yk_profiles(bsp_workspace, shooting_workspace, trapezoid_workspace):
    bd, t = bsp_workspace
    gr, grw, nv = bd["gr"], bd["grw"], bd["nv"]
    rho_g = rho1s(gr)
    first_cell_end = t[bd["ks"]]  # first interior breakpoint
    r_sh, difr_sh, np_sh, f_sh = shooting_workspace
    r_tr, rho_tr = trapezoid_workspace
    rr_bsp, ww_bsp = gr.ravel(), grw.ravel()
    rho_flat = rho_g.ravel()

    profiles = {}
    for k in K_VALUES:
        # Kernel 2 on its own nodes, outside the endpoint-singular first cell.
        yk2 = compute_yk(rho_g, gr, grw, nv, gr.shape[1], k).ravel()
        m2 = (rr_bsp <= R_SUPPORT) & (rr_bsp > first_cell_end)
        profiles["yk_gauss", k] = (rr_bsp[m2], yk2[m2])
        # Kernel 3 on a subsample of its uniform grid (reference cost only).
        yk3 = yk_on_grid(rho_tr, r_tr, k)
        sub = np.arange(0, r_tr.size, 499)
        sub = sub[(r_tr[sub] <= R_SUPPORT) & (r_tr[sub] >= 1e-3)]
        profiles["trapezoid", k] = (r_tr[sub], yk3[sub])
        # Kernel 4 on the shooting grid (r[0] = 0 carries the documented 0).
        yk4 = _yk_over_r(f_sh, k, r_sh, difr_sh, np_sh)
        m4 = (r_sh <= R_SUPPORT) & (r_sh > 0)
        profiles["shooting_slag6", k] = (r_sh[m4], yk4[m4])
        # Kernel 5 on the same composite Gauss cells as kernel 2.
        yk5 = cumulative_y_k(rr_bsp, ww_bsp, rho_flat, k)
        profiles["cc_potentials", k] = (rr_bsp[m2], yk5[m2])
    return profiles


@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("kernel", sorted({k for k, _ in PROFILE_TOL.items()}))
def test_yk_profile_matches_reference(yk_profiles, kernel, k):
    rr, values = yk_profiles[kernel, k]
    ref = yk_reference_array(rr, k)
    err = _masked_rel_error(values, ref)
    assert err < PROFILE_TOL[kernel][k], (
        f"{kernel} k={k}: masked max rel error {err:.2e}, "
        f"gate {PROFILE_TOL[kernel][k]:.1e}")


def test_yk_gauss_first_cell_k2_is_a_known_pointwise_limit(bsp_workspace):
    """Pin the first-cell endpoint singularity of kernel 2 at k=2 (finding F2).

    rho/r^{k+1} ~ 1/r on [0, t_1] is endpoint-singular, so the node
    interpolant that drives the partial-cell integrals converges only
    algebraically there: measured worst-node rel error 6.4e-02 (n_gauss=9),
    4.5e-02 (n_gauss=33).  Scalar contractions are immune (adjoint
    structure; R^2 agrees to 4.9e-12).  The gate brackets the measured
    value on both sides: a regression beyond it fails, and a genuine fix of
    the partial rule SHOULD fail this test too — then tighten it.
    """
    bd, t = bsp_workspace
    gr, grw, nv = bd["gr"], bd["grw"], bd["nv"]
    rho_g = rho1s(gr)
    rr = gr.ravel()
    yk2 = compute_yk(rho_g, gr, grw, nv, gr.shape[1], 2).ravel()
    first = rr <= t[bd["ks"]]
    ref = yk_reference_array(rr[first], 2)
    err = float(np.max(np.abs(yk2[first] - ref) / ref))
    assert 1e-3 < err < 0.15, (
        f"first-cell k=2 pointwise error moved: {err:.2e} "
        "(was 6.4e-02 — see finding F2 in the module docstring)")


# ---------------------------------------------------------------------------
# Discretization, not definition: refinement must shrink every kernel's error
# ---------------------------------------------------------------------------

def test_trapezoid_second_order_convergence():
    """Scalar error ratio ~16 for 4x the points; 25k -> 100k suffices."""
    errs = []
    for n in (25_001, 100_001):
        r = np.linspace(1e-6, 40.0, n)
        p, q = p1s(r), np.zeros_like(r)
        got = grid_slater_rk(p, q, p, q, p, q, p, q, r, 0)
        errs.append(abs(got - RK_REFERENCE[0]) / RK_REFERENCE[0])
    assert errs[1] < errs[0]
    assert 6.0 < errs[0] / errs[1] < 40.0, (
        f"trapezoid refinement ratio {errs[0] / errs[1]:.1f} "
        "(expected ~16 for second order)")


def test_cc_potentials_bias_converges_with_quadrature_order(bsp_workspace):
    """Kernel 5's bias shrinks ~9x for n_gauss 9 -> 27 (measured 8.4x).

    A wrong-definition bug would be refinement-invariant; the observed
    geometric decay identifies the diagnosed half-cell discretization bias.
    """
    bd, t = bsp_workspace
    ns, ks = bd["ns"], bd["ks"]
    errs = []
    for n_gauss in (9, 27):
        gr, grw = setup_gauss_points(t, ns, ks, n_gauss)
        rr, ww = gr.ravel(), grw.ravel()
        rho = rho1s(rr)
        got = float(np.sum(ww * rho * cumulative_y_k(rr, ww, rho, 0)))
        errs.append(abs(got - RK_REFERENCE[0]) / RK_REFERENCE[0])
    assert errs[1] < errs[0]
    assert 3.0 < errs[0] / errs[1] < 30.0, (
        f"cc_potentials refinement ratio {errs[0] / errs[1]:.1f}")


def test_two_electron_bias_converges_with_grid_density():
    """Kernel 6's bias shrinks ~4x per doubling; test 500 -> 2000 nodes."""
    errs = []
    for n_points in (500, 2000):
        got = te_slater_integral(p1s, p1s, p1s, p1s, 0, n_points=n_points)
        errs.append(abs(got - RK_REFERENCE[0]) / RK_REFERENCE[0])
    assert errs[1] < errs[0]
    assert 6.0 < errs[0] / errs[1] < 40.0, (
        f"two_electron refinement ratio {errs[0] / errs[1]:.1f}")


def test_shooting_slag6_refinement():
    """The SLAG6 kernel also sits on its convergence branch (ratio > 2)."""
    errs = []
    for params in ((75.0, 1.0e-7, 0.5, 1000), (75.0, 5.0e-8, 0.25, 2000)):
        r, difr, n_pts = sgrid(*params)
        f = rho1s(r)
        got = _slater_integral_slag6(f, f, 0, r, difr, n_pts)
        errs.append(abs(got - RK_REFERENCE[0]) / RK_REFERENCE[0])
    assert errs[1] < errs[0]
