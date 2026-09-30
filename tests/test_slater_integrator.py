"""Independent radial Coulomb quadratures, including the coincident-cell cusp.

The default O(N) Poisson recursion is checked against a direct double
integral that splits each coincident cell into two triangles. Both are
also checked against closed-form and high-precision integral values, so
agreement between implementations alone is never the accuracy argument.
"""

import numpy as np
import pytest

from atomx.numerics.bspline.slater import (
    AVAILABLE_METHODS,
    DEFAULT_METHOD,
    slater_rk,
    slater_rk_direct,
    slater_rk_poisson,
)


@pytest.fixture(scope="module")
def ne_grid():
    from atomx.methods.dirac_fock import DFSolver, neon_subshells
    from atomx.numerics.bspline.evaluate import eval_orbital_on_grid

    solver = DFSolver(Z=10, subshells=neon_subshells(), ns=60, ks=9, exchange="hf")
    res = solver.solve(max_iter=60, tol=1e-7)
    orbs = solver._flatten_orbitals(res.orbitals_raw)
    bsp = res._bsp_data
    nv = bsp["nv"]
    ksp = bsp.get("ksp", bsp["ks"])
    nsp = bsp.get("nsp", bsp["ns"])

    def P(coeffs):
        return eval_orbital_on_grid(coeffs, bsp["bsp_p"], nv, ksp, nsp).reshape(-1)

    od = {o[4]: o for o in orbs}
    rho13 = P(od["1s1/2"][0]) * P(od["2p1/2"][0])
    rho24 = P(od["2s1/2"][0]) * P(od["2p3/2"][0])
    return bsp, rho13, rho24


def test_defaults_are_the_fast_method():
    assert DEFAULT_METHOD == "poisson"
    assert set(AVAILABLE_METHODS) == {"poisson", "direct"}


@pytest.mark.parametrize('n_gauss,p,q', [(2,0,0), (5,2,3), (9,4,6)])
@pytest.mark.parametrize('rank', [0,3,12])
def test_direct_origin_cell_against_monomial_integral(n_gauss, p, q, rank):
    """Analytic double integral probes the cusp, corner and multipole order."""
    x, w = np.polynomial.legendre.leggauss(n_gauss)
    radius = .7
    r, weights = (radius*(x+1)/2)[None,:], (radius*w/2)[None,:]
    expected = radius**(p+q+1)/(p+q+1)*(1/(p+rank+1)+1/(q+rank+1))
    assert slater_rk_direct(r**p, r**q, r, weights, rank) == pytest.approx(expected, rel=2e-13)
    assert slater_rk_direct(r**q, r**p, r, weights, rank) == pytest.approx(expected, rel=2e-13)


@pytest.mark.parametrize("k", range(7))
def test_poisson_matches_direct(ne_grid, k):
    """Independent integration paths agree after resolving the r=s cusp."""
    bsp, rho13, rho24 = ne_grid
    fast = slater_rk(rho13, rho24, bsp, k, method="poisson")
    slow = slater_rk(rho13, rho24, bsp, k, method="direct")
    assert fast == pytest.approx(slow, rel=2e-12, abs=1e-14)


@pytest.mark.parametrize("Z", [1, 10, 26])
def test_r0_of_hydrogenic_1s_is_the_closed_form(ne_grid, Z):
    """R⁰(1s,1s) = F⁰(1s,1s) = 5Z/8 for a hydrogenic 1s (Slater–Condon).

    The 1s radial function is P(r) = 2 Z^{3/2} r e^{−Zr}, normalised as
    ∫P²dr = 1, so with ρ₁₁ = P²

        R⁰(1s,1s) = ∫∫ ρ₁₁(r) (r_<⁰/r_>¹) ρ₁₁(r′) dr dr′ = 5Z/8

    This analytic value validates each independent kernel, including the
    near-origin triangle where the Coulomb kernel itself is unbounded.
    """
    bsp = ne_grid[0]
    gr = bsp["gr"]
    P = 2.0 * Z**1.5 * gr * np.exp(-Z * gr)
    rho11 = P * P
    for method in AVAILABLE_METHODS:
        got = slater_rk(rho11, rho11, bsp, 0, method=method)
        assert got == pytest.approx(5.0 * Z / 8.0, rel=1e-8), (method, Z, got)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_rk_of_exponential_density_against_mpmath(ne_grid, k):
    """R^k(ρ, ρ) for ρ = e^{−r} against a 40-digit mpmath reference.

    k=0 is the closed form ln 4 = 1.38629436112…; k>0 uses mpmath's
    ``gammainc``.  This density is *not* a B-spline polynomial, so it
    exercises the interpolatory partial-interval rule beyond the class it
    interpolates exactly; with the old half-weight rule it measured
    1.37e-4 / 1.03e-3 / 2.41e-3 / 4.17e-3 for k = 0…3 (growing as the
    diagnosed (k+1)(3k+2)/2 signature).  After the fix it measures at
    machine precision; gate rel=1e-8.  Skipped if mpmath is unavailable —
    the closed-form gates above still stand alone.
    """
    pytest.importorskip("mpmath")
    from mpmath import exp, gammainc, inf, mp, mpf, quad

    mp.dps = 40

    def yk_ref(rv):
        r = mpf(rv)
        return (
            gammainc(k + 1, 0, r)
            + r ** (2 * k + 1) * quad(lambda s: exp(-s) / s ** (k + 1), [r, inf])
        ) / r ** (k + 1)

    ref = quad(lambda r: exp(-r) * yk_ref(r), [0, 1, 3, 10, inf])
    ref = float(ref)

    bsp = ne_grid[0]
    gr = bsp["gr"]
    rho = np.exp(-gr)
    for method in AVAILABLE_METHODS:
        got = slater_rk(rho, rho, bsp, k, method=method)
        assert got == pytest.approx(ref, rel=1e-8), (method, k, got, ref)


@pytest.mark.parametrize("n_gauss", [5, 9, 17])
def test_yk_partial_interval_matrices(n_gauss):
    """Pin the construction of the interpolatory partial-interval matrices.

    With reference-cell weights g_j = wi_j f(ξ_j) (unit Jacobian), the
    outward partial at node i is sum_j g_j mo[i, j] and must integrate the
    degree n_gauss−1 interpolant of f exactly:

    * mo + mi == 1 elementwise — θ[i,j] + ψ[i,j] = ∫₋₁¹ L_j = wi_j, so
      mo[i,j] + mi[i,j] = wi_j/wi_j (the two partials tile the cell);
    * constant density: mo @ wi == 1 + ξ and mi @ wi == 1 − ξ, i.e. the
      partials of f ≡ 1 are ∫₋₁^{ξ_i} 1 = 1 + ξ_i;
    * degree-exactness: for f(ξ) = ξ³ + 2ξ the partials match the
      polynomial integral to machine precision.
    """
    from atomx.numerics.bspline.basis import gauss_legendre_points
    from atomx.numerics.yk import yk_partial_matrices

    xi, wi = gauss_legendre_points(n_gauss)
    mo, mi = yk_partial_matrices(n_gauss)
    assert np.allclose(mo + mi, 1.0, atol=1e-15)
    assert np.allclose(mo @ wi, 1.0 + xi, atol=1e-15)
    assert np.allclose(mi @ wi, 1.0 - xi, atol=1e-15)

    f = xi**3 + 2.0 * xi
    g = wi * f
    exact_out = xi**4 / 4.0 + xi**2 - 1.25
    exact_in = 1.25 - xi**4 / 4.0 - xi**2
    assert np.allclose(g @ mo.T, exact_out, atol=1e-15)
    assert np.allclose(g @ mi.T, exact_in, atol=1e-15)


def test_dispatch_matches_direct_helpers(ne_grid):
    bsp, rho13, rho24 = ne_grid
    gr, grw, nv = bsp["gr"], bsp["grw"], bsp["nv"]
    n_gauss = gr.shape[1]
    assert slater_rk(rho13, rho24, bsp, 1, method="poisson") == pytest.approx(
        slater_rk_poisson(rho13, rho24, gr, grw, nv, n_gauss, 1), rel=1e-12
    )
    assert slater_rk(rho13, rho24, bsp, 1, method="direct") == pytest.approx(
        slater_rk_direct(rho13, rho24, gr, grw, 1),
        rel=1e-12,
    )


def test_unknown_method_raises(ne_grid):
    bsp, rho13, rho24 = ne_grid
    with pytest.raises(ValueError, match="unknown Slater integrator"):
        slater_rk(rho13, rho24, bsp, 1, method="simpson")


def test_auger_api_honours_method(ne_grid):
    from atomx.methods.dirac_fock import DFSolver, neon_subshells
    from atomx.scattering.autoionization.auger import auger_rate_two_electron

    solver = DFSolver(Z=10, subshells=neon_subshells(), ns=60, ks=9, exchange="hf")
    res = solver.solve(max_iter=60, tol=1e-7)
    orbs = {o[4]: o for o in solver._flatten_orbitals(res.orbitals_raw)}
    bsp = res._bsp_data
    a, b, c, e = (orbs["1s1/2"], orbs["2s1/2"], orbs["2p1/2"], orbs["2p3/2"])
    common = dict(angular_W=0.3, k=1, bsp_data=bsp)
    fast = auger_rate_two_electron(
        a[0], a[1], b[0], b[1], c[0], c[1], e[0], e[1], method="poisson", **common
    )
    slow = auger_rate_two_electron(
        a[0], a[1], b[0], b[1], c[0], c[1], e[0], e[1], method="direct", **common
    )
    # rate ∝ R_k², so the method disagreement roughly doubles the integral's:
    # measured 1.67e-4 relative after the Y^k partial-interval fix
    # (2026-09-23), the direct side carrying the (k+1)(3k+2)/2 quadrature
    # error the poisson path used to share.  Gate 1e-3, ~600x the
    # measurement; the absolute correctness of the poisson path is pinned
    # by the closed-form / mpmath gates above.
    assert fast == pytest.approx(slow, rel=2e-12, abs=1e-14)
    assert fast > 0.0
