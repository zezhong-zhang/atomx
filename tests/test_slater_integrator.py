"""Slater R^k integrator: method choice, agreement, and defaults.

The numerics integrator (:mod:`atomx.numerics.bspline.slater`) offers two
interchangeable kernels — the fast O(N) Poisson method (default) and the
textbook O(N²) direct double sum.  These tests verify:

* both kernels compute the same R^k on physical densities;
* the default is the fast Poisson method;
* the Auger public API honours the ``method`` choice;
* one R^k is compared to a **closed form** (hydrogenic 1s, R⁰ = 5Z/8) rather
  than to the other kernel.

Measured agreement between the two kernels on the Ne 1s×2p / 2s×2p densities
(2026-09-22): k=0 2.19e-14, k=1 4.68e-14, k=2 2.72e-14, k=3 3.06e-14 —
machine precision.  The previous gate was rel=5e-2 (and rel=1e-1 on the
Auger rate) with a comment claiming a "~1e-3 rank-dependent residual from
the direct method's diagonal (r1≈r2) product-quadrature error"; that comment
was false — there is no such residual — and a 5 % gate on a 1e-14 agreement
cannot see a wrong R^k at all.
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

    solver = DFSolver(Z=10, subshells=neon_subshells(), ns=60, ks=9,
                      exchange="hf")
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


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_poisson_matches_direct(ne_grid, k):
    """The two kernels must agree to machine precision.

    Measured 2026-09-22: k=0 2.19e-14, k=1 4.68e-14, k=2 2.72e-14,
    k=3 3.06e-14 relative; gate 1e-10 is ~2000x the measurement (the old
    rel=5e-2 was 5x10^11 times looser and could not fail for a wrong R^k).
    """
    bsp, rho13, rho24 = ne_grid
    fast = slater_rk(rho13, rho24, bsp, k, method="poisson")
    slow = slater_rk(rho13, rho24, bsp, k, method="direct")
    assert fast == pytest.approx(slow, rel=1e-10)


@pytest.mark.parametrize("Z", [1, 10, 26])
def test_r0_of_hydrogenic_1s_is_the_closed_form(ne_grid, Z):
    """R⁰(1s,1s) = F⁰(1s,1s) = 5Z/8 for a hydrogenic 1s (Slater–Condon).

    The 1s radial function is P(r) = 2 Z^{3/2} r e^{−Zr}, normalised as
    ∫P²dr = 1, so with ρ₁₁ = P²

        R⁰(1s,1s) = ∫∫ ρ₁₁(r) (r_<⁰/r_>¹) ρ₁₁(r′) dr dr′ = 5Z/8

    (independently re-integrated with scipy: 0.62500004 at Z=1,
    6.25000013 at Z=10).  Evaluated here on the Ne physics quadrature grid,
    which is what the rest of this file uses; measured 2.84e-5 relative at
    Z = 1, 10 and 26 — a *fixed* relative deviation, i.e. the Poisson
    kernel's quadrature accuracy rather than a peak-resolution effect (a
    smooth ρ = e^{−r} shows 1.37e-4 on the same grid).  Gate rel=1e-4,
    3.5x the measurement — and this is the only assertion in the file that
    would catch a *systematically* wrong R^k, since the two kernels share
    the Y^k construction.
    """
    bsp = ne_grid[0]
    gr = bsp["gr"]
    P = 2.0 * Z ** 1.5 * gr * np.exp(-Z * gr)
    rho11 = P * P
    got = slater_rk(rho11, rho11, bsp, 0, method="poisson")
    assert got == pytest.approx(5.0 * Z / 8.0, rel=1e-4), (
        f"R0(1s,1s) Z={Z}: atomx={got!r}, closed form 5Z/8={5.0*Z/8.0!r}"
    )


def test_dispatch_matches_direct_helpers(ne_grid):
    bsp, rho13, rho24 = ne_grid
    gr, grw, nv = bsp["gr"], bsp["grw"], bsp["nv"]
    n_gauss = gr.shape[1]
    assert slater_rk(rho13, rho24, bsp, 1, method="poisson") == pytest.approx(
        slater_rk_poisson(rho13, rho24, gr, grw, nv, n_gauss, 1), rel=1e-12
    )
    assert slater_rk(rho13, rho24, bsp, 1, method="direct") == pytest.approx(
        slater_rk_direct(rho13.ravel(), rho24.ravel(),
                         gr.ravel(), grw.ravel(), 1), rel=1e-12
    )


def test_unknown_method_raises(ne_grid):
    bsp, rho13, rho24 = ne_grid
    with pytest.raises(ValueError, match="unknown Slater integrator"):
        slater_rk(rho13, rho24, bsp, 1, method="simpson")


def test_auger_api_honours_method(ne_grid):
    from atomx.methods.dirac_fock import DFSolver, neon_subshells
    from atomx.scattering.autoionization.auger import auger_rate_two_electron

    solver = DFSolver(Z=10, subshells=neon_subshells(), ns=60, ks=9,
                      exchange="hf")
    res = solver.solve(max_iter=60, tol=1e-7)
    orbs = {o[4]: o for o in solver._flatten_orbitals(res.orbitals_raw)}
    bsp = res._bsp_data
    a, b, c, e = (orbs["1s1/2"], orbs["2s1/2"], orbs["2p1/2"], orbs["2p3/2"])
    common = dict(angular_W=0.3, k=1, bsp_data=bsp)
    fast = auger_rate_two_electron(a[0], a[1], b[0], b[1], c[0], c[1],
                                   e[0], e[1], method="poisson", **common)
    slow = auger_rate_two_electron(a[0], a[1], b[0], b[1], c[0], c[1],
                                   e[0], e[1], method="direct", **common)
    # rate ∝ R_k², so the rate tolerance is ~2x the integral tolerance.
    # Measured agreement 6.86e-14 relative (gate 1e-10; was rel=1e-1).
    assert fast == pytest.approx(slow, rel=1e-10)
    assert fast > 0.0
