"""Slater R^k integrator: method choice, agreement, and defaults.

The numerics integrator (:mod:`atomx.numerics.bspline.slater`) offers two
interchangeable kernels — the fast O(N) Poisson method (default) and the
textbook O(N²) direct double sum.  These tests verify:

* both kernels compute the same R^k on physical densities;
* the default is the fast Poisson method;
* the Auger public API honours the ``method`` choice.

The direct O(N²) result serves as the independent cross-check for the
Poisson default; the small residual (rank-dependent, ``~1e-3``) is the
direct method's diagonal (r₁≈r₂) product-quadrature error, which Poisson
resolves analytically per interval.
"""

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
    bsp, rho13, rho24 = ne_grid
    fast = slater_rk(rho13, rho24, bsp, k, method="poisson")
    slow = slater_rk(rho13, rho24, bsp, k, method="direct")
    # Agreement to a few percent; the residual is the direct method's
    # diagonal (r₁≈r₂) product-quadrature error, which is larger for sharp
    # inner-shell densities (here 1s×2p) and grows mildly with rank k.
    # Poisson is the accurate reference here — it resolves the cusp per
    # interval — so this bound characterises the *direct* method's error.
    assert fast == pytest.approx(slow, rel=5e-2)


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
    # rate ∝ R_k², so tolerance is ~2× the integral tolerance
    assert fast == pytest.approx(slow, rel=1e-1)
    assert fast > 0.0
