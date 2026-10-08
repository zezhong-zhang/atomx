"""Physical complex-density, dilation, c-metric and general-N CI gates."""

import numpy as np
import pytest

from atomx.methods.csf.complex_ci import ComplexScaledCIModel
from atomx.numerics.bspline.basis import setup_gauss_points
from atomx.numerics.bspline.complex_dirac import ComplexScaledDiracWorkspace
from atomx.numerics.bspline.knots import make_exponential_knots
from atomx.numerics.bspline.slater import slater_rk_direct, slater_rk_poisson
from atomx.numerics.yk import compute_yk
from atomx.physics.csf.complex_radial import ComplexCIRadialSnapshot


@pytest.mark.parametrize('theta', [0, .2, .4])
def test_complex_dilated_hydrogen_density_has_analytic_coulomb(theta):
    ns, ks = 85, 9
    gr, w = setup_gauss_points(make_exponential_knots(ns, ks, 1e-4, 25), ns, ks, 17)
    # Normalized c-product exponential density; no conjugation.
    lam = 4*np.exp(1j*theta)
    rho = lam**3/2*gr**2*np.exp(-lam*gr)
    expected = 5*lam/16
    fast = slater_rk_poisson(rho, rho, gr, w, *gr.shape, 0)
    direct = slater_rk_direct(rho, rho, gr, w, 0)
    assert abs(fast-expected) < 1e-9
    assert abs(direct-expected) < 1e-9
    assert abs(np.exp(-1j*theta)*fast-1.25) < 1e-9


def test_complex_origin_polynomials_and_integer_density_preserve_quadrature():
    x, w = np.polynomial.legendre.leggauss(9)
    r, w = ((x+1)/2)[None, :], (w/2)[None, :]
    a, b = 1+2j, -.3+.7j
    # Independent exact two-triangle monomial integral for p=2,q=3,k=0.
    exact = a*b/6*(1/3+1/4)
    for f in (slater_rk_poisson, slater_rk_direct):
        args = (1, 9, 0) if f is slater_rk_poisson else (0,)
        assert abs(f(a*r**2, b*r**3, r, w, *args)-exact) < 1e-13
    got = compute_yk(np.ones(r.shape, dtype=int), r, w, 1, 9, 0)
    ref = compute_yk(np.ones(r.shape), r, w, 1, 9, 0)
    np.testing.assert_array_equal(got, ref)


def workspaces():
    t = make_exponential_knots(55, 9, 1e-4, 20)
    return {k: ComplexScaledDiracWorkspace(t, 9, k, 6, n_gauss=17) for k in (-1, 1, -2)}


def test_selected_physical_spinors_have_c_gram_and_actual_reciprocal_integrals():
    ws = workspaces()
    s = ComplexCIRadialSnapshot(ws, {-1:3, 1:2, -2:2}, .3)
    for k, _w in ws.items():
        keys = [key for key in s.keys if key[1] == k]
        p, q = (np.array([s.orbitals[key][j] for key in keys]) for j in (0, 1))
        gram = (p*s.grw.ravel())@p.T + (q*s.grw.ravel())@q.T
        np.testing.assert_allclose(gram, np.eye(len(keys)), atol=1e-8)
    a, b, c, d = (1,-1), (2,1), (2,-1), (2,-2)
    for rank in (0, 1, 2):
        fast = s.coulomb(rank, a, c, b, d)
        direct = s.coulomb(rank, a, c, b, d, method='direct')
        assert abs(fast-direct) < 1e-8
        assert s.coulomb(rank, b, d, a, c) == fast
        assert s.coulomb(rank, c, a, d, b) == fast
        assert abs(fast.imag) > 1e-10
    o = ws[-1].solve(.3)
    with pytest.raises(ValueError, match='different'):
        ws[1].sample_spinors(o, [0])
    tiny_cache = ComplexCIRadialSnapshot(ws, {-1:3, 1:2, -2:2}, .3, cache_bytes=0)
    assert tiny_cache.coulomb(0, a, a, c, c) == s.coulomb(0, a, a, c, c)
    assert not tiny_cache._cache


def test_N1_N2_and_N3_actual_CI_not_synthetic_provider():
    knots = make_exponential_knots(55, 9, 1e-4, 20)
    one = ComplexScaledCIModel(knots, 9, 6, 1, {-1:3}, n_gauss=17).solve(.2, 1, 1)
    np.testing.assert_allclose(one.energies, list(one.radial.energies.values()), atol=1e-12)
    two = ComplexScaledCIModel(knots, 9, 6, 2, {-1:1}, n_gauss=17).solve(.2, 0, 1)
    k=(1,-1)
    exact = 2*two.radial.energies[k]+two.radial.coulomb(0,k,k,k,k)
    assert abs(two.energies[0]-exact) < 1e-12
    three = ComplexScaledCIModel(knots, 9, 6, 3, {-1:2}, n_gauss=17).solve(.2, 1, 1)
    assert len(three.energies) == 2
    assert three.residuals.max() < 1e-12
    np.testing.assert_allclose(three.coefficients.T@three.coefficients, np.eye(2), atol=1e-12)
    # Independent Slater-Condon diagonal of 1s^2 2s: 2h1+h2+F11+2F12-G12.
    i = next(i for i,c in enumerate(three.csfs) if {sh.n:sh.occupation for sh in c.subshells} == {1:2,2:1})
    R = three.radial
    b=(2,-1)
    expected = (2*R.energies[k]+R.energies[b]+R.coulomb(0,k,k,k,k)
                +2*R.coulomb(0,k,k,b,b)-R.coulomb(0,k,b,b,k))
    reconstructed = (three.coefficients*three.energies)@three.coefficients.T
    assert abs(reconstructed[i,i]-expected) < 1e-11


def test_cost_and_snapshot_identity_gates_precede_physical_calculation():
    t = make_exponential_knots(35, 9, 1e-4, 20)
    with pytest.raises(ValueError, match='cap'):
        ComplexScaledCIModel(t, 9, 6, 3, {-1:20}, magnetic_state_cap=10)
    ws = workspaces()
    other = ComplexScaledDiracWorkspace(t, 9, 1, 6, n_gauss=17)
    with pytest.raises(ValueError, match='common'):
        ComplexCIRadialSnapshot({-1:ws[-1],1:other}, {-1:2,1:2}, .2)
    with pytest.raises(ValueError, match='exceeds'):
        ComplexCIRadialSnapshot({-1:ws[-1]}, {-1:1000}, .2)


@pytest.mark.parametrize('ng', [17, 25, 33])
def test_orthogonal_partial_rule_at_high_order_against_legendre_antiderivatives(ng):
    from numpy.polynomial.legendre import Legendre

    from atomx.numerics.yk import yk_partial_matrices
    x, weights = np.polynomial.legendre.leggauss(ng)
    mo, mi = yk_partial_matrices(ng)
    np.testing.assert_allclose(mo+mi, np.ones_like(mo), rtol=0, atol=np.finfo(float).eps)
    for degree in (0, 1, 6, ng-1):
        poly = Legendre.basis(degree)
        anti = poly.integ()
        values = weights*poly(x)
        np.testing.assert_allclose(mo@values, anti(x)-anti(-1), atol=2e-14)
        np.testing.assert_allclose(mi@values, anti(1)-anti(x), atol=2e-14)


def test_grid_power_retains_default_bits_and_valid_knots():
    default = make_exponential_knots(45, 9, 1e-4, 15)
    explicit = make_exponential_knots(45, 9, 1e-4, 15, power=1)
    np.testing.assert_array_equal(default, explicit)
    other = make_exponential_knots(45, 9, 1e-4, 15, power=.4)
    assert np.all(np.diff(other[8:-8]) > 0)
    assert np.all(other[:9]==0) and np.all(other[-9:]==15)
    with pytest.raises(ValueError, match='power'):
        make_exponential_knots(45, 9, 1e-4, 15, power=0)
