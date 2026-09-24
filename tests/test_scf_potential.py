"""SCF potential factory tests — decoupling check."""

import numpy as np
import pytest

from atomx import Atom
from atomx.numerics.shooting.scf_potential import (
    bare_nuclear_potential,
    static_hartree_potential,
    tabulated_potential,
)


def test_bare_nuclear_returns_minus_z_over_r():
    V = bare_nuclear_potential(Z=10)
    r = np.array([0.1, 1.0, 10.0])
    expected = -10.0 / r
    np.testing.assert_allclose(V(r), expected, rtol=1e-12)


def test_bare_nuclear_handles_zero_r():
    """r = 0 must not crash; the integrator handles the singularity."""
    V = bare_nuclear_potential(Z=1)
    out = V(np.array([0.0, 1.0]))
    assert np.isfinite(out[0]) or abs(out[0]) > 1e20  # huge but finite


def test_tabulated_potential_interpolates():
    r = np.array([0.0, 1.0, 2.0, 3.0])
    V = np.array([-10.0, -5.0, -1.0, 0.0])
    f = tabulated_potential(r, V)
    # Inside grid: linear interpolation
    np.testing.assert_allclose(f(np.array([0.5, 1.5, 2.5])),
                                np.array([-7.5, -3.0, -0.5]), rtol=1e-12)
    # Outside grid: fill_outer
    assert f(np.array([10.0]))[0] == 0.0  # default fill_outer=0


def test_tabulated_potential_validates():
    with pytest.raises(ValueError):
        tabulated_potential(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        tabulated_potential(np.array([2.0, 1.0]), np.array([1.0, 2.0]))


def test_static_hartree_neutralises_at_large_r():
    """The native quadrature preserves the neutral helium tail."""
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    V_hf = static_hartree_potential(atom._result)
    r = np.array([20.0, 30.0, 50.0])
    rv = V_hf(r) * r
    np.testing.assert_allclose(rv, 0.0, rtol=0.0, atol=5.0e-5)


def test_static_hartree_excluding_one_electron_gives_minus_one_over_r():
    """Removing one electron from helium gives residual charge +1."""
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    V_hf = static_hartree_potential(atom._result, exclude_label="1s1/2")
    r = np.array([20.0, 30.0, 50.0])
    np.testing.assert_allclose(V_hf(r) * r, -1.0, rtol=0.0, atol=2.0e-5)


def test_static_hartree_can_include_local_exchange():
    """Exchange is attractive in the core and does not remove a second electron."""
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    V_h = static_hartree_potential(atom._result, exclude_label="1s1/2")
    V_hx = static_hartree_potential(
        atom._result, exclude_label="1s1/2", include_exchange=True,
    )
    r_inner = np.array([0.2, 0.5, 1.0])
    assert np.all(V_hx(r_inner) < V_h(r_inner))
    r = np.array([100.0, 200.0, 500.0])
    np.testing.assert_allclose(V_hx(r) * r, -1.0, rtol=0.0, atol=1.0e-9)


def test_static_hartree_inner_region_approaches_bare_nuclear():
    """The nuclear field is evaluated separately from the smooth screening."""
    atom = Atom(Z=10, config="1s2 2s2 2p6", engine="bspline",
                corrections="none")
    atom.solve()
    V_hf = static_hartree_potential(atom._result, n_radial=4000)
    # r -> 0 limit: the bare nuclear charge, unscreened.
    r_in = np.array([5.0e-4, 1.0e-3])
    np.testing.assert_allclose(V_hf(r_in) * r_in, -10.0, rtol=0.0, atol=0.05)
    # Outer: screening sets in and r·V rises monotonically towards 0.
    r_out = np.array([5.0e-3, 1.0e-2, 2.0e-2, 5.0e-2])
    rv_out = V_hf(r_out) * r_out
    assert np.all(rv_out > -10.0), f"more attractive than −Z/r: {rv_out}"
    assert np.all(np.diff(rv_out) > 0.0), f"not monotone: {rv_out}"


def test_static_hartree_preserves_nuclear_field_below_tabulation():
    atom = Atom(Z=10, config="1s2 2s2 2p6", engine="bspline",
                corrections="none", nuclear="point").solve()
    potential = static_hartree_potential(atom._result)
    radius = np.array([1e-9, 1e-8, 1e-7])
    np.testing.assert_allclose(radius * potential(radius), -10., atol=4e-6)
    # Finite direct screening survives the singular nuclear contribution.
    screening = potential(radius) + 10. / radius
    assert np.all(screening > 0.)
    np.testing.assert_allclose(screening, screening[-1], rtol=1e-5)


def test_static_hartree_for_neutral_argon():
    """The native quadrature preserves the neutral argon tail."""
    atom = Atom(
                Z=18,
                config="1s2 2s2 2p6 3s2 3p6",
                engine="bspline",
                corrections="none",
           )
    atom.solve()
    V_hf = static_hartree_potential(atom._result)
    r = np.array([20.0, 30.0, 50.0])
    np.testing.assert_allclose(V_hf(r) * r, 0.0, rtol=0.0, atol=3.0e-3)


def test_neutral_atom_asymptote_is_exact():
    """Neutral charge is preserved without integrating a resampled density."""
    atom = Atom(Z=18, config="1s2 2s2 2p6 3s2 3p6", engine="bspline",
                corrections="none")
    atom.solve()
    V_hf = static_hartree_potential(atom._result)
    r = np.array([20.0, 30.0, 50.0])
    np.testing.assert_allclose(V_hf(r) * r, 0.0, rtol=0.0, atol=1.0e-4)


def test_tabulated_hydrogenic_density_has_analytic_hartree_field():
    """An independent Poisson integral also checks the shooting-result path."""
    from types import SimpleNamespace

    from atomx.methods.dirac_fock import Subshell
    radius = np.linspace(0., 20., 20001)
    p = 2 * radius * np.exp(-radius)
    result = SimpleNamespace(
        Z=2, _bsp_data=None, _grid=radius,
        _difr=np.full_like(radius, radius[1]),
        _pa=p[None, :], _qa=np.zeros((1, radius.size)),
        subshells=[Subshell(1, -1, 2)])
    sample = np.array([0.05, 0.2, 1., 3., 10.])
    one_electron = -np.expm1(-2 * sample) / sample - np.exp(-2 * sample)
    for hole, number in ((None, 2), ("1s1/2", 1)):
        potential = static_hartree_potential(result, exclude_label=hole)
        np.testing.assert_allclose(potential(sample) + 2 / sample,
                                   number * one_electron, atol=2e-5, rtol=0.)
        assert potential(np.array([100.]))[0] == pytest.approx((number - 2) / 100.)
