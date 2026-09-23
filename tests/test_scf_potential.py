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
    """Exact asymptote for a neutral atom: r·V_HF(r) → −Z + N = 0.

    V_HF(r) = −Z/r + ∫ρ(r′)/max(r,r′) dr′, so beyond the last electron
    r·V_HF(r) → −Z + N.  For neutral He that is 0, and the only thing that
    can make it non-zero is the enclosed charge, i.e. the density
    normalisation.

    Measured: r·V_HF(20) = r·V_HF(50) = +2.456e-5, i.e. the tabulated
    density carries 2.2e-5 electrons too many (``static_hartree_potential``
    linearly re-grids the B-spline density onto a uniform ``n_radial=4000``
    mesh before integrating it with the trapezoid rule; that regridding, not
    the SCF, is the residual).  Gate 5e-5 = 2x the measurement, and 2e4x
    tighter than the ``|V_HF(20)| < 0.05`` band it replaces.
    """
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    V_hf = static_hartree_potential(atom._result)
    r = np.array([20.0, 30.0, 50.0])
    rv = V_hf(r) * r
    np.testing.assert_allclose(rv, 0.0, rtol=0.0, atol=5.0e-5)


def test_static_hartree_excluding_one_electron_gives_minus_one_over_r():
    """For He (Z=2) with exclude='1s1/2' the residual core holds exactly one
    electron, so the escaping electron sees a 1+ ion and V(r)·r → −(Z−N_core)
    = −1 as r → ∞ (not merely "between −0.2 and −0.05").

    Measured r·V = −0.9999877 flat from r = 20 to r = 50 — 1.2e-5 short of
    the exact −1, again the re-gridding quadrature of
    ``static_hartree_potential`` (it loses 1.2e-5 e of the remaining
    electron).  Gate 2e-5 = 1.7x the measurement.  The old band
    ``−0.2 < V(10) < −0.05`` admitted both −0.1 (this, the correct Hartree
    answer) and −0.2 (the Latter-capped answer of the exchange branch below),
    i.e. it gated nothing.
    """
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    V_hf = static_hartree_potential(atom._result, exclude_label="1s1/2")
    r = np.array([20.0, 30.0, 50.0])
    np.testing.assert_allclose(V_hf(r) * r, -1.0, rtol=0.0, atol=2.0e-5)


def test_static_hartree_can_include_local_exchange():
    """Local exchange is attractive on the core density, and the exchange
    branch applies FAC's Latter cap, whose asymptote is a *different* charge
    from the Hartree branch's.

    Two distinct exact statements, both gated here:

    1. on the core (r <= 1) exchange makes the potential more attractive,
       V_hx(r) < V_h(r);
    2. asymptotically the FAC branch caps r·V at
       ``rvinf = −(1 + Z − N_eff) = −(1 + 2 − 1) = −2``
       (``scf_potential.py`` Latter cap, the neutral-atom convention also
       pinned in ``test_fac_asymptotic.py``), i.e. the escaping electron
       sees a 2+ ion — whereas the pure-Hartree branch above returns
       exactly −1/r.  The two branches genuinely disagree; this is the FAC
       convention, not a bug, and the test pins both numbers so a silent
       change of either is caught.

    Measured: r·V_hx = −1.9999999999999998 … −2.0 (exact to machine
    precision; the code nudges the capped value one ULP toward zero).
    """
    atom = Atom(Z=2, config="1s2", engine="bspline", corrections="none")
    atom.solve()
    V_h = static_hartree_potential(atom._result, exclude_label="1s1/2")
    V_hx = static_hartree_potential(
        atom._result, exclude_label="1s1/2", include_exchange=True,
    )
    r_inner = np.array([0.2, 0.5, 1.0])
    assert np.all(V_hx(r_inner) < V_h(r_inner))
    r = np.array([20.0, 30.0, 50.0])
    np.testing.assert_allclose(V_hx(r) * r, -2.0, rtol=0.0, atol=1.0e-9)


def test_static_hartree_inner_region_approaches_bare_nuclear():
    """V_HF(r)·r → −Z as r → 0: V = −Z/r + V_H(r) with
    V_H(0) = ∫ρ(r′)/r′ dr′ finite, so r·V(r) = −Z + r·V_H(0) + O(r²).

    The limit has to be measured above the first tabulation point, because
    ``static_hartree_potential`` returns the endpoint value below it (see
    ``test_static_hartree_is_clamped_below_the_first_tabulation_point``),
    and above the mesh's resolution of the Ne 1s peak.  Measured with
    ``n_radial=400000`` (Δr = 1.3e-4 a₀):

        r·V(5e-4) = −10.018, r·V(1e-3) = −9.983      → −Z within 0.05
        r·V(5e-3) = −9.846, r·V(1e-2) = −9.690,
        r·V(2e-2) = −9.385, r·V(5e-2) = −8.543      → screened, monotone

    At the default ``n_radial=4000`` the same quantity is −10.587 at
    r = 0.02, i.e. *more* attractive than the bare nucleus — the default
    mesh does not resolve the 1s peak and must not be used for inner-region
    claims.  The old assertion, ``0.5 < V_HF(0.01)/V_bare(0.01) < 1.5``,
    was measured at r = 0.01 inside the clamped region, where the ratio is a
    property of the interpolator rather than of the atom.
    """
    atom = Atom(Z=10, config="1s2 2s2 2p6", engine="bspline",
                corrections="none")
    atom.solve()
    V_hf = static_hartree_potential(atom._result, n_radial=400000)
    # r -> 0 limit: the bare nuclear charge, unscreened.
    r_in = np.array([5.0e-4, 1.0e-3])
    np.testing.assert_allclose(V_hf(r_in) * r_in, -10.0, rtol=0.0, atol=0.05)
    # Outer: screening sets in and r·V rises monotonically towards 0.
    r_out = np.array([5.0e-3, 1.0e-2, 2.0e-2, 5.0e-2])
    rv_out = V_hf(r_out) * r_out
    assert np.all(rv_out > -10.0), f"more attractive than −Z/r: {rv_out}"
    assert np.all(np.diff(rv_out) > 0.0), f"not monotone: {rv_out}"


def test_static_hartree_is_clamped_below_the_first_tabulation_point():
    """Documented interpolation contract, measured: the tabulated potential
    starts at ``r_max/n_radial`` and ``np.interp(..., left=V[0])`` returns
    that endpoint value for every smaller r.  Measured at the default
    ``n_radial=4000`` (first point 1.26e-2 a₀): V_HF(1e-5) == V_HF(1e-2) ==
    −759.926461909543 bit-for-bit.

    This is why no "r → 0" claim can be tested below 1.26e-2 a₀, and why the
    test above refines the mesh instead of shrinking r.
    """
    atom = Atom(Z=10, config="1s2 2s2 2p6", engine="bspline",
                corrections="none")
    atom.solve()
    V_hf = static_hartree_potential(atom._result)
    below = V_hf(np.array([1.0e-5, 1.0e-3, 1.0e-2]))
    assert below[0] == below[1] == below[2]
    # ...and the potential does vary once the tabulation starts.
    assert V_hf(np.array([0.02]))[0] != below[0]


def test_static_hartree_for_neutral_argon():
    """For neutral Ar (Z=18, 18 electrons), r·V_HF(r) → −Z + N = 0.

    Measured r·V_HF(20) = r·V_HF(50) = +2.293e-3, i.e. the tabulated density
    encloses 2.3e-3 electrons too many.  The cause is the same regridding as
    in the He case above — ``static_hartree_potential`` interpolates the
    B-spline density onto a uniform ``n_radial=4000`` mesh (Δr = 1.3e-2 a₀)
    with ``np.interp`` and integrates with the trapezoid rule; Ar's 1s peak
    is ~1e-2 a₀ wide, so the 4000-point mesh cannot carry its norm.  Gate
    3e-3 = 1.3x the measurement, and 17x tighter than the replaced
    ``|V_HF(20)| < 0.05`` band.
    """
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Neutral-atom asymptote is not exact: r*V_HF(20) = +2.293e-3 for Ar "
        "(+2.456e-5 for He, +1.487e-3 for Ne) instead of 0, because "
        "static_hartree_potential re-grids the B-spline density onto a "
        "uniform n_radial=4000 mesh with np.interp + trapezoid and loses "
        "~2.6e-3 e of the 18-e norm; the audit's target atol=1e-4 fails by "
        "23x.  Fix = integrate with the B-spline quadrature weights."
    ),
)
def test_neutral_atom_asymptote_is_exact():
    """Target: r·V_HF(r) → 0 for a neutral atom to 1e-4 (the exact statement
    the He/Ar gates above can only assert at the regridding's accuracy)."""
    atom = Atom(Z=18, config="1s2 2s2 2p6 3s2 3p6", engine="bspline",
                corrections="none")
    atom.solve()
    V_hf = static_hartree_potential(atom._result)
    r = np.array([20.0, 30.0, 50.0])
    np.testing.assert_allclose(V_hf(r) * r, 0.0, rtol=0.0, atol=1.0e-4)
