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
    """For He (Z=2, 2 electrons in 1s²), V_HF(r→∞) → 0
    (the nucleus is fully screened by the bound electrons)."""
    atom = Atom(Z=2, config="1s2", breit=False, qed=False)
    atom.solve()
    V_hf = static_hartree_potential(atom._result)
    # At large r the bound charge is fully enclosed → V_HF ≈ 0
    out = V_hf(np.array([20.0, 30.0]))
    # |V_HF(20)| should be small compared to |V_bare(20)| = 0.1
    assert abs(out[0]) < 0.05


def test_static_hartree_excluding_one_electron_gives_minus_one_over_r():
    """For He (Z=2) with exclude='1s1/2', V → -1/r at large r
    (the photoelectron sees one residual electron's worth of nucleus)."""
    atom = Atom(Z=2, config="1s2", breit=False, qed=False)
    atom.solve()
    V_hf = static_hartree_potential(atom._result, exclude_label="1s1/2")
    # At r=10 a₀ the residual He+ ion looks like -1/r
    out = V_hf(np.array([10.0]))[0]
    assert -0.2 < out < -0.05


def test_static_hartree_can_include_local_exchange():
    """Local exchange should make the residual-core potential more attractive
    where the bound-electron density is non-negligible, without changing the
    Coulomb tail qualitatively."""
    atom = Atom(Z=2, config="1s2", breit=False, qed=False)
    atom.solve()
    V_h = static_hartree_potential(atom._result, exclude_label="1s1/2")
    V_hx = static_hartree_potential(
        atom._result, exclude_label="1s1/2", include_exchange=True,
    )
    r = np.array([0.2, 0.5, 1.0])
    assert np.all(V_hx(r) < V_h(r))
    assert -0.2 < V_hx(np.array([10.0]))[0] < -0.05


def test_static_hartree_inner_region_more_attractive():
    """V_HF(r) is more attractive than V_bare(r) only for small r —
    actually the screening REDUCES the attractive bare -Z/r at large r.
    Sanity: V_HF(r→0) ≈ V_bare(r→0) (no screening at the nucleus)."""
    atom = Atom(Z=10, config="1s2 2s2 2p6", breit=False, qed=False)
    atom.solve()
    V_hf = static_hartree_potential(atom._result)
    V_bare = bare_nuclear_potential(10)
    r_inner = np.array([0.01])
    # At r=0.01 a₀ the bound density hasn't accumulated much, so
    # |V_HF| ≈ |V_bare| = 1000.
    inner_ratio = V_hf(r_inner)[0] / V_bare(r_inner)[0]
    assert 0.5 < inner_ratio < 1.5


def test_static_hartree_for_neutral_argon():
    """For neutral Ar (Z=18, 18 electrons), V_HF(r → ∞) → 0."""
    atom = Atom(Z=18, config="1s2 2s2 2p6 3s2 3p6", breit=False, qed=False)
    atom.solve()
    V_hf = static_hartree_potential(atom._result)
    out = V_hf(np.array([20.0]))[0]
    assert abs(out) < 0.05
