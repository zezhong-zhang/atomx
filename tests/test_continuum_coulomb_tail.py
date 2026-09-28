"""Guards for the wide-step Coulomb-tail continuation (e > 0).

The walker (``doutw_coulomb_tail``) replaces the spline-integrated tail
by exact point-charge propagation once the sampled potential is close
enough to the ``-Z_res/r`` asymptote — "close enough" being the phase
budget ``_TAIL_PHASE_BUDGET`` (plus a pointwise guard and a minimum tail
length).  Physics pinned here:

* on potentials whose tail deviates from the asymptote, the walker wave
  and the historical per-interval wave agree to far inside the FAC-parity
  gate, and the extracted amplitude moves by ~1e-8 (measured on the real
  Fe/Xe SCF bench);
* the budget is per-wave: slower waves (smaller k) integrate the same
  deviation more strongly, so their tail start moves out;
* potentials with no Coulomb-like tail keep the reference path exactly
  (bit-identical output);
* deviation-free potentials (hydrogenic, free particle) engage the
  walker from the first grid cells and recover the exact Coulomb/free
  solution.
"""

import math

import numpy as np

import atomx.numerics.shooting.dirac_continuum as dc
from atomx.numerics.shooting.dirac_continuum import (
    solve_dirac_continuum_shooting,
)


def _V_screened(Z, z_res, r_core):
    def V(r):
        r = np.asarray(r, dtype=float)
        return (-z_res / np.maximum(r, 1e-30)
                - (Z - z_res) * np.exp(-r / r_core) / np.maximum(r, 1e-30))
    return V


def _solve(walker, E, kappa, V, z_res, r_max=200.0, n_grid=8000):
    dc._TAIL_WALKER = walker
    try:
        return solve_dirac_continuum_shooting(
            E, kappa, V, r_max=r_max, n_grid=n_grid,
            Z_residual=z_res, match_phase=False)
    finally:
        dc._TAIL_WALKER = True


def _window_a(P):
    n_match = max(20, len(P) // 5)
    return math.sqrt(2.0 * float(np.mean(P[-n_match:] ** 2)))


def test_tail_walker_matches_reference_path():
    """Gently screened core: continuation vs historical path, same grid."""
    V = _V_screened(26.0, 1.0, 2.0)
    for E in (1.0, 8.0, 0.05):
        r_a, P_a, Q_a, _ = _solve(True, E, -1, V, 1.0)
        r_b, P_b, Q_b, _ = _solve(False, E, -1, V, 1.0)
        assert np.array_equal(r_a, r_b)
        scale = np.max(np.abs(P_b[len(r_b) // 2:]))
        dP = np.max(np.abs(P_a[len(r_a) // 2:] - P_b[len(r_b) // 2:])) / scale
        assert dP < 1.0e-3, f"E={E}: window wave drifted by {dP}"
        assert abs(_window_a(P_a) / _window_a(P_b) - 1.0) < 1.0e-3


def test_tail_start_budget_tightens_for_slow_waves():
    """The action integral criterion scales with k: slow waves start later.

    A 1/r^3 correction gives a deviation ~1/r^2: the pointwise mask then
    fixes the fast-wave start, while the integrated action (budget ~ k)
    pushes the slow-wave start further out.
    """
    def V(r):
        r = np.asarray(r, dtype=float)
        return -1.0 / np.maximum(r, 1e-30) - 0.25 / np.maximum(r, 1e-30) ** 3

    prep = dc._prepare_potential(V, 200.0, 8000, 1.0)
    i_fast = dc._coulomb_tail_start(prep, math.sqrt(2.0 * 4.0))
    i_slow = dc._coulomb_tail_start(prep, math.sqrt(2.0 * 0.005))
    assert 0 < i_fast < i_slow
    r = prep[0]
    # The fast wave is mask-bound: dev(r) = |rV + z_res| = 0.25/r^2 - eps0
    # with eps0 = 0.25/r_max^2, because z_res (taken at the box edge)
    # absorbs a sliver of the 1/r^3 term.  The boundary therefore sits at
    # ~(0.25/(1e-4 + eps0))^0.5 — marginally inside the naive 50.0.
    r_mask = math.sqrt(0.25 / (1.0e-4 + 0.25 / 200.0 ** 2))
    assert abs(r[i_fast] - r_mask) < 1.0


def test_no_coulomb_tail_falls_back_bit_identical():
    """A potential that never approaches -Z_res/r keeps the reference path."""
    def V_long_tail(r):
        r = np.asarray(r, dtype=float)
        return -25.0 / np.maximum(r, 1e-30) ** 0.8

    prep = dc._prepare_potential(V_long_tail, 50.0, 4000, 0.0)
    # dev = |r*V| grows without bound; the mask must exclude every point.
    assert dc._coulomb_tail_start(prep, 1.0) == -1
    r_a, P_a, _, _ = _solve(True, 1.0, -1, V_long_tail, 0.0, r_max=50.0, n_grid=4000)
    r_b, P_b, _, _ = _solve(False, 1.0, -1, V_long_tail, 0.0, r_max=50.0, n_grid=4000)
    assert np.array_equal(r_a, r_b)
    assert np.array_equal(P_a, P_b)


def test_constant_tail_entry_used_past_last_knot():
    """vint freezes near-asymptotic tails; grid points past the last fitted
    knot must use the constant-asymptote entry, not extrapolate the last
    fitted cubic (the old mapping drifted ~5e-4 from -Z_res by mid-box)."""
    def V(r):
        r = np.asarray(r, dtype=float)
        return (-1.0 - 25.0 * np.exp(-r / 2.0)) / np.maximum(r, 1e-30)

    prep = dc._prepare_potential(V, 200.0, 8000, 1.0)
    r, n_out, va, vb, vc, vd, ind, rv, z_res = prep[:9]
    j = int(np.searchsorted(r, 150.0))
    i = ind[j]
    # exact constant asymptote: power form collapses to va == -z_res
    assert vb[i] == 0.0 and vc[i] == 0.0 and vd[i] == 0.0
    implied = va[i] + r[j] * (vb[i] + r[j] * (vc[i] + r[j] * vd[i]))
    assert implied == -z_res


def test_deviating_intervals_reinterpolated_to_track_callable():
    """Where the tail never freezes (production screened potentials), the
    per-interval cubic must still track V: deviations beyond the fix
    tolerance are re-interpolated, verified at fresh Gauss nodes."""
    def V(r):
        r = np.asarray(r, dtype=float)
        return (-1.0 - 0.05 * np.sin(2.0 * r) / np.maximum(r, 1.0)
                - 0.5 / np.maximum(r, 1.0) ** 2) / np.maximum(r, 1e-30)

    from atomx.numerics.shooting.spline import vint

    prep = dc._prepare_potential(V, 200.0, 600, 1.0)
    r, n_out, va, vb, vc, vd, ind, rv, z_res = prep[:9]
    m = n_out - 1
    h = np.diff(r)
    g = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
    nodes = r[:m, None] + h[:, None] * g[None, :]
    rn = np.where(nodes > 0.0, nodes, 1.0e-30)
    rvn = rn * V(rn)
    a = va[ind[:m]][:, None]
    b = vb[ind[:m]][:, None]
    c = vc[ind[:m]][:, None]
    d = vd[ind[:m]][:, None]
    implied = a + nodes * (b + nodes * (c + nodes * d))
    dev = np.abs(implied - rvn).max()
    # after the fix the implied potential tracks the callable to
    # interpolation accuracy everywhere, far inside the fix tolerance
    assert dev < 1.0e-8 * max(abs(z_res), 1.0)
    # and the fix really engaged: the plain natural spline misses V by
    # far more on this coarse grid / structured tail
    rg, rvg, va0, vb0, vc0, vd0, nvt = vint(r, rv, n_out)
    ind0 = np.clip(np.searchsorted(rg[:nvt], r, side="right") - 1, 0, nvt - 2)
    a0 = va0[ind0[:m]][:, None]
    b0 = vb0[ind0[:m]][:, None]
    c0 = vc0[ind0[:m]][:, None]
    d0 = vd0[ind0[:m]][:, None]
    old = a0 + nodes * (b0 + nodes * (c0 + nodes * d0))
    assert np.abs(old - rvn).max() > 10.0 * dev


def test_deviation_free_potentials_engage_from_origin():
    """Hydrogenic and free-particle tails are exact-Coulomb everywhere."""
    def V_hydrogen(r):
        return -1.0 / np.where(r > 0, r, 1e-30)

    prep = dc._prepare_potential(V_hydrogen, 50.0, 4000, 1.0)
    assert dc._coulomb_tail_start(prep, 1.0) == 2
    r, P, Q, _ = _solve(True, 1.0, -1, V_hydrogen, 1.0, r_max=50.0, n_grid=4000)
    k = math.sqrt(2.0)
    envelope = math.sqrt(2.0 / (math.pi * k))
    outer = P[r > 40.0]
    assert abs(math.sqrt(2.0 * float(np.mean(outer ** 2))) / envelope - 1.0) < 5e-2

    def V_zero(r):
        return np.zeros_like(r)

    prep = dc._prepare_potential(V_zero, 50.0, 4000, 0.0)
    assert dc._coulomb_tail_start(prep, 1.0) == 2
    r, P, Q, _ = _solve(True, 1.0, -1, V_zero, 0.0, r_max=50.0, n_grid=4000)
    assert np.all(np.isfinite(P)) and np.all(np.isfinite(Q))
