"""Z-aware continuum-knot generator regression tests."""

import numpy as np
import pytest

from atomx.numerics.bspline.knots import (
    make_continuum_knots,
    make_physics_knots,
)


def test_basic_construction():
    """Returns a valid (t, ns) tuple."""
    t, ns = make_continuum_knots(ks=9, Z=10, k_max_au=10.0)
    assert len(t) == ns + 9
    assert ns > 9
    assert t[0] == 0.0
    assert t[-1] >= 50.0


def test_inner_spacing_resolves_lambda():
    """Inner-region Δr ≤ λ_min / N_pts (where r < 5/Z)."""
    k_max = 30.0
    n_pts = 4
    Z = 26
    t, _ = make_continuum_knots(ks=9, Z=Z, k_max_au=k_max,
                                  points_per_wavelength=n_pts)
    lambda_min = 2.0 * np.pi / k_max
    # Inner region cutoff matches the generator (5/Z, capped at 0.5).
    r_inner = min(5.0 / Z, 0.5)
    # Pick knots that fall inside [0, r_inner] (strictly tight mesh).
    inner_mask = (t > 0) & (t <= r_inner)
    inner_knots = t[inner_mask]
    # Asserted, not guarded: the old ``if len(inner_knots) >= 3`` silently
    # voided the whole test whenever the generator produced fewer inner
    # knots.  Measured here: 20 inner knots.
    assert len(inner_knots) >= 3, f"only {len(inner_knots)} inner knots"
    spacings = np.diff(inner_knots)
    # Allow the last spacing to be the transition step into the
    # exponential region, so check the median rather than max.
    # Measured median spacing 9.615e-3 <= lambda_min/4 = 5.236e-2.
    median_spacing = float(np.median(spacings))
    assert median_spacing <= lambda_min / n_pts * 1.5


def test_low_k_max_close_to_physics_grid():
    """At k_max ≈ 2 a.u. the continuum grid has similar size to the
    standard physics grid."""
    t_phys, ns_phys = make_physics_knots(ks=9, Z=26, r_max=50.0)
    t_cont, ns_cont = make_continuum_knots(ks=9, Z=26, k_max_au=2.0,
                                             r_max=50.0)
    assert abs(ns_cont - ns_phys) < 5


def test_high_k_max_more_dense_in_inner_region():
    """High k_max → strictly more inner-region knots than the physics default.

    Measured (Z=10, ks=9, r_max=50): physics grid ns = 78 with 14 knots at
    r < 5/Z = 0.5; k_max = 50 a.u. gives ns = 89 with 19 inner knots.
    The replaced assertion was ``ns_cont >= ns_phys * 0.7`` — it passed for a
    30 % *coarsening*, i.e. it gated the opposite of its own name.  Both
    counts are now asserted strictly (the audit's alternative ">= 3x the
    physics count" does not hold and would have failed: 19/14 = 1.36x).
    """
    t_phys, ns_phys = make_physics_knots(ks=9, Z=10, r_max=50.0)
    t_cont, ns_cont = make_continuum_knots(ks=9, Z=10, k_max_au=50.0,
                                             r_max=50.0)
    r_inner = 5.0 / 10
    inner_phys = int(np.sum((t_phys > 0) & (t_phys <= r_inner)))
    inner_cont = int(np.sum((t_cont > 0) & (t_cont <= r_inner)))
    assert ns_cont > ns_phys, f"ns_cont={ns_cont} ns_phys={ns_phys}"
    assert inner_cont > inner_phys, (
        f"inner knots: continuum={inner_cont} physics={inner_phys}")


def test_validates_inputs():
    with pytest.raises(ValueError):
        make_continuum_knots(ks=9, Z=10, k_max_au=0.0)
    with pytest.raises(ValueError):
        make_continuum_knots(ks=9, Z=0, k_max_au=10.0)
    with pytest.raises(ValueError):
        make_continuum_knots(ks=9, Z=10, k_max_au=10.0, r_max=0.0)


def test_monotonic_knots():
    """All knots are non-decreasing."""
    t, _ = make_continuum_knots(ks=9, Z=26, k_max_au=30.0)
    assert np.all(np.diff(t) >= -1e-12)
