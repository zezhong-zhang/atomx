"""Coulomb quadrature: independent dense kernels, dynamic range and endpoints."""

import numpy as np
import pytest

from atomx.numerics.quadrature import (
    GridResourceError,
    grid_slater_rk,
    make_fine_grid,
    yk_on_grid,
    yk_pair_dot_batch_on_grid,
)
from atomx.numerics.radial_metric import RadialMetric


def weights(r):
    # Independent accumulation of each interval's two half weights.
    out = np.zeros_like(r)
    out[:-1] += 0.5 * np.diff(r)
    out[1:] += 0.5 * np.diff(r)
    return out


@pytest.mark.parametrize("rank", [0, 1, 12, 32, 64, 128, 256])
@pytest.mark.parametrize("immutable", [False, True])
def test_scaled_action_and_slater_agree_with_dense_green_matrix(rank, immutable):
    r = np.geomspace(1e-9, 100.0, 193)
    r.setflags(write=not immutable)
    rho = np.sin(3 * np.log(r)) * np.exp(-r / 12)
    other = np.cos(np.log(r)) * np.exp(-r / 9)
    lo, hi = np.minimum(r[:, None], r), np.maximum(r[:, None], r)
    green = (lo / hi) ** rank / hi
    reference = green @ (weights(r) * rho)
    actual = yk_on_grid(rho, r, rank)
    np.testing.assert_allclose(actual, reference, atol=2e-14, rtol=5e-13)
    ones, zeros = np.ones_like(r), np.zeros_like(r)
    value = grid_slater_rk(other, zeros, rho, zeros, ones, zeros, ones, zeros, r, rank)
    expected = (other * weights(r)) @ reference
    assert value == pytest.approx(expected, rel=5e-13, abs=2e-14)
    # Swap both transition densities: self-adjointness of the Green action.
    reverse = grid_slater_rk(
        rho, zeros, other, zeros, ones, zeros, ones, zeros, r, rank
    )
    assert reverse == pytest.approx(expected, rel=5e-13, abs=2e-14)
    pa, qa, pb, qb = rho[None, :], zeros[None, :], ones[None, :], zeros[None, :]
    batched = yk_pair_dot_batch_on_grid(
        pa,
        qa,
        pb,
        qb,
        np.array([0]),
        np.array([0]),
        np.array([rank]),
        np.array([True]),
        (other * weights(r))[None, :],
        np.array([0, 1]),
        np.array([0]),
        r,
    )
    assert batched[0] == pytest.approx(expected, rel=5e-13, abs=2e-14)


def test_scalar_integral_keeps_half_interval_after_last_nonzero_sample():
    r = np.array([0.1, 0.4, 1.0, 2.0])
    a = np.array([1.0, 2.0, 0.0, 0.0])
    b = np.ones(4)
    zero = np.zeros(4)
    green = 1 / np.maximum(r[:, None], r)
    expected = (a * weights(r)) @ green @ weights(r)
    assert grid_slater_rk(a, zero, b, zero, b, zero, b, zero, r, 0) == pytest.approx(
        expected, rel=3e-14
    )


@pytest.mark.parametrize("rank", [0, 1, 8])
def test_compact_consumer_retains_exterior_continuum_source(rank):
    r = np.geomspace(0.01, 40.0, 121)
    source = np.cos(r)
    consumer = np.where(r < 2.0, r * np.exp(-r), 0.0)
    cut = np.flatnonzero(consumer)[-1] + 1
    green = (np.minimum(r[:, None], r) / np.maximum(r[:, None], r)) ** rank
    green /= np.maximum(r[:, None], r)
    expected = (consumer * weights(r)) @ green @ (weights(r) * source)
    zeros, ones = np.zeros((1, len(r))), np.ones((1, len(r)))
    actual = yk_pair_dot_batch_on_grid(
        source[None, :],
        zeros,
        ones,
        zeros,
        np.array([0]),
        np.array([0]),
        np.array([rank]),
        np.array([True]),
        (consumer * weights(r))[None, : cut + 1],
        np.array([0, 1]),
        np.array([0]),
        r,
        i_cut=cut,
    )
    assert actual[0] == pytest.approx(expected, rel=5e-13, abs=2e-14)


def test_pointwise_tiny_tail_cannot_be_discarded():
    r = np.array([1e-7, 1.0, 1e20])
    a = np.array([1.0, 1e-20, 1e-20])
    b = np.ones(3)
    zero = np.zeros(3)
    green = 1 / np.maximum(r[:, None], r)
    expected = (a * weights(r)) @ green @ weights(r)
    assert grid_slater_rk(a, zero, b, zero, b, zero, b, zero, r, 0) == pytest.approx(
        expected, rel=3e-14
    )


def test_writable_grid_mutation_cannot_reuse_stale_geometry():
    r = np.geomspace(0.1, 10.0, 30)
    rho = np.exp(-r)
    yk_on_grid(rho, r, 3)
    r *= 2
    green = (np.minimum(r[:, None], r) / np.maximum(r[:, None], r)) ** 3 / np.maximum(
        r[:, None], r
    )
    np.testing.assert_allclose(
        yk_on_grid(rho, r, 3), green @ (weights(r) * rho), rtol=5e-14
    )


def test_trapezoidal_metric_integrates_constant_and_linear_functions():
    r = np.array([0.03, 0.04, 0.2, 1.0, 1.4, 8.0])
    metric = RadialMetric.trapezoid(r)
    assert metric.integrate(np.ones_like(r)) == pytest.approx(r[-1] - r[0], rel=1e-15)
    assert metric.integrate(r) == pytest.approx(
        0.5 * (r[-1] ** 2 - r[0] ** 2), rel=1e-15
    )
    np.testing.assert_array_equal(metric.weights, weights(r))


def test_radial_refinement_is_nested_and_enforces_resource_cap():
    coarse = make_fine_grid(3.0, 2.0, z_inner=2.0)
    fine = make_fine_grid(3.0, 2.0, z_inner=2.0, refinement=4)
    np.testing.assert_array_equal(fine[::4], coarse)
    assert len(fine) == 4 * (len(coarse) - 1) + 1
    assert not fine.flags.writeable
    with pytest.raises(GridResourceError, match="refinement exceeded"):
        make_fine_grid(3.0, 2.0, z_inner=2.0, refinement=4, max_points=len(fine) - 1)


@pytest.mark.parametrize("refinement", [0, -1, 1.5, True, float("nan")])
def test_invalid_radial_refinement_raises(refinement):
    with pytest.raises(ValueError, match="refinement"):
        make_fine_grid(3.0, 2.0, refinement=refinement)
