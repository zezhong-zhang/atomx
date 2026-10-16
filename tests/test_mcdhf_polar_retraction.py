"""The numerical polar successor preserves its map and tiny tangent changes."""

import mpmath as mp
import numpy as np
import pytest

from atomx.numerics.bspline.orbital_optimization import _polar_retraction


def wide_polar(y):
    # Independent high precision spectral oracle, including the near-degenerate
    # eigenproblem rather than the production binomial correction.
    with mp.workdps(65):
        matrix = mp.matrix(y.tolist())
        values, vectors = mp.eigsy(matrix.T * matrix)
        inverse = vectors * mp.diag([1 / mp.sqrt(v) for v in values]) * vectors.T
        return np.array((matrix * inverse).tolist(), dtype=float)


@pytest.mark.parametrize("scale", [1e-3, 1e-8, 1e-13, 0.2])
def test_same_polar_map_against_wide_spectral_oracle(scale):
    rng = np.random.default_rng(81)
    y, _ = np.linalg.qr(rng.normal(size=(24, 4)))
    y = y + scale * rng.normal(size=y.shape)
    result = _polar_retraction(y)
    np.testing.assert_allclose(result, wide_polar(y), rtol=0, atol=9e-16)
    np.testing.assert_allclose(result.T @ result, np.eye(4), rtol=0, atol=2e-15)


def test_tiny_tangent_increment_preserved_without_core_rotation_noise():
    y = np.zeros((24, 5))
    y[:5] = np.eye(5)
    direction = np.zeros_like(y)
    direction[5:, 4] = np.arange(1, 20) * 1e-13
    result = _polar_retraction(y + direction)
    np.testing.assert_array_equal(result[:, :4], y[:, :4])
    np.testing.assert_array_equal(result, wide_polar(y + direction))
    np.testing.assert_array_equal(_polar_retraction(y), y)


def test_rank_deficient_trial_is_rejected():
    with pytest.raises(ValueError, match="rank-deficient"):
        _polar_retraction(np.zeros((24, 4)))
