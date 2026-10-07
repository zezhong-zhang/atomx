"""Independent moments and cache ownership for the shared numerical rule."""

import numpy as np
import pytest

from atomx.numerics.quadrature import gauss_legendre_rule


@pytest.mark.parametrize("order", [1, 2, 7, 32, 128])
def test_polynomial_exactness_positive_weights_and_symmetry(order):
    x, w = gauss_legendre_rule(order)
    assert np.all(w > 0)
    np.testing.assert_allclose(x, -x[::-1], atol=2e-15, rtol=0)
    for degree in range(2*order):
        expected = 0.0 if degree % 2 else 2.0/(degree + 1)
        assert w @ x**degree == pytest.approx(expected, abs=6e-14, rel=0)
    assert gauss_legendre_rule(np.int64(order))[0] is x
    with pytest.raises(ValueError, match="read-only"):
        x[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        w[0] = 0.0


@pytest.mark.parametrize("order", [0, -1, True, np.bool_(True), 2.5, float("nan")])
def test_rule_rejects_invalid_order(order):
    with pytest.raises(ValueError, match="positive integer"):
        gauss_legendre_rule(order)
