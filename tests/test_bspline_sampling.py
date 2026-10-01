"""Independent polynomial checks of native, kinetic-balanced sampling."""
import numpy as np
import pytest

from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.evaluate import eval_P, eval_Q


@pytest.fixture
def basis():
    # A single clamped interval is a Bernstein basis. Kinetic balance gives
    # quadratic P and cubic Q, whose analytic values are independent of the
    # B-spline evaluator. P's padded coefficient must not become a basis term.
    return prepare_bspline_data(np.array([0.]*4+[2.]*4), 4, 4,
                                kinetic_balance=True)


def test_direct_sampling_matches_bernstein_polynomials_and_native_grid(basis):
    r = np.array([0., 1e-12, .03, .7, 1.2, 2.])
    t = r/2
    p = np.array([0., 1., 0., 12345.])
    q = np.array([0., 0., 1., 0.])
    np.testing.assert_allclose(eval_P(p, basis, r), 2*t*(1-t), atol=2e-16)
    np.testing.assert_allclose(eval_Q(q, basis, r), 3*t*t*(1-t), atol=2e-16)
    for evaluate, coeffs in ((eval_P, p), (eval_Q, q)):
        np.testing.assert_allclose(evaluate(coeffs, basis, basis['gr']),
                                   evaluate(coeffs, basis), atol=3e-16)


@pytest.mark.parametrize('r', [[-1e-12], [2.+1e-12], [np.nan], [np.inf]])
@pytest.mark.parametrize('evaluate', [eval_P, eval_Q])
def test_sampling_does_not_silently_clamp_or_extrapolate(basis, r, evaluate):
    with pytest.raises(ValueError, match='inside the B-spline box'):
        evaluate(np.ones(4), basis, r)
