"""Full-interval quadrature checked against analytic integrals."""
import numpy as np
import pytest

from atomx.numerics.quadrature import sinh_squared_quadrature


@pytest.mark.parametrize('upper,scale', [(1e-10, 3.), (.5, 1.), (1e6, 2.),
                                       (1e-5, 1e-9), (1e5, 1e8)])
def test_mapped_rule_integrates_endpoints_and_physical_scales(upper, scale):
    e, w = sinh_squared_quadrature(48, upper, scale)
    assert np.all(np.diff(e) > 0) and 0 < e[0] < e[-1] < upper
    assert np.all(w > 0)
    # Integrable endpoint singularity and regular half-integer moments:
    # the square-root change of variables must include its full Jacobian.
    for power in [-.5, 0., .5, 1., 2.]:
        exact = upper**(power+1)/(power+1)
        assert np.dot(w, e**power) == pytest.approx(exact, rel=2e-12)
    # An independent nonpolynomial integral resolving the supplied scale.
    assert np.dot(w, 1/(scale+e)) == pytest.approx(np.log1p(upper/scale), rel=2e-12)


@pytest.mark.parametrize('n,upper,scale', [(0, 1., 1.), (3.5, 1., 1.),
    (True, 1., 1.), (np.inf, 1., 1.), (4, 0., 1.), (4, -1., 1.),
    (4, 1., 0.), (4, np.inf, 1.), (4, 1., np.nan)])
def test_invalid_quadrature_parameters_rejected(n, upper, scale):
    with pytest.raises(ValueError):
        sinh_squared_quadrature(n, upper, scale)
