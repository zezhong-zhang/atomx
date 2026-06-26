"""FAC angular-convention regression tests.

The reference constants here were generated from the local FAC checkout
(`~/code/fac/faclib/angular.c`) via `InitAngular(); W3j/W6j(...)`. They
lock AtomX's twice-angular-momentum Wigner convention to FAC for the
small half-integer cases used by the Be-like odd-J=1 recoupling audit.
"""

import pytest

from atomx.angular.coefficients import exchange_lambda, reduced_cl_j2
from atomx.angular.wigner import wigner_3j, wigner_6j


@pytest.mark.parametrize(("args", "fac_value"), [
    ((1, 2, 1, -1, 0, 1), 0.40824829046386302),
    ((1, 2, 3, -1, 0, 1), -0.40824829046386313),
    ((3, 2, 1, -1, 0, 1), -0.40824829046386313),
    ((3, 2, 3, -1, 0, 1), -0.12909944487358060),
    ((1, 2, 3, 1, 0, -1), 0.40824829046386313),
])
def test_wigner_3j_matches_fac_half_integer_reference(args, fac_value):
    assert wigner_3j(*args) == pytest.approx(fac_value, abs=1e-14)


@pytest.mark.parametrize(("args", "fac_value"), [
    ((1, 3, 2, 1, 1, 2), -0.33333333333333354),
    ((1, 1, 0, 1, 1, 2), 0.50000000000000022),
    ((3, 1, 2, 1, 1, 2), -0.33333333333333354),
    ((1, 1, 2, 3, 3, 2), 0.26352313834736496),
    ((3, 3, 2, 1, 1, 2), 0.26352313834736496),
])
def test_wigner_6j_matches_fac_half_integer_reference(args, fac_value):
    assert wigner_6j(*args) == pytest.approx(fac_value, abs=1e-14)


@pytest.mark.parametrize(("args", "fac_value"), [
    ((1, 2, 1), -0.8164965809277261),
    ((3, 2, 1), 1.154700538379252),
    ((1, 2, 3), -1.1547005383792517),
    ((3, 2, 3), -0.5163977794943225),
])
def test_reduced_cl_j2_matches_fac_reference(args, fac_value):
    assert reduced_cl_j2(*args) == pytest.approx(fac_value, abs=1e-14)


def test_exchange_lambda_matches_fac_reduced_cl_squared():
    # FAC ReducedCL(ja, k, jb)^2 = (ja+1)(jb+1) W3j(ja,k,jb;1,0,-1)^2.
    assert exchange_lambda(1, -1, 1) == pytest.approx(2.0 / 3.0)
    assert exchange_lambda(-2, -1, 1) == pytest.approx(4.0 / 3.0)
    assert exchange_lambda(-2, -1, 1) == pytest.approx(
        reduced_cl_j2(3, 2, 1) ** 2
    )
