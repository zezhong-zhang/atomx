"""Independent high-angular-momentum and compiled-caller Wigner checks."""
import pytest
from sympy import Rational
from sympy.physics.wigner import wigner_3j as exact_3j, wigner_6j as exact_6j

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.wigner import wigner_3j, wigner_6j, wigner_9j


@pytest.mark.parametrize('args', [
    (63,128,129,-1,0,1), (129,128,127,-1,0,1),
    (129,160,127,-1,0,1), (127,136,129,-1,0,1),
    (100,100,100,0,0,0), (101,100,101,1,0,-1),
    (60,60,60,20,-10,-10), (40,40,40,0,0,0),
    (100,100,100,100,0,-100),
    (600,2,600,0,0,0), (1100,0,1100,0,0,0),
])
def test_3j_cancellation_and_large_factorials(args):
    reference = float(exact_3j(*(Rational(x,2) for x in args)).evalf(40))
    assert wigner_3j(*args) == pytest.approx(reference, rel=3e-12, abs=3e-15)


@pytest.mark.parametrize('args', [
    (100,100,100,100,100,100), (64,64,64,64,64,64),
    (129,63,76,127,127,160), (127,129,66,127,127,136),
    (95,95,116,97,129,160), (1,1,0,1,1,0),
    (600,600,0,600,600,0), (100,100,0,100,100,100),
])
def test_6j_cancellation_and_large_factorials(args):
    reference = float(exact_6j(*(Rational(x,2) for x in args)).evalf(40))
    assert wigner_6j(*args) == pytest.approx(reference, rel=3e-12, abs=3e-15)


def test_compiled_callers_reach_exact_fallback():
    from sympy.physics.wigner import clebsch_gordan as exact_cg, wigner_9j as exact_9j
    reference = float(exact_cg(20,20,20,0,0,0).evalf(40))
    assert clebsch_gordan(40,0,40,0,40,0) == pytest.approx(reference, rel=3e-12)
    reference = float(exact_9j(*([10]*9)).evalf(40))
    assert wigner_9j(*([20]*9)) == pytest.approx(reference, rel=3e-12)


def test_representable_symbol_does_not_underflow_when_squared():
    args = (4000,4000,4000,4000,-4000,0)
    reference = float(exact_3j(*(Rational(x,2) for x in args)).evalf(40))
    assert 0.0 < reference < 1e-170
    assert wigner_3j(*args) == pytest.approx(reference, rel=3e-12, abs=0.0)
