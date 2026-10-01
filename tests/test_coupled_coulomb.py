"""Racah reduction checked against magnetic sums and exact symbolic algebra."""
import math
from itertools import product

import pytest
from sympy import Rational, sqrt
from sympy.physics.wigner import wigner_3j, wigner_6j

from atomx.angular.coefficients import (
    cg,
    ckm,
    coupled_coulomb_direct,
    coupled_coulomb_exchange,
)


def magnetic_sum(c, d, a, b, tJ, k, tM, exchange=False):
    """Explicit ordered-pair contraction, independent of the 6j reduction."""
    ja, jb, jc, jd = [2*abs(q)-1 for q in (a, b, c, d)]
    terms = []
    for ma in range(-ja, ja+1, 2):
        mb = tM-ma
        if abs(mb) > jb:
            continue
        for mc in range(-jc, jc+1, 2):
            md = tM-mc
            if abs(md) > jd:
                continue
            left = cg(ja, ma, jb, mb, tJ, tM) * cg(jc, mc, jd, md, tJ, tM)
            q = (md if exchange else mc)-ma
            if exchange:
                vertex = ckm(d, md, a, ma, k, q)*ckm(c, mc, b, mb, k, -q)
            else:
                vertex = ckm(c, mc, a, ma, k, q)*ckm(d, md, b, mb, k, -q)
            terms.append(left * (-1 if (q//2) % 2 else 1) * vertex)
    return math.fsum(terms)


@pytest.mark.parametrize('exchange', [False, True])
def test_all_low_j_channels_and_m_projections(exchange):
    coefficient = coupled_coulomb_exchange if exchange else coupled_coulomb_direct
    for c, d, a, b in product((-1, 1, -2, 2), repeat=4):
        js = [2*abs(q)-1 for q in (a, b, c, d)]
        lo, hi = max(abs(js[0]-js[1]), abs(js[2]-js[3])), min(sum(js[:2]), sum(js[2:]))
        for tJ in range(lo, hi+1, 2):
            for k in range(4):
                value = coefficient(c, d, a, b, tJ, k)
                for tM in {-tJ, 0, tJ}:
                    ref = magnetic_sum(c, d, a, b, tJ, k, tM, exchange)
                    assert value == pytest.approx(ref, abs=5e-14), (c, d, a, b, tJ, k, tM)


def exact_coefficient(c, d, a, b, tJ, k):
    jc, jd, ja, jb = [Rational(2*abs(q)-1, 2) for q in (c, d, a, b)]
    half = Rational(1, 2)
    def red(x, y):
        return (-1)**(x+half)*sqrt((2*x+1)*(2*y+1))*wigner_3j(x,k,y,-half,0,half)
    return float(((-1)**(ja+jd+tJ//2)*wigner_6j(jc,jd,tJ//2,jb,ja,k)
                  * red(jc,ja)*red(jd,jb)).evalf(40))


@pytest.mark.parametrize('channel', [
    (-33, -33, -1, -33, 64, 32),
    (-49, -49, 4, -65, 126, 52),
    (-65, -65, 2, 64, 128, 64),
    (-65, 32, 64, 64, 76, 80),
    (64, -65, 64, 64, 66, 68),
    (48, 48, -65, -49, 116, 80),
])
def test_high_partial_waves_against_exact_algebra(channel):
    c, d, a, b, tJ, k = channel
    expected = exact_coefficient(*channel)
    assert coupled_coulomb_direct(*channel) == pytest.approx(expected, rel=3e-12, abs=3e-15)
    # Reversed bra + exchange tests the coupled-state permutation phase.
    phase = (-1)**(abs(c)+abs(d)-1-tJ//2)
    assert coupled_coulomb_exchange(d,c,a,b,tJ,k) == pytest.approx(phase*expected, rel=3e-12, abs=3e-15)


def test_monopole_identity_and_pauli_cancellation():
    for a, b in product((-1, 1, -4, 4, -33), repeat=2):
        for tJ in range(2*abs(abs(a)-abs(b)), 2*(abs(a)+abs(b)), 2):
            assert coupled_coulomb_direct(a,b,a,b,tJ,0) == pytest.approx(1., abs=2e-12)
    for a in (-1, -2, -4):
        for tJ in range(0, 4*abs(a)-1, 2):
            if tJ//2 % 2:  # equivalent electrons: odd J is Pauli forbidden
                for k in range(2*abs(a)):
                    assert coupled_coulomb_direct(a,a,a,a,tJ,k) == coupled_coulomb_exchange(a,a,a,a,tJ,k)


@pytest.mark.parametrize('tJ', [-2, 1, 6])
def test_forbidden_total_angular_momentum(tJ):
    assert coupled_coulomb_direct(-1,-1,-1,-1,tJ,0) == 0.
    assert coupled_coulomb_exchange(-1,-1,-1,-1,tJ,0) == 0.
