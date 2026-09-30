"""Unit-amplitude outgoing Dirac–Coulomb reference spinor.

The real and imaginary parts are the irregular and regular solutions in
Salvat's DCOULF phase convention. Both large components have unit amplitude
at infinity. Charge is positive for V=-charge/r; energy excludes rest mass.

For a Coulomb tail y'=(A0+A1/r)y, diagonalizing A0 gives eigenvalues +/-ip.
Factoring exp(i[p*r+beta*log(2*p*r)]) yields a two-component inverse-radius
series. Its coupled recurrence retains the relativistic amplitude variation;
there is no WKB envelope approximation. If that asymptotic series cannot
reach tolerance, evaluate the exact Coulomb-function representation instead.
"""
from __future__ import annotations

import cmath
import math

import numpy as np
from scipy.special import loggamma, spherical_jn, spherical_yn

from atomx.constants import C


def _outgoing_series(energy, kappa, charge, radius, c, tolerance):
    momentum = math.sqrt(energy*(2+energy/c**2))
    ratio = momentum*c/(energy+2*c*c)
    beta = charge*(1+energy/c**2)/momentum
    gamma = math.sqrt(kappa*kappa-(charge/c)**2)
    ell = kappa if kappa > 0 else -kappa-1
    # Stable gamma+kappa for negative kappa and small alpha*charge.
    kp = gamma+kappa if kappa > 0 else -(charge/c)**2/(gamma-kappa)
    delta = (math.atan2(-kp*momentum*c, -charge/c*(energy+2*c*c))
             -(gamma-ell-1)*math.pi/2+loggamma(complex(gamma, -beta)).imag)
    if charge > 0 and kappa < 0:
        delta -= math.pi
    phase = momentum*radius+beta*math.log(2*momentum*radius)-ell*math.pi/2+delta
    plus = complex(-kappa, charge/momentum)
    minus = plus.conjugate()
    previous_a, previous_b = 1+0j, 0j
    total_a, total_b = previous_a, previous_b
    previous_size = math.inf
    small_terms = 0
    for order in range(1, 129):
        b = ((order-1-2j*beta)*previous_b+minus*previous_a)/(2j*momentum*radius)
        a = -plus*b/order
        size = abs(a)+abs(b)
        # Stop before the divergent part, or before destructive cancellation
        # would lose the requested double-precision accuracy.
        if size > previous_size or size > 1e4:
            return None
        total_a += a
        total_b += b
        if size < tolerance*(abs(total_a)+abs(total_b)):
            small_terms += 1
            if small_terms >= 2:
                rotation = cmath.exp(1j*phase)
                return np.array([(total_a+total_b)*rotation,
                                 1j*ratio*(total_a-total_b)*rotation])
        else:
            small_terms = 0
        previous_a, previous_b, previous_size = a, b, size
    return None


def _outgoing_coulomb_functions(energy, kappa, charge, radius, c):
    # A private context avoids mutating mpmath's global precision when batch
    # continuum solves use threads. The usual outer-tail path never imports it.
    import mpmath

    context = mpmath.mp.clone()
    context.dps = 40
    energy, c, zeta = context.mpf(energy), context.mpf(c), -context.mpf(charge)/c
    rest = c*c
    pc = context.sqrt(energy*(energy+2*rest))
    gamma = context.sqrt(kappa*kappa-zeta*zeta)
    eta = zeta*(energy+rest)/pc
    a = kappa+gamma
    b = gamma*rest-kappa*(energy+rest)
    norm = 1/(context.sqrt((zeta*(energy+2*rest))**2+(a*pc)**2)*gamma)
    if charge > 0 and kappa < 0:
        norm = -norm
    coefficient = context.sqrt(gamma*gamma+eta*eta)*pc*norm
    p1, p0 = a*coefficient, zeta*b*norm
    q1, q0 = zeta*coefficient, a*b*norm
    x = pc/c*radius
    high = context.coulombg(gamma, eta, x)+1j*context.coulombf(gamma, eta, x)
    low = context.coulombg(gamma-1, eta, x)+1j*context.coulombf(gamma-1, eta, x)
    return np.array([complex(p1*high+p0*low), complex(-q1*high-q0*low)])


def dirac_coulomb_outgoing(energy, kappa, charge, radius, *, c=C, tolerance=2e-13):
    """Return complex (P,Q), with Re=irregular G and Im=regular F.

    Both real reference solutions have unit large-component amplitude at
    infinity. Their radial determinant F_P*G_Q-F_Q*G_P equals
    -sqrt(E/(E+2*c**2)). No continuum energy-normalization factor is applied.
    ``c`` permits comparisons with reference codes using other constants.
    """
    if (not all(math.isfinite(x) for x in (energy, charge, radius, c, tolerance))
            or energy <= 0 or radius <= 0 or c <= 0 or not 0 < tolerance < 1
            or kappa == 0 or int(kappa) != kappa):
        raise ValueError('Coulomb reference requires positive E/r/c/tolerance and nonzero integer kappa')
    if abs(charge/c) >= abs(kappa):
        raise ValueError('Coulomb reference requires real sqrt(kappa²-(charge/c)²)')
    if charge == 0:
        momentum = math.sqrt(energy*(2+energy/c**2))
        x = momentum*radius
        ell = kappa if kappa > 0 else -kappa-1
        small_ell = ell-1 if kappa > 0 else ell+1
        factor = math.sqrt(energy/(energy+2*c*c))*(1 if kappa > 0 else -1)
        value = x*np.array([
            complex(-spherical_yn(ell, x), spherical_jn(ell, x)),
            factor*complex(-spherical_yn(small_ell, x), spherical_jn(small_ell, x))])
    else:
        value = _outgoing_series(energy, kappa, charge, radius, c, tolerance)
        if value is None:
            value = _outgoing_coulomb_functions(energy, kappa, charge, radius, c)
    if not np.all(np.isfinite(value)):
        raise FloatingPointError('nonfinite Dirac–Coulomb reference')
    return value
