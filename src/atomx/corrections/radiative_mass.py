"""Finite-mass changes of leading one-loop hydrogenic QED terms.

CODATA 2018, Eqs. (14)–(16), https://doi.org/10.1063/5.0064853.
Scope: alpha*(Z alpha)^4, including its logarithm. This is not complete
radiative recoil or an all-order QED model.
"""
import math

from atomx.constants import ALPHA
from atomx.data.nist_bethe_log import BETHE_LOG_NIST


def hydrogenic_radiative_mass_leading(z, n, kappa, nuclear_mass_me):
    """Finite-minus-infinite-mass SE and VP shifts, in Hartree.

    Ordinary orbital recoil is excluded. The magnetic A40 term scales as
    mu², other leading terms as mu³; the s-state logarithm gains log(1/mu).
    mu=M/(M+1), with nuclear mass M in electron masses.
    """
    if (isinstance(z,bool) or not isinstance(z,int) or not 1<=z<1/ALPHA
            or isinstance(n,bool) or not isinstance(n,int)
            or isinstance(kappa,bool) or not isinstance(kappa,int) or kappa==0):
        raise ValueError('invalid hydrogenic quantum numbers')
    ell=kappa if kappa>0 else -kappa-1
    if n<=ell or n not in BETHE_LOG_NIST:
        raise ValueError('state requires a valid tabulated Bethe logarithm')
    if not math.isfinite(nuclear_mass_me) or nuclear_mass_me<=0:
        raise ValueError('nuclear mass must be positive and finite')
    log_mu=(-math.log1p(1/nuclear_mass_me) if nuclear_mass_me>=1
            else math.log(nuclear_mass_me)-math.log1p(nuclear_mass_me))
    delta_mu3=math.expm1(3*log_mu)
    mu3=math.exp(3*log_mu)
    magnetic=0. if ell==0 else -1/(2*kappa*(2*ell+1))
    a41=4/3 if ell==0 else 0.
    a40=-4/3*BETHE_LOG_NIST[n][ell]+(10/9 if ell==0 else magnetic)
    stationary=a41*math.log((z*ALPHA)**-2)+a40
    magnetic_scaling=-math.exp(2*log_mu)*math.expm1(log_mu)
    factor=ALPHA**3*z**4/(math.pi*n**3)
    se=factor*(delta_mu3*stationary-mu3*a41*log_mu+magnetic_scaling*magnetic)
    vp=factor*delta_mu3*(-4/15 if ell==0 else 0.)
    return {'self_energy_mass_ha':se,'vacuum_polarization_mass_ha':vp}
