"""Static transverse operator gates independent of old Gaunt snapshots."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_legendre
from atomx import Atom
from atomx.physics.csf.breit_ci import (
    _ci_radial_inputs, breit_operator_energy, gaunt_angular_terms,
)
from atomx.physics.csf.retardation import distance_potential, radial_evaluator


@pytest.mark.parametrize('L',range(5))
def test_distance_multipole_against_angular_projection(L):
    a,b=.7,1.8
    angular=(2*L+1)/2*quad(lambda x: np.sqrt(a*a+b*b-2*a*b*x)*eval_legendre(L,x),-1,1,epsabs=1e-13)[0]
    radial=a**(L+2)*b**(-L-1)/(2*L+3)-a**L*b**(1-L)/(2*L-1)
    assert angular==pytest.approx(radial,abs=1e-13)


def test_distance_radial_contraction_against_split_adaptive_quadrature():
    x,w=np.polynomial.legendre.leggauss(12)
    edges=np.linspace(.01,8,80);half=np.diff(edges)[:,None]/2
    r=edges[:-1,None]+half*(1+x);weights=half*w
    rho=r*r*np.exp(-2*r)
    for L in (0,1,2,3):
        pot=distance_potential(rho,r,weights,L)
        for i,j in ((4,3),(30,6),(65,2)):
            target=r[i,j]
            def f(s):
                lo,hi=sorted([s,target])
                kernel=lo**(L+2)*hi**(-L-1)/(2*L+3)-lo**L*hi**(1-L)/(2*L-1)
                return s*s*np.exp(-2*s)*kernel
            exact=quad(f,edges[0],target,epsabs=1e-13)[0]+quad(f,target,edges[-1],epsabs=1e-13)[0]
            assert pot[i,j]==pytest.approx(exact,abs=3e-12)


def test_gaunt_includes_rank_two_for_j_half_transition():
    # alpha (rank 1) couples photon L=2 to electronic K=1. The previous
    # j_a+j_c ceiling wrongly discarded this allowed s1/2-p1/2 channel.
    terms=gaunt_angular_terms(-1,1,-1,-1,1,1,1,-1)
    assert any(L==2 and abs(c)>1e-12 for L,*_,c in terms)


@pytest.mark.parametrize('z,config,reference',[
    (2,'1s2',.000063777432919),
    (10,'1s2 2s2 2p6',.016643630840490),
    (18,'1s2 2s2 2p6 3s2 3p6',.132364619357042),
])
def test_full_operator_matches_independent_dbsr_hf(z,config,reference):
    r=Atom(z,config=config,engine='bspline',corrections='none').solve().result
    orbitals,ws,z,ns=_ci_radial_inputs(r)
    assert breit_operator_energy(orbitals,ws,z,ns,config)==pytest.approx(reference,rel=8e-7)


def test_retardation_radial_hermiticity_and_zero_diagonal_current():
    r=Atom(4,config='1s2 2s2',engine='bspline',corrections='none').solve().result
    orbitals,ws,_,_=_ci_radial_inputs(r);ev=radial_evaluator(orbitals,ws)
    a,c=(1,-1),(2,-1)
    value=ev(('t',0,a,c,a,c))
    assert abs(value)>1e-12
    assert ev(('t',0,c,a,c,a))==pytest.approx(value,rel=1e-13)
    assert ev(('t',0,a,c,c,a))==pytest.approx(-value,rel=1e-13)
    assert ev(('t',0,a,a,a,c))==0.
