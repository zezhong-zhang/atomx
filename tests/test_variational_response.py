"""Independent eigenenergy, polar-path and native DC radial derivative anchors."""
import itertools
import math

import numpy as np
import pytest

from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.convol import ConvolCache, compute_slater_rk_cell
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.orbital_optimization import _polar_retraction
from atomx.numerics.variational_response import (
    polar_path_derivatives,
    real_eigenstate_response,
)
from atomx.physics.csf.radial_increment import bspline_dc_radial_derivatives


@pytest.mark.parametrize('origin',[0.,-16384.])
@pytest.mark.parametrize('sign',[-1.,1.])
def test_two_level_analytic_energy_and_signed_response(origin,sign):
    h = np.diag([origin,origin+2])
    first = np.array([[.25,sign*.375],[sign*.375,-.5]])
    second = np.array([[.125,sign*-.25],[sign*-.25,.75]])
    out = real_eigenstate_response(h,first,second,np.diag(h),np.eye(2),0)
    g,acc,delta = sign*.375,sign*-.25,-2
    assert out.first == .25
    assert out.second == .125+2*g*g/delta
    assert out.ci_third_coefficient == g*acc/delta + g*g*(-.5-.25)/delta**2
    np.testing.assert_array_equal(out.coefficient_derivative,[0,g/delta])
    assert out.response_residual == 0
    assert out.coefficient_derivative.flags.writeable is False


def test_third_coefficient_against_independent_intermediate_normalization_recursion():
    h = np.diag([-2.,.5,3.])
    h1 = np.array([[.1,.4,-.2],[.4,-.3,.5],[-.2,.5,.2]])
    h2 = np.array([[.2,-.3,.1],[-.3,.2,-.1],[.1,-.1,.5]])
    root,c0 = 1,np.array([0.,1.,0.])
    # Independently solve each order of H(t)c(t)=E(t)c(t), c0.T c(t)=1.
    matrix = h-.5*np.eye(3)+np.outer(c0,c0)
    states,energies = [c0],[.5]
    for order in range(1,4):
        operator = h1@states[order-1]
        if order>=2:
            operator += .5*h2@states[order-2]
        energy = c0@operator
        rhs = -operator+energy*c0
        rhs += sum((energies[j]*states[order-j] for j in range(1,order)),np.zeros(3))
        states.append(np.linalg.solve(matrix,rhs))
        energies.append(energy)
    out = real_eigenstate_response(h,h1,h2,np.diag(h),np.eye(3),root)
    assert out.second == pytest.approx(2*energies[2],abs=2e-16)
    assert out.ci_third_coefficient == pytest.approx(energies[3],abs=2e-16)
    np.testing.assert_allclose(out.coefficient_derivative,states[1],atol=2e-16,rtol=0)
    # A signed change of basis must retain the physical energy derivatives.
    phase = np.diag([-1.,1.,-1.])
    phased = real_eigenstate_response(phase@h@phase,phase@h1@phase,phase@h2@phase,
                                     np.diag(h),phase,root)
    assert phased.ci_third_coefficient == out.ci_third_coefficient


@pytest.mark.parametrize('bad',['partial','degenerate','mismatch','complex','nonsymmetric','cap','root'])
def test_real_response_rejects_unqualified_arrays(bad):
    h,h1,h2,e,v,root = np.diag([0.,2.]),np.eye(2),np.eye(2),np.array([0.,2.]),np.eye(2),0
    kw = {}
    if bad=='partial':
        e,v = e[:1],v[:,:1]
    elif bad=='degenerate':
        h,e = np.zeros((2,2)),np.zeros(2)
    elif bad=='mismatch':
        e = np.array([0.,3.])
    elif bad=='complex':
        h1 = h1.astype(complex)
    elif bad=='nonsymmetric':
        h1[0,1] = .2
    elif bad=='cap':
        kw['max_dimension'] = 1
    else:
        root = True
    with pytest.raises(ValueError):
        real_eigenstate_response(h,h1,h2,e,v,root,**kw)


def test_polar_single_column_against_normalized_vector_formula():
    y,d = np.array([[1.2],[-.5],[.3]]),np.array([[-.2],[.4],[.1]])
    p,p1,p2 = polar_path_derivatives(y,d)
    r,s,dd = np.linalg.norm(y),float((y.T@d).item()),float((d.T@d).item())
    np.testing.assert_allclose(p,y/r,atol=3e-16,rtol=0)
    np.testing.assert_allclose(p1,d/r-y*s/r**3,atol=3e-16,rtol=0)
    np.testing.assert_allclose(p2,-2*d*s/r**3-y*dd/r**3+3*y*s*s/r**5,atol=3e-16,rtol=0)


def test_polar_repeated_gram_full_constraint_derivatives_and_native_path():
    y = np.eye(4)[:,:2]
    d = np.array([[.1,-.2],[.3,.15],[.4,.2],[-.1,.3]])
    p,p1,p2 = polar_path_derivatives(y,d)
    np.testing.assert_allclose(p.T@p1+p1.T@p,0,atol=3e-16)
    np.testing.assert_allclose(p.T@p2+p2.T@p+2*p1.T@p1,0,atol=5e-16)
    step = .0002
    plus,minus = _polar_retraction(y+step*d),_polar_retraction(y-step*d)
    np.testing.assert_allclose((plus-minus)/(2*step),p1,atol=8e-9,rtol=0)
    np.testing.assert_allclose((plus+minus-2*p)/step**2,p2,atol=3e-8,rtol=0)


def test_radial_first_second_against_independent_two_four_slot_integrals():
    ns,Z = 24,3
    data = prepare_bspline_data(make_knots(ns,7,.005,20),ns,7,kinetic_balance=True)
    cache,rng = ConvolCache(data),np.random.default_rng(122)
    waves,first,second,raw = {},{},{},{}
    for key in ((1,-1),(2,-1),(2,1)):
        n,k = key
        if k not in raw:
            raw[k] = build_dirac_matrices(ns,k,Z,data)
        sp = solve_dirac(*raw[k],ns,k,nsp=data['nsp'],nsq=data['nsq'])
        p,q,_ = sp.get_orbital(n-(k if k>0 else -k-1)-1)
        waves[key] = p,q
        first[key] = tuple(rng.normal(size=ns)*.01 for _ in range(2))
        second[key] = tuple(rng.normal(size=ns)*.02 for _ in range(2))
    derivative = bspline_dc_radial_derivatives(waves,first,second,data,Z,ns,cache=cache)
    for a,b in (((1,-1),(1,-1)),((1,-1),(2,-1))):
        h,s = raw[a[1]]
        x,y = map(np.concatenate,(waves[a],waves[b]))
        u,v = map(np.concatenate,(first[a],first[b]))
        aa,bb = map(np.concatenate,(second[a],second[b]))
        N,D = x@h@y,y@s@y
        N1,D1 = u@h@y+x@h@v,2*v@s@y
        N2,D2 = aa@h@y+2*u@h@v+x@h@bb,2*bb@s@y+2*v@s@v
        oracle = (N1/D-N*D1/D**2,
                  N2/D-2*N1*D1/D**2-N*D2/D**2+2*N*D1**2/D**3)
        np.testing.assert_allclose(derivative.components(('h',a[0],a[1],b[0])),oracle,atol=5e-14,rtol=1e-13)
    for rank,keys in ((0,((1,-1),(2,-1),(2,1),(2,1))),
                      (2,((2,1),(2,1),(2,1),(2,1)))):
        def integral(args,rank=rank):
            return float(compute_slater_rk_cell(*args[0],*args[1],*args[2],*args[3],data,rank,cache=cache))

        term1,term2 = [],[]
        for slot in range(4):
            args = [waves[k] for k in keys]
            args[slot] = first[keys[slot]]
            term1.append(integral(args))
            args[slot] = second[keys[slot]]
            term2.append(integral(args))
        for i,j in itertools.combinations(range(4),2):
            args = [waves[k] for k in keys]
            args[i],args[j] = first[keys[i]],first[keys[j]]
            term2.append(2*integral(args))
        np.testing.assert_allclose(derivative.components(('r',rank,*keys)),
                                   (math.fsum(term1),math.fsum(term2)),atol=2e-16,rtol=3e-13)
