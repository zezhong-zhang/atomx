"""Analytic derivative, continuity, complex kernel and full static Breit gates."""
from collections import OrderedDict

import numpy as np
import pytest

from atomx.angular.csf_hamiltonian import csf_signature, exact_block_matrix
from atomx.constants import ALPHA, C
from atomx.methods.csf.complex_ci import ComplexScaledCIModel
from atomx.numerics.bspline.complex_dirac import ComplexScaledDiracWorkspace
from atomx.numerics.bspline.knots import make_exponential_knots
from atomx.physics.csf.breit_ci import breit_terms
from atomx.physics.csf.complex_radial import ComplexCIRadialSnapshot
from atomx.physics.csf.retardation import distance_potential


def snapshot(theta, *, derivatives=True):
    # Fixed thresholds require both a resolved nuclear cusp and a dense
    # log grid. The ns65/r0=1e-4 and intermediate failures are archived
    # in docs/xfel_complex_breit_2026_10_04; energy residuals alone missed them.
    knots=make_exponential_knots(125,9,1e-6,25)
    workspaces={k:ComplexScaledDiracWorkspace(knots,9,k,6,n_gauss=25) for k in (-1,1,-2)}
    return ComplexCIRadialSnapshot(workspaces,{-1:3,1:2,-2:2},theta,
                                  with_derivatives=derivatives)


@pytest.mark.parametrize('theta',[0,.2,.4])
def test_analytic_DKB_derivative_against_Sommerfeld_1s_shape(theta):
    s=snapshot(theta)
    r=s.gr.ravel()
    gamma=np.sqrt(1-(6*ALPHA)**2)
    for component in (0,1):
        f=s.orbitals[(1,-1)][component]
        df=s.derivatives[(1,-1)][component]
        mask=(r>.02)&(abs(f)>1e-5*abs(f).max())
        expected=(gamma/r-6*np.exp(1j*theta))*f
        weighted=s.grw.ravel()[mask]
        error=np.sqrt(np.sum(weighted*abs(df[mask]-expected[mask])**2)/
                      np.sum(weighted*abs(expected[mask])**2))
        assert error<1e-6


@pytest.mark.parametrize('theta',[0,.3])
def test_current_continuity_for_common_dilated_local_Dirac_operator(theta):
    s=snapshot(theta)
    z=np.exp(-1j*theta)
    for a,c in (((1,-1),(2,-1)),((1,-1),(2,1)),((2,-2),(2,1))):
        pa,qa=s.orbitals[a]
        pc,qc=s.orbitals[c]
        expected=(s.energies[c]-s.energies[a])/(z*C)*(pa*pc+qa*qc)
        actual=s._current((a,c))
        weighted=s.grw.ravel()
        assert np.sqrt(np.sum(weighted*abs(actual-expected)**2))<1e-7
        # Separate evaluations reorder floating additions. Antisymmetry of
        # the continuum identity allows roundoff, not a physical tolerance;
        # integral pair canonicalization is tested bitwise below.
        np.testing.assert_allclose(s._current((c,a)),-actual,rtol=0,
                                   atol=4*np.finfo(float).eps*max(1.,abs(actual).max()))
    assert np.count_nonzero(s._current(((1,-1),(1,-1))))==0


@pytest.mark.parametrize('theta',[0,.2,.4])
def test_retardation_dilation_against_closed_polynomial_double_integrals(theta):
    # Independent exact densities: Pa=A*r^2, Qc=B*r^3 -> Dac=-5AB*r^4;
    # Pb=C*r^2, Qd=D*r^4 -> Dbd=-6CD*r^5. No orbital-energy identity used.
    x,w=np.polynomial.legendre.leggauss(17)
    r=((x+1)/2)[None,:]
    weights=(w/2)[None,:]
    A,B,C0,D=1+2j,.7-.3j,.6+.4j,1.2-.8j
    a,c,b,d=(1,-1),(2,-1),(3,-1),(4,-1)
    s=object.__new__(ComplexCIRadialSnapshot)
    s.theta,s.gr,s.grw=theta,r,weights
    zero=np.zeros(r.size)
    s.orbitals={a:(A*r.ravel()**2,zero),c:(zero,B*r.ravel()**3),
                b:(C0*r.ravel()**2,zero),d:(zero,D*r.ravel()**4)}
    s.derivatives={a:(2*A*r.ravel(),zero),c:(zero,3*B*r.ravel()**2),
                   b:(2*C0*r.ravel(),zero),d:(zero,4*D*r.ravel()**3)}
    s._cache,s._integrals,s._cache_bytes,s._used_bytes=OrderedDict(),{},0,0
    for L in range(4):
        p,q=4,5
        exact=((1/(q+L+3)+1/(p+L+3))/(2*L+3)
               -(1/(q+L+1)+1/(p+L+1))/(2*L-1))/(p+q+3)
        expected=np.exp(-1j*theta)*30*A*B*C0*D*exact
        value=s.retardation(L,a,c,b,d)
        assert abs(value-expected)<1e-13
        assert s.retardation(L,c,a,b,d)==-value
        assert s.retardation(L,b,d,a,c)==value


def test_actual_gaunt_split_quadrature_and_retardation_continuity_contraction():
    s=snapshot(.3)
    for L in (0,1,2):
        keys=(('P',1,-1),('Q',2,1),('Q',2,-1),('P',2,-2))
        assert abs(s.gaunt(L,*keys)-s.gaunt(L,*keys,method='direct'))<1e-10
    a,c,b,d=(1,-1),(2,-1),(2,1),(2,-2)
    z=np.exp(-1j*s.theta)
    def charge_current(a,c):
        pa,qa=s.orbitals[a]
        pc,qc=s.orbitals[c]
        return ((s.energies[c]-s.energies[a])/(z*C)*(pa*pc+qa*qc)).reshape(s.gr.shape)
    da,db=charge_current(a,c),charge_current(b,d)
    expected=z*np.sum(s.grw*da*distance_potential(db,s.gr,s.grw,1))
    assert abs(s.retardation(1,a,c,b,d)-expected)<1e-9
    without=snapshot(.3,derivatives=False)
    with pytest.raises(ValueError,match='derivatives'):
        without.retardation(0,a,c,a,c)
    with pytest.raises(ValueError,match='unsupported'):
        s.evaluate_breit(('missing',0,a,c,b,d))


def test_static_DCB_is_complete_non_scalar_operator_and_N1_has_no_self_pair():
    knots=make_exponential_knots(55,9,1e-4,20)
    model=ComplexScaledCIModel(knots,9,6,2,{-1:3,1:2,-2:2},n_gauss=17,include_breit=True)
    result=model.solve(.3,0,1)
    sig=[csf_signature(c) for c in result.csfs]
    radial=result.radial
    dc=exact_block_matrix(sig,radial.evaluate,integral_dtype=complex)
    full=exact_block_matrix(sig,radial.evaluate,integral_dtype=complex,
                            breit_evaluator=radial.evaluate_breit)
    gaunt=exact_block_matrix(sig,radial.evaluate,integral_dtype=complex,
        breit_evaluator=lambda key:radial.evaluate_breit(key) if key[0]=='g' else 0j)
    assert np.linalg.norm(full-gaunt)>1e-8
    assert np.linalg.norm((full-dc)-np.eye(len(full))*np.trace(full-dc)/len(full))>1e-6
    np.testing.assert_allclose((result.coefficients*result.energies)@result.coefficients.T,
                               full,atol=1e-10)
    assert result.include_breit and 'retardation' in result.hamiltonian
    one=ComplexScaledCIModel(knots,9,6,1,{-1:3},n_gauss=17,include_breit=True).solve(.3,1,1)
    np.testing.assert_allclose(one.energies,list(one.radial.energies.values()),atol=1e-11)
    with pytest.raises(ValueError,match='bool'):
        ComplexScaledCIModel(knots,9,6,2,{-1:2},include_breit='gaunt')


def test_N3_static_DCB_diagonal_against_explicit_antisymmetrized_pairs():
    # 1s(up) 1s(down) 2s(up) is the stretched 1s²2s CSF. Independently
    # count each physical electron pair once, then direct minus exchange;
    # shared Breit angular elements have separate Pauli/SymPy tests.
    knots=make_exponential_knots(55,9,1e-4,20)
    block=ComplexScaledCIModel(knots,9,6,3,{-1:2},n_gauss=17,
                              include_breit=True).solve(.2,1,1)
    labels=((1,-1,1),(1,-1,-1),(2,-1,1))
    radial=block.radial
    a,b=(1,-1),(2,-1)
    expected=(2*radial.energies[a]+radial.energies[b]+radial.coulomb(0,a,a,a,a)
              +2*radial.coulomb(0,a,a,b,b)-radial.coulomb(0,a,b,b,a))
    correction=0j
    for i in range(3):
        for j in range(i+1,3):
            for k,l,sign in ((i,j,1),(j,i,-1)):
                correction+=sum(coefficient*radial.evaluate_breit(key) for key,coefficient
                    in breit_terms(i,j,k,l,sign,labels))
    row=next(i for i,c in enumerate(block.csfs) if
             {sh.n:sh.occupation for sh in c.subshells}=={1:2,2:1})
    reconstructed=(block.coefficients*block.energies)@block.coefficients.T
    assert abs(correction)>1e-6
    assert abs(reconstructed[row,row]-expected-correction)<1e-11
