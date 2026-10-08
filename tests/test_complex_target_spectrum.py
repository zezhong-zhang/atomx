"""Original-operator and analytic anchors for partial complex CI spectra."""
import numpy as np
import pytest
from scipy.sparse.linalg import ArpackNoConvergence, aslinearoperator

from atomx.methods.csf.complex_ci import ComplexScaledCIModel
from atomx.numerics.bspline.knots import make_exponential_knots
from atomx.numerics.complex_eigensolver import complex_target_spectrum


def operator(matrix):
    result=aslinearoperator(matrix)
    result.diagonal=lambda:np.diag(matrix).copy()
    return result


def analytic_matrix():
    # Exact complex orthogonal rotation, nonunitary: Q.T Q=I, Q†Q!=I.
    # Its diagonal eigenvalues are an independent anchor, not LAPACK output.
    values=np.arange(-5,7,dtype=float)-.01j*np.arange(12)
    Q=np.eye(12,dtype=complex)
    c,s=np.cosh(.5),1j*np.sinh(.5)
    Q[:2,:2]=[[c,s],[-s,c]]
    return Q@np.diag(values)@Q.T,values


@pytest.mark.parametrize('backend',['matrix_free','lu'])
def test_nonhermitian_analytic_spectrum_original_residual_and_c_product(backend):
    H,values=analytic_matrix()
    assert np.linalg.norm(H-H.conj().T)>.1
    result=complex_target_spectrum(operator(H) if backend=='matrix_free' else H,
                                   sigma=-4.2-.1j,roots=3)
    expected=values[np.argsort(abs(values-(-4.2-.1j)))[:3]]
    np.testing.assert_allclose(result.energies,expected,rtol=0,atol=1e-11)
    np.testing.assert_allclose(result.coefficients.T@result.coefficients,np.eye(3),atol=1e-11)
    np.testing.assert_allclose(H@result.coefficients,result.coefficients*result.energies,atol=1e-10)
    assert result.residuals.max()<1e-10
    assert result.diagnostics['max_linear_relative_residual']<=1e-12
    assert result.diagnostics['inverse_calls']>0
    assert not result.coefficients.flags.writeable
    with pytest.raises(TypeError):
        result.diagnostics['full_spectrum']=True


def test_real_operator_uses_complex_shift_transform():
    H=np.diag(np.arange(-5,7,dtype=float))
    result=complex_target_spectrum(operator(H),sigma=-3.8+.3j,roots=2)
    np.testing.assert_allclose(result.energies,[-4,-3],atol=1e-11)


@pytest.mark.parametrize('N,J2,theta,breit',[(2,0,0,False),(2,0,.3,True),(3,1,.2,True)])
@pytest.mark.parametrize('backend',['matrix_free','lu'])
def test_actual_full_retained_CI_target_matches_full_spectrum(N,J2,theta,breit,backend):
    knots=make_exponential_knots(45,9,1e-4,15)
    model=ComplexScaledCIModel(knots,9,6,N,{-1:3,1:2,-2:2},
                              n_gauss=25,include_breit=breit)
    full=model.solve(theta,J2,1)
    sigma=complex(full.energies[2])+.02+.01j
    subset=model.solve_target(theta,J2,1,sigma=sigma,roots=2,backend=backend)
    expected=full.energies[np.argsort(abs(full.energies-sigma))[:2]]
    np.testing.assert_allclose(subset.energies,expected,rtol=0,atol=1e-10)
    assert subset.csfs==full.csfs
    assert subset.radial.keys==full.radial.keys
    assert subset.coefficients.shape==(len(full.csfs),2)
    assert full.full_spectrum and not subset.full_spectrum
    assert subset.include_breit==breit
    assert subset.hamiltonian==full.hamiltonian
    assert subset.residuals.max()<1e-10


@pytest.mark.parametrize('options',[
    {'roots':True},{'roots':0},{'roots':11},{'sigma':np.nan},
    {'sigma':True},{'sigma':'bad'},{'tol':1e-8},{'tol':1j},
    {'ncv':3},{'ncv':13},{'maxiter':False},{'linear_maxiter':0},
])
def test_invalid_target_controls_reject(options):
    H,_=analytic_matrix()
    args={'sigma':-4.2-.1j,'roots':3,**options}
    with pytest.raises(ValueError):
        complex_target_spectrum(operator(H),**args)


def test_model_rejects_controls_before_radial_allocation(monkeypatch):
    model=ComplexScaledCIModel(make_exponential_knots(35,9,1e-4,15),9,6,2,{-1:3},n_gauss=25)
    def forbidden(*args,**kwargs):
        pytest.fail('invalid request allocated a physical radial snapshot')
    monkeypatch.setattr('atomx.methods.csf.complex_ci.ComplexCIRadialSnapshot',forbidden)
    with pytest.raises(ValueError,match='sigma'):
        model.solve_target(.2,0,1,sigma=np.inf,roots=1,backend='matrix_free')
    with pytest.raises(ValueError,match='backend'):
        model.solve_target(.2,0,1,sigma=-8,roots=1,backend='auto')


def test_successful_inner_flag_does_not_bypass_original_linear_residual(monkeypatch):
    H,_=analytic_matrix()
    monkeypatch.setattr('atomx.numerics.complex_eigensolver.gmres',
                        lambda A,b,**kwargs:(np.zeros_like(b),0))
    with pytest.raises(ValueError,match='linear residual gate'):
        complex_target_spectrum(operator(H),sigma=-4.2-.1j,roots=3)


def test_outer_failure_rejects_partial_ARPACK_output(monkeypatch):
    H,_=analytic_matrix()
    def fail(*args,**kwargs):
        raise ArpackNoConvergence('partial',np.array([-4+0j]),np.eye(12,dtype=complex)[:,:1])
    monkeypatch.setattr('atomx.numerics.complex_eigensolver.eigs',fail)
    with pytest.raises(ValueError,match='all requested roots'):
        complex_target_spectrum(operator(H),sigma=-4.2-.1j,roots=3)


@pytest.mark.parametrize('kind',['self_orthogonal','nonorthogonal','bad_residual'])
def test_post_solver_validation_is_independent_of_convergence_flag(monkeypatch,kind):
    H,_=analytic_matrix()
    vectors=np.eye(12,dtype=complex)[:,:2]
    if kind=='self_orthogonal':
        vectors[:,0]=0
        vectors[:2,0]=[1,1j]
    elif kind=='nonorthogonal':
        vectors[:,1]=vectors[:,0]
    monkeypatch.setattr('atomx.numerics.complex_eigensolver.eigs',
                        lambda *args,**kwargs:(np.array([-4,-3],dtype=complex),vectors))
    with pytest.raises(ValueError,match={'self_orthogonal':'self-orthogonal',
                       'nonorthogonal':'orthogonality','bad_residual':'original-operator'}[kind]):
        complex_target_spectrum(operator(H),sigma=-4.2-.1j,roots=2)


def test_model_target_path_never_requests_dense_CSF_matrix(monkeypatch):
    from atomx.angular.csf_hamiltonian import exact_block_matrix
    seen=[]
    def guarded(*args,**kwargs):
        assert kwargs['as_operator'] is True
        seen.append(True)
        return exact_block_matrix(*args,**kwargs)
    monkeypatch.setattr('atomx.methods.csf.complex_ci.exact_block_matrix',guarded)
    knots=make_exponential_knots(45,9,1e-4,15)
    model=ComplexScaledCIModel(knots,9,6,2,{-1:3},n_gauss=25)
    result=model.solve_target(.2,0,1,sigma=-8.2-.01j,roots=1,backend='matrix_free')
    assert seen==[True] and not result.full_spectrum


def test_singular_shifted_LU_rejects_without_changing_shift():
    H=np.diag(np.arange(12,dtype=float))
    with pytest.raises(ValueError,match='LU factorization'):
        complex_target_spectrum(H,sigma=3,roots=2)


def test_LU_inverse_also_has_original_system_residual_gate(monkeypatch):
    H,_=analytic_matrix()
    monkeypatch.setattr('atomx.numerics.complex_eigensolver.linalg.lu_solve',
                        lambda factor,b,**kwargs:np.zeros_like(b))
    with pytest.raises(ValueError,match='linear residual gate'):
        complex_target_spectrum(H,sigma=-4.2-.1j,roots=3)


def test_LU_refinement_corrects_solution_without_loosening_gate(monkeypatch):
    from scipy.linalg import lu_solve
    H,_=analytic_matrix()
    calls=[]
    def inaccurate_first(factor,b,**kwargs):
        value=lu_solve(factor,b,**kwargs)
        calls.append(True)
        return value*(1+1e-9) if len(calls)==1 else value
    monkeypatch.setattr('atomx.numerics.complex_eigensolver.linalg.lu_solve',inaccurate_first)
    result=complex_target_spectrum(H,sigma=-4.2-.1j,roots=3)
    assert result.diagnostics['refinement_steps']>=1
    assert result.diagnostics['max_linear_relative_residual']<=1e-12
    assert result.residuals.max()<1e-10
