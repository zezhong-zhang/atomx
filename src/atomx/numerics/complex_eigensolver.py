"""Controlled non-Hermitian target spectrum; no physical pole classification."""
import warnings
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import (
    ArpackNoConvergence,
    LinearOperator,
    aslinearoperator,
    eigs,
    gmres,
)


@dataclass(frozen=True)
class ComplexTargetSpectrum:
    energies: np.ndarray
    coefficients: np.ndarray
    residuals: np.ndarray
    absolute_residuals: np.ndarray
    diagnostics: object


def _positive_integer(value,name):
    if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def validate_complex_target_request(n, *, sigma, roots=4, ncv=None,
                                    tol=1e-11, maxiter=1000, linear_maxiter=100,
                                    linear_restart=60):
    """Validate search/resource controls before physical radial allocation."""
    roots=_positive_integer(roots,'roots')
    if roots>=n-1:
        raise ValueError('target roots must be smaller than dimension-1; use full solve')
    if (isinstance(sigma,bool) or not isinstance(sigma,(int,float,complex,np.number))
            or not np.isfinite(sigma)):
        raise ValueError('sigma must be a finite scalar')
    sigma=complex(sigma)
    if (isinstance(tol,bool) or not isinstance(tol,(int,float,np.integer,np.floating))
            or not np.isfinite(tol) or not 0<tol<=1e-10):
        raise ValueError('tol must be finite and in (0,1e-10]')
    ncv=min(n,max(2*roots+1,20)) if ncv is None else _positive_integer(ncv,'ncv')
    if not roots+2<=ncv<=n:
        raise ValueError('ncv must be between roots+2 and dimension')
    maxiter=_positive_integer(maxiter,'maxiter')
    linear_maxiter=_positive_integer(linear_maxiter,'linear_maxiter')
    restart=min(n,_positive_integer(linear_restart,'linear_restart'))
    return dict(sigma=sigma,roots=roots,ncv=ncv,tol=float(tol),maxiter=maxiter,
                linear_maxiter=linear_maxiter,linear_restart=restart)


def complex_target_spectrum(operator, *, sigma, roots=4, ncv=None,
                            tol=1e-11, maxiter=1000, linear_maxiter=100,
                            linear_restart=60):
    """Arnoldi shift-invert on a complete complex-symmetric operator.

    A LinearOperator requires ``diagonal()`` for Jacobi/GMRES. An explicit
    ndarray instead uses dense LU of H-sigma I; this is not matrix-free.
    Neither route is a Hermitian eigensolver. Inverses independently check
    original-system relative residual <=1e-12, never a hidden default solve.
    Returned eigenpairs also pass the original-operator pair residual gate
    1e-10 and c-product self/orthogonality gates. All requested roots must
    converge; no partial ARPACK exception or dense fallback is accepted.

    Selection is by distance to the numerical shift, not by experimental
    fit or resonance identity. Repeated/exceptional subspaces that cannot
    satisfy scalar c-normalization are rejected, not arbitrarily mixed.
    No Hamiltonian is constructed here. Supplied operators may own sparse
    determinant tables/projection; supplied dense matrices retain N² storage.
    """
    matrix=None
    if isinstance(operator,np.ndarray):
        matrix=np.asarray(operator,dtype=complex)
        if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError('finite square Hamiltonian matrix required')
        operator=aslinearoperator(matrix)
        operator.diagonal=lambda:np.diag(matrix).copy()
    if not isinstance(operator,LinearOperator) or operator.shape[0]!=operator.shape[1]:
        raise ValueError('square LinearOperator or ndarray required')
    if not hasattr(operator,'diagonal'):
        raise ValueError('operator requires diagonal preconditioner data')
    n=operator.shape[0]
    options=validate_complex_target_request(n,sigma=sigma,roots=roots,ncv=ncv,
        tol=tol,maxiter=maxiter,linear_maxiter=linear_maxiter,linear_restart=linear_restart)
    sigma,roots,ncv=options['sigma'],options['roots'],options['ncv']
    restart=options['linear_restart']
    diagonal=np.asarray(operator.diagonal(),dtype=complex)
    if diagonal.shape!=(n,) or not np.isfinite(diagonal).all():
        raise ValueError('operator diagonal must be a finite vector')
    shifted=LinearOperator((n,n),matvec=lambda x:operator@x-sigma*x,dtype=complex)
    divisor=diagonal-sigma
    # Only the preconditioner changes at an exact zero; H itself never does.
    divisor=np.where(divisor==0,1,divisor)
    preconditioner=LinearOperator((n,n),matvec=lambda x:x/divisor,dtype=complex)
    factors=None
    if matrix is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error',linalg.LinAlgWarning)
                factors=linalg.lu_factor(matrix-sigma*np.eye(n),check_finite=True)
        except (linalg.LinAlgWarning,linalg.LinAlgError) as error:
            raise ValueError('target shifted LU factorization failed') from error
    counts={'inverse_calls':0,'linear_iterations':0,'refinement_steps':0,
            'max_linear_relative_residual':0.}

    def iteration(_residual):
        counts['linear_iterations']+=1

    def inverse(rhs):
        counts['inverse_calls']+=1
        rhs_norm=np.linalg.norm(rhs)
        if rhs_norm==0:
            return np.zeros(n,dtype=complex)
        if factors is None:
            value,info=gmres(shifted,rhs,M=preconditioner,rtol=1e-12,atol=0,
                             restart=restart,maxiter=linear_maxiter,
                             callback=iteration,callback_type='pr_norm')
        else:
            value,info=linalg.lu_solve(factors,rhs,check_finite=True),0
        error=np.linalg.norm(shifted@value-rhs)/rhs_norm
        # Bounded original-system iterative refinement of a direct factor.
        # It corrects the solution, never the Hamiltonian or the 1e-12 gate.
        if factors is not None:
            for _ in range(2):
                if np.isfinite(error) and error<=1e-12:
                    break
                value+=linalg.lu_solve(factors,rhs-shifted@value,check_finite=True)
                counts['refinement_steps']+=1
                error=np.linalg.norm(shifted@value-rhs)/rhs_norm
        if info!=0 or not np.isfinite(error) or error>1e-12:
            raise ValueError(f'target inverse linear residual gate failed: info={info}, residual={error:g}')
        counts['max_linear_relative_residual']=max(counts['max_linear_relative_residual'],float(error))
        return value

    inverse_operator=LinearOperator((n,n),matvec=inverse,dtype=complex)
    # Force complex Arnoldi even for an explicitly real operator at theta=0:
    # real ARPACK's OPpart interpretation is a different shift transform.
    complex_operator=LinearOperator((n,n),matvec=lambda x:operator@x,
                                     matmat=lambda x:operator@x,dtype=complex)
    rng=np.random.default_rng(0)
    v0=rng.standard_normal(n)+1j*rng.standard_normal(n)
    try:
        energies,vectors=eigs(complex_operator,k=roots,sigma=sigma,which='LM',
                             OPinv=inverse_operator,v0=v0,ncv=ncv,tol=tol,maxiter=maxiter)
    except ArpackNoConvergence as error:
        raise ValueError('target Arnoldi did not converge all requested roots') from error
    if not np.isfinite(energies).all() or not np.isfinite(vectors).all():
        raise ValueError('nonfinite targeted eigensystem')
    order=np.argsort(abs(energies-sigma),kind='stable')
    energies,vectors=energies[order],vectors[:,order]
    norms=np.sum(vectors*vectors,axis=0)
    if np.any(abs(norms)<=1e-10*np.sum(abs(vectors)**2,axis=0)):
        raise ValueError('self-orthogonal targeted root: exceptional subspace treatment required')
    vectors=vectors/np.sqrt(norms)[None,:]
    gram_error=float(abs(vectors.T@vectors-np.eye(roots)).max())
    if gram_error>1e-8:
        raise ValueError('target c-product orthogonality gate failed: degenerate subspace treatment required')
    action=operator@vectors
    absolute=np.linalg.norm(action-vectors*energies,axis=0)
    scale=np.linalg.norm(action,axis=0)+abs(energies)*np.linalg.norm(vectors,axis=0)
    residuals=absolute/np.where(scale==0,1,scale)
    if not np.isfinite(residuals).all() or np.any(residuals>1e-10):
        raise ValueError('target original-operator eigenpair residual gate failed')
    for array in (energies,vectors,residuals,absolute):
        array.setflags(write=False)
    backend='GMRES' if factors is None else 'dense LU'
    diagnostics=MappingProxyType({**counts,'algorithm':f'non-Hermitian Arnoldi / {backend} shift-invert',
        'ncv':ncv,'roots':roots,'dimension':n,'c_gram_error':gram_error,
        'linear_rtol':1e-12,'eigenpair_residual_gate':1e-10,
        'full_spectrum':False,'dense_hamiltonian_built_by_solver':False,
        'matrix_free':matrix is None,'dense_matrix_input':matrix is not None,
        'dense_shift_factorization':factors is not None})
    return ComplexTargetSpectrum(energies,vectors,residuals,absolute,diagnostics)


__all__=['ComplexTargetSpectrum','complex_target_spectrum','validate_complex_target_request']
