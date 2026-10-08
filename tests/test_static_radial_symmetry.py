"""Exact permutation reduction against unreduced signed CI contractions."""
from itertools import product

import numpy as np
import pytest

from atomx.angular.csf_hamiltonian import (
    block_terms,
    breit_block_terms,
    csf_signature,
    evaluate_block,
    exact_block_matrix,
)
from atomx.config.csf import enumerate_jj_orbital_space
from atomx.methods.csf.complex_ci import ComplexScaledCIModel
from atomx.numerics.bspline.knots import make_exponential_knots
from atomx.physics.csf.radial_symmetry import canonical_static_radial_key


def permutations(a, c, b, d):
    for reverse_a, reverse_b, exchange in product((False, True), repeat=3):
        p = (c, a) if reverse_a else (a, c)
        q = (d, b) if reverse_b else (b, d)
        yield (*q, *p) if exchange else (*p, *q), (-1)**(reverse_a+reverse_b)


def analytic_integral(key):
    # Independent closed integral of polynomial densities on [0,1]:
    # int r^p s^q r_<^L/r_>^(L+1) dr ds. Components have complex
    # amplitudes, so accidental conjugation cannot pass this oracle.
    def amplitude(a):
        return (a[-2]+2)*.03+1j*(a[-1]+4)*.02
    def power(a):
        return a[-2]+abs(a[-1])+2
    if key[0] == 'h':
        _, a, kappa, c = key
        return (a*c+kappa)*.07+1j*(a+c)*.03
    kind, rank, a, c, b, d = key
    p, q = power(a)+power(c), power(b)+power(d)
    factor = amplitude(a)*amplitude(c)*amplitude(b)*amplitude(d)
    if kind == 't':
        # Antisymmetric polynomial transition currents, reciprocal kernel.
        factor *= (power(c)-power(a))*(power(d)-power(b))
        p, q = p-1, q-1
    return factor*(1/(p+rank+1)+1/(q+rank+1))/(p+q+1)


@pytest.mark.parametrize('kind', ['r', 'g', 't'])
def test_eight_permutations_against_independent_polynomial_integral(kind):
    labels = ((1,-1),(2,1),(3,-2),(4,1))
    if kind == 'g':
        labels = tuple((component,*a) for component,a in zip('PQPQ',labels, strict=True))
    original = (kind,2,*labels)
    value = analytic_integral(original)
    representative, _ = canonical_static_radial_key(original)
    for indices, current_sign in permutations(*labels):
        key = (kind,2,*indices)
        expected = value*(current_sign if kind=='t' else 1)
        np.testing.assert_allclose(analytic_integral(key),expected,rtol=2e-15,atol=0)
        canonical, sign = canonical_static_radial_key(key)
        assert canonical == representative
        np.testing.assert_allclose(sign*analytic_integral(canonical),expected,rtol=2e-15,atol=0)
    if kind == 't':
        key = (kind,2,labels[0],labels[0],*labels[2:])
        assert analytic_integral(key) == 0
        assert canonical_static_radial_key(key)[1] == 0


@pytest.mark.parametrize('N,J2,parity', [(2,0,1),(2,2,-1),(3,1,1),(3,3,-1)])
@pytest.mark.parametrize('complex_values', [False,True])
def test_full_static_CI_and_adjoint_equal_unreduced(N,J2,parity,complex_values):
    csfs = [c for c in enumerate_jj_orbital_space(((1,-1),(2,-1),(2,1),(2,-2)),N)
            if (c.total_J2,c.parity)==(J2,parity)]
    sigs = [csf_signature(c) for c in csfs]
    evaluate = analytic_integral if complex_values else lambda key: analytic_integral(key).real
    dtype = complex if complex_values else float
    args = dict(breit_evaluator=evaluate,integral_dtype=dtype)
    raw = exact_block_matrix(sigs,evaluate,**args)
    reduced = exact_block_matrix(sigs,evaluate,radial_symmetry=True,**args)
    np.testing.assert_allclose(reduced,raw,rtol=0,atol=2e-14)
    x = np.arange(len(sigs))+.3j
    operator = exact_block_matrix(sigs,evaluate,radial_symmetry=True,as_operator=True,**args)
    np.testing.assert_allclose(operator@x,raw@x,rtol=0,atol=2e-13)
    np.testing.assert_allclose(operator.rmatvec(x),raw.conj().T@x,rtol=0,atol=2e-13)
    np.testing.assert_allclose(operator.diagonal(),np.diag(raw),rtol=0,atol=2e-14)
    reversed_H = exact_block_matrix(sigs[::-1],evaluate,radial_symmetry=True,**args)
    np.testing.assert_allclose(reversed_H,raw[::-1,::-1],rtol=0,atol=2e-14)
    patterns = tuple(sorted({s.pattern for s in sigs}))
    for compiler in (block_terms,breit_block_terms):
        old = compiler(patterns,J2)
        new = compiler(patterns,J2,radial_symmetry=True)
        assert len(new.keys)<len(old.keys)
        assert len(new.pairs)<len(old.pairs)
        assert np.all(new.pairs[:,0]<=new.pairs[:,1])
        # Check before CSF projection/final roundoff symmetrization, which
        # could otherwise hide a wrong transpose assumption or current sign.
        old_determinants = evaluate_block(old,evaluate,integral_dtype=dtype)
        new_determinants = evaluate_block(new,evaluate,integral_dtype=dtype)
        np.testing.assert_allclose(old_determinants,old_determinants.T,rtol=0,atol=2e-14)
        np.testing.assert_allclose(new_determinants,old_determinants,rtol=0,atol=2e-14)


@pytest.mark.parametrize('N,J2,theta', [(2,0,.0),(2,0,.3),(3,1,.2)])
def test_actual_owned_complex_DCB_equal_unreduced(N,J2,theta):
    knots = make_exponential_knots(55,9,1e-4,20)
    block = ComplexScaledCIModel(knots,9,6,N,{-1:3,1:2,-2:2},n_gauss=17,
                                include_breit=True).solve(theta,J2,1)
    sigs = [csf_signature(c) for c in block.csfs]
    args = dict(integral_dtype=complex,breit_evaluator=block.radial.evaluate_breit)
    raw = exact_block_matrix(sigs,block.radial.evaluate,**args)
    new = exact_block_matrix(sigs,block.radial.evaluate,radial_symmetry=True,**args)
    np.testing.assert_allclose(new,raw,rtol=0,atol=1e-11)
    np.testing.assert_allclose((block.coefficients*block.energies)@block.coefficients.T,
                               raw,rtol=0,atol=1e-10)


def test_symmetry_contract_is_explicit_and_preserves_small_terms():
    sigs = [csf_signature(c) for c in enumerate_jj_orbital_space(((1,-1),),2)]
    for value in (None,1,'yes'):
        with pytest.raises(ValueError,match='bool'):
            exact_block_matrix(sigs,lambda k:0,radial_symmetry=value)
    tiny = exact_block_matrix(sigs,lambda k:1e-100 if k[0]=='r' else 0,
                              radial_symmetry=True)
    assert 0<tiny[0,0]<2e-100
    with pytest.raises(ValueError,match='h/r/g/t'):
        canonical_static_radial_key(('dynamic',0,(1,-1),(2,-1),(1,-1),(2,-1)))
