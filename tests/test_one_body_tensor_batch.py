"""Every signed batch entry versus the original independent pair contraction."""
from unittest.mock import patch

import numpy as np
import pytest

from atomx.angular.csf_hamiltonian import csf_signature, one_body_tensor_terms
from atomx.angular.one_body_tensor import TensorProjectionCache, one_body_tensor_matrix
from atomx.config.csf import enumerate_jj_csfs, enumerate_jj_orbital_space


def _elements(blocks):
    keys = sorted({(n, k) for s in blocks for n, k, _ in s.pattern})
    return {(a, b): complex((i+1)/7, -(j+1)/11)
            for i, a in enumerate(keys) for j, b in enumerate(keys)}


def _oracle(bra, ket, rank, elements):
    return np.asarray([[sum(c*elements.get((a, b), 0.)
                             for a, b, c in one_body_tensor_terms(x, y, rank))
                        for y in ket] for x in bra], dtype=complex)


@pytest.mark.parametrize('jf,ji', [(1, 1), (1, 3), (3, 1), (3, 3), (5, 3)])
def test_complete_three_electron_signed_matrix(jf, ji):
    csfs = enumerate_jj_orbital_space(((2, -1), (2, 1), (2, -2)), 3)
    bra = tuple(csf_signature(c) for c in csfs if c.total_J2 == jf)
    ket = tuple(csf_signature(c) for c in csfs if c.total_J2 == ji)
    elements = _elements(bra+ket)
    expected = _oracle(bra, ket, 1, elements)
    actual = one_body_tensor_matrix(bra, ket, 1, elements)
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=3e-14)
    permutation = list(reversed(range(len(bra))))
    reordered = one_body_tensor_matrix([bra[i] for i in permutation], ket, 1, elements)
    np.testing.assert_allclose(reordered, expected[permutation], atol=3e-14, rtol=3e-14)
    tiny = one_body_tensor_matrix(bra, ket, 1, {k: v*1e-80 for k, v in elements.items()})
    np.testing.assert_allclose(tiny, expected*1e-80, atol=1e-94, rtol=3e-14)


def test_degenerate_parentages_and_rank_two():
    csfs = enumerate_jj_csfs('4f4')
    block = tuple(csf_signature(c) for c in csfs if c.total_J2 == 4)
    elements = _elements(block)
    np.testing.assert_allclose(one_body_tensor_matrix(block, block, 2, elements),
                               _oracle(block, block, 2, elements), atol=1e-13, rtol=1e-13)


def test_full_frozen_core_masks_above_64_bits():
    prefix = '1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 4f14 5s2 '
    block = tuple(csf_signature(c) for c in enumerate_jj_csfs(prefix+'5p1') if c.total_J2 == 3)
    elements = _elements(block)
    np.testing.assert_allclose(one_body_tensor_matrix(block, block, 1, elements),
                               _oracle(block, block, 1, elements), atol=1e-13, rtol=1e-13)


def test_caps_reject_before_vector_allocation():
    block = tuple(csf_signature(c) for c in enumerate_jj_csfs('4f4') if c.total_J2 == 4)
    elements = _elements(block)
    with patch('atomx.angular.one_body_tensor.csf_magnetic_vector', side_effect=AssertionError):
        with pytest.raises(ValueError, match='determinant'):
            one_body_tensor_matrix(block, block, 2, elements, max_determinants=1)
        with pytest.raises(MemoryError):
            one_body_tensor_matrix(block, block, 2, elements, memory_bytes=1)
        with pytest.raises(ValueError, match='distinct'):
            one_body_tensor_matrix(block+block, block, 2, elements)


def test_invalid_operator_and_sector_reject():
    block = tuple(csf_signature(c) for c in enumerate_jj_csfs('2p1') if c.total_J2 == 1)
    with pytest.raises(ValueError, match='finite'):
        one_body_tensor_matrix(block, block, 1, {((2, 1), (2, 1)): float('nan')})
    with pytest.raises(ValueError, match='outside'):
        one_body_tensor_matrix(block, block, 1, {((3, 1), (2, 1)): 1.})


@pytest.mark.parametrize('which', ['both', 'bra', 'ket'])
def test_early_complex_projection_preserves_complete_csf_contraction(which):
    csfs = enumerate_jj_orbital_space(((2, -1), (2, 1), (2, -2)), 3)
    bra = tuple(csf_signature(c) for c in csfs if c.total_J2 == 3)
    ket = tuple(csf_signature(c) for c in csfs if c.total_J2 == 1)
    elements = _elements(bra+ket)
    b = np.arange(len(bra)*2).reshape(-1,2)/17+1j/23
    k = np.arange(len(ket)*3).reshape(-1,3)/29-1j/31
    raw = _oracle(bra, ket, 1, elements)
    expected = b.conj().T@raw if which != 'ket' else raw
    expected = expected@k if which != 'bra' else expected
    actual = one_body_tensor_matrix(bra, ket, 1, elements,
                                   bra_coefficients=b if which != 'ket' else None,
                                   ket_coefficients=k if which != 'bra' else None)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize('projected', [False, True])
def test_scoped_projection_reuse_preserves_changed_operator_and_signed_coefficients(projected):
    csfs = enumerate_jj_orbital_space(((2, -1), (2, 1), (2, -2)), 3)
    bra = tuple(csf_signature(c) for c in csfs if c.total_J2 == 3)
    ket = tuple(csf_signature(c) for c in csfs if c.total_J2 == 1)
    elements = _elements(bra+ket)
    b = np.arange(len(bra)*2).reshape(-1,2)/17+1j/23
    k = np.arange(len(ket)*3).reshape(-1,3)/29-1j/31
    kwargs = dict(bra_coefficients=b, ket_coefficients=k) if projected else {}
    raw = _oracle(bra, ket, 1, elements)
    expected = b.conj().T@raw@k if projected else raw
    cache = TensorProjectionCache(1024**2)
    actual = one_body_tensor_matrix(bra, ket, 1, elements, projection_cache=cache, **kwargs)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
    with patch('atomx.angular.one_body_tensor.csf_magnetic_vector', side_effect=AssertionError):
        changed = one_body_tensor_matrix(bra, ket, 1, {p:2*v for p,v in elements.items()},
                                        projection_cache=cache, **kwargs)
    np.testing.assert_allclose(changed, 2*expected, atol=1e-13, rtol=1e-13)
    assert cache.diagnostics()['hits'] == 2
    assert cache.diagnostics()['accounted_resident_bytes'] <= cache.memory_bytes
    if projected:
        b[0,0] *= -1
        changed = one_body_tensor_matrix(bra, ket, 1, elements, projection_cache=cache, **kwargs)
        np.testing.assert_allclose(changed, b.conj().T@raw@k, atol=1e-13, rtol=1e-13)
        assert cache.diagnostics()['misses'] == 3


def test_projection_cache_budget_cannot_change_the_operator():
    block = tuple(csf_signature(c) for c in enumerate_jj_csfs('2p1') if c.total_J2 == 1)
    elements = _elements(block)
    cache = TensorProjectionCache(1)
    np.testing.assert_allclose(one_body_tensor_matrix(block, block, 1, elements, projection_cache=cache),
                               _oracle(block, block, 1, elements), atol=1e-14)
    assert cache.diagnostics()['entries'] == cache.diagnostics()['accounted_resident_bytes'] == 0
    with pytest.raises(ValueError, match='budget'):
        TensorProjectionCache(True)
