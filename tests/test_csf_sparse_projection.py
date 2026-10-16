"""Sparse exact projection preserves global orthogonality and J-span guards."""
import itertools

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from atomx.angular import csf_hamiltonian as ci
from atomx.config.csf import enumerate_jj_orbital_space


def _groups():
    csfs = enumerate_jj_orbital_space(((1, -1), (2, -1), (2, 1)), 2,
                                     magnetic_state_cap=100)
    sigs = [ci.csf_signature(c) for c in csfs if c.total_J2 == 0]
    patterns = sorted({s.pattern for s in sigs})
    groups = tuple(tuple(s for s in sigs if s.pattern == p) for p in patterns)
    return patterns, groups, sigs


def test_dense_compatibility_and_exact_sparse_coefficients():
    patterns, groups, _ = _groups()
    dense = ci.csf_block_matrix(patterns, 0, groups)
    sparse = ci.csf_block_matrix(patterns, 0, groups, sparse=True)
    assert isinstance(dense, np.ndarray) and issparse(sparse)
    np.testing.assert_array_equal(dense, sparse.toarray())
    ci.verify_csf_span(patterns, 0, sparse)
    with pytest.raises(ValueError, match='sparse must be bool'):
        ci.csf_block_matrix(patterns, 0, groups, sparse=1)


@pytest.mark.parametrize('sparse', [False, True])
def test_global_cross_configuration_orthogonality_is_still_checked(sparse):
    patterns, groups, _ = _groups()
    matrix = ci.csf_block_matrix(patterns, 0, groups).copy()
    # A normalized column transplanted from a different configuration must
    # not pass just because every local configuration has a valid J basis.
    matrix[:, -1] = matrix[:, 0]
    with pytest.raises(ci.ExactCISpaceError, match='not orthonormal'):
        ci.verify_csf_span(patterns, 0, csr_matrix(matrix) if sparse else matrix)


@pytest.mark.parametrize('sparse', [False, True])
def test_normalized_wrong_J_vector_is_still_rejected(sparse):
    pattern = ((1, -1, 1), (2, -1, 1))
    # One normalized M=0 determinant spans neither the required singlet nor
    # the kernel of J_+. Its norm alone must not certify J=0.
    matrix = np.array([[1.], [0.]])
    with pytest.raises(ci.ExactCISpaceError, match='not an eigenstate of J'):
        ci.verify_csf_span([pattern], 0, csr_matrix(matrix) if sparse else matrix)


def test_exact_assembly_does_not_materialize_dense_C(monkeypatch):
    _, _, sigs = _groups()
    ci._verify_cached.cache_clear()

    def dense_forbidden(*args, **kwargs):
        raise AssertionError('dense CSF projection requested')

    monkeypatch.setattr(ci, '_block_csf_matrix', dense_forbidden)
    matrix = ci.exact_block_matrix(sigs, lambda key: 0.2)
    assert matrix.shape == (len(sigs), len(sigs))
    assert np.isfinite(matrix).all()


@pytest.mark.parametrize('electrons', [0, 1, 2, 3, 4])
@pytest.mark.parametrize('triangular', [False, True])
@pytest.mark.parametrize('frozen', [False, True])
def test_encoded_connections_match_full_slater_condon_graph(electrons, triangular, frozen):
    dets = [sum(1 << i for i in occupied)
            for occupied in itertools.combinations(range(1, 9), electrons)]
    if frozen:
        dets = [mask | 1 for mask in dets]
    expected = [(i, j) for i, a in enumerate(dets) for j, b in enumerate(dets)
                if (a ^ b).bit_count() <= 4 and (not triangular or i <= j)]
    assert list(ci._connected_pairs(dets, triangular=triangular)) == expected
