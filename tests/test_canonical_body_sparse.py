"""Independent all-word and spectator controls for sparse signed projection."""

from itertools import combinations
from math import comb

import numpy as np
import pytest

from atomx.numerics.canonical_sparse import canonical_body_sparse
from atomx.numerics.fermion import apply_terms, canonical_body_matrix


@pytest.mark.parametrize('number', range(5))
@pytest.mark.parametrize('row_index', ('colex', 'hash'))
def test_all_complex_directed_entries_against_independent_actions(number, row_index):
    rng = np.random.default_rng(951+number)
    n = 7
    coefficients, terms = [], []
    for rank in range(4):
        tuples = list(combinations(range(n), rank))
        c = rng.normal(size=(len(tuples), len(tuples)))+1j*rng.normal(
            size=(len(tuples), len(tuples)))
        c[rng.random(c.shape) < .7] = 0
        c[0, -1] = 1e-25+2e-25j
        coefficients.append(c)
        for i, creators in enumerate(tuples):
            for j, annihilators in enumerate(tuples):
                if c[i, j] != 0:
                    operations = (tuple((bit, True) for bit in creators)
                                  +tuple((bit, False) for bit in reversed(annihilators)))
                    terms.append((c[i, j], operations))
    masks = [sum(1 << bit for bit in row) for row in combinations(range(n), number)]
    masks = masks[::-1]  # The declared order is not canonical or sorted.
    actual = canonical_body_sparse(masks, coefficients, n, row_index=row_index).toarray()
    expected = np.zeros_like(actual)
    for ket, mask in enumerate(masks):
        action = apply_terms(terms, {mask: 1})
        for bra, left in enumerate(masks):
            expected[bra, ket] = action.get(left, 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual, canonical_body_matrix(
        masks, coefficients, n), rtol=1e-13, atol=1e-13)
    assert np.max(abs(actual-actual.conj().T)) > 0


@pytest.mark.parametrize('row_index', ('colex', 'hash'))
def test_cross_tile_colex_selected_space_and_tiny_real_values(row_index):
    n = 21
    masks = [sum(1 << bit for bit in row) for row in [(0, 10), (10, 20), (0, 20)]]
    one = np.zeros((n, n))
    one[0, 20] = -2e-25
    one[20, 0] = 3e-25
    coefficients = (np.zeros((1, 1)), one)
    actual = canonical_body_sparse(masks, coefficients, n, row_index=row_index).toarray()
    np.testing.assert_array_equal(actual, canonical_body_matrix(masks, coefficients, n))
    assert actual[0, 1] != 0 and actual[1, 0] != 0


def test_integer_coefficients_accumulate_in_floating_storage():
    n = 4
    coefficients = (np.array([[120]], dtype=np.int8), np.full((n, n), 120, dtype=np.int8))
    masks = [3, 5, 6]
    actual = canonical_body_sparse(masks, coefficients, n).toarray()
    assert actual.dtype == np.float64 and actual[0, 0] == 360
    np.testing.assert_array_equal(actual, canonical_body_matrix(masks, coefficients, n))


def test_exact_zero_output_and_admission_failures():
    coefficients = (np.zeros((1, 1)), np.zeros((8, 8)), np.eye(comb(8, 2)))
    masks = [3, 5, 6]
    with pytest.raises(MemoryError, match='candidate work'):
        canonical_body_sparse(masks, coefficients, 8, max_work=1)
    with pytest.raises(MemoryError, match='index/conversion'):
        canonical_body_sparse(masks, coefficients, 8, memory_bytes=1024)
    with pytest.raises(MemoryError, match='index/conversion'):
        canonical_body_sparse([(1 << 31)-1], (np.zeros((1, 1)),), 63)
    zero = canonical_body_sparse(masks, (np.zeros((1, 1)),), 8)
    assert zero.nnz == 0
    with pytest.raises(ValueError, match='fixed electron'):
        canonical_body_sparse([1, 3], coefficients, 8)


def test_counted_output_requires_separate_admission():
    coefficients = tuple(np.ones((comb(4, r), comb(4, r))) for r in range(3))
    masks = [sum(1 << bit for bit in row) for row in combinations(range(4), 2)]
    with pytest.raises(MemoryError, match='counted canonical sparse matrix'):
        canonical_body_sparse(masks, coefficients, 4, memory_bytes=61000)
    full = canonical_body_sparse(masks, coefficients, 4)
    np.testing.assert_array_equal(full.toarray(), canonical_body_matrix(masks, coefficients, 4))


def test_hash_admits_small_declared_space_with_enormous_full_N_sector():
    n, number = 40, 18
    masks = [(1 << number)-1, ((1 << number)-1) ^ (1 << 17) ^ (1 << 39)]
    one = np.diag(np.arange(n, dtype=float))
    one = one.astype(complex)
    one[39, 17], one[17, 39] = .3+.2j, -.1+.4j
    coefficients = (np.array([[1.25]]), one)
    with pytest.raises(MemoryError, match='index/conversion'):
        canonical_body_sparse(masks, coefficients, n, memory_bytes=2*1024**2)
    actual = canonical_body_sparse(
        masks, coefficients, n, memory_bytes=2*1024**2, row_index='hash').toarray()
    terms = [(1.25, ())] + [
        (one[p, q], ((p, True), (q, False)))
        for p in range(n) for q in range(n) if one[p, q] != 0]
    expected = np.array([[apply_terms(terms, {right: 1}).get(left, 0)
                          for right in masks] for left in masks])
    np.testing.assert_array_equal(actual, expected)


def test_row_index_rejects_unknown_mode_and_preserves_exact_CSC_bytes():
    masks = [27, 15, 30, 23, 29]
    rng = np.random.default_rng(20261016)
    coefficients = tuple(rng.normal(size=(comb(5, r), comb(5, r)))
                         for r in range(4))
    old = canonical_body_sparse(masks, coefficients, 5)
    new = canonical_body_sparse(masks, coefficients, 5, row_index='hash')
    for name in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
    with pytest.raises(ValueError, match='row_index'):
        canonical_body_sparse(masks, coefficients, 5, row_index='approximate')


def test_large_CSC_index_admission_matches_actual_scipy_index_rule():
    from scipy.sparse._sputils import get_index_dtype

    from atomx.numerics.canonical_sparse import _csc_row_dtype

    # No billion-entry allocation: independently ask the installed consumer
    # which width a valid CSC indptr with each terminal count requires.
    for count in (0, 507892914, 2**31-1, 2**31, 3_000_000_000):
        scipy_dtype = get_index_dtype(
            arrays=(np.array([0, count], dtype=np.int64),),
            maxval=count, check_contents=True)
        assert np.dtype(_csc_row_dtype(count)) == np.dtype(scipy_dtype)


def test_population_prefilter_retains_every_directed_word_in_selected_space():
    from atomx.numerics.canonical_sparse import _population_prefix

    n = 12
    masks = [sum(1 << i for i in row) for row in combinations(range(n), 4)
             if sum(i < 4 for i in row) >= 3][::-1]
    prefix, support = _population_prefix(np.array(masks, dtype=np.uint64), n, 4)
    assert int(prefix) == 15 and int(support) == (1 << 3) | (1 << 4)
    rng = np.random.default_rng(719)
    coefficients, terms = [], []
    for rank in range(4):
        tuples = list(combinations(range(n), rank))
        values = np.zeros((len(tuples), len(tuples)), dtype=complex)
        for _unused in range(24):
            i, j = rng.integers(len(tuples), size=2)
            values[i, j] = rng.normal()+1j*rng.normal()
        values[0, -1] = 1e-25+2e-25j
        coefficients.append(values)
        for i, j in zip(*np.nonzero(values), strict=True):
            ops = (tuple((p, True) for p in tuples[i])
                   +tuple((p, False) for p in reversed(tuples[j])))
            terms.append((values[i, j], ops))
    actual = canonical_body_sparse(masks, coefficients, n, row_index='hash')
    old = canonical_body_sparse(masks, coefficients, n)
    for key in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(actual, key), getattr(old, key))
    for j, mask in enumerate(masks):
        independent = apply_terms(terms, {mask: 1})
        np.testing.assert_allclose(actual.toarray()[:, j],
                                   [independent.get(bra, 0) for bra in masks],
                                   rtol=1e-13, atol=1e-13)
