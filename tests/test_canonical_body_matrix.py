"""Independent exhaustive signed action, including complex three-body terms."""
from itertools import combinations

import numpy as np
import pytest

from atomx.numerics.fermion import apply_terms, canonical_body_matrix


@pytest.mark.parametrize('number', range(5))
def test_every_canonical_coefficient_against_independent_all_word_action(number):
    rng = np.random.default_rng(73942)
    n = 6
    subsets = [list(combinations(range(n), r)) for r in range(4)]
    coefs = tuple(rng.normal(size=(len(s), len(s)))+1j*rng.normal(size=(len(s), len(s)))
                  for s in subsets)
    masks = [sum(1 << i for i in row) for row in combinations(range(n), number)]
    terms = []
    for rank, c in enumerate(coefs):
        for i, left in enumerate(subsets[rank]):
            for j, right in enumerate(subsets[rank]):
                terms.append((c[i, j], tuple((p, True) for p in left)
                              +tuple((p, False) for p in reversed(right))))
    expected = np.zeros((len(masks), len(masks)), complex)
    for ket, mask in enumerate(masks):
        action = apply_terms(terms, {mask: 1.})
        for bra, target in enumerate(masks):
            expected[bra, ket] = action.get(target, 0.)
    actual = canonical_body_matrix(masks, coefs, n)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-13)
    if number:
        assert np.max(abs(actual-actual.conj().T)) > 1


def test_selected_space_preserves_phases_and_every_spectator():
    n = 7
    rng = np.random.default_rng(640)
    s = [list(combinations(range(n), r)) for r in range(4)]
    c = tuple(rng.normal(size=(len(a), len(a))) for a in s)
    masks = [sum(1 << i for i in a) for a in combinations(range(n), 4)]
    full = canonical_body_matrix(masks, c, n)
    order = [19, 4, 27, 0]
    partial = canonical_body_matrix([masks[i] for i in order], c, n)
    np.testing.assert_array_equal(partial, full[np.ix_(order, order)])
    with pytest.raises(MemoryError):
        canonical_body_matrix(masks, c, n, max_work=1)
    with pytest.raises(MemoryError):
        canonical_body_matrix(masks, c, n, memory_bytes=1)


@pytest.mark.parametrize('masks', [[1, 1], [1, 3], [-1], [64], [True]])
def test_malformed_spaces_reject(masks):
    with pytest.raises(ValueError):
        canonical_body_matrix(masks, (np.zeros((1, 1)),), 6)
