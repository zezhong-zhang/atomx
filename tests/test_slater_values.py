"""Independent complete second-quantized products, with no Slater case split."""
from itertools import combinations

import numpy as np
import pytest

from atomx.numerics.slater_values import slater_values


def _act(occupied, operators):
    occupied, sign = list(occupied), 1
    for site, creating in operators:
        position = sum(i < site for i in occupied)
        if (site in occupied) == creating:
            return None
        sign *= (-1)**position
        if creating:
            occupied.insert(position, site)
        else:
            occupied.pop(position)
    return tuple(occupied), sign


@pytest.mark.parametrize('electrons', [1, 2, 3, 4])
def test_complete_graph_against_all_creation_annihilation_products(electrons):
    n = 5
    h = np.arange(n*n, dtype=float).reshape(n, n)/19
    h = h+h.T
    pairs = tuple(combinations(range(n), 2))
    x = np.arange(len(pairs)**2, dtype=float).reshape(len(pairs), -1)/73
    x = x+x.T
    g = np.zeros((n, n, n, n))
    for i, (p, q) in enumerate(pairs):
        for j, (r, s) in enumerate(pairs):
            g[p,q,r,s] = g[q,p,s,r] = x[i,j]
            g[q,p,r,s] = g[p,q,s,r] = -x[i,j]
    occupied = tuple(combinations(range(n), electrons))
    masks = np.asarray([sum(1 << i for i in row) for row in occupied], dtype=np.uint64)
    indices = {o: i for i, o in enumerate(occupied)}
    expected = np.zeros((len(masks), len(masks)))
    for col, source in enumerate(occupied):
        for p in range(n):
            for q in range(n):
                acted = _act(source, [(q, False), (p, True)])
                if acted is not None:
                    expected[indices[acted[0]], col] += acted[1]*h[p,q]
                for r in range(n):
                    for s in range(n):
                        acted = _act(source, [(r, False), (s, False), (q, True), (p, True)])
                        if acted is not None:
                            expected[indices[acted[0]], col] += .25*acted[1]*g[p,q,r,s]
    size = len(masks)
    ptr = np.arange(0, size*size+1, size, dtype=np.int64)
    cols = np.tile(np.arange(size, dtype=np.int64), size)
    actual = slater_values(masks, h, g, ptr, cols).reshape(size, size)
    np.testing.assert_allclose(actual, expected, atol=5e-14, rtol=5e-14)
