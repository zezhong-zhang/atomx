"""Batched Slater contractions against an independent dense Green matrix."""
import numba
import numpy as np
import pytest

from atomx.numerics.quadrature import yk_pair_dot_batch_on_grid


@pytest.mark.parametrize('cut', [0, 41, 80])
def test_batch_matches_symmetric_coulomb_green_quadrature(cut):
    rng = np.random.default_rng(8217)
    r = np.geomspace(.013, 9., 81)
    pa, qa, pb, qb = [rng.normal(size=(4, len(r))) * np.exp(-r/4)
                      for _ in range(4)]
    a, b = np.array([0, 2, 1, 3]), np.array([1, 0, 2, 0])
    ranks = np.array([0, 1, 3, 2])
    valid = np.array([True, True, False, True])
    ptr, consumers = np.array([0, 2, 2, 5, 6]), np.array([0, 2, 1, 0, 3, 1])
    outer = rng.normal(size=(4, cut+1))
    # Independent dense representation of the symmetric Coulomb Green
    # function, with true trapezoidal endpoint weights for the source.
    weights = np.gradient(r)
    weights[[0, -1]] *= .5
    lo = np.minimum(r[:, None], r[None, :])
    hi = np.maximum(r[:, None], r[None, :])
    expected = np.zeros(len(consumers))
    for row in range(len(a)):
        if not valid[row]:
            continue
        density = pa[a[row]]*pb[b[row]] + qa[a[row]]*qb[b[row]]
        potential = (lo**ranks[row]/hi**(ranks[row]+1)) @ (weights*density)
        for slot in range(ptr[row], ptr[row+1]):
            expected[slot] = outer[consumers[slot]] @ potential[:cut+1]
    initial_threads = numba.get_num_threads()
    outputs = []
    try:
        for threads in [1, min(4, numba.config.NUMBA_NUM_THREADS)]:
            numba.set_num_threads(threads)
            value = yk_pair_dot_batch_on_grid(pa, qa, pb, qb, a, b, ranks,
                valid, outer, ptr, consumers, r, i_cut=cut)
            np.testing.assert_allclose(value, expected, rtol=3e-13, atol=2e-14)
            outputs.append(value)
    finally:
        numba.set_num_threads(initial_threads)
    np.testing.assert_array_equal(outputs[0].view(np.uint64), outputs[1].view(np.uint64))
