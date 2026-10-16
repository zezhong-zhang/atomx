"""Single-continuum action against independent full second quantization."""

from itertools import combinations, product

import numpy as np
import pytest

from atomx.numerics.continuum_actions import ContinuumFockAction


def operate(occupied, operations):
    state, sign = list(occupied), 1
    for index, create in operations:
        if (index in state) == create:
            return None, 0
        sign *= (-1)**sum(i < index for i in state)
        if create:
            state.append(index)
            state.sort()
        else:
            state.remove(index)
    return tuple(state), sign


@pytest.mark.parametrize('number', [1, 2, 3, 4])
def test_continuum_action_matches_full_unscreened_fock_operator(number):
    n = 5
    initial, final = list(combinations(range(n), number)), list(combinations(range(n), number-1))
    rng = np.random.default_rng(9104+number)
    ci, cf = rng.normal(size=len(initial)), rng.normal(size=len(final))
    ci /= np.linalg.norm(ci)
    cf /= np.linalg.norm(cf)
    g = rng.normal(size=(n, n, n))
    g -= g.swapaxes(1, 2)
    h, core = rng.normal(size=n), rng.normal(size=n)
    action = ContinuumFockAction([sum(1 << i for i in c) for c in initial], ci,
                                [sum(1 << i for i in c) for c in final], cf, n)
    actual = action.amplitudes(h, core, g)
    full = np.zeros((n+1,)*4)
    for p, q, r in product(range(n), repeat=3):
        full[n, p, q, r] = g[p, q, r]
        full[p, n, q, r] = -g[p, q, r]
        full[q, r, n, p] = g[p, q, r]
        full[q, r, p, n] = -g[p, q, r]
    expected_h, expected_v = 0., 0.
    for ket, wi in zip(initial, ci, strict=True):
        for final_occupied, wf in zip(final, cf, strict=True):
            bra = final_occupied+(n,)
            for q in range(n):
                out, sign = operate(ket, [(q, False), (n, True)])
                if out == bra:
                    expected_h += wi*wf*sign*h[q]
                    expected_v += wi*wf*sign*core[q]
            for a, b, c, d in zip(*np.nonzero(full), strict=True):
                out, sign = operate(ket, [(c, False), (d, False), (b, True), (a, True)])
                if out == bra:
                    expected_v += wi*wf*sign*full[a, b, c, d]/4
    np.testing.assert_allclose(actual, [expected_h, expected_v], atol=2e-14, rtol=2e-13)
    assert action.step_bound > 0


def test_continuum_action_owns_vectors_and_rejects_wrong_sectors():
    initial, ci, final, cf = np.array([3]), np.array([1.]), np.array([1]), np.array([1.])
    action = ContinuumFockAction(initial, ci, final, cf, 3)
    before = action.amplitudes(np.array([0., 2., 0.]), np.zeros(3), np.zeros((3,)*3))
    ci[:] = -9
    cf[:] = 0
    assert action.amplitudes(np.array([0., 2., 0.]), np.zeros(3), np.zeros((3,)*3)) == before
    with pytest.raises(ValueError, match='particle'):
        ContinuumFockAction([3], [1.], [3], [1.], 3)
    with pytest.raises(ValueError, match='antisymmetric'):
        action.amplitudes(np.zeros(3), np.zeros(3), np.ones((3,)*3))
