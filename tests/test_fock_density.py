"""Every directed density coordinate against independent sorted-list algebra."""

from itertools import combinations

import numpy as np
import pytest

from atomx.numerics.fock_density import fock_bilinear_density


def apply(occupied, operations):
    occupied, sign = list(occupied), 1
    for orbital, create in operations:
        if (orbital in occupied) == create:
            return None, 0
        sign *= (-1)**sum(i < orbital for i in occupied)
        if create:
            occupied.append(orbital)
            occupied.sort()
        else:
            occupied.remove(orbital)
    return tuple(occupied), sign


@pytest.mark.parametrize('electrons', [0, 1, 2, 3, 4])
def test_all_directed_one_and_canonical_two_body_entries(electrons):
    n = 5
    basis = list(combinations(range(n), electrons))
    rng = np.random.default_rng(7341+electrons)
    left, right = rng.normal(size=(2, len(basis)))
    if len(basis) > 2:
        basis, left, right = basis[:-1], left[:-1], right[:-1]
    masks = [sum(1 << i for i in state) for state in basis]
    one, two, pairs = fock_bilinear_density(masks, left, right, n)
    rho = np.zeros_like(one)
    gamma = np.zeros_like(two)
    for ket, r in zip(basis, right, strict=True):
        for bra, l in zip(basis, left, strict=True):
            for p in range(n):
                for q in range(n):
                    out, sign = apply(ket, [(q, False), (p, True)])
                    if out == bra:
                        rho[p, q] += l*r*sign
            for i, (p, q) in enumerate(pairs):
                for j, (a, b) in enumerate(pairs):
                    out, sign = apply(ket, [(a, False), (b, False), (q, True), (p, True)])
                    if out == bra:
                        gamma[i, j] += l*r*sign
    np.testing.assert_allclose(one, rho, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(two, gamma, atol=1e-14, rtol=1e-14)
    assert np.trace(one) == pytest.approx(electrons*(left@right), abs=1e-13)
    assert np.trace(two) == pytest.approx(electrons*(electrons-1)/2*(left@right), abs=1e-13)


def test_tiny_terms_high_bit_and_exact_resource_admission():
    one, two, pairs = fock_bilinear_density(
        np.array([1 | (1 << 62)], dtype=np.uint64), [1e-12], [-2e-13], 63)
    assert one[62, 62] == pytest.approx(-2e-25, abs=1e-39)
    i = pairs.index((0, 62))
    assert two[i, i] == pytest.approx(-2e-25, abs=1e-39)
    with pytest.raises(MemoryError):
        fock_bilinear_density([3], [1.], [1.], 5, memory_bytes=1)
    with pytest.raises(MemoryError):
        fock_bilinear_density([3], [1.], [1.], 5, max_steps=1)
    with pytest.raises(ValueError):
        fock_bilinear_density([1, 3], [1., 1.], [1., 1.], 5)
