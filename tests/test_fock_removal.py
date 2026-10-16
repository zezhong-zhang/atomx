"""Independent sorted-list signs, high bits and integer bitset forms."""

from itertools import combinations

import numpy as np
import pytest

from atomx.numerics.fock_removal import fock_removal_amplitudes, mask_linear_forms


@pytest.mark.parametrize('number', [1, 2, 3, 4])
def test_removal_matches_independent_sorted_particle_lists(number):
    rng = np.random.default_rng(400+number)
    n = 6
    ket = list(combinations(range(n), number))
    bra = list(combinations(range(n), number-1))
    cm, cp = rng.normal(size=len(ket)), rng.normal(size=len(bra))
    cm[-1] = 1e-20
    expected = np.zeros(n)
    for particles, ck in zip(ket, cm, strict=True):
        for position, removed in enumerate(particles):
            final = particles[:position]+particles[position+1:]
            expected[removed] += cp[bra.index(final)]*ck*(-1)**position
    actual = fock_removal_amplitudes([sum(1 << i for i in p) for p in bra], cp,
                                    [sum(1 << i for i in p) for p in ket], cm, n)
    np.testing.assert_array_equal(actual, expected)


def test_high_bits_coherence_darkness_and_integer_forms():
    ket = np.array([(1 << 62)|1, (1 << 61)|1], dtype=np.uint64)
    a = fock_removal_amplitudes([1], [1.], ket, [1., -1.], 63)
    assert a[62] == -1. and a[61] == 1. and a.sum() == 0
    weights = np.arange(-31, 32, dtype=np.int64)
    masks = np.array([0, 1, (1 << 62)|1, (1 << 63)-1], dtype=np.uint64)
    expected = [sum(int(w) for i, w in enumerate(weights) if int(mask)&(1 << i)) for mask in masks]
    np.testing.assert_array_equal(mask_linear_forms(masks, weights), expected)


def test_explicit_admission_caps_ownership_and_no_inferred_sector():
    mask, coefficients = np.array([1], dtype=np.uint64), np.array([1.])
    before = coefficients.copy()
    np.testing.assert_array_equal(fock_removal_amplitudes(mask, coefficients, mask, coefficients, 1), [0.])
    np.testing.assert_array_equal(coefficients, before)
    for kwargs in (dict(memory_bytes=1), dict(max_steps=1)):
        with pytest.raises(MemoryError):
            fock_removal_amplitudes(mask, coefficients, mask, coefficients, 2, **kwargs)
    for bad in ([-1], [4], [1, 1]):
        with pytest.raises(ValueError):
            fock_removal_amplitudes(bad, np.ones(len(bad)), mask, coefficients, 2)
    with pytest.raises(ValueError, match='real'):
        fock_removal_amplitudes(mask, [1j], mask, coefficients, 2)
    with pytest.raises(ValueError, match='overflow'):
        mask_linear_forms(mask, np.array([2**63], dtype=np.uint64))
    with pytest.raises(MemoryError):
        mask_linear_forms(mask, [1, 2], max_steps=1)
