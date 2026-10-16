"""Exact real one-/two-particle bilinears on a supplied finite Fock basis."""

from math import comb

import numba as nb
import numpy as np
from numba.typed import Dict

from atomx.numerics.slater_values import _first, _phase


@nb.njit(cache=True)
def _density(masks, left, right, lookup, pair_index, nspinors):
    one = np.zeros((nspinors, nspinors))
    count = nspinors*(nspinors-1)//2
    two = np.zeros((count, count))
    bit_one = np.uint64(1)
    for column in range(len(masks)):
        mask, cr = masks[column], right[column]
        rrest = mask
        while rrest:
            r = _first(rrest)
            rb = bit_one << np.uint64(r)
            rrest ^= rb
            reduced = mask ^ rb
            sign_r = _phase(mask, r)
            for p in range(nspinors):
                pb = bit_one << np.uint64(p)
                if reduced & pb:
                    continue
                target = reduced | pb
                if target in lookup:
                    row = lookup[target]
                    one[p, r] += left[row]*cr*sign_r*_phase(reduced, p)
            srest = rrest
            while srest:
                s = _first(srest)
                sb = bit_one << np.uint64(s)
                srest ^= sb
                remainder = reduced ^ sb
                sign_rs = sign_r*_phase(reduced, s)
                rs = pair_index[r, s]
                for q in range(1, nspinors):
                    qb = bit_one << np.uint64(q)
                    if remainder & qb:
                        continue
                    with_q = remainder | qb
                    sign_q = sign_rs*_phase(remainder, q)
                    for p in range(q):
                        pb = bit_one << np.uint64(p)
                        if with_q & pb:
                            continue
                        target = with_q | pb
                        if target in lookup:
                            row = lookup[target]
                            two[pair_index[p, q], rs] += left[row]*cr*sign_q*_phase(with_q, p)
    return one, two


def fock_bilinear_density(masks, left, right, nspinors, *,
                          memory_bytes=128*1024**2, max_steps=500000000):
    """Return directed real rho[p,q] and Gamma[(p<q),(r<s)].

    rho=<L|a+_p a_q|R>, Gamma=<L|a+_p a+_q a_s a_r|R>.
    Canonical pair order is lexicographic. Contraction with arbitrary real
    h/g is sum rho*h + sum Gamma*g_canonical; no factor1/4 is needed for
    canonical antisymmetric g pairs. No symmetry, state normalization,
    orbital physics, integral screening or complete-particle-space inference
    occurs here. The supplied basis may be a proper finite subset.
    """
    for value, name in ((nspinors, 'nspinors'), (memory_bytes, 'memory_bytes'),
                        (max_steps, 'max_steps')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'positive integer {name} required')
    if nspinors > 63:
        raise ValueError('at most63 supplied spinors required')
    masks, left, right = map(np.asarray, (masks, left, right))
    if (masks.ndim != 1 or not masks.size or masks.dtype.kind not in 'iu'
            or len(set(map(int, masks))) != len(masks)
            or any(int(m) < 0 or int(m).bit_length() > nspinors for m in masks)
            or left.shape != masks.shape or right.shape != masks.shape
            or any(not np.isrealobj(a) or not np.isfinite(a).all() for a in (left, right))):
        raise ValueError('unique masks and matching finite real coefficients required')
    numbers = {int(m).bit_count() for m in masks}
    if len(numbers) != 1:
        raise ValueError('one fixed particle sector required')
    occupied = numbers.pop()
    count = comb(nspinors, 2)
    steps = len(masks)*(occupied*(nspinors-occupied+1)
                        + comb(occupied, 2)*comb(nspinors-occupied+2, 2))
    bound = 16*(nspinors**2+count**2)+160*len(masks)
    if bound > memory_bytes or steps > max_steps:
        raise MemoryError('complete Fock density exceeds declared payload/work cap; no terms screened')
    pairs = tuple((p, q) for p in range(nspinors) for q in range(p+1, nspinors))
    indices = np.full((nspinors, nspinors), -1, dtype=np.int64)
    for k, (p, q) in enumerate(pairs):
        indices[p, q] = k
    lookup = Dict.empty(key_type=nb.types.uint64, value_type=nb.types.int64)
    for i, mask in enumerate(masks):
        lookup[np.uint64(mask)] = i
    one, two = _density(np.asarray(masks, dtype=np.uint64), np.asarray(left, dtype=float),
                        np.asarray(right, dtype=float), lookup, indices, int(nspinors))
    return one, two, pairs
