"""Numeric signed one/two-body values on an explicit determinant graph."""
import numba as nb
import numpy as np


@nb.njit(cache=True, inline='always')
def _count(x):
    total = 0
    while x:
        x &= x-np.uint64(1)
        total += 1
    return total


@nb.njit(cache=True, inline='always')
def _first(x):
    index = 0
    while not x & np.uint64(1):
        x >>= np.uint64(1)
        index += 1
    return index


@nb.njit(cache=True, inline='always')
def _phase(mask, bit):
    return -1. if _count(mask & ((np.uint64(1) << np.uint64(bit))-np.uint64(1))) % 2 else 1.


@nb.njit(cache=True)
def slater_values(dets, h, g, indptr, columns):
    """Real CSR entries; g[p,q,r,s]=<pq||rs>, H2=1/4 sum g a+a+aa.

    Masks have fewer than64 active spinors. Graph, shapes and the complete
    requested space are validated by the physical consumer before this call.
    No screening, physics selection, basis construction or fastmath occurs.
    """
    values = np.zeros(len(columns))
    one = np.uint64(1)
    for row in range(len(dets)):
        a = dets[row]
        for entry in range(indptr[row], indptr[row+1]):
            b = dets[columns[entry]]
            diff = a ^ b
            number = _count(diff)
            value = 0.
            if number == 0:
                rest = a
                while rest:
                    i = _first(rest)
                    rest ^= one << np.uint64(i)
                    value += h[i, i]
                    partners = rest
                    while partners:
                        j = _first(partners)
                        partners ^= one << np.uint64(j)
                        value += g[i, j, i, j]
            elif number == 2:
                p, q = _first(a & diff), _first(b & diff)
                removed = b ^ (one << np.uint64(q))
                sign = _phase(b, q)*_phase(removed, p)
                value = h[p, q]
                spectators = a & b
                while spectators:
                    i = _first(spectators)
                    spectators ^= one << np.uint64(i)
                    value += g[p, i, q, i]
                value *= sign
            elif number == 4:
                added, removed = a & diff, b & diff
                p = _first(added)
                q = _first(added ^ (one << np.uint64(p)))
                r = _first(removed)
                s = _first(removed ^ (one << np.uint64(r)))
                state = b
                sign = _phase(state, r)
                state ^= one << np.uint64(r)
                sign *= _phase(state, s)
                state ^= one << np.uint64(s)
                sign *= _phase(state, q)
                state |= one << np.uint64(q)
                sign *= _phase(state, p)
                value = sign*g[p, q, r, s]
            values[entry] = value
    return values
