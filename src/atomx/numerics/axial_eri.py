"""Exact pair-symmetry packing of a supplied signed integral coordinate space.

No physical selection, value threshold or Pauli compression is performed.
The original outer-p ABI ordering, including its Hermitian duplicates, stays
unchanged after removal of forbidden total-M/parity coordinates.
"""
from dataclasses import dataclass

import numpy as np


def _owned(a):
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, init=False)
class AxialPairLayout:
    symmetry: np.ndarray
    ranks: np.ndarray
    prefix: np.ndarray
    offsets: np.ndarray
    pairs: tuple
    count: int
    max_m: int

    def __init__(self, symmetry):
        s = np.asarray(symmetry)
        if (s.ndim != 2 or s.shape[1] != 2 or len(s) < 2
                or s.dtype.kind not in 'iu' or np.any(abs(s[:, 0]) > 126)
                or np.any((s[:, 1] != 1) & (s[:, 1] != -1))):
            raise ValueError('bounded integer M/parity coordinate pairs required')
        s = np.asarray(s, dtype=np.int64)
        n, m = len(s), int(np.max(abs(s[:, 0])))
        groups = [[] for _ in range(2*(4*m+1))]
        ranks = np.full((n, n), -1, np.int64)
        prefix = np.zeros((n, len(groups)), np.int64)
        for p in range(n):
            prefix[p] = [len(g) for g in groups]
            for q in range(p+1, n):
                g = 2*(int(s[p, 0]+s[q, 0])+2*m)+int(s[p, 1] != s[q, 1])
                ranks[p, q] = len(groups[g])
                groups[g].append((p, q))
        offsets = np.zeros((n, n), np.int64)
        count = 0
        for p in range(n-1):
            for q in range(p+1, n):
                g = 2*(int(s[p, 0]+s[q, 0])+2*m)+int(s[p, 1] != s[q, 1])
                offsets[p, q] = count
                count += len(groups[g])-int(prefix[p, g])
        values = dict(symmetry=_owned(s), ranks=_owned(ranks),
                      prefix=_owned(prefix), offsets=_owned(offsets),
                      pairs=tuple(_owned(np.asarray(g, dtype=np.int64).reshape(-1, 2))
                                  for g in groups), count=count, max_m=m)
        for name, value in values.items():
            object.__setattr__(self, name, value)

    @property
    def index_bytes(self):
        return (self.symmetry.nbytes+self.ranks.nbytes+self.prefix.nbytes
                +self.offsets.nbytes+sum(g.nbytes for g in self.pairs))

    def group(self, p, q):
        s = self.symmetry
        return 2*(int(s[p, 0]+s[q, 0])+2*self.max_m)+int(s[p, 1] != s[q, 1])

    def rows(self):
        """Canonical outer pairs and every allowed right pair, in ABI order."""
        n = len(self.symmetry)
        for p in range(n-1):
            for q in range(p+1, n):
                g = self.group(p, q)
                yield p, q, self.pairs[g][self.prefix[p, g]:]

    def value(self, bank, p, q, r, s):
        """Original signed/conjugated query, including exact forbidden zeros."""
        if min(p, q, r, s) < 0 or max(p, q, r, s) >= len(self.symmetry):
            raise IndexError('integral coordinate out of range')
        if p == q or r == s:
            return 0j
        sign = 1
        if p > q:
            p, q, sign = q, p, -sign
        if r > s:
            r, s, sign = s, r, -sign
        conjugate = p > r
        if conjugate:
            p, q, r, s = r, s, p, q
        g = self.group(p, q)
        if g != self.group(r, s):
            return 0j
        i = self.offsets[p, q]+self.ranks[r, s]-self.prefix[p, g]
        value = complex(bank[i])
        return sign*(value.conjugate() if conjugate else value)
