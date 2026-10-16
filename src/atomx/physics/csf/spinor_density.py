"""Complete signed DC radial weights through exact finite spinor densities."""

from functools import lru_cache

import numpy as np

from atomx.angular.csf_hamiltonian import _radial_terms, csf_signature, pattern_m_counts
from atomx.angular.one_body_tensor import magnetic_csf_projection
from atomx.angular.projection_cache import MagneticCSFProjectionCache
from atomx.numerics.fock_density import fock_bilinear_density
from atomx.physics.csf.radial_symmetry import canonical_static_radial_key


class SpinorRadialDensity:
    """Same real scalar bilinears as CIRadialDensity, with no angular pair bank.

    A common filled subshell core is contracted exactly, including every
    core radial derivative and exterior-product phase. Complete signed root
    projections and directed RDMs retain all active substitutions. Only static
    real Dirac-Coulomb radial keys are supplied. This object does not certify
    state normalization, eigen-equations or orbital/nuclear identity.
    """

    def __init__(self, csfs, *, max_determinants=80000,
                 memory_bytes=512*1024**2, max_steps=500000000, projection_cache=None):
        if projection_cache is not None and not isinstance(projection_cache, MagneticCSFProjectionCache):
            raise ValueError('explicit bounded magnetic CSF projection cache required')
        self.projection_cache = projection_cache
        for value in (max_determinants, memory_bytes, max_steps):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError('positive integer density resource caps required')
        self.signatures = tuple(csf_signature(c) for c in csfs)
        if (not self.signatures or len({s.key for s in self.signatures}) != len(csfs)
                or len({c.n_electrons for c in csfs}) != 1):
            raise ValueError('distinct nonempty CSFs in one particle sector required')
        self.keys = tuple(sorted({(n, k) for s in self.signatures for n, k, _ in s.pattern}))
        frozen_keys = {key for key in self.keys
                       if all((*key, 2*abs(key[1])) in s.pattern for s in self.signatures)}
        self.labels = tuple((n, k, m) for n, k in self.keys for m in range(-2*abs(k)+1, 2*abs(k), 2))
        self.core = tuple(i for i, label in enumerate(self.labels) if label[:2] in frozen_keys)
        self.active = tuple(i for i, label in enumerate(self.labels) if label[:2] not in frozen_keys)
        if len(self.active) > 63:
            raise MemoryError('complete spinor density exceeds63 active spinors; no orbital removed')
        self.frozen_mask = sum(1 << i for i in self.core)
        groups = {}
        for i, c in enumerate(csfs):
            groups.setdefault((c.total_J2, c.parity), []).append(i)
        # Functional consumers use these column inventories for exact eligibility
        # checks. Projections are constructed per requested bilinear, not stored.
        self.blocks = tuple((sector, None, tuple(indices)) for sector, indices in sorted(groups.items()))
        self.max_determinants, self.memory_bytes, self.max_steps = max_determinants, memory_bytes, max_steps
        for sector, _, columns in self.blocks:
            patterns = {self.signatures[i].pattern for i in columns}
            raw = sum(dict(pattern_m_counts(p)).get(sector[0], 0) for p in patterns)
            if raw > max_determinants:
                raise MemoryError('complete spinor density magnetic basis exceeds determinant cap')

        @lru_cache(maxsize=32768)
        def gterms(a, b, c, d):
            terms = {}
            for cc, dd, sign in ((c, d, 1.), (d, c, -1.)):
                for key, value in _radial_terms(a, b, cc, dd, sign, self.labels):
                    key, phase = canonical_static_radial_key(key)
                    terms[key] = terms.get(key, 0.)+phase*value
            return tuple((k, v) for k, v in terms.items() if v != 0)
        self._gterms = gterms

    def contract_bilinear(self, left, right):
        left, right = np.asarray(left), np.asarray(right)
        if left.ndim != 1 or right.ndim != 1:
            raise ValueError('finite real CSF vectors required')
        return self.contract_bilinears(left[:, None], right[:, None])[0]

    def contract_bilinears(self, left, right):
        left, right = np.asarray(left), np.asarray(right)
        if (left.ndim != 2 or left.shape != right.shape or left.shape[0] != len(self.signatures)
                or left.shape[1] < 1 or any(not np.isrealobj(x) or not np.isfinite(x).all() for x in (left, right))):
            raise ValueError('matching finite real CSF column arrays required')
        result = tuple({} for _ in range(left.shape[1]))
        for (j, _), _, columns in self.blocks:
            columns = list(columns)
            l, r = left[columns], right[columns]
            if not np.any(l) or not np.any(r):
                continue
            signatures = tuple(self.signatures[i] for i in columns)
            raw = sum(dict(pattern_m_counts(p)).get(j, 0) for p in {s.pattern for s in signatures})
            if 64*raw*left.shape[1] > self.memory_bytes:
                raise MemoryError('complete spinor root projection exceeds density cap')
            if self.projection_cache is None:
                dets, _, projected = magnetic_csf_projection(signatures, j, self.keys,
                                  np.column_stack((l, r)), engine='streamed')
            else:
                dets, _, projected = self.projection_cache.project(signatures, j, self.keys,
                                                                  np.column_stack((l, r)))
            masks, signs = [], []
            for mask in dets:
                if mask&self.frozen_mask != self.frozen_mask:
                    raise ValueError('spinor density opens a contracted closed core')
                masks.append(sum(1 << k for k, bit in enumerate(self.active) if mask&(1 << bit)))
                signs.append((-1)**sum((mask&~self.frozen_mask&((1 << bit)-1)).bit_count() for bit in self.core))
            projected *= np.asarray(signs)[:, None]
            for k, weights in enumerate(result):
                a, b = projected[:, k], projected[:, left.shape[1]+k]
                if not np.any(a) or not np.any(b):
                    continue
                overlap = float(a@b)
                if self.active:
                    one, two, pairs = fock_bilinear_density(np.asarray(masks, dtype=np.uint64), a, b,
                        len(self.active), memory_bytes=self.memory_bytes, max_steps=self.max_steps)
                else:
                    one, two, pairs = np.empty((0, 0)), np.empty((0, 0)), ()
                def add(key, value, weights=weights):
                    key, sign = canonical_static_radial_key(key)
                    weights[key] = weights.get(key, 0.)+sign*value
                def addg(p, q, r, s, value):
                    for key, angular in self._gterms(p, q, r, s):
                        add(key, value*angular)
                for p in self.core:
                    n, kap, _ = self.labels[p]
                    add(('h', n, kap, n), overlap)
                for i, p in enumerate(self.core):
                    for q in self.core[i+1:]:
                        addg(p, q, p, q, overlap)
                for p, q in zip(*np.nonzero(one), strict=True):
                    aa, bb = self.active[p], self.active[q]
                    a_label, b_label = self.labels[aa], self.labels[bb]
                    if a_label[1:] == b_label[1:]:
                        add(('h', a_label[0], a_label[1], b_label[0]), one[p, q])
                    for c in self.core:
                        addg(aa, c, bb, c, one[p, q])
                for pq, rs in zip(*np.nonzero(two), strict=True):
                    p, q = (self.active[i] for i in pairs[pq])
                    r, s = (self.active[i] for i in pairs[rs])
                    addg(p, q, r, s, two[pq, rs])
        return tuple({key: float(v) for key, v in row.items() if v != 0} for row in result)

    def contract(self, ci, state_weights=None):
        if state_weights is None:
            state_weights = abs(ci.state_energies-ci.ground_energy) < 1e-10
            state_weights = state_weights/state_weights.sum()
        w = np.asarray(state_weights, dtype=float)
        if (w.shape != ci.state_energies.shape or np.any(w < 0) or not np.isfinite(w).all()
                or abs(w.sum()-1) > 1e-12):
            raise ValueError('normalized nonnegative finite state weights required')
        selected = w > 0
        vectors = ci.mixing_coefficients[:, selected]
        terms = self.contract_bilinears(vectors, vectors)
        keys = set().union(*terms)
        combined = {key: sum(weight*row.get(key, 0.) for weight, row in zip(w[selected], terms, strict=True))
                    for key in keys}
        return {key: value for key, value in combined.items() if value != 0}
