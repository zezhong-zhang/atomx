"""Batched signed reduced one-body operators in complete declared CSF blocks.

Magnetic Slater actions and sparse CSF projection are shared across every
pair of roots. This is numerically equivalent to contracting the existing
pairwise one_body_tensor_terms, including phases and arbitrarily small terms.
No radial model, electric parity rule or empirical strength cutoff lives here.
"""
import hashlib
from collections import OrderedDict
from types import MappingProxyType

import numpy as np
from scipy.sparse import csr_matrix

from atomx.angular.csf_hamiltonian import csf_magnetic_vector, pattern_m_counts
from atomx.angular.wigner import wigner_3j


def magnetic_csf_projection(block, m, keys, coefficients=None, *, engine='csr'):
    """Signed sparse CSF projection, optionally contracted to complete root columns.

    ``block`` contains CSFSignature objects in a bound ordering; keys fixes
    the sorted magnetic layout. The consuming physical operator preflights
    complete determinant/projection resources and binds coefficients to states.
    No CSF contribution or small root coefficient is screened.
    """
    if engine not in ('csr', 'streamed') or (engine == 'streamed' and coefficients is None):
        raise ValueError('explicit CSR projection or streamed complete root columns required')
    if engine == 'streamed':
        counts = {s.pattern: dict(pattern_m_counts(s.pattern)).get(m, 0) for s in block}
        bound = sum(counts.values())
        result = np.zeros((bound, coefficients.shape[1]), dtype=np.result_type(float, coefficients))
        dets, index = [], {}
        for column, sig in enumerate(block):
            vector = csf_magnetic_vector(sig, m, keys)
            rows, amplitudes = [], []
            for determinant, amplitude in vector:
                if amplitude == 0:
                    continue
                if determinant not in index:
                    index[determinant] = len(dets)
                    dets.append(determinant)
                rows.append(index[determinant])
                amplitudes.append(amplitude)
            result[rows] += np.asarray(amplitudes)[:, None]*coefficients[column]
        order = sorted(range(len(dets)), key=dets.__getitem__)
        dets = tuple(dets[i] for i in order)
        return dets, {d: i for i, d in enumerate(dets)}, result[order]
    vectors = [csf_magnetic_vector(s, m, keys) for s in block]
    dets = tuple(sorted({d for v in vectors for d, _ in v}))
    index = {d: i for i, d in enumerate(dets)}
    row, col, value = [], [], []
    for j, vector in enumerate(vectors):
        for determinant, coefficient in vector:
            if coefficient != 0:
                row.append(index[determinant])
                col.append(j)
                value.append(coefficient)
    matrix = csr_matrix((value, (row, col)), shape=(len(dets), len(block)))
    return dets, index, matrix if coefficients is None else matrix @ coefficients


class TensorProjectionCache:
    """Bounded, caller-owned reuse of immutable magnetic root projections.

    Identity contains ordered CSFs, magnetic/orbital layout and all signed
    coefficient bytes. No radial operator or state/column approximation is
    inferred. Entries larger than the budget are computed without caching.
    """

    def __init__(self, memory_bytes):
        if isinstance(memory_bytes, bool) or not isinstance(memory_bytes, int) or memory_bytes < 1:
            raise ValueError('positive integer projection-cache byte budget required')
        self.memory_bytes = memory_bytes
        self.resident_bytes = self.hits = self.misses = 0
        self._entries = OrderedDict()

    def _get(self, block, m, keys, coefficients, build):
        identity = (tuple(s.key for s in block), m, keys,
                    None if coefficients is None else
                    (coefficients.shape, coefficients.dtype.str,
                     hashlib.sha256(coefficients.tobytes()).digest()))
        if identity in self._entries:
            self.hits += 1
            entry, size = self._entries.pop(identity)
            self._entries[identity] = (entry, size)
            return entry
        self.misses += 1
        dets, index, values = build()
        if isinstance(values, np.ndarray):
            values = np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)
            size = values.nbytes
        else:
            values = values.copy()
            for name in ('data', 'indices', 'indptr'):
                a = getattr(values, name)
                setattr(values, name, np.frombuffer(a.tobytes(), dtype=a.dtype))
            size = sum(getattr(values, name).nbytes for name in ('data', 'indices', 'indptr'))
        # Include a conservative allowance for determinant/index/signature keys.
        size += 256*len(dets)+1024*len(block)+4096
        entry = (dets, MappingProxyType(index), values)
        if size <= self.memory_bytes:
            while self.resident_bytes+size > self.memory_bytes:
                _, (_, removed) = self._entries.popitem(last=False)
                self.resident_bytes -= removed
            self._entries[identity] = (entry, size)
            self.resident_bytes += size
        return entry

    def diagnostics(self):
        return dict(hits=self.hits, misses=self.misses, entries=len(self._entries),
                    accounted_resident_bytes=self.resident_bytes, byte_budget=self.memory_bytes)


def one_body_tensor_matrix(bra, ket, rank, reduced_elements, *,
                           max_determinants=5000, memory_bytes=512*1024**2,
                           bra_coefficients=None, ket_coefficients=None, projection_cache=None):
    """Return <bra||T_rank||ket> from signed one-electron reduced elements.

    Blocks each have one N/J. ``reduced_elements[(destination, source)]``
    is a finite real/complex scalar in a common orthonormal orbital layout.
    Counts and conservative projection/action bounds reject before expanding
    magnetic vectors; no state, parentage, coefficient or orbital is pruned.
    Optional complete CSF-coordinate columns are contracted BEFORE the Slater
    action; the output is B^dagger T K with every supplied CSF contribution.
    """
    bra, ket = tuple(bra), tuple(ket)
    if projection_cache is not None and not isinstance(projection_cache, TensorProjectionCache):
        raise ValueError('explicit caller-owned TensorProjectionCache required')
    for name, value in [('rank', rank), ('max_determinants', max_determinants),
                        ('memory_bytes', memory_bytes)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'positive integer {name} required')
    if not bra or not ket:
        raise ValueError('two nonempty CSF blocks required')
    sectors = []
    for block in (bra, ket):
        sector = {(sum(q for _, _, q in s.pattern), s.total_J2) for s in block}
        if len(sector) != 1 or len({s.key for s in block}) != len(block):
            raise ValueError('one N/J sector and distinct CSFs per block required')
        sectors.append(next(iter(sector)))
    if sectors[0][0] != sectors[1][0]:
        raise ValueError('one-body operator requires equal electron counts')
    jf, ji = sectors[0][1], sectors[1][1]
    elements = {}
    for pair, value in reduced_elements.items():
        if (not isinstance(pair, tuple) or len(pair) != 2
                or any(not isinstance(k, tuple) or len(k) != 2 or k[0] < 1 or k[1] == 0 for k in pair)
                or not np.isscalar(value) or not np.isfinite(value)):
            raise ValueError('finite scalar elements with (n,kappa) pair keys required')
        elements[pair] = value
    transforms = []
    for block, transform in ((bra, bra_coefficients), (ket, ket_coefficients)):
        if transform is not None:
            transform = np.asarray(transform)
            if (transform.ndim != 2 or transform.shape[0] != len(block) or transform.shape[1] < 1
                    or transform.dtype.kind not in 'iufc' or not np.all(np.isfinite(transform))):
                raise ValueError('finite CSF-coordinate coefficient columns required')
        transforms.append(transform)
    dtype = np.result_type(float, *[np.asarray(x).dtype for x in elements.values()],
                           *[c.dtype for c in transforms if c is not None])
    if dtype.kind not in 'fc' or dtype.itemsize > 16:
        raise ValueError('floating or complex tensor elements required')
    keys = tuple(sorted({(n, k) for s in bra+ket for n, k, _ in s.pattern}))
    if any(a not in keys or b not in keys for a, b in elements):
        raise ValueError('tensor element orbital outside the common CSF layout')
    counts = []
    for block, j in ((bra, jf), (ket, ji)):
        patterns = {s.pattern for s in block}
        raw = sum(dict(pattern_m_counts(p)).get(j, 0) for p in patterns)
        if not 0 < raw <= max_determinants:
            raise ValueError('complete magnetic determinant block exceeds cap')
        counts.append((raw, sum(dict(pattern_m_counts(s.pattern)).get(j, 0) for s in block)))
    spinors = [(n, k, m) for n, k in keys for m in range(-2*abs(k)+1, 2*abs(k), 2)]
    # Sparse C and sparse rectangular Slater action plus dense final output.
    # Per occupied source, tensor selection permits at most 2*rank+1
    # destinations per orbital pair; overbound by the full spinor count.
    output_sizes = [len(block) if c is None else c.shape[1]
                    for block, c in zip((bra, ket), transforms, strict=True)]
    bytes_bound = (32*sum(c[1] for c in counts)
                   + 32*counts[1][0]*sectors[1][0]*len(spinors)
                   + 48*output_sizes[0]*output_sizes[1] + 64*sum(c[0] for c in counts)
                   + 48*sum(counts[i][0]*output_sizes[i] for i in range(2) if transforms[i] is not None))
    if bytes_bound > memory_bytes:
        raise MemoryError(f'complete projected tensor needs bound {bytes_bound} bytes; cap={memory_bytes}')
    if abs(jf-ji) > 2*rank or jf+ji < 2*rank or (jf-ji) % 2:
        return np.zeros(tuple(output_sizes), dtype=dtype)
    q = jf-ji
    denominator = wigner_3j(jf, 2*rank, ji, -jf, q, ji)
    if denominator == 0:
        return np.zeros(tuple(output_sizes), dtype=dtype)

    def projection(block, m, transform):
        return magnetic_csf_projection(block, m, keys, transform)

    def prepared(block, m, transform):
        if projection_cache is None:
            return projection(block, m, transform)
        return projection_cache._get(block, m, keys, transform,
                                     lambda: projection(block, m, transform))

    _, bra_index, cb = prepared(bra, jf, transforms[0])
    ket_dets, _, ck = prepared(ket, ji, transforms[1])
    connections = [[] for _ in spinors]
    for ib, (nb, kb, mb) in enumerate(spinors):
        for ia, (na, ka, ma) in enumerate(spinors):
            pair = ((na, ka), (nb, kb))
            if ma != mb+q or pair not in elements:
                continue
            value = ((-1)**((2*abs(ka)-1-ma)//2)
                     * wigner_3j(2*abs(ka)-1, 2*rank, 2*abs(kb)-1, -ma, q, mb)
                     * elements[pair])
            if value != 0:
                connections[ib].append((ia, value))
    rows, cols, values = [], [], []
    for col, mask in enumerate(ket_dets):
        occupied = mask
        while occupied:
            low = occupied & -occupied
            ib = low.bit_length()-1
            occupied ^= low
            removed = mask ^ low
            sign_b = (-1)**((mask & (low-1)).bit_count())
            for ia, value in connections[ib]:
                target = 1 << ia
                if removed & target:
                    continue
                row = bra_index.get(removed | target)
                if row is None:
                    continue
                sign_a = (-1)**((removed & (target-1)).bit_count())
                rows.append(row)
                cols.append(col)
                values.append(sign_a*sign_b*value)
    action = csr_matrix((np.asarray(values, dtype=dtype), (rows, cols)),
                        shape=(len(bra_index), len(ket_dets)))
    result = cb.conjugate().T @ (action @ ck)
    return (result.toarray() if hasattr(result, 'toarray') else np.asarray(result))/denominator
