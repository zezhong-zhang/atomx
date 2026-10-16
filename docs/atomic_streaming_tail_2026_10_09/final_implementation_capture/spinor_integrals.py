"""Signed native Dirac–Coulomb integrals for tensor correlation engines.

The Hamiltonian is E_frozen + h_eff[p,q] a†p aq +
1/4 g[p,q,r,s] a†p a†q as ar. Every retained radial orbital contributes
its complete magnetic multiplet. Frozen electrons enter both the constant
and one-body operator; correlated occupied spinors enter the Fock operator.
No determinant space is constructed.
"""
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from math import comb
from pathlib import Path
from types import MappingProxyType

import numpy as np


def packed_size(n):
    return sum((n-p-1) * comb(n-p, 2) for p in range(n-1))


def _packed_offsets(n):
    offsets, offset = [], 0
    for p in range(n-1):
        for _ in range(p+1, n):
            offsets.append(offset)
            offset += comb(n-p, 2)
    return tuple(offsets)


def _packed_value(n, data, offsets, p, q, r, s):
    if min(p, q, r, s) < 0 or max(p, q, r, s) >= n:
        raise IndexError('spinor integral index out of range')
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
    pq = n*p+q-(p+1)*(p+2)//2
    m, rx, sx = n-p, r-p, s-p
    rs = m*rx+sx-(rx+1)*(rx+2)//2
    value = data[offsets[pq]+rs]
    return sign*(value.conjugate() if conjugate else value)


def normal_ordered_one_body(fock, packed, noccupied):
    """Recover h_eff = F - sum_i g[p,i,q,i] in the retained core space."""
    n = len(fock)
    offsets = _packed_offsets(n)
    h = np.asarray(fock, dtype=complex).copy()
    for p in range(n):
        for q in range(n):
            h[p, q] -= sum(_packed_value(n, packed, offsets, p, i, q, i)
                           for i in range(noccupied))
    return h


@dataclass(frozen=True)
class SpinorIntegrals:
    labels: tuple
    noccupied: int
    one_body_Ha: np.ndarray
    fock_Ha: np.ndarray
    packed_eri_Ha: np.ndarray
    frozen_energy_Ha: float
    reference_energy_Ha: float
    identity: object

    def __post_init__(self):
        n = len(self.labels)
        if not 0 < self.noccupied < n or len(set(self.labels)) != n:
            raise ValueError('require unique spinors and occupied/virtual spaces')
        for name, shape in [('one_body_Ha', (n, n)), ('fock_Ha', (n, n)),
                            ('packed_eri_Ha', (packed_size(n),))]:
            a = np.asarray(getattr(self, name), dtype=np.complex128)
            if a.shape != shape or not np.all(np.isfinite(a)):
                raise ValueError('nonfinite or malformed integral arrays')
            if a.ndim == 2 and not np.array_equal(a, a.conj().T):
                raise ValueError('one-body and Fock operators must be Hermitian')
            # Bytes backing prevents the caller making these mutable again.
            owned = np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(shape)
            object.__setattr__(self, name, owned)
        if not np.isfinite(self.frozen_energy_Ha + self.reference_energy_Ha):
            raise ValueError('nonfinite energy constants')
        object.__setattr__(self, '_offsets', _packed_offsets(n))
        object.__setattr__(self, 'identity', MappingProxyType(dict(self.identity)))
        # The backend consumes Fock, not h: verify the occupied contraction
        # and reference energy to prevent exporting a different Hamiltonian.
        actual_fock = self.one_body_Ha.copy()
        for p in range(n):
            for q in range(n):
                actual_fock[p, q] += sum(self.eri(p, i, q, i) for i in range(self.noccupied))
        err = float(np.max(abs(actual_fock-self.fock_Ha)))
        o = self.noccupied
        energy = self.frozen_energy_Ha + np.trace(self.one_body_Ha[:o, :o]) + .5*sum(
            self.eri(i, j, i, j) for i in range(o) for j in range(o))
        if err > 1e-9 or abs(energy-self.reference_energy_Ha) > 1e-9:
            raise ValueError('Fock/reference energy do not match signed normal-ordered Hamiltonian')
        # Entries with equal first pair index occur twice in this packing.
        # Check their conjugacy instead of quietly selecting one triangle.
        for p in range(n-1):
            for q in range(p+1, n):
                for s in range(p+1, n):
                    if abs(self.eri(p, q, p, s)-self.eri(p, s, p, q).conjugate()) > 1e-10:
                        raise ValueError('packed two-body duplicate entries are not Hermitian')

    def eri(self, p, q, r, s):
        return _packed_value(len(self.labels), self.packed_eri_Ha, self._offsets, p, q, r, s)

    @property
    def dense_eri_bytes(self):
        return self.packed_eri_Ha.nbytes

    def diagnostics(self):
        return dict(representation='packed', resident_matrix_bytes=self.one_body_Ha.nbytes+self.fock_Ha.nbytes,
                    resident_eri_bytes=self.packed_eri_Ha.nbytes, dense_eri_bytes=self.dense_eri_bytes)

    def iter_packed_eri(self, *, max_chunk_bytes=4*1024**2):
        if max_chunk_bytes < 16:
            raise ValueError('integral chunk must hold at least one complex value')
        count = int(max_chunk_bytes)//16
        for begin in range(0, len(self.packed_eri_Ha), count):
            yield self.packed_eri_Ha[begin:begin+count]


def _pair_symmetry(labels, p, q):
    a, b = labels[p], labels[q]
    la = a[1] if a[1] > 0 else -a[1]-1
    lb = b[1] if b[1] > 0 else -b[1]-1
    return a[2]+b[2], (la+lb) % 2


@dataclass(frozen=True)
class OnDemandSpinorIntegrals:
    """Native spherical DC operator with bounded radial/angular caches.

    Only the two one-body matrices are resident. The owned immutable native
    snapshot evaluates requested signed ERIs; complete magnetic multiplets
    and exact total-M/parity zeros are preserved. The backend still needs its
    full integral/tensor representation, checked separately before export.
    This factory-only representation does not accept arbitrary rotated ERIs.
    """
    labels: tuple
    noccupied: int
    one_body_Ha: np.ndarray
    fock_Ha: np.ndarray
    frozen_energy_Ha: float
    reference_energy_Ha: float
    identity: object
    _evaluate: object
    _diagnostics: object

    def __post_init__(self):
        n = len(self.labels)
        if not 0 < self.noccupied < n or len(set(self.labels)) != n:
            raise ValueError('require unique spinors and occupied/virtual spaces')
        for name in ('one_body_Ha', 'fock_Ha'):
            a = np.asarray(getattr(self, name), dtype=np.complex128)
            if a.shape != (n, n) or not np.all(np.isfinite(a)) or not np.array_equal(a, a.conj().T):
                raise ValueError('nonfinite, malformed or non-Hermitian one-body operator')
            for p, row in enumerate(a):
                if any(self.labels[q][1:] != self.labels[p][1:] for q in np.flatnonzero(row)):
                    raise ValueError('native spherical one-body operator has nonscalar entries')
            object.__setattr__(self, name, np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(n, n))
        if not np.isfinite(self.frozen_energy_Ha+self.reference_energy_Ha):
            raise ValueError('nonfinite energy constants')
        object.__setattr__(self, 'identity', MappingProxyType(dict(self.identity)))
        # A closed spherical core is scalar in kappa and m. Validate its
        # surviving contraction, rather than iterating exact zero sectors.
        radial, groups = {}, {}
        for i, label in enumerate(self.labels):
            radial.setdefault(label[:2], {})[label[2]] = i
        for key, multiplet in radial.items():
            if set(multiplet) != set(range(-2*abs(key[1])+1, 2*abs(key[1]), 2)):
                raise ValueError('native operator requires complete magnetic multiplets')
            occupied = [i < self.noccupied for i in multiplet.values()]
            if any(occupied) and not all(occupied):
                raise ValueError('native reference requires a closed spherical core')
            groups.setdefault(key[1], []).append(multiplet)
        for group in groups.values():
            for left in group:
                for right in group:
                    p, q = left[min(left)], right[min(right)]
                    h = self.one_body_Ha[p, q]
                    actual = h+sum(self.eri(p, i, q, i) for i in range(self.noccupied))
                    for m in left:
                        if (abs(self.one_body_Ha[left[m], right[m]]-h) > 1e-10
                                or abs(self.fock_Ha[left[m], right[m]]-actual) > 1e-9):
                            raise ValueError('native spherical Fock contraction failed')
        o = self.noccupied
        energy = self.frozen_energy_Ha+np.trace(self.one_body_Ha[:o, :o])+.5*sum(
            self.eri(i, j, i, j) for i in range(o) for j in range(o))
        if abs(energy-self.reference_energy_Ha) > 1e-9:
            raise ValueError('native reference energy failed')

    @property
    def dense_eri_bytes(self):
        return 16*packed_size(len(self.labels))

    def diagnostics(self):
        return dict(self._diagnostics(), representation='on_demand',
                    resident_matrix_bytes=self.one_body_Ha.nbytes+self.fock_Ha.nbytes,
                    resident_eri_bytes=0, dense_eri_bytes=self.dense_eri_bytes)

    def eri(self, p, q, r, s):
        n = len(self.labels)
        if min(p, q, r, s) < 0 or max(p, q, r, s) >= n:
            raise IndexError('spinor integral index out of range')
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
        if _pair_symmetry(self.labels, p, q) != _pair_symmetry(self.labels, r, s):
            return 0j
        value = complex(self._evaluate(*(self.labels[i] for i in (p, q, r, s))))
        return sign*(value.conjugate() if conjugate else value)

    def iter_packed_eri(self, *, max_chunk_bytes=4*1024**2):
        """Yield the original ABI order with a bounded complex output buffer."""
        if max_chunk_bytes < 16:
            raise ValueError('integral chunk must hold at least one complex value')
        n, count = len(self.labels), int(max_chunk_bytes)//16
        pairs = {}
        for p in range(n-1):
            for q in range(p+1, n):
                pairs.setdefault(_pair_symmetry(self.labels, p, q), []).append((p, q))
        for p in range(n-1):
            m, length = n-p, comb(n-p, 2)
            for q in range(p+1, n):
                allowed = pairs[_pair_symmetry(self.labels, p, q)]
                for begin in range(0, length, count):
                    chunk = np.zeros(min(count, length-begin), complex)
                    for r, s in allowed:
                        if r < p:
                            continue
                        rx, sx = r-p, s-p
                        index = m*rx+sx-(rx+1)*(rx+2)//2-begin
                        if 0 <= index < len(chunk):
                            # Canonical ABI pairs and exact symmetry were
                            # already selected; avoid repeating those checks.
                            chunk[index] = self._evaluate(self.labels[p], self.labels[q],
                                self.labels[r], self.labels[s])
                    yield chunk


def build_spinor_integrals(orbitals, workspace, Z, ns, *, occupied_keys,
                           virtual_keys, frozen_keys=(), nuclear_potential=None,
                           max_integral_bytes=2*1024**3, metric_tolerance=1e-8,
                           representation='packed', radial_pair_cache_size=1024):
    """Export the complete declared orthonormal B-spline DC space.

    Caller supplies all retained virtuals explicitly, including continuum
    pseudo-states. Missing orbitals, nonorthogonal metrics and resource excess
    reject the whole request. ``on_demand`` retains the native immutable radial
    snapshot instead of a quartic ERI array; the payload cap then applies to
    resident one-body matrices. Cache/Python overhead is not a peak-RSS bound.
    Export and backend still check their full logical storage separately.
    Static Breit/QED are not included in this API.
    """
    from atomx.angular.csf_hamiltonian import two_body_angular
    from atomx.physics.csf.effective_one_body import CIReferenceBinding
    from atomx.physics.csf.matrix import _dirac_blocks
    from atomx.physics.csf.radial_snapshot import CIRadialSnapshot

    if representation not in ('packed', 'on_demand'):
        raise ValueError('integral representation must be packed or on_demand')
    groups = [tuple(tuple(k) for k in keys) for keys in
              (occupied_keys, virtual_keys, frozen_keys)]
    occ, vir, frozen = groups
    keys = occ+vir+frozen
    if (not occ or not vir or len(set(keys)) != len(keys)
            or any(k not in orbitals or len(k) != 2 or k[1] == 0 for k in keys)):
        raise ValueError('require disjoint occupied/virtual/frozen orbital keys')
    def multiplets(selected):
        return tuple((n, k, m) for n, k in selected
                     for m in range(-2*abs(k)+1, 2*abs(k), 2))
    labels = multiplets(occ+vir)
    core = multiplets(occ+frozen)
    frozen_labels = multiplets(frozen)
    n = len(labels)
    required = 16*(2*n*n+(packed_size(n) if representation == 'packed' else 0))
    if required > max_integral_bytes or max_integral_bytes < 1:
        raise MemoryError(f'complete integral space needs {required} bytes; cap {max_integral_bytes}')
    selected = {k: tuple(orbitals[k]) for k in keys}
    # Metric check includes correlated core, frozen core and all virtuals.
    metric_error = 0.0
    for kappa in sorted({k for _, k in keys}):
        subset = [key for key in keys if key[1] == kappa]
        _, S = _dirac_blocks(ns, kappa, Z, workspace)
        a = np.column_stack([np.r_[selected[key][0], selected[key][1]] for key in subset])
        metric_error = max(metric_error, float(np.max(abs(a.T @ S @ a-np.eye(len(subset))))))
    if metric_error > metric_tolerance:
        raise ValueError(f'common orbital metric failed: {metric_error}')
    snap = CIRadialSnapshot(selected, workspace, Z, ns, nuclear_potential=nuclear_potential,
                           pair_potential_cache_size=radial_pair_cache_size)
    radial = snap.evaluator_for(selected, workspace, Z, ns, nuclear_potential=nuclear_potential)
    @lru_cache(maxsize=16384)
    def h(a, b):
        if a[1:] != b[1:]:
            return 0.0
        return radial(('h', a[0], a[1], b[0]))
    def direct(a, b, c, d):
        return sum(v * radial(('r', rank, a[:2], c[:2], b[:2], d[:2]))
                   for rank, v in two_body_angular(a[1], a[2], b[1], b[2],
                                                 c[1], c[2], d[1], d[2]))
    max_projection = 0.0
    @lru_cache(maxsize=65536)
    def g(a, b, c, d):
        nonlocal max_projection
        if a[2]+b[2] != c[2]+d[2]:
            return 0.0
        x = direct(a, b, c, d)-direct(a, b, d, c)
        y = direct(c, d, a, b)-direct(c, d, b, a)
        error = abs(x-y)
        if error > 1e-10:
            raise ValueError(f'two-electron Hermitian gate failed: {error}')
        max_projection = max(max_projection, .5*error)
        return .5*(x+y)
    frozen_energy = sum(h(a, a) for a in frozen_labels)+.5*sum(
        g(a, b, a, b) for a in frozen_labels for b in frozen_labels)
    reference_energy = sum(h(a, a) for a in core)+.5*sum(g(a, b, a, b) for a in core for b in core)
    heff = np.empty((n, n), complex)
    fock = np.empty((n, n), complex)
    @lru_cache(maxsize=32768)
    def scalar_one_body(left, right):
        m = -2*abs(left[1])+1
        a, b = (*left, m), (*right, m)
        return (h(a, b)+sum(g(a, i, b, i) for i in frozen_labels),
                h(a, b)+sum(g(a, i, b, i) for i in core))
    for p, a in enumerate(labels):
        for q, b in enumerate(labels):
            if representation == 'on_demand' and a[1:] != b[1:]:
                # All occupied/frozen radial keys are complete closed shells.
                heff[p, q] = fock[p, q] = 0.0
                continue
            if representation == 'on_demand':
                heff[p, q], fock[p, q] = scalar_one_body(a[:2], b[:2])
                continue
            heff[p, q] = h(a, b)+sum(g(a, i, b, i) for i in frozen_labels)
            fock[p, q] = h(a, b)+sum(g(a, i, b, i) for i in core)
    error = max(float(np.max(abs(heff-heff.T.conj()))), float(np.max(abs(fock-fock.T.conj()))))
    if error > 1e-10:
        raise ValueError('one-body Hermitian gate failed')
    max_projection = max(max_projection, .5*error)
    heff, fock = .5*(heff+heff.T.conj()), .5*(fock+fock.T.conj())
    binding = CIReferenceBinding.capture([(*selected[k], k[1]) for k in occ+vir],
        {k: selected[k] for k in frozen+occ}, workspace, Z, ns, nuclear_potential)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
               (Path(__file__), Path(__file__).with_name('matrix.py'),
                Path(__file__).parents[2]/'angular/csf_hamiltonian.py',
                Path(__file__).with_name('radial_snapshot.py'))}
    def identity():
        return dict(hamiltonian='Dirac-Coulomb', binding=repr(binding),
            physical_source_sha256=sources, occupied_keys=occ, virtual_keys=vir,
            frozen_keys=frozen, metric_error=metric_error,
            hermitian_projection_max_Ha=max_projection, integral_bytes=required,
            radial_pair_cache_size=radial_pair_cache_size)
    if representation == 'on_demand':
        def diagnostics():
            return dict(snap.diagnostics(), hermitian_projection_max_Ha=max_projection,
                        signed_cache_entries=g.cache_info().currsize,
                        signed_cache_capacity=g.cache_info().maxsize,
                        pair_cache_array_bound_bytes=radial_pair_cache_size*2*ns*max(
                            workspace.get('ksp', workspace['ks']), workspace.get('ksq', workspace['ks']))*8)
        return OnDemandSpinorIntegrals(labels, len(multiplets(occ)), heff, fock,
            frozen_energy, reference_energy,
            dict(identity(), representation='native on-demand radial/angular DC',
                 construction_scalar_radial_pairs=scalar_one_body.cache_info().misses,
                 hermitian_projection_scope='core/one-body at construction; every requested ERI checked on evaluation'),
            g, diagnostics)
    # Fill only exactly allowed total-M/parity pairs. Every ABI coordinate
    # remains represented, including the exact zeros, without a weak cutoff.
    packed = np.zeros(packed_size(n), complex)
    pair_blocks = {}
    def pair_sym(p, q):
        a, b = labels[p], labels[q]
        la = a[1] if a[1] > 0 else -a[1]-1
        lb = b[1] if b[1] > 0 else -b[1]-1
        return a[2]+b[2], (la+lb) % 2
    for p in range(n-1):
        for q in range(p+1, n):
            pair_blocks.setdefault(pair_sym(p, q), []).append((p, q))
    offset = 0
    for p in range(n-1):
        for q in range(p+1, n):
            for r, s in pair_blocks[pair_sym(p, q)]:
                if r < p:
                    continue
                m, rx, sx = n-p, r-p, s-p
                index = offset+m*rx+sx-(rx+1)*(rx+2)//2
                packed[index] = g(labels[p], labels[q], labels[r], labels[s])
            offset += comb(n-p, 2)
    return SpinorIntegrals(labels, len(multiplets(occ)), heff, fock, packed,
        frozen_energy, reference_energy, identity())
