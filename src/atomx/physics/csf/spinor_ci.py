"""Exact native frozen-core spinor integrals consumed by complete CSF CI."""
from array import array

import numpy as np
from scipy.sparse import csr_matrix, diags

from atomx.angular.csf_hamiltonian import (
    _connected_pairs,
    _permutation_sign,
    csf_magnetic_vector,
    csf_signature,
    pattern_m_counts,
)
from atomx.numerics.slater_values import slater_values
from atomx.physics.csf.spinor_integrals import build_spinor_integrals


class SpinorCIHamiltonian:
    """One common scalar frozen-core Hamiltonian, not an MBPT correction.

    ``reference_keys`` supplies the integral exporter's closed reference.
    CI consumes h_eff/g/E_frozen, not its reference Fock or energy. The CI
    particle number and all opened reference-shell holes remain explicit.
    Explicit Breit options use the shared signed integral exporter, including
    its frozen-core ledger. Finite-frequency exchange replaces static Breit;
    fixed orbitals and frequencies do not include relaxation or QED.
    """

    def __init__(self, csfs, orbitals, workspace, Z, ns, *, reference_keys,
                 nuclear_potential=None, memory_bytes=512*1024**2,
                 include_breit=False, breit_orbital_energies=None,
                 transverse_cache_size=128, projection_cache=None):
        from atomx.angular.projection_cache import MagneticCSFProjectionCache

        if projection_cache is not None and not isinstance(projection_cache, MagneticCSFProjectionCache):
            raise ValueError('explicit bounded magnetic CSF projection cache required')
        self.projection_cache = projection_cache
        csfs = tuple(csfs)
        signatures = tuple(csf_signature(c) for c in csfs)
        all_keys = {(n, k) for s in signatures for n, k, _ in s.pattern}
        frozen = tuple(sorted(k for k in all_keys if all((*k, 2*abs(k[1])) in s.pattern
                                                       for s in signatures)))
        active = all_keys-set(frozen)
        reference_keys = tuple(reference_keys)
        if not reference_keys or not set(reference_keys) < active:
            raise ValueError('explicit closed reference keys must be a proper subset of active orbitals')
        virtuals = tuple(sorted(active-set(reference_keys)))
        n = sum(2*abs(k) for _, k in active)
        if n > 63 or (8*n**4+32*n*n) > memory_bytes:
            raise MemoryError('complete real active spinor integrals exceed declared cap/63-spinor layout')
        self.integrals = build_spinor_integrals(orbitals, workspace, Z, ns,
            occupied_keys=reference_keys, virtual_keys=virtuals, frozen_keys=frozen,
            nuclear_potential=nuclear_potential, max_integral_bytes=memory_bytes,
            include_breit=include_breit, breit_orbital_energies=breit_orbital_energies,
            transverse_cache_size=transverse_cache_size)
        obj = self.integrals
        if np.max(abs(obj.one_body_Ha.imag)) > 1e-12 or np.max(abs(obj.packed_eri_Ha.imag)) > 1e-12:
            raise ValueError('real spinor CI requires real h and g')
        self.h = np.ascontiguousarray(obj.one_body_Ha.real)
        self.g = np.empty((n, n, n, n))
        # Fill exact signed permutations once, avoiding Python callbacks in
        # every determinant matrix element.
        self.g.fill(0.)
        for p in range(n):
            for q in range(p+1, n):
                for r in range(n):
                    for s in range(r+1, n):
                        value = obj.eri(p, q, r, s).real
                        self.g[p,q,r,s] = self.g[q,p,s,r] = value
                        self.g[q,p,r,s] = self.g[p,q,s,r] = -value
        active_keys = tuple(dict.fromkeys(label[:2] for label in obj.labels))
        self.keys = frozen+active_keys
        self.frozen_spinors = sum(2*abs(k) for _, k in frozen)
        sorted_labels = tuple((n,k,m) for n,k in sorted(self.keys)
                              for m in range(-2*abs(k)+1, 2*abs(k), 2))
        target_labels = tuple((n,k,m) for n,k in self.keys
                              for m in range(-2*abs(k)+1, 2*abs(k), 2))
        old_index = {label:i for i,label in enumerate(sorted_labels)}
        self.old_frozen_mask = sum(1 << old_index[label] for label in target_labels[:self.frozen_spinors])
        self.active_remap = tuple((1 << old_index[label], 1 << j) for j,label in enumerate(obj.labels))
        self.mask_cache = {}
        self.identities = {s.key for s in signatures}

    def block_matrix(self, csfs, *, max_determinants):
        sigs = tuple(csf_signature(c) for c in csfs)
        if not sigs or any(s.key not in self.identities for s in sigs):
            raise ValueError('spinor block does not belong to its declared parent CSF space')
        sectors = {(sum(q for _, _, q in s.pattern), s.total_J2) for s in sigs}
        if len(sectors) != 1:
            raise ValueError('one N/J block required')
        j2 = sigs[0].total_J2
        raw_bound = sum(dict(pattern_m_counts(p)).get(j2, 0) for p in {s.pattern for s in sigs})
        if raw_bound > max_determinants:
            raise ValueError('complete magnetic spinor determinant cap exceeded')
        if self.projection_cache is not None:
            dets, projection = self._cached_projection(sigs, j2, max_determinants)
            return self._project_matrix(dets, projection)
        vectors = []
        order = {key:i for i,key in enumerate(self.keys)}
        for sig in sigs:
            vector = csf_magnetic_vector(sig, sig.total_J2, self.keys)
            if any(mask & self.old_frozen_mask != self.old_frozen_mask for mask, _ in vector):
                raise ValueError('CSF opens a contracted frozen shell')
            # csf_magnetic_vector ALWAYS sorts its orbital layout. Convert
            # its exterior-product phase as well as the occupation bitmask.
            phase = _permutation_sign(tuple(sorted(sig.pattern)), order)
            converted = []
            for mask, value in vector:
                if mask not in self.mask_cache:
                    self.mask_cache[mask] = sum(destination for source,destination in self.active_remap
                                                if mask & source)
                converted.append((self.mask_cache[mask], phase*value))
            vectors.append(tuple(converted))
        dets = tuple(sorted({mask for v in vectors for mask, _ in v}))
        if len(dets) > max_determinants:
            raise ValueError('complete magnetic spinor determinant cap exceeded')
        position = {mask: i for i, mask in enumerate(dets)}
        rows, cols, values = [], [], []
        for j, vector in enumerate(vectors):
            for mask, value in vector:
                if value != 0:
                    rows.append(position[mask])
                    cols.append(j)
                    values.append(value)
        projection = csr_matrix((values, (rows, cols)), shape=(len(dets), len(sigs)))
        return self._project_matrix(dets, projection)

    def _cached_projection(self, sigs, j2, max_determinants):
        active = tuple(dict.fromkeys(label[:2] for label in self.integrals.labels))
        frozen = self.keys[:len(self.keys)-len(active)]
        return self.projection_cache.project_active(sigs, j2, frozen_keys=frozen,
            active_keys=active, max_determinants=max_determinants)

    def _project_matrix(self, dets, projection):
        ptr, columns = array('q', [0]), array('q')
        current = 0
        for row, col in _connected_pairs(dets, triangular=True):
            while current < row:
                ptr.append(len(columns))
                current += 1
            columns.append(col)
        while current < len(dets):
            ptr.append(len(columns))
            current += 1
        ptr = np.frombuffer(ptr, dtype=np.int64)
        columns = np.frombuffer(columns, dtype=np.int64)
        numeric = slater_values(np.asarray(dets, dtype=np.uint64), self.h, self.g, ptr, columns)
        upper = csr_matrix((numeric, columns, ptr), shape=(len(dets), len(dets)))
        determinant_H = upper+upper.T-diags(upper.diagonal(), format='csr')
        h = (projection.T @ (determinant_H @ projection)).toarray()
        h[np.diag_indices(len(h))] += self.integrals.frozen_energy_Ha
        return .5*(h+h.T)
