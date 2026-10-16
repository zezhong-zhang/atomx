"""Bounded reuse of canonical signed CSF bases across scalar consumers."""

import numpy as np
from scipy.sparse import csr_matrix

from atomx.angular.csf_hamiltonian import _permutation_sign, pattern_m_counts
from atomx.angular.one_body_tensor import TensorProjectionCache, magnetic_csf_projection


class MagneticCSFProjectionCache(TensorProjectionCache):
    """Reuse complete angular bases, independently of radial/root coefficients.

    Consumers preflight construction scratch and bind CSF/state identities.
    This uses the existing bounded projection-cache accounting and eviction;
    an oversized entry is calculated without retention. Only exact zero
    angular entries are absent. No root, radial or physical channel is removed.
    """

    def project(self, block, m, keys, coefficients=None):
        """Return canonical masks/index and owned-root or immutable CSR values.

        Keys are canonicalized to the angular engine's sorted spinor order.
        Complete signed coefficient columns are applied after retrieving the
        basis. Returned CSR wrappers cannot mutate a cached object's arrays
        or attributes. Coefficients are never part of the retained cache.
        """
        block, keys = tuple(block), tuple(sorted(keys))
        if coefficients is not None:
            coefficients = np.asarray(coefficients)
            if (coefficients.ndim not in (1, 2) or coefficients.shape[0] != len(block)
                    or coefficients.dtype.kind not in 'biufc' or not np.isfinite(coefficients).all()):
                raise ValueError('finite complete signed CSF coefficient columns required')
        dets, index, basis = self._get(block, m, keys, None,
                            lambda: magnetic_csf_projection(block, m, keys))
        if coefficients is not None:
            return dets, index, basis @ coefficients
        view = csr_matrix((basis.data, basis.indices, basis.indptr), shape=basis.shape, copy=False)
        return dets, index, view

    def project_active(self, block, m, *, frozen_keys, active_keys, max_determinants):
        """Contract filled shells and reorder the complete signed angular basis.

        Explicit key order is the consumer's spinor order, not a cache key.
        Exterior-product phases and determinant bit positions both transform.
        No radial data, Hamiltonian or reference state is invented here.
        Consumers must also admit their projection-construction memory/work.
        """
        block = tuple(block)
        frozen_keys, active_keys = tuple(frozen_keys), tuple(active_keys)
        keys = frozen_keys+active_keys
        if (not block or not active_keys or len(set(keys)) != len(keys)
                or isinstance(max_determinants, (bool, np.bool_))
                or not isinstance(max_determinants, (int, np.integer))
                or max_determinants < 1):
            raise ValueError('explicit distinct frozen/active keys and determinant cap required')
        if any((n, k) not in keys for sig in block for n, k, _ in sig.pattern):
            raise ValueError('CSF orbital outside declared frozen/active layout')
        if any((*key, 2*abs(key[1])) not in sig.pattern
               for sig in block for key in frozen_keys):
            raise ValueError('CSF opens a contracted frozen shell')
        raw_bound = sum(dict(pattern_m_counts(pattern)).get(m, 0)
                        for pattern in {sig.pattern for sig in block})
        if raw_bound > max_determinants:
            raise ValueError('complete magnetic spinor determinant cap exceeded')
        masks, _, basis = self.project(block, m, keys)
        sorted_labels = tuple((n, k, value) for n, k in sorted(keys)
                              for value in range(-2*abs(k)+1, 2*abs(k), 2))
        old_index = {label: i for i, label in enumerate(sorted_labels)}
        target_labels = tuple((n, k, value) for n, k in active_keys
                              for value in range(-2*abs(k)+1, 2*abs(k), 2))
        remap = tuple((1 << old_index[label], 1 << i)
                      for i, label in enumerate(target_labels))
        converted = [sum(destination for source, destination in remap if mask & source)
                     for mask in masks]
        if len(set(converted)) != len(converted) or len(converted) > max_determinants:
            raise ValueError('complete magnetic spinor determinant cap or mapping failed')
        row_order = np.argsort(converted)
        dets = tuple(converted[i] for i in row_order)
        order = {key: i for i, key in enumerate(keys)}
        phases = np.array([_permutation_sign(tuple(sorted(sig.pattern)), order) for sig in block])
        projection = basis[row_order].multiply(phases).tocsr()
        projection.sort_indices()
        return dets, projection
