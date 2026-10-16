"""Full CI radial integral weights, shared by both radial representations."""
from dataclasses import dataclass

import numpy as np

from atomx.angular import csf_hamiltonian as exact_ci
from atomx.physics.csf import matrix


@dataclass(frozen=True)
class _StreamedBlock:
    """Determinant/projection identity without an angular table or pair array."""

    patterns: tuple
    j2: int
    dets: tuple
    spinor_labels: tuple


class CIRadialDensity:
    """Signed DC radial weights in an explicit, ordered CSF basis.

    ``angular_storage='streamed'`` contracts connected determinant pairs
    directly into radial keys. The default cached angular table is unchanged.
    Bilinears retain every nonzero signed term, without a magnitude cutoff.
    Raw vectors do not certify an eigenstate, normalization, Hamiltonian or
    radial snapshot; those identities remain the consuming caller's contract.
    """

    def __init__(self, csfs, *, max_determinants=None, angular_storage='cached'):
        if angular_storage not in ('cached', 'streamed'):
            raise ValueError("angular_storage must be 'cached' or 'streamed'")
        self.angular_storage = angular_storage
        if max_determinants is None:
            max_determinants = exact_ci.MAX_BLOCK_DETERMINANTS
        if (isinstance(max_determinants, (bool, np.bool_))
                or not isinstance(max_determinants, (int, np.integer))
                or max_determinants < 1):
            raise ValueError('max_determinants must be a positive integer')
        self.max_determinants = int(max_determinants)
        self.signatures = tuple(exact_ci.csf_signature(c) for c in csfs)
        self.blocks = []
        for _, indices in matrix._group_csf_indices(csfs):
            signatures = [exact_ci.csf_signature(csfs[i]) for i in indices]
            groups = {}
            for sig in signatures:
                groups.setdefault(sig.pattern, []).append(sig)
            if angular_storage == 'cached':
                structure = exact_ci.block_terms(
                    tuple(groups), signatures[0].total_J2,
                    max_determinants=self.max_determinants)
            else:
                patterns, dets, _, labels = exact_ci._block_basis(
                    tuple(groups), signatures[0].total_J2, self.max_determinants)
                structure = _StreamedBlock(patterns, signatures[0].total_J2, dets, labels)
            grouped = tuple(tuple(groups[p]) for p in structure.patterns)
            projection = exact_ci.csf_block_matrix(
                structure.patterns, structure.j2, grouped, sparse=True)
            original = {sig: i for i, sig in enumerate(signatures)}
            columns = [indices[original[sig]] for group in grouped for sig in group]
            self.blocks.append((structure, projection, columns))

    def contract(self, result, state_weights=None):
        ground_average = state_weights is None
        if state_weights is None:
            ground = abs(result.state_energies-result.ground_energy) < 1e-10
            state_weights = ground / ground.sum()
        else:
            state_weights = np.asarray(state_weights, dtype=float)
            if (state_weights.shape != result.state_energies.shape
                    or not np.isfinite(state_weights).all() or np.any(state_weights < 0)
                    or not np.isclose(state_weights.sum(), 1., rtol=0, atol=1e-12)):
                raise ValueError('CI state weights must be normalized, finite and nonnegative')
        selected = state_weights > 0
        if self.angular_storage == 'streamed':
            vectors = result.mixing_coefficients[:, selected]
            terms = self.contract_bilinears(vectors, vectors)
            weights = {}
            for state_weight, state in zip(state_weights[selected], terms, strict=True):
                for key, value in state.items():
                    weights[key] = weights.get(key, 0.) + state_weight * value
            # Preserve the legacy ensemble API's cutoff. Complete signed
            # derivative consumers explicitly use the unfiltered bilinears.
            return {key: value for key, value in weights.items() if abs(value) > 1e-15}
        weights = {}
        for structure, projection, columns in self.blocks:
            amplitudes = projection @ result.mixing_coefficients[columns][:, selected]
            products = (amplitudes[structure.pairs[:, 0]] *
                        amplitudes[structure.pairs[:, 1]])
            values = (np.mean(products, axis=1) if ground_average else
                      products @ state_weights[selected])
            coefficients = np.asarray(structure.table.T @ values).ravel()
            for key, coefficient in zip(structure.keys, coefficients, strict=True):
                if abs(coefficient) > 1e-15:
                    weights[key] = weights.get(key, 0.) + coefficient
        return weights

    def contract_bilinear(self, left, right):
        """Signed weights of left.T H right in this exact CSF order.

        Transition amplitudes need not be normalized or positive. Internal
        symmetry blocks remain separate; structural scalar zeros are exact.
        """
        left, right = np.asarray(left), np.asarray(right)
        shape = (len(self.signatures),)
        if (left.shape != shape or right.shape != shape
                or np.iscomplexobj(left) or np.iscomplexobj(right)
                or not np.isfinite(left).all() or not np.isfinite(right).all()):
            raise ValueError('bilinear CI amplitudes require finite real CSF vectors')
        if self.angular_storage == 'streamed':
            return self._streamed_bilinears(left[:, None], right[:, None])[0]
        weights = {}
        for structure, projection, columns in self.blocks:
            a, b = projection @ left[columns], projection @ right[columns]
            i, j = structure.pairs.T
            products = .5 * (a[i] * b[j] + b[i] * a[j])
            values = np.asarray(structure.table.T @ products).ravel()
            for key, value in zip(structure.keys, values, strict=True):
                if value != 0:
                    weights[key] = weights.get(key, 0.) + float(value)
        return weights

    def contract_bilinears(self, left, right):
        """Independent signed bilinears, one per pair of input columns.

        Arrays have shape (number of ordered CSFs, number of bilinears).
        The streamed path traverses each complete connection graph once,
        sharing angular work across columns. It never forms a pair-by-state
        array. Zero outputs alone are removed; arbitrary real signed vectors
        and independently scaled columns are supported, without normalization.
        """
        left, right = np.asarray(left), np.asarray(right)
        if (left.ndim != 2 or left.shape != right.shape
                or left.shape[0] != len(self.signatures) or left.shape[1] == 0
                or np.iscomplexobj(left) or np.iscomplexobj(right)
                or not np.isfinite(left).all() or not np.isfinite(right).all()):
            raise ValueError('bilinear CI amplitudes require matching finite real CSF column arrays')
        if self.angular_storage == 'streamed':
            return self._streamed_bilinears(left, right)
        return tuple(self.contract_bilinear(a, b) for a, b in zip(left.T, right.T, strict=True))

    def _streamed_bilinears(self, left, right):
        """Use the Hamiltonian's exact signed core/pair algebra, row by row."""
        weights = tuple({} for _ in range(left.shape[1]))
        for structure, projection, columns in self.blocks:
            a, b = projection @ left[columns], projection @ right[columns]
            pair_terms = exact_ci._block_pair_terms(structure.dets, structure.spinor_labels)
            for i, j in exact_ci._connected_pairs(structure.dets):
                values = tuple(.5 * (ai * bj + bi * aj)
                               for ai, bj, bi, aj in zip(a[i], b[j], b[i], a[j], strict=True))
                # Skip only an exact structural zero of all requested bilinears.
                if not any(values):
                    continue
                for key, coefficient in pair_terms(structure.dets[i], structure.dets[j]):
                    for state, value in zip(weights, values, strict=True):
                        state[key] = state.get(key, 0.) + coefficient * value
        return tuple({key: float(value) for key, value in state.items() if value != 0}
                     for state in weights)
