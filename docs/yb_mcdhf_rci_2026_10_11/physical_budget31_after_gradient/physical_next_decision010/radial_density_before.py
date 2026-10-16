"""Full CI radial integral weights, shared by both radial representations."""
import numpy as np

from atomx.angular import csf_hamiltonian as exact_ci
from atomx.physics.csf import matrix


class CIRadialDensity:
    """Contract its exact angular table through sparse signed CSF projections."""

    def __init__(self, csfs, *, max_determinants=None):
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
            structure = exact_ci.block_terms(
                tuple(groups), signatures[0].total_J2,
                max_determinants=self.max_determinants)
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
