"""Full CI radial integral weights, shared by both radial representations."""
import numpy as np
from atomx.methods.mcdhf import exact_ci, matrix


class CIRadialDensity:
    """Contract the Hamiltonian's own angular table with CI amplitudes."""

    def __init__(self, csfs):
        self.blocks = []
        for _, indices in matrix._group_csf_indices(csfs):
            signatures = [exact_ci.csf_signature(csfs[i]) for i in indices]
            groups = {}
            for sig in signatures:
                groups.setdefault(sig.pattern, []).append(sig)
            structure = exact_ci.block_terms(tuple(groups), signatures[0].total_J2)
            grouped = tuple(tuple(groups[p]) for p in structure.patterns)
            projection = exact_ci.csf_block_matrix(structure.patterns, structure.j2, grouped)
            original = {sig: i for i, sig in enumerate(signatures)}
            columns = [indices[original[sig]] for group in grouped for sig in group]
            self.blocks.append((structure, projection, columns))

    def contract(self, result):
        ground = abs(result.state_energies-result.ground_energy) < 1e-10
        weights = {}
        for structure, projection, columns in self.blocks:
            amplitudes = projection @ result.mixing_coefficients[columns][:, ground]
            values = np.mean(amplitudes[structure.pairs[:, 0]] *
                             amplitudes[structure.pairs[:, 1]], axis=1)
            coefficients = np.asarray(structure.table.T @ values).ravel()
            for key, coefficient in zip(structure.keys, coefficients, strict=True):
                if abs(coefficient) > 1e-15:
                    weights[key] = weights.get(key, 0.) + coefficient
        return weights
