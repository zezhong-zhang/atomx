"""Frozen-reference CI ensemble identities in changing common radial bases.

This is an explicit optimization target, not a resonance-pole solver. A
target is resolved against the same owned reference on every trial, including
rejected trials. Its selection cannot drift with line-search call order.
"""
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from atomx.physics.csf.levels import CILevel, ci_level_blocks, normalize_ci_weights
from atomx.physics.csf.overlap import ci_subspace_overlaps
from atomx.physics.csf.state_average import CIStateAverage


class CITrackingAmbiguity(ValueError):
    """A trial cannot uniquely retain the declared physical CI subspace."""


def _owned_array(value):
    array = np.array(value, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class _Spectrum:
    state_energies: np.ndarray
    mixing_coefficients: np.ndarray


@dataclass(frozen=True, init=False)
class CIBasisTracking:
    """Keep the reference CIStateAverage weights on physical eigensubspaces.

    Equal normalized weights within each J/parity are grouped into a single
    projector, including complete exact-degenerate clusters. Candidate roots
    maximize its projected probability. A unique score boundary and every
    squared principal overlap > 1/2 are required. Colliding unequal-weight
    groups and unequal weights at a trial degeneracy reject the trial.

    This selected-eigenvalue objective is locally differentiable away from
    selection boundaries; it is not a globally smooth diabatic functional.
    Weights never depend on experimental energy, and the reference never
    updates after an accepted step. Large physical changes can therefore be
    refused; a new reference requires an explicit new target. No near-root
    smoothing, coherent superposition gradient, or complex pole is implied.
    """

    state_average: CIStateAverage
    _csfs: tuple
    _ci: _Spectrum
    _orbitals: object
    _metric: object
    _groups: tuple
    _pair_cap: int

    def __init__(self, state_average, csfs, result, orbitals, metric,
                 *, max_determinant_pairs=1_000_000):
        if not isinstance(state_average, CIStateAverage):
            raise TypeError('tracking requires CIStateAverage')
        weights = state_average.weights(csfs, result)
        _, blocks = ci_level_blocks(csfs, result, require_full=True)
        groups = []
        for (j2, parity), columns in blocks.items():
            for weight in sorted(set(weights[columns]) - {0.0}):
                levels = tuple(CILevel(j2, parity, i) for i, col in enumerate(columns)
                               if weights[col] == weight)
                groups.append((levels, float(weight)))
        owned_ci = _Spectrum(_owned_array(result.state_energies),
                             _owned_array(result.mixing_coefficients))
        owned_waves = MappingProxyType({key: tuple(_owned_array(v) for v in pair)
                                        for key, pair in orbitals.items()})
        owned_metric = MappingProxyType({k: _owned_array(s) for k, s in metric.items()})
        # Validate basis, cluster closure and pair budget before optimization.
        # Only selected bra subspaces are needed; ket spans their symmetry blocks.
        levels = tuple(level for group, _ in groups for level in group)
        candidates = tuple(CILevel(j, p, i) for (j, p), columns in blocks.items()
                           if (j, p) in {(s.J2, s.parity) for s in levels}
                           for i in range(len(columns)))
        ci_subspace_overlaps(csfs, owned_ci, owned_waves, levels,
                             csfs, owned_ci, owned_waves, candidates, owned_metric,
                             max_determinant_pairs=max_determinant_pairs)
        for name, value in (('state_average', state_average), ('_csfs', tuple(deepcopy(csfs))),
                            ('_ci', owned_ci), ('_orbitals', owned_waves),
                            ('_metric', owned_metric), ('_groups', tuple(groups)),
                            ('_pair_cap', int(max_determinant_pairs))):
            object.__setattr__(self, name, value)

    def resolve(self, csfs, result, orbitals, metric):
        """Return fixed identity weights and diagnostics without changing state."""
        if set(metric) != set(self._metric) or any(
                not np.array_equal(metric[k], self._metric[k]) for k in metric):
            raise ValueError('tracking requires the reference radial geometry metric')
        energies, blocks = ci_level_blocks(csfs, result, require_full=True)
        reference = tuple(level for group, _ in self._groups for level in group)
        symmetries = {(s.J2, s.parity) for s in reference}
        candidates = tuple(CILevel(j, p, i) for (j, p), columns in blocks.items()
                           if (j, p) in symmetries for i in range(len(columns)))
        overlaps = ci_subspace_overlaps(
            self._csfs, self._ci, self._orbitals, reference,
            csfs, result, orbitals, candidates, metric,
            max_determinant_pairs=self._pair_cap)
        weights = np.zeros(len(energies))
        diagnostics = []
        offset = 0
        for group, weight in self._groups:
            symmetry = group[0].J2, group[0].parity
            positions = [i for i, s in enumerate(candidates) if (s.J2, s.parity) == symmetry]
            matrix = overlaps[offset:offset+len(group), positions]
            offset += len(group)
            if matrix.shape[1] < len(group):
                raise CITrackingAmbiguity('tracking target rank exceeds the trial symmetry space')
            scores = np.sum(matrix**2, axis=0)
            order = np.argsort(-scores, kind='stable')
            chosen = order[:len(group)]
            if len(order) > len(group) and scores[chosen[-1]]-scores[order[len(group)]] <= 1e-10:
                raise CITrackingAmbiguity('ambiguous CI subspace probability boundary')
            principal = np.linalg.svd(matrix[:, chosen], compute_uv=False)**2
            if principal.min() <= .5 or principal.max() > 1+1e-8:
                raise CITrackingAmbiguity('CI subspace lost its majority principal overlap')
            selectors = [candidates[positions[i]] for i in chosen]
            columns = [int(blocks[s.J2, s.parity][s.root]) for s in selectors]
            if np.any(weights[columns] != 0):
                raise CITrackingAmbiguity('unequal-weight CI identities select colliding trial roots')
            weights[columns] = weight
            diagnostics.append(dict(
                reference_levels=[(s.J2, s.parity, s.root) for s in group],
                selected_levels=[(s.J2, s.parity, s.root) for s in selectors],
                per_level_weight=weight,
                principal_overlap_squared=principal.tolist(),
                projected_probability=scores[chosen].tolist()))
        try:
            weights = normalize_ci_weights(weights, energies, blocks)
        except ValueError as error:
            raise CITrackingAmbiguity(str(error)) from error
        weights.flags.writeable = False
        return weights, diagnostics
