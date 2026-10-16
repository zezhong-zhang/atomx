"""Bounded complete-CSF orbital objectives with explicit symmetry-root ensembles."""

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from atomx.angular.projection_cache import MagneticCSFProjectionCache
from atomx.methods.ci_blocks import plan_ci_blocks
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.physics.csf.effective_one_body import CIReferenceBinding
from atomx.physics.csf.spinor_ci import SpinorCIHamiltonian
from atomx.physics.csf.state_average import CIStateAverage


def _owned(array):
    array = np.asarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True, init=False)
class BlockedOrbitalTrial:
    """Owned selected eigenvectors from complete native DC symmetry matrices.

    This is an orbital-objective trial, not a complete blocked spectrum for
    continuum/thermodynamic consumers. Build through the functional below.
    """

    energy_Ha: float
    energies_Ha: np.ndarray
    coefficients: np.ndarray
    state_ids: tuple
    weights: np.ndarray
    radial_binding: CIReferenceBinding
    objective_identity: tuple
    full_H_residual_Ha: float
    hamiltonian_hashes: tuple

    def __init__(self, *args, **kwargs):
        raise TypeError('build an orbital trial through BlockedCSFOrbitalFunctional')


class BlockedCSFOrbitalFunctional(CSFOrbitalFunctional):
    """Fixed CIStateAverage on full signed blocks, with exact spinor RDM gradients.

    Every CSF is retained. Only explicitly selected observable roots are solved,
    with a neighboring-root degeneracy check, full-H residual and Gram gates.
    Complete radial-key accumulation is canonical across Python hash seeds;
    no signed term or weak coefficient is removed by this ordering.
    No global dense CI matrix/vector bank is assembled. Bare DC orbital
    optimization and subsequent Breit/QED RCI are separate physical stages.
    The positive-energy projector and constrained convergence remain the
    numerical orbital optimizer's explicit responsibility.
    """

    def __init__(self, csfs, workspace, Z, *, state_average, reference_keys,
                 nuclear_potential=None, max_determinants=80000,
                 memory_bytes=8*1024**3, projection_cache_bytes=128*1024**2):
        if not isinstance(state_average, CIStateAverage) or state_average.states is None:
            raise ValueError('explicit fixed CIStateAverage root selectors required')
        for value in (max_determinants, memory_bytes):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError('positive integer orbital-objective caps required')
        if (isinstance(projection_cache_bytes, (bool, np.bool_))
                or not isinstance(projection_cache_bytes, (int, np.integer)) or projection_cache_bytes < 0):
            raise ValueError('nonnegative integer angular projection cache budget required')
        csfs = tuple(csfs)
        plan = plan_ci_blocks(csfs)
        if plan['estimated_peak_bytes']+projection_cache_bytes > memory_bytes:
            raise MemoryError('complete blocked orbital objective exceeds matrix/scratch cap')
        if any(r['raw_determinants'] > max_determinants for r in plan['blocks']):
            raise MemoryError('complete orbital objective exceeds determinant cap')
        sectors = {(r['J2'], r['parity']): r for r in plan['blocks']}
        for j, p, root, _ in state_average.states:
            if (j, p) not in sectors or root >= sectors[j, p]['size']:
                raise ValueError('selected orbital-objective root outside full CSF space')
        super().__init__(csfs, workspace, Z, nuclear_potential=nuclear_potential,
                         max_determinants=max_determinants, density_storage='spinor')
        self.projection_cache = (MagneticCSFProjectionCache(int(projection_cache_bytes))
                                 if projection_cache_bytes else None)
        self.radial_density.projection_cache = self.projection_cache
        self.state_average = state_average
        self.reference_keys, self.memory_bytes = tuple(reference_keys), memory_bytes
        self.plan = plan
        self.objective_identity = (tuple(s.key for s in self.radial_density.signatures),
                 state_average.states, state_average.statistical, self.reference_keys, 'bare Dirac-Coulomb')
        self.target_description = 'fixed symmetry-root ensemble of complete blocked DC CI matrices'

    def _solve(self, orbitals):
        builder = SpinorCIHamiltonian(self.csfs, orbitals, self.data, self.Z, self.ns,
                   reference_keys=self.reference_keys, nuclear_potential=self.nuclear_potential,
                   memory_bytes=min(self.memory_bytes//4, 512*1024**2),
                   projection_cache=self.projection_cache)
        selectors = self.state_average.states
        scale = max(s[3] for s in selectors)
        weights = np.array([w/scale*(j+1 if self.state_average.statistical else 1) for j, p, r, w in selectors])
        weights /= weights.sum()
        if np.any(weights == 0):
            raise ValueError('fixed orbital-objective weight underflow')
        vectors = np.zeros((len(self.csfs), len(selectors)))
        energies = np.empty(len(selectors))
        hashes = []
        residual = 0.
        for row in self.plan['blocks']:
            sector = row['J2'], row['parity']
            requested = [i for i, s in enumerate(selectors) if s[:2] == sector]
            if not requested:
                continue
            indices = row['indices']
            h = builder.block_matrix([self.csfs[i] for i in indices], max_determinants=self.max_determinants)
            last = min(len(h)-1, max(selectors[i][2] for i in requested)+1)
            shift = float(np.rint(h[0, 0]))
            q = h.copy()
            q[np.diag_indices(len(h))] -= shift
            e, c = eigh(q, subset_by_index=[0, last], driver='evr')
            e += shift
            local_weights = np.zeros(len(e))
            for i in requested:
                root = selectors[i][2]
                energies[i], vectors[indices, i] = e[root], c[:, root]
                local_weights[root] = weights[i]
            for i in range(len(e)):
                same = abs(e-e[i]) < 1e-10
                if np.ptp(local_weights[same]) != 0:
                    raise ValueError('orbital objective has unequal weights in an exactly degenerate root cluster')
            defect = float(np.max(abs(h@c-c*e)))
            if defect > 1e-8 or np.max(abs(c.T@c-np.eye(len(e)))) > 1e-10:
                raise RuntimeError('complete blocked orbital-objective H residual/Gram gate failed')
            residual = max(residual, defect)
            hashes.append((sector, hashlib.sha256(h.tobytes()).hexdigest()))
        trial = object.__new__(BlockedOrbitalTrial)
        values = dict(energy_Ha=float(weights@energies), energies_Ha=_owned(energies),
            coefficients=_owned(vectors), weights=_owned(weights), state_ids=tuple(s[:3] for s in selectors),
            radial_binding=CIReferenceBinding.capture([(*orbitals[k], k[1]) for k in self.keys],
                       {}, self.data, self.Z, self.ns, self.nuclear_potential),
            objective_identity=self.objective_identity, full_H_residual_Ha=residual, hamiltonian_hashes=tuple(hashes))
        for key, value in values.items():
            object.__setattr__(trial, key, value)
        return trial

    def ground_variational_upper_change(self, reference, trial, reference_ci, trial_ci,
                                        *, increments=None):
        """Fixed-old-CI ground Rayleigh change on the same orbital manifold.

        For isolated root zero in each selected complete symmetry block and
        nonnegative fixed weights, Rayleigh-Ritz bounds the optimized trial
        ground energy by the old CI vector's trial Rayleigh quotient. Contract
        all signed old CI density terms and subtract the full metric constraint
        defect, as in ``constrained_energy_difference``. Both endpoint CI
        solutions are still required and bound to their own physical points.

        Both endpoints must retain the native homogeneous Dirac boundary
        coordinates. Unsupported boundary extensions return ``None``.
        This opt-in comparison is an exact finite-cell variational statement
        on the orthogonal manifold, not a floating-point forward-error bound.
        It is neither the optimized energy difference nor a stationarity gate.
        Excited roots or nonphysical constraint defects return ``None``.
        """
        for orbitals, result in ((reference, reference_ci), (trial, trial_ci)):
            if (not isinstance(result, BlockedOrbitalTrial)
                    or result.objective_identity != self.objective_identity):
                raise ValueError('blocked trial belongs to a different physical objective')
            result.radial_binding.validate(self.keys, orbitals, self.data, self.Z,
                                          self.ns, self.nuclear_potential)
        from atomx.numerics.bspline.dirac_assembly import _get_keep_grid

        keep, _ = _get_keep_grid(self.ns, self.data.get('nsp', self.ns),
                                self.data.get('nsq', self.ns), 1, 2)
        excluded = np.ones(2*self.ns, dtype=bool)
        excluded[keep] = False
        if any(np.any(np.concatenate(orbitals[key])[excluded] != 0)
               for orbitals in (reference, trial) for key in self.keys):
            return None
        if (any(root != 0 for _, _, root in reference_ci.state_ids)
                or not np.array_equal(reference_ci.weights, trial_ci.weights)
                or np.any(reference_ci.weights < 0)):
            return None
        terms = self.radial_density.contract_bilinears(
            reference_ci.coefficients, reference_ci.coefficients)
        weights = {key: sum(w*row.get(key, 0.) for w, row in
                            zip(reference_ci.weights, terms, strict=True))
                   for key in sorted(set().union(*terms))}
        weights = {key: value for key, value in weights.items() if value != 0}
        return self._constrained_density_energy_difference(
            reference, trial, weights, increments=increments)

    def evaluate(self, orbitals, *, gradient=True, ci_result=None, gradient_keys=None):
        trial = self._solve(orbitals) if ci_result is None else ci_result
        if not isinstance(trial, BlockedOrbitalTrial) or trial.objective_identity != self.objective_identity:
            raise ValueError('blocked trial belongs to a different physical objective')
        trial.radial_binding.validate(self.keys, orbitals, self.data, self.Z, self.ns, self.nuclear_potential)
        if not gradient:
            return trial.energy_Ha, trial
        terms = self.radial_density.contract_bilinears(trial.coefficients, trial.coefficients)
        weights = {key: sum(w*row.get(key, 0.) for w, row in zip(trial.weights, terms, strict=True))
                   for key in sorted(set().union(*terms))}
        weights = {key: value for key, value in weights.items() if value != 0}
        self.last_state_weights = trial.weights
        derivative = self.gradient_from_weights(orbitals, weights, gradient_keys=gradient_keys)
        return trial.energy_Ha, derivative, trial
