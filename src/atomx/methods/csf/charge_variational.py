"""Shared-orbital charge-sector CI functional composed from existing kernels."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import comb
from types import MappingProxyType

import numpy as np

from atomx.config.csf import enumerate_jj_orbital_space
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.physics.csf.radial_density import CIRadialDensity
from atomx.physics.csf.state_average import CIChargeStateAverage


def orbital_signature(orbitals):
    """Exact content identity for accepted CI trials, scoped to one functional."""
    return tuple(
        (key, np.asarray(p).tobytes(), np.asarray(q).tobytes())
        for key, (p, q) in sorted(orbitals.items())
    )


@dataclass(frozen=True)
class ChargeCITrial:
    parent: object
    sectors: dict
    signature: tuple
    owner: object
    tracking: Mapping | None = None


class ChargeStateOrbitalFunctional(CSFOrbitalFunctional):
    """Minimize a fixed charge/level ensemble in common orthonormal spinors.

    One-/two-body RDMs are summed before differentiating radial integrals.
    Every line-search trial diagonalizes all target sectors. Its accepted
    CI spectra are reused for the gradient only with exact orbital identity.
    The final parent spectrum remains distinct from the averaged objective.

    ``level_tracking={N: CIBasisTracking(...)}`` explicitly replaces sorted
    roots by frozen-reference physical subspaces in those sectors. Matching
    weights are part of the accepted trial and reused with its signed RDM.
    ``dense_cluster_gap`` is an explicit numerical Rayleigh--Ritz refinement
    of complete real CI blocks; the declared ensemble remains unchanged.
    """

    def __init__(
        self, csfs, bsp_data, Z, nuclear_potential=None, cache=None, *, state_average,
        level_tracking=None, dense_cluster_gap=None
    ):
        if not isinstance(state_average, CIChargeStateAverage):
            raise TypeError("charge functional requires CIChargeStateAverage")
        numbers = {sum(s.occupation for s in c.subshells) for c in csfs}
        if len(numbers) != 1:
            raise ValueError("one fixed parent electron number required")
        self.parent_electrons = int(numbers.pop())
        self.charge_sectors = state_average.normalized_sectors(self.parent_electrons)
        from atomx.physics.csf.tracking import CIBasisTracking
        self.sector_tracking = MappingProxyType({} if level_tracking is None else dict(level_tracking))
        targets = {n: target for n, _, target in self.charge_sectors}
        if any(n not in targets or not isinstance(tracker, CIBasisTracking)
               or tracker.state_average != targets[n] for n, tracker in self.sector_tracking.items()):
            raise ValueError('level_tracking must reference the declared charge-sector targets')
        keys = tuple(sorted({(s.n, s.kappa) for c in csfs for s in c.subshells}))
        spinors = sum(2 * abs(k) for _, k in keys)
        size = sum(c.total_J2 + 1 for c in csfs)
        size += sum(
            comb(spinors, n)
            for n, _, _ in self.charge_sectors
            if n != self.parent_electrons
        )
        if size > state_average.magnetic_state_cap:
            raise ValueError(
                f"charge objective needs {size} magnetic states; cap={state_average.magnetic_state_cap}"
            )
        parent_target = next(
            target for n, _, target in self.charge_sectors if n == self.parent_electrons
        )
        super().__init__(
            csfs, bsp_data, Z, nuclear_potential, cache, state_average=parent_target,
            dense_cluster_gap=dense_cluster_gap
        )
        self.charge_state_average = state_average
        self.target_description = (
            "fixed charge-sector CI Hamiltonian average in shared orthonormal orbitals"
        )
        if self.sector_tracking:
            self.target_description += '; frozen-reference physical CI subspaces in explicit sectors'
        self.sector_csfs = {
            n: (
                csfs
                if n == self.parent_electrons
                else enumerate_jj_orbital_space(
                    keys, n, magnetic_state_cap=state_average.magnetic_state_cap
                )
            )
            for n, _, _ in self.charge_sectors
        }
        self.sector_densities = {
            n: (
                self.radial_density
                if n == self.parent_electrons
                else CIRadialDensity(space)
            )
            for n, space in self.sector_csfs.items()
        }
        self.last_sector_weights = {}
        self.last_sector_energies = {}
        self.last_trial = None
        self._trial_owner = object()

    def evaluate(self, orbitals, *, gradient=True, ci_result=None, gradient_keys=None):
        signature = orbital_signature(orbitals)
        if ci_result is None:
            from atomx.physics.csf.radial_snapshot import CIRadialSnapshot

            radial = CIRadialSnapshot(
                orbitals,
                self.data,
                self.Z,
                self.ns,
                nuclear_potential=self.nuclear_potential,
            )
            results = {
                n: radial.solve(
                    space,
                    self.keys,
                    convol_cache=self.cache,
                    dense_cluster_gap=self.dense_cluster_gap,
                )
                for n, space in self.sector_csfs.items()
            }
            self.last_radial_cache_statistics = radial.diagnostics()
            trial = ChargeCITrial(
                results[self.parent_electrons], results, signature, self._trial_owner
            )
        else:
            if (
                not isinstance(ci_result, ChargeCITrial)
                or ci_result.owner is not self._trial_owner
                or ci_result.signature != signature
                or set(ci_result.sectors) != set(self.sector_csfs)
            ):
                raise ValueError(
                    "charge CI trial does not match the shared orbital snapshot"
                )
            trial = ci_result
        if self.sector_tracking and trial.tracking is None:
            tracking = {
                n: tracker.resolve(self.sector_csfs[n], trial.sectors[n], orbitals, self.metric)
                for n, tracker in self.sector_tracking.items()
            }
            trial = ChargeCITrial(trial.parent, trial.sectors, signature,
                                  self._trial_owner, MappingProxyType(tracking))
        energy = 0.0
        levels = {}
        for n, charge_weight, target in self.charge_sectors:
            weights = (trial.tracking[n][0] if n in self.sector_tracking else
                       target.weights(self.sector_csfs[n], trial.sectors[n]))
            levels[n] = weights
            energy += charge_weight * float(weights @ trial.sectors[n].state_energies)
        if not gradient:
            return float(energy), trial
        combined = {}
        for n, charge_weight, _ in self.charge_sectors:
            weights = self.sector_densities[n].contract(trial.sectors[n], levels[n])
            for key, value in weights.items():
                combined[key] = combined.get(key, 0.0) + charge_weight * value
        derivative = self.gradient_from_weights(
            orbitals, combined, gradient_keys=gradient_keys
        )
        self.last_state_weights = levels[self.parent_electrons]
        self.last_sector_weights = levels
        self.last_sector_energies = {
            n: result.state_energies.copy() for n, result in trial.sectors.items()
        }
        self.last_trial = trial
        self.last_tracking_diagnostics = ({n: item[1] for n, item in trial.tracking.items()}
                                         if trial.tracking is not None else {})
        return float(energy), derivative, trial

    def objective_diagnostics(self):
        return dict(
            frozen_reference_ci_tracking=dict(self.last_tracking_diagnostics),
            charge_sector_weights={n: w for n, w, _ in self.charge_sectors},
            charge_sector_level_weights={
                n: w.tolist() for n, w in self.last_sector_weights.items()
            },
            charge_sector_energies_hartree={
                n: e.tolist() for n, e in self.last_sector_energies.items()
            },
            objective_mean_electrons=sum(n * w for n, w, _ in self.charge_sectors),
            charge_sector_magnetic_dimensions={
                n: sum(c.total_J2 + 1 for c in csfs)
                for n, csfs in self.sector_csfs.items()
            },
            charge_ci_radial_cache=dict(self.last_radial_cache_statistics),
            dense_ci_cluster_gap_hartree=self.dense_cluster_gap,
            charge_orbital_model="one common orthonormal set; fixed charge and CI level optimization weights; no independent-charge orbitals",
        )
