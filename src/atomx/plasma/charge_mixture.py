"""Charge-neutral finite-level Saha EOS with excitation and ionization energy.

Every charge partition and adjacent ionization threshold is caller-supplied.
No hydrogenic partition, density cutoff or missing charge state is invented.
This ideal, finite bound-level LTE model has no continuum lowering/nonideal EOS.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp

from atomx.constants import CODATA, DEFAULT_CODATA_VERSION

_C = CODATA[DEFAULT_CODATA_VERSION]


@dataclass(frozen=True)
class ChargePartition:
    charge: int
    energies_eV: np.ndarray
    statistical_weights: np.ndarray
    state_ids: tuple[str, ...]
    source_id: str


class SahaMixture:
    """Self-consistent electron density and thermodynamic derivatives in SI/eV.

    Ionization thresholds connect neighboring ground states. Energy zero is
    the lowest included charge's ground state. All charge fractions sum to one;
    truncation at the charge boundaries remains an explicit physical limitation.
    """

    def __init__(self, partitions, ionization_potentials_eV, *, threshold_source_id):
        partitions = tuple(partitions)
        if not partitions or not threshold_source_id:
            raise ValueError("explicit charge partitions and threshold provenance required")
        charges = np.array([p.charge for p in partitions])
        if (charges.dtype.kind not in "iu" or charges[0] < 1
                or np.any(np.diff(charges) != 1)):
            raise ValueError("positive contiguous explicitly supplied charges required")
        ips = np.array(ionization_potentials_eV, dtype=float, copy=True)
        if ips.shape != (len(charges) - 1,) or np.any(~np.isfinite(ips)) or np.any(ips <= 0):
            raise ValueError("one positive ground-state ionization threshold per adjacent charge required")
        offsets = np.concatenate(([0.0], np.cumsum(ips)))
        energies, weights, state_charges, ids, slices = [], [], [], [], []
        owned_partitions = []
        start = 0
        for i, p in enumerate(partitions):
            E = np.array(p.energies_eV, dtype=float, copy=True)
            g = np.array(p.statistical_weights, dtype=float, copy=True)
            if (E.ndim != 1 or E.size == 0 or E.shape != g.shape or E.size != len(p.state_ids)
                    or np.any(~np.isfinite(E)) or np.any(E < 0) or E.min() != 0
                    or np.any(~np.isfinite(g)) or np.any(g <= 0)
                    or len(set(p.state_ids)) != E.size or not p.source_id):
                raise ValueError("complete declared finite partition with unique states and source identity required")
            if i < ips.size and np.any(ips[i] <= E):
                raise ValueError("partition contains levels above its ground-continuum threshold; explicit resonance/DR treatment required")
            E.setflags(write=False)
            g.setflags(write=False)
            owned_partitions.append(ChargePartition(int(p.charge), E, g, tuple(p.state_ids), p.source_id))
            energies.extend(E + offsets[i])
            weights.extend(g)
            state_charges.extend([p.charge] * E.size)
            ids.extend((p.charge, key) for key in p.state_ids)
            slices.append(slice(start, start + E.size))
            start += E.size
        self.energy = np.array(energies)
        self.partitions = tuple(owned_partitions)
        self.g = np.array(weights)
        self.q = np.array(state_charges)
        self.slices = tuple(slices)
        self.state_ids = tuple(ids)
        self.charges = charges
        self.ionization_potentials_eV = ips
        self.threshold_source_id = threshold_source_id
        for array in (self.energy, self.g, self.q, self.charges, self.ionization_potentials_eV):
            array.setflags(write=False)
        identity = dict(threshold_source_id=threshold_source_id,
                        partition_sources=[p.source_id for p in self.partitions],
                        state_ids=self.state_ids)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
        for array in (self.energy, self.g, self.q, self.ionization_potentials_eV):
            digest.update(array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes())
        self.source_id = digest.hexdigest()

    def equilibrium(self, temperature_eV, ion_density_per_m3):
        """LTE state including exact finite-model dU/dT and dU/dln(n_i).

        U is per nucleus in eV, including ideal electrons/ions, excitation and
        cumulative ionization energy. Derivatives include charge-neutrality
        response, needed when density changes during expansion.
        """
        T, ni = temperature_eV, ion_density_per_m3
        if not all(np.isfinite(v) and v > 0 for v in (T, ni)):
            raise ValueError("positive finite temperature and ion density required")
        # Two electron spin states; kT is converted from eV directly to joules.
        log_s = np.log(2) + 1.5 * np.log(2 * np.pi * _C["_me"] * _C["_e"] * T / _C["_hplanck"]**2)
        k = self.q - self.charges[0]
        base = np.log(self.g) - self.energy / T + k * log_s

        def populations(log_ne):
            lw = base - k * log_ne
            return np.exp(lw - logsumexp(lw))

        if self.charges.size == 1:
            log_ne = np.log(ni) + np.log(self.charges[0])
        else:
            def residual(value):
                p = populations(value)
                return value - np.log(ni) - np.log(p @ self.q)

            lo = np.log(ni) + np.log(self.charges[0])
            hi = np.log(ni) + np.log(self.charges[-1])
            log_ne = brentq(residual, lo - 1e-12, hi + 1e-12, xtol=1e-13)
        p = populations(log_ne)
        mean_q, mean_E = float(p @ self.q), float(p @ self.energy)
        dq, dE = self.q - mean_q, self.energy - mean_E
        var_q, var_E, covariance = float(p @ dq**2), float(p @ dE**2), float(p @ (dq * dE))
        log_ne_T = (covariance / T**2 + 1.5 * var_q / T) / (mean_q + var_q)
        q_T = mean_q * log_ne_T
        E_T = var_E / T**2 + 1.5 * covariance / T - covariance * log_ne_T
        density_factor = mean_q / (mean_q + var_q)
        U_density = -density_factor * (1.5 * T * var_q + covariance)
        U = 1.5 * (1 + mean_q) * T + mean_E
        cv = 1.5 * (1 + mean_q) + 1.5 * T * q_T + E_T
        return dict(populations=p, charge_fractions=np.array([p[s].sum() for s in self.slices]),
                    electron_density_per_m3=ni * mean_q, mean_charge=mean_q,
                    mean_charge_squared=float(p @ self.q**2), internal_energy_eV=U,
                    heat_capacity_eV_per_eV=cv, energy_derivative_log_density_eV=U_density,
                    mean_charge_derivative_T=q_T,
                    pressure_Pa=ni * (1 + mean_q) * T * _C["_e"],
                    neutrality_log_residual=log_ne - np.log(ni * mean_q))
