"""General N -> N-1 signed CI channels on one common orbital space.

Every daughter electron sector spans the full retained orbital Fock space.
Residual orbitals share the parent basis, optionally charge-averaged EOL, and
are fixed during channel preparation. Continuum is projected direct Hartree
DW, not nonlocal exchange or coupled-channel/Fano.
"""

import copy
from functools import lru_cache
from math import comb

import numpy as np

from atomx.angular.csf_hamiltonian import continuum_hamiltonian_terms
from atomx.config.csf import enumerate_jj_orbital_space
from atomx.constants import AU_RATE_PER_S
from atomx.numerics.bspline.slater import slater_rk
from atomx.scattering.ci_continuum import CIContinuumChannel, _CommonOrbitalCIChannels


class MultiElectronCIChannels(_CommonOrbitalCIChannels):
    """Complete residual CI cascade, signed Dyson PI and full-h+V Auger.

    The parent space is explicit in Atom; all N-1,...,1 residual spaces are
    complete over its retained orbitals. Cost limits fail before enumeration.
    This defines a finite-space model; it does not certify active-space
    convergence or replace unspecified channels with configuration averages.
    """

    def __init__(self, atom, *, continuum_model, magnetic_state_cap=512):
        if (isinstance(magnetic_state_cap, bool)
                or not isinstance(magnetic_state_cap, (int, np.integer))
                or magnetic_state_cap < 1):
            raise ValueError("positive integer magnetic_state_cap required")
        self.magnetic_state_cap = int(magnetic_state_cap)
        super().__init__(atom, continuum_model=continuum_model)
        self.sector_atoms = {}
        for number, result in self.sectors.items():
            value = copy.copy(self.atom)
            value._result = result
            self.sector_atoms[number] = value
        self._angular_terms = {}

    def _electron_number(self, result):
        numbers = {sum(s.occupation for s in c.subshells) for c in result.csfs}
        if len(numbers) != 1:
            raise ValueError("one fixed parent electron number required")
        number = numbers.pop()
        if int(number) != number or not 2 <= number <= result.Z:
            raise ValueError("multi-electron channels require 2 <= N <= Z")
        number = int(number)
        spinors = sum(2 * abs(k) for _, k in result.orbitals_dict)
        size = sum(c.total_J2 + 1 for c in result.csfs)
        size += sum(comb(spinors, n) for n in range(1, number)) + 1
        if size > self.magnetic_state_cap:
            raise ValueError(f"complete joint CI cascade needs {size} magnetic states; cap={self.magnetic_state_cap}")
        return number

    def _make_sectors(self, diagonalize):
        if self.electron_number == 2:
            return super()._make_sectors(diagonalize)
        sectors = {self.electron_number: self.initial}
        for number in range(self.electron_number - 1, 1, -1):
            csfs = enumerate_jj_orbital_space(
                self.keys, number, magnetic_state_cap=self.magnetic_state_cap
            )
            ci = diagonalize(csfs)
            result = copy.copy(self.initial)
            result.csfs = csfs
            result.state_energies = ci.state_energies
            result.mixing_coefficients = ci.mixing_coefficients
            result.ground_energy = ci.ground_energy
            # copy.copy shares the diagnostics dict with the parent —
            # rebind a merged copy before recording the sector's model.
            result.solver_diagnostics = {
                **result.solver_diagnostics,
                "_radial_optimization": self._sector_orbital_optimization(
                    number, "frozen parent orbitals; complete fixed-N ionic CI"),
            }
            result.__dict__.pop("_radial_state", None)
            keys, density = result.one_body_density_matrix()
            result.occupations = {key: float(density[i, i]) for i, key in enumerate(keys)}
            sectors[number] = result
        sectors[1] = self.ion
        return sectors

    def auger(self, *, energy_overrides=None):
        overrides = {} if energy_overrides is None else energy_overrides
        plan = []
        for number in self.sectors:
            if number == 1:
                continue
            for initial, (ji, pi) in self.quantum[number].items():
                for final, (jf, pf) in self.quantum[number - 1].items():
                    energy = float(self.sectors[number].state_energies[initial]
                                   - self.sectors[number - 1].state_energies[final])
                    if energy <= 0:
                        continue
                    energy = float(overrides.get((number, initial, final), energy))
                    if not np.isfinite(energy) or energy <= 0:
                        raise ValueError("continuum sampling energy must be finite and positive")
                    for je in range(abs(ji - jf), ji + jf + 1, 2):
                        if je % 2 != 1:
                            continue
                        for kappa in (-(je + 1) // 2, (je + 1) // 2):
                            orbital_l = kappa if kappa > 0 else -kappa - 1
                            if pf * (-1) ** orbital_l == pi:
                                plan.append((number, initial, final, ji, jf, kappa, energy))
        for target in dict.fromkeys((n - 1, f) for n, _, f, *_ in plan):
            number, final = target
            self._prepare_waves(self._field_token(number, final),
                                [(k, e) for n, _, f, _, _, k, e in plan
                                 if (n - 1, f) == target])
        channels = []
        for number, initial, final, ji, jf, kappa, energy in plan:
            p, q = self._wave(self._field_token(number - 1, final), kappa, energy)
            continuum_key = (-1, kappa)
            waves = dict(self.bound)
            waves[continuum_key] = p, q

            @lru_cache(maxsize=None)
            def evaluate(key, p=p, q=q, continuum_key=continuum_key, waves=waves):
                if key[0] == "h":
                    _, na, ka, nb = key
                    if na != -1:
                        raise ValueError("continuum one-body term has incorrect role order")
                    hp, hq = self._h_bound[nb, ka]
                    return float(np.sum(self.weights * (p * hp + q * hq)))
                _, rank, a, c, b, d = key
                # Keep the bound pair first, matching the existing N=2 kernel.
                if continuum_key in (a, c):
                    a, c, b, d = b, d, a, c
                pa, qa = waves[a]
                pc, qc = waves[c]
                pb, qb = waves[b]
                pd, qd = waves[d]
                return slater_rk(pa * pc + qa * qc, pb * pd + qb * qd, self.data, rank)

            for me in range(-2 * abs(kappa) + 1, 2 * abs(kappa), 2):
                components = []
                for mf in range(-jf, jf + 1, 2):
                    for mi in range(-ji, ji + 1, 2):
                        if mi != mf + me:
                            continue
                        token = (number, initial, final, kappa, me, mf, mi)
                        if token not in self._angular_terms:
                            self._angular_terms[token] = continuum_hamiltonian_terms(
                                self.vectors[number - 1][final, mf],
                                self.vectors[number][initial, mi], self.spinors, kappa, me,
                            )
                        value = sum(angular * evaluate(key)
                                    for key, angular in self._angular_terms[token])
                        if value != 0:
                            components.append(((mf, mi), float(np.sqrt(2 * np.pi * AU_RATE_PER_S) * value)))
                if components:
                    channels.append(CIContinuumChannel("auger", number, initial, final,
                                                       kappa, me, energy, tuple(components)))
        return tuple(channels)
