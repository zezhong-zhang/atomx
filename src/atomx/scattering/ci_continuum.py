"""Signed two-electron CI -> one-electron ion continuum channels.

The residual ion is diagonalized in the parent's common orbital space.
Continuum waves use the explicit frozen residual Hartree field, with
Pauli projection out of the retained bound orbital space. This is a
projected distorted-wave model, not relaxed-ion or close-coupling PI.
KNOWN GAP: N>2, Breit/QED continuum, interchannel/Fano coupling and
shake/double emission. Nothing is inferred from averaged rates.
"""

import copy
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from atomx.angular.coefficients import ckm
from atomx.angular.csf import csf_parity, csf_twice_j
from atomx.angular.csf_hamiltonian import (
    csf_magnetic_vector,
    csf_signature,
    two_body_angular,
)
from atomx.config.csf import enumerate_jj_csfs
from atomx.config.subshell import L_LETTERS
from atomx.constants import (
    ALPHA,
    AU_RATE_PER_S,
    BOHR_RADIUS_M,
    HARTREE_EV,
)
from atomx.methods.mcdhf.solver import MCDHFResult
from atomx.numerics.bspline.dirac_assembly import apply_dirac_hamiltonian
from atomx.numerics.bspline.evaluate import eval_P, eval_Q
from atomx.numerics.bspline.slater import slater_rk
from atomx.numerics.radial_workspace import make_grid_workspace
from atomx.numerics.shooting.dirac_continuum import solve_dirac_continuum_many
from atomx.numerics.yk import compute_yk
from atomx.properties.transitions.multipole import radial_eL
from atomx.scattering.models import ContinuumModel
from atomx.scattering.radiative_recombination import _final_e1_kappas


@dataclass(frozen=True)
class CIContinuumChannel:
    """One energy-resolved electron bath; signed magnetic jump amplitudes.

    components are ((final 2M, initial 2M), amplitude) pairs. PI amplitudes
    have units sqrt(cm²), Auger amplitudes sqrt(s⁻¹). Each includes the
    full coherent CSF sum before squaring. No additional occupation or
    statistical weights are applied by the consumer.
    """

    process: str
    initial_electrons: int
    initial_level: int
    final_level: int
    electron_kappa: int
    electron_m2: int
    electron_energy_hartree: float
    components: tuple


def _quantum(result):
    mix = np.asarray(result.mixing_coefficients)
    energies = np.asarray(result.state_energies)
    if (
        np.iscomplexobj(mix)
        or mix.shape != (len(result.csfs), len(energies))
        or not np.isfinite(mix).all()
        or not np.isfinite(energies).all()
        or not np.allclose(mix.T @ mix, np.eye(len(energies)), atol=1e-10, rtol=0)
    ):
        raise ValueError("finite real orthonormal CI eigenvectors required")
    quantum = {}
    for level in range(len(result.state_energies)):
        active = np.flatnonzero(abs(result.mixing_coefficients[:, level]) > 1e-12)
        blocks = {
            (csf_twice_j(result.csfs[i]), csf_parity(result.csfs[i])) for i in active
        }
        if len(blocks) != 1:
            raise ValueError("resolved CI states must have pure J and parity")
        quantum[level] = blocks.pop()
    return quantum


def _vectors(result, keys, quantum):
    output = {}
    for level, (j2, _) in quantum.items():
        for m in range(-j2, j2 + 1, 2):
            vector = {}
            for c, csf in zip(
                result.mixing_coefficients[:, level], result.csfs, strict=True
            ):
                if abs(c) < 1e-12:
                    continue
                for mask, amplitude in csf_magnetic_vector(csf_signature(csf), m, keys):
                    vector[mask] = vector.get(mask, 0.0) + float(c) * amplitude
            if abs(sum(v * v for v in vector.values()) - 1) > 1e-10:
                raise ValueError("CI magnetic expansion is not normalized")
            output[level, m] = vector
    return output


class _CommonOrbitalCIChannels:
    """Actual CI PI/Auger tensors in an explicit residual Hartree continuum.

    Require Dirac-Coulomb MCDHF without Breit/QED, so the initial and
    projected ionic CI Hamiltonians use identical energy accounting.
    One-electron ionic PI to the bare ion is also supplied. Bound orbitals
    and ionic radial CI share the parent's orthonormal B-spline basis.
    Auger uses the full h_D + 1/r12 Slater-Condon amplitude: the one-body
    term must not be dropped merely because the named channel is Auger.
    """

    def __init__(self, atom, *, continuum_model):
        result = atom.result
        if (
            not isinstance(result, MCDHFResult)
            or not result.converged
            or result._grid is not None
            or result.per_block_orbitals
        ):
            raise ValueError("converged shared B-spline MCDHF required")
        if atom.breit or atom.qed:
            raise ValueError(
                "consistent Dirac-Coulomb channels require corrections='none'"
            )
        self.electron_number = self._electron_number(result)
        if (
            not isinstance(continuum_model, ContinuumModel)
            or continuum_model.method != "shooting"
            or continuum_model.potential is not None
            or continuum_model.orbital_removal != "none"
        ):
            raise ValueError(
                "explicit shooting(potential=None, orbital_removal='none') required; final-ion Hartree field is constructed here"
            )
        if continuum_model.shooting_n_grid < 1000:
            raise ValueError("at least 1000 shooting nodes required")
        _quantum(result)
        self.model = continuum_model
        self.atom = copy.copy(atom)
        self.initial = copy.copy(result)
        self.initial.csfs = copy.deepcopy(result.csfs)
        self.initial.__dict__.pop("_radial_state", None)
        self.initial.orbitals_dict = {
            key: tuple(np.array(v, copy=True) for v in wave)
            for key, wave in result.orbitals_dict.items()
        }
        self.initial.state_energies = result.state_energies.copy()
        self.initial.mixing_coefficients = result.mixing_coefficients.copy()
        self.initial.bsp_data = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in result.bsp_data.items()
        }
        self.initial._bsp_data = self.initial.bsp_data
        self.atom._result = self.initial
        self.data = self.initial.bsp_data
        self.keys = tuple(sorted(self.initial.orbitals_dict))
        self.spinors = tuple(
            (n, k, m)
            for n, k in self.keys
            for m in range(-2 * abs(k) + 1, 2 * abs(k), 2)
        )
        self.radius = self.data["gr"].ravel()
        self.weights = self.data["grw"].ravel()
        self.bound = {
            key: (
                eval_P(wave[0], self.data).ravel(),
                eval_Q(wave[1], self.data).ravel(),
            )
            for key, wave in self.initial.orbitals_dict.items()
        }
        nuclear = self.initial._nuclear_potential
        potential = (
            -self.initial.Z / self.radius
            if nuclear is None
            else np.array([nuclear(r) for r in self.radius])
        )
        self._h_bound = {
            key: tuple(
                value.ravel()
                for value in apply_dirac_hamiltonian(
                    *coefficients,
                    key[1],
                    self.data,
                    potential=potential.reshape(self.data["gr"].shape),
                )
            )
            for key, coefficients in self.initial.orbitals_dict.items()
        }
        gram = np.array(
            [
                [
                    sum(
                        self.weights
                        * (
                            self.bound[a][0] * self.bound[b][0]
                            + self.bound[a][1] * self.bound[b][1]
                        )
                    )
                    if a[1] == b[1]
                    else 0
                    for b in self.keys
                ]
                for a in self.keys
            ]
        )
        if (
            not np.isfinite(gram).all()
            or abs(gram - np.eye(len(self.keys))).max() > 1e-7
        ):
            raise ValueError("common bound orbitals must be orthonormal")
        self._projectors = {}
        for kappa in {key[1] for key in self.keys}:
            indices = [i for i, key in enumerate(self.keys) if key[1] == kappa]
            self._projectors[kappa] = (
                tuple(self.bound[self.keys[i]] for i in indices),
                gram[np.ix_(indices, indices)],
            )
        self.workspace = make_grid_workspace(self.radius, self.weights)
        ionic_csfs = []
        for n, k in self.keys:
            l = k if k > 0 else -k - 1
            ionic_csfs.extend(
                c
                # Canonical spectroscopic alphabet — the same one
                # ``enumerate_jj_csfs`` parses through
                # :data:`atomx.config.parsing.L_MAP`, so the generated
                # string round-trips for every l.
                for c in enumerate_jj_csfs(f"{n}{L_LETTERS[l]}1")
                if c.subshells[0].kappa == k
            )

        from atomx.physics.csf.radial_snapshot import CIRadialSnapshot

        radial = CIRadialSnapshot(
            self.initial.orbitals_dict,
            self.data,
            self.initial.Z,
            self.data["ns"],
            nuclear_potential=self.initial._nuclear_potential,
        )

        def diagonalize(csfs):
            return radial.solve(
                csfs,
                self.keys,
            )

        control = diagonalize(self.initial.csfs)
        if not np.allclose(
            control.state_energies, self.initial.state_energies, atol=1e-8, rtol=0
        ):
            raise ValueError(
                "parent energies do not match the shared Dirac-Coulomb Hamiltonian"
            )
        hamiltonian = (
            control.mixing_coefficients * control.state_energies
        ) @ control.mixing_coefficients.T
        if not np.allclose(
            hamiltonian @ self.initial.mixing_coefficients,
            self.initial.mixing_coefficients * self.initial.state_energies,
            atol=1e-8,
            rtol=0,
        ):
            raise ValueError(
                "parent CI vectors do not diagonalize the shared Hamiltonian"
            )
        ionic = diagonalize(ionic_csfs)
        if np.any(ionic.state_energies >= 0):
            raise ValueError(
                "the retained ionic CI space contains unbound ionic levels"
            )
        self.ion = copy.copy(self.initial)
        self.ion.csfs = ionic_csfs
        self.ion.state_energies = ionic.state_energies
        self.ion.mixing_coefficients = ionic.mixing_coefficients
        self.ion.ground_energy = ionic.ground_energy
        self.ion.occupations = {}
        # copy.copy shares the diagnostics dict with the parent — rebind a
        # merged copy before recording the sector's own radial model.
        self.ion.solver_diagnostics = {
            **self.ion.solver_diagnostics,
            "_radial_optimization": self._sector_orbital_optimization(
                1, "frozen parent orbitals; one-electron ionic CI diagonalization"
            ),
        }
        self.ion_atom = copy.copy(self.atom)
        self.ion_atom._result = self.ion
        self.sectors = self._make_sectors(diagonalize)
        self.ci_radial_cache_statistics = radial.diagnostics()
        self.quantum = {n: _quantum(sector) for n, sector in self.sectors.items()}
        self.vectors = {
            n: _vectors(sector, self.keys, self.quantum[n])
            for n, sector in self.sectors.items()
        }
        self.projection_diagnostics = []
        self._waves = {}
        self._fields = {}

    def _electron_number(self, result):
        if any(sum(s.occupation for s in c.subshells) != 2 for c in result.csfs):
            raise ValueError("this continuum adapter requires two-electron CI states")
        return 2

    def _make_sectors(self, diagonalize):
        return {2: self.initial, 1: self.ion}

    def _sector_orbital_optimization(self, number, legacy):
        objective = self.initial.solver_diagnostics.get("_charge_state_average")
        if objective is None:
            return legacy
        targeted = {n for n, _, _ in objective.normalized_sectors(self.electron_number)}
        return (
            "shared orbitals jointly optimized for charge-sector CI Hamiltonian average; fixed during channel preparation"
            if number in targeted
            else "shared charge-average orbitals; sector outside optimization objective; fixed during channel preparation"
        )

    @staticmethod
    def _field_token(number, final):
        return final if number <= 1 else (number, final)

    def _field(self, final):
        if final in self._fields:
            return self._fields[final]
        number, level = final if isinstance(final, tuple) else (int(final != -1), final)
        density = np.zeros_like(self.data["gr"])
        if number > 1:
            keys, rho = self.sectors[number].one_body_density_matrix(level)
            if not np.isclose(np.trace(rho), number, atol=1e-9, rtol=0):
                raise ValueError(
                    "final-ion one-body density has incorrect particle number"
                )
            for ia, a in enumerate(keys):
                for ib, b in enumerate(keys):
                    if a[1] == b[1] and rho[ia, ib] != 0:
                        pa, qa = self.bound[a]
                        pb, qb = self.bound[b]
                        density += (rho[ia, ib] * (pa * pb + qa * qb)).reshape(
                            density.shape
                        )
        elif number == 1:
            # A single electron's radial wave is independent of M; retaining
            # its same-kappa CI interference is essential to its direct field.
            j = self.quantum[1][final][0]
            vector = self.vectors[1][final, j]
            p, q = np.zeros_like(self.radius), np.zeros_like(self.radius)
            for mask, coefficient in vector.items():
                bit = mask.bit_length() - 1
                n, k, _ = self.spinors[bit]
                bp, bq = self.bound[n, k]
                p += coefficient * bp
                q += coefficient * bq
            density = (p * p + q * q).reshape(self.data["gr"].shape)
        hartree = compute_yk(
            density,
            self.data["gr"],
            self.data["grw"],
            self.data["nv"],
            self.data["gr"].shape[1],
            0,
        ).ravel()
        number = float(number)
        nuclear = self.initial._nuclear_potential

        def field(r):
            r = np.asarray(r)
            direct = np.interp(r, self.radius, hartree, left=hartree[0], right=0.0)
            direct = np.where(r > self.radius[-1], number / r, direct)
            attraction = (
                -self.initial.Z / r
                if nuclear is None
                else np.vectorize(nuclear, otypes=[float])(r)
            )
            return attraction + direct

        self._fields[final] = field
        return field

    def _prepare_waves(self, final, requests):
        """One existing batch/dense-output kernel per immutable ionic field."""
        missing = list(
            dict.fromkeys(
                (int(k), float(e))
                for k, e in requests
                if (final, int(k), float(e)) not in self._waves
            )
        )
        if not missing:
            return
        r_max = self.model.shooting_r_max or max(50.0, float(self.radius[-1]) * 1.01)
        if r_max < self.radius[-1]:
            raise ValueError("shooting box must contain the common orbital quadrature")
        _, waves = solve_dirac_continuum_many(
            missing,
            self._field(final),
            n_grid=self.model.shooting_n_grid,
            r_max=r_max,
            Z_residual=self.initial.Z
            - (final[0] if isinstance(final, tuple) else int(final != -1)),
            interp_grid=self.radius,
            workers=1,
        )
        for kappa, energy in missing:
            self._store_wave(final, kappa, energy, *waves[kappa, energy])

    def _wave(self, final, kappa, energy):
        self._prepare_waves(final, [(kappa, energy)])
        return self._waves[final, kappa, float(energy)]

    def _store_wave(self, final, kappa, energy, p, q):
        key = (final, kappa, float(energy))
        basis, gram = self._projectors.get(kappa, ((), None))
        before, after = 0.0, 0.0
        if basis:
            overlap = np.array([sum(self.weights * (a * p + b * q)) for a, b in basis])
            before = float(abs(overlap).max())
            for coefficient, (a, b) in zip(
                np.linalg.solve(gram, overlap), basis, strict=True
            ):
                p -= coefficient * a
                q -= coefficient * b
            after = float(
                max(abs(sum(self.weights * (a * p + b * q))) for a, b in basis)
            )
            if after > 1e-10:
                raise ValueError("continuum Pauli projection failed")
        self.projection_diagnostics.append(
            dict(
                final_level=final,
                kappa=kappa,
                energy_hartree=float(energy),
                before=before,
                after=after,
            )
        )
        self._waves[key] = (p, q)
        return p, q

    def photoionization(
        self, photon_energy_eV, *, polarization_q=0, energy_overrides=None
    ):
        if (
            not np.isfinite(photon_energy_eV)
            or photon_energy_eV <= 0
            or polarization_q not in (-1, 0, 1)
        ):
            raise ValueError(
                "positive photon energy and valid spherical polarization required"
            )
        omega = photon_energy_eV / HARTREE_EV
        overrides = {} if energy_overrides is None else energy_overrides
        scale = np.sqrt(4 * np.pi**2 * ALPHA * omega) * (BOHR_RADIUS_M * 100)
        kappas = sorted(
            {k for _, bound_k in self.keys for k in _final_e1_kappas(bound_k)}
        )
        for number, result in self.sectors.items():
            targets = self.quantum[number - 1] if number > 1 else {-1: (0, 1)}
            for final in targets:
                ef = (
                    self.sectors[number - 1].state_energies[final]
                    if number > 1
                    else 0.0
                )
                self._prepare_waves(
                    self._field_token(number - 1, final),
                    [
                        (k, float(overrides.get((number, i, final), omega + ei - ef)))
                        for i, ei in enumerate(result.state_energies)
                        if omega + ei - ef > 0
                        for k in kappas
                    ],
                )
        channels = []
        for number, result in self.sectors.items():
            for initial, (ji, _) in self.quantum[number].items():
                targets = self.quantum[number - 1] if number > 1 else {-1: (0, 1)}
                for final, (jf, _) in targets.items():
                    ef = (
                        self.sectors[number - 1].state_energies[final]
                        if number > 1
                        else 0.0
                    )
                    physical_energy = float(omega + result.state_energies[initial] - ef)
                    if physical_energy <= 0:
                        continue
                    energy = float(
                        overrides.get((number, initial, final), physical_energy)
                    )
                    if not np.isfinite(energy) or energy <= 0:
                        raise ValueError(
                            "continuum sampling energy must be finite and positive"
                        )
                    for kappa in kappas:
                        p, q = self._wave(
                            self._field_token(number - 1, final), kappa, energy
                        )
                        radial = {
                            key: radial_eL(
                                *self.bound[key],
                                p,
                                q,
                                self.workspace,
                                1,
                                kappa_i=key[1],
                                kappa_f=kappa,
                                omega=omega,
                            )
                            for key in self.keys
                        }
                        for me in range(-2 * abs(kappa) + 1, 2 * abs(kappa), 2):
                            components = []
                            for mf in range(-jf, jf + 1, 2):
                                final_vector = (
                                    self.vectors[number - 1][final, mf]
                                    if number > 1
                                    else {0: 1.0}
                                )
                                for mi in range(-ji, ji + 1, 2):
                                    value = 0.0
                                    for mask, ci in self.vectors[number][
                                        initial, mi
                                    ].items():
                                        occupied = [
                                            bit
                                            for bit in range(len(self.spinors))
                                            if mask & (1 << bit)
                                        ]
                                        for position, bit in enumerate(occupied):
                                            spectator = mask ^ (1 << bit)
                                            cf = final_vector.get(spectator, 0.0)
                                            n, k, m = self.spinors[bit]
                                            value += (
                                                cf
                                                * ci
                                                * (-1) ** position
                                                * ckm(
                                                    kappa,
                                                    me,
                                                    k,
                                                    m,
                                                    1,
                                                    2 * polarization_q,
                                                )
                                                * radial[n, k]
                                            )
                                    if value != 0:
                                        components.append(
                                            ((mf, mi), float(scale * value))
                                        )
                            if components:
                                channels.append(
                                    CIContinuumChannel(
                                        "pi",
                                        number,
                                        initial,
                                        final,
                                        kappa,
                                        me,
                                        energy,
                                        tuple(components),
                                    )
                                )
        return tuple(channels)


class TwoElectronCIChannels(_CommonOrbitalCIChannels):
    """Two-/one-electron specialization of the shared CI continuum kernels."""

    def auger(self):
        plan = []
        for initial, (ji, pi) in self.quantum[2].items():
            for final, (jf, pf) in self.quantum[1].items():
                energy = float(
                    self.initial.state_energies[initial]
                    - self.ion.state_energies[final]
                )
                if energy <= 0:
                    continue
                for je in range(abs(ji - jf), ji + jf + 1, 2):
                    if je % 2 != 1:
                        continue
                    for kappa in (-(je + 1) // 2, (je + 1) // 2):
                        l = kappa if kappa > 0 else -kappa - 1
                        if pf * (-1) ** l == pi:
                            plan.append((initial, final, ji, jf, kappa, energy))
        for final in dict.fromkeys(row[1] for row in plan):
            self._prepare_waves(
                final, [(row[4], row[5]) for row in plan if row[1] == final]
            )
        channels = []
        for initial, final, ji, jf, kappa, energy in plan:
            je = 2 * abs(kappa) - 1
            p, q = self._wave(final, kappa, energy)
            one_body = {
                key: float(np.sum(self.weights * (p * hp + q * hq)))
                if key[1] == kappa
                else 0.0
                for key, (hp, hq) in self._h_bound.items()
            }

            @lru_cache(maxsize=None)
            def radial(a, b, c, k, p=p, q=q):
                pa, qa = self.bound[a]
                pb, qb = self.bound[b]
                pc, qc = self.bound[c]
                return slater_rk(pa * pb + qa * qb, p * pc + q * qc, self.data, k)

            def element(a, ma, b, mb, c, mc, me, kappa=kappa, radial=radial):
                return sum(
                    coefficient * radial(a, b, c, rank)
                    for rank, coefficient in two_body_angular(
                        a[1], ma, kappa, me, b[1], mb, c[1], mc
                    )
                )

            for me in range(-je, je + 1, 2):
                components = []
                for mf in range(-jf, jf + 1, 2):
                    for mi in range(-ji, ji + 1, 2):
                        if mi != mf + me:
                            continue
                        value = 0.0
                        for mask, ci in self.vectors[2][initial, mi].items():
                            occupied = [
                                bit
                                for bit in range(len(self.spinors))
                                if mask & (1 << bit)
                            ]
                            (nb, kb, mb), (nc, kc, mc) = (
                                self.spinors[bit] for bit in occupied
                            )
                            for ionmask, cf in self.vectors[1][final, mf].items():
                                na, ka, ma = self.spinors[ionmask.bit_length() - 1]
                                single = (
                                    one_body[nc, kc]
                                    if (na, ka, ma) == (nb, kb, mb) and me == mc
                                    else 0.0
                                ) - (
                                    one_body[nb, kb]
                                    if (na, ka, ma) == (nc, kc, mc) and me == mb
                                    else 0.0
                                )
                                value += (
                                    ci
                                    * cf
                                    * (
                                        element(
                                            (na, ka),
                                            ma,
                                            (nb, kb),
                                            mb,
                                            (nc, kc),
                                            mc,
                                            me,
                                        )
                                        - element(
                                            (na, ka),
                                            ma,
                                            (nc, kc),
                                            mc,
                                            (nb, kb),
                                            mb,
                                            me,
                                        )
                                        + single
                                    )
                                )
                        if value != 0:
                            components.append(
                                (
                                    (mf, mi),
                                    float(np.sqrt(2 * np.pi * AU_RATE_PER_S) * value),
                                )
                            )
                if components:
                    channels.append(
                        CIContinuumChannel(
                            "auger",
                            2,
                            initial,
                            final,
                            kappa,
                            me,
                            energy,
                            tuple(components),
                        )
                    )
        return tuple(channels)
