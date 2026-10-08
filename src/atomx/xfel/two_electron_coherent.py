"""Charge-resolved two-electron CI XFEL drive, PI, Auger and E1 decay.

KNOWN GAP: frozen shared ion orbitals and projected Hartree continuum;
no interchannel/Fano or continuum-memory effects, shake/double emission,
N>2, Breit/QED, higher multipoles or strong-field imaging response.
"""

from dataclasses import asdict

import numpy as np

from atomx.constants import AU_RATE_PER_S, HARTREE_EV
from atomx.scattering.ci_continuum import TwoElectronCIChannels
from atomx.xfel.coherent import CoherentGaussianPulse, CoherentSystem
from atomx.xfel.mcdhf_coherent import MCDHFCoherentModel


class _JointCICoherentModel:
    """Explicit two-, one- and zero-electron CI magnetic density matrix.

    Provide photon ladders for all parent AND ion CI levels. For off-resonant
    ionic pumping use equal ionic ladders explicitly. Continuous-electron
    loss uses central-energy, far-from-threshold secular Markov baths;
    exactly degenerate initial levels share emitted-electron operators.
    Pulse fluence can vary, but carrier and transform-limited width cannot
    reuse a model prepared at different continuum energies.
    ``initial_state`` selects the parent CI state/population with the same
    contract as MCDHFCoherentModel; ionic/coherent preparations can supply an
    explicit joint magnetic density to ``solve(initial_density=...)``.
    """

    def __init__(
        self,
        atom,
        pulse,
        *,
        photon_numbers,
        ionic_photon_numbers,
        continuum_model,
        initial_state=0,
        magnetic_state_cap=128,
    ):
        if not isinstance(pulse, CoherentGaussianPulse):
            raise ValueError("an explicit coherent pulse is required")
        self.channels = self._make_channels(atom, continuum_model, magnetic_state_cap)
        self.bound_models = self._make_bound_models(
            photon_numbers, ionic_photon_numbers, initial_state, magnetic_state_cap
        )
        parent = self.bound_models[self.channels.electron_number]
        self.photon_energy_eV = pulse.photon_energy_eV
        self.bandwidth_fwhm_eV = pulse.bandwidth_fwhm_eV
        energies, ladders, charges = {}, {}, {}
        self.by_level = {}
        self.level_by_state = {}
        self.maps = {}
        origin = self.channels.initial.state_energies.min()
        for number, bound in self.bound_models.items():
            result = self.channels.sectors[number]
            self.maps[number] = {}
            for level, magnetic in bound.by_level.items():
                self.by_level[number, level] = {}
                for m, label in magnetic.items():
                    sid = f"{number}e|{label}"
                    self.maps[number][label] = sid
                    self.by_level[number, level][m] = sid
                    self.level_by_state[sid] = (number, level)
                    energies[sid] = (
                        float(result.state_energies[level] - origin) * HARTREE_EV
                    )
                    ladders[sid] = int(bound.system.ladders[bound.indices[label]])
                    charges[sid] = atom.Z - number
        self.by_level[0, -1] = {0: "bare"}
        self.level_by_state["bare"] = (0, -1)
        energies["bare"], ladders["bare"], charges["bare"] = (
            float(-origin) * HARTREE_EV,
            0,
            atom.Z,
        )
        if len(energies) > magnetic_state_cap:
            raise ValueError("joint magnetic CI basis exceeds the density-matrix cap")
        self.indices = {s: i for i, s in enumerate(energies)}
        n = len(energies)
        dipole, decay = np.zeros((n, n), complex), []
        self.initial_density = np.zeros((n, n), complex)
        for number, bound in self.bound_models.items():
            indices = [
                self.indices[self.maps[number][s]] for s in bound.system.state_ids
            ]
            dipole[np.ix_(indices, indices)] = bound.system.dipole
            if number == self.channels.electron_number:
                self.initial_density[np.ix_(indices, indices)] = bound.initial_density
            for op in bound.system.decay_operators:
                padded = np.zeros((n, n), complex)
                padded[np.ix_(indices, indices)] = op
                decay.append(self._finish_jump(padded))
        # A continuum threshold intersecting the pulse needs memory/Fano,
        # not a constant center-frequency Lindblad cross section.
        width = 9 * pulse.bandwidth_fwhm_eV / np.sqrt(8 * np.log(2))
        for number, initial in self.channels.sectors.items():
            final_energies = self.channels.sectors[number - 1].state_energies if number > 1 else [0]
            for ei in initial.state_energies:
                for ef in final_energies:
                    if abs((ef - ei) * HARTREE_EV - pulse.photon_energy_eV) <= width:
                        raise ValueError(
                            "near-threshold CI PI requires continuum memory"
                        )
        self.pi_channels, self.auger_channels = self._make_loss_channels(pulse)

        def jumps(channels):
            baths = []
            for ch in channels:
                result = self.channels.sectors[ch.initial_electrons]
                source_energy = self._bath_energy(ch, result)
                key = (
                    ch.initial_electrons,
                    ch.final_level,
                    ch.electron_kappa,
                    ch.electron_m2,
                )
                op = next(
                    (
                        v
                        for k, e, v in baths
                        if k == key and abs(e - source_energy) <= 1e-9
                    ),
                    None,
                )
                if op is None:
                    op = self._new_jump(n)
                    baths.append((key, source_energy, op))
                for (mf, mi), amplitude in ch.components:
                    lower = self.by_level[ch.initial_electrons - 1, ch.final_level][mf]
                    upper = self.by_level[ch.initial_electrons, ch.initial_level][mi]
                    op[self.indices[lower], self.indices[upper]] += amplitude
            return tuple(self._finish_jump(op) for _, _, op in baths)

        photo = jumps(self.pi_channels)
        radiative_count = len(decay)
        decay.extend(jumps(self.auger_channels))
        reference = {}
        if hasattr(self, "reference_pi_bath_channels"):
            reference = {
                "reference_pi_operators": jumps(self.reference_pi_bath_channels),
                "reference_decay_operators": tuple(decay[:radiative_count])
                    + jumps(self.reference_auger_bath_channels),
            }
        self.system = CoherentSystem(
            energies,
            ladders,
            dipole,
            charges=charges,
            decay_operators=decay,
            pi_operators=photo,
            **reference,
            provenance={
                "atomic_model": f"Dirac-Coulomb {self.channels.electron_number}-electron CI; frozen common-basis ionic CI",
                "parent_radial_optimization": self.channels.initial.solver_diagnostics["_radial_optimization"],
                "ionic_radial_optimization": self.channels.ion.solver_diagnostics["_radial_optimization"],
                "default_initial_parent_level_populations": parent.system.provenance[
                    "default_initial_level_populations"
                ],
                "basis_sha256": ":".join(
                    b.system.provenance["basis_sha256"] for b in self.bound_models.values()
                ),
                "sector_radial_optimization": {
                    n: result.solver_diagnostics["_radial_optimization"] for n, result in self.channels.sectors.items()
                },
                "radiative_arithmetic_generator_difference_bound_per_s": (
                    sum(b.system.provenance["radiative_arithmetic_generator_difference_bound_per_s"]
                        for b in self.bound_models.values())
                    if all(b.system.provenance["radiative_arithmetic_generator_difference_bound_per_s"] is not None
                           for b in self.bound_models.values()) else None
                ),
                "radiative_arithmetic_trace_norm_error_budget": 1e-3,
                "radiative_error_reference": "same clusters/Hamiltonian, arithmetic dissipative coefficients; no microscopic bath/Lamb-shift bound",
                "continuum_model": asdict(continuum_model),
                "continuum_field": "final-ion frozen direct Hartree; no nonlocal exchange",
                "continuum_projection": "orthogonal complement of retained common bound orbitals",
                "continuum_bandwidth_fwhm_eV": pulse.bandwidth_fwhm_eV,
                "photon_energy_eV": pulse.photon_energy_eV,
                "pi_source": "signed magnetic Dyson annihilation amplitudes; occupations once",
                "auger_source": "signed full Dirac-Coulomb CI amplitudes (one- and two-body); no J averaging",
                "known_gap": "N>2, relaxed ionic orbitals, interchannel/Fano/memory, shake/double, Breit/QED, coherent imaging",
            },
        )
        self._decay_rates = self._column_rates(self.system.decay_operators, n)
        self._pi_sigma = self._column_rates(self.system.pi_operators, n)

    @staticmethod
    def _new_jump(n):
        return np.zeros((n, n), complex)

    @staticmethod
    def _finish_jump(op):
        return op

    @staticmethod
    def _column_rates(operators, n):
        # Preserve the historical dense summation order exactly.
        return np.diag(sum((c.conj().T @ c for c in operators),
                           np.zeros((n, n), complex))).real

    def _make_channels(self, atom, continuum_model, magnetic_state_cap):
        return TwoElectronCIChannels(atom, continuum_model=continuum_model)

    def _make_loss_channels(self, pulse):
        return self.channels.photoionization(pulse.photon_energy_eV), self.channels.auger()

    @staticmethod
    def _bath_energy(channel, result):
        return result.state_energies[channel.initial_level] * HARTREE_EV

    def _make_bound_models(self, photon_numbers, ionic_photon_numbers, initial_state,
                           magnetic_state_cap):
        return {
            2: MCDHFCoherentModel(self.channels.atom, photon_numbers=photon_numbers,
                                  bound_only=True, initial_state=initial_state,
                                  magnetic_state_cap=magnetic_state_cap),
            1: MCDHFCoherentModel(self.channels.ion_atom, photon_numbers=ionic_photon_numbers,
                                  bound_only=True, magnetic_state_cap=magnetic_state_cap),
        }
    def solve(self, pulse, *, initial_density=None, **kwargs):
        if (
            not isinstance(pulse, CoherentGaussianPulse)
            or not np.isclose(
                pulse.photon_energy_eV, self.photon_energy_eV, atol=1e-10, rtol=0
            )
            or not np.isclose(
                pulse.bandwidth_fwhm_eV, self.bandwidth_fwhm_eV, atol=1e-12, rtol=0
            )
        ):
            raise ValueError(
                "rebuild CI continuum tensors for a changed carrier or duration"
            )
        rates = self._decay_rates + float(pulse.flux(pulse.center_fs)) * self._pi_sigma
        self._validate_continuum_separation(pulse, rates)
        return self.system.solve(
            pulse,
            self.initial_density if initial_density is None else initial_density,
            **kwargs,
        )

    def _validate_continuum_separation(self, pulse, rates):
        for number, bound in self.bound_models.items():
            levels = list(bound.by_level)
            for a, i in enumerate(levels):
                for j in levels[a + 1 :]:
                    states_i, states_j = (
                        self.by_level[number, i],
                        self.by_level[number, j],
                    )
                    xi, xj = (
                        self.indices[next(iter(states_i.values()))],
                        self.indices[next(iter(states_j.values()))],
                    )
                    difference = abs(
                        self.system.energies[xi] - self.system.energies[xj]
                    )
                    width = (
                        max(
                            rates[self.indices[s]]
                            for s in (*states_i.values(), *states_j.values())
                        )
                        * HARTREE_EV
                        / AU_RATE_PER_S
                    )
                    if 1e-9 < difference < 10 * width:
                        raise ValueError(
                            "unresolved CI electron-loss baths require nonsecular continuum dynamics"
                        )

    def level_populations(self, dynamics):
        if (
            dynamics.state_ids != self.system.state_ids
            or dynamics.provenance.get("basis_sha256")
            != self.system.provenance["basis_sha256"]
        ):
            raise ValueError("density belongs to another joint CI basis")
        return {
            key: sum(
                (dynamics.populations[:, self.indices[s]] for s in magnetic.values()),
                np.zeros(len(dynamics.times_fs)),
            )
            for key, magnetic in self.by_level.items()
        }


class TwoElectronCoherentModel(_JointCICoherentModel):
    """Two-/one-/zero-electron specialization of the common CI XFEL driver."""
