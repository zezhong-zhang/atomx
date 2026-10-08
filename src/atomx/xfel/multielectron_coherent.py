"""General many-electron resonance drive and complete finite-space CI losses.

KNOWN GAP: frozen parent-EOL ionic orbitals and projected Hartree DW;
explicit secular/clustered Markov baths, no coupled/Fano continuum, shake/double emission,
nonlocal exchange or joint-charge orbital relaxation. Applicability gates remain.
"""

from dataclasses import replace

import numpy as np
from scipy.sparse import csc_matrix, lil_matrix

from atomx.constants import AU_RATE_PER_S, HARTREE_EV
from atomx.scattering.multielectron_ci import MultiElectronCIChannels
from atomx.xfel.mcdhf_coherent import MCDHFCoherentModel
from atomx.xfel.reservoir_errors import lindblad_difference_bound
from atomx.xfel.two_electron_coherent import _JointCICoherentModel


class MultiElectronCoherentModel(_JointCICoherentModel):
    """Joint N,...,1,0 magnetic CI density with E1, Dyson PI and full-H Auger.

    ``photon_numbers`` maps electron number to a full CI-column/RWA-ladder map.
    No ladders are guessed for daughter sectors. Sector column counts can be
    inspected with MultiElectronCIChannels. Parent preparation uses the same
    CILevel/CILevelPopulation or snapshot-column contract as the bound driver.
    The retained orbital Fock space is complete in every residual sector.
    ``atom`` can also be a prepared MultiElectronCIChannels snapshot: repeated
    drives then reuse its daughter CI, angular contractions and DW cache.
    The continuum model must match; never mutate a prepared snapshot.
    ``radiative_reservoir="geometric_bounded"`` and
    ``electron_reservoir="clustered_bounded"`` opt into finite-time absolute
    trace-norm bounds against explicitly specified same-cluster references.
    These do not bound microscopic reservoir physics. Legacy relative gates
    are unchanged. Large (>128 M-state) jump storage and Hamiltonian action
    use sparse matrices without removing phases, coherences or transitions.
    """

    def __init__(self, atom, pulse, *, photon_numbers, continuum_model,
                 initial_state=0, radiative_reservoir="secular",
                 electron_reservoir="secular", magnetic_state_cap=512):
        if electron_reservoir not in ("secular", "clustered", "clustered_bounded"):
            raise ValueError("electron_reservoir must be secular, clustered or clustered_bounded")
        self._prepared_channels = atom if isinstance(atom, MultiElectronCIChannels) else None
        if self._prepared_channels is not None:
            if continuum_model != self._prepared_channels.model:
                raise ValueError("prepared CI channels require the same continuum model")
            atom = self._prepared_channels.atom
        self.electron_reservoir = electron_reservoir
        self.radiative_reservoir = radiative_reservoir
        parent = {sum(s.occupation for s in c.subshells) for c in atom.result.csfs}
        if len(parent) != 1:
            raise ValueError("one fixed parent electron number required")
        number = int(parent.pop())
        if set(photon_numbers) != set(range(1, number + 1)):
            raise ValueError("assign photon ladders to every retained electron sector")
        super().__init__(
            atom, pulse, photon_numbers=photon_numbers[number],
            ionic_photon_numbers={n: photon_numbers[n] for n in range(1, number)},
            continuum_model=continuum_model, initial_state=initial_state,
            magnetic_state_cap=magnetic_state_cap,
        )
        self.system.provenance["known_gap"] = (
            "joint-charge relaxation, nonlocal continuum exchange, interchannel/Fano/memory, "
            "radiative Hamiltonian renormalization, loss-generated final-ion coherences, "
            "shake/double, Breit/QED, coherent imaging"
        )
        objective = self.channels.initial.solver_diagnostics.get("_charge_state_average")
        if objective is not None:
            diagnostics = self.channels.initial.orbital_optimization
            self.system.provenance["charge_orbital_objective"] = {
                key: diagnostics[key] for key in (
                    "charge_sector_weights", "charge_sector_level_weights",
                    "objective_mean_electrons", "charge_orbital_model")
            }
            self.system.provenance["known_gap"] = (
                "independent charge-specific or dynamic orbital relaxation, nonlocal continuum exchange, "
                "interchannel/Fano/memory, radiative Hamiltonian renormalization, "
                "loss-generated final-ion coherences, shake/double, Breit/QED, coherent imaging"
            )
        self.system.provenance["residual_space"] = "complete Fock sectors over retained parent orbitals"
        self.system.provenance["radiative_reservoir"] = radiative_reservoir
        self.system.provenance["sector_radiative_clusters"] = {
            n: bound.system.provenance["radiative_clusters"]
            for n, bound in self.bound_models.items()
        }
        self.system.provenance["radiative_cluster_spectral_tolerance"] = None if radiative_reservoir == "geometric_bounded" else 1e-3
        self.system.provenance["radiative_hamiltonian_renormalization"] = "omitted"
        self.system.provenance["electron_reservoir"] = electron_reservoir
        self.system.provenance["electron_clusters"] = tuple(self.electron_clusters)
        self.system.provenance["electron_cluster_tensor_relative_tolerance"] = 1e-3
        self.system.provenance["electron_final_state_resolution"] = (
            "distinct final ionic CI levels are resolved reservoirs; final-state coherences omitted"
        )

        if electron_reservoir == "clustered_bounded":
            self.system.provenance["electron_pi_generator_difference_bound_cm2"] = sum(
                r["generator_difference_bound"] for r in self.electron_clusters if r["process"] == "pi")
            self.system.provenance["electron_auger_generator_difference_bound_per_s"] = sum(
                r["generator_difference_bound"] for r in self.electron_clusters if r["process"] == "auger")
            self.system.provenance["reservoir_trace_norm_error_budget"] = 1e-3
            self.system.provenance["electron_error_reference"] = (
                "same retained final-level/kappa/m frequency clusters, signed amplitudes at each original electron energy; "
                "no microscopic bath/intercluster/final-ion-coherence bound")
            self.system.provenance["electron_cluster_tensor_relative_tolerance"] = None

    @staticmethod
    def _new_jump(n):
        # Representation only: no matrix elements, phases or baths are removed.
        return lil_matrix((n, n), dtype=complex) if n > 128 else np.zeros((n, n), complex)

    @staticmethod
    def _finish_jump(op):
        return csc_matrix(op) if op.shape[0] > 128 else op

    @staticmethod
    def _column_rates(operators, n):
        if n <= 128:
            return _JointCICoherentModel._column_rates(operators, n)
        return sum(((c.conj().T @ c).diagonal().real for c in operators), np.zeros(n))

    def _make_channels(self, atom, continuum_model, magnetic_state_cap):
        if self._prepared_channels is not None:
            return self._prepared_channels
        return MultiElectronCIChannels(atom, continuum_model=continuum_model,
                                        magnetic_state_cap=magnetic_state_cap)

    def _make_bound_models(self, photon_numbers, ionic_photon_numbers, initial_state,
                           magnetic_state_cap):
        parent = self.channels.electron_number
        return {
            n: MCDHFCoherentModel(
                self.channels.sector_atoms[n],
                photon_numbers=photon_numbers if n == parent else ionic_photon_numbers[n],
                bound_only=True, initial_state=initial_state if n == parent else 0,
                radiative_reservoir=self.radiative_reservoir,
                magnetic_state_cap=magnetic_state_cap,
            )
            for n in self.channels.sectors
        }

    @staticmethod
    def _bath_energy(channel, result):
        return channel.electron_energy_hartree * HARTREE_EV

    def _level_rates(self, pulse, photo, auger):
        rates = {}
        for number, bound in self.bound_models.items():
            for level, magnetic in bound.by_level.items():
                for m in magnetic:
                    rates[number, level, m] = bound.system.provenance[
                        "prepared_radiative_level_rates_per_s"
                    ][level]
        flux = float(pulse.flux(pulse.center_fs))
        for group, multiplier in ((photo, flux), (auger, 1.0)):
            for channel in group:
                for (_, m), amplitude in channel.components:
                    rates[channel.initial_electrons, channel.initial_level, m] += multiplier * abs(amplitude) ** 2
        return {
            (n, level): max(rates[n, level, m] for m in bound.by_level[level])
            for n, bound in self.bound_models.items() for level in bound.by_level
        }

    @staticmethod
    def _components(channels):
        return {
            (ch.initial_electrons, ch.initial_level, ch.final_level,
             ch.electron_kappa, ch.electron_m2, mf, mi): amplitude
            for ch in channels for (mf, mi), amplitude in ch.components
        }

    @staticmethod
    def _cluster_operator_error(keys, original, sampled):
        # Bath membership is exactly the retained (final level,kappa,m) cluster.
        # Do not discard tiny/forbidden-looking amplitudes or relative errors.
        total = 0.0
        baths = {(key[3], key[4]) for key in keys}
        for kappa, m in sorted(baths):
            paths = [key for key in keys if (key[3], key[4]) == (kappa, m)]
            rows = {mf: i for i, mf in enumerate(sorted({key[5] for key in paths}))}
            cols = {source: i for i, source in enumerate(sorted({(key[1], key[6]) for key in paths}))}
            c = np.zeros((len(rows), len(cols)), complex)
            b = np.zeros_like(c)
            for key in paths:
                index = rows[key[5]], cols[key[1], key[6]]
                c[index], b[index] = original.get(key, 0), sampled.get(key, 0)
            total += lindblad_difference_bound(c, b)
        return total

    def _make_loss_channels(self, pulse):
        photo, auger = super()._make_loss_channels(pulse)
        self.electron_clusters = []
        if self.electron_reservoir == "secular":
            return photo, auger
        rates = self._level_rates(pulse, photo, auger)
        outputs = []
        hbar = HARTREE_EV / AU_RATE_PER_S
        for process, channels in (("pi", photo), ("auger", auger)):
            targets = {}
            for channel in channels:
                targets.setdefault((channel.initial_electrons, channel.final_level), {})[
                    channel.initial_level
                ] = channel.electron_energy_hartree
            overrides, records = {}, []
            for (number, final), sources in targets.items():
                final_width = rates.get((number - 1, final), 0.0)
                clusters = []
                for initial, energy in sorted(sources.items(), key=lambda item: item[1]):
                    width = hbar * (rates[number, initial] + final_width)
                    if (clusters and (energy - clusters[-1][-1][1]) * HARTREE_EV
                            < 10 * max(width, clusters[-1][-1][2])):
                        clusters[-1].append((initial, energy, width))
                    else:
                        clusters.append([(initial, energy, width)])
                for cluster in clusters:
                    if len(cluster) < 2:
                        continue
                    center = 0.5 * (cluster[0][1] + cluster[-1][1])
                    for initial, _, _ in cluster:
                        overrides[number, initial, final] = center
                    records.append({
                        "process": process, "initial_electrons": number, "final_level": final,
                        "initial_levels": tuple(i for i, _, _ in cluster),
                        "physical_electron_energies_hartree": tuple(e for _, e, _ in cluster),
                        "sampled_electron_energy_hartree": center,
                    })
            updated = (
                self.channels.photoionization(pulse.photon_energy_eV, energy_overrides=overrides)
                if process == "pi" else self.channels.auger(energy_overrides=overrides)
            ) if overrides else channels
            original, sampled = self._components(channels), self._components(updated)
            for record in records:
                members = set(record["initial_levels"])
                keys = {key for key in original.keys() | sampled.keys()
                        if key[0] == record["initial_electrons"]
                        and key[1] in members and key[2] == record["final_level"]}
                norm = sum(abs(original.get(key, 0)) ** 2 for key in keys)
                difference = sum(abs(original.get(key, 0) - sampled.get(key, 0)) ** 2 for key in keys)
                error = np.sqrt(difference / norm) if norm else (0.0 if difference == 0 else np.inf)
                if self.electron_reservoir == "clustered" and error > 1e-3:
                    raise ValueError(
                        f"electron cluster varies too much in signed continuum tensor: {error:.6g}; "
                        f"process={process}, N={record['initial_electrons']}, "
                        f"final_level={record['final_level']}, initial_levels={record['initial_levels']}, "
                        f"energies_Ha={record['physical_electron_energies_hartree']}, "
                        f"sample_Ha={record['sampled_electron_energy_hartree']}, "
                        f"squared_tensor_norm={norm:.8g}, squared_difference={difference:.8g}"
                    )
                record["signed_tensor_relative_change"] = float(error)
                if self.electron_reservoir == "clustered_bounded":
                    # The bound treats these as separate frequency baths. Refuse
                    # accidental overlap with an independently retained cluster.
                    baths = {(key[3], key[4]) for key in keys}
                    for channel in channels:
                        if (channel.initial_electrons == record["initial_electrons"]
                            and channel.final_level == record["final_level"]
                            and channel.initial_level not in members
                            and (channel.electron_kappa, channel.electron_m2) in baths):
                            energy = overrides.get((channel.initial_electrons, channel.initial_level, channel.final_level),
                                                   channel.electron_energy_hartree)
                            if abs(energy-record["sampled_electron_energy_hartree"])*HARTREE_EV <= 1e-9:
                                raise ValueError("distinct electron clusters coincide; combined bath bound required")
                    record["generator_difference_bound"] = self._cluster_operator_error(keys, original, sampled)
                    record["generator_difference_units"] = "cm^2" if process == "pi" else "s^-1"
            if self.electron_reservoir == "clustered_bounded":
                # Keep original physical-energy amplitudes. Only relabel bath
                # membership to the same common-energy clusters as production.
                reference = tuple(replace(ch, electron_energy_hartree=overrides.get(
                    (ch.initial_electrons, ch.initial_level, ch.final_level),
                    ch.electron_energy_hartree)) for ch in channels)
                setattr(self, f"reference_{process}_bath_channels", reference)
            self.electron_clusters.extend(records)
            outputs.append(updated)
        return tuple(outputs)

    def _validate_continuum_separation(self, pulse, rates):
        # Only paths sharing the same emitted electron bath can interfere.
        # A broad level in an unrelated kappa/final-ion channel is not a veto.
        hbar = HARTREE_EV / AU_RATE_PER_S
        level_rates = {
            (number, level): max(rates[self.indices[label]] for label in magnetic.values())
            for (number, level), magnetic in self.by_level.items()
        }
        for channels in (self.pi_channels, self.auger_channels):
            baths = {}
            for ch in channels:
                key = ch.initial_electrons, ch.final_level, ch.electron_kappa, ch.electron_m2
                baths.setdefault(key, {})[ch.initial_level] = ch.electron_energy_hartree * HARTREE_EV
            for (number, final, _, _), sources in baths.items():
                ordered = sorted(sources.items(), key=lambda item: item[1])
                for (i, ei), (j, ej) in zip(ordered, ordered[1:], strict=False):
                    width = hbar * (max(level_rates[number, i], level_rates[number, j])
                                    + level_rates.get((number - 1, final), 0.0))
                    if 1e-9 < ej - ei < 10 * width:
                        raise ValueError("unresolved CI electron-loss baths require clustered or non-Markov continuum dynamics")
