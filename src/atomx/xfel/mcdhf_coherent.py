"""Phase-preserving magnetic-state dynamics on an existing MCDHF CI basis.

KNOWN GAP: this adapter is explicitly bound-only. It cannot infer level- and
magnetic-resolved PI/Auger reservoirs from configuration-average rates.
"""

import hashlib

import numpy as np

from atomx.angular.csf import csf_has_explicit_j, csf_parity, csf_twice_j
from atomx.constants import AU_RATE_PER_S, HARTREE_EV
from atomx.methods.dirac_fock import _orbital_overlap
from atomx.methods.mcdhf.solver import MCDHFResult
from atomx.physics.csf.levels import initial_ci_level_weights
from atomx.properties.transitions.rates import einstein_A_eL
from atomx.xfel.coherent import CoherentSystem, magnetic_e1_component
from atomx.xfel.radiative_reservoir import cluster_radiative_baths


class MCDHFCoherentModel:
    """Expand all existing shared-orbital CI eigenstates; linear z drive.

    ``photon_numbers`` maps every CI column index to an integer RWA ladder.
    ``bound_only=True`` is mandatory acknowledgement that PI and Auger are
    absent. No configuration-average loss or UTA amplitude is substituted.
    ``initial_state`` accepts a snapshot column, CILevel symmetry/root selector,
    or CILevelPopulation with total level weights. These choices do not change
    the orbital optimization ensemble. Each selected level is unpolarized;
    ``solve(initial_density=...)`` accepts explicitly phased magnetic states.

    Emission paths at equal frequency share a spherical-photon jump operator.
    The default secular reservoir rejects lines within ten natural widths.
    Opt-in clustered reservoirs retain signed near-degenerate interference:
    "clustered" samples the vacuum spectrum at a common frequency;
    "geometric_clustered" preserves individual Einstein rates and uses
    geometric dissipative cross coefficients. Both gate spectral variation
    at 1e-3. These approximations omit radiation Lamb-shift renormalization;
    the gate bounds spectral coefficients, not the full dynamical error.
    "geometric_bounded" instead gates a conservative finite-time trace-norm
    bound versus the same-cluster arithmetic dissipator. This comparison
    omits Lamb shifts and does not certify the microscopic vacuum model.
    """

    def __init__(
        self,
        atom,
        *,
        photon_numbers,
        bound_only=False,
        initial_state=0,
        radiative_reservoir="secular",
        magnetic_state_cap=128,
    ):
        if bound_only is not True:
            raise ValueError(
                "bound_only=True required; level-resolved PI/Auger are absent"
            )
        if radiative_reservoir not in ("secular", "clustered", "geometric_clustered", "geometric_bounded"):
            raise ValueError("radiative_reservoir must be secular, clustered, geometric_clustered or geometric_bounded")
        result = atom.result
        if not isinstance(result, MCDHFResult) or not result.converged:
            raise ValueError("a converged MCDHF result is required")
        if result._grid is not None or not result.bsp_data:
            raise ValueError(
                "MCDHF coherent adapter currently requires B-spline orbitals"
            )
        if result.per_block_orbitals:
            raise ValueError("one shared orthonormal orbital basis is required")
        csfs, mixing = result.csfs, np.asarray(result.mixing_coefficients)
        count = len(result.state_energies)
        if (
            mixing.shape != (len(csfs), count)
            or not np.isfinite(mixing).all()
            or np.any(np.imag(mixing) != 0)
            or not np.allclose(
                mixing.conj().T @ mixing, np.eye(count), atol=1e-10, rtol=0
            )
        ):
            raise ValueError(
                "CI eigenvectors must be real and orthonormal in one CSF basis"
            )
        if set(photon_numbers) != set(range(count)):
            raise ValueError(
                "assign every CI level a photon ladder"
            )
        if not all(csf_has_explicit_j(c) for c in csfs):
            raise ValueError("explicit J/parity-resolved CSFs are required")
        initial_weights = initial_ci_level_weights(initial_state, csfs, result)
        quantum = [(csf_twice_j(c), csf_parity(c)) for c in csfs]
        self.two_J = {}
        for state in range(count):
            active = np.flatnonzero(abs(mixing[:, state]) > 1e-10)
            blocks = {quantum[i] for i in active}
            if len(blocks) != 1:
                raise ValueError("CI eigenstate mixes different J/parity blocks")
            self.two_J[state] = blocks.pop()[0]
        electrons = {sum(s.occupation for s in c.subshells) for c in csfs}
        if len(electrons) != 1:
            raise ValueError("CI states must have one fixed electron number")
        charge = result.Z - electrons.pop()
        if sum(j + 1 for j in self.two_J.values()) > magnetic_state_cap:
            raise ValueError(
                "magnetic CI basis exceeds the explicit density-matrix size cap"
            )
        keys = sorted(result.orbitals_dict)
        gram = np.array(
            [
                [
                    _orbital_overlap(
                        *result.orbitals_dict[a],
                        *result.orbitals_dict[b],
                        result.bsp_data,
                    )
                    if a[1] == b[1]
                    else 0.0
                    for b in keys
                ]
                for a in keys
            ]
        )
        error = float(abs(gram - np.eye(len(keys))).max())
        if not np.isfinite(gram).all() or error > 1e-7:
            raise ValueError("shared CI orbital basis must be orthonormal")
        energies, ladders, charges = {}, {}, {}
        self.by_level = {}
        self.level_by_state = {}
        for level, j in self.two_J.items():
            self.by_level[level] = {}
            for m in range(-j, j + 1, 2):
                label = f"CI:{level}|m2={m}"
                self.by_level[level][m] = label
                self.level_by_state[label] = level
                energies[label] = (
                    float(result.state_energies[level] - result.state_energies.min())
                    * HARTREE_EV
                )
                ladders[label] = photon_numbers[level]
                charges[label] = charge
        self.indices = {s: i for i, s in enumerate(energies)}
        n = len(energies)
        dipole = np.zeros((n, n), complex)
        self.lines = tuple(atom.transitions(multipoles=["E1"], gauge="length"))
        widths = {i: 0.0 for i in range(count)}
        for line in self.lines:
            if (
                line.reduced_matrix_element_au is None
                or line.initial_state_index not in self.two_J
                or line.final_state_index not in self.two_J
                or line.initial_two_J != self.two_J[line.initial_state_index]
                or line.final_two_J != self.two_J[line.final_state_index]
            ):
                raise ValueError("signed shared-basis CI amplitudes are required")
            widths[line.final_state_index] += line.A_rate_per_s
        hbar_eV_s = HARTREE_EV / AU_RATE_PER_S
        baths = []
        for line in self.lines:
            lower, upper = line.initial_state_index, line.final_state_index
            rme = line.reduced_matrix_element_au
            if (
                rme is None
                or line.resolution != "ci"
                or line.provenance.get("orbital_basis") != "shared orthonormal"
                or not np.isclose(abs(rme) ** 2, line.line_strength, rtol=1e-12, atol=0)
            ):
                raise ValueError("signed shared-basis CI amplitudes are required")
            jl, ju = self.two_J[lower], self.two_J[upper]
            # Hermitian spherical tensor: <l||D||u>=(-1)^(Jl-Ju)<u||D||l>*.
            reverse = (-1.0) ** ((jl - ju) // 2) * np.conj(rme)
            scale = np.sqrt(
                einstein_A_eL(line.energy_diff_hartree, 1.0, 1, ju) * (ju + 1)
            )
            bath = next(
                (entry for entry in baths if abs(entry[0] - line.energy_eV) <= 1e-9),
                None,
            )
            if bath is None:
                group = [np.zeros((n, n), complex) for _ in range(3)]
                bath = [line.energy_eV, group, 0.0]
                baths.append(bath)
            group = bath[1]
            # A transition coherence decays at (Gamma_u+Gamma_l)/2;
            # its natural FWHM is hbar*(Gamma_u+Gamma_l). An unrelated
            # broad line must not forbid two narrow frequency groups.
            bath[2] = max(bath[2], (widths[lower] + widths[upper]) * hbar_eV_s)
            for mu, up in self.by_level[upper].items():
                for ml, low in self.by_level[lower].items():
                    u, l = self.indices[up], self.indices[low]
                    if photon_numbers[upper] - photon_numbers[lower] == 1:
                        dipole[u, l] = magnetic_e1_component(ju, mu, jl, ml, 0, rme)
                    for q in (-1, 0, 1):
                        group[q + 1][l, u] += scale * magnetic_e1_component(
                            jl, ml, ju, mu, q, reverse
                        )
        frequencies = sorted((energy, width) for energy, _, width in baths)
        unresolved = any(
            b[0] - a[0] < 10 * max(a[1], b[1])
            for a, b in zip(frequencies, frequencies[1:], strict=False)
        )
        if unresolved and radiative_reservoir == "secular":
            raise ValueError(
                "unresolved CI emission lines require a nonsecular reservoir"
            )
        cluster_records = []
        expected_widths = dict(widths)
        if radiative_reservoir != "secular":
            merged, factors, cluster_records = cluster_radiative_baths(baths, radiative_reservoir)
            expected_widths = {i: 0.0 for i in range(count)}
            for line in self.lines:
                energy = next(e for e in factors if abs(e - line.energy_eV) <= 1e-9)
                expected_widths[line.final_state_index] += line.A_rate_per_s * factors[energy]
            baths = merged
        decay = tuple(op for _, group, _ in baths for op in group if np.any(op))
        rates = sum((c.conj().T @ c for c in decay), np.zeros((n, n), complex))
        for level, magnetic in self.by_level.items():
            actual = [
                rates[self.indices[s], self.indices[s]].real for s in magnetic.values()
            ]
            if not np.allclose(actual, expected_widths[level], rtol=1e-12, atol=1e-18):
                raise ValueError(
                    "magnetic decay sum disagrees with the CI Einstein rate"
                )
        signature = hashlib.sha256()
        for array in (result.state_energies, mixing):
            signature.update(np.asarray(array).tobytes())
        for c in csfs:
            signature.update(repr(c).encode())
        for key in keys:
            signature.update(repr(key).encode())
            for array in result.orbitals_dict[key]:
                signature.update(np.asarray(array).tobytes())
        # Coefficient vectors alone do not identify physical orbitals:
        # their knots, quadrature and P/Q basis maps are part of the basis.
        for key in ("t", "tp", "gr", "grw", "bsp_p", "bsp_q"):
            if key in result.bsp_data:
                array = np.asarray(result.bsp_data[key])
                signature.update(repr((key, array.shape, str(array.dtype))).encode())
                signature.update(array.tobytes())
        self.system = CoherentSystem(
            energies,
            ladders,
            dipole,
            charges=charges,
            decay_operators=decay,
            provenance={
                "atomic_model": "shared-orbital MCDHF CI; signed finite-wavelength length E1",
                "basis_sha256": signature.hexdigest(),
                "radial_optimization": result.solver_diagnostics["_radial_optimization"],
                "max_orbital_orthogonality_error": error,
                "polarization": "linear z, q=0",
                "bound_only": True,
                "default_initial_level_populations": tuple(
                    (int(i), float(initial_weights[i]))
                    for i in np.flatnonzero(initial_weights)
                ),
                "radiative_level_rates_per_s": widths,
                "prepared_radiative_level_rates_per_s": expected_widths,
                "radiative_reservoir": radiative_reservoir,
                "radiative_cluster_spectral_tolerance": None if radiative_reservoir == "geometric_bounded" else 1e-3,
                "radiative_hamiltonian_renormalization": "omitted",
                "radiative_arithmetic_generator_difference_bound_per_s": sum(
                    record.get("arithmetic_generator_difference_bound_per_s", 0.0)
                    for record in cluster_records
                ) if radiative_reservoir == "geometric_bounded" else None,
                "radiative_arithmetic_trace_norm_error_budget": 1e-3,
                "radiative_error_reference": "same clusters/Hamiltonian, arithmetic dissipative coefficients; no microscopic bath/Lamb-shift bound",

                "radiative_clusters": tuple(cluster_records),
                "emission_frequency_equality_tolerance_eV": 1e-9,
                "emission_group_fwhm_eV": tuple(frequencies),
                "secular_separation_minimum_widths": 10,
                "known_gap": "no level-resolved PI/Auger, Fano continuum, SASE or coherent imaging",
            },
        )
        self.initial_density = np.zeros((n, n), complex)
        for level in np.flatnonzero(initial_weights):
            for label in self.by_level[level].values():
                self.initial_density[self.indices[label], self.indices[label]] = (
                    initial_weights[level] / (self.two_J[level] + 1)
                )

    def solve(self, pulse, *, initial_density=None, **kwargs):
        return self.system.solve(
            pulse,
            self.initial_density if initial_density is None else initial_density,
            **kwargs,
        )

    def level_populations(self, dynamics):
        if (
            dynamics.state_ids != self.system.state_ids
            or dynamics.provenance.get("basis_sha256")
            != self.system.provenance["basis_sha256"]
        ):
            raise ValueError("density belongs to a different CI magnetic basis")
        return {
            level: sum(
                (dynamics.populations[:, self.indices[s]] for s in magnetic.values()),
                np.zeros(len(dynamics.times_fs)),
            )
            for level, magnetic in self.by_level.items()
        }
