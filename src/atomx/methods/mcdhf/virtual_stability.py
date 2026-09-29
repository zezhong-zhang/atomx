"""Finite orbital trials for nearly empty correlated radial directions.

A vanishing occupation can make the orbital gradient uninformative. Trial
orbitals are generated with confinement, projected into the *existing*
nuclear no-pair space, and scored with the unmodified CI Hamiltonian.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac


def populate_weak_virtuals(functional, orbitals, *, occupation_threshold=1e-7,
                          strengths=(.01, .1, 1., 10.), acceptance_tolerance=1e-10):
    """Try compact orbitals in weak natural-density directions.

    This is a finite stability search, not a global-minimum certificate.
    Every accepted trial lowers the original CI energy. Natural rotations
    are used only if a fresh CI evaluation verifies energy equivalence.
    The current implementation requires the nuclear no-pair projector.
    """
    from atomx.methods.mcdhf.solver import MCDHFResult
    current = dict(orbitals)
    energy, ci = functional.evaluate(current, gradient=False)
    initial_energy = energy
    model = MCDHFResult(True, 0, ci.state_energies, ci.mixing_coefficients,
                        ci.ground_energy, {}, functional.csfs,
                        orbitals_dict=current, bsp_data=functional.data, Z=functional.Z)
    keys, density = model.one_body_density_matrix()
    trials, rotations = [], []
    groups = {}
    for i, key in enumerate(keys):
        groups.setdefault(key[1], []).append(i)
    for kappa, indices in groups.items():
        shell_keys = [keys[i] for i in indices]
        populations, rotation = eigh(density[np.ix_(indices, indices)])
        order = np.argsort(populations)[::-1]
        populations, rotation = populations[order], rotation[:, order]
        weak = np.flatnonzero(populations < occupation_threshold)
        if not len(weak):
            continue
        coefficients = np.column_stack([np.concatenate(current[k]) for k in shell_keys])@rotation
        rotated = dict(current)
        for i, key in enumerate(shell_keys):
            rotated[key] = (coefficients[:functional.ns, i], coefficients[functional.ns:, i])
        rotated_energy, rotated_ci = functional.evaluate(rotated, gradient=False)
        # A truncated CSF set need not be closed under this orbital rotation.
        if abs(rotated_energy-energy) > 1e-9:
            rotations.append({'kappa': kappa, 'accepted': False,
                              'energy_change_ha': rotated_energy-energy})
            continue
        rotations.append({'kappa': kappa, 'accepted': True,
                          'energy_change_ha': rotated_energy-energy,
                          'natural_occupations': populations.tolist()})
        current, energy, ci = rotated, rotated_energy, rotated_ci
        metric = functional.metric[kappa]
        nuclear = solve_dirac(functional.one_body[kappa], metric, functional.ns, kappa,
                              nsp=functional.data.get('nsp'), nsq=functional.data.get('nsq'))
        l = kappa if kappa > 0 else -kappa-1
        kb = functional.data.get('ksp', functional.data['ks']) != functional.data.get('ksq', functional.data['ks'])
        first = min(k[0] for k in shell_keys)-l-1+int(kappa > 0 and not kb)
        projector = nuclear.coefficients[:, first:]
        potential = functional.nuclear_potential
        if potential is None:
            def potential(r):
                return -functional.Z/r
        for column in weak:
            for strength in strengths:
                h, _ = build_dirac_matrices(functional.ns, kappa, functional.Z, functional.data,
                    lambda r, strength=strength, potential=potential: potential(r)+strength*r*r)
                h = np.tril(h)+np.tril(h, -1).T
                _, eigenvectors = eigh(projector.T@h@projector)
                others = [np.concatenate(current[k]) for i, k in enumerate(shell_keys) if i != column]
                retained = np.column_stack(others) if others else np.empty((len(h), 0))
                trial = None
                for vector in eigenvectors.T:
                    candidate = projector@vector
                    for _ in range(2):
                        candidate -= retained@(retained.T@metric@candidate)
                    norm = float(candidate@metric@candidate)
                    if norm > 1e-8:
                        trial = candidate/np.sqrt(norm)
                        break
                if trial is None:
                    continue
                proposed = dict(current)
                key = shell_keys[column]
                proposed[key] = (trial[:functional.ns], trial[functional.ns:])
                trial_energy, trial_ci = functional.evaluate(proposed, gradient=False)
                delta = trial_energy-energy
                accepted = delta < -acceptance_tolerance
                trials.append({'orbital': key, 'strength': strength,
                               'energy_change_ha': delta, 'accepted': accepted})
                if accepted:
                    current, energy, ci = proposed, trial_energy, trial_ci
    return current, ci, {'initial_energy_ha': initial_energy, 'final_energy_ha': energy,
                         'occupation_threshold': occupation_threshold,
                         'rotations': rotations, 'trials': trials,
                         'scope': 'finite trials in nuclear positive-energy space; original Hamiltonian'}
