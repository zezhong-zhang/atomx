"""Coherent monoconfiguration DF using B-spline Dirac source equations.

This finds the self-consistent electron-branch stationary solution, matching
the direct shooting equations. It does not minimize over a fixed nuclear
positive-energy basis.
"""
import numpy as np
from atomx.config.subshell import Subshell
from atomx.physics.hartree_fock import RadialHartreeFock
from atomx.numerics.bspline.inhomogeneous import solve_constrained_source
from atomx.numerics.orbital_rotations import rotate_occupied
from atomx.numerics.radial_metric import orbital_distance


def solve_direct_bspline(functional, orbitals, *, max_iter, tolerance,
                         orbital_tolerance=None):
    if functional.data.get('ksp') == functional.data.get('ksq'):
        raise ValueError('direct DF source equations require a kinetically balanced B-spline basis')
    keys, ns = functional.keys, functional.ns
    shells = [Subshell(n, k, 0.) for n, k in keys]
    vectors = {key: np.concatenate(orbitals[key]).copy() for key in keys}
    groups = {k: [i for i, key in enumerate(keys) if key[1] == k]
              for k in {key[1] for key in keys}}
    tol_e = min(tolerance, 1e-7)
    threshold = max(1e-8, np.sqrt(tol_e))
    update_tolerance = threshold if orbital_tolerance is None else min(threshold, orbital_tolerance)
    history = []
    converged = False
    frozen_ci = None

    def pairs():
        return {key: (v[:ns], v[ns:]) for key, v in vectors.items()}

    def fixed_energy():
        return functional.energy_for_weights(pairs(), frozen_weights, integral_cache=rotation_integrals)

    class RotationFunctional(RadialHartreeFock):
        rotation_harmonics = (1, 2, 3, 4)
        def total_energy(self, pa, qa, *args):
            for i, key in enumerate(keys):
                vectors[key] = np.concatenate((pa[i], qa[i]))
            return fixed_energy()
    rotation = RotationFunctional(shells)
    for iteration in range(1, max_iter+1):
        before_sweep = {key: v.copy() for key, v in vectors.items()} if orbital_tolerance is not None else None
        _, _, frozen_ci = functional.evaluate(pairs())
        frozen_weights = functional.last_weights
        for shell, key in zip(shells, keys, strict=True):
            q = functional.last_occupations[key]
            shell.occupation = float(2*abs(key[1])) if abs(q-2*abs(key[1])) < 1e-12 else q
        update = 0.
        for i, key in enumerate(keys):
            q = shells[i].occupation
            if q < 1e-12:
                continue  # no ground-state Euler equation for an empty orbital
            _, gradient, _ = functional.evaluate(pairs(), ci_result=frozen_ci,
                                                  gradient_keys=(key,))
            local = np.zeros_like(functional.one_body[key[1]])
            for j, other in enumerate(keys):
                coefficient = shells[j].occupation-(i == j)
                if abs(coefficient) < 1e-14:
                    continue
                v = vectors[other]
                pp, qq = functional._pair_potential(0, (v[:ns], v[ns:]), (v[:ns], v[ns:]))
                local[:ns,:ns] += coefficient*pp
                local[ns:,ns:] += coefficient*qq
            h = functional.one_body[key[1]]+local
            source = h @ vectors[key]-gradient[key]/(2*q)
            group = groups[key[1]]
            constraints = [vectors[keys[j]] for j in group if j != i]
            new, energy = solve_constrained_source(h, functional.metric[key[1]],
                source, vectors[key], principal=key[0], kappa=key[1],
                constraints=constraints, lower_occupied=sum(keys[j][0]<key[0] for j in group))
            metric = functional.metric[key[1]]
            update = max(update, orbital_distance(new, vectors[key], metric))
            mixed = .5*(new+vectors[key])
            vectors[key] = mixed/np.sqrt(float(mixed @ metric @ mixed))
        pa, qa = [vectors[k][:ns].copy() for k in keys], [vectors[k][ns:].copy() for k in keys]
        for pair in rotation.rotation_pairs(groups):
            rotation_integrals = {}
            rotate_occupied(pa, qa, [pair], lambda: rotation.total_energy(pa, qa),
                            harmonics=rotation.rotation_harmonics)
        for i,key in enumerate(keys): vectors[key] = np.concatenate((pa[i],qa[i]))
        energy, gradient, ci = functional.evaluate(pairs())
        history.append(float(energy))
        multiplier_residual = 0.
        for group in groups.values():
            matrix = np.array([[vectors[keys[j]] @ gradient[keys[i]]/2 for i in group] for j in group])
            multiplier_residual = max(multiplier_residual, float(np.max(abs(matrix-matrix.T))))
        population_change = float(np.max(abs(np.array([functional.last_occupations[k] for k in keys])
                                            -np.array([s.occupation for s in shells]))))
        state_residual = max((abs(functional.last_weights.get(k, 0.)-frozen_weights.get(k, 0.)) /
            max(1., abs(functional.last_weights.get(k, 0.)), abs(frozen_weights.get(k, 0.)))
            for k in functional.last_weights.keys() | frozen_weights.keys()), default=0.)
        applied_update = (max(orbital_distance(vectors[key], before_sweep[key],
                          functional.metric[key[1]]) for key in keys)
                          if before_sweep is not None else None)
        residual = max(update, multiplier_residual, population_change, state_residual)
        orbital_ok = (orbital_tolerance is None or
                      max(update, applied_update, state_residual) <= orbital_tolerance)
        if (len(history)>1 and abs(history[-1]-history[-2]) <= tol_e*max(1.,abs(energy))
                and residual <= threshold and orbital_ok):
            converged = True
            break
    # Canonicalize fully occupied same-kappa blocks, whose rotations are a
    # gauge freedom. This gives physical orbital multipliers for corrections.
    for group in rotation.canonical_groups():
        fock = np.array([[vectors[keys[j]] @ gradient[keys[i]]/(2*shells[i].occupation)
                          for i in group] for j in group])
        _, transform = np.linalg.eigh(.5*(fock+fock.T))
        block = np.array([vectors[keys[i]] for i in group])
        block = transform.T @ block
        for row,i in zip(block, group, strict=True): vectors[keys[i]] = row
    _, _, ci = functional.evaluate(pairs())
    return pairs(), ci, dict(converged=converged, iterations=iteration,
        termination_reason='converged' if converged else 'maximum_iterations',
        model='monoconfiguration Dirac-Coulomb DF',
        functional='coherent CSF Hamiltonian expectation',
        radial_solver='constrained B-spline Dirac source equations',
        electron_projector='self-consistent electron branch, radial index constraints',
        orbital_residual=residual, orbital_tolerance=threshold,
        orbital_update_tolerance=update_tolerance,
        applied_orbital_update_norm=applied_update,
        orbital_update_metric='Dirac overlap matrix',
        state_density_tolerance=update_tolerance,
        occupied_rotation_residual_Ha=multiplier_residual,
        undamped_orbital_update_norm=update, population_change=population_change,
        state_density_residual=state_residual,
        energy_history=history, full_correlated_orbital_gradient=True,
        radial_space_converged=False, radial_space_convergence_reason='requires knot and box refinement')
