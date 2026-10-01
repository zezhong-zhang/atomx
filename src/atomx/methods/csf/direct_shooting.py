"""Monoconfiguration Hamiltonian DF via direct coupled radial shooting."""
from copy import deepcopy

import numpy as np

from atomx.config.jj_coupling import generate_jj_average
from atomx.methods.ci import ci_solve
from atomx.numerics.shooting.hartree_fock import ShootingHFSolver
from atomx.physics.csf.matrix import GridCIWorkspace
from atomx.physics.csf.radial_fock import CSFRadialFock


def solve_direct_shooting(solver, *, max_iter, tolerance, orbital_tolerance=None):
    shells, ensemble = generate_jj_average(solver.configuration)
    native = ShootingHFSolver(solver.Z, shells, jj_configs=ensemble,
        n_grid=getattr(solver, 'n_grid', 1000),
        r_max=solver.r_max or 75., nuclear=solver.nuclear,
        screening_potential=solver.screening_potential)
    seed = native.solve(max_iter=max_iter, tol=min(tolerance, 1e-7))
    workspace = GridCIWorkspace.from_shooting(native, seed)
    reference = native.functional
    functional = CSFRadialFock(deepcopy(shells), solver.csfs, workspace, reference)
    keys = functional.keys
    def diagonalize(pa, qa):
        orbitals = {key: (pa[i], qa[i]) for i, key in enumerate(keys)}
        return ci_solve(solver.csfs, orbitals, keys, workspace, solver.Z, workspace.n)
    ci = diagonalize(seed._pa, seed._qa)
    functional.set_state(ci)
    native.functional = functional
    native.subshells = functional.subshells
    native.jj_configs = None
    history = [float(ci.ground_energy)]
    def update_state(ci):
        previous = functional.coefficients
        functional.set_state(ci)
        current = functional.coefficients
        return max((abs(current.get(k, 0.)-previous.get(k, 0.)) /
                    max(1., abs(current.get(k, 0.)), abs(previous.get(k, 0.)))
                    for k in current.keys() | previous.keys()), default=0.)

    def begin_sweep(pa, qa):
        nonlocal ci
        ci = diagonalize(pa, qa)
        change = update_state(ci)
        history.append(float(ci.ground_energy))
        return change
    native.state_update = begin_sweep
    result = native.solve(max_iter=max_iter, tol=tolerance, seed=seed,
                          orbital_tolerance=orbital_tolerance)
    ci = diagonalize(result._pa, result._qa)
    state_residual = update_state(ci)
    threshold = max(1e-8, np.sqrt(min(tolerance, 1e-7)))
    residual = max(state_residual,
        result.solver_diagnostics['occupied_rotation_residual_Ha'],
        result.solver_diagnostics['undamped_orbital_update_norm'])
    converged = bool(result.converged and residual <= threshold)
    # The final diagonalization may change the coherent density. Keep that
    # dimensionless check separate from the occupied-rotation defect in Ha.
    if orbital_tolerance is not None:
        converged = bool(converged and state_residual <= orbital_tolerance)
    orbitals = {key: (result._pa[i], result._qa[i]) for i, key in enumerate(keys)}
    diagnostics = dict(result.solver_diagnostics,
        converged=converged, iterations=result.iterations,
        termination_reason='converged' if converged else 'stationarity_not_reached',
        exchange='nonlocal_coherent_csf',
        initializer_iterations=seed.iterations, initializer_converged=seed.converged,
        state_density_residual=state_residual,
        model='monoconfiguration Dirac-Coulomb DF',
        functional='coherent CSF Hamiltonian expectation',
        radial_solver='direct coupled inhomogeneous Dirac boundary-value equations',
        electron_projector='self-consistent electron branch, radial node constraints',
        continuum_basis=False, full_correlated_orbital_gradient=True,
        energy_history=history+[float(ci.ground_energy)],
        orbital_residual=residual,
        orbital_tolerance=max(1e-8, np.sqrt(min(tolerance, 1e-7))),
        state_density_tolerance=(threshold if orbital_tolerance is None
                                 else min(threshold, orbital_tolerance)),
        radial_space_converged=False,
        radial_space_convergence_reason='requires radial grid and box refinement',
        zero_occupation_orbitals='reference DF fields; no artificial Hamiltonian populations',
        n_grid=workspace.n, radial_box_bohr=float(workspace.r[-1]))
    return orbitals, ci, workspace, result, diagnostics
