"""Full CI-gradient orbital optimization with a shooting electron basis.

A fixed nuclear no-pair projector is a model choice shared with the B-spline
MCDHF route. Radial cutoff convergence is reported separately from stationarity.
"""
import numpy as np

from atomx.angular.recoupling import lj_from_kappa
from atomx.config.subshell import l_letter


def solve_variational_shooting(solver, *, max_iter, tolerance):
    """Return the optimized orbitals, CI and native radial context."""
    from atomx.numerics.shooting.electron_space import continuum_enriched_electron_space
    from atomx.numerics.shooting.grid import sgrid
    from atomx.numerics.shooting.orbital_optimization import optimize_grid_shooting
    from atomx.physics.csf.matrix import GridCIWorkspace
    from atomx.potentials.radial_nucleus import screened_nuclear_rv
    from atomx.results import ShootingResult

    from .grid_functional import GridCSFFunctional

    controls = solver.shooting_space
    required = max(n-(k if k > 0 else -k-1) for n, k in solver.orbital_basis)
    n_bound = max(controls.bound_roots, required)
    radius = solver.r_max if solver.r_max is not None else max(75., 16.*n_bound**2/solver.Z)
    r, jacobian, n = sgrid(radius, 1e-7, radius/controls.outer_intervals, controls.n_grid)
    rv = screened_nuclear_rv(solver.Z, r, jacobian, n, nuclear=solver.nuclear,
                              screening_potential=solver.screening_potential)
    workspace = GridCIWorkspace(r, jacobian, rv)
    functional = GridCSFFunctional(solver.csfs, workspace, solver.Z,
                                  max_determinants=solver.max_determinants)
    space, initial = continuum_enriched_electron_space(functional,
        n_bound=n_bound, n_continuum=controls.continuum_roots)
    initializer = 'nuclear eigenfunctions'
    initializer_converged = None
    shifts = {key: 1.1*min(space.root_energies[key[1]]) for key in functional.keys}
    orbitals, ci, diagnostics = optimize_grid_shooting(functional, initial, space,
        shifts, max_iter=max_iter, tolerance=tolerance)
    _, derivative, _ = functional.evaluate(orbitals, ci_result=ci)
    # The grid functional differentiates h_ab/<b|b>. Restore the
    # removed normal derivative only for reporting physical multipliers;
    # it must remain removed in the tangent optimization gradient.
    normal = {key: 0. for key in functional.keys}
    for term, coefficient in functional.last_weights.items():
        if term[0] == 'h':
            _, na, kappa, nb = term
            a, b = (na, kappa), (nb, kappa)
            norm_b = float(np.sum(orbitals[b]**2*functional.weights))
            h_ab = float(np.sum(orbitals[a]*functional.one_body_action(orbitals[b], kappa)))
            normal[b] += coefficient*h_ab/norm_b
    energies = {}
    for key in functional.keys:
        n_shell, kappa = key
        ell, tj = lj_from_kappa(kappa)
        label = f'{n_shell}{l_letter(ell)}{tj}/2'
        occupation = functional.last_occupations[key]
        if occupation > 1e-12:
            energies[label] = float((.5*np.sum(orbitals[key]*derivative[key])+normal[key])/occupation)
    diagnostics.update(
        initializer=initializer,
        seed_converged=initializer_converged,
        full_correlated_orbital_gradient=True,
        model='fixed nuclear no-pair Dirac-Coulomb MCDHF',
        scope='stationarity in the declared finite shooting electron space',
        radial_space_converged=False,
        radial_space_convergence_reason='requires box, grid and continuum-cutoff refinement',
        continuum_diagnostics=space.continuum_diagnostics,
        radial_box_bohr=float(radius), n_grid=n, bound_roots_per_kappa=n_bound,
        continuum_roots_per_kappa=controls.continuum_roots,
        maximum_electron_energy_Ha={k: max(v) for k, v in space.root_energies.items()})
    context = ShootingResult(bool(diagnostics['converged']), diagnostics['iterations'],
        float(ci.ground_energy), energies, Z=solver.Z, _grid=r, _difr=jacobian)
    return orbitals, ci, workspace, context, diagnostics
