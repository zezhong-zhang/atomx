"""Constrained grid optimization of a supplied energy/gradient functional."""
import numpy as np

from atomx.numerics.shooting.dirac_response import DiracResponse
from atomx.numerics.shooting.problem import ShootingRadialProblem


def optimize_grid_shooting(functional, orbitals, electron_space, response_energies,
                           *, max_iter=100, tolerance=1e-5,
                           progress_callback=None):
    """Optimize the full correlated grid CI functional within a fixed space.

    ``response_energies[(n,kappa)]`` selects a non-pole bound energy for the
    reference Dirac Green operator. The full CI gradient carries screening,
    exchange and correlation; the local response is an update operator.
    Same-kappa horizontal constraints are solved as a coupled border, and
    occupied-orbital rotations are retained separately. Armijo rejection
    and projected-gradient fallback protect the original energy functional.
    When the Armijo decrease threshold is below eight machine epsilons
    of the total energy and the measured energy change is unresolved, acceptance instead
    requires a measured decrease of the projected gradient norm; the final
    residual tolerance is never relaxed. Such steps are recorded explicitly.

    Secants, constraints, and line searches use orthonormal electron-basis
    coefficients. Only physical functional and Green evaluations use radial
    arrays. This is algebraically the same fixed-space minimization and
    prevents floating-point secant amplification outside the electron space.

    This is an experimental building block, not the default public engine.
    It needs a supplied, independently validated electron space; constructing
    a sufficiently complete all-element space is not implemented here.
    """
    if max_iter < 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('invalid iteration limit or tolerance')
    if not np.array_equal(electron_space.weights, functional.weights):
        raise ValueError('electron space and CI functional use different quadrature metrics')
    keys, ws, weight = functional.keys, functional.data, functional.weights
    if any(key[1] not in electron_space.basis for key in keys):
        raise ValueError('electron space is missing an active kappa')
    if set(response_energies) != set(keys):
        raise ValueError('provide one response energy for every active orbital')
    groups = {kappa: [k for k in keys if k[1] == kappa] for kappa in {k[1] for k in keys}}

    def analyze(vectors):
        result = {}
        for kappa, group in groups.items():
            block = electron_space.analyze(np.array([vectors[k] for k in group]), kappa)
            result.update(zip(group, block, strict=True))
        return result

    def expand(coefficients):
        result = {}
        for kappa, group in groups.items():
            block = electron_space.synthesize(np.array([coefficients[k] for k in group]), kappa)
            result.update(zip(group, block, strict=True))
        return result

    def retract(coefficients):
        result = {}
        for group in groups.values():
            block = np.array([coefficients[k] for k in group])
            values, vectors = np.linalg.eigh(block @ block.T)
            if values.min() <= 1e-14:
                raise ValueError('rank-deficient electron-space orbital trial')
            block = ((vectors/np.sqrt(values)) @ vectors.T) @ block
            result.update(zip(group, block, strict=True))
        return result

    def tangent(point, vector):
        result = {}
        for group in groups.values():
            x = np.array([point[k] for k in group])
            g = np.array([vector[k] for k in group])
            cross = g @ x.T
            block = g-.5*(cross+cross.T) @ x
            result.update(zip(group, block, strict=True))
        return result

    def gradient_at(point, derivative):
        # dE/d(radial samples) is a coordinate derivative. Analyze its
        # Riesz representative once; all secants and constraint operations
        # then stay in the orthonormal electron basis by construction.
        return tangent(point, analyze({k: derivative[k]/weight for k in keys}))

    point = retract(analyze(orbitals))
    waves = expand(point)
    reference_rv = getattr(electron_space, 'reference_rv', ws.rv_nuc)
    response_problem = ShootingRadialProblem(ws.r, ws.difr, reference_rv)
    responses = {k: DiracResponse(response_energies[k], k[1],
                                  response_problem,
                                  boundary='box')
                 for k in keys}
    history, fallbacks = [], []
    secants = []
    previous_point = previous_gradient = None
    def dot(a, b):
        return sum(float(a[k] @ b[k]) for k in keys)
    energy, derivative, ci = functional.evaluate(waves)
    history.append(energy)
    converged = False
    reason = 'maximum_iterations'
    step_guess = 1.
    worst_constraint = 0.
    roundoff_acceptances = 0
    for iteration in range(max_iter+1):
        gradient = gradient_at(point, derivative)
        residual = float(np.sqrt(dot(gradient, gradient)))
        if progress_callback is not None:
            progress_callback(iteration, waves, ci, {'energy_ha': energy, 'orbital_residual': residual})
        if residual < tolerance:
            converged, reason = True, 'converged'
            break
        if iteration == max_iter:
            break
        if previous_point is not None:
            transported = []
            for s, y, _ in secants:
                s, y = tangent(point, s), tangent(point, y)
                sy = dot(s, y)
                if sy > 1e-12*np.sqrt(dot(s, s)*dot(y, y)):
                    transported.append((s, y, 1./sy))
            secants = transported
            s = tangent(point, {k: point[k]-previous_point[k] for k in keys})
            old_gradient = tangent(point, previous_gradient)
            y = {k: gradient[k]-old_gradient[k] for k in keys}
            sy = dot(s, y)
            if sy > 1e-12*np.sqrt(dot(s, s)*dot(y, y)):
                secants.append((s, y, 1./sy))
                secants = secants[-8:]
        search_gradient = {k: gradient[k].copy() for k in keys}
        alphas = []
        for s, y, rho in reversed(secants):
            alpha = rho*dot(s, search_gradient)
            alphas.append(alpha)
            search_gradient = {k: search_gradient[k]-alpha*y[k] for k in keys}
        direction = {}
        failure = None
        try:
            for kappa, group in groups.items():
                constraints = np.array([waves[k] for k in group])
                # Horizontal Green solve plus the skew rotation component:
                # freezing all active-active rotations would change MCDHF.
                for key in group:
                    g = search_gradient[key]
                    active = np.array([point[k] for k in group])
                    rotation = (active @ g) @ active
                    source = electron_space.synthesize(-(g-rotation), kappa)
                    response, _, diagnostics = responses[key].solve(source, constraints)
                    worst_constraint = max(worst_constraint, diagnostics['constraint_residual'])
                    scale = 1./(2*max(abs(response_energies[key]), .5))
                    direction[key] = electron_space.analyze(response, kappa)-scale*rotation
            # Projection and numerical integration can introduce tiny
            # constraint normals; use the symmetric tangent projection.
            inverse_q = {k: -direction[k] for k in keys}
            for (s, y, rho), alpha in zip(secants, reversed(alphas), strict=True):
                beta = rho*dot(y, inverse_q)
                inverse_q = {k: inverse_q[k]+s[k]*(alpha-beta) for k in keys}
            direction = tangent(point, {k: -inverse_q[k] for k in keys})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            failure = str(exc)
        slope = dot(gradient, direction) if failure is None else np.inf
        if not np.isfinite(slope) or slope >= 0:
            failure = failure or 'Dirac response is not a descent direction in the supplied electron space'
        accepted = False
        for attempt in range(2):
            if failure is not None or attempt:
                secants.clear()
                fallbacks.append({'iteration': iteration, 'reason': failure or 'line search exhausted'})
                scale = 1./(2*max(functional.Z**2, 1.))
                direction = {k: -scale*gradient[k] for k in keys}
                slope = dot(gradient, direction)
                failure = None
                # The fallback has a different scale; carrying a tiny
                # rejected quasi-Newton step into it can freeze the solve.
                step_guess = 1.
            step = min(1., 2*step_guess)
            for _ in range(24):
                trial_derivative = None
                try:
                    trial_point = retract({k: point[k]+step*direction[k] for k in keys})
                    trial = expand(trial_point)
                    trial_energy, trial_ci = functional.evaluate(trial, gradient=False)
                except (ValueError, FloatingPointError):
                    step *= .5
                    continue
                # Deeply bound ions can resolve the gradient after their
                # total CI energy has reached floating-point resolution.
                # In that regime require a measured Armijo-like residual
                # reduction, never a small update or a relaxed tolerance.
                # Check this BEFORE Armijo: rounding energy+c*step*slope
                # back to energy otherwise accepts unchanged energies with
                # an increasing residual, causing near-converged cycles.
                resolution = 8*np.finfo(float).eps*max(1., abs(energy))
                if (np.isfinite(trial_energy) and -1e-4*step*slope <= resolution
                        and abs(trial_energy-energy) <= resolution):
                    _, trial_derivative, _ = functional.evaluate(trial, ci_result=trial_ci)
                    tg = gradient_at(trial_point, trial_derivative)
                    trial_residual = np.sqrt(dot(tg, tg))
                    if trial_residual < (1.-1e-4*step)*residual:
                        roundoff_acceptances += 1
                        accepted = True
                        break
                elif np.isfinite(trial_energy) and trial_energy <= energy+1e-4*step*slope:
                    accepted = True
                    break
                step *= .5
            if accepted:
                break
        if not accepted:
            reason = 'line_search_failed'
            break
        previous_point, previous_gradient = point, gradient
        point, waves = trial_point, trial
        if trial_derivative is None:
            energy, derivative, ci = functional.evaluate(waves, ci_result=trial_ci)
        else:
            energy, derivative, ci = trial_energy, trial_derivative, trial_ci
        history.append(energy)
        # An energy-noise backtrack says nothing about the useful step for
        # the next gradient. Do not propagate a near-zero noise-selected
        # step through subsequent secant iterations.
        step_guess = 1. if trial_derivative is not None else step
    return waves, ci, {
        'converged': converged, 'iterations': iteration, 'termination_reason': reason,
        'orbital_residual': residual, 'orbital_tolerance': tolerance,
        'energy_history': history, 'fallbacks': fallbacks,
        'roundoff_acceptances': roundoff_acceptances,
        'energy_resolution_ha': 8*np.finfo(float).eps*max(1., abs(energy)),
        'maximum_border_constraint_residual': worst_constraint,
        'electron_space_leakage': max(float(np.sqrt(np.sum(
            (waves[k]-electron_space.project(waves[k], k[1]))**2*weight))) for k in keys),
        'functional': 'full correlated Dirac-Coulomb grid CI',
        'corrections': 'Breit, QED, and recoil are outside this orbital functional',
        'electron_space': electron_space.provenance,
        'electron_reference': getattr(electron_space, 'reference_label', 'caller-supplied electron space'),
        'electron_reference_sha256': getattr(electron_space, 'reference_sha256', None),
        'electron_space_dimensions': {k: len(v) for k, v in electron_space.basis.items()},
        'update': 'constrained inhomogeneous Dirac shooting plus active-orbital rotations',
        'acceleration': 'transported Riemannian L-BFGS (8 secant pairs)',
        'optimizer_coordinates': 'orthonormal electron-basis coefficients',
        'scope': 'experimental; supplied finite electron space; no all-element completeness claim',
    }
