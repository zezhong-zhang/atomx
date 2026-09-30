"""Bound radial refinement using independently converged SCF samples.

The callback owns the Hamiltonian and state selection. This controller owns
only box/grid changes and comparisons. It retains SGRID's uniform computational
coordinate: inserting arbitrary physical-radius points would invalidate SLAG6.
Tail mass is checked for *every* orbital, including exchange-induced core tails.
It is a warning indicator, not an estimate of the omitted infinite-domain norm;
an enlarged-box calculation provides the separate boundary check.
"""
from dataclasses import dataclass
import math
from numbers import Integral
import numpy as np

@dataclass(frozen=True)
class RadialRefinement:
    """Opt-in checks for single-configuration shooting calculations.

    Energy tolerances are absolute Hartree differences between successive
    discretizations, not certified error bounds. All orbital energies and, when
    present, state excitation energies are checked. An explicit radial box is
    held fixed; otherwise the box can grow. The sampling point count is the
    starting count once this policy is requested. SCF convergence is independent.
    """
    energy_atol: float = 1e-6
    orbital_atol: float = 1e-5
    tail_norm: float = 1e-10
    scf_tol: float = 1e-11
    max_points: int = 8000
    max_radius: float = 1200.
    max_solves: int = 8

    def __post_init__(self):
        for key in ('energy_atol', 'orbital_atol', 'tail_norm', 'scf_tol', 'max_radius'):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.tail_norm >= 1:
            raise ValueError('tail_norm must be less than one')
        for key, minimum in (('max_points', 50), ('max_solves', 2)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{key} must be an integer >= {minimum}')


@dataclass
class RadialSample:
    """Numerical measurements and opaque result from the supplied solve."""
    value: object
    energy: float
    orbital_energies: dict
    radius: np.ndarray
    weights: np.ndarray
    densities: np.ndarray
    converged: bool
    excitation_energies: np.ndarray

    def tail(self):
        norm = self.densities @ self.weights
        if (np.any(norm <= 0) or not np.all(np.isfinite(norm))
                or not np.all(np.isfinite(self.densities))):
            raise FloatingPointError('invalid orbital norm during radial refinement')
        outer = self.radius >= .8*self.radius[-1]
        return float(np.max((self.densities[:, outer] @ self.weights[outer])/norm))


def _differences(a, b):
    if (a.orbital_energies.keys() != b.orbital_energies.keys()
            or a.excitation_energies.shape != b.excitation_energies.shape):
        raise ValueError('radial refinement changed the requested state/orbital set')
    return dict(energy_change_Ha=abs(b.energy-a.energy),
        orbital_change_Ha=max((abs(v-b.orbital_energies[k])
                              for k, v in a.orbital_energies.items()), default=0.),
        excitation_change_Ha=float(np.max(np.abs(
            b.excitation_energies-a.excitation_energies), initial=0.)))


def refine_radial_grid(solve, *, n_grid, r_max, policy, vary_box):
    """Return (sample, audit) after grid and optional box refinement.

    ``solve(n_grid, r_max)`` must rebuild the complete state on that grid and
    return a RadialSample. No interpolation of spinors, frozen exchange sources,
    or changed orbital boundary conditions is used. A box increase also adds
    points; a failed box comparison therefore triggers another grid check.
    """
    from .dirac_orbital import RadialBoxError
    if n_grid > policy.max_points or r_max > policy.max_radius:
        raise ValueError('initial radial grid exceeds refinement limits')
    history, current = [], None
    n, radius, phase = int(n_grid), float(r_max), 'initial'
    grid_ok = box_ok = False
    reason = 'solve_limit'

    def grow_box():
        return (min(policy.max_points, math.ceil(n*1.5)),
                min(policy.max_radius, radius*1.5))

    for _ in range(policy.max_solves):
        try:
            candidate = solve(n, radius)
        except RadialBoxError as exc:
            history.append(dict(n_grid=n, r_max=radius, check=phase,
                                status='box_too_small', detail=str(exc)))
            if not vary_box or radius >= policy.max_radius:
                raise
            n, radius = grow_box()
            current, phase, grid_ok = None, 'box_recovery', False
            continue
        tail = candidate.tail()
        row = dict(n_grid=n, actual_points=len(candidate.radius), r_max=radius,
            check=phase, energy_Ha=candidate.energy, tail_norm=tail,
            scf_converged=bool(candidate.converged))
        history.append(row)
        # SGRID may raise a too-small requested point count. Refine the
        # actual grid, otherwise two requests can produce identical meshes
        # and falsely pass a zero-difference comparison.
        n = max(n, len(candidate.radius))
        if n > policy.max_points:
            current, reason = candidate, 'point_limit'
            grid_ok = False
            break
        numbers = [candidate.energy, *candidate.orbital_energies.values(),
                   *candidate.excitation_energies]
        if not np.all(np.isfinite(numbers)):
            raise FloatingPointError('nonfinite energy during radial refinement')
        if not candidate.converged:
            current, reason = candidate, 'scf_not_converged'
            break
        agrees = False
        if current is not None:
            changes = _differences(current, candidate)
            row.update(changes)
            agrees = (changes['energy_change_Ha'] <= policy.energy_atol
                and changes['orbital_change_Ha'] <= policy.orbital_atol
                and changes['excitation_change_Ha'] <= policy.energy_atol)
        current = candidate
        if tail > policy.tail_norm:
            grid_ok = False
            if not vary_box:
                reason = 'fixed_box_tail_not_resolved'
                break
            if radius >= policy.max_radius:
                reason = 'radius_limit'
                break
            n, radius = grow_box()
            phase = 'tail_expansion'
            continue
        if phase == 'box' and agrees:
            box_ok, reason = True, 'grid_and_box_converged'
            break
        if phase == 'grid' and agrees:
            grid_ok = True
            if not vary_box:
                reason = 'grid_converged_fixed_box'
                break
            if radius >= policy.max_radius:
                reason = 'radius_limit'
                break
            n, radius = grow_box()
            phase = 'box'
            continue
        grid_ok = False
        if n >= policy.max_points:
            reason = 'point_limit'
            break
        n, phase = min(policy.max_points, math.ceil(n*1.5)), 'grid'
    if current is None:
        raise RadialBoxError('radial refinement exhausted without a resolved bound state')
    accepted = grid_ok and (box_ok or not vary_box)
    return current, dict(converged=accepted, grid_converged=grid_ok,
        box_converged=box_ok, box_fixed=not vary_box, termination_reason=reason,
        criterion='absolute total, orbital and excitation energy changes; outer 20% norm',
        radial_space_converged=grid_ok and box_ok, history=history)
