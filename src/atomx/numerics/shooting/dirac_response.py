"""Reusable inhomogeneous Dirac response with coupled orbital constraints.

Solves (h_D[V]-energy) x = source + sum_j multiplier_j constraint_j,
with prescribed overlaps <constraint_j|x>. The border is solved through
Green responses, not by orthogonalizing the unconstrained solution later.
Energy must be in the bound-state gap and away from poles. This linear
response alone is not a positive-energy/no-pair projector.
"""
from __future__ import annotations

import numpy as np

from atomx.numerics.shooting.constants import SL
from atomx.numerics.shooting.dirac_green import (
    green_solve as _green_solve,
)
from atomx.numerics.shooting.dirac_green import (
    homogeneous_pair as _homogeneous_pair,
)
from atomx.numerics.shooting.dirac_green import (
    prepare_scaled_green,
    scaled_box_pair,
    scaled_green_solve,
)
from atomx.numerics.shooting.problem import ShootingRadialProblem


class DiracResponse:
    """Factor a local radial Dirac boundary-value problem once per energy.

    With ``boundary='decaying'`` the existing Salvat integrators set a
    practical infinity; sources beyond it are rejected, never silently
    truncated. ``boundary='box'`` instead sets P(R)=0 and retains sources
    over the entire finite radial box. The same homogeneous pair is reused
    for every source and constraint. Neither boundary selects a no-pair
    subspace by itself.
    """

    def __init__(self, energy, kappa, radius, jacobian=None, rv_potential=None,
                 tail_tolerance=1e-10, boundary='decaying'):
        self.problem = ShootingRadialProblem.coerce(radius, jacobian, rv_potential)
        self.r, self.difr = self.problem.r, self.problem.jacobian
        n = len(self.r)
        if not np.isfinite(energy) or not -SL**2 < energy < 0:
            raise ValueError('Dirac response energy must lie in (-c², 0)')
        if not isinstance(kappa, (int, np.integer)) or kappa == 0:
            raise ValueError('kappa must be a nonzero integer')
        if not np.isfinite(tail_tolerance) or not 0 < tail_tolerance < 1:
            raise ValueError('tail_tolerance must lie between zero and one')
        self.tail_tolerance = tail_tolerance
        if boundary not in ('decaying', 'box'):
            raise ValueError("boundary must be 'decaying' or 'box'")
        self.boundary = boundary
        self.weights = self.problem.weights
        potential = self.problem.spline
        self.scaled_pair = None
        self.pair = None
        if boundary == 'box':
            self.scaled_pair = scaled_box_pair(energy, kappa, self.r, *potential)
            self.scaled_quadrature = prepare_scaled_green(self.scaled_pair)
        else:
            self.pair = _homogeneous_pair(energy, kappa, self.r,
                self.problem.indices, n, *potential)
            if not np.isfinite(self.pair[4]) or abs(self.pair[4]) < np.finfo(float).tiny:
                raise FloatingPointError('singular Dirac Green response; shift energy away from pole')

    def apply(self, source):
        source = np.asarray(source, dtype=float)
        if source.shape != (2, len(self.r)) or not np.all(np.isfinite(source)):
            raise ValueError('source must be a finite (2, radial_points) spinor')
        if self.scaled_pair is not None:
            out = np.array(scaled_green_solve(*source, self.scaled_pair, self.difr,
                                             self.scaled_quadrature))
            if not np.all(np.isfinite(out)):
                raise FloatingPointError('non-finite Dirac response')
            return out
        p0, q0, pinf, qinf, wronskian, cutoff, _, _ = self.pair
        density = np.sum(source*source, axis=0)*self.weights
        total = float(np.sum(density))
        if total and np.sum(density[cutoff:]) > self.tail_tolerance**2*total:
            raise ValueError('source extends beyond practical infinity; use a shallower response energy')
        p, q = _green_solve(*source, p0, q0, pinf, qinf, wronskian,
                            self.difr, cutoff, stable_tail=True)
        out = np.zeros_like(source)
        out[:, :cutoff] = p, q
        if not np.all(np.isfinite(out)):
            raise FloatingPointError('non-finite Dirac response')
        return out

    def solve(self, source, constraints=(), overlaps=None):
        """Return (spinor, multipliers, diagnostics) for the bordered solve."""
        initial = self.apply(source)
        constraints = np.asarray(constraints, dtype=float)
        if constraints.size == 0:
            if overlaps is not None and np.asarray(overlaps).size:
                raise ValueError('overlaps supplied without constraints')
            return initial, np.empty(0), {'constraint_residual': 0., 'border_condition': 1.}
        if constraints.ndim != 3 or constraints.shape[1:] != initial.shape:
            raise ValueError('constraints must have shape (count, 2, radial_points)')
        count = len(constraints)
        target = np.zeros(count) if overlaps is None else np.asarray(overlaps, dtype=float)
        if target.shape != (count,) or not np.all(np.isfinite(target)):
            raise ValueError('invalid target overlaps')
        responses = np.array([self.apply(c) for c in constraints])
        border = np.einsum('iar,r,jar->ij', constraints, self.weights, responses)
        rhs = target-np.einsum('iar,r,ar->i', constraints, self.weights, initial)
        condition = float(np.linalg.cond(border))
        if not np.isfinite(condition) or condition > 1e12:
            raise np.linalg.LinAlgError('dependent constraints or singular Dirac border')
        multipliers = np.linalg.solve(border, rhs)
        result = initial+np.einsum('i,iar->ar', multipliers, responses)
        residual = np.einsum('iar,r,ar->i', constraints, self.weights, result)-target
        return result, multipliers, {
            'constraint_residual': float(np.max(abs(residual))),
            'border_condition': condition,
            'border_asymmetry': float(np.max(abs(border-border.T))),
        }
