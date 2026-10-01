"""Owned quadrature contract shared by radial engines and observables."""
from dataclasses import dataclass

import numpy as np


def orbital_distance(left, right, metric):
    """Phase-aligned L2 distance of normalized spinors in their own metric.

    ``metric`` is an overlap matrix for coefficients or diagonal quadrature
    weights for samples. Subtract vectors before forming the norm so nearly
    identical states do not lose precision through ``2 - 2*overlap``.
    """
    left, right = np.asarray(left), np.asarray(right)
    metric = np.asarray(metric)
    action = (lambda x: metric*x) if metric.ndim == 1 else (lambda x: metric @ x)
    sign = -1. if float(left @ action(right)) < 0 else 1.
    delta = left-sign*right
    norm_squared = float(delta @ action(delta))
    if not np.isfinite(norm_squared):
        raise FloatingPointError('nonfinite orbital update norm')
    return float(np.sqrt(max(0., norm_squared)))


@dataclass(frozen=True, eq=False)
class RadialMetric:
    radius: np.ndarray
    weights: np.ndarray
    rule: str

    def __post_init__(self):
        r, w = (np.array(v, dtype=float, copy=True) for v in (self.radius, self.weights))
        if (r.ndim != 1 or w.shape != r.shape or np.any(np.diff(r) <= 0)
                or np.any(w <= 0) or not np.all(np.isfinite(r))
                or not np.all(np.isfinite(w))):
            raise ValueError('invalid radial quadrature')
        r.setflags(write=False)
        w.setflags(write=False)
        object.__setattr__(self, 'radius', r)
        object.__setattr__(self, 'weights', w)

    @classmethod
    def trapezoid(cls, radius):
        """Composite trapezoid on an arbitrary grid, including half endpoints."""
        r = np.asarray(radius, dtype=float)
        if r.ndim != 1 or r.size < 2:
            raise ValueError('trapezoidal radial quadrature needs at least two radii')
        intervals = np.diff(r)
        weights = np.empty_like(r)
        weights[0], weights[-1] = .5*intervals[0], .5*intervals[-1]
        weights[1:-1] = .5*(intervals[:-1]+intervals[1:])
        return cls(r, weights, 'trapezoid')

    @classmethod
    def shooting(cls, radius, jacobian):
        from .shooting.quadrature import slag6_weights
        return cls(radius, slag6_weights(len(radius))*jacobian, 'slag6')

    @classmethod
    def bspline(cls, workspace):
        return cls(workspace['gr'].ravel(), workspace['grw'].ravel(), 'gauss-legendre')

    def integrate(self, values):
        values = np.asarray(values)
        if values.shape[-1:] != self.radius.shape:
            raise ValueError('integrand does not match radial metric')
        return values @ self.weights
