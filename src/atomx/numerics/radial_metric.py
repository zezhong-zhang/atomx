"""Owned quadrature contract shared by radial engines and observables."""
from dataclasses import dataclass
import numpy as np


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
