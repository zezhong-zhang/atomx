"""Owned, immutable data for a local radial shooting problem.

This numerical context does not decide whether its potential defines the
physical Hamiltonian, an electron projector, or a preconditioner. That role
belongs to the calling method. Units are bohr, Hartree and atomic units.
"""
from dataclasses import dataclass, field
from functools import cached_property
import hashlib

import numpy as np

from .quadrature import slag6_weights
from .spline import splin0


def _readonly(values, dtype=float):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, eq=False)
class ShootingRadialProblem:
    """Uniform-index SGRID metric and r*V potential, including the origin.

    Input arrays are copied so a later caller mutation cannot invalidate a
    cached spline or fundamental solution. Interpolation is constructed only
    when requested, then reused. Equality is identity, not array comparison.
    """
    r: np.ndarray
    jacobian: np.ndarray
    rv: np.ndarray
    weights: np.ndarray = field(init=False, repr=False)

    @classmethod
    def coerce(cls, radius, jacobian=None, rv_potential=None):
        """Accept an existing context or the legacy three-array interface."""
        if isinstance(radius, cls):
            if jacobian is not None or rv_potential is not None:
                raise ValueError('supply a radial problem or arrays, not both')
            return radius
        if jacobian is None or rv_potential is None:
            raise ValueError('radius, Jacobian, and potential are required')
        return cls(radius, jacobian, rv_potential)

    def __post_init__(self):
        arrays = [_readonly(v) for v in (self.r, self.jacobian, self.rv)]
        r, jacobian, rv = arrays
        if (r.ndim != 1 or len(r) < 9 or jacobian.shape != r.shape
                or rv.shape != r.shape or r[0] != 0 or np.any(np.diff(r) <= 0)
                or np.any(jacobian <= 0)
                or not all(np.all(np.isfinite(v)) for v in arrays)):
            raise ValueError('invalid shooting radius, Jacobian, or potential')
        weights = slag6_weights(len(r))*jacobian
        if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
            raise ValueError('shooting quadrature metric must be finite and positive')
        for name, value in zip(('r', 'jacobian', 'rv'), arrays, strict=True):
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'weights', _readonly(weights))

    @cached_property
    def spline(self):
        return tuple(_readonly(v) for v in splin0(self.r, self.rv, len(self.r), 0., 0.))

    @cached_property
    def indices(self):
        return _readonly(np.arange(len(self.r)), dtype=np.int64)

    @cached_property
    def fingerprint(self):
        """Identity of geometry, quadrature metric and sampled potential."""
        digest = hashlib.sha256()
        for values in (self.r, self.jacobian, self.rv):
            digest.update(values.tobytes())
        return digest.hexdigest()
