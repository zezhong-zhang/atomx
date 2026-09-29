"""Two-sided radial Green solves for correlated orbital update directions.

This is a numerical preconditioner, not an approximate atomic Hamiltonian.
The full CI derivative remains the source, and the caller must project into
its fixed no-pair space, enforce orthogonality, and test the actual energy.
We invert -1/2 d²/dr² + l(l+1)/(2r²) + shift with zero endpoint
values. Its homogeneous modified Bessel functions permit inward/outward sweeps
without a matrix factorization. Applying it to both spinor components and
projecting back gives a symmetric positive preconditioner. It is not a full
GRASP-style coupled Dirac orbital solver.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _sweep(source, decay, origin, outer, denominator):
    n, columns = source.shape
    result = np.empty_like(source)
    acc = np.zeros(columns)
    for i in range(n):
        if i:
            acc *= decay[i - 1]
        acc += origin[i] * source[i]
        result[i] = outer[i] * acc
    acc[:] = 0.
    for i in range(n - 1, -1, -1):
        # The diagonal term was already included in the outward sweep.
        result[i] += origin[i] * acc
        acc += outer[i] * source[i]
        if i:
            acc *= decay[i - 1]
    return result / denominator


class RadialGreen:
    """Quadrature Nyström inverse of a positive Dirichlet radial operator."""

    def __init__(self, radius, weights, boundary, shift=0.5, ell=0):
        self.radius = np.asarray(radius, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        if (self.radius.ndim != 1 or self.radius.size < 2
                or self.weights.shape != self.radius.shape
                or not np.all(np.isfinite(self.radius))
                or not np.all(np.isfinite(self.weights))
                or np.any(np.diff(self.radius) <= 0)
                or self.radius[0] <= 0 or np.any(self.weights <= 0)
                or not np.isfinite(boundary) or boundary <= self.radius[-1]
                or not np.isfinite(shift) or shift <= 0
                or not isinstance(ell, (int, np.integer)) or ell < 0):
            raise ValueError("invalid radial Green grid, boundary, or shift")
        a = np.sqrt(2 * shift)
        self.decay = np.exp(-a * np.diff(self.radius))
        self.origin = -np.expm1(-2 * a * self.radius)
        self.outer = -np.expm1(-2 * a * (boundary - self.radius))
        self.denominator = a * -np.expm1(-2 * a * boundary)
        if ell:
            from scipy.special import ive, kve
            # Scaled regular/irregular solutions sqrt(r) I/K_{l+1/2}(ar).
            # Their unscaled Wronskian is -1, giving Green prefactor 2.
            nu = ell + .5
            regular = ive(nu, a*self.radius)
            irregular = kve(nu, a*self.radius)
            log_ratio = (np.log(kve(nu, a*boundary)) - np.log(ive(nu, a*boundary))
                         + np.log(regular) - np.log(irregular)
                         - 2*a*(boundary-self.radius))
            self.origin = np.sqrt(self.radius)*regular
            self.outer = np.sqrt(self.radius)*irregular*(-np.expm1(log_ratio))
            self.denominator = .5
            if not np.all(np.isfinite(self.outer)):
                raise FloatingPointError('non-finite centrifugal Green solution')

    def apply(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[0] != self.radius.size:
            raise ValueError("source must have shape (radial nodes, columns)")
        return _sweep(values * self.weights[:, None], self.decay,
                      self.origin, self.outer, self.denominator)


class ShootingPreconditioner:
    """Fixed positive radial inverse in the optimizer's no-pair coordinates.

    Occupation scaling is frozen at construction, with the same floor as
    the spectral preconditioner. No division by a zero CI population occurs.
    """

    def __init__(self, functional, layout, occupation_floor, point,
                 reference_matrices=None):
        data = functional.data
        radius, weights = data['gr'].ravel(), data['grw'].ravel()
        boundary = float(data['t'][-1])
        self.groups = []
        ns = functional.ns
        for keys, projector, shape, where in layout:
            sampled = []
            for component, offset in [('p', 0), ('q', ns)]:
                basis = data['bsp_' + component]
                ncomp = data.get('ns' + component, ns)
                values = np.zeros((*basis.shape[:2], projector.shape[1]))
                cells = np.arange(basis.shape[0])
                for local in range(basis.shape[2]):
                    indices = cells + local
                    valid = indices < ncomp
                    values[valid] += (basis[valid, :, local, None]
                                      * projector[offset + indices[valid], None, :])
                sampled.append(values.reshape(-1, projector.shape[1]))
            population = np.array([max(functional.last_occupations.get(k, 0.),
                                       occupation_floor) for k in keys])
            # The radial operators and projector are fixed throughout this
            # solve. Shoot all projected basis sources once and cache the
            # small reduced inverse, avoiding repeated grid transforms.
            inverses = []
            for column, key in enumerate(keys):
                orbital = projector @ point[where].reshape(shape)[:, column]
                h = (reference_matrices[key] if reference_matrices is not None
                     else functional.one_body[key[1]])
                # Match the bound orbital's asymptotic kinetic scale. The
                # positive floor regularizes diffuse/near-zero modes only
                # in the preconditioner, never in the physical functional.
                shift = max(abs(float(orbital @ h @ orbital)), .5)
                inverse = np.zeros((projector.shape[1], projector.shape[1]))
                kappa = key[1]
                ell = kappa if kappa > 0 else -kappa-1
                small_ell = ell-1 if kappa > 0 else ell+1
                for basis, angular in zip(sampled, (ell, small_ell), strict=True):
                    green = RadialGreen(radius, weights, boundary, shift, angular)
                    inverse += basis.T @ (weights[:, None] * green.apply(basis))
                inverses.append(.5 * (inverse + inverse.T))
            self.groups.append((inverses, shape, where, population))

    def __call__(self, vector):
        result = np.zeros_like(vector)
        for inverses, shape, where, population in self.groups:
            rhs = vector[where].reshape(shape)
            update = np.column_stack([inverse @ rhs[:, i]
                                      for i, inverse in enumerate(inverses)])
            result[where] = (update / (2 * population)).ravel()
        return result
