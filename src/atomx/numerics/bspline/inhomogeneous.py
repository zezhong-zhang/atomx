"""Constrained Dirac source equation in a B-spline representation."""
import numpy as np
from scipy.linalg import cholesky, eigh, qr, solve_triangular
from scipy.optimize import brentq

from atomx.numerics.shooting.constants import SL


def solve_constrained_source(hamiltonian, metric, source, reference, *,
                             principal, kappa, constraints=(), lower_occupied=0):
    """Solve (H-eS)x=b with x†Sx=1 and occupied orthogonality.

    The positive bound root is selected by its radial index. The Green
    response retains both signs of the Dirac spectrum, as does direct ODE
    shooting; this is a stationary DF root, not minimization in a fixed
    nuclear positive-energy subspace.
    """
    active = np.flatnonzero(np.diag(metric) > 0)
    s = metric[np.ix_(active, active)]
    h = hamiltonian[np.ix_(active, active)]
    lower = cholesky(s, lower=True)
    c = np.asarray(constraints)
    if c.size:
        y = lower.T @ c[:, active].T
        q, _ = qr(y, mode='full')
        complement = q[:, len(c):]
    else:
        complement = np.eye(len(active))
    transform = solve_triangular(lower.T, complement, lower=False)
    energies, rotation = eigh(transform.T @ h @ transform)
    states = transform @ rotation
    ell = kappa if kappa > 0 else -kappa-1
    index = principal-ell-1-lower_occupied
    positive = np.flatnonzero(energies > -SL**2)
    if index < 0 or index >= len(positive):
        raise ValueError('requested radial branch is absent from the constrained basis')
    target = positive[index]
    pole = energies[target]
    amplitudes = states.T @ source[active]
    if pole >= 0:
        raise ValueError('requested local Dirac branch is not bound')
    if np.linalg.norm(amplitudes) < 1e-13:
        energy, vector = pole, states[:, target]
    else:
        # Attractive exchange places the physical frozen-source root below
        # the matching local pole. With lower occupied roots removed this
        # is the same branch used by the coupled shooting boundary solver.
        low = max(-.999*SL**2, energies[target-1]+1e-9) if index > 0 else -.999*SL**2
        high = np.nextafter(pole, -np.inf)
        def norm_defect(energy):
            return float(np.sum((amplitudes/(energies-energy))**2)-1.)
        if norm_defect(high) < 0:
            # A vanishing source residue at the pole admits a homogeneous
            # component. Keep the continuous reference phase.
            coefficient = amplitudes/(energies-pole + (np.arange(len(energies)) == target))
            coefficient[target] = 0.
            norm = float(coefficient@coefficient)
            coefficient[target] = np.sqrt(max(0., 1.-norm))
            if float(states[:, target] @ s @ reference[active]) < 0:
                coefficient[target] *= -1
            energy, vector = pole, states @ coefficient
        else:
            if norm_defect(low) > 0:
                # A source residue at the preceding pole makes the norm
                # diverge at both ends. It is strictly convex between
                # poles; bracket the upper root from its unique minimum.
                def norm_derivative(energy):
                    return float(2*np.sum(amplitudes**2/(energies-energy)**3))

                if norm_derivative(low) >= 0:
                    raise ValueError('no normalized source root on the requested branch')
                low = brentq(norm_derivative, low, high, xtol=1e-11)
                if norm_defect(low) > 0:
                    raise ValueError('no normalized source root between adjacent poles')
            energy = brentq(norm_defect, low, high, xtol=1e-11)
            vector = states @ (amplitudes/(energies-energy))
    out = np.zeros_like(reference)
    out[active] = vector
    out /= np.sqrt(float(out @ metric @ out))
    if float(out @ metric @ reference) < 0:
        out *= -1
    return out, float(energy)
