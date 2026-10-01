"""Stationary rotations of two orthonormal orbitals for a quartic energy.

The physical functional supplies the rotation pairs and harmonic content;
this routine operates on either radial samples or basis coefficients.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def rotate_occupied(pa, qa, pairs, energy, *, harmonics=(2, 4), pair_energy=None):
    """Update orbital buffers and return the final callback energy.

    ``pairs`` contains (first, second, follow_stationary_saddle) tuples.
    Coherent quartic states require harmonics (1, 2, 3, 4); fixed diagonal
    ensembles can use (2, 4). The buffers must already be orthonormal.
    Optional ``pair_energy(a,b)`` creates an equivalent energy callback
    whose caches are valid only during that pair's rotation.
    """
    if tuple(harmonics) not in ((2, 4), (1, 2, 3, 4)):
        raise ValueError('unsupported quartic orbital rotation harmonics')
    for a, b, core_hole in pairs:
        ap, aq, bp, bq = (v.copy() for v in (pa[a], qa[a], pa[b], qa[b]))
        sample_energy = energy if pair_energy is None else pair_energy(a, b)
        def rotate(t, *, a=a, ap=ap, aq=aq, b=b, bp=bp, bq=bq):
            c, s = np.cos(t), np.sin(t)
            pa[a], qa[a] = c*ap+s*bp, c*aq+s*bq
            pa[b], qa[b] = c*bp-s*ap, c*bq-s*aq
        angles = (np.arange(5)*np.pi/5 if harmonics == (2, 4)
                  else np.arange(9)*2*np.pi/9)
        values = []
        for t in angles:
            rotate(t)
            values.append(sample_energy())
        def row(t):
            return np.array([1., *(x for m in harmonics
                                  for x in (np.cos(m*t), np.sin(m*t)))])
        coef = np.linalg.solve(np.array([row(t) for t in angles]), values)
        # Evaluate E(t)-E(0) without subtracting nearly equal
        # cosine terms. This resolves rotations far below the
        # total-energy ulp (e.g. a deep K core/valence pair).
        def fit(t, *, coef=coef):
            return float(sum(-2*coef[2*i+1]*np.sin(m*t/2)**2
                             +coef[2*i+2]*np.sin(m*t)
                             for i, m in enumerate(harmonics)))
        if core_hole:
            # A constrained excited configuration may be a saddle.
            # Minimization can exchange radial identities and fill
            # the core hole. Follow the nearest stationary rotation,
            # just as the B-spline method does, including maxima.
            degree = max(harmonics)
            polynomial = np.zeros(2*degree+1, dtype=complex)
            for i, m in enumerate(harmonics):
                ac, bs = coef[2*i+1:2*i+3]
                polynomial[degree-m] = m*(bs+1j*ac)
                polynomial[degree+m] = m*(bs-1j*ac)
            roots = np.roots(polynomial)
            stationary = np.angle(roots[abs(abs(roots)-1)<1e-6])
            angle = (float(np.clip(stationary[np.argmin(abs(stationary))],
                                   -.1,.1)) if stationary.size else 0.)
            rotate(angle)
        else:
            optimum = minimize_scalar(fit, bounds=(-.15, .15), method='bounded',
                                       options={'xatol': 1e-11})
            rotate(optimum.x if fit(optimum.x) < fit(0.) else 0.)
    return energy()
