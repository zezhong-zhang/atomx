"""Shared no-pair boundaries and frozen nuclear virtual correlation orbitals.

These are correlation basis functions, not certified spectroscopic excited
states or scattering channels. Virtuals minimize the supplied nuclear Dirac
one-body operator in the retained positive subspace orthogonal to all frozen
spinors of the same kappa. No empirical energies or screening models enter.
"""
import numpy as np
from scipy.linalg import eigh, null_space

from atomx.numerics.bspline.dirac_assembly import solve_dirac


def radial_projector_start(kappa, minimum_n, data):
    """First retained nuclear root, including equal-order kappa>0 exclusion.

    Missing lower reference shells are outside the active no-pair subspace.
    This is the same boundary used in constrained orbital minimization.
    """
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer))
           for x in (kappa, minimum_n)):
        raise ValueError('integer kappa and principal shell required')
    ell = kappa if kappa > 0 else -kappa-1
    if kappa == 0 or minimum_n < ell+1:
        raise ValueError('valid kappa and principal shell required')
    balanced = data.get('ksp', data['ks']) != data.get('ksq', data['ks'])
    return minimum_n-ell-1+int(kappa > 0 and not balanced)


def extend_nuclear_virtuals(orbitals, one_body, metric, data, *, layers=1, kappas=None):
    """Append lowest orthogonal virtuals in each existing kappa symmetry.

    Frozen inputs must be S-orthonormal and contained in the declared nuclear
    no-pair subspace. Return owned spinors and explicit construction metadata;
    reject inconsistent seeds instead of projecting or relaxing them silently.
    New labels are consecutive above the largest existing n for each kappa.
    An explicit ``kappas`` selects symmetries to extend, including new ones
    with supplied nuclear matrices. New symmetries start at n=l+1 and use
    the same physical branch exclusion as the orbital optimizer.
    """
    if isinstance(layers, (bool, np.bool_)) or not isinstance(layers, (int, np.integer)) or layers < 1:
        raise ValueError('positive integer number of virtual radial layers required')
    if not orbitals:
        raise ValueError('frozen orbitals required')
    if any(np.iscomplexobj(a) for pair in orbitals.values() for a in pair):
        raise ValueError('real frozen radial spinors required')
    requested = sorted({k for n, k in orbitals}) if kappas is None else tuple(kappas)
    if (not requested or len(set(requested)) != len(requested) or
            any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k == 0
                for k in requested)):
        raise ValueError('distinct nonzero integer kappa symmetries required')
    ns = data['ns']
    output = {key: tuple(np.array(a, dtype=float, copy=True) for a in pair)
              for key, pair in orbitals.items()}
    diagnostics = {}
    for kappa in sorted(requested):
        keys = sorted(key for key in orbitals if key[1] == kappa)
        frozen = (np.column_stack([np.concatenate(output[key]) for key in keys])
                  if keys else np.empty((2*ns, 0)))
        if kappa not in one_body or kappa not in metric:
            raise ValueError('nuclear Hamiltonian and metric required for every requested kappa')
        h, s = one_body[kappa], metric[kappa]
        if (frozen.shape[0] != 2*ns or not np.isfinite(frozen).all() or
                not np.allclose(frozen.T @ s @ frozen, np.eye(len(keys)), atol=1e-10, rtol=0)):
            raise ValueError('frozen spinors must be finite and S-orthonormal')
        spectral = solve_dirac(h, s, ns, kappa, nsp=data.get('nsp'), nsq=data.get('nsq'))
        ell = kappa if kappa > 0 else -kappa-1
        first = radial_projector_start(kappa, min(n for n, k in keys) if keys else ell+1, data)
        positive = spectral.coefficients[:, first:]
        overlaps = positive.T @ s @ frozen
        missing = frozen-positive @ overlaps
        leakage = float(np.max(np.sum(missing*(s @ missing), axis=0))) if keys else 0.
        if leakage > 1e-12:
            raise ValueError('frozen spinors lie outside the declared nuclear no-pair subspace')
        complement = null_space(frozen.T @ s @ positive) if keys else np.eye(positive.shape[1])
        if complement.shape[1] < layers:
            raise ValueError('not enough independent positive-energy virtual directions')
        basis = positive @ complement
        eps, vectors = eigh(basis.T @ h @ basis, basis.T @ s @ basis,
                            subset_by_index=(0, layers-1))
        virtuals = basis @ vectors
        for i in range(layers):
            key = (max(n for n, k in keys)+1 if keys else ell+1)+i, kappa
            output[key] = (virtuals[:ns, i].copy(), virtuals[ns:, i].copy())
            diagnostics[key] = dict(one_body_energy_hartree=float(eps[i]),
                excluded_nuclear_roots=first, frozen_projector_leakage=leakage,
                old_core_overlap=float(np.max(abs(frozen.T @ s @ virtuals[:, i]))) if keys else 0.,
                radial_label='orthogonal correlation orbital; not a certified excited state')
    return output, diagnostics
