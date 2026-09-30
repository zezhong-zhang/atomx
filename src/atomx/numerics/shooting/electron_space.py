"""Experimental full-gradient MCDHF updates on a shooting grid.

The electron space is an explicit input, not silently inferred from an
unrestricted Dirac grid. Callers must supply a validated positive-energy
radial basis for the intended no-pair Hamiltonian. This module performs no
B-spline evaluation or radial matrix diagonalization. Basis completeness
and its no-pair model remain independent convergence requirements.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


class FixedElectronSpace:
    """Fixed sampled electron-branch spinors grouped by kappa.

    Each value has shape (basis_size, 2, radial_points). Gram whitening
    corrects quadrature overlap defects. Supplying arbitrary spinors here
    does not prove they belong to the electron branch: provenance is
    mandatory, and branch validation belongs to their generating solver.
    """

    def __init__(self, basis, weights, *, provenance):
        if not isinstance(provenance, str) or not provenance.strip():
            raise ValueError('electron-space provenance is required')
        self.provenance = provenance
        self.weights = np.asarray(weights, dtype=float).copy()
        if self.weights.ndim != 1 or np.any(self.weights <= 0) or not np.all(np.isfinite(self.weights)):
            raise ValueError('invalid electron-space quadrature weights')
        self.basis = {}
        for kappa, vectors in basis.items():
            x = np.asarray(vectors, dtype=float)
            if (not isinstance(kappa, (int, np.integer)) or kappa == 0
                    or x.ndim != 3 or x.shape[1:] != (2, len(self.weights))
                    or not len(x) or not np.all(np.isfinite(x))):
                raise ValueError('invalid electron-space spinors')
            flat = x.reshape(len(x), -1)
            gram = (x*self.weights).reshape(len(x), -1) @ flat.T
            values, vectors = np.linalg.eigh(gram)
            if values.min() <= 1e-12*values.max():
                raise ValueError('linearly dependent electron-space spinors')
            self.basis[kappa] = (((vectors/np.sqrt(values))@vectors.T) @ flat).reshape(x.shape)

    def project(self, spinor, kappa):
        return self.synthesize(self.analyze(spinor, kappa), kappa)

    def analyze(self, spinors, kappa):
        """Metric coefficients of one spinor or a batch of spinors."""
        spinors = np.asarray(spinors)
        basis = self.basis[kappa]
        if spinors.shape[-2:] != (2, len(self.weights)):
            raise ValueError('spinors must end in (2, radial_points)')
        flat = basis.reshape(len(basis), -1)
        return (spinors*self.weights).reshape(*spinors.shape[:-2], -1) @ flat.T

    def synthesize(self, coefficients, kappa):
        """Expand electron coefficients; no projection or approximation."""
        coefficients = np.asarray(coefficients)
        basis = self.basis[kappa]
        if coefficients.shape[-1] != len(basis):
            raise ValueError('coefficient count does not match the electron space')
        return (coefficients @ basis.reshape(len(basis), -1)).reshape(
            *coefficients.shape[:-1], 2, len(self.weights))


def _electron_reference(functional, reference_rv, reference_label):
    if reference_rv is None:
        if reference_label is not None:
            raise ValueError('reference_label requires reference_rv')
        rv, label = functional.data.rv_nuc, 'Hamiltonian one-body potential'
    else:
        if not isinstance(reference_label, str) or not reference_label.strip():
            raise ValueError('an explicit electron reference needs a reference_label')
        rv, label = reference_rv, reference_label
    rv = np.array(rv, dtype=float, copy=True)
    if rv.shape != (functional.ns,) or not np.all(np.isfinite(rv)):
        raise ValueError('reference_rv must be finite on the functional grid')
    rv.setflags(write=False)
    return rv, label


def bound_electron_space(functional, n_roots=3, *, reference_rv=None,
                         reference_label=None):
    """Diagnostic electron space generated entirely by bound-state shooting.

    This intentionally incomplete space excludes the positive continuum.
    It tests the optimizer without using any B-spline solver, but must not
    be used as an all-element production accuracy model. Increasing only
    n_roots cannot repair the missing continuum.

    An explicit local reference changes the electron projector only, not
    the physical one-body potential in the CI Hamiltonian. Its label is
    mandatory so unlike projectors cannot be silently treated as one model.
    """
    from atomx.numerics.shooting.constants import SL
    from atomx.numerics.shooting.dirac_bound import dbound
    from atomx.numerics.shooting.spline import splin0
    if not isinstance(n_roots, int) or n_roots < 1:
        raise ValueError('n_roots must be a positive integer')
    ws = functional.data
    rv, label = _electron_reference(functional, reference_rv, reference_label)
    potential = splin0(ws.r, rv, ws.n, 0., 0.)
    basis, orbitals, energies = {}, {}, {}
    for kappa in sorted({k[1] for k in functional.keys}):
        ell = kappa if kappa > 0 else -kappa-1
        basis[kappa], energies[kappa] = [], []
        for principal in range(ell+1, ell+1+n_roots):
            e, ier, p, q, _ = dbound(-functional.Z**2/(2.*principal**2), 1e-11,
                principal, kappa, ws.r, rv, *potential, ws.n, ws.r, ws.n)
            if ier or not -SL**2 < e < 0:
                raise RuntimeError(f'bound electron root failed: n={principal}, kappa={kappa}, ier={ier}')
            phi = np.array([p, q])
            norm = float(np.sum(phi*phi*functional.weights))
            if not np.isfinite(norm) or norm <= 1e-20:
                raise FloatingPointError('invalid bound electron spinor')
            phi /= np.sqrt(norm)
            # A large boundary amplitude means the finite grid does not
            # represent this infinite-domain bound state adequately.
            if np.linalg.norm(phi[:, -1]) > 1e-7*np.max(abs(phi)):
                raise ValueError('radial box is too short for the requested bound basis')
            basis[kappa].append(phi)
            energies[kappa].append(float(e))
            if (principal, kappa) in functional.keys:
                orbitals[principal, kappa] = phi
    if set(orbitals) != set(functional.keys):
        raise ValueError('bound basis does not contain all active principal labels')
    space = FixedElectronSpace(basis, functional.weights,
        provenance=f'local Dirac bound states from dbound; reference={label}; '
                   f'{n_roots} per kappa; continuum omitted')
    space.root_energies = energies
    space.reference_rv = rv
    space.reference_label = label
    space.reference_sha256 = hashlib.sha256(ws.r.tobytes()+rv.tobytes()).hexdigest()
    return space, functional.retract(orbitals)


def continuum_enriched_electron_space(functional, *, n_bound=3, n_continuum=12,
                                      include_negative_box=True, reference_rv=None,
                                      reference_label=None):
    """Enrich localized nuclear bound roots with positive box pseudostates.

    This is a controlled diagnostic extension with a finite positive-energy
    cutoff. By default diffuse negative box roots are included too. Bound roots must
    have negligible boundary amplitude (checked by bound_electron_space),
    and the combined raw basis must satisfy quadrature orthogonality before
    whitening. The number of negative box roots omitted is reported.
    """
    from atomx.numerics.shooting.box_continuum import BoxDiracContinuum
    space, orbitals = bound_electron_space(functional, n_roots=n_bound,
        reference_rv=reference_rv, reference_label=reference_label)
    ws = functional.data
    solver = BoxDiracContinuum(ws.r, ws.difr, space.reference_rv)
    combined, energies, diagnostics = {}, {}, {}
    for kappa, bound in space.basis.items():
        positive, waves, report = solver.solve(kappa, n_continuum)
        negative = np.empty(0)
        diffuse = np.empty((0, 2, ws.n))
        if include_negative_box:
            negative, diffuse, matching = solver.negative_roots_above(kappa,
                .99*max(space.root_energies[kappa]), expected_below=n_bound)
            report.update(matching)
        basis = np.concatenate([bound, diffuse, waves])
        flat = basis.reshape(len(basis), -1)
        gram = (basis*functional.weights).reshape(len(basis), -1) @ flat.T
        defect = float(np.max(abs(gram-np.eye(len(basis)))))
        if defect > 5e-6:
            raise ValueError(f'bound/continuum overlap defect {defect:.3g}; refine the grid or box')
        report['combined_overlap_defect'] = defect
        report['omitted_negative_box_roots'] = max(0, report['initial_zero_energy_nodes']-n_bound-len(negative))
        report['scope'] = ('localized bound plus finite positive box space; '
                           f'diffuse negative box roots included={include_negative_box}')
        diagnostics[kappa] = report
        combined[kappa] = basis
        energies[kappa] = list(space.root_energies[kappa])+negative.tolist()+positive.tolist()
    enriched = FixedElectronSpace(combined, functional.weights,
        provenance=f'local Dirac shooting; reference={space.reference_label}: '
                   f'{n_bound} localized bound roots and '
                   f'{n_continuum} positive P(R)=0 box roots per kappa; '
                   f'diffuse negative box roots included={include_negative_box}; '
                   'positive-energy cutoff convergence not established')
    enriched.root_energies = energies
    enriched.continuum_diagnostics = diagnostics
    enriched.reference_rv = space.reference_rv
    enriched.reference_label = space.reference_label
    enriched.reference_sha256 = space.reference_sha256
    return enriched, functional.retract({k: enriched.project(v, k[1]) for k, v in orbitals.items()})


@dataclass(frozen=True)
class ShootingSpace:
    """Finite electron-space controls; sizes are per active kappa."""
    n_grid: int = 6000
    bound_roots: int = 3
    continuum_roots: int = 384
    outer_intervals: int = 4000

    def __post_init__(self):
        for key in ('n_grid', 'bound_roots', 'continuum_roots', 'outer_intervals'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.n_grid < 100 or self.outer_intervals >= self.n_grid:
            raise ValueError('require n_grid >= 100 and outer_intervals < n_grid')
