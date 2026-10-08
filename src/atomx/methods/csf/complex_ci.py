"""Finite positive-band nuclear-projector complex DC or static DCB CI.

Reuse full signed jj CSF algebra and shared Y^k numerics. Explicitly
separate this complex result from real bound-state CIResult/XFEL rates.
No automatic pole classifier, empirical shifts, QED or extra self-energy.
"""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy import linalg

from atomx.angular.csf_hamiltonian import csf_signature, exact_block_matrix
from atomx.config.csf import enumerate_jj_orbital_space
from atomx.numerics.bspline.complex_dirac import ComplexScaledDiracWorkspace
from atomx.numerics.complex_eigensolver import (
    complex_target_spectrum,
    validate_complex_target_request,
)
from atomx.physics.csf.complex_radial import ComplexCIRadialSnapshot


@dataclass(frozen=True)
class ComplexCIBlock:
    theta: float
    J2: int
    parity: int
    energies: np.ndarray
    coefficients: np.ndarray
    residuals: np.ndarray
    csfs: tuple
    radial: ComplexCIRadialSnapshot
    include_breit: bool = False

    @property
    def full_spectrum(self):
        return True

    @property
    def hamiltonian(self):
        return ('point-nuclear static DCB (Gaunt plus retardation)' if self.include_breit
                else 'point-nuclear DC')


@dataclass(frozen=True,kw_only=True)
class ComplexCITargetBlock(ComplexCIBlock):
    """Partial numerical spectrum in the full retained CSF basis, not poles."""
    sigma: complex
    absolute_residuals: np.ndarray
    solver_diagnostics: object

    @property
    def full_spectrum(self):
        return False


class ComplexScaledCIModel:
    """Retained-basis nuclear DC or full static DCB for arbitrary fixed N.

    All allowed CSFs in the explicit orbital counts are enumerated before
    radial allocation, with a magnetic-state cap. ``solve`` selects an exact
    J/parity block from that space, retaining every CSF in it. Angular/radial
    caches remain bounded by the original angular cap and owned snapshots.
    Count convergence, positive-band root identity and stationary poles
    require external scans; a box eigenvalue alone is not certification.
    ``include_breit=True`` uses both Gaunt and static retardation from the
    shared signed operator. Neither a scalar shift nor a Gaunt-only mode
    is substituted for the full Breit operator.
    ``radial_symmetry=True`` compiles exact reciprocal static radial keys
    and one determinant triangle. The owned c-product provider satisfies
    this contract; False retains the unreduced path for paired validation.
    """

    def __init__(self, knots, ks, Z, electrons, counts, *, n_gauss=None,
                 magnetic_state_cap=20000, max_determinants=20000,
                 radial_cache_bytes=64*1024**2, include_breit=False,
                 radial_symmetry=True):
        if not isinstance(include_breit, bool):
            raise ValueError("include_breit must be bool")
        if not isinstance(radial_symmetry, bool):
            raise ValueError('radial_symmetry must be bool')
        if not counts or any(isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k == 0
                             for k in counts):
            raise ValueError("nonzero integer kappa counts required")
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in counts.values()):
            raise ValueError("positive integer retained counts required")
        if isinstance(max_determinants, bool) or not isinstance(max_determinants, int) or max_determinants < 1:
            raise ValueError("positive determinant cap required")
        keys = tuple(sorted((abs(k)+(k > 0)+i, k) for k, n in counts.items() for i in range(n)))
        self.csfs = tuple(enumerate_jj_orbital_space(keys, electrons,
                                                  magnetic_state_cap=magnetic_state_cap))
        self.counts = MappingProxyType(dict(counts))
        self.workspaces = MappingProxyType({k: ComplexScaledDiracWorkspace(knots, ks, k, Z,
                                              n_gauss=n_gauss) for k in counts})
        self.max_determinants, self.radial_cache_bytes = max_determinants, radial_cache_bytes
        self.include_breit = include_breit
        self.radial_symmetry = radial_symmetry

    def _block_csfs(self,J2,parity):
        if (isinstance(J2, bool) or not isinstance(J2, (int, np.integer)) or J2 < 0
                or isinstance(parity, bool) or parity not in (-1, 1)):
            raise ValueError("nonnegative integer J2 and parity +/-1 required")
        csfs = tuple(c for c in self.csfs if (c.total_J2, c.parity) == (J2, parity))
        if not csfs:
            raise ValueError("requested J/parity has no CSFs in retained space")
        return csfs

    def _prepare_block(self, theta, J2, parity, *, as_operator=False):
        csfs=self._block_csfs(J2,parity)
        radial = ComplexCIRadialSnapshot(self.workspaces, self.counts, theta,
                                       cache_bytes=self.radial_cache_bytes,
                                       with_derivatives=self.include_breit)
        H = exact_block_matrix([csf_signature(c) for c in csfs], radial.evaluate,
                              max_determinants=self.max_determinants, integral_dtype=complex,
                              breit_evaluator=radial.evaluate_breit if self.include_breit else None,
                              radial_symmetry=self.radial_symmetry,as_operator=as_operator)
        return csfs,radial,H

    def solve(self, theta, J2, parity):
        """Full complex DC/DCB spectrum with C.T C normalization.

        Imaginary energies are exposed, not turned into physical decay rates.
        """
        csfs,radial,H=self._prepare_block(theta,J2,parity)
        energies, coefficients = linalg.eig(H)
        order = np.argsort(energies.real)
        energies, coefficients = energies[order], coefficients[:, order]
        norms = np.sum(coefficients*coefficients, axis=0)
        if np.any(abs(norms) < 1e-10):
            raise ValueError("self-orthogonal complex CI root: exceptional point treatment required")
        coefficients /= np.sqrt(norms)[None, :]
        residuals = np.linalg.norm(H@coefficients-coefficients*energies, axis=0)
        residuals /= ((np.linalg.norm(H)+abs(energies))*np.linalg.norm(coefficients, axis=0))
        if not np.isfinite(energies).all() or np.max(residuals) > 1e-10:
            raise ValueError("complex CI algebraic residual gate failed")
        for array in (energies, coefficients, residuals):
            array.setflags(write=False)
        return ComplexCIBlock(float(theta), int(J2), int(parity), energies,
                              coefficients, residuals, csfs, radial, self.include_breit)

    def solve_target(self, theta, J2, parity, *, sigma, backend, roots=4, ncv=None,
                     tol=1e-11,maxiter=1000,linear_maxiter=100,linear_restart=60):
        """Partial near-shift spectrum; retain every CSF and owned radial state.

        The shift is a numerical search parameter, not an empirical energy
        correction or root classifier. ``backend='matrix_free'`` retains
        full determinant tables/projection and uses controlled GMRES without
        dense CSF H; ``backend='lu'`` explicitly builds CSF H and dense LU
        of H-sigma I. Neither builds a full eigenspectrum. Backend is required:
        avoiding dense H does not guarantee faster iterations.
        Residuals use the stricter original-action pair scale, different
        from the full solver's Frobenius scale. Failed inner/outer solves
        or unresolved c-product degeneracy reject without dense fallback.
        """
        if backend not in ('matrix_free','lu'):
            raise ValueError("target backend must be 'matrix_free' or 'lu'")
        options=validate_complex_target_request(len(self._block_csfs(J2,parity)),
            sigma=sigma,roots=roots,ncv=ncv,tol=tol,maxiter=maxiter,
            linear_maxiter=linear_maxiter,linear_restart=linear_restart)
        csfs,radial,H=self._prepare_block(theta,J2,parity,as_operator=backend=='matrix_free')
        result=complex_target_spectrum(H,**options)
        return ComplexCITargetBlock(theta=float(theta),J2=int(J2),parity=int(parity),
            energies=result.energies,coefficients=result.coefficients,
            residuals=result.residuals,csfs=csfs,radial=radial,
            include_breit=self.include_breit,sigma=complex(sigma),
            absolute_residuals=result.absolute_residuals,solver_diagnostics=result.diagnostics)


__all__ = ['ComplexScaledCIModel', 'ComplexCIBlock','ComplexCITargetBlock']
