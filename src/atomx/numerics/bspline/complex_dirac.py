"""Uniformly complex-scaled point-Coulomb Dirac equation in a DKB basis.

Implements the one-electron prerequisites of Zaytsev et al., arXiv:1909.11487,
Eqs. (2), (3), (6). This is NOT a many-electron resonance-pole provider.
The theta-dependent DKB basis and c-product (transpose, never adjoint)
are essential. Scaling an existing real CI matrix is not equivalent.

The workspace owns its knot/quadrature data and precontracts polynomial
coefficients in z=exp(-i*theta). Angle scans reuse all radial integrations.
Regularity at the origin retains the linear spline when its DKB partner
cancels (positive branch kappa=-1, negative branch kappa=+1). Both components
vanish at the outer box boundary. No Bloch surface operator is used.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg
from scipy.interpolate import BSpline

from atomx.constants import ALPHA, C_SQ, C
from atomx.numerics.bspline.basis import setup_gauss_points
from atomx.physics.dirac import Dirac


def _angle(theta):
    if (not np.isscalar(theta) or np.iscomplexobj(theta)
            or not np.isfinite(theta) or abs(theta) >= np.pi / 2):
        raise ValueError("theta must be real, finite and |theta| < pi/2")
    return float(theta)


@dataclass(frozen=True)
class ComplexDiracSpectrum:
    """All finite-box roots, with no physical resonance classification.

    Vectors are in the equilibrated DKB basis, c-normalized as v.T S v=1.
    Residuals measure algebraic accuracy only, not basis convergence.
    Imaginary parts of box roots MUST NOT be reported as decay rates.
    """

    theta: float
    energies: np.ndarray
    coefficients: np.ndarray
    residuals: np.ndarray
    angle_slopes: np.ndarray
    _owner: object = field(repr=False, compare=False)


class ComplexScaledDiracWorkspace:
    """Reusable point-nucleus radial DKB workspace for a single kappa.

    ``knots`` is the ordinary real atomx clamped B-spline knot vector;
    ``ks`` is its order. Subcritical Z including Z=0 is supported. No SCF,
    finite-nucleus, exchange, Breit or QED potential is inferred.
    """

    def __init__(self, knots, ks, kappa, Z, *, n_gauss=None):
        if np.iscomplexobj(knots):
            raise ValueError("DKB requires real knots; rotate the operator and basis, not quadrature")
        t = np.array(knots, dtype=float, copy=True)
        if (t.ndim != 1 or not np.isfinite(t).all()
                or not isinstance(ks, (int, np.integer)) or ks < 4
                or len(t) < 2 * ks + 1 or np.any(np.diff(t) < 0)
                or np.any(t[:ks] != 0) or t[-1] <= 0
                or np.any(t[-ks:] != t[-1])
                or np.any(np.diff(t[ks-1:-ks+1]) <= 0)):
            raise ValueError("DKB requires clamped, simple interior real knots, order >= 4")
        if (not isinstance(kappa, (int, np.integer)) or kappa == 0
                or not np.isscalar(Z) or np.iscomplexobj(Z) or not np.isfinite(Z)
                or Z < 0 or abs(kappa) <= ALPHA * Z):
            raise ValueError("nonzero integer kappa and subcritical nonnegative Z required")
        ng = ks + 2 if n_gauss is None else n_gauss
        if not isinstance(ng, (int, np.integer)) or ng < ks + 2:
            raise ValueError("DKB quadrature needs at least ks+2 points per cell")
        ns = len(t) - ks
        gr, grw = setup_gauss_points(t, ns, ks, ng)
        r, w = gr.ravel(), grw.ravel()
        # Existing bsplvd only supplies first derivatives. SciPy's analytic
        # spline derivatives supply the second derivative required by DKB.
        splines = BSpline(t, np.eye(ns), ks - 1, extrapolate=False)
        evaluated = tuple(splines(r, nu=d).T for d in range(3))
        positive_start = 1 if kappa == -1 else 2
        negative_start = 1 if kappa == 1 else 2
        Bp, dp, ddp = (v[positive_start:-2] for v in evaluated)
        Bn, dn, ddn = (v[negative_start:-2] for v in evaluated)
        zp, zn = np.zeros_like(Bp), np.zeros_like(Bn)
        P = (np.vstack((Bp, zn)), np.vstack((zp, (dn-kappa*Bn/r)/(2*C))))
        Q = (np.vstack((zp, Bn)), np.vstack(((dp+kappa*Bp/r)/(2*C), zn)))
        dP = (np.vstack((dp, zn)), np.vstack((zp, (ddn-kappa*dn/r+kappa*Bn/r**2)/(2*C))))
        dQ = (np.vstack((zp, dn)), np.vstack(((ddp+kappa*dp/r-kappa*Bp/r**2)/(2*C), zn)))
        dimension = len(Bp) + len(Bn)
        overlap = np.zeros((3, dimension, dimension))
        hamiltonian = np.zeros((4, dimension, dimension))
        equation = Dirac()
        for a in range(2):
            for b in range(2):
                overlap[a+b] += (P[a]*w) @ P[b].T + (Q[a]*w) @ Q[b].T
                # Share the signs and rest-energy convention with the
                # ordinary physical Dirac action; separate mass from z*h.
                zeros = np.zeros_like(P[b])
                radius = np.broadcast_to(r, P[b].shape)
                hP, _ = equation.radial_action(zeros, Q[b], zeros, dQ[b],
                                               radius, zeros, kappa)
                _, kinetic_Q = equation.radial_action(P[b], zeros, dP[b], zeros,
                                                     radius, zeros, kappa)
                hamiltonian[a+b] += -2*C_SQ*(Q[a]*w) @ Q[b].T
                hamiltonian[a+b+1] += ((P[a]*w) @ (hP-Z*P[b]/r).T
                    + (Q[a]*w) @ (kinetic_Q-Z*Q[b]/r).T)
        # Equilibrate once using the real-angle positive metric. This is
        # a congruence (not a similarity) and uses transpose at every angle.
        metric = overlap.sum(axis=0)
        diagonal_scale = 1 / np.sqrt(np.diag(metric))
        balanced_metric = metric * diagonal_scale[:, None] * diagonal_scale[None, :]
        L = linalg.cholesky(balanced_metric, lower=True)
        transform = diagonal_scale[:, None] * linalg.solve_triangular(L.T, np.eye(dimension))
        for bank in (overlap, hamiltonian):
            for i in range(len(bank)):
                defect = np.linalg.norm(bank[i]-bank[i].T) / max(1., np.linalg.norm(bank[i]))
                if defect > 1e-10:
                    raise ValueError(f"DKB quadrature violates complex symmetry: {defect:g}")
                bank[i] = (bank[i] + bank[i].T) / 2
                bank[i] = transform.T @ bank[i] @ transform
                bank[i] = (bank[i] + bank[i].T) / 2
            bank.setflags(write=False)
        self.kappa, self.Z, self.ks = int(kappa), float(Z), int(ks)
        self.dimension = dimension
        self._overlap, self._hamiltonian = overlap, hamiltonian
        self._transform = transform
        self._transform.setflags(write=False)
        self._splines = splines
        self._positive_start, self._negative_start = positive_start, negative_start
        self._token = object()
        self.gr, self.grw = gr, grw
        self.gr.setflags(write=False)
        self.grw.setflags(write=False)
        self.knots = t
        self.knots.setflags(write=False)

    def matrices(self, theta):
        """Return fresh H(theta), S(theta), complex symmetric c-product matrices."""
        theta = _angle(theta)
        z = np.exp(-1j*theta)
        return (np.polynomial.polynomial.polyval(z, self._hamiltonian),
                np.polynomial.polynomial.polyval(z, self._overlap))

    def angle_derivatives(self, theta):
        """Analytic dH/dtheta, dS/dtheta for pole-stationarity diagnostics.

        A small derivative at one box size/angle is not resonance evidence;
        a physical pole also needs continuum/root and radial-basis stability.
        """
        z = np.exp(-1j*_angle(theta))
        return tuple(-1j*z*np.polynomial.polynomial.polyval(
            z, np.polynomial.polynomial.polyder(bank))
            for bank in (self._hamiltonian, self._overlap))

    def solve(self, theta):
        """Solve all roots without dropping the negative-energy sector.

        Near self-orthogonal eigenvectors/exceptional points reject. There
        is deliberately no ``bound_energies`` or ``widths`` classifier.
        """
        theta = _angle(theta)
        H, S = self.matrices(theta)
        energies, vectors = linalg.eig(H, S, check_finite=True)
        if not np.isfinite(energies).all():
            raise ValueError("complex DKB generalized spectrum contains nonfinite roots")
        order = np.argsort(energies.real)
        energies, vectors = energies[order], vectors[:, order]
        norms = np.einsum('ij,ij->j', vectors, S @ vectors)
        norm_scale = np.linalg.norm(S) * np.sum(abs(vectors)**2, axis=0)
        if np.any(abs(norms) <= 1e-12*norm_scale):
            raise ValueError("c-product self-orthogonal root: pole tracking requires special treatment")
        vectors = vectors / np.sqrt(norms)[None, :]
        residuals = np.linalg.norm(H @ vectors - (S @ vectors)*energies, axis=0)
        residuals /= ((np.linalg.norm(H)+abs(energies)*np.linalg.norm(S))
                      * np.linalg.norm(vectors, axis=0))
        dH, dS = self.angle_derivatives(theta)
        slopes = np.einsum('ij,ij->j', vectors,
                          dH @ vectors - (dS @ vectors)*energies)
        for array in (energies, vectors, residuals, slopes):
            array.setflags(write=False)
        return ComplexDiracSpectrum(theta, energies, vectors, residuals, slopes, self._token)

    def sample_spinors(self, spectrum, indices, *, derivative=0):
        """Owned-spectrum radial P/Q on this workspace's exact Gauss grid.

        Selection is explicit: this method neither infers a no-pair
        projector nor relabels continuum box roots as physical n states.
        Rows follow ``indices``; no conjugation occurs in DKB evaluation.
        ``derivative=1`` returns analytic dP/dr,dQ/dr with respect to the
        real integration coordinate. Operator dilation factors stay with
        the consumer, avoiding a second rotation of these derivatives.
        """
        if not isinstance(spectrum, ComplexDiracSpectrum) or spectrum._owner is not self._token:
            raise ValueError("spectrum belongs to a different DKB workspace")
        if isinstance(derivative, bool) or derivative not in (0, 1):
            raise ValueError("spinor derivative must be 0 or 1")
        indices = np.asarray(indices)
        if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices)
                or len(np.unique(indices)) != len(indices) or np.any(indices < 0)
                or np.any(indices >= self.dimension)):
            raise ValueError("distinct valid integer spectrum indices required")
        r = self.gr.ravel()
        B, dB = (self._splines(r, nu=d).T for d in range(2))
        Bp, dp = B[self._positive_start:-2], dB[self._positive_start:-2]
        Bn, dn = B[self._negative_start:-2], dB[self._negative_start:-2]
        z = np.exp(-1j*spectrum.theta)
        if derivative == 0:
            P = np.vstack((Bp, z*(dn-self.kappa*Bn/r)/(2*C)))
            Q = np.vstack((z*(dp+self.kappa*Bp/r)/(2*C), Bn))
        else:
            ddB = self._splines(r, nu=2).T
            ddp, ddn = ddB[self._positive_start:-2], ddB[self._negative_start:-2]
            P = np.vstack((dp, z*(ddn-self.kappa*(dn/r-Bn/r**2))/(2*C)))
            Q = np.vstack((z*(ddp+self.kappa*(dp/r-Bp/r**2))/(2*C), dn))
        coefficients = self._transform @ spectrum.coefficients[:, indices]
        p, q = coefficients.T @ P, coefficients.T @ Q
        p.setflags(write=False)
        q.setflags(write=False)
        return p, q


__all__ = ["ComplexScaledDiracWorkspace", "ComplexDiracSpectrum"]
