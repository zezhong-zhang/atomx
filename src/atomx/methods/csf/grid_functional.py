"""Full correlated CI energy and its exact discrete shooting-grid derivative.

This supplies the physical residual for a future grid orbital optimizer.
It is not safe to minimize an unrestricted Dirac grid functional: the
electron branch/no-pair constraint must be supplied by the orbital solver.
No B-spline coefficients, diagonalization, or density-only SCF weights enter
the radial derivatives here. All configuration interference is retained.
"""
from __future__ import annotations

import numpy as np

from atomx.methods.ci import ci_solve
from atomx.numerics.shooting.constants import SL
from atomx.numerics.shooting.coulomb import _yk_over_r
from atomx.numerics.shooting.operators import _FD_CENTRAL, _FD_EDGE
from atomx.numerics.shooting.quadrature import slag6_adjoint, slag6_weights
from atomx.physics.csf import matrix
from atomx.physics.csf.radial_density import CIRadialDensity


def derivative_adjoint(source):
    """Exact transpose of matrix._ds_derivative, including edge stencils."""
    n = len(source)
    if n < 9:
        raise ValueError('grid needs at least 9 points')
    out = np.zeros_like(source)
    for offset, coeff in enumerate(_FD_CENTRAL, 1):
        out[4+offset:n-4+offset] += coeff*source[4:n-4]
        out[4-offset:n-4-offset] -= coeff*source[4:n-4]
    for i in range(4):
        out[:9] += source[i]*_FD_EDGE[i]
        out[-9:] -= source[n-1-i]*_FD_EDGE[i][::-1]
    return out


def coulomb_adjoint(source, rank, workspace):
    """Transpose of the *implemented* forward/backward radial Coulomb map."""
    r, difr, n = workspace.r, workspace.difr, workspace.n
    low = r**rank
    high = workspace.inv_r**(rank+1)
    active = source.copy()
    active[r <= 1e-30] = 0.
    forward = low*difr*slag6_adjoint(1., active*high, n)
    backward = high*difr*slag6_adjoint(1., (active*low)[::-1].copy(), n)[::-1]
    # The original backwards integrand explicitly drops index zero even
    # on a positive-origin grid. Match that operation, not its continuum
    # approximation. Standard SGRID starts at exactly zero.
    backward[0] = 0.
    return forward+backward


class GridCSFFunctional:
    """Ground-state Dirac–Coulomb CI functional on tabulated radial spinors.

    ``evaluate`` returns coordinate derivatives dE/dP_i,dE/dQ_i. Divide
    by radial quadrature weights to obtain the L2 Riesz gradient. The
    discrete adjoints make directional derivatives agree with the exact
    grid CI energy, rather than merely approach it on grid refinement.
    Breit/QED/recoil are not included in this orbital functional.
    """

    def __init__(self, csfs, workspace, Z, *, max_determinants=None):
        if not isinstance(workspace, matrix.GridCIWorkspace):
            raise TypeError('GridCIWorkspace is required')
        if workspace.n < 9 or np.any(np.diff(workspace.r) <= 0):
            raise ValueError('shooting grid must have at least 9 increasing points')
        self.csfs, self.data, self.Z = csfs, workspace, Z
        self.ns = workspace.n
        self.keys = tuple(sorted({(s.n, s.kappa) for c in csfs for s in c.subshells}))
        self.radial_density = CIRadialDensity(csfs, max_determinants=max_determinants)
        self.max_determinants = self.radial_density.max_determinants
        self.index_weights = slag6_weights(workspace.n)
        self.weights = self.index_weights*workspace.difr
        if np.any(self.weights <= 0) or not np.all(np.isfinite(self.weights)):
            raise ValueError('radial quadrature metric must be finite and positive')
        self.last_occupations = {}

    def one_body_action(self, pair, kappa):
        """Symmetric discrete h_D times a spinor (quadrature included)."""
        p, q = pair
        w, ws, s = self.weights, self.data, self.index_weights
        potential = ws.rv_nuc*ws.inv_r
        coupling = SL*kappa*ws.inv_r*w
        hp = (w*potential*p + coupling*q + .5*SL*(
            derivative_adjoint(s*q) - s*matrix._ds_derivative(q)))
        hq = (w*(potential-2*SL**2)*q + coupling*p + .5*SL*(
            s*matrix._ds_derivative(p) - derivative_adjoint(s*p)))
        return np.array([hp, hq])

    def evaluate(self, orbitals, *, gradient=True, ci_result=None):
        result = ci_result
        if result is None:
            result = ci_solve(self.csfs, orbitals, self.keys, self.data, self.Z, self.ns,
                              max_determinants=self.max_determinants)
        energy = float(result.ground_energy)
        if not gradient:
            return energy, result
        weights = self.radial_density.contract(result)
        self.last_weights = weights
        self.last_occupations = {key: max(0., weights.get(('h', key[0], key[1], key[0]), 0.))
                                 for key in self.keys}
        derivative = {key: np.zeros((2, self.ns)) for key in self.keys}
        spinors = {key: np.asarray(orbitals[key]) for key in self.keys}
        h_action = {}
        densities, forward, backward = {}, {}, {}

        def density(a, b):
            pair = tuple(sorted((a, b)))
            if pair not in densities:
                densities[pair] = np.sum(spinors[a]*spinors[b], axis=0)
            return pair, densities[pair]

        for key, weight in weights.items():
            if key[0] == 'h':
                _, na, kappa, nb = key
                a, b = (na, kappa), (nb, kappa)
                for source in (a, b):
                    if source not in h_action:
                        h_action[source] = self.one_body_action(spinors[source], kappa)
                # The CI radial evaluator uses h_ab / <b|b>. Its normal
                # derivative cancels after exact tangent projection, but
                # retain it here so this is also the exact coordinate
                # derivative of that implemented integral (and avoids
                # carrying a large nuclear-energy normal into projection).
                norm_b = float(np.sum(spinors[b]**2*self.weights))
                h_ab = float(np.sum(spinors[a]*h_action[b]))
                derivative[a] += weight*h_action[b]/norm_b
                derivative[b] += weight*h_action[a]/norm_b
                derivative[b] -= 2*weight*h_ab/norm_b**2*self.weights*spinors[b]
            else:
                _, rank, a, b, c, d = key
                ab, rho_ab = density(a, b)
                cd, rho_cd = density(c, d)
                fa, bc = (rank, ab), (rank, cd)
                if fa not in forward:
                    forward[fa] = self.weights*_yk_over_r(
                        rho_ab, rank, self.data.r, self.data.difr, self.ns)
                if bc not in backward:
                    backward[bc] = coulomb_adjoint(self.weights*rho_cd, rank, self.data)
                derivative[a] += weight*backward[bc]*spinors[b]
                derivative[b] += weight*backward[bc]*spinors[a]
                derivative[c] += weight*forward[fa]*spinors[d]
                derivative[d] += weight*forward[fa]*spinors[c]
        self.last_gradient = derivative
        return energy, derivative, result

    def constrained_residual(self, orbitals, derivative):
        """L2 tangent gradient and same-kappa Lagrange multiplier matrices.

        Input orbitals must already be metric-orthonormal. This projection
        includes off-diagonal constraints; it is not post-hoc Schmidt
        orthogonalization of independently solved orbital equations.
        It does not enforce the separate electron/no-pair constraint.
        """
        residual, multipliers = {}, {}
        for kappa in sorted({k[1] for k in self.keys}):
            keys = [k for k in self.keys if k[1] == kappa]
            x = np.stack([orbitals[k] for k in keys], axis=-1)
            g = np.stack([derivative[k] for k in keys], axis=-1)/self.weights[None, :, None]
            gram = np.einsum('ari,r,arj->ij', x, self.weights, x)
            if not np.allclose(gram, np.eye(len(keys)), rtol=0., atol=1e-7):
                raise ValueError('constraint residual requires orthonormal orbitals')
            cross = np.einsum('ari,r,arj->ij', x, self.weights, g)
            multiplier = .5*(cross+cross.T)
            tangent = g-np.einsum('ari,ij->arj', x, multiplier)
            multipliers[kappa] = multiplier
            for i, key in enumerate(keys):
                residual[key] = tangent[:, :, i]
        return residual, multipliers

    def retract(self, orbitals):
        """Symmetric polar orthonormalization in the actual CI grid metric."""
        out = {}
        for kappa in sorted({k[1] for k in self.keys}):
            keys = [k for k in self.keys if k[1] == kappa]
            x = np.stack([orbitals[k] for k in keys], axis=-1)
            if x.shape != (2, self.ns, len(keys)) or not np.all(np.isfinite(x)):
                raise ValueError('invalid grid spinors')
            gram = np.einsum('ari,r,arj->ij', x, self.weights, x)
            values, vectors = np.linalg.eigh(gram)
            if values.min() <= 1e-14:
                raise ValueError('rank-deficient grid orbital trial')
            x = np.einsum('ari,ij->arj', x, (vectors/np.sqrt(values))@vectors.T)
            for i, key in enumerate(keys):
                out[key] = x[:, :, i].copy()
        return out


def occupied_1s2_reference_potential(functional, orbital):
    """Local reference for the occupied radial equation of a 1s² HF state.

    For its occupied 1s spinor, direct 2J minus exchange J gives V_nuc+J.
    This identity does NOT replace the nonlocal Fock operator on arbitrary
    virtual orbitals, nor extend to general open shells. The supplied
    normalized spinor must come from the intended HF reference; this
    function does not establish its self-consistency. The resulting local
    operator defines an explicit fixed electron projector for subsequent CI.
    The physical Hamiltonian remains the original nuclear plus Coulomb one.
    """
    from atomx.numerics.shooting.coulomb import _yk_over_r
    if not functional.csfs or any(c.n_electrons != 2 for c in functional.csfs):
        raise ValueError('the 1s2 reference requires a two-electron problem')
    phi = np.asarray(orbital, dtype=float)
    if phi.shape != (2, functional.ns) or not np.all(np.isfinite(phi)):
        raise ValueError('invalid reference 1s spinor')
    density = np.sum(phi*phi, axis=0)
    if abs(float(density@functional.weights)-1.) > 1e-7:
        raise ValueError('reference 1s spinor must be normalized')
    ws = functional.data
    return ws.rv_nuc + ws.r*_yk_over_r(density, 0, ws.r, ws.difr, ws.n)
