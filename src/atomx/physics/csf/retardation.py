"""Static Breit retardation from transition-current divergences.

R = -1/2 (alpha_1.grad_1)(alpha_2.grad_2) r12. Integrating by parts
on both electrons gives -1/2 int div(j_ac) div(j_bd) r12. With real
radial spinors div(j_ac) = +/- i D_ac/r^2 Omega_a^dagger Omega_c,
D_ac = (Q_a P_c-P_a Q_c)' + (kappa_c-kappa_a)(Q_a P_c+P_a Q_c)/r.
The common phase sign cancels between the two electrons. Therefore the
radial bilinear has a positive 1/2 angular prefactor. No local-potential
or orbital-energy substitution is made: this also holds for nonlocal HF.

The Racah-normalized multipole of r12 is
r_<^(L+2) r_>^(-L-1)/(2L+3) - r_<^L r_>^(1-L)/(2L-1).
Both terms have linear-cost semiseparable radial contractions.
"""

from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline

from atomx.numerics.yk import yk_partial_matrices
from atomx.physics.csf.breit_ci import _clm, _spinor_coeffs, twoj_of_kappa


@lru_cache(maxsize=None)
def scalar_spinor_element(ka, ma, kc, mc, L, q2):
    return sum(
        a * c * _clm(la, mla, L, q2, lc, mlc)
        for a, la, mla, msa in _spinor_coeffs(ka, ma)
        for c, lc, mlc, msc in _spinor_coeffs(kc, mc)
        if msa == msc
    )


def retardation_terms(ia, ib, ic, id_, coeff, spinor_labels):
    na, ka, ma = spinor_labels[ia]
    nb, kb, mb = spinor_labels[ib]
    nc, kc, mc = spinor_labels[ic]
    nd, kd, md = spinor_labels[id_]
    if ma + mb != mc + md:
        return ()
    # A diagonal transition current is divergence-free identically.
    if (na, ka) == (nc, kc) or (nb, kb) == (nd, kd):
        return ()
    q2 = ma - mc
    terms = []
    for L in range(
        min(
            (twoj_of_kappa(ka) + twoj_of_kappa(kc)) // 2,
            (twoj_of_kappa(kb) + twoj_of_kappa(kd)) // 2,
        )
        + 1
    ):
        if abs(q2) > 2 * L:
            continue
        angular = (
            0.5
            * (-1.0) ** (q2 // 2)
            * scalar_spinor_element(ka, ma, kc, mc, L, q2)
            * scalar_spinor_element(kb, mb, kd, md, L, -q2)
        )
        if angular:
            terms.append(
                (("t", L, (na, ka), (nc, kc), (nb, kb), (nd, kd)), coeff * angular)
            )
    return tuple(terms)


def distance_potential(rho, r, weights, L):
    """Integral of the L multipole of |r1-r2|, on Gauss cells or raw grid."""

    def channel(a, b):
        f = weights * rho * r**a
        g = weights * rho * r**b
        if r.ndim == 2:
            mo, mi = yk_partial_matrices(r.shape[1])
            ft = f.sum(axis=1)
            gt = g.sum(axis=1)
            outward = (np.cumsum(ft) - ft)[:, None] + f @ mo.T
            inward = (np.cumsum(gt[::-1])[::-1] - gt)[:, None] + g @ mi.T
        else:
            # The grid path uses a continuous cubic interpolant with exact
            # antiderivatives, rather than differentiating noisy differences.
            fa = CubicSpline(r, rho * r**a).antiderivative()
            ga = CubicSpline(r, rho * r**b).antiderivative()
            outward = fa(r) - fa(r[0])
            inward = ga(r[-1]) - ga(r)
        return r**b * outward + r**a * inward

    return channel(L + 2, -L - 1) / (2 * L + 3) - channel(L, 1 - L) / (2 * L - 1)


def transition_current_fields(orbitals, workspace):
    """Shared Gauss cells, spinors and exact transition-current divergences.

    Return ``r, weights, wave(key), divergence(a, c)``. B-spline spinors
    use analytic derivatives; shooting spinors use the same cubic
    interpolant for both values and derivatives. Caches belong to this
    orbital set and are never shared between solves.
    """
    from atomx.numerics.bspline.evaluate import eval_orbital_on_grid
    from atomx.physics.csf.matrix import GridCIWorkspace

    grid = isinstance(workspace, GridCIWorkspace)
    if grid:
        # Interpolate each native interval and integrate on interior Gauss
        # nodes. This excludes the singular origin without dropping its
        # interval, and differentiates the same cubic orbital interpolant.
        x, w = np.polynomial.legendre.leggauss(8)
        half = 0.5 * np.diff(workspace.r)[:, None]
        r = workspace.r[:-1, None] + half * (1 + x)
        weights = half * w
    else:
        r = workspace["gr"]
        weights = workspace["grw"]
    waves = {}
    densities = {}

    def wave(key):
        if key not in waves:
            P, Q = orbitals[key][:2]
            if grid:
                sp = CubicSpline(workspace.r, P)
                sq = CubicSpline(workspace.r, Q)
                waves[key] = (sp(r), sq(r), sp(r, 1), sq(r, 1))
            else:

                def evaluate(c, comp, derivative=False):
                    name = "bsp_" + comp + ("_d" if derivative else "")
                    return eval_orbital_on_grid(
                        c,
                        workspace[name],
                        workspace["nv"],
                        workspace.get("ks" + comp, workspace["ks"]),
                        workspace.get("ns" + comp, workspace["ns"]),
                    )

                waves[key] = (
                    evaluate(P, "p"),
                    evaluate(Q, "q"),
                    evaluate(P, "p", True),
                    evaluate(Q, "q", True),
                )
        return waves[key]

    def density(a, c):
        if (a, c) not in densities:
            if a == c:
                densities[a, c] = np.zeros_like(r)
            elif (c, a) in densities:
                densities[a, c] = -densities[c, a]
            else:
                pa, qa, dpa, dqa = wave(a)
                pc, qc, dpc, dqc = wave(c)
                densities[a, c] = (
                    dqa * pc
                    + qa * dpc
                    - dpa * qc
                    - pa * dqc
                    + (c[1] - a[1]) * (qa * pc + pa * qc) / r
                )
        return densities[a, c]

    return r, weights, wave, density


def radial_evaluator(orbitals, workspace):
    """Create a solve-local evaluator with analytic B-spline derivatives."""
    r, weights, _, density = transition_current_fields(orbitals, workspace)
    potentials = {}
    values = {}

    def potential(L, a, c):
        if (L, a, c) not in potentials:
            potentials[L, a, c] = distance_potential(density(a, c), r, weights, L)
        return potentials[L, a, c]

    def evaluate(key):
        _, L, a, c, b, d = key
        if key not in values:
            # Symmetrize quadrature roundoff; not the physical operator.
            values[key] = float(
                0.5
                * np.sum(
                    weights
                    * (
                        density(a, c) * potential(L, b, d)
                        + density(b, d) * potential(L, a, c)
                    )
                )
            )
        return values[key]

    return evaluate
