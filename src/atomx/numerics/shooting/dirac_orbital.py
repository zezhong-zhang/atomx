"""Bracketed normalization corrector for a sourced Dirac orbital.

The caller supplies the local potential and nonlocal source. This module
owns the eigenvalue bracket, normalization Newton step and wrong-root guard;
occupations, angular coefficients and SCF mixing belong to the method.
"""
import math

import numpy as np

from atomx.angular.recoupling import l_of_kappa

from .constants import SL
from .dirac_bound import dbound
from .dirac_green import green_solve as _green_solve
from .dirac_green import homogeneous_pair as _homogeneous_pair
from .dirac_green import prepare_scaled_green, scaled_box_pair, scaled_green_solve
from .quadrature import slag6, slag6_weights
from .spline import splin0


def _count_nodes(p: np.ndarray, ncut: int) -> int:
    """Sign changes of P on (0, r_cut), counting only O(1) lobes.

    Used as a wrong-root backstop for the inhomogeneous solve: landing
    on the root of a neighbouring pole produces a solution dominated by
    the wrong eigenfunction, i.e. full-amplitude lobes.  The threshold
    must stay LOOSE (1e-2 of the peak): exact HF orbitals are only
    nodeless/nr-noded in their dominant region — the non-local exchange
    source adds ~1e-4-amplitude sign wiggles in the far tail (canonical
    same-kappa HF orbitals are mutually orthogonal), which a tight
    threshold would miscount as extra nodes (seen for Ar 1s).
    """
    amax = np.max(np.abs(p[:ncut]))
    if amax <= 0.0:
        return 0
    thresh = 1.0e-2 * amax
    nodes = 0
    last_sign = 0
    for i in range(ncut):
        v = p[i]
        if abs(v) < thresh:
            continue
        s = 1 if v > 0.0 else -1
        if last_sign != 0 and s != last_sign:
            nodes += 1
        last_sign = s
    return nodes


class OrbitalSolveError(RuntimeError):
    pass


class RadialBoxError(OrbitalSolveError):
    """A requested bound state does not fit in the radial box."""


class FrozenSourceConvergenceError(OrbitalSolveError):
    """The frozen-source solve cannot resolve the requested normalized root."""


def _resolve_weak_source_pole(e0, n, kappa, r, difr, rv, potential, chi_p, chi_q):
    """Separate a nearly singular local bound-state pole from its response.

    Write psi=a*phi+u with u orthogonal to the normalized local eigenstate phi.
    Then e=epsilon-<phi,chi>/a and (H-e)u=Q chi. Evaluate the latter through
    an off-pole constrained inverse and its resolvent iteration. No exchange
    source is dropped, and the full radial box remains in the response.
    """
    from .dirac_response import DiracResponse
    count = len(r)
    epsilon, error, p, q, _ = dbound(float(e0), 1e-11, int(n), int(kappa),
        r, rv, *potential, count, r, count)
    if error:
        raise OrbitalSolveError(f'local pole reference failed: n={n} kappa={kappa}, ier={error}')
    weights = slag6_weights(count)*difr
    phi = np.array([p, q])
    phi /= np.sqrt(np.sum(phi*phi*weights))
    chi = np.array([chi_p, chi_q])
    overlap = float(np.sum(phi*chi*weights))
    if overlap < 0:
        phi *= -1
        overlap *= -1
    # Small compared with a typical adjacent radial gap, but well above the
    # floating-point uncertainty of the local pole. Iteration verifies use.
    shift = max(abs(epsilon)*1e-3, 1e-8)
    reference = epsilon-shift
    response = DiracResponse(reference, int(kappa), r, difr, rv, boundary='box')
    u = np.zeros_like(phi)
    energy = epsilon-overlap
    for _ in range(80):
        updated, _, report = response.solve(chi+(energy-reference)*u, [phi])
        norm = float(np.sum(updated*updated*weights))
        if not np.isfinite(norm) or norm >= .25:
            raise FrozenSourceConvergenceError('local-pole separation is not in the weak-source regime')
        amplitude = np.sqrt(1.-norm)
        next_energy = epsilon-overlap/amplitude
        change = float(np.sqrt(np.sum((updated-u)**2*weights)))
        u = updated
        if (change < 1e-11 and abs(next_energy-energy) < 1e-12*max(1.,abs(energy))
                and report['constraint_residual'] < 1e-11):
            psi = amplitude*phi+u
            return psi[0], psi[1], float(next_energy)
        energy = next_energy
    raise OrbitalSolveError('projected local-pole response did not converge')


def solve_inhomogeneous_orbital(e0, n, kappa, rv_loc, chi_p, chi_q,
                      r, ind, difr, np_,
                      tol_e=1.0e-10, max_newton=60, *, constraints=(), stable_tail=True,
                      excluded_lower_roots=0):
    """Solve one inhomogeneous DHF orbital equation for ``(P, Q, eps)``.

    Newton iteration on the normalisation ``N(eps) = 1`` with a Sturm
    node-count bracket.  Falls back to the plain homogeneous eigenvalue
    solver (:func:`dbound`) when the exchange source vanishes
    (single-shell systems like He, hydrogen-like ions).
    """
    l = l_of_kappa(kappa)
    nr = n - l - 1  # required radial node count

    va, vb, vc, vd = splin0(r, rv_loc, np_, 0.0, 0.0)
    constraints = np.asarray(constraints, dtype=float)
    constrained = constraints.size > 0
    if constrained and (constraints.ndim != 3 or constraints.shape[1:] != (2, np_)
                        or not np.all(np.isfinite(constraints))):
        raise ValueError('orbital constraints must be finite (count, 2, grid) spinors')
    if (not isinstance(excluded_lower_roots, (int, np.integer))
            or excluded_lower_roots < 0 or excluded_lower_roots > nr
            or (excluded_lower_roots and not constrained)):
        raise ValueError('invalid excluded lower-root count')
    metric = slag6_weights(np_)*difr if constrained else None

    chi_scale = float(np.max(np.abs(chi_p)) + np.max(np.abs(chi_q)))
    if chi_scale < 1.0e-12 and not constrained:
        # No exchange source: ordinary local eigenproblem.
        e_val, ier, p_sol, q_sol, _ = dbound(
            float(min(e0, -1.0e-4)), 1.0e-11, int(n), int(kappa),
            r, rv_loc, va, vb, vc, vd, np_, r, np_)
        if ier != 0:
            raise OrbitalSolveError(
                f"dbound failed for n={n} kappa={kappa}: ier={ier}")
        norm = slag6(1.0, (p_sol ** 2 + q_sol ** 2) * difr, np_)[np_ - 1]
        fn = 1.0 / math.sqrt(norm) if norm > 1.0e-30 else 1.0
        return p_sol * fn, q_sol * fn, float(e_val)

    e = float(min(e0, -1.0e-4))
    e_lo = -SL * SL          # deep floor (as in dbound)
    e_hi = -1.0e-9           # bound states only
    p = q = None
    ncut = np_
    converged = False

    for _ in range(max_newton):
        if stable_tail:
            pair = scaled_box_pair(e, kappa, r, va, vb, vc, vd)
            quadrature = prepare_scaled_green(pair)
            nzero, ncut = pair[4], np_
            def green(cp, cq, *, pair=pair, quadrature=quadrature):
                return scaled_green_solve(cp, cq, pair, difr, quadrature)
        else:
            (p_reg, q_reg, p_irr, q_irr, w_const,
             ncut, iotp, nzero) = _homogeneous_pair(
                e, kappa, r, ind, np_, va, vb, vc, vd)
            def green(cp, cq, *, ncut=ncut, p_irr=p_irr, p_reg=p_reg, q_irr=q_irr, q_reg=q_reg, w_const=w_const):
                return _green_solve(cp, cq, p_reg, q_reg, p_irr, q_irr,
                                    w_const, difr, ncut, stable_tail=False)

        try:
            p, q = green(chi_p, chi_q)
        except FloatingPointError:
            if constrained or not stable_tail:
                raise
            return _resolve_weak_source_pole(e, n, kappa, r, difr, rv_loc,
                (va,vb,vc,vd), chi_p, chi_q)
        if constrained:
            c = constraints[:, :, :ncut]
            responses = np.array([green(*v) for v in constraints])
            border = np.einsum('iar,r,jar->ij', c, metric[:ncut], responses)
            condition = np.linalg.cond(border)
            if not np.isfinite(condition) or condition > 1e12:
                raise OrbitalSolveError('singular same-kappa constraint border')
            def enforce(pair, *, border=border, c=c, ncut=ncut, responses=responses):
                overlap = np.einsum('iar,r,ar->i', c, metric[:ncut], pair)
                multipliers = np.linalg.solve(border, -overlap)
                return pair+np.einsum('i,iar->ar', multipliers, responses)
            p, q = enforce(np.array([p, q]))
        # Inertia of a constrained self-adjoint operator:
        # n_-(H-e restricted to C^perp) = n_-(H-e) - n_-(C^T G C).
        # Orthogonality shifts the poles; a bare Sturm count would reject
        # valid roots between an unconstrained pole and its shifted pole.
        if constrained:
            nzero -= int(np.sum(np.linalg.eigvalsh((border+border.T)*.5) < 0.))
        target = nr-excluded_lower_roots
        if nzero > target:
            e_hi = min(e_hi, e)
            e = 0.5 * (e_lo + e) if e_lo > -SL * SL + 1.0 else 1.5 * e
            e = max(e, -SL * SL)
            continue
        if nzero < target:
            e_lo = max(e_lo, e)
            e = 0.5 * (e + e_hi)
            continue

        nrm = slag6(1.0, (p * p + q * q) * difr[:ncut], ncut)[ncut - 1]

        # dN/de via one extra Green solve with the solution as source.
        pd, qd = green(
            np.concatenate((p, np.zeros(np_ - ncut))),
            np.concatenate((q, np.zeros(np_ - ncut))))
        if constrained:
            pd, qd = enforce(np.array([pd, qd]))
        dn = 2.0 * slag6(1.0, (p * pd + q * qd) * difr[:ncut],
                         ncut)[ncut - 1]

        if dn > 0.0:
            # N(e)=sum_j |<j|chi>|²/(epsilon_j-e)² is convex between
            # poles. On its increasing branch, N<1 is a lower root bound
            # and N>1 an upper bound. Retain these bounds: otherwise a
            # slight pole overshoot discards all progress and jumps back
            # to 1.5*e, repeatedly exhausting Newton for weak exchange.
            if nrm < 1.0:
                e_lo = max(e_lo, e)
            elif nrm > 1.0:
                e_hi = min(e_hi, e)
            de = (1.0 - nrm) / dn
        else:
            # Within this pole interval N is convex. A negative slope
            # places us left of its minimum, hence left of the desired
            # increasing-branch root regardless of N-1. Moving left when
            # N>1 instead converges toward the preceding pole's divergence.
            e_lo = max(e_lo, e)
            de = 0.1 * abs(e)

        # Bounded, bracketed step.
        max_step = 0.5 * abs(e)
        de = max(-max_step, min(max_step, de))
        e_new = e + de
        if e_new >= e_hi:
            e_new = 0.5 * (e + e_hi)
        if e_new <= e_lo:
            e_new = 0.5 * (e + e_lo)

        if abs(e_new - e) <= tol_e * abs(e):
            if stable_tail and not constrained and abs(nrm-1.) > 1e-5:
                # A tiny energy step is not normalization convergence. Near
                # a weakly driven pole the required step can be below one ulp.
                return _resolve_weak_source_pole(e, n, kappa, r, difr, rv_loc,
                    (va,vb,vc,vd), chi_p, chi_q)
            e = e_new
            converged = True
            break
        e = e_new

    if p is None:
        raise OrbitalSolveError(
            f"orbital n={n} kappa={kappa}: node bracket never reached the "
            f"target window (nr={nr})")
    if not converged:
        raise OrbitalSolveError(
            f"orbital n={n} kappa={kappa}: eps Newton did not converge "
            f"(last e={e:.8e})")

    # Wrong-root guard: the accepted solution must carry the target
    # radial node count.
    # Nonlocal constraint sources can introduce tail nodes; Sturm
    # node identities apply only to the unconstrained local problem.
    if not constrained and _count_nodes(p, ncut) != nr:
        raise OrbitalSolveError(
            f"orbital n={n} kappa={kappa}: converged onto a root with "
            f"{_count_nodes(p, ncut)} nodes (want {nr}) at e={e:.8e}")

    nrm = slag6(1.0, (p * p + q * q) * difr[:ncut], ncut)[ncut - 1]
    fn = 1.0 / math.sqrt(nrm) if nrm > 1.0e-30 else 1.0
    p_full = np.zeros(np_)
    q_full = np.zeros(np_)
    p_full[:ncut] = p * fn
    q_full[:ncut] = q * fn
    return p_full, q_full, float(e)
