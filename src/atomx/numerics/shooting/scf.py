"""Numerical SCF sweeps for supplied nonlocal radial fields.

Methods supply a local potential and exchange source for each orbital, the
energy objective, and an occupied-space stationarity measure. This engine
owns root solves, normalization, damping and numerical stopping criteria.
Callbacks act on whole radial arrays; there is no per-grid-point callback.
"""
from dataclasses import dataclass
import math
import numpy as np
from atomx.numerics.shooting.dirac_bound import dbound
from atomx.numerics.shooting.dirac_orbital import (
    solve_inhomogeneous_orbital, FrozenSourceConvergenceError,
)
from atomx.numerics.shooting.quadrature import slag6
from atomx.numerics.shooting.spline import splin0


@dataclass
class SCFIteration:
    converged: bool
    iterations: int
    energy: float
    components: object
    orbital_update_norm: float
    nonlocal_recoveries: int = 0
    nonlocal_residual: float = 0.0
    exchange_continuations: int = 0
    continuation_min_overlap: float = 1.0
    continuation_max_residual: float = 0.0
    stagnation_promotions: int = 0


def validate_scf_controls(max_iter, tolerance, damping):
    """Validate numerical controls before initializing any physical state."""
    if max_iter < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('max_iter must be positive and tol finite and positive')
    if not np.isfinite(damping) or not 0 <= damping < 1:
        raise ValueError('damping must lie in [0, 1)')


def iterate_nonlocal_orbitals(pa, qa, ev, *, ns, kappas, groups, r, difr,
                             initial_energy, max_iter, tolerance, damping,
                             constraint_mode, first_sweep, field, evaluate,
                             stationarity, exchange_action=None, orbital_solver="adaptive"):
    """Update owned iteration buffers and return the numerical SCF outcome.

    Field/evaluation callbacks may update the supplied buffers in place
    (e.g. occupied-space rotations). The caller owns state identity and
    final physical result construction.
    """
    validate_scf_controls(max_iter, tolerance, damping)
    if orbital_solver not in ('adaptive', 'nonlocal_resolvent'):
        raise ValueError('unknown shooting orbital solver')
    if orbital_solver == 'nonlocal_resolvent' and (constraint_mode != 'coupled' or exchange_action is None):
        raise ValueError('nonlocal_resolvent requires coupled mode and an exchange action')
    nsh = len(ns)
    np_ = len(r)
    ind = np.arange(np_, dtype=np.int64)
    tol_e = min(tolerance, 1.0e-7)
    e_total_prev = None
    e_total = initial_energy
    converged = False
    iterations = 0
    nonlocal_recoveries = 0
    nonlocal_residual = 0.0
    exchange_continuations = 0
    continuation_min_overlap = 1.0
    continuation_max_residual = 0.0
    nonlocal_orbitals = ({a for a in range(nsh) if len(groups[kappas[a]]) == 1}
                         if orbital_solver == 'nonlocal_resolvent' else set())
    update_history = [[] for _ in range(nsh)]
    stagnation_promotions = 0

    for it in range(1, max_iter + 1):
        iterations = it
        orbital_update_norm = 0.0
        for a in range(nsh):
            rv_loc, chi_p, chi_q = field(a, pa, qa)

            # --- seed: first sweep uses the homogeneous eigenvalue
            #     plus the first-order exchange shift ---
            e_seed = ev[a]
            if first_sweep and (np.max(np.abs(chi_p))
                                + np.max(np.abs(chi_q))) > 1.0e-12:
                va_, vb_, vc_, vd_ = splin0(r, rv_loc, np_, 0.0, 0.0)
                e_h, ier, p_h, q_h, _ = dbound(
                    float(min(ev[a], -1.0e-4)), 1.0e-11,
                    int(ns[a]), int(kappas[a]),
                    r, rv_loc, va_, vb_, vc_, vd_, np_, r, np_)
                if ier == 0:
                    nh = slag6(1.0, (p_h ** 2 + q_h ** 2) * difr,
                               np_)[np_ - 1]
                    if nh > 1.0e-30:
                        fnh = 1.0 / math.sqrt(nh)
                        shift = slag6(
                            1.0, (p_h * chi_p + q_h * chi_q) * difr,
                            np_)[np_ - 1] * fnh
                        e_seed = e_h - shift

            use_nonlocal = a in nonlocal_orbitals
            if not use_nonlocal:
                try:
                    p_new, q_new, e_new = solve_inhomogeneous_orbital(
                        e_seed, ns[a], kappas[a], rv_loc, chi_p, chi_q,
                        r, ind, difr, np_, constraints=(
                            np.array([[pa[b], qa[b]] for b in groups[kappas[a]] if b != a])
                            if constraint_mode == 'coupled' else ()),
                        stable_tail=constraint_mode == 'coupled',
                        excluded_lower_roots=(sum(ns[b] < ns[a] for b in groups[kappas[a]])
                            if constraint_mode == 'coupled' else 0))
                except FrozenSourceConvergenceError:
                    use_nonlocal = True
            if use_nonlocal:
                # Keep the same local field, but apply exchange to each trial
                # orbital instead of retaining an inconsistent frozen source.
                if (constraint_mode != 'coupled' or exchange_action is None
                        or len(groups[kappas[a]]) != 1):
                    raise FrozenSourceConvergenceError('nonlocal recovery requires an unconstrained orbital')
                from .nonlocal_eigen import solve_nonlocal_bound_state
                from .problem import ShootingRadialProblem
                p_new, q_new, e_new, report = solve_nonlocal_bound_state(
                    e_seed, ns[a], kappas[a], ShootingRadialProblem(r, difr, rv_loc),
                    np.array([pa[a], qa[a]]),
                    lambda trial: exchange_action(a, trial, pa, qa))
                # Retain this formulation: alternating it with frozen-source
                # roots can destabilize an otherwise converged orbital.
                nonlocal_orbitals.add(a)
                nonlocal_recoveries += 1
                exchange_continuations += bool(report.get('exchange_continuation'))
                for step in report.get('exchange_continuation', ()):
                    continuation_min_overlap = min(continuation_min_overlap, step['overlap'])
                    continuation_max_residual = max(continuation_max_residual, step['green_equation_residual'])
                nonlocal_residual = max(nonlocal_residual, report['green_equation_residual'])

            # sign alignment with the previous orbital
            if slag6(1.0, (p_new * pa[a] + q_new * qa[a]) * difr,
                     np_)[np_ - 1] < 0.0:
                p_new = -p_new
                q_new = -q_new

            # Measure the fixed-point defect BEFORE mixing: strong damping
            # must not turn a large orbital correction into convergence.
            update_norm = math.sqrt(max(0.0, float(slag6(
                1.0, ((p_new-pa[a])**2+(q_new-qa[a])**2)*difr,
                np_)[-1])))
            orbital_update_norm = max(orbital_update_norm, update_norm)
            history = update_history[a]
            history.append(update_norm)
            if len(history) > 12:
                del history[0]

            # damped update
            p_mix = damping * pa[a] + (1.0 - damping) * p_new
            q_mix = damping * qa[a] + (1.0 - damping) * q_new

            # Schmidt orthogonalisation against lower same-kappa
            # shells (off-diagonal Lagrange multipliers).
            for b in groups[kappas[a]]:
                if constraint_mode == 'coupled':
                    break
                if ns[b] >= ns[a]:
                    break
                ov = slag6(1.0, (p_mix * pa[b] + q_mix * qa[b]) * difr,
                           np_)[np_ - 1]
                p_mix -= ov * pa[b]
                q_mix -= ov * qa[b]

            nrm = slag6(1.0, (p_mix ** 2 + q_mix ** 2) * difr,
                        np_)[np_ - 1]
            fn = 1.0 / math.sqrt(nrm) if nrm > 1.0e-30 else 1.0
            pa[a] = p_mix * fn
            qa[a] = q_mix * fn
            ev[a] = e_new

        first_sweep = False

        e_total, components = evaluate(pa, qa, ev)

        if e_total_prev is not None and abs(
                e_total - e_total_prev) <= tol_e * max(1.0, abs(e_total)):
            rotation_ok = True
            if constraint_mode == 'coupled':
                defect = stationarity(pa, qa)
                rotation_ok = (defect <= max(1e-8, math.sqrt(tol_e))
                               and orbital_update_norm <= math.sqrt(tol_e))
            if rotation_ok:
                converged = True
                break
            # A frozen exchange source can reach an energy plateau while its
            # orbital fixed-point defect remains finite. Escalate only stalled
            # singleton kappa channels to the same Hamiltonian's implicit
            # exchange solve; preserve all root and convergence acceptance gates.
            if constraint_mode == 'coupled' and exchange_action is not None:
                for a, history in enumerate(update_history):
                    if (a not in nonlocal_orbitals and len(groups[kappas[a]]) == 1
                            and len(history) == 12
                            and min(history) > math.sqrt(tol_e)
                            and min(history) >= .8 * history[0]):
                        nonlocal_orbitals.add(a)
                        stagnation_promotions += 1
        e_total_prev = e_total

    return SCFIteration(converged, iterations, e_total, components, orbital_update_norm,
                        nonlocal_recoveries, nonlocal_residual, exchange_continuations,
                        continuation_min_overlap, continuation_max_residual,
                        stagnation_promotions)
