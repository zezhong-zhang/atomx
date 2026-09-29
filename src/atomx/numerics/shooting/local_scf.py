"""Compiled local-potential SCF iteration on a supplied radial grid.

The method supplies the initial density, nuclear/environment field and a
compiled whole-array field callback. This module owns potential mixing,
orbital solves, normalization and convergence; it chooses no exchange model
or nuclear profile and assembles no physical total-energy functional.

The compiled callback is part of this dispatcher's specialization identity.
Callbacks must have stable module ownership (for example ``physics.slater``);
moving one requires rebuilding this module's Numba cache as well.
"""
import math
import numba as nb
import numpy as np
from .dirac_bound import dbound
from .schrodinger_bound import sbound
from .quadrature import slag6
from .spline import splin0


@nb.njit(cache=True)
def iterate_local_orbitals(r, difr, vnuc, rden, shell_n, shell_kappa,
                           shell_occ, shell_l, equation, max_iter, epsc,
                           single_electron, field, field_args):
    """Return orbital buffers and numerical outcome; density is a work buffer.

    ``field(r, difr, density, vnuc, field_args)`` returns the trial rV,
    direct and exchange arrays. Callbacks stay compiled across the SCF loop.
    """
    np_ = len(r)
    nshell = len(shell_n)
    eps = 1.0e-11
    # Spline work arrays
    rg = r.copy()
    rvg = np.zeros(np_)
    va = np.zeros(np_)
    vb_arr = np.zeros(np_)
    vc = np.zeros(np_)
    vd = np.zeros(np_)

    # Orbital storage
    pa = np.zeros((nshell, np_))
    qa = np.zeros((nshell, np_))
    ev = np.zeros(nshell)

    # Initial SFIELD (DHFS.f:771)
    rvt, vel, vex = field(r, difr, rden, vnuc, field_args)
    rv = rvt.copy()

    # SCF iteration (DHFS.f:776-846)
    factor = 0.0
    chi = 1.0
    chiant = 1.0e30
    converged = False
    #: Worst (largest) ``dbound``/``sbound`` status code seen.  0 = ok,
    #: 5 = the eigenvalue bracket collapsed, i.e. the shooting box is too
    #: small for the requested state and the returned eigenvalue is the
    #: bracketing ceiling rather than a solved state.  DHFS.f discards
    #: this; we surface it so ``converged`` cannot be True on a failure.
    ier_max = 0
    iteration = 0

    for it in range(1, max_iter + 1):
        iteration = it

        # Adaptive mixing (DHFS.f:777-784)
        if chiant > chi:
            factor += 0.025
            if factor > 0.5:
                factor = 0.5
        else:
            factor -= 0.05
            if factor < 0.025:
                factor = 0.025
        faccom = 1.0 - factor
        chiant = chi

        # Mix potential (DHFS.f:790-794)
        for i in range(np_):
            rv[i] = faccom * rv[i] + factor * rvt[i]
            rden[i] = 0.0
            rvg[i] = rv[i]

        # Refit cubic spline (DHFS.f:795) — CRITICAL for DBOUND accuracy
        va, vb_arr, vc, vd = splin0(rg, rvg, np_, 0.0, 0.0)

        # Solve each shell (DHFS.f:799-827)
        for s in range(nshell):
            nnn = shell_n[s]
            kkk = shell_kappa[s]
            e_guess = ev[s] if abs(ev[s]) > 1.0e-15 else -0.1

            if equation == 0:
                # Dirac
                e_val, ier, p_sol, q_sol, ilast = dbound(
                    e_guess, eps, nnn, kkk,
                    rg, rvg, va, vb_arr, vc, vd, np_,
                    r, np_)
                if ier > ier_max:
                    ier_max = ier
            else:
                # Schrödinger
                lll = shell_l[s]
                e_val, ier, p_sol, q_sol, ilast = sbound(
                    e_guess, eps, nnn, lll,
                    rg, rvg, va, vb_arr, vc, vd, np_,
                    r, np_)
                if ier > ier_max:
                    ier_max = ier

            # Renormalize with SLAG6 (DHFS.f:810-818)
            # Schrödinger: norm = ∫P² dr; Dirac: norm = ∫(P²+Q²) dr
            f1 = np.zeros(np_)
            for i in range(np_):
                if equation == 0:
                    f1[i] = (p_sol[i] ** 2 + q_sol[i] ** 2) * difr[i]
                else:
                    f1[i] = p_sol[i] ** 2 * difr[i]
            f1_int = slag6(1.0, f1, np_)
            fnorm_val = 1.0 / math.sqrt(f1_int[np_ - 1]) if f1_int[np_ - 1] > 1.0e-30 else 1.0

            for i in range(np_):
                pa[s, i] = p_sol[i] * fnorm_val
                qa[s, i] = q_sol[i] * fnorm_val

            ev[s] = e_val

            # Density: Schrödinger uses P² only; Dirac uses P²+Q²
            for i in range(np_):
                if equation == 0:
                    rden[i] += shell_occ[s] * (pa[s, i] ** 2 + qa[s, i] ** 2)
                else:
                    rden[i] += shell_occ[s] * pa[s, i] ** 2

        # New trial potential (DHFS.f:829)
        rvt, vel, vex = field(r, difr, rden, vnuc, field_args)

        # Convergence check (DHFS.f:830-834)
        chi = -1.0e10
        for i in range(np_):
            chip = abs(rv[i] - rvt[i]) / max(abs(rv[i]), 1.0e-18)
            if chip > chi:
                chi = chip

        # Test (DHFS.f:839)
        if chi <= epsc and factor >= 0.25:
            converged = True
            break

        if single_electron:
            # Single-electron: no SCF needed beyond first iteration
            converged = True
            break

    # Final potential mix (DHFS.f:851-856)
    for i in range(np_):
        rv[i] = faccom * rv[i] + factor * rvt[i]
        rvg[i] = rv[i]
        rden[i] = 0.0
    va, vb_arr, vc, vd = splin0(rg, rvg, np_, 0.0, 0.0)

    # Final solve (DHFS.f:862-885)
    for s in range(nshell):
        nnn = shell_n[s]
        kkk = shell_kappa[s]
        e_guess = ev[s]

        if equation == 0:
            e_val, ier, p_sol, q_sol, ilast = dbound(
                e_guess, eps, nnn, kkk,
                rg, rvg, va, vb_arr, vc, vd, np_, r, np_)
        else:
            lll = shell_l[s]
            e_val, ier, p_sol, q_sol, ilast = sbound(
                e_guess, eps, nnn, lll,
                rg, rvg, va, vb_arr, vc, vd, np_, r, np_)
        if ier > ier_max:
            ier_max = ier

        f1 = np.zeros(np_)
        for i in range(np_):
            if equation == 0:
                f1[i] = (p_sol[i] ** 2 + q_sol[i] ** 2) * difr[i]
            else:
                f1[i] = p_sol[i] ** 2 * difr[i]
        f1_int = slag6(1.0, f1, np_)
        fnorm_val = 1.0 / math.sqrt(f1_int[np_ - 1]) if f1_int[np_ - 1] > 1.0e-30 else 1.0

        for i in range(np_):
            pa[s, i] = p_sol[i] * fnorm_val
            qa[s, i] = q_sol[i] * fnorm_val
        ev[s] = e_val

        for i in range(np_):
            if equation == 0:
                rden[i] += shell_occ[s] * (pa[s, i] ** 2 + qa[s, i] ** 2)
            else:
                rden[i] += shell_occ[s] * pa[s, i] ** 2

    return ev, converged, iteration, ier_max, rv, pa, qa
