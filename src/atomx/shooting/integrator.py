"""Dirac ODE shooting solver using grid-based RK4.

Integrates step-by-step on a dense logarithmic grid (like Salvat),
giving consistent accuracy for eigenvalue refinement via Q-matching.
"""

import numpy as np
import numba as nb


@nb.njit(cache=True)
def _rk4_step(r, P, Q, h, kappa, E, V_func_r, V_func_rph):
    """Single RK4 step for Dirac equation with V at endpoints."""
    c = 137.03599908
    V_mid = 0.5*(V_func_r + V_func_rph)  # midpoint V

    def f(rr, pp, qq, V):
        if rr < 1e-30:
            return 0.0, 0.0
        return (-kappa/rr*pp + (E-V+2*c*c)/c*qq,
                kappa/rr*qq - (E-V)/c*pp)

    k1p, k1q = f(r, P, Q, V_func_r)
    k2p, k2q = f(r+.5*h, P+.5*h*k1p, Q+.5*h*k1q, V_mid)
    k3p, k3q = f(r+.5*h, P+.5*h*k2p, Q+.5*h*k2q, V_mid)
    k4p, k4q = f(r+h, P+h*k3p, Q+h*k3q, V_func_rph)
    return (P + h/6*(k1p+2*k2p+2*k3p+k4p),
            Q + h/6*(k1q+2*k2q+2*k3q+k4q))


@nb.njit(cache=True)
def find_bound_energy(r_grid, V_grid, n_grid, n, kappa, Z,
                       tol=1e-10, max_iter=80, E_guess=0.0):
    """Find Dirac bound-state energy using grid-based shooting.

    Grid-based RK4: steps exactly from grid point to grid point,
    using V values known at each point. No interpolation needed.
    """
    c = 137.03599908
    alpha = 1.0/c
    l = kappa if kappa > 0 else -kappa - 1
    n_nodes_target = n - l - 1

    aZ = alpha*Z
    gam = abs(kappa)*0.99
    if aZ < abs(kappa):
        gam = np.sqrt(kappa*kappa - aZ*aZ)
    nr = n - abs(kappa)
    if nr < 0:
        nr = 0

    if E_guess < -1e-15:
        E = E_guess
    else:
        E = c*c*(1.0/np.sqrt(1.0 + aZ**2/(nr+gam)**2) - 1.0)

    E_lo = min(3.0*E, -2.0*Z*Z) if E < -1e-10 else -2.0*Z*Z
    E_hi = max(0.3*E, -1e-12) if E < -1e-10 else -1e-12

    r_min = r_grid[1]
    converged = False
    P_out = np.zeros(n_grid)
    Q_out = np.zeros(n_grid)
    P_in = np.zeros(n_grid)
    Q_in = np.zeros(n_grid)

    for iteration in range(max_iter):
        # Find classical turning point
        iotp = n_grid - 2
        for i in range(n_grid-1, 1, -1):
            if E > V_grid[i]:
                iotp = i
                break
        iotp = max(iotp, 10)
        iotp = min(iotp, n_grid - 10)

        # === OUTWARD from r_min to iotp ===
        P0 = r_min**max(gam, 0.5)
        V0 = V_grid[1]
        denom = (E - V0 + 2*c*c)/c
        Q0 = (gam+kappa)/r_min * P0/denom if abs(denom) > 1e-20 else 0.0

        for i in range(n_grid):
            P_out[i] = 0.0
            Q_out[i] = 0.0
        P_out[1] = P0; Q_out[1] = Q0
        Pc, Qc = P0, Q0
        n_nodes = 0
        norm_out = 0.0

        for i in range(2, iotp+1):
            h = r_grid[i] - r_grid[i-1]
            Pn, Qn = _rk4_step(r_grid[i-1], Pc, Qc, h, kappa, E,
                                 V_grid[i-1], V_grid[i])
            if Pn*Pc < 0 and abs(Pc) > 1e-30:
                n_nodes += 1
            Pc, Qc = Pn, Qn
            P_out[i] = Pc; Q_out[i] = Qc
            norm_out += (Pc*Pc + Qc*Qc)*h

            # Renormalize
            amp = max(abs(Pc), abs(Qc))
            if amp > 1e10:
                fac = 1.0/amp
                Pc *= fac; Qc *= fac
                for j in range(i+1):
                    P_out[j] *= fac; Q_out[j] *= fac
                norm_out *= fac*fac

        # Node check → bisection
        if n_nodes > n_nodes_target:
            E_hi = E
            E = 0.5*(E_lo + E_hi)
            continue
        if n_nodes < n_nodes_target:
            E_lo = E
            E = 0.5*(E_lo + E_hi)
            continue

        # === INWARD from r_max to iotp ===
        V_last = V_grid[n_grid-1]
        mu = np.sqrt(max(2.0*(V_last-E), 0.01))
        P_inf = 1e-10
        denom_in = (E - V_last + 2*c*c)/c
        Q_inf = -P_inf*mu/denom_in if abs(denom_in) > 1e-20 else 0.0

        for i in range(n_grid):
            P_in[i] = 0.0; Q_in[i] = 0.0
        P_in[n_grid-1] = P_inf; Q_in[n_grid-1] = Q_inf
        Pc, Qc = P_inf, Q_inf
        norm_in = 0.0

        for i in range(n_grid-2, iotp-1, -1):
            h = r_grid[i] - r_grid[i+1]  # negative
            Pn, Qn = _rk4_step(r_grid[i+1], Pc, Qc, h, kappa, E,
                                 V_grid[i+1], V_grid[i])
            Pc, Qc = Pn, Qn
            P_in[i] = Pc; Q_in[i] = Qc
            norm_in += (Pc*Pc + Qc*Qc)*abs(h)

            amp = max(abs(Pc), abs(Qc))
            if amp > 1e10:
                fac = 1.0/amp
                Pc *= fac; Qc *= fac
                for j in range(i, n_grid):
                    P_in[j] *= fac; Q_in[j] *= fac
                norm_in *= fac*fac

        # Match at iotp
        P_out_m = P_out[iotp]
        Q_out_m = Q_out[iotp]
        P_in_m = P_in[iotp]
        Q_in_m = Q_in[iotp]

        if abs(P_in_m) > 1e-30:
            scale = P_out_m/P_in_m
        else:
            scale = 1.0

        # Q-discontinuity → eigenvalue correction (Salvat ME-5.15)
        Q_disc = Q_out_m - Q_in_m*scale
        denom2 = norm_out + norm_in*scale*scale
        if denom2 > 1e-30:
            dE = c*P_out_m*Q_disc/denom2
        else:
            dE = 0.0

        E_new = E + dE
        if E_new >= E_hi:
            E_new = 0.5*(E + E_hi)
        if E_new <= E_lo:
            E_new = 0.5*(E + E_lo)

        if abs(dE) < tol*abs(E) and iteration > 3:
            converged = True
            E = E_new
            break

        E = E_new

    # Build final matched wavefunction
    P_final = np.zeros(n_grid)
    Q_final = np.zeros(n_grid)

    # Re-integrate with converged E
    P0 = r_min**max(gam, 0.5)
    denom = (E - V_grid[1] + 2*c*c)/c
    Q0 = (gam+kappa)/r_min*P0/denom if abs(denom) > 1e-20 else 0.0
    P_final[1] = P0; Q_final[1] = Q0
    Pc, Qc = P0, Q0

    for i in range(2, iotp+1):
        h = r_grid[i] - r_grid[i-1]
        Pc, Qc = _rk4_step(r_grid[i-1], Pc, Qc, h, kappa, E,
                             V_grid[i-1], V_grid[i])
        P_final[i] = Pc; Q_final[i] = Qc
        amp = max(abs(Pc), abs(Qc))
        if amp > 1e10:
            fac = 1.0/amp
            Pc *= fac; Qc *= fac
            for j in range(i+1):
                P_final[j] *= fac; Q_final[j] *= fac

    P_out_m = Pc; Q_out_m = Qc

    # Inward
    V_last = V_grid[n_grid-1]
    mu = np.sqrt(max(2.0*(V_last-E), 0.01))
    P_inf = 1e-10
    denom_in = (E - V_last + 2*c*c)/c
    Q_inf = -P_inf*mu/denom_in if abs(denom_in) > 1e-20 else 0.0
    P_final[n_grid-1] = P_inf; Q_final[n_grid-1] = Q_inf
    Pc, Qc = P_inf, Q_inf

    for i in range(n_grid-2, iotp-1, -1):
        h = r_grid[i] - r_grid[i+1]
        Pc, Qc = _rk4_step(r_grid[i+1], Pc, Qc, h, kappa, E,
                             V_grid[i+1], V_grid[i])
        P_final[i] = Pc; Q_final[i] = Qc
        amp = max(abs(Pc), abs(Qc))
        if amp > 1e10:
            fac = 1.0/amp
            Pc *= fac; Qc *= fac
            for j in range(i, n_grid):
                P_final[j] *= fac; Q_final[j] *= fac

    # Scale inward to match outward at iotp
    P_in_m = P_final[iotp]
    if abs(P_in_m) > 1e-30 and abs(P_out_m) > 1e-30:
        # The outward part was already written to P_final up to iotp
        # Scale the inward part (iotp onward)
        scale = P_out_m/P_in_m
        # But P_final[:iotp] is already from outward, P_final[iotp:] from inward
        # Need to scale inward part
        for i in range(iotp, n_grid):
            P_final[i] *= scale
            Q_final[i] *= scale

    # Normalize
    norm = 0.0
    for i in range(1, n_grid):
        norm += (P_final[i]**2 + Q_final[i]**2)*(r_grid[i]-r_grid[i-1])
    if norm > 1e-30:
        fac = 1.0/np.sqrt(norm)
        for i in range(n_grid):
            P_final[i] *= fac; Q_final[i] *= fac

    return E, converged, P_final, Q_final
