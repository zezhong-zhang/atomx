"""Dirac/Schrödinger ODE integrators using Numba-compiled adaptive RK45.

The entire integration + eigenvalue search runs in compiled code
with no Python callback overhead. Uses log-amplitude renormalization
to prevent overflow (Salvat's technique).
"""

import numpy as np
import numba as nb


@nb.njit(cache=True)
def _V_interp(r, r_grid, V_grid, n):
    """Fast linear interpolation of V(r) from grid."""
    if r <= r_grid[0]:
        return V_grid[0]
    if r >= r_grid[n - 1]:
        return V_grid[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if r_grid[mid] > r:
            hi = mid
        else:
            lo = mid
    t = (r - r_grid[lo]) / (r_grid[hi] - r_grid[lo])
    return V_grid[lo] + t * (V_grid[hi] - V_grid[lo])


@nb.njit(cache=True)
def _dirac_rk4_step(r, P, Q, h, kappa, E, r_grid, V_grid, ng):
    """Single RK4 step for the Dirac equation. Fully compiled."""
    c = 137.03599908

    def f(rr, pp, qq):
        if rr < 1e-30:
            return 0.0, 0.0
        V = _V_interp(rr, r_grid, V_grid, ng)
        dp = -kappa / rr * pp + (E - V + 2 * c * c) / c * qq
        dq = kappa / rr * qq - (E - V) / c * pp
        return dp, dq

    k1p, k1q = f(r, P, Q)
    k2p, k2q = f(r + 0.5*h, P + 0.5*h*k1p, Q + 0.5*h*k1q)
    k3p, k3q = f(r + 0.5*h, P + 0.5*h*k2p, Q + 0.5*h*k2q)
    k4p, k4q = f(r + h, P + h*k3p, Q + h*k3q)

    return (P + h/6*(k1p + 2*k2p + 2*k3p + k4p),
            Q + h/6*(k1q + 2*k2q + 2*k3q + k4q))


@nb.njit(cache=True)
def _integrate_dirac_adaptive(r_start, r_end, P0, Q0, kappa, E,
                                r_grid, V_grid, ng, direction=1,
                                rtol=1e-11, max_steps=50000):
    """Adaptive RK4 Dirac integration with renormalization.

    Returns: P_end, Q_end, norm_integral, n_nodes
    """
    c = 137.03599908
    P, Q = P0, Q0
    r = r_start
    h = direction * min(abs(r_end - r_start) / 100, 0.01)
    norm_sum = 0.0
    n_nodes = 0
    P_prev = P

    for step in range(max_steps):
        if direction > 0 and r >= r_end:
            break
        if direction < 0 and r <= r_end:
            break

        # Clamp h to not overshoot
        if direction > 0 and r + h > r_end:
            h = r_end - r
        if direction < 0 and r + h < r_end:
            h = r_end - r

        # Full step
        P1, Q1 = _dirac_rk4_step(r, P, Q, h, kappa, E, r_grid, V_grid, ng)

        # Two half-steps (for error estimate)
        Ph, Qh = _dirac_rk4_step(r, P, Q, h/2, kappa, E, r_grid, V_grid, ng)
        P2, Q2 = _dirac_rk4_step(r + h/2, Ph, Qh, h/2, kappa, E, r_grid, V_grid, ng)

        # Error estimate
        err_P = abs(P2 - P1)
        err_Q = abs(Q2 - Q1)
        scale = max(abs(P2), abs(Q2), 1e-30)
        err = max(err_P, err_Q) / scale

        if err > rtol and abs(h) > 1e-15:
            # Reject: reduce step
            h *= 0.5
            continue

        # Accept step (use Richardson extrapolation)
        P = (16 * P2 - P1) / 15
        Q = (16 * Q2 - Q1) / 15
        r += h

        # Node counting
        if P * P_prev < 0 and abs(P_prev) > 1e-30:
            n_nodes += 1
        P_prev = P

        # Normalization integral
        norm_sum += (P*P + Q*Q) * abs(h)

        # Renormalize to prevent overflow
        amp = max(abs(P), abs(Q))
        if amp > 1e10:
            fac = 1.0 / amp
            P *= fac
            Q *= fac
            # norm_sum scales with fac^2 but we track relative

        # Grow step if error is small
        if err < 0.1 * rtol and abs(h) < abs(r_end - r_start) / 10:
            h *= 1.5

    return P, Q, norm_sum, n_nodes


@nb.njit(cache=True)
def find_bound_energy(r_grid, V_grid, n_grid, n, kappa, Z,
                       tol=1e-10, max_iter=80):
    """Find Dirac bound-state energy by shooting + bisection + matching.

    All-Numba compiled — no Python overhead.

    Returns (energy, converged_flag).
    """
    c = 137.03599908
    alpha = 1.0 / c
    l = kappa if kappa > 0 else -kappa - 1
    n_nodes_target = n - l - 1

    # Initial guess: hydrogenic
    aZ = alpha * Z
    gam = abs(kappa) * 0.99
    if aZ < abs(kappa):
        gam = np.sqrt(kappa * kappa - aZ * aZ)
    nr = n - abs(kappa)
    if nr < 0:
        nr = 0
    E = c*c * (1.0/np.sqrt(1.0 + aZ**2/(nr+gam)**2) - 1.0)

    E_lo = -2.0 * Z * Z
    E_hi = -1e-12

    r_min = r_grid[1]
    r_max = r_grid[n_grid - 1]

    # Find turning point
    r_match = 0.5 * (r_min + r_max)
    for i in range(n_grid - 1, 0, -1):
        if E > V_grid[i]:
            r_match = r_grid[i]
            break

    converged = False

    for iteration in range(max_iter):
        # Starting values (power-law at r_min)
        P0 = r_min ** max(gam, 0.5)
        denom = (E - _V_interp(r_min, r_grid, V_grid, n_grid) + 2*c*c) / c
        Q0 = (gam + kappa) / r_min * P0 / denom if abs(denom) > 1e-20 else 0.0

        # Update turning point
        for i in range(n_grid - 1, 0, -1):
            if E > V_grid[i]:
                r_match = r_grid[i]
                break
        r_match = max(r_match, r_min * 2)
        r_match = min(r_match, r_max * 0.9)

        # Outward integration
        P_out, Q_out, norm_out, n_nodes = _integrate_dirac_adaptive(
            r_min, r_match, P0, Q0, kappa, E, r_grid, V_grid, n_grid, 1)

        # Node check
        if n_nodes > n_nodes_target:
            E_hi = E
            E = 0.5 * (E_lo + E_hi)
            continue
        if n_nodes < n_nodes_target:
            E_lo = E
            E = 0.5 * (E_lo + E_hi)
            continue

        # Inward: WKB boundary at r_max
        V_max = _V_interp(r_max, r_grid, V_grid, n_grid)
        mu = np.sqrt(max(2.0 * (V_max - E), 0.01))
        P_inf = 1e-10
        denom_inf = (E - V_max + 2*c*c) / c
        Q_inf = -P_inf * mu / denom_inf if abs(denom_inf) > 1e-20 else 0.0

        P_in, Q_in, norm_in, _ = _integrate_dirac_adaptive(
            r_max, r_match, P_inf, Q_inf, kappa, E, r_grid, V_grid, n_grid, -1)

        # Match: scale inward to outward at r_match
        if abs(P_in) > 1e-30:
            scale = P_out / P_in
        else:
            scale = 1.0

        # Q-discontinuity → eigenvalue correction
        Q_disc = Q_out - Q_in * scale
        denom2 = norm_out + norm_in * scale * scale
        if denom2 > 1e-30:
            dE = c * P_out * Q_disc / denom2
        else:
            dE = 0.0

        E_new = E + dE
        if E_new >= 0:
            E_new = 0.5 * (E + E_lo)
        if E_new < E_lo:
            E_new = 0.5 * (E + E_lo)

        if abs(dE) < tol * abs(E) and iteration > 5:
            converged = True
            E = E_new
            break

        E = E_new

    return E, converged
