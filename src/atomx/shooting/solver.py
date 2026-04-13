"""Fast SCF solver using Numba-compiled ODE shooting.

All-Numba integration for performance. Supports Slater exchange.
Reuses AtomX constants and configuration infrastructure.
"""

import numpy as np
import numba as nb
from dataclasses import dataclass, field
from typing import Dict
from atomx.shooting.integrator import find_bound_energy


@nb.njit(cache=True)
def _V_interp_inline(r, r_grid, V_grid, n):
    if r <= r_grid[0]: return V_grid[0]
    if r >= r_grid[n-1]: return V_grid[n-1]
    lo, hi = 0, n-1
    while hi - lo > 1:
        mid = (lo+hi)//2
        if r_grid[mid] > r: hi = mid
        else: lo = mid
    t = (r - r_grid[lo])/(r_grid[hi]-r_grid[lo])
    return V_grid[lo] + t*(V_grid[hi]-V_grid[lo])


@nb.njit(cache=True)
def _dirac_rk4_step_inline(r, P, Q, h, kappa, E, r_grid, V_grid, n):
    c = 137.03599908
    def f(rr, pp, qq):
        if rr < 1e-30: return 0.0, 0.0
        V = _V_interp_inline(rr, r_grid, V_grid, n)
        return (-kappa/rr*pp + (E-V+2*c*c)/c*qq,
                kappa/rr*qq - (E-V)/c*pp)
    k1p,k1q = f(r,P,Q)
    k2p,k2q = f(r+.5*h, P+.5*h*k1p, Q+.5*h*k1q)
    k3p,k3q = f(r+.5*h, P+.5*h*k2p, Q+.5*h*k2q)
    k4p,k4q = f(r+h, P+h*k3p, Q+h*k3q)
    return P+h/6*(k1p+2*k2p+2*k3p+k4p), Q+h/6*(k1q+2*k2q+2*k3q+k4q)


@nb.njit(cache=True)
def _build_hartree(r, rho, n):
    """Hartree potential on grid: V_H(r) = (1/r)∫ρ dr' + ∫ρ/r' dr'."""
    V_H = np.zeros(n)
    cum_out = np.zeros(n)
    for i in range(1, n):
        h = r[i] - r[i-1]
        cum_out[i] = cum_out[i-1] + 0.5*h*(rho[i]+rho[i-1])
    cum_in = np.zeros(n)
    for i in range(n-2, -1, -1):
        h = r[i+1] - r[i]
        cum_in[i] = cum_in[i+1] + 0.5*h*(rho[i]/max(r[i],1e-30) + rho[i+1]/max(r[i+1],1e-30))
    for i in range(n):
        V_H[i] = cum_out[i]/max(r[i], 1e-30) + cum_in[i]
    return V_H


@nb.njit(cache=True)
def _build_exchange(r, rho, n, alpha_x):
    """Slater exchange potential."""
    V_x = np.zeros(n)
    for i in range(n):
        if rho[i] > 1e-30 and r[i] > 1e-30:
            rho_vol = rho[i] / (4.0 * np.pi * r[i]**2)
            if rho_vol > 1e-30:
                V_x[i] = -alpha_x * (3.0*rho_vol/np.pi)**(1.0/3.0)
    return V_x


@nb.njit(cache=True)
def _scf_loop(Z, n_shells, shell_n, shell_kappa, shell_occ,
              r_grid, n_grid, alpha_x, max_iter, tol):
    """Full SCF loop in Numba — no Python overhead.

    Returns: orbital_energies, total_energy, converged, iterations.
    """
    n = n_grid

    # Nuclear potential
    V_nuc = np.zeros(n)
    for i in range(1, n):
        V_nuc[i] = -Z / r_grid[i]
    V_nuc[0] = V_nuc[1]

    V_total = V_nuc.copy()
    orb_energies = np.zeros(n_shells)
    converged = False

    for iteration in range(1, max_iter + 1):
        rho = np.zeros(n)

        # Solve each orbital
        for s in range(n_shells):
            E, conv = find_bound_energy(
                r_grid, V_total, n, shell_n[s], shell_kappa[s], Z, tol=1e-10)
            orb_energies[s] = E

            # Re-integrate to get P,Q on grid for density
            c = 137.03599908
            alpha = 1.0/c; aZ = alpha*Z
            gam = abs(shell_kappa[s])*0.99
            if aZ < abs(shell_kappa[s]):
                gam = np.sqrt(shell_kappa[s]**2 - aZ**2)
            r_min = r_grid[1]

            P0 = r_min**max(gam, 0.5)
            denom = (E - V_total[1] + 2*c*c)/c
            Q0 = (gam + shell_kappa[s])/r_min * P0/denom if abs(denom) > 1e-20 else 0.0

            # Find turning point
            r_match = r_grid[n-2]
            for i in range(n-1, 0, -1):
                if E > V_total[i]:
                    r_match = r_grid[i]
                    break

            # Outward to each grid point — accumulate density
            P, Q = P0, Q0
            r_cur = r_min
            norm = 0.0
            P_grid = np.zeros(n)
            Q_grid = np.zeros(n)
            P_grid[1] = P; Q_grid[1] = Q

            for i in range(2, n):
                h = r_grid[i] - r_grid[i-1]
                P, Q = _dirac_rk4_step_inline(r_grid[i-1], P, Q, h, shell_kappa[s], E,
                                               r_grid, V_total, n)
                P_grid[i] = P; Q_grid[i] = Q
                norm += (P*P + Q*Q) * h
                # Renormalize
                amp = max(abs(P), abs(Q))
                if amp > 1e8:
                    fac = 1.0/amp
                    P *= fac; Q *= fac
                    for j in range(i+1):
                        P_grid[j] *= fac
                        Q_grid[j] *= fac
                    norm *= fac*fac

            if norm > 1e-30:
                fac = 1.0/np.sqrt(norm)
                for i in range(n):
                    P_grid[i] *= fac
                    Q_grid[i] *= fac

            for i in range(n):
                rho[i] += shell_occ[s] * (P_grid[i]**2 + Q_grid[i]**2)

        # Build new potential
        V_H = _build_hartree(r_grid, rho, n)
        V_x = _build_exchange(r_grid, rho, n, alpha_x)
        V_trial = np.zeros(n)
        for i in range(n):
            V_trial[i] = V_nuc[i] + V_H[i] + V_x[i]

        # Convergence
        chi = 0.0
        v_max = 1e-18
        for i in range(n):
            d = abs(V_trial[i] - V_total[i])
            if d > chi: chi = d
            a = abs(V_total[i])
            if a > v_max: v_max = a
        chi /= v_max

        # Mix
        mixing = min(0.4, 0.02 + 0.02*iteration)
        for i in range(n):
            V_total[i] = (1-mixing)*V_total[i] + mixing*V_trial[i]

        if chi < tol and iteration > 5:
            converged = True
            break

    # Total energy
    E_sum = 0.0
    for s in range(n_shells):
        E_sum += shell_occ[s] * orb_energies[s]
    E_J = 0.0
    V_H = _build_hartree(r_grid, rho, n)
    for i in range(1, n):
        h = r_grid[i] - r_grid[i-1]
        E_J += 0.5*h*(rho[i]*V_H[i] + rho[i-1]*V_H[i-1])
    E_J *= 0.5
    E_xc = 0.0
    for i in range(1, n):
        ri = r_grid[i]
        if ri > 1e-30 and rho[i] > 1e-30:
            rv = rho[i]/(4*np.pi*ri*ri)
            if rv > 1e-30:
                h = r_grid[i] - r_grid[i-1]
                E_xc += h*alpha_x*0.75*(3.0/np.pi)**(1.0/3.0)*rv**(4.0/3.0)*4*np.pi*ri*ri
    E_total = E_sum - E_J - E_xc/3.0

    return orb_energies, E_total, converged, iteration


@dataclass
class ShootingResult:
    converged: bool
    iterations: int
    total_energy: float
    orbital_energies: Dict[str, float]
    method: str = "shooting"
    energy_components: Dict[str, float] = field(default_factory=dict)


class ShootingSolver:
    """Fast DFS SCF using all-Numba ODE shooting."""

    def __init__(self, Z, subshells, n_grid=1000, r_max=75.0, alpha_x=1.5):
        self.Z = Z
        self.subshells = subshells
        self.n_grid = n_grid
        self.alpha_x = alpha_x

        # Logarithmic grid
        r_min = 1e-6 / max(Z, 1)
        self.r_grid = np.zeros(n_grid)
        self.r_grid[0] = 0.0
        self.r_grid[1:] = np.geomspace(r_min, r_max, n_grid - 1)

    def solve(self, max_iter=200, tol=1e-8):
        n_shells = len(self.subshells)
        shell_n = np.array([sh.n for sh in self.subshells], dtype=np.int64)
        shell_kappa = np.array([sh.kappa for sh in self.subshells], dtype=np.int64)
        shell_occ = np.array([sh.occupation for sh in self.subshells], dtype=np.float64)

        orb_energies, E_total, converged, iterations = _scf_loop(
            float(self.Z), n_shells, shell_n, shell_kappa, shell_occ,
            self.r_grid, self.n_grid, self.alpha_x, max_iter, tol)

        oe_dict = {}
        for i, sh in enumerate(self.subshells):
            oe_dict[sh.label] = orb_energies[i]

        return ShootingResult(
            converged=converged, iterations=iterations,
            total_energy=E_total, orbital_energies=oe_dict)
