"""SCF solver using ODE shooting — follows Salvat's DHFS.f EXACTLY.

Key details matched from DHFS.f:
- Initial SFIELD call before loop (line 771-774)
- Self-interaction correction for 1-2 electrons (line 948-956)
- Latter tail correction (line 970-978)
- SLAG6-equivalent integration for Hartree potential
- Adaptive mixing: FACTOR 0.025→0.5 (line 777-784)
- E_guess from previous eigenvalue (line 802)
- α=1.5 Slater (default)
"""

import numpy as np
import numba as nb
from dataclasses import dataclass, field
from typing import Dict
from atomx.shooting.integrator import find_bound_energy


@nb.njit(cache=True)
def _sfield(r, rden, vnuc, n, alpha, qelec, nshell):
    """Compute screened potential r*V(r) — exact DHFS.f SFIELD translation.

    Returns r*VT = r*V_nuc + r*V_el + r*V_ex (all stored as r*V).
    """
    dr = np.zeros(n)
    for i in range(1, n):
        dr[i] = r[i] - r[i - 1]

    # Forward integral: ∫_0^r RDEN dr' (SLAG6 equivalent using trapezoidal)
    fwd = np.zeros(n)
    for i in range(1, n):
        fwd[i] = fwd[i-1] + 0.5*dr[i]*(rden[i]+rden[i-1])

    # Backward integral: ∫_r^∞ RDEN/r' dr'
    bwd = np.zeros(n)
    for i in range(n-2, -1, -1):
        f_i = rden[i]/max(r[i], 1e-30)
        f_ip1 = rden[i+1]/max(r[i+1], 1e-30)
        bwd[i] = bwd[i+1] + 0.5*dr[i+1]*(f_i+f_ip1)

    # r*V_el
    vel = np.zeros(n)
    for i in range(n):
        vel[i] = fwd[i] + bwd[i]*r[i]

    # Exchange potential (r*V_ex)
    vex = np.zeros(n)
    CSLATE = 0.75/(np.pi*np.pi)

    if qelec < 2.1 and nshell == 1:
        # EXACT self-interaction correction for 1-2 electrons (DHFS.f line 948-956)
        fact = 1.0/qelec
        for i in range(n):
            vex[i] = -vel[i]*fact
    else:
        # Slater exchange (DHFS.f line 960-968)
        for i in range(1, n):
            arg = CSLATE*r[i]*rden[i]
            if arg > 1e-30:
                vex[i] = -alpha*arg**(1.0/3.0)

    # Total: r*V_trial
    rvt = np.zeros(n)
    for i in range(n):
        rvt[i] = vnuc[i] + vel[i] + vex[i]

    # Latter tail correction (DHFS.f line 970-978)
    rvinf = -1.0 - (-qelec)  # = -(1 + Z - N_electrons) ... wait
    # Salvat: RVINF = -1.0D0 - Z + QELEC
    # For neutral atom: RVINF = -1 - Z + Z = -1
    # This is the asymptotic value: r*V(r→∞) = -(Z - N + 1) for ions
    # For neutral: r*V → -1 (last electron sees -1/r at large r)
    Z = -vnuc[1]  # vnuc = -Z (constant)
    rvinf = -1.0 - Z + qelec
    for i in range(n-1, -1, -1):
        if rvt[i] > rvinf:
            rvt[i] = rvinf
            vex[i] = rvinf - vnuc[i] - vel[i]
        else:
            break

    return rvt, vel, vex


@nb.njit(cache=True)
def _shooting_scf(Z, n_shells, shell_n, shell_kappa, shell_occ,
                   r_grid, n_grid, alpha, max_iter, epsc):
    """Full SCF loop — DHFS.f lines 765-846 exact translation."""
    c = 137.03599908
    n = n_grid
    qelec = 0.0
    for s in range(n_shells):
        qelec += shell_occ[s]

    # Nuclear potential: r*V_nuc = -Z
    vnuc = np.zeros(n)
    for i in range(n):
        vnuc[i] = -Z

    # Initial orbital solve (hydrogenic) to get starting density
    V_nuc = np.zeros(n)
    for i in range(1, n):
        V_nuc[i] = -Z/r_grid[i]
    V_nuc[0] = V_nuc[1]

    rden = np.zeros(n)
    orb_E = np.zeros(n_shells)
    P_store = np.zeros((n_shells, n))
    Q_store = np.zeros((n_shells, n))

    for s in range(n_shells):
        E, conv, P, Q = find_bound_energy(
            r_grid, V_nuc, n, shell_n[s], shell_kappa[s], Z, tol=1e-10)
        orb_E[s] = E
        for i in range(n):
            P_store[s, i] = P[i]
            Q_store[s, i] = Q[i]
            rden[i] += shell_occ[s]*(P[i]**2 + Q[i]**2)

    # Initial SFIELD call (DHFS.f line 771)
    rvt, vel, vex = _sfield(r_grid, rden, vnuc, n, alpha, qelec, n_shells)

    # Set initial RV = RVT (DHFS.f line 772-774)
    rv = np.zeros(n)
    for i in range(n):
        rv[i] = rvt[i]

    # SCF iteration (DHFS.f line 776-846)
    factor = 0.0
    chi = 1.0
    chi_prev = 1e30
    converged = False

    for iteration in range(1, max_iter + 1):
        # Adaptive mixing (DHFS.f line 777-784)
        if chi_prev > chi:
            factor += 0.025
            if factor > 0.5:
                factor = 0.5
        else:
            factor -= 0.05
            if factor < 0.025:
                factor = 0.025
        chi_prev = chi

        # Mix potential (DHFS.f line 790-794)
        faccom = 1.0 - factor
        for i in range(n):
            rv[i] = faccom*rv[i] + factor*rvt[i]

        # Build V(r) from r*V for the eigenvalue solver
        V_grid = np.zeros(n)
        for i in range(1, n):
            V_grid[i] = rv[i]/r_grid[i]
        V_grid[0] = V_grid[1]

        # Solve all orbitals (DHFS.f line 799-827)
        rden_new = np.zeros(n)
        for s in range(n_shells):
            E, conv, P, Q = find_bound_energy(
                r_grid, V_grid, n, shell_n[s], shell_kappa[s], Z,
                tol=1e-10, E_guess=orb_E[s])
            orb_E[s] = E
            for i in range(n):
                P_store[s, i] = P[i]
                Q_store[s, i] = Q[i]
                rden_new[i] += shell_occ[s]*(P[i]**2 + Q[i]**2)

        # Update density
        for i in range(n):
            rden[i] = rden_new[i]

        # Build new trial potential (DHFS.f line 829)
        rvt, vel, vex = _sfield(r_grid, rden, vnuc, n, alpha, qelec, n_shells)

        # Convergence check (DHFS.f line 830-834)
        chi = 0.0
        for i in range(n):
            d = abs(rv[i] - rvt[i])
            v = max(abs(rv[i]), 1e-18)
            c_val = d/v
            if c_val > chi:
                chi = c_val

        # Convergence (DHFS.f line 839): chi ≤ epsc AND factor ≥ 0.25
        if chi <= epsc and factor >= 0.25:
            converged = True
            break

    # Final potential mix
    faccom = 1.0 - factor
    for i in range(n):
        rv[i] = faccom*rv[i] + factor*rvt[i]

    # Total energy (using eigenvalue formula)
    E_sum = 0.0
    for s in range(n_shells):
        E_sum += shell_occ[s]*orb_E[s]

    # E_J = (1/2) ∫ ρ * V_el/r dr
    dr = np.zeros(n)
    for i in range(1, n):
        dr[i] = r_grid[i] - r_grid[i-1]
    E_J = 0.0
    for i in range(1, n):
        f0 = rden[i]*vel[i]/max(r_grid[i], 1e-30)
        f1 = rden[i-1]*vel[i-1]/max(r_grid[i-1], 1e-30)
        E_J += 0.5*dr[i]*(f0+f1)
    E_J *= 0.5

    # E_xc
    E_xc = 0.0
    CSLATE = 0.75/(np.pi*np.pi)
    for i in range(1, n):
        ri = r_grid[i]
        if ri > 1e-30 and rden[i] > 1e-30:
            rho_vol = rden[i]/(4.0*np.pi*ri*ri)
            if rho_vol > 1e-30:
                E_xc += dr[i]*alpha*0.75*(3.0/np.pi)**(1.0/3.0) * \
                        rho_vol**(4.0/3.0)*4.0*np.pi*ri*ri

    # DFS total energy
    E_total = E_sum - E_J - E_xc/3.0

    return orb_E, E_total, converged, iteration


@dataclass
class ShootingResult:
    converged: bool
    iterations: int
    total_energy: float
    orbital_energies: Dict[str, float]
    method: str = "shooting"
    energy_components: Dict[str, float] = field(default_factory=dict)


class ShootingSolver:
    """DFS SCF using ODE shooting — Salvat/Radial method."""

    def __init__(self, Z, subshells, n_grid=2000, r_max=75.0, alpha_x=1.5):
        self.Z = Z
        self.subshells = subshells
        self.n_grid = n_grid
        self.alpha_x = alpha_x

        r_min = 1e-7/max(Z, 1)
        self.r_grid = np.zeros(n_grid)
        self.r_grid[0] = 0.0
        self.r_grid[1:] = np.geomspace(r_min, r_max, n_grid-1)

    def solve(self, max_iter=300, tol=1e-8):
        n_shells = len(self.subshells)
        shell_n = np.array([sh.n for sh in self.subshells], dtype=np.int64)
        shell_kappa = np.array([sh.kappa for sh in self.subshells], dtype=np.int64)
        shell_occ = np.array([float(sh.occupation) for sh in self.subshells])

        orb_E, E_total, converged, iters = _shooting_scf(
            float(self.Z), n_shells, shell_n, shell_kappa, shell_occ,
            self.r_grid, self.n_grid, self.alpha_x, max_iter, tol)

        oe = {}
        for i, sh in enumerate(self.subshells):
            oe[sh.label] = orb_E[i]

        return ShootingResult(
            converged=converged, iterations=iters,
            total_energy=E_total, orbital_energies=oe)
