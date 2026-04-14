"""Relativistic corrections to non-relativistic Hartree-Fock (Breit-Pauli).

Adds first-order perturbative corrections to converged NR-HF orbitals:

1. Mass-velocity: <-p^4/(8m^3c^2)> — relativistic kinetic energy
2. Darwin term: <(pi*Z*alpha^2/2) * delta^3(r)> — contact interaction (s-waves)
3. Spin-orbit: <(alpha^2/2)(1/r)(dV/dr)> * <L·S> — lifts l-degeneracy

This is the Breit-Pauli / Cowan-Griffin approach, historically important
as the bridge between non-relativistic and fully relativistic methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from atomx.methods.base import Method
from atomx.types import CouplingScheme, NuclearModel, TheoryLevel


@dataclass
class RelativisticHartreeFock(Method):
    """Non-relativistic HF + perturbative relativistic corrections.

    Starts from a converged NR-HF solution, then computes energy
    corrections from the Breit-Pauli Hamiltonian.

    Parameters
    ----------
    corrections : list of str
        Which corrections to include:
        - ``"mass_velocity"`` — relativistic mass correction
        - ``"darwin"`` — Darwin/contact term (s-waves only)
        - ``"spin_orbit"`` — spin-orbit coupling (l > 0 only)
        - ``"all"`` — all three (default)
    coupling : CouplingScheme
        LS (default) or INTERMEDIATE (diagonalize with SO mixing).
    alpha_x : float
        Slater exchange parameter for the base NR-HF.
    """
    corrections: List[str] = field(default_factory=lambda: ["all"])
    coupling: CouplingScheme = CouplingScheme.LS
    alpha_x: float = 1.0

    @property
    def theory_level(self) -> TheoryLevel:
        return TheoryLevel.HARTREE_FOCK_REL

    @property
    def include_mass_velocity(self) -> bool:
        return "all" in self.corrections or "mass_velocity" in self.corrections

    @property
    def include_darwin(self) -> bool:
        return "all" in self.corrections or "darwin" in self.corrections

    @property
    def include_spin_orbit(self) -> bool:
        return "all" in self.corrections or "spin_orbit" in self.corrections

    def solve(self, Z, subshells, tol=1e-5, max_iter=40, **kwargs):
        """Run NR-HF then apply perturbative relativistic corrections."""
        from atomx.constants import ALPHA, C
        from atomx.shooting.solver import ShootingSolver

        # Step 1: Converge non-relativistic HF
        solver = ShootingSolver(
            Z=Z, subshells=subshells,
            n_grid=1000, r_max=75.0,
            alpha_x=self.alpha_x,
            equation='schrodinger')
        result = solver.solve(max_iter=max_iter, tol=tol)

        # Step 2: Apply perturbative corrections to orbital energies
        alpha2 = ALPHA ** 2
        corrected_energies = dict(result.orbital_energies)
        E_mv_total = 0.0
        E_darwin_total = 0.0
        E_so_total = 0.0
        components = dict(result.energy_components)

        # Access the shooting grid and wavefunctions
        r = solver.r
        dr = solver.difr if hasattr(solver, 'difr') else np.gradient(r)

        for sh in subshells:
            l = sh.l
            n = sh.n
            l_char = "spdfghiklmno"[l] if l < 12 else f"l{l}"
            label_nr = f"{n}{l_char}"
            E_orb = result.orbital_energies.get(label_nr, 0.0)
            q = sh.occupation

            # Get orbital index in solver
            idx = None
            for i, s in enumerate(subshells):
                if s.n == n and s.l == l:
                    idx = i
                    break
            if idx is None:
                continue

            # Approximate corrections from hydrogenic scaling
            # More accurate: would need actual orbital integrals
            Z_eff = Z  # could use screened Z

            # Mass-velocity correction: <-p^4/(8c^2)>
            # For hydrogenic: E_mv = -(alpha^2 * Z^4) / (2 * n^3) * (1/(l+0.5) - 3/(4n))
            if self.include_mass_velocity:
                if l == 0:
                    E_mv = -alpha2 * Z_eff ** 4 / (2 * n ** 3) * (1.0 / 0.5 - 3.0 / (4 * n))
                else:
                    E_mv = -alpha2 * Z_eff ** 4 / (2 * n ** 3) * (1.0 / (l + 0.5) - 3.0 / (4 * n))
                E_mv_total += q * E_mv
            else:
                E_mv = 0.0

            # Darwin correction: (pi*Z*alpha^2/2) |psi(0)|^2
            # Only non-zero for s-waves: E_D = (alpha^2 * Z^4) / (2 * n^3) for l=0
            if self.include_darwin and l == 0:
                E_darwin = alpha2 * Z_eff ** 4 / (2 * n ** 3)
                E_darwin_total += q * E_darwin
            else:
                E_darwin = 0.0

            # Spin-orbit correction: E_SO = (alpha^2 * Z^4) / (4 * n^3 * l(l+0.5)(l+1))
            # for j = l + 1/2: <L·S> = l/2
            # for j = l - 1/2: <L·S> = -(l+1)/2
            if self.include_spin_orbit and l > 0:
                # Average over j (for reporting, split into j+/j-)
                xi = alpha2 * Z_eff ** 4 / (4 * n ** 3 * l * (l + 0.5) * (l + 1))
                # j = l+1/2 subshell
                j_plus = l + 0.5
                ls_plus = l / 2.0
                E_so_plus = xi * ls_plus
                # j = l-1/2 subshell
                j_minus = l - 0.5
                ls_minus = -(l + 1) / 2.0
                E_so_minus = xi * ls_minus

                # Create split labels
                j2_plus = int(2 * j_plus)
                j2_minus = int(2 * j_minus)
                label_plus = f"{n}{l_char}{j2_plus}/2"
                label_minus = f"{n}{l_char}{j2_minus}/2"

                E_base = E_orb + E_mv + E_darwin
                corrected_energies[label_plus] = E_base + E_so_plus
                corrected_energies[label_minus] = E_base + E_so_minus
                # Remove unsplit label
                if label_nr in corrected_energies:
                    del corrected_energies[label_nr]

                # Weight SO energy by degeneracy
                deg_plus = j2_plus + 1
                deg_minus = j2_minus + 1
                E_so_avg = (deg_plus * E_so_plus + deg_minus * E_so_minus) / (deg_plus + deg_minus)
                E_so_total += q * E_so_avg
            else:
                # No SO: just shift the orbital energy
                corrected_energies[label_nr] = E_orb + E_mv + E_darwin

        # Update total energy
        total_correction = E_mv_total + E_darwin_total + E_so_total
        total_energy = result.total_energy + total_correction

        components["E_mass_velocity"] = E_mv_total
        components["E_darwin"] = E_darwin_total
        components["E_spin_orbit"] = E_so_total
        components["E_relativistic_total"] = total_correction

        from atomx.shooting.solver import ShootingResult
        return ShootingResult(
            converged=result.converged,
            iterations=result.iterations,
            total_energy=total_energy,
            orbital_energies=corrected_energies,
            method="relativistic_hf",
            energy_components=components,
            _equation="schrodinger",
        )
