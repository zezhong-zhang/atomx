"""Relativistic corrections to Hartree-Fock (Breit-Pauli / Cowan-Griffin).

Adds first-order perturbative corrections to converged NR-HF orbitals:

1. Mass-velocity: <-p^4 / (8 m^3 c^2)>
   Relativistic kinetic energy correction.  Largest for inner shells.

2. Darwin term: <(pi Z alpha^2 / 2) delta^3(r)>
   Contact interaction from zitterbewegung.  s-orbitals only.

3. Spin-orbit: <(alpha^2 / 2)(1/r)(dV/dr) L*S>
   Splits l>0 shells into j = l +/- 1/2 subshells.

This was the standard approach before fully relativistic Dirac codes.
Each correction can be toggled independently for pedagogical comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from atomx.methods.base import Method
from atomx.types import CouplingScheme, NuclearModel, RelativisticCorrection, TheoryLevel


@dataclass
class BreitPauli(Method):
    """Non-relativistic HF + perturbative relativistic corrections.

    Parameters
    ----------
    corrections : list of RelativisticCorrection
        Which corrections to include.  Default: all three.
    coupling : CouplingScheme
        LS (default) or INTERMEDIATE (diagonalize with SO mixing).
    alpha_x : float
        Slater exchange parameter for the base NR-HF.
    """
    corrections: List[RelativisticCorrection] = field(
        default_factory=lambda: list(RelativisticCorrection))
    coupling: CouplingScheme = CouplingScheme.LS
    alpha_x: float = 1.0

    @property
    def theory_level(self) -> TheoryLevel:
        return TheoryLevel.BREIT_PAULI

    @property
    def include_mass_velocity(self) -> bool:
        return RelativisticCorrection.MASS_VELOCITY in self.corrections

    @property
    def include_darwin(self) -> bool:
        return RelativisticCorrection.DARWIN in self.corrections

    @property
    def include_spin_orbit(self) -> bool:
        return RelativisticCorrection.SPIN_ORBIT in self.corrections

    def solve(self, Z, subshells, tol=1e-5, max_iter=40, **kwargs):
        """Run NR-HF then apply perturbative relativistic corrections."""
        from atomx.constants import ALPHA
        from atomx.shooting.solver import ShootingSolver, ShootingResult

        # Step 1: Converge non-relativistic HF
        solver = ShootingSolver(
            Z=Z, subshells=subshells,
            n_grid=1000, r_max=75.0,
            alpha_x=self.alpha_x,
            equation='schrodinger')
        nr_result = solver.solve(max_iter=max_iter, tol=tol)

        # Step 2: Perturbative corrections
        alpha_sq = ALPHA ** 2
        corrected_energies = dict(nr_result.orbital_energies)
        delta_mv = 0.0
        delta_darwin = 0.0
        delta_so = 0.0
        components = dict(nr_result.energy_components)

        for sh in subshells:
            n, l = sh.n, sh.l
            l_char = "spdfghiklmno"[l] if l < 12 else f"l{l}"
            label_nr = f"{n}{l_char}"
            E_orb = nr_result.orbital_energies.get(label_nr, 0.0)
            q = sh.occupation
            Z_eff = Z

            # Mass-velocity
            if self.include_mass_velocity:
                E_mv = -alpha_sq * Z_eff**4 / (2 * n**3) * (
                    1.0 / (l + 0.5) - 3.0 / (4 * n))
                delta_mv += q * E_mv
            else:
                E_mv = 0.0

            # Darwin (s-waves only)
            if self.include_darwin and l == 0:
                E_d = alpha_sq * Z_eff**4 / (2 * n**3)
                delta_darwin += q * E_d
            else:
                E_d = 0.0

            # Spin-orbit (l > 0 only)
            if self.include_spin_orbit and l > 0:
                xi = alpha_sq * Z_eff**4 / (
                    4 * n**3 * l * (l + 0.5) * (l + 1))

                j2_plus = 2 * l + 1
                j2_minus = 2 * l - 1
                E_so_plus = xi * (l / 2.0)        # j = l + 1/2
                E_so_minus = xi * (-(l + 1) / 2.0)  # j = l - 1/2

                E_base = E_orb + E_mv + E_d
                corrected_energies[f"{n}{l_char}{j2_plus}/2"] = E_base + E_so_plus
                corrected_energies[f"{n}{l_char}{j2_minus}/2"] = E_base + E_so_minus
                del corrected_energies[label_nr]

                deg_plus, deg_minus = j2_plus + 1, j2_minus + 1
                E_so_avg = (deg_plus * E_so_plus + deg_minus * E_so_minus
                            ) / (deg_plus + deg_minus)
                delta_so += q * E_so_avg
            else:
                corrected_energies[label_nr] = E_orb + E_mv + E_d

        total_correction = delta_mv + delta_darwin + delta_so
        components["E_mass_velocity"] = delta_mv
        components["E_darwin"] = delta_darwin
        components["E_spin_orbit"] = delta_so
        components["E_relativistic_total"] = total_correction

        return ShootingResult(
            converged=nr_result.converged,
            iterations=nr_result.iterations,
            total_energy=nr_result.total_energy + total_correction,
            orbital_energies=corrected_energies,
            method="BreitPauli",
            energy_components=components,
            _equation="schrodinger",
        )
