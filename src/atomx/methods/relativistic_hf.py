"""Breit-Pauli perturbative relativistic corrections to Hartree-Fock.

Adds first-order perturbative corrections to converged NR-HF orbitals
using the Breit-Pauli Hamiltonian (also known as the Cowan-Griffin approach):

1. Mass-velocity: <-p^4 / (8 m^3 c^2)>
   Relativistic kinetic energy correction.  Affects all orbitals,
   largest for inner shells where electrons move fastest.

2. Darwin term: <(pi Z alpha^2 / 2) delta^3(r)>
   Contact interaction arising from electron zitterbewegung.
   Only non-zero for s-orbitals (l=0) which have |psi(0)|^2 != 0.

3. Spin-orbit: <(alpha^2 / 2)(1/r)(dV/dr)> * <L*S>
   Lifts the l-degeneracy by splitting shells with l>0 into
   j = l + 1/2 and j = l - 1/2 subshells.

Historical context: This approach was the standard way to include
relativistic effects before fully relativistic (Dirac) codes became
practical in the 1970s.  It remains useful pedagogically because
each correction can be turned on/off independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from atomx.methods.base import Method
from atomx.types import (
    CouplingScheme, NuclearModel, RelativisticCorrection, TheoryLevel,
)


@dataclass
class BreitPauli(Method):
    """Non-relativistic Hartree-Fock + perturbative relativistic corrections.

    Starts from a converged NR-HF solution (Schrodinger equation), then
    applies first-order energy corrections from the Breit-Pauli Hamiltonian.

    Students can turn on each correction independently to see its effect:

        # Scalar relativistic only (mass-velocity + Darwin)
        atom.solve(BreitPauli(corrections=[
            RelativisticCorrection.MASS_VELOCITY,
            RelativisticCorrection.DARWIN,
        ]))

        # All corrections including spin-orbit
        atom.solve(BreitPauli())  # default: all three

    Parameters
    ----------
    corrections : list of RelativisticCorrection
        Which corrections to include.  Default: all three.
    coupling : CouplingScheme
        LS (default) or INTERMEDIATE (diagonalize with SO mixing).
    slater_exchange_alpha : float
        Slater exchange parameter for the base NR-HF calculation.
        alpha=1.0 corresponds to the free-electron gas value.
    """
    corrections: List[RelativisticCorrection] = field(
        default_factory=lambda: list(RelativisticCorrection))
    coupling: CouplingScheme = CouplingScheme.LS
    slater_exchange_alpha: float = 1.0

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

        # Step 1: Converge non-relativistic Hartree-Fock
        solver = ShootingSolver(
            Z=Z, subshells=subshells,
            n_grid=1000, r_max=75.0,
            alpha_x=self.slater_exchange_alpha,
            equation='schrodinger')
        nr_result = solver.solve(max_iter=max_iter, tol=tol)

        # Step 2: Compute perturbative corrections to orbital energies
        alpha_sq = ALPHA ** 2
        corrected_energies = dict(nr_result.orbital_energies)
        delta_mass_velocity = 0.0
        delta_darwin = 0.0
        delta_spin_orbit = 0.0
        energy_components = dict(nr_result.energy_components)

        for sh in subshells:
            n, l = sh.n, sh.l
            l_char = "spdfghiklmno"[l] if l < 12 else f"l{l}"
            label_nr = f"{n}{l_char}"
            epsilon_nr = nr_result.orbital_energies.get(label_nr, 0.0)
            q = sh.occupation

            # Hydrogenic-scaled corrections (Z_eff = Z for now)
            Z_eff = Z

            # --- Mass-velocity: -(alpha^2 Z^4) / (2 n^3) (1/(l+1/2) - 3/(4n)) ---
            if self.include_mass_velocity:
                E_mv = -alpha_sq * Z_eff**4 / (2 * n**3) * (
                    1.0 / (l + 0.5) - 3.0 / (4 * n))
                delta_mass_velocity += q * E_mv
            else:
                E_mv = 0.0

            # --- Darwin: (alpha^2 Z^4) / (2 n^3) for l=0 only ---
            if self.include_darwin and l == 0:
                E_darwin = alpha_sq * Z_eff**4 / (2 * n**3)
                delta_darwin += q * E_darwin
            else:
                E_darwin = 0.0

            # --- Spin-orbit: splits l>0 into j = l +/- 1/2 ---
            if self.include_spin_orbit and l > 0:
                # Spin-orbit parameter
                xi_nl = alpha_sq * Z_eff**4 / (
                    4 * n**3 * l * (l + 0.5) * (l + 1))

                # j = l + 1/2:  <L*S> = +l/2
                E_so_plus = xi_nl * (l / 2.0)
                j2_plus = 2 * l + 1
                label_plus = f"{n}{l_char}{j2_plus}/2"

                # j = l - 1/2:  <L*S> = -(l+1)/2
                E_so_minus = xi_nl * (-(l + 1) / 2.0)
                j2_minus = 2 * l - 1
                label_minus = f"{n}{l_char}{j2_minus}/2"

                # Apply scalar + SO to split levels
                E_scalar = epsilon_nr + E_mv + E_darwin
                corrected_energies[label_plus] = E_scalar + E_so_plus
                corrected_energies[label_minus] = E_scalar + E_so_minus
                del corrected_energies[label_nr]

                # Degeneracy-weighted average for total energy
                deg_plus = j2_plus + 1
                deg_minus = j2_minus + 1
                E_so_avg = (deg_plus * E_so_plus + deg_minus * E_so_minus
                            ) / (deg_plus + deg_minus)
                delta_spin_orbit += q * E_so_avg
            else:
                corrected_energies[label_nr] = epsilon_nr + E_mv + E_darwin

        # Assemble total correction
        delta_relativistic = delta_mass_velocity + delta_darwin + delta_spin_orbit
        total_energy = nr_result.total_energy + delta_relativistic

        energy_components["E_mass_velocity"] = delta_mass_velocity
        energy_components["E_darwin"] = delta_darwin
        energy_components["E_spin_orbit"] = delta_spin_orbit
        energy_components["E_relativistic_total"] = delta_relativistic

        return ShootingResult(
            converged=nr_result.converged,
            iterations=nr_result.iterations,
            total_energy=total_energy,
            orbital_energies=corrected_energies,
            method="BreitPauli",
            energy_components=energy_components,
            _equation="schrodinger",
        )


# Backward-compatible alias
RelativisticHartreeFock = BreitPauli
