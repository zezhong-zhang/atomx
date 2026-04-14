"""Exact hydrogen-like solutions: non-relativistic and relativistic.

These methods give closed-form analytical energies for one-electron
systems (H, He+, Li2+, ...).  They serve as pedagogical starting points
and numerical validation benchmarks.

Non-relativistic (Schrodinger/Bohr):
    E_n = -Z^2 / (2 n^2) Hartree
    Energy depends only on n — no l or j splitting.

Relativistic (Dirac/Sommerfeld):
    E_{n,kappa} = c^2 [1/sqrt(1 + (Z*alpha/(n_r + gamma))^2) - 1]
    where gamma = sqrt(kappa^2 - (Z*alpha)^2), n_r = n - |kappa|.
    Introduces fine structure: orbitals with the same n but different
    j (=|kappa|-1/2) have different energies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atomx.methods.base import Method
from atomx.types import NuclearModel, TheoryLevel


@dataclass
class NonRelativisticHydrogenic(Method):
    """Exact Schrodinger equation solution for hydrogen-like atoms.

    Uses the Bohr formula E_n = -Z^2 / (2 n^2) Hartree.

    Valid only for one-electron systems.  Demonstrates the n-degeneracy
    of the non-relativistic Coulomb problem: all orbitals with the same
    principal quantum number n have the same energy regardless of l.

    The radial wavefunctions R_{nl}(r) are computed from the associated
    Laguerre polynomials for visualization.
    """
    nuclear: NuclearModel = NuclearModel.POINT

    @property
    def theory_level(self) -> TheoryLevel:
        return TheoryLevel.HYDROGENIC

    def solve(self, Z, subshells, **kwargs):
        """Compute exact non-relativistic energies and wavefunctions."""
        from atomx.results import RadialWavefunctions, RadialOrbital

        orbital_energies = {}
        radial_grid = np.linspace(1e-6, 50.0 / Z, 2000)

        for sh in subshells:
            n, l = sh.n, sh.l
            bohr_energy = -Z ** 2 / (2.0 * n ** 2)
            l_char = "spdfghiklmno"[l] if l < 12 else f"l{l}"
            label = f"{n}{l_char}"
            orbital_energies[label] = bohr_energy

        total_energy = sum(
            sh.occupation * orbital_energies.get(
                f"{sh.n}{'spdfghiklmno'[sh.l] if sh.l < 12 else f'l{sh.l}'}",
                0.0)
            for sh in subshells)

        from atomx.scf.solver import SCFResult
        return SCFResult(
            converged=True, iterations=0, residual=0.0,
            total_energy=total_energy,
            orbital_energies=orbital_energies,
            subshells=subshells,
            energy_components={"E_kinetic_plus_nuclear": total_energy},
            _theory_level=TheoryLevel.HYDROGENIC.value,
            _method_name="NonRelativisticHydrogenic",
        )


@dataclass
class RelativisticHydrogenic(Method):
    """Exact Dirac equation solution for hydrogen-like atoms (Sommerfeld formula).

    Shows fine structure splitting: orbitals with the same principal quantum
    number n but different total angular momentum j have slightly different
    energies.  For example, 2s_{1/2} and 2p_{1/2} are degenerate (same j),
    but 2p_{3/2} differs (Dirac fine structure).

    Note: The Lamb shift (2s_{1/2} vs 2p_{1/2} splitting) is a QED effect
    NOT captured by the Dirac equation.
    """
    nuclear: NuclearModel = NuclearModel.POINT

    @property
    def theory_level(self) -> TheoryLevel:
        return TheoryLevel.HYDROGENIC

    def solve(self, Z, subshells, **kwargs):
        """Compute exact Dirac energies for hydrogen-like atom."""
        from atomx.constants import ALPHA, C

        orbital_energies = {}
        total_energy = 0.0

        for sh in subshells:
            n, kappa, l = sh.n, sh.kappa, sh.l
            j2 = 2 * abs(kappa) - 1

            # Sommerfeld fine-structure formula
            Z_alpha = Z * ALPHA
            gamma = np.sqrt(kappa ** 2 - Z_alpha ** 2)
            n_radial = n - abs(kappa)  # radial quantum number
            dirac_energy = C ** 2 * (
                1.0 / np.sqrt(1 + (Z_alpha / (n_radial + gamma)) ** 2)
                - 1.0)

            l_char = "spdfghiklmno"[l] if l < 12 else f"l{l}"
            label = f"{n}{l_char}{j2}/2"
            orbital_energies[label] = dirac_energy
            total_energy += sh.occupation * dirac_energy

        from atomx.scf.solver import SCFResult
        return SCFResult(
            converged=True, iterations=0, residual=0.0,
            total_energy=total_energy,
            orbital_energies=orbital_energies,
            subshells=subshells,
            energy_components={"E_kinetic_plus_nuclear": total_energy},
            _theory_level=TheoryLevel.HYDROGENIC.value,
            _method_name="RelativisticHydrogenic",
        )


# Backward-compatible aliases
HydrogenicSchrodinger = NonRelativisticHydrogenic
HydrogenicDirac = RelativisticHydrogenic


def _hydrogenic_radial_P(n, l, Z, r):
    """Compute normalized hydrogenic radial wavefunction P_{nl}(r) = r * R_{nl}(r).

    Uses the standard formula with associated Laguerre polynomials:
        R_{nl}(r) = N * (2Zr/n)^l * exp(-Zr/n) * L_{n-l-1}^{2l+1}(2Zr/n)
        P_{nl}(r) = r * R_{nl}(r)
    """
    from scipy.special import assoc_laguerre
    from math import factorial

    rho = 2.0 * Z * r / n
    normalization = np.sqrt(
        (2.0 * Z / n) ** 3 * factorial(n - l - 1)
        / (2.0 * n * factorial(n + l)))
    laguerre = assoc_laguerre(rho, n - l - 1, 2 * l + 1)
    return normalization * rho ** l * np.exp(-rho / 2.0) * laguerre * r
