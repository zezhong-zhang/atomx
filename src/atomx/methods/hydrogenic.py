"""Exact hydrogen-like solutions: Schrodinger and Dirac."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atomx.methods.base import Method
from atomx.types import CouplingScheme, NuclearModel, TheoryLevel


@dataclass
class HydrogenicSchrodinger(Method):
    """Exact non-relativistic hydrogen-like solution.

    Energy: E_n = -Z^2 / (2 n^2) Hartree (Bohr formula).

    Valid only for one-electron systems (H, He+, Li2+, ...).
    Demonstrates that without relativity, energies depend only on n
    (no l or j splitting).
    """
    nuclear: NuclearModel = NuclearModel.POINT

    @property
    def theory_level(self) -> TheoryLevel:
        return TheoryLevel.HYDROGENIC

    def solve(self, Z, subshells, **kwargs):
        """Compute exact Schrodinger energies for hydrogen-like atom."""
        from atomx.results import Wavefunctions, OrbitalWavefunction

        orbital_energies = {}
        wf_orbitals = {}
        # Standard grid for wavefunction evaluation
        grid = np.linspace(1e-6, 50.0 / Z, 2000)

        for sh in subshells:
            n = sh.n
            l = sh.l
            E_n = -Z ** 2 / (2.0 * n ** 2)
            l_char = "spdfghiklmno"[l] if l < 12 else f"l{l}"
            label = f"{n}{l_char}"
            orbital_energies[label] = E_n

            # Compute hydrogenic radial wavefunction R_{nl}(r) * r
            P = _hydrogenic_P(n, l, Z, grid)
            wf_orbitals[label] = OrbitalWavefunction(
                P=P, Q=None, n=n, l=l, kappa=None,
                occupation=sh.occupation, energy=E_n, label=label)

        total_energy = sum(
            sh.occupation * orbital_energies.get(
                f"{sh.n}{'spdfghiklmno'[sh.l] if sh.l < 12 else f'l{sh.l}'}",
                0.0)
            for sh in subshells)

        from atomx.scf.solver import SCFResult, Subshell
        return SCFResult(
            converged=True, iterations=0, residual=0.0,
            total_energy=total_energy,
            orbital_energies=orbital_energies,
            subshells=subshells,
            energy_components={"E_1e": total_energy},
            _theory_level=TheoryLevel.HYDROGENIC.value,
            _method_name="HydrogenicSchrodinger",
        )


@dataclass
class HydrogenicDirac(Method):
    """Exact relativistic hydrogen-like solution (Sommerfeld formula).

    Energy: E_{n,kappa} = c^2 (1/sqrt(1 + (Z*alpha)^2 / (n - |kappa| + sqrt(kappa^2 - (Z*alpha)^2))^2) - 1)

    Shows fine structure splitting: orbitals with same n but different kappa
    have different energies (e.g. 2s1/2 vs 2p1/2 vs 2p3/2).
    """
    nuclear: NuclearModel = NuclearModel.POINT

    @property
    def theory_level(self) -> TheoryLevel:
        return TheoryLevel.HYDROGENIC

    def solve(self, Z, subshells, **kwargs):
        """Compute exact Dirac energies for hydrogen-like atom."""
        from atomx.constants import ALPHA, C
        from atomx.results import Wavefunctions, OrbitalWavefunction

        Za = Z * ALPHA
        orbital_energies = {}
        total_energy = 0.0

        for sh in subshells:
            n = sh.n
            kappa = sh.kappa
            l = sh.l
            j2 = 2 * abs(kappa) - 1

            # Sommerfeld formula
            gamma = np.sqrt(kappa ** 2 - Za ** 2)
            nr = n - abs(kappa)  # radial quantum number
            denom = np.sqrt((nr + gamma) ** 2 + Za ** 2)
            E_dirac = C ** 2 * (1.0 / np.sqrt(1 + (Za / (nr + gamma)) ** 2) - 1.0)

            l_char = "spdfghiklmno"[l] if l < 12 else f"l{l}"
            label = f"{n}{l_char}{j2}/2"
            orbital_energies[label] = E_dirac
            total_energy += sh.occupation * E_dirac

        from atomx.scf.solver import SCFResult
        return SCFResult(
            converged=True, iterations=0, residual=0.0,
            total_energy=total_energy,
            orbital_energies=orbital_energies,
            subshells=subshells,
            energy_components={"E_1e": total_energy},
            _theory_level=TheoryLevel.HYDROGENIC.value,
            _method_name="HydrogenicDirac",
        )


def _hydrogenic_P(n, l, Z, r):
    """Compute normalized hydrogenic radial wavefunction P_{nl}(r) = r * R_{nl}(r).

    Uses the standard formula with associated Laguerre polynomials.
    """
    from scipy.special import assoc_laguerre
    rho = 2.0 * Z * r / n
    # Normalization
    from math import factorial
    norm = np.sqrt((2.0 * Z / n) ** 3 * factorial(n - l - 1)
                   / (2.0 * n * factorial(n + l)))
    L = assoc_laguerre(rho, n - l - 1, 2 * l + 1)
    P = norm * rho ** l * np.exp(-rho / 2.0) * L * r
    return P
