"""Exact hydrogen-like solutions: Schrodinger and Dirac.

Non-relativistic (Schrodinger/Bohr):
    E_n = -Z^2 / (2 n^2) Hartree
    Energy depends only on n — no l or j splitting.

Relativistic (Dirac/Sommerfeld):
    E_{n,kappa} = c^2 [1/sqrt(1 + (Z*alpha/(n_r + gamma))^2) - 1]
    where gamma = sqrt(kappa^2 - (Z*alpha)^2), n_r = n - |kappa|.
    Introduces fine structure: same n, different j → different energies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atomx.methods.base import Method
from atomx.types import NuclearModel, TheoryLevel


@dataclass
class HydrogenicSchrodinger(Method):
    """Exact non-relativistic hydrogen-like solution (Bohr formula).

    E_n = -Z^2 / (2 n^2) Hartree.  Valid for one-electron systems
    (H, He+, Li2+, ...).  All orbitals with the same n are degenerate.
    """
    nuclear: NuclearModel = NuclearModel.POINT

    @property
    def theory_level(self) -> TheoryLevel:
        return TheoryLevel.HYDROGENIC

    def solve(self, Z, subshells, **kwargs):
        """Compute exact Schrodinger energies for hydrogen-like atom."""
        orbital_energies = {}
        for sh in subshells:
            n, l = sh.n, sh.l
            l_char = "spdfghiklmno"[l] if l < 12 else f"l{l}"
            orbital_energies[f"{n}{l_char}"] = -Z ** 2 / (2.0 * n ** 2)

        total_energy = sum(
            sh.occupation * orbital_energies.get(
                f"{sh.n}{'spdfghiklmno'[sh.l] if sh.l < 12 else f'l{sh.l}'}",
                0.0)
            for sh in subshells)

        from atomx.methods.dirac_fock import SCFResult
        from atomx.calculation_spec import CalculationSpec
        equation = 'dirac' if isinstance(self, HydrogenicDirac) else 'schrodinger'
        spec = CalculationSpec(float(Z), self.name, 'analytic',
            dict(equation=equation, nuclear='point', exchange='none',
                 rest_energy='subtracted', units='atomic'),
            dict(subshells=tuple((s.n, s.kappa, float(s.occupation)) for s in subshells)),
            {}, {})
        return SCFResult(
            calculation_spec=spec,
            Z=Z,
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

    Introduces fine structure: orbitals with same n but different j
    have different energies (e.g. 2s1/2 and 2p1/2 are degenerate,
    but 2p3/2 differs).

    Note: the Lamb shift (2s1/2 vs 2p1/2) is a QED effect not captured here.
    """
    nuclear: NuclearModel = NuclearModel.POINT

    @property
    def theory_level(self) -> TheoryLevel:
        return TheoryLevel.HYDROGENIC

    def solve(self, Z, subshells, **kwargs):
        """Compute exact Dirac energies for hydrogen-like atom."""
        from atomx.constants import ALPHA, C

        Z_alpha = Z * ALPHA
        orbital_energies = {}
        total_energy = 0.0

        for sh in subshells:
            n, kappa, l = sh.n, sh.kappa, sh.l
            j2 = 2 * abs(kappa) - 1

            # Sommerfeld fine-structure formula
            gamma = np.sqrt(kappa ** 2 - Z_alpha ** 2)
            n_radial = n - abs(kappa)
            E = C ** 2 * (
                1.0 / np.sqrt(1 + (Z_alpha / (n_radial + gamma)) ** 2)
                - 1.0)

            l_char = "spdfghiklmno"[l] if l < 12 else f"l{l}"
            label = f"{n}{l_char}{j2}/2"
            orbital_energies[label] = E
            total_energy += sh.occupation * E

        from atomx.methods.dirac_fock import SCFResult
        from atomx.calculation_spec import CalculationSpec
        equation = 'dirac' if isinstance(self, HydrogenicDirac) else 'schrodinger'
        spec = CalculationSpec(float(Z), self.name, 'analytic',
            dict(equation=equation, nuclear='point', exchange='none',
                 rest_energy='subtracted', units='atomic'),
            dict(subshells=tuple((s.n, s.kappa, float(s.occupation)) for s in subshells)),
            {}, {})
        return SCFResult(
            calculation_spec=spec,
            Z=Z,
            converged=True, iterations=0, residual=0.0,
            total_energy=total_energy,
            orbital_energies=orbital_energies,
            subshells=subshells,
            energy_components={"E_1e": total_energy},
            _theory_level=TheoryLevel.HYDROGENIC.value,
            _method_name="HydrogenicDirac",
        )
