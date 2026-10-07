"""Relativistic corrections to Hartree-Fock (Breit-Pauli / Cowan-Griffin).

Uses an HF–Slater reference and hydrogenic first-order shifts evaluated at Z.
These shifts are not expectation values on the converged many-electron orbitals.
The implemented terms are:

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

from atomx.config.subshell import l_letter
from atomx.methods.base import Method
from atomx.types import (
    CouplingScheme,
    RelativisticCorrection,
    TheoryLevel,
)


@dataclass
class BreitPauli(Method):
    """Non-relativistic HF + perturbative relativistic corrections.

    Parameters
    ----------
    corrections : list of RelativisticCorrection
        Which corrections to include.  Default: all three.
    coupling : CouplingScheme
        LS only; intermediate-coupling diagonalization is not implemented.
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

    def solve(self, Z, subshells, tol=1e-5, max_iter=40, *,
              n_grid=1000, r_max=75.0, r_inner=1e-7, screening_potential=None):
        """Run NR-HF then apply perturbative relativistic corrections.

        Output ``orbital_energies`` key convention
        ------------------------------------------
        Keys are jj subshell labels (``"{n}{l}{2j}/2"``, e.g. ``"1s1/2"``,
        ``"2p3/2"``) — the same labelling the NR :class:`ShootingSolver`
        uses for its base result:

        - ``l = 0`` (or spin-orbit disabled): each input jj key is
          updated **in place** with the corrected energy; no extra
          NR-style key (``"1s"``) is created.
        - ``l > 0`` with spin-orbit enabled: exactly two jj keys,
          ``"{n}{l}{2l-1}/2"`` and ``"{n}{l}{2l+1}/2"``, holding the
          corrected energies of the ``j = l - 1/2`` / ``j = l + 1/2``
          fine-structure levels.  Both share one NR base energy, so
          their difference is exactly the spin-orbit splitting.  The
          input key(s) of the shell — including any user-customised
          ``Subshell.label`` — are removed, so no uncorrected NR energy
          survives under a stale key.

        When the input lists several subshells sharing the same
        ``(n, l)`` (e.g. ``2p1/2`` + ``2p3/2`` from a ``2p6`` token),
        they form one NR shell: the shared base energy is the
        occupation-weighted average of their NR orbital energies (the
        Breit-Pauli shifts depend only on ``(n, l)``), and the two
        output jj keys are written once.

        Raises
        ------
        KeyError
            If a subshell's NR orbital energy is missing from the base
            result — there is no silent zero fallback.
        """
        if self.coupling != CouplingScheme.LS:
            raise NotImplementedError('this local-reference method supports LS coupling only')
        from atomx.numerics.shooting.local import ShootingSolver
        from atomx.physics.breit_pauli import (
            darwin_shift,
            mass_velocity_shift,
            spin_orbit_shifts,
            spin_orbit_xi,
        )

        # Step 1: Converge non-relativistic HF
        solver = ShootingSolver(
            Z=Z, subshells=subshells,
            n_grid=n_grid, r_max=r_max, r_inner=r_inner,
            nuclear=getattr(self.nuclear, "value", self.nuclear),
            screening_potential=screening_potential,
            alpha_x=self.alpha_x,
            equation='schrodinger')
        nr_result = solver.solve(max_iter=max_iter, tol=tol)

        # Step 2: Perturbative corrections via physics/breit_pauli operators
        corrected_energies = dict(nr_result.orbital_energies)
        delta_mv = 0.0
        delta_darwin = 0.0
        delta_so = 0.0
        components = dict(nr_result.energy_components)

        # Group subshells by (n, l): jj-split inputs of one NR shell
        # (e.g. "2p6" -> 2p1/2 + 2p3/2) share a single NR base energy.
        # Resolve each base energy via the subshell's own jj label —
        # the key the NR solver actually wrote — never a synthesized
        # NR label with a silent 0.0 fallback.
        groups = []  # [{"n", "l", "members": [(subshell, E_orb)]}]
        group_index = {}
        for sh in subshells:
            if sh.label not in nr_result.orbital_energies:
                raise KeyError(
                    f"NR base energy for subshell {sh.label!r} is missing "
                    f"from the ShootingSolver result; available keys: "
                    f"{sorted(nr_result.orbital_energies)}")
            key = (sh.n, sh.l)
            member = (sh, nr_result.orbital_energies[sh.label])
            if key in group_index:
                groups[group_index[key]]["members"].append(member)
            else:
                group_index[key] = len(groups)
                groups.append({"n": sh.n, "l": sh.l, "members": [member]})

        for group in groups:
            n, l, members = group["n"], group["l"], group["members"]
            l_char = l_letter(l)
            q_tot = sum(sh.occupation for sh, _ in members)
            Z_eff = Z

            E_mv = mass_velocity_shift(n, l, Z_eff) if self.include_mass_velocity else 0.0
            delta_mv += q_tot * E_mv

            E_d = darwin_shift(n, l, Z_eff) if self.include_darwin else 0.0
            delta_darwin += q_tot * E_d

            if self.include_spin_orbit and l > 0:
                # Occupation-weighted NR base shared by both j levels.
                E_orb = (sum(sh.occupation * E for sh, E in members) / q_tot
                         if q_tot > 0 else 0.0)
                xi = spin_orbit_xi(n, l, Z_eff)
                E_so_plus, E_so_minus = spin_orbit_shifts(l, xi)

                j2_plus = 2 * l + 1
                j2_minus = 2 * l - 1
                E_base = E_orb + E_mv + E_d
                # Replace the real NR jj key(s) of this (n, l) shell with
                # the two corrected fine-structure levels.  Pop each
                # member's own label — the key the NR solver actually
                # wrote — because users may override ``Subshell.label``,
                # in which case no synthesized jj key exists to pop.
                for sh, _ in members:
                    corrected_energies.pop(sh.label, None)
                corrected_energies[f"{n}{l_char}{j2_plus}/2"] = E_base + E_so_plus
                corrected_energies[f"{n}{l_char}{j2_minus}/2"] = E_base + E_so_minus

                # First-order SO contribution to the total energy is the
                # occupation-weighted sum of each member's jj shift.  A
                # (2j+1)-weighted statistical average of the two shifts
                # cancels identically for any l, so it cannot be used here
                # (it is only correct for statistically filled shells).
                delta_so += sum(
                    sh.occupation * (E_so_plus if sh.j2 == j2_plus else E_so_minus)
                    for sh, _ in members
                )
            else:
                # l = 0 (or spin-orbit off): update each jj key in place.
                for sh, E_orb in members:
                    corrected_energies[sh.label] = E_orb + E_mv + E_d

        total_correction = delta_mv + delta_darwin + delta_so
        components["E_mass_velocity"] = delta_mv
        components["E_darwin"] = delta_darwin
        components["E_spin_orbit"] = delta_so
        components["E_relativistic_total"] = total_correction

        from dataclasses import replace
        spec = replace(nr_result.calculation_spec, method='BreitPauli',
            corrections=dict(mass_velocity=self.include_mass_velocity,
                             darwin=self.include_darwin, spin_orbit=self.include_spin_orbit,
                             operator_model='hydrogenic effective-charge perturbation'))
        return replace(nr_result,
            calculation_spec=spec,
            total_energy=nr_result.total_energy + total_correction,
            orbital_energies=corrected_energies,
            method="BreitPauli",
            energy_components=components,
        )
