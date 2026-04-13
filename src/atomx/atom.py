"""High-level Atom API for AtomX.

Simple interface for common atomic structure calculations:

    from atomx import Atom

    ne = Atom(Z=10)                    # Neon
    ne.solve()                          # Run DHF SCF
    print(ne.total_energy)              # -128.04 Hartree
    print(ne.orbital_energies)          # {1s: -32.6, 2s: -1.6, ...}

    # With QED corrections:
    ne = Atom(Z=10, breit=True, qed=True)
    ne.solve()

    # Multi-configuration:
    be = Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"])
    be.solve()
    print(be.state_energies)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from atomx.constants import HARTREE_EV, HARTREE_CM1
from atomx.scf.solver import DFSolver, Subshell, neon_subshells, argon_subshells


# Standard ground-state configurations for Z=1..86
# d/f shells use nonrelativistic notation (auto-split into j-1/2, j+1/2)
_GROUND_CONFIGS = {
    1:  "1s1",
    2:  "1s2",
    3:  "1s2 2s1",
    4:  "1s2 2s2",
    5:  "1s2 2s2 2p1",
    6:  "1s2 2s2 2p2",
    7:  "1s2 2s2 2p3",
    8:  "1s2 2s2 2p4",
    9:  "1s2 2s2 2p5",
    10: "1s2 2s2 2p6",
    11: "1s2 2s2 2p6 3s1",
    12: "1s2 2s2 2p6 3s2",
    13: "1s2 2s2 2p6 3s2 3p1",
    14: "1s2 2s2 2p6 3s2 3p2",
    15: "1s2 2s2 2p6 3s2 3p3",
    16: "1s2 2s2 2p6 3s2 3p4",
    17: "1s2 2s2 2p6 3s2 3p5",
    18: "1s2 2s2 2p6 3s2 3p6",
    19: "1s2 2s2 2p6 3s2 3p6 4s1",
    20: "1s2 2s2 2p6 3s2 3p6 4s2",
    21: "1s2 2s2 2p6 3s2 3p6 3d1 4s2",
    22: "1s2 2s2 2p6 3s2 3p6 3d2 4s2",
    23: "1s2 2s2 2p6 3s2 3p6 3d3 4s2",
    24: "1s2 2s2 2p6 3s2 3p6 3d5 4s1",  # Cr exception
    25: "1s2 2s2 2p6 3s2 3p6 3d5 4s2",
    26: "1s2 2s2 2p6 3s2 3p6 3d6 4s2",
    27: "1s2 2s2 2p6 3s2 3p6 3d7 4s2",
    28: "1s2 2s2 2p6 3s2 3p6 3d8 4s2",
    29: "1s2 2s2 2p6 3s2 3p6 3d10 4s1", # Cu exception
    30: "1s2 2s2 2p6 3s2 3p6 3d10 4s2",
    31: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p1",
    32: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p2",
    33: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p3",
    34: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p4",
    35: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p5",
    36: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6",
    37: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 5s1",
    38: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 5s2",
    39: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d1 5s2",
    40: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d2 5s2",
    41: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d4 5s1",
    42: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d5 5s1",
    43: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d5 5s2",
    44: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d7 5s1",
    45: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d8 5s1",
    46: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10",
    47: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s1",
    48: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2",
    49: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p1",
    50: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p2",
    51: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p3",
    52: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p4",
    53: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p5",
    54: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6",
    55: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 6s1",
    56: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 6s2",
    57: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 5d1 6s2",
    71: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d1 6s2",
    72: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d2 6s2",
    73: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d3 6s2",
    74: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d4 6s2",
    75: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d5 6s2",
    76: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d6 6s2",
    77: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d7 6s2",
    78: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d9 6s1",
    79: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s1",
    80: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s2",
    81: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s2 6p1",
    82: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s2 6p2",
    83: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s2 6p3",
    84: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s2 6p4",
    85: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s2 6p5",
    86: "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14 5d10 6s2 6p6",
}


class Atom:
    """High-level interface for atomic structure calculations.

    Parameters
    ----------
    Z : int
        Nuclear charge.
    config : str, optional
        Electron configuration (e.g., "1s2 2s2 2p6").
        Default: ground state from built-in table.
    configs : list of str, optional
        Multiple configurations for MCDHF.
    charge : int, optional
        Ion charge (0=neutral, 1=singly ionized, etc.). Default: 0.
    ns : int
        Number of B-splines.
    ks : int
        B-spline order.
    breit : bool
        Include Breit (Gaunt) interaction.
    qed : bool
        Include QED (VP + SE) corrections.
    exchange : str
        "hf" for B-spline Hartree-Fock (default),
        "slater" for B-spline local Slater exchange,
        "shooting" for ODE shooting with Slater exchange (Radial-style).
    """

    def __init__(self, Z, config=None, configs=None, charge=0,
                 ns=None, ks=9, breit=False, qed=False, exchange="hf",
                 correlation=None, starting_potential=None,
                 plasma_ne=None, plasma_Te=None):
        """
        Parameters
        ----------
        correlation : str or None
            None = single-config DHF (default).
            "sd" = add singles+doubles excitations (MCDHF with correlation).
            "s" = singles only.
        exchange : str
            "hf" for Hartree-Fock, "slater" for local Slater exchange,
            "tfda" for Thomas-Fermi-Dirac-Amaldi (fast, no SCF),
            "gsz" for Green-Sellin-Zachor parametric potential.
        starting_potential : str or None
            "tfda" or "gsz" for faster SCF convergence.
        plasma_ne : float, optional
            Electron density (cm^-3) for plasma screening.
        plasma_Te : float, optional
            Electron temperature (eV) for plasma screening.
        """
        self.Z = Z
        self.charge = charge
        # Unified physics-based grid for all Z
        self._use_physics_grid = (ns is None)
        if ns is None:
            ns = 113  # will be overridden by physics grid
        self.ns = ns
        self.ks = ks
        self.breit = breit
        self.qed = qed
        self.exchange = exchange
        self.correlation = correlation
        self.starting_potential = starting_potential
        self.plasma_ne = plasma_ne
        self.plasma_Te = plasma_Te

        # Determine configuration(s)
        if configs is not None:
            self._multi_config = True
            self._config_strings = configs
        elif correlation is not None:
            # Auto-generate correlation CSFs
            self._multi_config = True
            N_electrons = Z - charge
            base_config = config or _GROUND_CONFIGS.get(N_electrons)
            if base_config is None:
                raise ValueError(f"No built-in config for {N_electrons} electrons.")
            from atomx.mcdhf.excitations import generate_sd_excitations, suggest_virtual_orbitals
            virtuals = suggest_virtual_orbitals(Z, n_virtual=1)
            max_exc = 2 if correlation == "sd" else 1
            # Freeze core for Z > 10
            core = None
            if Z > 10:
                core = ["1s", "2s", "2p"]
            elif Z > 4:
                core = ["1s"]
            self._config_strings = generate_sd_excitations(
                base_config, virtuals, core_orbitals=core, max_excitations=max_exc)
        else:
            self._multi_config = False
            if config is None:
                N_electrons = Z - charge
                config = _GROUND_CONFIGS.get(N_electrons)
                if config is None:
                    raise ValueError(f"No built-in config for {N_electrons} electrons. "
                                     f"Provide config= explicitly.")
            self._config_strings = [config]

        # Results (populated after solve())
        self._result = None
        self._mcdhf_result = None
        self._bsp_data = None

    def solve(self, tol=1e-5, max_iter=40):
        """Run the self-consistent field calculation.

        Parameters
        ----------
        tol : float
            Convergence tolerance.
        max_iter : int
            Maximum SCF iterations.

        Returns
        -------
        self (for chaining)
        """
        if self._multi_config:
            from atomx.mcdhf.config import make_csf_list
            from atomx.mcdhf.solver import MCDHFSolver
            csfs = make_csf_list(self.Z, self._config_strings)
            solver = MCDHFSolver(Z=self.Z, csfs=csfs, ns=self.ns, ks=self.ks,
                                  include_breit=self.breit, include_qed=self.qed)
            self._mcdhf_result = solver.solve(max_iter=max_iter, scf_tol=tol)
            self._bsp_data = self._mcdhf_result.bsp_data
        elif self.exchange == "shooting":
            # ODE shooting backend (Radial/Salvat-style)
            from atomx.mcdhf.config import _parse_config_string
            from atomx.shooting.solver import ShootingSolver
            subshells = _parse_config_string(self._config_strings[0])
            solver = ShootingSolver(Z=self.Z, subshells=subshells,
                                    n_grid=1000, r_max=75.0, alpha_x=1.5)
            shooting_result = solver.solve(max_iter=max_iter, tol=tol)
            # Wrap in a compatible result object
            from atomx.scf.solver import SCFResult, Subshell
            self._result = SCFResult(
                converged=shooting_result.converged,
                iterations=shooting_result.iterations,
                residual=0.0,
                total_energy=shooting_result.total_energy,
                orbital_energies=shooting_result.orbital_energies,
                subshells=subshells,
                energy_components={
                    "E_1e": 0.0,
                    "E_J": 0.0,
                    "E_K": 0.0,
                },
            )
        else:
            # B-spline DHF/DFS backend
            from atomx.mcdhf.config import _parse_config_string
            subshells = _parse_config_string(self._config_strings[0])
            solver = DFSolver(Z=self.Z, subshells=subshells,
                              ns=self.ns, ks=self.ks,
                              exchange=self.exchange,
                              include_breit=self.breit,
                              include_qed=self.qed,
                              starting_potential=self.starting_potential)

            if self._use_physics_grid:
                from atomx.bspline.knots import make_physics_knots
                from atomx.dirac.matrix import prepare_bspline_data
                t_phys, ns_phys = make_physics_knots(
                    ks=self.ks, Z=self.Z, ns=113)
                data = prepare_bspline_data(t_phys, ns_phys, self.ks)
                solver.t = t_phys
                solver.bsp_data = data
                solver.ns = ns_phys
                solver._precompute()

            self._result = solver.solve(max_iter=max_iter, tol=tol)
            self._bsp_data = solver.bsp_data

        return self

    @property
    def converged(self):
        if self._mcdhf_result:
            return self._mcdhf_result.converged
        return self._result.converged if self._result else False

    @property
    def total_energy(self):
        """Total energy in Hartree."""
        if self._mcdhf_result:
            return self._mcdhf_result.ground_energy
        return self._result.total_energy if self._result else None

    @property
    def total_energy_eV(self):
        E = self.total_energy
        return E * HARTREE_EV if E is not None else None

    @property
    def orbital_energies(self):
        """Dict of orbital energies {label: energy_hartree}."""
        if self._mcdhf_result:
            return self._mcdhf_result.orbital_energies
        return self._result.orbital_energies if self._result else {}

    @property
    def energy_components(self):
        """Energy breakdown dict."""
        if self._result:
            return self._result.energy_components
        return {}

    @property
    def state_energies(self):
        """CI state energies (MCDHF only)."""
        if self._mcdhf_result:
            return self._mcdhf_result.state_energies
        return np.array([self.total_energy]) if self.total_energy else None

    def excitation_energies(self, unit="cm-1"):
        """Excitation energies relative to ground state."""
        if self._mcdhf_result:
            return self._mcdhf_result.excitation_energies(unit)
        return np.array([0.0])

    def transitions(self, multipoles=None, gauge='length'):
        """Compute radiative transitions between atomic states.

        Parameters
        ----------
        multipoles : list of str, optional
            Multipole types: 'E1', 'M1', 'E2', 'M2', etc. Default: ['E1'].
        gauge : str
            'length' (Babushkin) or 'velocity' (Coulomb).

        Returns
        -------
        list of TransitionResult
            Transitions sorted by wavelength.
        """
        from atomx.transitions.calculator import TransitionCalculator
        calc = TransitionCalculator(self)
        return calc.compute_all(multipoles, gauge)

    def line_list(self, multipoles=None, gauge='length', unit='angstrom'):
        """Print a formatted line list of transitions.

        Parameters
        ----------
        multipoles : list of str, optional
        gauge : str
        unit : str
            'angstrom', 'nm', 'eV', or 'cm-1'.
        """
        transitions = self.transitions(multipoles, gauge)
        if not transitions:
            print("No allowed transitions found.")
            return transitions

        print(f"{'Transition':>25s}  {'Type':>4s}  {'Wavelength':>12s}  "
              f"{'A (s^-1)':>12s}  {'gf':>10s}")
        print("-" * 70)
        for tr in transitions:
            label = f"{tr.initial_label} -> {tr.final_label}"
            if unit == 'nm':
                wl = f"{tr.wavelength_nm:.3f} nm"
            elif unit == 'eV':
                wl = f"{tr.energy_eV:.4f} eV"
            elif unit == 'cm-1':
                wl = f"{tr.energy_cm1:.1f} cm-1"
            else:
                wl = f"{tr.wavelength_angstrom:.3f} A"
            print(f"{label:>25s}  {tr.multipole:>4s}  {wl:>12s}  "
                  f"{tr.A_rate_per_s:>12.4e}  {tr.gf_value:>10.4e}")

        return transitions

    @property
    def iterations(self):
        if self._mcdhf_result:
            return self._mcdhf_result.iterations
        return self._result.iterations if self._result else 0

    def excitation_cross_section(self, final_atom, energies_eV, **kwargs):
        """Compute electron-impact excitation cross-section (DW approximation).

        Parameters
        ----------
        final_atom : Atom
            Solved excited-state atom.
        energies_eV : array_like
            Incident electron energies in eV.

        Returns
        -------
        sigma : ndarray
            Cross-section in cm^2.
        omega : ndarray
            Collision strength (dimensionless).
        """
        from atomx.collisions.distorted_wave import dw_excitation_cross_section
        return dw_excitation_cross_section(self, final_atom, energies_eV, **kwargs)

    def __repr__(self):
        sym = _element_symbol(self.Z)
        charge_str = ""
        if self.charge > 0:
            charge_str = f"^{self.charge}+"
        elif self.charge < 0:
            charge_str = f"^{abs(self.charge)}-"
        status = "solved" if self.converged else "unsolved"
        return f"Atom({sym}{charge_str}, Z={self.Z}, {status})"

    def summary(self):
        """Print a summary of the calculation."""
        sym = _element_symbol(self.Z)
        print(f"{'='*50}")
        print(f"  {sym} (Z={self.Z})  AtomX DHF Calculation")
        print(f"{'='*50}")
        print(f"  Config: {self._config_strings[0]}")
        print(f"  Converged: {self.converged} ({self.iterations} iterations)")
        print(f"  Total energy: {self.total_energy:.6f} Ha"
              f" ({self.total_energy_eV:.2f} eV)")
        print(f"\n  Orbital energies:")
        for label, E in sorted(self.orbital_energies.items(),
                                key=lambda x: x[1]):
            print(f"    {label:8s}: {E:12.6f} Ha ({E*HARTREE_EV:10.2f} eV)")

        if self.breit or self.qed:
            print(f"\n  Corrections:")
            ec = self.energy_components
            for key in ["E_Breit", "E_VP", "E_SE"]:
                if key in ec:
                    print(f"    {key:8s}: {ec[key]:12.2e} Ha"
                          f" ({ec[key]*HARTREE_CM1:10.1f} cm-1)")

        if self._multi_config and self._mcdhf_result:
            print(f"\n  CI States:")
            for i, E in enumerate(self.state_energies):
                de = (E - self.state_energies[0]) * HARTREE_CM1
                print(f"    State {i}: {E:.6f} Ha  (ΔE = {de:.0f} cm-1)")


def _element_symbol(Z):
    """Element symbol from atomic number."""
    syms = [
        "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "In", "Sn", "Sb", "Te", "I", "Xe",
        "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
        "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    ]
    return syms[Z] if Z < len(syms) else f"Z{Z}"
