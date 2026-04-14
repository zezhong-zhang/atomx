"""Comparison and pedagogical tools for AtomX.

Side-by-side comparison of results from different theory levels,
automatic theory ladder computation, and plain-English explanations
of what physics each correction adds.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from atomx.constants import HARTREE_EV, HARTREE_CM1
from atomx.types import EnergyUnit, RelativisticCorrection, TheoryLevel


# ---------------------------------------------------------------------------
# compare: side-by-side table
# ---------------------------------------------------------------------------

def compare(*results, unit: EnergyUnit = EnergyUnit.HARTREE):
    """Print side-by-side comparison of results from different theory levels.

    Parameters
    ----------
    *results
        Two or more result objects satisfying the AtomicResult protocol.
    unit : EnergyUnit
        Energy unit for display.

    Example
    -------
    >>> r1 = atom.solve(HartreeFock())
    >>> r2 = atom.solve(DiracFock())
    >>> compare(r1, r2)
    """
    if len(results) < 2:
        print("Need at least 2 results to compare.")
        return

    conv = _unit_factor(unit)
    unit_str = unit.value

    # Column headers
    names = [_get_method_name(r) for r in results]
    col_w = max(14, max(len(n) for n in names) + 2)

    # Header
    header = f"{'Property':>25s}"
    for name in names:
        header += f"  {name:>{col_w}s}"
    print(header)
    print("-" * len(header))

    # Total energy
    row = f"{'Total energy (' + unit_str + ')':>25s}"
    for r in results:
        E = r.total_energy * conv
        row += f"  {E:>{col_w}.6f}"
    print(row)

    # Delta from previous
    row = f"{'Delta from previous':>25s}"
    row += f"  {'---':>{col_w}s}"
    for i in range(1, len(results)):
        dE = (results[i].total_energy - results[i - 1].total_energy) * conv
        row += f"  {dE:>{col_w}.6f}"
    print(row)

    # Theory level
    row = f"{'Theory level':>25s}"
    for r in results:
        lvl = _get_theory_level(r)
        row += f"  {lvl:>{col_w}s}"
    print(row)

    # Coupling scheme
    row = f"{'Coupling':>25s}"
    for r in results:
        cs = _get_coupling(r)
        row += f"  {cs:>{col_w}s}"
    print(row)

    print()

    # Orbital energies
    all_labels = set()
    for r in results:
        all_labels.update(r.orbital_energies.keys())
    sorted_labels = sorted(all_labels,
                           key=lambda s: (int(s[0]), s[1:]))

    if sorted_labels:
        print(f"{'Orbital energies (' + unit_str + ')':>25s}")
        for label in sorted_labels:
            row = f"  {label:>23s}"
            for r in results:
                E = r.orbital_energies.get(label)
                if E is not None:
                    row += f"  {E * conv:>{col_w}.6f}"
                else:
                    row += f"  {'---':>{col_w}s}"
            print(row)

    # Energy breakdown
    all_keys = set()
    for r in results:
        bd = _get_breakdown(r)
        all_keys.update(bd.keys())
    correction_keys = [k for k in sorted(all_keys)
                       if k not in ("E_1e", "E_total", "E_QED_total")]

    if correction_keys:
        print()
        print(f"{'Energy breakdown (' + unit_str + ')':>25s}")
        for key in correction_keys:
            row = f"  {key:>23s}"
            for r in results:
                bd = _get_breakdown(r)
                val = bd.get(key)
                if val is not None and abs(val) > 1e-15:
                    row += f"  {val * conv:>{col_w}.6f}"
                else:
                    row += f"  {'---':>{col_w}s}"
            print(row)


# ---------------------------------------------------------------------------
# theory_ladder: automatic progressive calculation
# ---------------------------------------------------------------------------

def theory_ladder(Z: int, config: str = None) -> list:
    """Run all applicable theory levels and return a list of results.

    Automatically selects methods appropriate for the given Z:
    - Z=1: HydrogenicSchrodinger, HydrogenicDirac
    - Z>1: HartreeFock, BreitPauli(scalar),
            BreitPauli(all), DiracFock
    - Z>=20: adds DiracFock(breit=True)
    - Z>=36: adds DiracFock(breit=True, qed=True)

    Parameters
    ----------
    Z : int
        Atomic number.
    config : str, optional
        Electron configuration.  Uses ground state if not given.

    Returns
    -------
    list of results
        Results from each theory level, ordered by increasing complexity.
    """
    from atomx.atom import Atom
    from atomx.methods import (
        HydrogenicSchrodinger, HydrogenicDirac,
        HartreeFock, BreitPauli,
        DiracFock,
    )

    atom = Atom(Z=Z, config=config) if config else Atom(Z=Z)
    results = []

    if Z == 1:
        results.append(atom.solve(HydrogenicSchrodinger()))
        results.append(atom.solve(HydrogenicDirac()))
    else:
        # Non-relativistic HF
        try:
            results.append(atom.solve(HartreeFock()))
        except Exception:
            pass

        # Scalar relativistic corrections
        try:
            results.append(atom.solve(
                BreitPauli(corrections=[
                    RelativisticCorrection.MASS_VELOCITY,
                    RelativisticCorrection.DARWIN])))
        except Exception:
            pass

        # Full relativistic corrections (+ spin-orbit)
        try:
            results.append(atom.solve(BreitPauli()))
        except Exception:
            pass

        # Full Dirac-Fock
        results.append(atom.solve(DiracFock()))

        # Breit for heavier atoms
        if Z >= 20:
            results.append(atom.solve(DiracFock(breit=True)))

        # QED for heavy atoms
        if Z >= 36:
            results.append(atom.solve(DiracFock(breit=True, qed=True)))

    return results


# ---------------------------------------------------------------------------
# what_changed: plain-English explanation
# ---------------------------------------------------------------------------

_EXPLANATIONS = {
    (TheoryLevel.HYDROGENIC, TheoryLevel.HYDROGENIC):
        "Comparing non-relativistic (Schrodinger) with relativistic (Dirac) "
        "hydrogen-like solutions.  The Dirac equation introduces fine structure: "
        "states with the same n but different j have slightly different energies.",

    (TheoryLevel.HARTREE_FOCK, TheoryLevel.BREIT_PAULI):
        "Adding perturbative relativistic corrections to non-relativistic HF:\n"
        "  - Mass-velocity: p^4 correction to kinetic energy (all orbitals)\n"
        "  - Darwin term: s-wave contact interaction (s-orbitals only)\n"
        "  - Spin-orbit: L·S coupling splits l>0 shells into j=l±1/2",

    (TheoryLevel.BREIT_PAULI, TheoryLevel.DIRAC_HARTREE_FOCK):
        "Switching from perturbative corrections to the full Dirac equation.\n"
        "The Dirac equation treats relativity variationally (not perturbatively), "
        "giving more accurate results especially for heavy atoms where alpha*Z "
        "is not small.",

    (TheoryLevel.HARTREE_FOCK, TheoryLevel.DIRAC_HARTREE_FOCK):
        "Replacing the Schrodinger equation with the Dirac equation adds all "
        "relativistic effects at once: scalar relativistic corrections "
        "(mass-velocity + Darwin), spin-orbit coupling, and the small component "
        "of the wavefunction.",

    (TheoryLevel.DIRAC_HARTREE_FOCK, TheoryLevel.DIRAC_FOCK_BREIT):
        "Adding the Breit interaction: magnetic interaction between electrons "
        "and retardation of the Coulomb interaction.  Important for inner-shell "
        "energies of heavy atoms.",

    (TheoryLevel.DIRAC_FOCK_BREIT, TheoryLevel.DIRAC_FOCK_QED):
        "Adding QED corrections:\n"
        "  - Vacuum polarization (Uehling): virtual electron-positron pairs "
        "screen the nuclear charge\n"
        "  - Self-energy (Lamb shift): electron interaction with its own "
        "radiation field",

    (TheoryLevel.DIRAC_HARTREE_FOCK, TheoryLevel.MCDHF):
        "Adding electron correlation via multi-configuration mixing.  The "
        "single-configuration approximation (DHF) misses instantaneous "
        "electron-electron correlation; MCDHF recovers some of this by "
        "mixing multiple configurations.",
}


def what_changed(before, after) -> str:
    """Explain what physics the new method adds compared to the previous one.

    Parameters
    ----------
    before, after
        Result objects from two different theory levels.

    Returns
    -------
    str
        Plain-English explanation.
    """
    lvl_before = _get_theory_level_enum(before)
    lvl_after = _get_theory_level_enum(after)

    dE = after.total_energy - before.total_energy
    dE_eV = dE * HARTREE_EV
    dE_cm1 = dE * HARTREE_CM1

    explanation = _EXPLANATIONS.get((lvl_before, lvl_after))
    if explanation is None:
        explanation = (
            f"Going from {_get_method_name(before)} to {_get_method_name(after)}.")

    energy_note = (
        f"\nEnergy change: {dE:.6f} Ha = {dE_eV:.4f} eV = {dE_cm1:.1f} cm-1")

    return explanation + energy_note


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_factor(unit: EnergyUnit) -> float:
    if unit == EnergyUnit.EV:
        return HARTREE_EV
    elif unit == EnergyUnit.CM1:
        return HARTREE_CM1
    elif unit == EnergyUnit.RYDBERG:
        return 2.0
    return 1.0


def _get_method_name(result) -> str:
    if hasattr(result, 'method_name'):
        return result.method_name
    if hasattr(result, 'method'):
        return result.method
    return "Unknown"


def _get_theory_level(result) -> str:
    if hasattr(result, 'theory_level'):
        return result.theory_level.name
    return "?"


def _get_theory_level_enum(result) -> TheoryLevel:
    if hasattr(result, 'theory_level'):
        return result.theory_level
    return TheoryLevel.HYDROGENIC


def _get_coupling(result) -> str:
    if hasattr(result, 'coupling_scheme'):
        return result.coupling_scheme.value
    return "?"


def _get_breakdown(result) -> dict:
    if hasattr(result, 'energy_breakdown'):
        return result.energy_breakdown
    if hasattr(result, 'energy_components'):
        return result.energy_components
    return {}
