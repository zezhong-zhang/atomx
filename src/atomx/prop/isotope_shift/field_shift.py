"""Field shift — nuclear-charge-radius dependence of atomic energy levels.

For two isotopes with different charge-distribution RMS radii,
the perturbative energy shift is

    ΔE_FS = F · (⟨r²⟩_A − ⟨r²⟩_A')                                  (1)

where F is the **field-shift factor** with dimensions of energy/length².
For a single-electron s-orbital, the Seltzer (1969) leading-order
expression is

    F_a = (2π Z / 3) · |ψ_a(0)|²       [atomic units, Hartree / Bohr²]

Including higher-order corrections, particularly the relativistic
enhancement for heavy atoms, can be done either perturbatively (with
the density at the nuclear boundary rather than the origin) or via
finite-difference SCF with two different nuclear radii.  atomx ships
the Seltzer leading-order factor here, and a numerical-derivative
helper is planned for a later release.

Because the Dirac radial wavefunction has P(r) ~ r^γ near the origin
(γ = √(1 − (Zα)²) < 1 for Z > 0), the quantity ``|ψ(0)|²`` must be
extracted as the coefficient of r^γ rather than by direct evaluation
at ``r = 0``.  For light atoms γ ≈ 1 and the difference from naive
``|P(r_min)/r_min|²`` is below 1 %; for heavy atoms the correct
approach is the finite-difference SCF derivative.

Reference
---------
E. C. Seltzer, *Phys. Rev.* **188**, 1916 (1969).
B.-O. King, *Isotope Shifts in Atomic Spectra* (Plenum, 1984).
S. G. Karshenboim, *Phys. Rep.* **422**, 1 (2005) — relativistic
corrections and the "field shift parameter" terminology review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from atomx.numerics.bspline.evaluate import eval_orbital_on_grid
from atomx.constants import ALPHA, BOHR_RADIUS_M
from atomx.corrections.qedmod.welton import _density_at_origin
from atomx.prop.hyperfine.dipole import HARTREE_MHZ


# 1 Bohr = BOHR_RADIUS_M metres;  1 fm = 1e-15 metres
# Bohr²/fm² = (BOHR_RADIUS_M × 1e15)²
BOHR_IN_FM = BOHR_RADIUS_M * 1e15             # ≈ 5.29177e4 fm
#: Conversion from Bohr⁻² to fm⁻² (divide F[Ha/Bohr²] by this to get Ha/fm²)
BOHR2_IN_FM2 = BOHR_IN_FM * BOHR_IN_FM         # Bohr² expressed in fm²


@dataclass
class FieldShiftParameter:
    """Per-orbital field-shift factor F_a.

    Only orbitals with non-negligible ``|ψ_a(0)|²`` (i.e. j = 1/2,
    κ = ±1) carry a first-order field shift; higher-l orbitals have
    a much smaller contribution from the small component Q(r) only.

    Attributes
    ----------
    F_MHz_per_fm2 : dict
        ``{orbital_label: F_MHz_per_fm2}`` — field-shift factor in the
        physicist's favourite units.  Multiply by Δ⟨r²⟩ in fm² to get
        the isotope shift in MHz.
    F_Ha_per_Bohr2 : dict
        Same, in atomic units (Hartree / Bohr²).
    total_F_MHz_per_fm2 : float
        Sum of per-orbital contributions weighted by occupation —
        appropriate for ground-state total-energy field shifts.
    """

    F_MHz_per_fm2: Dict[str, float] = field(default_factory=dict)
    F_Ha_per_Bohr2: Dict[str, float] = field(default_factory=dict)
    total_F_MHz_per_fm2: float = 0.0


def seltzer_F_Ha_per_Bohr2(psi0_sq: float, Z: int) -> float:
    """Seltzer leading-order field-shift factor in Hartree / Bohr².

        F = (2π Z / 3) · |ψ(0)|²           (atomic units)
    """
    return (2.0 * np.pi * Z / 3.0) * psi0_sq


def field_shift_factor_per_orbital(
    orbitals,
    bsp_data,
    Z: int,
) -> FieldShiftParameter:
    """Compute the Seltzer field-shift factor ``F_a`` for each orbital.

    Parameters
    ----------
    orbitals : list of ``(P, Q, occ, kappa, label)`` tuples
        Same structure as :func:`DFSolver._flatten_orbitals`.
    bsp_data : dict
        B-spline workspace.
    Z : int
        Nuclear charge.

    Returns
    -------
    FieldShiftParameter
    """
    per_F_Ha = {}
    per_F_MHz = {}
    total_MHz = 0.0

    conv_MHz_per_fm2 = HARTREE_MHZ / BOHR2_IN_FM2   # Ha/Bohr² → MHz/fm²

    for P, Q, occ, kappa, label in orbitals:
        # |ψ(0)|² estimator from Welton's density-at-origin helper
        psi0_sq = _density_at_origin(P, Q, bsp_data)

        F_Ha = seltzer_F_Ha_per_Bohr2(psi0_sq, Z)
        F_MHz = F_Ha * conv_MHz_per_fm2

        per_F_Ha[label] = F_Ha
        per_F_MHz[label] = F_MHz
        total_MHz += occ * F_MHz

    return FieldShiftParameter(
        F_MHz_per_fm2=per_F_MHz,
        F_Ha_per_Bohr2=per_F_Ha,
        total_F_MHz_per_fm2=total_MHz,
    )
