"""Combined isotope-shift calculation for a single atomic level.

For two isotopes A and A' of the same element,

    δν^{A,A'} = (K_NMS + K_SMS) · (1/M_A − 1/M_A')  [mass shift]
              + F · (⟨r²⟩_A − ⟨r²⟩_A')               [field shift]

This module packages the NMS + FS contributions (SMS not yet
implemented) into a single :class:`IsotopeShiftResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class IsotopeShiftResult:
    """Isotope shift between two isotopes of the same element.

    Attributes
    ----------
    isotope_a, isotope_b : str
        Labels of the two isotopes, e.g. ``"H-1"`` and ``"H-2 (D)"``.
    mass_me_a, mass_me_b : float
        Nuclear masses in electron-mass units.
    r_rms_fm_a, r_rms_fm_b : float
        Nuclear RMS charge radii in fm.  Zero if not tabulated.
    delta_r2_fm2 : float
        ``δ⟨r²⟩ = r_rms_A² − r_rms_B²``  in fm².  By the standard
        field-shift convention ``⟨r²⟩ ≡ ∫r² ρ_nuc d³r = r_rms²`` is
        the *second moment* of the nuclear charge density — no ``5/3``
        factor, despite the common misconception (see Seltzer 1969,
        or King 1984 §2.1).
    E_NMS_MHz : float
        Normal-mass-shift contribution to δν in MHz.
    E_FS_MHz : float
        Field-shift contribution to δν in MHz (per-orbital or total,
        depending on which F was used).
    E_SMS_MHz : float
        Specific-mass-shift contribution.  Zero in the current release
        (two-body operator, not yet implemented).
    E_total_MHz : float
        ``E_NMS + E_SMS + E_FS``.
    orbital_FS_MHz : dict
        Per-orbital field-shift breakdown, ``{label: MHz}``, useful for
        transition-frequency isotope shifts when users subtract two
        level contributions.
    """

    isotope_a: str
    isotope_b: str
    mass_me_a: float
    mass_me_b: float
    r_rms_fm_a: float
    r_rms_fm_b: float
    delta_r2_fm2: float
    E_NMS_MHz: float
    E_FS_MHz: float
    E_SMS_MHz: float = 0.0
    E_total_MHz: float = 0.0
    orbital_FS_MHz: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Unit-aware accessors (pint)
    # ------------------------------------------------------------------

    def _to_unit(self, value_MHz: float, unit: str):
        from atomx.results import _as_quantity
        return _as_quantity(value_MHz, "MHz", unit)

    def total_as(self, unit: str = "MHz"):
        """Total isotope shift in the target unit (pint Quantity).

        Internal unit is MHz; spectroscopic conversions (Hz, GHz,
        THz, cm⁻¹, eV, Hartree) are handled automatically.
        """
        return self._to_unit(self.E_total_MHz, unit)

    def NMS_as(self, unit: str = "MHz"):
        """Normal-mass-shift contribution in target unit."""
        return self._to_unit(self.E_NMS_MHz, unit)

    def SMS_as(self, unit: str = "MHz"):
        """Specific-mass-shift contribution in target unit."""
        return self._to_unit(self.E_SMS_MHz, unit)

    def FS_as(self, unit: str = "MHz"):
        """Field-shift contribution in target unit."""
        return self._to_unit(self.E_FS_MHz, unit)
