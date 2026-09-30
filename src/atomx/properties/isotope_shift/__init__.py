"""Isotope shifts: mass shift + field shift.

The isotope shift ``δν^{A,A'} = ν^A − ν^{A'}`` for an atomic transition
decomposes into a mass-shift piece (proportional to 1/M) and a
field-shift piece (proportional to δ⟨r²⟩ of the nuclear charge):

    δν = (K_NMS + K_SMS) · (1/M_A − 1/M_A')        [mass shift]
       + F · (⟨r²⟩_A − ⟨r²⟩_A')                     [field shift]

This module implements the **Normal Mass Shift** (NMS) via the virial
theorem and the **Field Shift** via the first-order Seltzer formula.
The **Specific Mass Shift** (SMS) is evaluated by direct Vinti momentum
integrals, with optional point-Coulomb Breit recoil terms. This shares the
radial implementation with the recoil correction stack. A user-supplied
SMS coefficient remains available for external or correlated results.

Public API
----------
>>> from atomx import Atom
>>> result = Atom(Z=1).solve(...)
>>> ms = result.mass_shift_NMS()          # MassShift for default isotope
>>> F  = result.field_shift_parameter()   # FieldShiftParameter per orbital
>>> iso = result.isotope_shift((1, 1), (1, 2))  # H vs D for every orbital
"""

from atomx.properties.isotope_shift.field_shift import (
    FieldShiftParameter,
    field_shift_factor_per_orbital,
    seltzer_F_Ha_per_Bohr2,
)
from atomx.properties.isotope_shift.isotope_shift import (
    IsotopeShiftResult,
)
from atomx.properties.isotope_shift.mass_shift import (
    MassShift,
    SpecificMassShift,
    normal_mass_shift,
    specific_mass_shift,
)
from atomx.properties.isotope_shift.sms_abinitio import (
    SMSResult,
    compute_sms_abinitio,
)

__all__ = [
    "FieldShiftParameter",
    "IsotopeShiftResult",
    "MassShift",
    "SMSResult",
    "SpecificMassShift",
    "compute_sms_abinitio",
    "field_shift_factor_per_orbital",
    "normal_mass_shift",
    "seltzer_F_Ha_per_Bohr2",
    "specific_mass_shift",
]
