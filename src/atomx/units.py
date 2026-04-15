"""Pint-based unit handling at the atomx API boundary.

atomx computes internally in Hartree atomic units.  This module lets
**API callers** express values in any unit (eV, cm⁻¹, MHz, fm, Kelvin,
barns …) without having to remember atomx's internal conventions, and
get results back in whatever units they like.

Typical usage
-------------
>>> from atomx import Atom
>>> from atomx.units import Hartree, eV, cm_inv, MHz, ureg
>>>
>>> ne = Atom(Z=10).solve()
>>> ne.total_energy                      # float, Hartree (internal)
-128.547
>>> ne.total_energy_as(eV)               # Quantity in eV
<Quantity(-3497.47, 'electron_volt')>
>>> ne.total_energy_as("cm^-1")           # string-as-unit also works
<Quantity(-28215000.0, '1 / centimeter')>
>>>
>>> # Any pint-supported unit can be consumed, too:
>>> atom_E = 1.0 * ureg.rydberg          # user wants Rydberg
>>> atomx_internal = to_hartree(atom_E)  # 0.5 (Hartree)

Optional dependency
-------------------
Pint is listed under the ``[units]`` optional-dependency group:

    pip install "atomx[units]"

Without pint, the convenience unit-conversion functions below raise
a clear RuntimeError — but atomx itself works normally (internal
calculations never depend on pint).
"""

from __future__ import annotations

from typing import Any, Optional, Union

from atomx.constants import (
    ALPHA,
    BOHR_RADIUS_M,
    HARTREE_EV,
    HARTREE_J,
    PROTON_ELECTRON_MASS_RATIO,
    DEFAULT_CODATA_VERSION,
)


# ---------------------------------------------------------------------------
# Optional pint import
# ---------------------------------------------------------------------------

try:
    import pint
    _PINT_AVAILABLE = True
    #: Shared pint :class:`~pint.UnitRegistry`.  Use this (not a fresh
    #: instance) so that Quantities produced by different atomx calls
    #: are mutually comparable.
    ureg = pint.UnitRegistry()
    Quantity = ureg.Quantity
except ImportError:  # pragma: no cover
    pint = None
    _PINT_AVAILABLE = False
    ureg = None
    Quantity = None


def _require_pint():
    if not _PINT_AVAILABLE:  # pragma: no cover
        raise RuntimeError(
            "pint is required for atomx.units conversions but is not "
            "installed.\nInstall with:  pip install pint   "
            "(or  pip install \"atomx[units]\")"
        )


# ---------------------------------------------------------------------------
# Convenience unit aliases (pint objects)
# ---------------------------------------------------------------------------
# Exported only when pint is available so users can write
#     from atomx.units import Hartree, eV, MHz, Bohr

if _PINT_AVAILABLE:
    # Energy
    Hartree = ureg.hartree
    Ha = ureg.hartree
    eV = ureg.electron_volt
    Rydberg = ureg.rydberg
    Joule = ureg.joule
    kelvin = ureg.kelvin
    cm_inv = ureg.cm ** -1
    invcm = ureg.cm ** -1

    # Frequency
    Hz = ureg.hertz
    kHz = ureg.kilohertz
    MHz = ureg.megahertz
    GHz = ureg.gigahertz
    THz = ureg.terahertz

    # Length
    Bohr = ureg.bohr
    a0 = ureg.bohr
    metre = ureg.meter
    nm = ureg.nanometer
    angstrom = ureg.angstrom
    fm = ureg.femtometer
    cm = ureg.centimeter

    # Mass
    amu = ureg.atomic_mass_constant
    electron_mass = ureg.electron_mass

    # Area (quadrupole moments)
    barn = ureg.barn

    # Misc
    dimensionless = ureg.dimensionless
else:  # pragma: no cover
    Hartree = Ha = eV = Rydberg = Joule = kelvin = None
    cm_inv = invcm = None
    Hz = kHz = MHz = GHz = THz = None
    Bohr = a0 = metre = nm = angstrom = fm = cm = None
    amu = electron_mass = None
    barn = None
    dimensionless = None


# ---------------------------------------------------------------------------
# Dimension-aware converters (accept Quantity OR plain float)
# ---------------------------------------------------------------------------

def _coerce_to_au(
    value: Any,
    default_unit: str,
) -> float:
    """Core helper — convert value to atomx internal atomic units.

    If ``value`` is a pint Quantity, convert it to ``default_unit``
    and return the magnitude.  If it is a plain float, assume it is
    already in ``default_unit`` and return as-is.
    """
    if _PINT_AVAILABLE and hasattr(value, "magnitude"):
        return float(value.to(default_unit).magnitude)
    return float(value)


def to_hartree(value: Any) -> float:
    """Convert an energy-valued input to Hartree (internal units).

    Accepts a pint Quantity with energy / frequency / temperature
    dimension (via spectroscopy context), or a plain float already
    in Hartree.
    """
    if _PINT_AVAILABLE and hasattr(value, "magnitude"):
        try:
            return float(value.to(ureg.hartree).magnitude)
        except pint.DimensionalityError:
            # Try spectroscopy context for cm^-1 / Hz → energy.
            return float(value.to(ureg.hartree, "spectroscopy").magnitude)
    return float(value)


def to_bohr(value: Any) -> float:
    """Convert a length-valued input to Bohr (internal units)."""
    return _coerce_to_au(value, "bohr")


def to_kelvin(value: Any) -> float:
    """Convert a temperature-like input to Kelvin."""
    if _PINT_AVAILABLE and hasattr(value, "magnitude"):
        try:
            return float(value.to(ureg.kelvin).magnitude)
        except pint.DimensionalityError:
            # Accept an energy via k_B
            return float(
                (value / ureg.boltzmann_constant).to(ureg.kelvin).magnitude
            )
    return float(value)


def to_MHz(value: Any) -> float:
    """Convert a frequency-valued input to MHz.

    Accepts energy (via spectroscopy context), wavenumber (cm⁻¹), or
    bare Hz / kHz / MHz / GHz / THz Quantity.
    """
    if _PINT_AVAILABLE and hasattr(value, "magnitude"):
        try:
            return float(value.to(ureg.megahertz).magnitude)
        except pint.DimensionalityError:
            return float(
                value.to(ureg.megahertz, "spectroscopy").magnitude)
    return float(value)


# ---------------------------------------------------------------------------
# Quantity factories (wrap internal atomic-units value as pint Quantity)
# ---------------------------------------------------------------------------

def energy_quantity(E_Ha: float) -> "Quantity":
    """Wrap an atomic-units energy in a pint Quantity (Hartree).

    Use ``.to(<unit>)`` on the returned Quantity to convert, including
    cross-dimension spectroscopy conversions::

        >>> q = energy_quantity(1.0)
        >>> q.to("eV")
        <Quantity(27.2114, 'electron_volt')>
        >>> q.to("cm^-1", "spectroscopy")
        <Quantity(219474.6, '1 / centimeter')>
        >>> q.to("MHz", "spectroscopy")
        <Quantity(6.58e9, 'megahertz')>

    Temperature conversion via ``k_B`` uses :func:`to_kelvin`
    (pint's spectroscopy context does not cover energy ↔ temperature).
    """
    _require_pint()
    return E_Ha * ureg.hartree


def length_quantity(r_Bohr: float) -> "Quantity":
    """Wrap an atomic-units length in a pint Quantity (Bohr)."""
    _require_pint()
    return r_Bohr * ureg.bohr


def frequency_quantity(nu_Hz: float) -> "Quantity":
    """Wrap a frequency (Hz) as a pint Quantity."""
    _require_pint()
    return nu_Hz * ureg.hertz


def wavenumber_quantity(sigma_cm_inv: float) -> "Quantity":
    """Wrap a wavenumber (cm⁻¹) as a pint Quantity."""
    _require_pint()
    return sigma_cm_inv * (ureg.cm ** -1)


def mass_quantity(m_au: float) -> "Quantity":
    """Wrap a mass in electron-mass units as a pint Quantity."""
    _require_pint()
    return m_au * ureg.electron_mass


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def pint_available() -> bool:
    """Return ``True`` when pint is importable in the current env."""
    return _PINT_AVAILABLE


def codata_version() -> str:
    """Return the CODATA version that atomx's internal constants use."""
    return DEFAULT_CODATA_VERSION


__all__ = [
    # Availability & metadata
    "pint_available",
    "codata_version",
    "ureg",
    "Quantity",
    # Conversion helpers (float-in, float-out, unit-aware)
    "to_hartree",
    "to_bohr",
    "to_kelvin",
    "to_MHz",
    # Factories (float-in, Quantity-out)
    "energy_quantity",
    "length_quantity",
    "frequency_quantity",
    "wavenumber_quantity",
    "mass_quantity",
    # Unit aliases (only populated when pint installed)
    "Hartree",
    "Ha",
    "eV",
    "Rydberg",
    "Joule",
    "kelvin",
    "cm_inv",
    "invcm",
    "Hz",
    "kHz",
    "MHz",
    "GHz",
    "THz",
    "Bohr",
    "a0",
    "metre",
    "nm",
    "angstrom",
    "fm",
    "cm",
    "amu",
    "electron_mass",
    "barn",
    "dimensionless",
]
