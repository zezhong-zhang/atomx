"""Line shapes for atomic emission and absorption spectra.

Provides three elementary profile families and their standard
line-width helpers:

* Gaussian (Doppler) profile
* Lorentzian (natural / pressure-broadened) profile
* Voigt profile = convolution of the two, via scipy's
  :func:`scipy.special.voigt_profile`

All profiles are **normalised** so that ``∫ f(ν) dν = 1`` over the
frequency axis, in whatever frequency unit (Hz, MHz, cm⁻¹, nm⁻¹) the
caller chooses — ``x`` and the widths must share the same unit.

Width helpers
-------------
:func:`doppler_width` returns the Gaussian σ (1 σ) from temperature,
atomic mass and line frequency via the standard formula
``σ = ν₀ · sqrt(kT/(Mc²))``.

:func:`natural_width` returns Lorentzian HWHM ``Γ = A/(4π)`` from an
Einstein A coefficient (units consistent with the chosen ``x``-axis).

:func:`stark_hwhm_quasistatic` implements the Griem quasi-static
Stark HWHM formula for hydrogenic ions in a dense-plasma environment
(leading-order only, sufficient for synthetic-spectrum use).

Design note
-----------
The elementary profile functions are vectorised over ``x``.  For
multi-line spectra the caller builds a :class:`Spectrum` (in
:mod:`atomx.spectrum`) and calls its ``.add_line`` method, which
delegates to these profiles and sums contributions.
"""

from atomx.lineshapes.profiles import (
    doppler_width,
    gaussian,
    lorentzian,
    natural_width,
    voigt,
)
from atomx.lineshapes.stark import (
    stark_hwhm_quasistatic,
)

__all__ = [
    "doppler_width",
    "gaussian",
    "lorentzian",
    "natural_width",
    "stark_hwhm_quasistatic",
    "voigt",
]
