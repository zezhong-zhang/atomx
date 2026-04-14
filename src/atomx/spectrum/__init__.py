"""Synthetic atomic spectra.

``Spectrum`` is a wavelength/frequency-gridded array of emissivity (or
absorption, opacity, …) built up line-by-line from atomic transition
results.  Each line is convolved with a user-chosen profile (Gaussian,
Lorentzian, or Voigt) from :mod:`atomx.lineshapes`.

Public API
----------
>>> from atomx.spectrum import Spectrum
>>> s = Spectrum.linear(lam_min_nm=100.0, lam_max_nm=200.0, n_points=5001)
>>> s.add_line(lam0_nm=121.567, intensity=1.0, sigma_nm=0.01)      # Gaussian
>>> s.add_line(lam0_nm=102.572, intensity=0.8, gamma_nm=0.005)     # Lorentzian
>>> s.add_line(lam0_nm=97.253, intensity=0.3,
...            sigma_nm=0.01, gamma_nm=0.005)                      # Voigt
>>> y = s.values   # ndarray of per-wavelength intensity

See :mod:`atomx.uta` for Unresolved-Transition-Array (Bauche-Arnoult)
moments, and :meth:`synthesize_from_transitions` below for the
standard emissivity-from-rates helper.
"""

from atomx.spectrum.crm_synth import crm_emission_spectrum
from atomx.spectrum.spectrum import (
    Spectrum,
    synthesize_from_transitions,
)

__all__ = [
    "Spectrum",
    "crm_emission_spectrum",
    "synthesize_from_transitions",
]
