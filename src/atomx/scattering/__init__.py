"""Continuum-state atomic physics — photoionisation, autoionisation,
dielectronic recombination, electron-impact scattering.

Every module here consumes the bound-state wavefunctions from an
:class:`~atomx.results.AtomicResult` and builds the corresponding
continuum orbital via the Dirac / Schrödinger continuum solver in
:mod:`atomx.numerics.bspline.continuum`.

Modules
-------
:mod:`atomx.scattering.photoionization`
    Non-resonant and Fano-resonant photoionisation cross-sections.

:mod:`atomx.scattering.autoionization`
    Auger decay of doubly-excited autoionising states.

:mod:`atomx.scattering.dielectronic_recombination`
    DR rate coefficients (resonant capture + radiative stabilisation).

:mod:`atomx.scattering.distorted_wave`
    Electron-impact excitation / ionisation in the distorted-wave
    approximation.

:mod:`atomx.scattering.r_matrix` (Phase-7 scaffold)
    Close-coupling R-matrix method for systematic continuum treatment.
"""

from atomx.scattering.models import ContinuumModel, PotentialModel


__all__ = ["ContinuumModel", "PotentialModel"]
