"""Wavelength-gridded synthetic-spectrum container.

A :class:`Spectrum` is a lightweight wrapper around a wavelength array
and a same-length intensity array.  It provides:

* factory constructors (linear / log / fromlist),
* a general-purpose :meth:`Spectrum.add_line` that delegates to the
  primitive profiles in :mod:`atomx.lineshapes`,
* a convenience :func:`synthesize_from_transitions` that takes a list
  of ``(wavelength, intensity)`` pairs (optionally with widths) and
  returns the rendered spectrum in one call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from atomx.lineshapes.profiles import gaussian, lorentzian, voigt


@dataclass
class Spectrum:
    """Wavelength-sampled synthetic spectrum.

    Attributes
    ----------
    wavelength_nm : np.ndarray
        Wavelength axis in nanometres, length ``N``.
    values : np.ndarray
        Per-wavelength intensity (emissivity, opacity, or other),
        length ``N``.  Starts as zeros; lines accumulate via
        :meth:`add_line`.
    unit : str
        Free-form label for the intensity column ("emissivity",
        "absorbance", "opacity cm⁻¹" …).  Purely informational.
    line_count : int
        Number of lines added so far.  Bookkeeping only.
    """

    wavelength_nm: np.ndarray
    values: np.ndarray
    unit: str = "emissivity (arb.)"
    line_count: int = 0
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def linear(
        cls,
        lam_min_nm: float,
        lam_max_nm: float,
        n_points: int,
        *,
        unit: str = "emissivity (arb.)",
    ) -> "Spectrum":
        """Uniform wavelength grid."""
        wl = np.linspace(lam_min_nm, lam_max_nm, n_points)
        return cls(wavelength_nm=wl, values=np.zeros_like(wl), unit=unit)

    @classmethod
    def logarithmic(
        cls,
        lam_min_nm: float,
        lam_max_nm: float,
        n_points: int,
        *,
        unit: str = "emissivity (arb.)",
    ) -> "Spectrum":
        """Log-spaced wavelength grid — useful for very broadband spectra."""
        wl = np.geomspace(lam_min_nm, lam_max_nm, n_points)
        return cls(wavelength_nm=wl, values=np.zeros_like(wl), unit=unit)

    @classmethod
    def from_grid(
        cls, wavelength_nm, *, unit: str = "emissivity (arb.)"
    ) -> "Spectrum":
        wl = np.asarray(wavelength_nm, dtype=float)
        return cls(wavelength_nm=wl, values=np.zeros_like(wl), unit=unit)

    # ------------------------------------------------------------------
    # Core: add a single line
    # ------------------------------------------------------------------

    def add_line(
        self,
        lam0_nm: float,
        intensity: float,
        *,
        sigma_nm: float | None = None,
        gamma_nm: float | None = None,
    ) -> None:
        """Add a single spectral line to the spectrum.

        Profile selected automatically from widths:

          * ``sigma_nm`` only       → Gaussian (Doppler)
          * ``gamma_nm`` only       → Lorentzian (natural / pressure)
          * both set                → Voigt (convolution)

        Parameters
        ----------
        lam0_nm : float
            Central wavelength, nm.
        intensity : float
            Integrated intensity (area) of the line; the normalised
            profile is multiplied by this scalar before accumulating.
        sigma_nm, gamma_nm : float, optional
            Gaussian σ and Lorentzian HWHM γ, both in nm.  At least one
            must be supplied.
        """
        if sigma_nm is None and gamma_nm is None:
            raise ValueError("Must supply at least one of sigma_nm, gamma_nm.")

        if intensity == 0.0:
            return

        wl = self.wavelength_nm

        if sigma_nm is not None and gamma_nm is not None:
            profile = voigt(wl, lam0_nm, sigma_nm, gamma_nm)
        elif sigma_nm is not None:
            profile = gaussian(wl, lam0_nm, sigma_nm)
        else:
            profile = lorentzian(wl, lam0_nm, gamma_nm)

        self.values += intensity * profile
        self.line_count += 1

    def add_delta(self, lam0_nm: float, intensity: float) -> None:
        """Add an un-broadened delta-function line at the nearest grid point.

        Useful for stick-plot visualisations.
        """
        idx = int(np.argmin(np.abs(self.wavelength_nm - lam0_nm)))
        self.values[idx] += intensity
        self.line_count += 1

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def integrate(
        self, lam_min: float | None = None, lam_max: float | None = None,
    ) -> float:
        """Trapezoidal area under the spectrum in ``[lam_min, lam_max]``."""
        wl = self.wavelength_nm
        y = self.values
        if lam_min is not None:
            mask = wl >= lam_min
            wl, y = wl[mask], y[mask]
        if lam_max is not None:
            mask = wl <= lam_max
            wl, y = wl[mask], y[mask]
        return float(np.trapezoid(y, wl))

    def peak(self) -> tuple[float, float]:
        """Return the ``(λ, value)`` of the global maximum."""
        idx = int(np.argmax(self.values))
        return float(self.wavelength_nm[idx]), float(self.values[idx])

    def normalize(self, *, peak: bool = False) -> "Spectrum":
        """Return a copy normalised by area (default) or peak."""
        if peak:
            denom = float(np.max(self.values))
        else:
            denom = float(np.trapezoid(self.values, self.wavelength_nm))
        if denom == 0.0:
            return Spectrum(
                wavelength_nm=self.wavelength_nm.copy(),
                values=self.values.copy(),
                unit=self.unit,
                line_count=self.line_count,
                metadata=dict(self.metadata),
            )
        return Spectrum(
            wavelength_nm=self.wavelength_nm.copy(),
            values=self.values / denom,
            unit=self.unit,
            line_count=self.line_count,
            metadata=dict(self.metadata),
        )

    def __len__(self) -> int:
        return len(self.wavelength_nm)


# ---------------------------------------------------------------------------
# Convenience: render many lines in one call
# ---------------------------------------------------------------------------

def synthesize_from_transitions(
    transitions: Iterable,
    *,
    lam_min_nm: float,
    lam_max_nm: float,
    n_points: int = 2001,
    default_sigma_nm: float | None = None,
    default_gamma_nm: float | None = None,
    intensity_key: str = "A_einstein",
) -> Spectrum:
    """Render a list of transitions into a :class:`Spectrum`.

    Each item in ``transitions`` may be

      * an object with ``wavelength_nm`` and ``A_einstein`` attributes
        (e.g. a :class:`~atomx.transitions.calculator.TransitionResult`),
      * a dict with keys ``wavelength_nm`` and ``A_einstein`` (or
        another ``intensity_key``), or
      * a 2- or 4-tuple  ``(λ_nm, intensity [, sigma [, gamma]])``.

    Lines outside ``[lam_min_nm, lam_max_nm]`` are dropped silently.

    Parameters
    ----------
    transitions : iterable
        See above for accepted formats.
    lam_min_nm, lam_max_nm : float
        Spectrum wavelength range (nm).
    n_points : int
        Grid resolution (linearly spaced).
    default_sigma_nm, default_gamma_nm : float, optional
        Fallback Doppler σ and natural γ used when a transition omits
        its own widths.  At least one must be supplied.
    intensity_key : str
        Attribute / dict key holding the line strength.

    Returns
    -------
    Spectrum
    """
    spec = Spectrum.linear(lam_min_nm, lam_max_nm, n_points)
    if default_sigma_nm is None and default_gamma_nm is None:
        raise ValueError(
            "Must supply default_sigma_nm or default_gamma_nm as a fallback width."
        )

    for t in transitions:
        # Extract wavelength / intensity / widths
        if hasattr(t, "wavelength_nm"):
            lam0 = float(getattr(t, "wavelength_nm"))
            inten = float(getattr(t, intensity_key))
            sigma = getattr(t, "sigma_nm", None)
            gamma = getattr(t, "gamma_nm", None)
        elif isinstance(t, dict):
            lam0 = float(t["wavelength_nm"])
            inten = float(t.get(intensity_key, t.get("intensity", 0.0)))
            sigma = t.get("sigma_nm")
            gamma = t.get("gamma_nm")
        else:
            lam0 = float(t[0])
            inten = float(t[1])
            sigma = t[2] if len(t) > 2 else None
            gamma = t[3] if len(t) > 3 else None

        if sigma is None:
            sigma = default_sigma_nm
        if gamma is None:
            gamma = default_gamma_nm

        if lam0 < lam_min_nm or lam0 > lam_max_nm:
            continue

        spec.add_line(lam0, inten, sigma_nm=sigma, gamma_nm=gamma)

    return spec
