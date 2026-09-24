"""Wavelength-gridded synthetic-spectrum container.

A :class:`Spectrum` is a lightweight wrapper around a wavelength array
and a same-length intensity array.  It provides:

* factory constructors (linear / log / fromlist),
* a general-purpose :meth:`Spectrum.add_line` that deposits normalized
  Gaussian, Lorentzian, and Voigt profiles on the wavelength grid,
* a convenience :func:`synthesize_from_transitions` that takes a list
  of ``(wavelength, intensity)`` pairs (optionally with widths) and
  returns the rendered spectrum in one call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from atomx.lineshapes.profiles import voigt


@dataclass
class Spectrum:
    """Wavelength-sampled synthetic spectrum.

    Attributes
    ----------
    wavelength_nm : np.ndarray
        Increasing wavelength axis in nanometres, length ``N``. Treat the
        grid as immutable after adding a line because bin geometry is cached.
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
    _bin_edges: np.ndarray | None = field(default=None, init=False, repr=False)
    _bin_widths: np.ndarray | None = field(default=None, init=False, repr=False)
    _min_step: float | None = field(default=None, init=False, repr=False)

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
        if wl.ndim != 1 or len(wl) < 2:
            raise ValueError("spectrum grid must contain at least two increasing points")
        if not np.isfinite(lam0_nm) or not np.isfinite(intensity):
            raise ValueError("line center and intensity must be finite")
        if sigma_nm is not None and (not np.isfinite(sigma_nm) or sigma_nm < 0):
            raise ValueError("Gaussian width must be finite and nonnegative")
        if gamma_nm is not None and (not np.isfinite(gamma_nm) or gamma_nm < 0):
            raise ValueError("Lorentzian width must be finite and nonnegative")
        sigma = float(sigma_nm or 0.0)
        gamma = float(gamma_nm or 0.0)
        if self._bin_edges is None:
            steps = np.diff(wl)
            if np.any(steps <= 0):
                raise ValueError("spectrum grid must be increasing")
            self._min_step = float(np.min(steps))
            self._bin_edges = np.r_[wl[0], (wl[:-1]+wl[1:])/2, wl[-1]]
            self._bin_widths = np.diff(self._bin_edges)
        edges = self._bin_edges
        cell_width = self._bin_widths
        if sigma == 0 and gamma == 0:
            # Unresolved delta: its finite-volume cell carries the full area.
            idx = int(np.argmin(np.abs(wl-lam0_nm)))
            self.values[idx] += intensity/cell_width[idx]
        elif gamma == 0:
            # Gaussian tails beyond 8 sigma contain < 1.3e-15 of the area.
            lo = max(0, int(np.searchsorted(edges, lam0_nm-8*sigma)-1))
            hi = min(len(wl), int(np.searchsorted(edges, lam0_nm+8*sigma)+1))
            masses = ndtr((edges[lo+1:hi+1]-lam0_nm)/sigma) - ndtr((edges[lo:hi]-lam0_nm)/sigma)
            self.values[lo:hi] += intensity*masses/cell_width[lo:hi]
        elif sigma == 0:
            masses = (np.arctan((edges[1:]-lam0_nm)/gamma)
                      - np.arctan((edges[:-1]-lam0_nm)/gamma))/np.pi
            self.values += intensity*masses/cell_width
        else:
            spacing = self._min_step
            if max(sigma, gamma) < 2*spacing:
                # Center cells need actual Voigt integrals; midpoint sampling
                # misses arbitrarily narrow profiles. Wings vary smoothly.
                mass = voigt(wl, lam0_nm, sigma, gamma)*cell_width
                near = np.flatnonzero(np.abs(wl-lam0_nm) < max(8*spacing, 6*sigma, 6*gamma))
                for i in near:
                    mass[i] = quad(lambda x: float(voigt(x, lam0_nm, sigma, gamma)),
                                   edges[i], edges[i+1], epsabs=1e-10)[0]
                self.values += intensity*mass/cell_width
            else:
                self.values += intensity*voigt(wl, lam0_nm, sigma, gamma)
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

    # ------------------------------------------------------------------
    # Visualisation convenience
    # ------------------------------------------------------------------

    def plot(self, *, backend: str = "mpl", **kwargs):
        """Render this Spectrum with the requested viz backend.

        Parameters
        ----------
        backend : {"mpl", "plotly"}
            Default ``"mpl"`` (matplotlib).
        **kwargs
            Forwarded to the backend's ``plot_spectrum``.

        Returns
        -------
        Figure object from the selected backend.
        """
        if backend == "mpl":
            from atomx.viz.mpl import plot_spectrum
            return plot_spectrum(self, **kwargs)
        if backend == "plotly":
            from atomx.viz.plotly import plot_spectrum
            return plot_spectrum(self, **kwargs)
        raise ValueError(
            f"Unknown backend {backend!r}; use 'mpl' or 'plotly'."
        )


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
        (e.g. a :class:`~atomx.properties.transitions.calculator.TransitionResult`),
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
            lam0 = float(t.wavelength_nm)
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
