"""Explicit detector-to-hemisphere energy contract for absolute EUV CE.

Raw ADU, arbitrary units and normalized spectra do not supply an energy
calibration. The angular multiplier must come from the experiment's declared
geometry/angular model, never a hidden 2pi extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EUVAbsoluteCalibration:
    wavelength_edges_nm: np.ndarray
    counts_per_collected_J: np.ndarray
    hemisphere_multiplier: float
    accumulated_shots: int
    source_id: str
    angular_model_source_id: str


def calibrated_conversion_efficiency(counts_per_bin, background_counts_per_bin,
                                      incident_laser_energy_J_per_shot, calibration, *,
                                      band_nm=(13.365, 13.635)):
    """Absolute band energy in one hemisphere / incident pulse energy.

    Counts and dark reference are integrated over the same spectral/spatial
    aperture and shot accumulation. Negative background-subtracted bins remain
    signed measurements. Partial bins use a declared piecewise-constant energy
    density. This routine does not fit a gain, wavelength shift or angular factor.
    """
    if not isinstance(calibration, EUVAbsoluteCalibration):
        raise ValueError("absolute detector response and angular calibration required; counts alone do not determine CE")
    c = calibration
    edges = np.asarray(c.wavelength_edges_nm, dtype=float)
    counts, dark, response = (np.asarray(v, dtype=float) for v in
                              (counts_per_bin, background_counts_per_bin, c.counts_per_collected_J))
    if (edges.ndim != 1 or edges.size < 2 or counts.shape != (edges.size - 1,)
            or dark.shape != counts.shape or response.shape != counts.shape
            or any(np.any(~np.isfinite(v)) for v in (edges, counts, dark, response))
            or np.any(np.diff(edges) <= 0) or edges[0] <= 0 or np.any(response <= 0)
            or not np.isfinite(c.hemisphere_multiplier) or c.hemisphere_multiplier < 1
            or not isinstance(c.accumulated_shots, (int, np.integer)) or c.accumulated_shots < 1
            or not c.source_id or not c.angular_model_source_id
            or not np.isfinite(incident_laser_energy_J_per_shot) or incident_laser_energy_J_per_shot <= 0
            or not 0 < band_nm[0] < band_nm[1]):
        raise ValueError("invalid detector, laser, band or absolute calibration inputs")
    fraction = np.maximum(0, np.minimum(edges[1:], band_nm[1]) - np.maximum(edges[:-1], band_nm[0])) / np.diff(edges)
    if edges[0] > band_nm[0] or edges[-1] < band_nm[1]:
        raise ValueError("detector wavelength range does not cover the complete acceptance band")
    energy_collected = float(fraction @ ((counts - dark) / response)) / c.accumulated_shots
    energy_hemisphere = energy_collected * c.hemisphere_multiplier
    return dict(band_energy_collected_J_per_shot=energy_collected,
                band_energy_hemisphere_J_per_shot=energy_hemisphere,
                incident_laser_energy_J_per_shot=incident_laser_energy_J_per_shot,
                conversion_efficiency=energy_hemisphere / incident_laser_energy_J_per_shot,
                band_nm=tuple(band_nm), calibration_source_id=c.source_id,
                angular_model_source_id=c.angular_model_source_id)
