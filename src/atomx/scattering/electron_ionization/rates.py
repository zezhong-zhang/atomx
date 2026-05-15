"""Maxwellian-averaged EII rate coefficients.

Public entry point: :func:`eii_rate_coefficient`.

For a precomputed cross section σ(E), the rate coefficient at
electron temperature T_e is

    α(T) = ⟨σ v⟩_M = (8 / (π m_e (k_B T)³))^{1/2}
                     · ∫₀^∞ σ(E) · E · exp(−E / k_B T) dE.

The integration uses :func:`atomx.plasma.rates.maxwellian_rate`
so the same trapezoidal kernel is shared with PI, RR, AI rate
coefficients.

Reference
---------
* M. F. Gu, *Can. J. Phys.* **86**, 675 (2008) — FAC's
  ``MaxwellRate`` knob.
"""

from __future__ import annotations

from typing import Union

import numpy as np


def eii_rate_coefficient(
    eii_result,
    temperature_eV: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Maxwellian-averaged ⟨σv⟩(T_e) from a precomputed
    :class:`EIIResult`.

    Parameters
    ----------
    eii_result : EIIResult
        Output of :func:`atomx.scattering.electron_ionization.eii_cross_section`
        on a sufficiently dense energy grid (extending to ≥ 10× T_max).
    temperature_eV : float or array_like
        Maxwellian electron temperature(s) in eV.

    Returns
    -------
    float or np.ndarray
        Rate coefficient in cm³/s.  Scalar when ``temperature_eV`` is
        scalar.
    """
    from atomx.plasma.rates import maxwellian_rate
    T_arr = np.atleast_1d(np.asarray(temperature_eV, dtype=float))
    out = np.array([
        maxwellian_rate(eii_result.sigma_total_cm2,
                         eii_result.energies_eV, float(T))
        for T in T_arr
    ])
    return out if out.shape != (1,) else float(out[0])


__all__ = ["eii_rate_coefficient"]
