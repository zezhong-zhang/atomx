"""Leading-order Stark broadening helpers (quasi-static ion approximation).

Stark broadening in a dense-plasma environment arises from the
statistically-distributed electric field of surrounding ions and
electrons.  A full treatment (Griem 1974 monograph) sums over impact
and quasi-static contributions with careful coupling to plasma
conditions.  atomx ships the leading-order quasi-static *ion*
contribution — the dominant term for high-density, hot plasmas — as a
quick synthetic-spectra helper.

Formula (Griem, *Plasma Spectroscopy*, §10.3)
---------------------------------------------

For hydrogenic transitions between principal quantum numbers
``(n_u, n_l)``, the Holtsmark normal field strength is

    F_0 = 2.60 · e · n_i^{2/3}

and the Stark FWHM at 10 % intensity is approximately

    Δλ_S = F_0 · α_H(n_u, n_l) · λ² / c                             (1)

with ``α_H`` a tabulated coefficient (Holtsmark α).  For order-of-
magnitude estimates we use the simplified expression

    HWHM_Stark [nm]  ≈  C(n_u, n_l) · (n_e / 10¹⁷ cm⁻³)^{2/3}        (2)

where C is the Griem coefficient — fine for plasma-spectrum
applications at the ~10 % level.  For quantitative work the full
Griem-tabulated profiles or the MERL code should be used.

Returns HWHM in the same unit as ``lambda_nm`` (nanometres here).
"""

from __future__ import annotations

# Griem Table II.5 (Plasma Spectroscopy 1964) — Holtsmark coefficients
# for H I lines at n_e = 1e17 cm⁻³ and T_e = 10⁴ K.  Keys: (n_u, n_l).
# Values: HWHM in angstroms.  Used for the simplified eq. (2).
_GRIEM_H_HWHM_ANGSTROM = {
    # (n_u, n_l): HWHM in Å at n_e = 1e17, T=1e4 K
    (2, 1): 0.06,   # Lyman-α
    (3, 2): 0.38,   # H-α
    (4, 2): 1.44,   # H-β
    (5, 2): 2.79,   # H-γ
    (6, 2): 4.00,   # H-δ
    (3, 1): 0.20,   # Lyman-β
}


def stark_hwhm_quasistatic(
    n_upper: int,
    n_lower: int,
    n_e_per_cm3: float,
    T_electron_K: float = 1.0e4,
    element: str = "H",
) -> float:
    """Leading-order Stark HWHM for a hydrogenic transition, in Å.

    Parameters
    ----------
    n_upper, n_lower
        Principal quantum numbers of the transition (``n_u → n_l``).
    n_e_per_cm3
        Electron density in cm⁻³.
    T_electron_K
        Electron temperature in K (weak dependence — for the
        leading-order result the HWHM is ≈ T-independent, since ions
        dominate the broadening).
    element
        Currently only ``"H"`` is tabulated; extension to heavier ions
        via their respective Griem tables is straightforward.

    Returns
    -------
    float
        HWHM in Ångstrom.  Multiply by 10 to get pm, or by 0.1 to get nm.
    """
    if element != "H":
        raise NotImplementedError(
            f"Stark broadening for element {element!r} not yet tabulated; "
            f"only hydrogenic lines are supported."
        )
    key = (int(n_upper), int(n_lower))
    if key not in _GRIEM_H_HWHM_ANGSTROM:
        raise KeyError(
            f"No Griem tabulation for H ({n_upper} → {n_lower}); "
            f"available transitions: {sorted(_GRIEM_H_HWHM_ANGSTROM)}."
        )
    hwhm_ref = _GRIEM_H_HWHM_ANGSTROM[key]
    # Griem scaling:  HWHM ∝ n_e^{2/3}, weak T dependence (ignored here).
    scaling = (n_e_per_cm3 / 1.0e17) ** (2.0 / 3.0)
    return hwhm_ref * scaling
