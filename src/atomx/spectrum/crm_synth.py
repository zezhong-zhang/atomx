"""CRM-weighted synthetic emission spectrum.

Given a :class:`~atomx.plasma.crm.CollisionalRadiativeModel` with rates
and populations, build a :class:`~atomx.spectrum.Spectrum` where each
radiative transition contributes a normalised line-profile weighted by
``n_upper · A_{ul}``.  Line widths are computed from Doppler (thermal)
and natural-lifetime contributions; an optional static Stark width for
hydrogenic ions can be folded in.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from atomx.constants import HARTREE_EV, HARTREE_INV_ANGSTROM
from atomx.lineshapes import doppler_width, natural_width
from atomx.spectrum.spectrum import Spectrum
from atomx.uta import uta_moments


def _hartree_to_angstrom(dE_hartree: float) -> float:
    """ΔE (Hartree) → λ (Å), standard 1/HARTREE_INV_ANGSTROM inversion."""
    return 1.0 / (HARTREE_INV_ANGSTROM * dE_hartree)


def crm_emission_spectrum(
    crm,
    n_e_cm3: float,
    T_e_eV: float,
    *,
    lam_min_nm: float,
    lam_max_nm: float,
    n_points: int = 2001,
    ion_mass_amu: float = 1.0,
    T_ion_K: Optional[float] = None,
    profile: str = "voigt",
    stark_hwhm_nm: Optional[float] = None,
    uta_threshold: int = 20,
    force_uta: bool = False,
) -> Spectrum:
    """Render a CRM's radiative transitions into a :class:`Spectrum`.

    Line intensities are the *emissivity-proportional* product
    ``ε_{ul} = n_upper · A_{ul} · (hν)``  — appropriate for most
    optical-plasma applications.  Multiply by a normalisation factor
    externally if you need absolute emissivity in W·m⁻³·sr⁻¹·nm⁻¹.

    Parameters
    ----------
    crm : CollisionalRadiativeModel
        A CRM with ``.levels`` and ``.transitions`` populated.
    n_e_cm3, T_e_eV : float
        Plasma electron density (cm⁻³) and temperature (eV).
    lam_min_nm, lam_max_nm : float
        Output wavelength range in nm.
    n_points : int
        Grid resolution.
    ion_mass_amu : float
        Atomic mass of the emitting ion (for Doppler broadening).
    T_ion_K : float, optional
        Ion kinetic temperature in K.  Defaults to
        ``T_e_eV × 11604.52``  (ion-electron equilibrium assumption).
    profile : {"gaussian", "lorentzian", "voigt"}
        Line profile family.  Voigt is the default when both Doppler
        and natural broadening are non-zero.
    stark_hwhm_nm : float, optional
        Additional HWHM added in quadrature to the Lorentzian width
        (set from :func:`atomx.lineshapes.stark_hwhm_quasistatic`
        for hydrogenic plasmas).
    uta_threshold : int
        Minimum number of explicit lines in a configuration-pair array
        before heavy M-/N-shell ions are rendered through UTA moments.
    force_uta : bool
        Route every eligible multi-line array through UTA moments. Intended
        for validation fixtures and external FAC comparisons.

    Returns
    -------
    Spectrum
        Emission intensity on a linear nm grid.
    """
    # Compute populations (steady-state, user may have solved time-dependent
    # elsewhere — for emission spectrum the steady-state is the usual choice).
    populations = crm.steady_state(n_e_cm3, T_e_eV)

    if T_ion_K is None:
        T_ion_K = T_e_eV * 11604.5250061657   # eV → K

    spec = Spectrum.linear(
        lam_min_nm, lam_max_nm, n_points,
        unit="emissivity (arb., n_u · A · hν)",
    )
    spec.metadata = {
        "n_e_cm3": n_e_cm3,
        "T_e_eV": T_e_eV,
        "T_ion_K": T_ion_K,
        "ion_mass_amu": ion_mass_amu,
        "profile": profile,
        "uta_threshold": uta_threshold,
        "uta_arrays": [],
    }
    grid_step_nm = (lam_max_nm - lam_min_nm) / max(n_points - 1, 1)

    if _use_uta_bridge(crm, force_uta=force_uta):
        consumed = _add_uta_arrays(
            spec,
            crm,
            populations,
            lam_min_nm=lam_min_nm,
            lam_max_nm=lam_max_nm,
            uta_threshold=uta_threshold,
        )
    else:
        consumed = set()

    for i_tr, tr in enumerate(crm.transitions):
        if i_tr in consumed:
            continue
        n_upper = populations[tr.upper]
        if n_upper <= 0 or tr.A_rate <= 0:
            continue

        dE_eV = crm.levels[tr.upper].energy_eV - crm.levels[tr.lower].energy_eV
        if dE_eV <= 0:
            continue

        lam_A = _hartree_to_angstrom(dE_eV / HARTREE_EV)
        lam_nm = lam_A / 10.0
        if lam_nm < lam_min_nm or lam_nm > lam_max_nm:
            continue

        # Doppler (thermal) σ in nm
        sigma_nm = doppler_width(lam_nm, T_ion_K, ion_mass_amu)
        sigma_nm = max(sigma_nm, grid_step_nm)

        # Natural width in nm:  Γ_λ (HWHM) = A · λ²/(4πc)·... units
        # Easier:  Γ_nu = A/(4π) [Hz], convert to Γ_λ via  Δλ = λ²/c · Δν
        # c in nm/s = 2.998e17
        gamma_nu_Hz = natural_width(tr.A_rate)   # in Hz (HWHM)
        c_nm_per_s = 2.99792458e17
        gamma_nm = (lam_nm ** 2 / c_nm_per_s) * gamma_nu_Hz

        if stark_hwhm_nm is not None:
            gamma_nm = np.hypot(gamma_nm, stark_hwhm_nm)
        gamma_nm = max(gamma_nm, grid_step_nm)

        # Intensity ~ n_upper · A · hν  — use photons·s⁻¹ weighted by energy
        intensity = n_upper * tr.A_rate * dE_eV

        if profile == "gaussian":
            spec.add_line(lam_nm, intensity, sigma_nm=sigma_nm)
        elif profile == "lorentzian":
            spec.add_line(lam_nm, intensity, gamma_nm=gamma_nm)
        elif profile == "voigt":
            spec.add_line(lam_nm, intensity,
                          sigma_nm=sigma_nm, gamma_nm=gamma_nm)
        else:
            raise ValueError(
                f"Unknown profile {profile!r}; use 'gaussian', "
                f"'lorentzian', or 'voigt'."
            )

    return spec


def _use_uta_bridge(crm, *, force_uta: bool) -> bool:
    """Return True for heavy W-like M-/N-shell CRM objects."""
    if force_uta:
        return True
    Z = getattr(crm, "Z", None)
    charge = getattr(crm, "charge", None)
    if Z is None or charge is None:
        return False
    return Z >= 60 and any(_level_shell(lev) in {"M", "N"} for lev in crm.levels)


def _level_shell(level) -> Optional[str]:
    text = " ".join(
        str(x) for x in (getattr(level, "label", ""), getattr(level, "config", ""))
        if x is not None
    )
    if any(tok in text for tok in ("4s", "4p", "4d", "4f", "n4", "N-shell")):
        return "N"
    if any(tok in text for tok in ("3s", "3p", "3d", "n3", "M-shell")):
        return "M"
    return None


def _array_key(crm, tr) -> tuple[str, str] | None:
    upper = crm.levels[tr.upper]
    lower = crm.levels[tr.lower]
    upper_cfg = getattr(upper, "config", None)
    lower_cfg = getattr(lower, "config", None)
    if upper_cfg is None or lower_cfg is None:
        return None
    return upper_cfg, lower_cfg


def _add_uta_arrays(
    spec: Spectrum,
    crm,
    populations: np.ndarray,
    *,
    lam_min_nm: float,
    lam_max_nm: float,
    uta_threshold: int,
) -> set[int]:
    grouped: dict[tuple[str, str], list[tuple[int, object, float, float]]] = defaultdict(list)
    for i_tr, tr in enumerate(crm.transitions):
        key = _array_key(crm, tr)
        if key is None:
            continue
        dE_eV = crm.levels[tr.upper].energy_eV - crm.levels[tr.lower].energy_eV
        n_upper = populations[tr.upper]
        strength = n_upper * tr.A_rate
        if dE_eV > 0 and strength > 0:
            grouped[key].append((i_tr, tr, dE_eV, strength))

    consumed: set[int] = set()
    for key, rows in grouped.items():
        if len(rows) <= uta_threshold:
            continue
        moments = uta_moments([(dE, strength) for _, _, dE, strength in rows])
        lam_nm = moments.centroid_nm
        if lam_nm < lam_min_nm or lam_nm > lam_max_nm:
            continue
        intensity = moments.total_strength * moments.centroid_eV
        sigma_nm = _uta_sigma_nm(moments.centroid_eV, moments.std_eV)
        sigma_nm = max(sigma_nm, (lam_max_nm - lam_min_nm) / max(len(spec) - 1, 1))
        spec.add_line(lam_nm, intensity, sigma_nm=sigma_nm)
        consumed.update(i_tr for i_tr, _, _, _ in rows)
        spec.metadata["uta_arrays"].append({
            "upper_config": key[0],
            "lower_config": key[1],
            "n_lines": moments.n_lines,
            "centroid_eV": moments.centroid_eV,
            "fwhm_eV": moments.fwhm_eV,
            "total_strength": moments.total_strength,
        })
    return consumed


def _uta_sigma_nm(centroid_eV: float, std_eV: float) -> float:
    if centroid_eV <= 0 or std_eV <= 0:
        return 0.0
    hc_ev_nm = 1239.841984
    return hc_ev_nm * std_eV / (centroid_eV ** 2)
